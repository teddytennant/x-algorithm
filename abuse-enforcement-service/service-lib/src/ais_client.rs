use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use tracing::info;
use xai_abuse_thrift_codec::{DeclaredException, Transcoder, SCHEMA_AIS};
use xai_thrift_mux::{ThriftMuxClient, ThriftMuxConfig};

use crate::metrics;
use crate::service::{classify_ais_failure, parse_ais_response};

const AIS_SERVICE: &str = "com.twitter.agenttools.ais.thriftscala.ActionIntakeService";

const INTAKE_RESPONSE_TYPE: &str = "com.twitter.agenttools.ais.thriftscala.ActionIntakeResponse";

const MULTI_RESPONSE_TYPE: &str =
    "com.twitter.agenttools.ais.thriftscala.MultiEntityActionIntakeResponse";

#[derive(Clone, Copy, Debug)]
pub struct AisMetricLabels<'a> {
    pub source: &'a str,
    pub topic: &'a str,
    pub entity_type: &'a str,
    pub action: &'a str,
}

impl AisMetricLabels<'static> {
    pub const HTTP: Self = Self {
        source: "http",
        topic: "",
        entity_type: "",
        action: "",
    };
}

#[derive(Clone)]
struct ExceptionMeta {
    field_id: i16,
    name: String,
    type_class: String,
}

#[derive(Clone)]
struct MethodMeta {
    thrift_method_name: String,
    arg_field_id: i16,
    request_type: String,
    response_type: String,
    exceptions: Vec<ExceptionMeta>,
}

#[derive(Debug, Clone)]
pub struct AisClientConfig {
    pub wily_path: String,
    pub tls_server_name: String,
    pub zone: String,
    pub s2s_cert_path: String,
    pub s2s_key_path: String,
    pub s2s_ca_path: String,
}

impl AisClientConfig {
    pub fn prod_defaults() -> Self {
        Self {
            wily_path: "/s/action-intake-service/action-intake-service:thrift".into(),
            tls_server_name: "action-intake-service.action-intake-service.prod.atla.s2s.twttr.net"
                .into(),
            zone: "atla".into(),
            s2s_cert_path: "/etc/strato-tls/client/tls.crt".into(),
            s2s_key_path: "/etc/strato-tls/client/tls.key".into(),
            s2s_ca_path: "/etc/strato-tls/ca/ca-bundle.crt".into(),
        }
    }
}

#[derive(Clone)]
pub struct AisClient {
    transcoder: Arc<Transcoder>,
    mux: Arc<ThriftMuxClient>,
    seq: Arc<AtomicI32>,
    intake: MethodMeta,
    multi: MethodMeta,
}

impl AisClient {
    pub fn connect(cfg: &AisClientConfig) -> Result<Self> {
        let transcoder = Transcoder::from_schema_bytes(SCHEMA_AIS)
            .context("failed to load AIS thrift schema")?;

        let intake = resolve_method(&transcoder, "intakeAction", INTAKE_RESPONSE_TYPE)?;
        let multi = resolve_method(&transcoder, "multiEntityIntakeAction", MULTI_RESPONSE_TYPE)?;

        let mux = ThriftMuxClient::new_mtls(
            ThriftMuxConfig {
                wily_path: cfg.wily_path.clone(),
                zone: cfg.zone.clone(),
                tls_server_name: Some(cfg.tls_server_name.clone()),
                refresh_interval: Some(Duration::from_secs(30)),
                ..Default::default()
            },
            &cfg.s2s_cert_path,
            &cfg.s2s_key_path,
            &cfg.s2s_ca_path,
        )
        .with_context(|| {
            format!(
                "failed to create AIS ThriftMux client for {}",
                cfg.wily_path
            )
        })?;

        info!(
            wily = %cfg.wily_path,
            tls = %cfg.tls_server_name,
            "AIS ThriftMux client initialised"
        );

        Ok(Self {
            transcoder: Arc::new(transcoder),
            mux: Arc::new(mux),
            seq: Arc::new(AtomicI32::new(0)),
            intake,
            multi,
        })
    }

    #[cfg(test)]
    pub(crate) fn unconnected() -> Self {
        let transcoder = Transcoder::from_schema_bytes(SCHEMA_AIS).expect("AIS thrift schema");
        let intake = resolve_method(&transcoder, "intakeAction", INTAKE_RESPONSE_TYPE)
            .expect("intakeAction");
        let multi = resolve_method(&transcoder, "multiEntityIntakeAction", MULTI_RESPONSE_TYPE)
            .expect("multiEntityIntakeAction");
        Self {
            transcoder: Arc::new(transcoder),
            mux: Arc::new(ThriftMuxClient::new(ThriftMuxConfig::default())),
            seq: Arc::new(AtomicI32::new(0)),
            intake,
            multi,
        }
    }

    #[tracing::instrument(skip_all, fields(topic = labels.topic, action = labels.action))]
    pub async fn intake_action(
        &self,
        request: Value,
        labels: AisMetricLabels<'_>,
    ) -> Result<Value> {
        let decoded = self.call_method(&self.intake, request, labels).await?;
        Ok(normalize_intake_response(decoded))
    }

    #[tracing::instrument(skip_all, fields(topic = labels.topic, action = labels.action))]
    pub async fn multi_entity_intake_action(
        &self,
        request: Value,
        labels: AisMetricLabels<'_>,
    ) -> Result<Value> {
        self.call_method(&self.multi, request, labels).await
    }

    async fn call_method(
        &self,
        meta: &MethodMeta,
        request: Value,
        labels: AisMetricLabels<'_>,
    ) -> Result<Value> {
        let start = Instant::now();
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);

        let payload = self
            .transcoder
            .encode_call(
                &meta.thrift_method_name,
                "",
                meta.arg_field_id,
                &meta.request_type,
                &request,
                seq,
            )
            .with_context(|| format!("AIS thrift encode failed ({})", meta.thrift_method_name))?;

        let response_bytes = match self.mux.call_raw(&payload).await {
            Ok(b) => {
                record_ais_metrics(&meta.thrift_method_name, "ok", labels, start.elapsed());
                b
            }
            Err(e) => {
                record_ais_metrics(&meta.thrift_method_name, "error", labels, start.elapsed());
                return Err(anyhow::anyhow!(
                    "AIS thriftmux {} failed: {e}",
                    meta.thrift_method_name
                ));
            }
        };

        let exceptions: Vec<DeclaredException<'_>> = meta
            .exceptions
            .iter()
            .map(|e| DeclaredException {
                field_id: e.field_id,
                name: e.name.as_str(),
                type_class: e.type_class.as_str(),
            })
            .collect();

        self.transcoder
            .decode_response_with_exceptions(&response_bytes, &meta.response_type, &exceptions)
            .with_context(|| format!("AIS thrift decode failed ({})", meta.thrift_method_name))
    }
}

fn resolve_method(
    transcoder: &Transcoder,
    method_name: &str,
    default_response: &str,
) -> Result<MethodMeta> {
    let method = transcoder
        .lookup_method(AIS_SERVICE, method_name)
        .or_else(|| transcoder.lookup_method("ais", method_name))
        .or_else(|| transcoder.lookup_method("ActionIntakeService", method_name))
        .with_context(|| {
            format!("AIS schema missing {method_name} (expected ActionIntakeService in schema-ais)")
        })?;
    Ok(MethodMeta {
        thrift_method_name: method.thrift_method_name.to_owned(),
        arg_field_id: method.arg_field_id,
        request_type: method.request_type.to_owned(),
        response_type: method.response_type.unwrap_or(default_response).to_owned(),
        exceptions: method
            .exceptions
            .iter()
            .map(|e| ExceptionMeta {
                field_id: e.field_id,
                name: e.name.to_owned(),
                type_class: e.type_class.to_owned(),
            })
            .collect(),
    })
}

fn record_ais_metrics(method: &str, status: &str, labels: AisMetricLabels<'_>, elapsed: Duration) {
    metrics::AIS_LATENCY
        .with_label_values(&[
            method,
            status,
            labels.source,
            labels.topic,
            labels.entity_type,
            labels.action,
        ])
        .observe(elapsed.as_secs_f64());
}

fn normalize_intake_response(decoded: Value) -> Value {
    if decoded.get("outcome").is_some() {
        return json!({ "success": decoded });
    }
    decoded
}

pub async fn call_ais(
    client: &AisClient,
    entity_id: i64,
    action_name: &str,
    request: Value,
    labels: AisMetricLabels<'_>,
) -> Result<Value> {
    let resp = client.intake_action(request, labels).await?;
    classify_ais_response(entity_id, action_name, resp)
}

pub fn classify_ais_response(entity_id: i64, action_name: &str, resp: Value) -> Result<Value> {
    if resp.get("aisException").is_some() {
        let msg = resp
            .pointer("/aisException/message")
            .and_then(Value::as_str)
            .unwrap_or("aisException");
        return Err(classify_ais_failure(action_name, msg, &resp));
    }
    let parsed = parse_ais_response(entity_id, &resp);
    if !parsed.success {
        return Err(classify_ais_failure(action_name, &parsed.message, &resp));
    }
    Ok(resp)
}

pub fn build_intake_request(
    actor_header: Value,
    action: Value,
    tool: &str,
    audit_note: Option<&str>,
) -> Value {
    let mut req = json!({
        "actorHeader": actor_header,
        "action": action,
        "tool": tool,
    });
    if let Some(note) = audit_note {
        req["auditHeader"] = json!({ "note": note });
    }
    req
}

pub fn build_multi_entity_request(
    actor_header: Value,
    actions: Vec<Value>,
    tool: &str,
    audit_note: Option<&str>,
) -> Value {
    let mut req = json!({
        "actorHeader": actor_header,
        "actions": actions,
        "tool": tool,
    });
    if let Some(note) = audit_note {
        req["auditHeader"] = json!({ "note": note });
    }
    req
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiEntityCounts {
    pub ok: usize,
    pub failed: usize,
}

impl MultiEntityCounts {
    pub fn from_response(resp: &Value) -> Self {
        Self {
            ok: map_len(
                resp.get("successesfulActions")
                    .or_else(|| resp.get("successfulActions")),
            ),
            failed: map_len(resp.get("failedActions")),
        }
    }

    pub fn all_ok(self) -> bool {
        self.ok > 0 && self.failed == 0
    }

    pub fn all_failed(self) -> bool {
        self.ok == 0 && self.failed > 0
    }

    pub fn partial(self) -> bool {
        self.ok > 0 && self.failed > 0
    }

    pub fn empty(self) -> bool {
        self.ok == 0 && self.failed == 0
    }

    pub fn outcome_label(self) -> &'static str {
        if self.all_ok() {
            "all_ok"
        } else if self.partial() {
            "partial"
        } else {
            "all_failed"
        }
    }
}

fn map_len(v: Option<&Value>) -> usize {
    match v {
        Some(Value::Object(m)) => m.len(),
        Some(Value::Array(a)) => a.len(),
        _ => 0,
    }
}

pub fn coerce_intake_request(body: Value) -> Result<Value> {
    if body.get("action").is_some() && body.get("actorHeader").is_some() {
        return Ok(body);
    }
    if let Some(inner) = body.get("actionIntakeRequest").cloned() {
        return Ok(inner);
    }
    bail!(
        "body must be an ActionIntakeRequest (actorHeader + action + tool) \
         or legacy {{\"actionIntakeRequest\": …}}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::PermanentAisFailure;
    use serde_json::json;

    #[test]
    fn normalize_wraps_thrift_outcome() {
        let thrift = json!({"outcome": {"success": {"actionId": "abc"}}});
        let n = normalize_intake_response(thrift);
        assert_eq!(
            n.pointer("/success/outcome/success/actionId")
                .and_then(|v| v.as_str()),
            Some("abc")
        );
    }

    #[test]
    fn coerce_accepts_inner_and_legacy_wrap() {
        let inner = json!({
            "actorHeader": {"plugin": {"pluginId": "x"}},
            "action": {"suspendUser": {"userId": 1}},
            "tool": "RtpPlugin",
        });
        assert!(coerce_intake_request(inner.clone()).is_ok());
        assert_eq!(
            coerce_intake_request(json!({"actionIntakeRequest": inner.clone()})).unwrap(),
            inner
        );
    }

    #[test]
    fn schema_has_intake_and_multi_entity() {
        let t = Transcoder::from_schema_bytes(SCHEMA_AIS).unwrap();
        assert!(
            t.lookup_method(AIS_SERVICE, "intakeAction").is_some()
                || t.lookup_method("ActionIntakeService", "intakeAction")
                    .is_some()
        );
        assert!(
            t.lookup_method(AIS_SERVICE, "multiEntityIntakeAction")
                .is_some()
                || t.lookup_method("ActionIntakeService", "multiEntityIntakeAction")
                    .is_some()
        );
    }

    #[test]
    fn encode_suspend_user_smoke() {
        let t = Transcoder::from_schema_bytes(SCHEMA_AIS).unwrap();
        let method = t
            .lookup_method(AIS_SERVICE, "intakeAction")
            .or_else(|| t.lookup_method("ActionIntakeService", "intakeAction"))
            .expect("intakeAction");
        let body = json!({
            "actorHeader": {"plugin": {"pluginId": "xai-abuse-enforcement-service"}},
            "action": {
                "suspendUser": {
                    "userId": 42,
                    "perm": false,
                    "policy": "PlatformManipulation",
                    "additionalInfo": ["note"]
                }
            },
            "tool": "RtpPlugin",
            "auditHeader": {"note": "test"}
        });
        let bytes = t
            .encode_call(
                method.thrift_method_name,
                method.arg_field_name,
                method.arg_field_id,
                method.request_type,
                &body,
                1,
            )
            .expect("encode");
        assert!(
            bytes.len() > 16,
            "encoded payload too small: {}",
            bytes.len()
        );
    }

    #[test]
    fn encode_multi_entity_smoke() {
        let t = Transcoder::from_schema_bytes(SCHEMA_AIS).unwrap();
        let method = t
            .lookup_method(AIS_SERVICE, "multiEntityIntakeAction")
            .or_else(|| t.lookup_method("ActionIntakeService", "multiEntityIntakeAction"))
            .expect("multiEntityIntakeAction");
        let body = build_multi_entity_request(
            json!({"plugin": {"pluginId": "xai-abuse-enforcement-service"}}),
            vec![
                json!({"suspendUser": {
                    "userId": 1,
                    "perm": false,
                    "policy": "PlatformManipulation",
                    "additionalInfo": []
                }}),
                json!({"bounceViaSelection": {
                    "target": {"user": {"userId": 1}},
                    "tags": [],
                    "uncheckedTags": ["FAKE"],
                    "bounceActor": {"simpleService": {"serviceName": "xai-abuse-enforcement-service"}}
                }}),
            ],
            "RtpPlugin",
            Some("audit"),
        );
        let bytes = t
            .encode_call(
                method.thrift_method_name,
                method.arg_field_name,
                method.arg_field_id,
                method.request_type,
                &body,
                1,
            )
            .expect("encode multiEntity");
        assert!(
            bytes.len() > 32,
            "encoded multi payload too small: {}",
            bytes.len()
        );
    }

    #[test]
    fn thrift_ais_exception_validation_failed_is_permanent() {
        let t = Transcoder::from_schema_bytes(SCHEMA_AIS).unwrap();
        let method = t
            .lookup_method(AIS_SERVICE, "intakeAction")
            .or_else(|| t.lookup_method("ActionIntakeService", "intakeAction"))
            .expect("intakeAction in schema");

        assert!(
            method
                .exceptions
                .iter()
                .any(|e| e.field_id == 4 && e.name == "aisException"),
            "schema must declare aisException as result field 4: {:?}",
            method
                .exceptions
                .iter()
                .map(|e| (e.field_id, e.name))
                .collect::<Vec<_>>()
        );

        let exception_type = "com.twitter.agenttools.ais.thriftscala.AISException";
        let body = json!({
            "kind": "Internal",
            "message": "Processor chain: 'add_labels_v2_processor_chain' threw an unhandled exception (Throwable) with message: 'User could not be successfully validated.'",
            "underlyingDetails": {
                "exceptionClass": "com.twitter.gizmoduck.thriftscala.ValidationFailed",
                "message": "User could not be successfully validated.",
                "stackTrace": [
                    "com.twitter.gizmoduck.thriftscala.ValidationFailed$.decodeInternal(ValidationFailed.scala:195)"
                ]
            }
        });
        let bytes = t
            .encode_exception_reply("intakeAction", 4, exception_type, &body, 1)
            .expect("encode exception reply");

        let decoded = t
            .decode_method_response(&bytes, &method)
            .expect("declared aisException must decode, not hard-fail");

        assert!(
            decoded.get("aisException").is_some(),
            "expected {{aisException: …}}, got {decoded}"
        );
        assert_eq!(
            decoded
                .pointer("/aisException/underlyingDetails/exceptionClass")
                .and_then(|v| v.as_str()),
            Some("com.twitter.gizmoduck.thriftscala.ValidationFailed")
        );

        let err = classify_ais_response(1672364227921682433, "addLabelsV2", decoded)
            .expect_err("ValidationFailed must not be Ok");
        let permanent = err
            .downcast_ref::<PermanentAisFailure>()
            .expect("must be PermanentAisFailure so Kafka path does not retry");
        assert!(
            permanent.0.contains("ValidationFailed"),
            "permanent message should name the class: {}",
            permanent.0
        );
    }

    #[test]
    fn multi_entity_counts_from_response() {
        let resp = json!({
            "successesfulActions": {"a": {}, "b": {}},
            "failedActions": {"c": {}}
        });
        let c = MultiEntityCounts::from_response(&resp);
        assert_eq!(c.ok, 2);
        assert_eq!(c.failed, 1);
        assert!(c.partial());
        assert_eq!(c.outcome_label(), "partial");

        let all_ok = MultiEntityCounts::from_response(&json!({
            "successesfulActions": {"a": {}},
            "failedActions": {}
        }));
        assert!(all_ok.all_ok());
        assert_eq!(all_ok.outcome_label(), "all_ok");

        let alt = MultiEntityCounts::from_response(&json!({
            "successfulActions": {"x": {}, "y": {}},
            "failedActions": {}
        }));
        assert_eq!(alt.ok, 2);

        let empty = MultiEntityCounts::from_response(&json!({
            "successesfulActions": {},
            "failedActions": {}
        }));
        assert!(empty.empty());
        assert!(!empty.all_ok());
        assert_eq!(empty.outcome_label(), "all_failed");
    }
}
