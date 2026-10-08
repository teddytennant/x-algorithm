use std::sync::{Arc, LazyLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::Json;
use axum::extract::{Extension, Path, State};
use axum::http::{Method, StatusCode, Uri};
use axum::response::IntoResponse;
use backon::ExponentialBuilder;
use serde::Serialize;
use serde_json::{Value, json};
use tracing::{error, info, warn};
use xai_build_version::current_build_information;
use xai_kafka::KafkaProducer;

use crate::ais_client::{
    AisClient, AisMetricLabels, MultiEntityCounts, build_intake_request,
    build_multi_entity_request, call_ais, coerce_intake_request,
};
use crate::allowlist::{AllowlistEntry, AllowlistRecord};
use crate::auth::KeyId;
use crate::decision::ActionSpec;
use crate::facts::EntityType;
use crate::growthbook::DynamicConfig;
use crate::growthbook_writer::GrowthBookWriter;
use crate::metrics;
use xai_abuse_proto::enforcement::{AdminAction, AdminActionEvent, ScoreResult};

const PLUGIN_ID: &str = env!("CARGO_PKG_NAME");

const STUB_BOTMAKER_BOT_ID: i64 = 0;

const AIS_TOOL: &str = "RtpPlugin";

const ARKOSE_BOUNCE_TAGS: &[&str] = &["FAKE", "FAKE_ARKOSE_LOG_RECAPTCHA_EXPERIMENT"];

const CAPTCHA_BOUNCE_TAGS: &[&str] = &[
    "FAKE",
    "FAKE_RECAPTCHA_ENTERPRISE_SCORE",
    "FAKE_RECAPTCHA_ENTERPRISE_SCORE_ONLY",
];

const SPAM_LIVENESS_CHECK_TEMPLATE_ID: &str = "spam_liveness_check";

const SPAM_LIVENESS_CHECK_POLICY_TAG: &str = "Spam";

fn bounce_actor_json() -> Value {
    json!({"simpleService": {"serviceName": PLUGIN_ID}})
}

static AUDIT_NOTE: LazyLock<String> = LazyLock::new(|| {
    let bi = current_build_information();
    let sha = &bi.git_commit_sha[..7.min(bi.git_commit_sha.len())];
    let environment = std::env::var("NAMESPACE").unwrap_or_else(|_| "local".to_string());
    format!(
        "Automated enforcement by {} {} ({} @ {}) [env: {}]",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        sha,
        bi.git_commit_timestamp.format("%Y-%m-%d"),
        environment,
    )
});

fn with_audit_note(mut additional_info: Vec<String>) -> Vec<String> {
    additional_info.push(AUDIT_NOTE.clone());
    additional_info
}

fn join_audit_info(mut additional_info: Vec<String>) -> String {
    additional_info.push(AUDIT_NOTE.clone());
    additional_info.join("; ")
}

fn compute_expires_at_msec(ttl_msec: i64) -> i64 {
    let now_msec = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    now_msec.saturating_add(ttl_msec)
}

static AUDIT_ENVIRONMENT: LazyLock<String> =
    LazyLock::new(|| std::env::var("NAMESPACE").unwrap_or_else(|_| "local".to_string()));

pub fn publish_admin_action_event(producer: Option<&Arc<KafkaProducer>>, event: &AdminActionEvent) {
    crate::spawn_publish(
        producer,
        crate::ADMIN_ACTIONS_PRODUCER,
        || {
            if event.entity_id != 0 {
                event.entity_id.to_string().into_bytes()
            } else {
                event.actor_key_id.clone().into_bytes()
            }
        },
        event,
    );
}

pub struct AppState {
        pub ais_client: AisClient,
    pub dynamic_config: DynamicConfig,
                pub rules_cache: std::sync::Arc<crate::rules::RulesCache>,
    pub dedup_cache: Option<crate::dedup_cache::ManhattanDedupCache>,
    pub limiter: Option<crate::limiter::LimiterClient>,
    pub allowlist: Option<crate::allowlist::ManhattanAllowlist>,
                    pub growthbook_writer: Option<GrowthBookWriter>,
                pub growthbook_feature_key: String,
    pub growthbook_environment: String,
        pub kafka_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
                pub kafka_producers: crate::KafkaProducers,
                pub hold_gate: std::sync::Arc<crate::overturn_hold::HoldGate>,
                pub boot_config: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct UserActionResult {
    pub user_id: i64,
    pub success: bool,
    pub message: String,
}

pub enum EnforcementAction {
    SuspendUser {
        user_id: i64,
        perm: bool,
        policy: String,
        additional_info: Vec<String>,
    },
                        AddLabelsV2 {
        user_id: i64,
        labels: Vec<String>,
                                expires_at_msec: Option<i64>,
                                audit_note: String,
    },
                                                AddPostLabelsV2 {
        tweet_id: i64,
        labels: Vec<String>,
                                        expires_at_msec: Option<i64>,
                        audit_note: String,
    },
                                    BounceArkose { user_id: i64, audit_note: String },
                BounceCaptcha { user_id: i64, audit_note: String },
                                                SpamLivenessCheck { user_id: i64, audit_note: String },
}

impl EnforcementAction {
                pub fn suspend_user(
        user_id: i64,
        perm: bool,
        policy: impl Into<String>,
        additional_info: Vec<String>,
    ) -> Self {
        Self::SuspendUser {
            user_id,
            perm,
            policy: policy.into(),
            additional_info: with_audit_note(additional_info),
        }
    }

                                        pub fn add_user_label(
        user_id: i64,
        labels: Vec<String>,
        ttl_msec: Option<i64>,
        additional_info: Vec<String>,
    ) -> Self {
        Self::AddLabelsV2 {
            user_id,
            labels,
            expires_at_msec: ttl_msec.map(compute_expires_at_msec),
            audit_note: join_audit_info(additional_info),
        }
    }

                                pub fn add_post_label(
        tweet_id: i64,
        labels: Vec<String>,
        ttl_msec: Option<i64>,
        additional_info: Vec<String>,
    ) -> Self {
        Self::AddPostLabelsV2 {
            tweet_id,
            labels,
            expires_at_msec: ttl_msec.map(compute_expires_at_msec),
            audit_note: join_audit_info(additional_info),
        }
    }

                pub fn bounce_arkose(user_id: i64, additional_info: Vec<String>) -> Self {
        Self::BounceArkose {
            user_id,
            audit_note: join_audit_info(additional_info),
        }
    }

            pub fn bounce_captcha(user_id: i64, additional_info: Vec<String>) -> Self {
        Self::BounceCaptcha {
            user_id,
            audit_note: join_audit_info(additional_info),
        }
    }

                    pub fn spam_liveness_check(user_id: i64, additional_info: Vec<String>) -> Self {
        Self::SpamLivenessCheck {
            user_id,
            audit_note: join_audit_info(additional_info),
        }
    }

                                                                    pub fn from_spec(
        spec: &ActionSpec,
        entity_id: i64,
        user_id: i64,
        additional_info: Vec<String>,
    ) -> Self {
        match spec {
            ActionSpec::SuspendUser { perm, policy } => {
                Self::suspend_user(user_id, *perm, policy.clone(), additional_info)
            }
            ActionSpec::AddLabelsV2 { labels, ttl_msec } => {
                Self::add_user_label(user_id, labels.clone(), *ttl_msec, additional_info)
            }
            ActionSpec::AddPostLabelsV2 { labels, ttl_msec } => {
                Self::add_post_label(entity_id, labels.clone(), *ttl_msec, additional_info)
            }
            ActionSpec::Arkose => Self::bounce_arkose(user_id, additional_info),
            ActionSpec::Captcha => Self::bounce_captcha(user_id, additional_info),
            ActionSpec::SpamLivenessCheck => Self::spam_liveness_check(user_id, additional_info),
        }
    }

                                            fn audit_note(&self) -> &str {
        match self {
            Self::SuspendUser { .. } => AUDIT_NOTE.as_str(),
            Self::AddLabelsV2 { audit_note, .. }
            | Self::AddPostLabelsV2 { audit_note, .. }
            | Self::BounceArkose { audit_note, .. }
            | Self::BounceCaptcha { audit_note, .. }
            | Self::SpamLivenessCheck { audit_note, .. } => audit_note.as_str(),
        }
    }

                                fn actor_header(&self) -> Value {
        match self {
            Self::AddPostLabelsV2 { .. } => {
                json!({"botmakerBot": {"botId": STUB_BOTMAKER_BOT_ID}})
            }
            Self::SuspendUser { .. }
            | Self::AddLabelsV2 { .. }
            | Self::BounceArkose { .. }
            | Self::BounceCaptcha { .. }
            | Self::SpamLivenessCheck { .. } => json!({"plugin": {"pluginId": PLUGIN_ID}}),
        }
    }

                                        pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::SuspendUser { .. } => "suspendUser",
            Self::AddLabelsV2 { .. } => "addLabelsV2",
            Self::AddPostLabelsV2 { .. } => "addPostLabelsV2",
            Self::BounceArkose { .. } => "bounceArkose",
            Self::BounceCaptcha { .. } => "bounceCaptcha",
            Self::SpamLivenessCheck { .. } => "spamLivenessCheck",
        }
    }

                pub(crate) fn metric_label(&self) -> String {
        match self {
            Self::AddLabelsV2 { labels, .. } | Self::AddPostLabelsV2 { labels, .. } => {
                labels.join(",")
            }
            Self::SuspendUser { .. }
            | Self::BounceArkose { .. }
            | Self::BounceCaptcha { .. }
            | Self::SpamLivenessCheck { .. } => String::new(),
        }
    }

                                                fn entity_id(&self) -> i64 {
        match self {
            Self::SuspendUser { user_id, .. }
            | Self::AddLabelsV2 { user_id, .. }
            | Self::BounceArkose { user_id, .. }
            | Self::BounceCaptcha { user_id, .. }
            | Self::SpamLivenessCheck { user_id, .. } => *user_id,
            Self::AddPostLabelsV2 { tweet_id, .. } => *tweet_id,
        }
    }

        fn to_json(&self) -> Value {
        match self {
            Self::SuspendUser {
                user_id,
                perm,
                policy,
                additional_info,
            } => json!({
                "suspendUser": {
                    "userId": user_id,
                    "perm": perm,
                    "policy": policy,
                    "additionalInfo": additional_info,
                }
            }),
            Self::AddLabelsV2 {
                user_id,
                labels,
                expires_at_msec,
                ..
            } => {
                let user_entity = json!({
                    "entity": {
                        "entityType": "User",
                        "entityId": {"longId": user_id}
                    }
                });
                let safety_label = match expires_at_msec {
                    Some(expires_at_msec) => json!({"expiresAtMsec": expires_at_msec}),
                    None => json!({}),
                };
                let labels_json: Vec<Value> = labels
                    .iter()
                    .map(|label| {
                        json!({
                            "nonPolicyBasedLabel": {
                                "label": {
                                    "labelReferenceWithSafetyLabel": {
                                        "labelReference": {
                                            "entityType": "User",
                                            "labelIdType": {"nativeUserLabels": label},
                                            "labelType": "Native"
                                        },
                                        "safetyLabel": safety_label.clone(),
                                    }
                                }
                            }
                        })
                    })
                    .collect();
                json!({
                    "addLabelsV2": {
                        "entityId": user_entity,
                        "entityRef": user_entity,
                        "labels": labels_json,
                        "violatingUserIds": [],
                    }
                })
            }
            Self::AddPostLabelsV2 {
                tweet_id,
                labels,
                expires_at_msec,
                ..
            } => {
                let tweet_entity = json!({
                    "entity": {
                        "entityType": "Tweet",
                        "entityId": {"longId": tweet_id}
                    }
                });
                let safety_label = match expires_at_msec {
                    Some(expires_at_msec) => json!({"expiresAtMsec": expires_at_msec}),
                    None => json!({}),
                };
                let labels_json: Vec<Value> = labels
                    .iter()
                    .map(|label| {
                        json!({
                            "nonPolicyBasedLabel": {
                                "label": {
                                    "labelReferenceWithSafetyLabel": {
                                        "labelReference": {
                                            "entityType": "Tweet",
                                            "labelIdType": {"safetyTweetLabels": label},
                                            "labelType": "Safety"
                                        },
                                        "safetyLabel": safety_label.clone(),
                                    }
                                }
                            }
                        })
                    })
                    .collect();
                json!({
                    "addLabelsV2": {
                        "entityId": tweet_entity,
                        "entityRef": tweet_entity,
                        "labels": labels_json,
                        "violatingUserIds": [],
                    }
                })
            }
            Self::BounceArkose { user_id, .. } => {
                bounce_via_selection(*user_id, ARKOSE_BOUNCE_TAGS)
            }
            Self::BounceCaptcha { user_id, .. } => {
                bounce_via_selection(*user_id, CAPTCHA_BOUNCE_TAGS)
            }
            Self::SpamLivenessCheck { user_id, .. } => bounce_via_template(
                *user_id,
                SPAM_LIVENESS_CHECK_TEMPLATE_ID,
                SPAM_LIVENESS_CHECK_POLICY_TAG,
            ),
        }
    }
}

fn bounce_via_selection(user_id: i64, tags: &[&str]) -> Value {
    json!({
        "bounceViaSelection": {
            "target": {"user": {"userId": user_id}},
            "tags": [],
            "uncheckedTags": tags,
            "bounceActor": bounce_actor_json(),
        }
    })
}

fn bounce_via_template(user_id: i64, template_id: &str, policy_tag: &str) -> Value {
    json!({
        "bounceViaTemplate": {
            "userId": user_id,
            "templateId": template_id,
            "policyTag": policy_tag,
            "additionalTags": [],
            "bounceActor": bounce_actor_json(),
        }
    })
}


#[tracing::instrument(skip_all)]
pub fn record_strato_metrics(
    column: &str,
    status: &str,
    elapsed: std::time::Duration,
    retries: u32,
) {
    let labels = &[column, status];
    metrics::STRATO_LATENCY
        .with_label_values(labels)
        .observe(elapsed.as_secs_f64());
    metrics::STRATO_RETRIES
        .with_label_values(labels)
        .observe(retries as f64);
}

pub fn strato_retry_strategy() -> ExponentialBuilder {
    ExponentialBuilder::default()
        .with_min_delay(std::time::Duration::from_millis(100))
        .with_max_delay(std::time::Duration::from_secs(2))
        .with_max_times(5)
        .with_jitter()
}


#[utoipa::path(
    post,
    path = "/action",
    tag = "debug",
    security(("BearerAuth" = [])),
    description = "[PROTECTED — bearer token] Manual/debug direct AIS intakeAction (ThriftMux). The production path is the Kafka consumer.",
    request_body(
        content = Object,
        description = "ActionIntakeRequest JSON (actorHeader + action + tool), or legacy {\"actionIntakeRequest\": …}",
    ),
    responses(
        (status = 200, description = "AIS responded successfully — body is the raw AIS response", body = Object),
        (status = 500, description = "AIS call failed (network / non-success response). Body shape: {\"action\": <error string>}"),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_action(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    metrics::HTTP_INFLIGHT.with_label_values(&["action"]).inc();
    let start = Instant::now();

    info!("POST /api/action: {payload}");

    let (status, result) = match coerce_intake_request(payload) {
        Ok(req) => match call_ais(&state.ais_client, 0, "action", req, AisMetricLabels::HTTP).await
        {
            Ok(resp) => ("ok", (StatusCode::OK, Json(resp))),
            Err(e) => {
                error!(error = format!("{e:#}"), "Action AIS call failed");
                let code = if e.downcast_ref::<PermanentAisFailure>().is_some() {
                    StatusCode::UNPROCESSABLE_ENTITY
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                };
                ("error", (code, Json(json!({"action": format!("{e:#}")}))))
            }
        },
        Err(e) => (
            "error",
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"action": e.to_string()})),
            ),
        ),
    };

    metrics::HTTP_LATENCY
        .with_label_values(&["action", status])
        .observe(start.elapsed().as_secs_f64());
    metrics::HTTP_INFLIGHT.with_label_values(&["action"]).dec();

    result
}

#[utoipa::path(
    get,
    path = "/config",
    tag = "config",
    security(("BearerAuth" = [])),
    responses(
        (status = 200, description = "Current effective GrowthBook config object. Empty object when GrowthBook is disabled.", body = Object),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_config(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(state.dynamic_config.config().unwrap_or_default())
}

#[utoipa::path(
    get,
    path = "/dry_run",
    tag = "debug",
    responses(
        (status = 200, description = "Current effective dry_run flag", body = crate::api_doc::DryRunStatus),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_dry_run(State(state): State<Arc<AppState>>) -> Json<Value> {
    let value = state.dynamic_config.is_dry_run();
    let ret = json!({ "dry_run": value });
    info!("{ret:?}");
    Json(ret)
}


#[tracing::instrument(skip_all)]
pub async fn enforce_actions(
    ais_client: &AisClient,
    dry_run: bool,
    score_result: Option<&ScoreResult>,
    topic: &str,
    entity_type: crate::facts::EntityType,
    actions: Vec<EnforcementAction>,
) -> Result<()> {
    if actions.len() == 1 {
        let action = actions.into_iter().next().expect("len checked == 1");
        return enforce_action(
            ais_client,
            dry_run,
            score_result,
            topic,
            entity_type,
            action,
        )
        .await;
    }

    let kind_label: String = actions
        .iter()
        .map(|a| a.name())
        .collect::<Vec<_>>()
        .join("+");

    let groups = partition_contiguous_by_actor_header(actions);
    let mut any_ok = false;
    let mut any_err = false;

    for (actor_header, group) in groups {
        if group.len() == 1 {
            let action = group.into_iter().next().expect("len checked == 1");
            let action_name = action.name();
            let res = Box::pin(enforce_action(
                ais_client,
                dry_run,
                score_result,
                topic,
                entity_type,
                action,
            ))
            .await;
            match res {
                Ok(()) => any_ok = true,
                Err(e) => {
                    any_err = true;
                    warn!(
                        topic,
                        "composite {kind_label}: {action_name} sub-call failed: {e}"
                    );
                }
            }
            continue;
        }

        match enforce_action_group_multi(
            ais_client,
            dry_run,
            score_result,
            topic,
            entity_type,
            actor_header,
            group,
        )
        .await
        {
            Ok(counts) => {
                if counts.ok > 0 {
                    any_ok = true;
                }
                if counts.failed > 0 || counts.empty() {
                    any_err = true;
                }
            }
            Err(e) => {
                any_err = true;
                warn!(
                    topic,
                    "composite {kind_label}: multiEntity group failed: {e}"
                );
            }
        }
    }

    let outcome = match (any_ok, any_err) {
        (true, false) => "all_ok",
        (true, true) => "partial",
        (false, _) => "all_failed",
    };
    metrics::ACTION_DISPATCH_TOTAL
        .with_label_values(&[kind_label.as_str(), outcome])
        .inc();

    Ok(())
}

fn partition_contiguous_by_actor_header(
    actions: Vec<EnforcementAction>,
) -> Vec<(Value, Vec<EnforcementAction>)> {
    let mut groups: Vec<(Value, Vec<EnforcementAction>)> = Vec::new();
    for action in actions {
        let header = action.actor_header();
        if let Some((h, bucket)) = groups.last_mut()
            && *h == header
        {
            bucket.push(action);
        } else {
            groups.push((header, vec![action]));
        }
    }
    groups
}

#[tracing::instrument(skip_all, fields(group_size))]
async fn enforce_action_group_multi(
    ais_client: &AisClient,
    dry_run: bool,
    score_result: Option<&ScoreResult>,
    topic: &str,
    entity_type: crate::facts::EntityType,
    actor_header: Value,
    group: Vec<EnforcementAction>,
) -> Result<MultiEntityCounts> {
    let group_size = group.len();
    tracing::Span::current().record("group_size", group_size);

    let entity_type_str = entity_type.as_str();
    let head = score_result.map(crate::facts::head_label).unwrap_or("");
    let model = score_result.map(|s| s.model_version.as_str()).unwrap_or("");

    let action_jsons: Vec<Value> = group.iter().map(|a| a.to_json()).collect();
    let audit_note = group.first().map(|a| a.audit_note());
    let payload = build_multi_entity_request(actor_header, action_jsons, AIS_TOOL, audit_note);

    if dry_run {
        info!("dry run; would multiEntityIntakeAction (n={group_size}): {payload}");
        for action in &group {
            emit_enforcement_metric(topic, entity_type_str, action, "dry_run", head, model);
        }
        return Ok(MultiEntityCounts {
            ok: group_size,
            failed: 0,
        });
    }

    let action_kind: String = group.iter().map(|a| a.name()).collect::<Vec<_>>().join("+");
    let ais_labels = AisMetricLabels {
        source: "kafka",
        topic,
        entity_type: entity_type_str,
        action: &action_kind,
    };
    let resp = match ais_client
        .multi_entity_intake_action(payload, ais_labels)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            for action in &group {
                emit_enforcement_metric(topic, entity_type_str, action, "error", head, model);
            }
            return Err(e);
        }
    };

    let counts = MultiEntityCounts::from_response(&resp);
    info!(
        ok = counts.ok,
        failed = counts.failed,
        outcome = counts.outcome_label(),
        "multiEntityIntakeAction complete"
    );

    if counts.empty() {
        warn!(
            group_size,
            "multiEntity returned empty successesfulActions and failedActions: {resp}"
        );
        for action in &group {
            emit_enforcement_metric(topic, entity_type_str, action, "error", head, model);
        }
        return Ok(MultiEntityCounts {
            ok: 0,
            failed: group_size,
        });
    }

    if counts.failed > 0 {
        warn!(
            failed = counts.failed,
            "multiEntity had failedActions: {resp}"
        );
    }

    let status_for = |i: usize| -> &'static str {
        if counts.all_ok() {
            "success"
        } else if counts.all_failed() {
            "ais_failure"
        } else if i < counts.ok {
            "success"
        } else {
            "ais_failure"
        }
    };
    for (i, action) in group.iter().enumerate() {
        emit_enforcement_metric(topic, entity_type_str, action, status_for(i), head, model);
    }

    Ok(counts)
}

fn emit_enforcement_metric(
    topic: &str,
    entity_type: &str,
    action: &EnforcementAction,
    status: &str,
    head: &str,
    model: &str,
) {
    let action_name = action.name();
    let metric_label = action.metric_label();
    let (step, action_name_label, label_slot) = if status == "success" || status == "dry_run" {
        ("final", action_name, metric_label.as_str())
    } else {
        (action_name, action_name, metric_label.as_str())
    };
    metrics::ENFORCEMENT_TOTAL
        .with_label_values(&[
            "kafka",
            topic,
            entity_type,
            step,
            status,
            action_name_label,
            label_slot,
            head,
            model,
        ])
        .inc();
}

#[tracing::instrument(skip_all, fields(action_name))]
pub async fn enforce_action(
    ais_client: &AisClient,
    dry_run: bool,
    score_result: Option<&ScoreResult>,
    topic: &str,
    entity_type: crate::facts::EntityType,
    action: EnforcementAction,
) -> Result<()> {
    let action_name = action.name();
    let metric_label = action.metric_label();
    let entity_type = entity_type.as_str();
    let entity_id = action.entity_id();
    let head = score_result.map(crate::facts::head_label).unwrap_or("");
    let model = score_result.map(|s| s.model_version.as_str()).unwrap_or("");
    let span = tracing::Span::current();
    span.record("action_name", action_name);

    let payload = build_intake_request(
        action.actor_header(),
        action.to_json(),
        AIS_TOOL,
        Some(action.audit_note()),
    );

    if dry_run {
        let mut log_payload = payload.clone();
        if let Some(arr) = log_payload
            .pointer_mut("/action/suspendUser/additionalInfo")
            .and_then(|v| v.as_array_mut())
        {
            for item in arr.iter_mut() {
                if let Some(s) = item.as_str()
                    && s.starts_with("decoded_actions: ")
                    && s.len() > 500
                {
                    *item = Value::String(truncate_utf8(s, 500));
                }
            }
        }
        if let Some(note_val) = log_payload.pointer_mut("/auditHeader/note")
            && let Some(s) = note_val.as_str()
            && s.len() > 1000
        {
            *note_val = Value::String(truncate_utf8(s, 1000));
        }
        let score_summary = score_result.map(|sr| {
            let mut v = serde_json::to_value(sr).unwrap_or_default();
            if let Some(da) = v.get("decoded_actions") {
                let s = da.to_string();
                if s.len() > 500 {
                    v["decoded_actions"] = Value::String(truncate_utf8(&s, 500));
                }
            }
            v
        });
        info!("dry run; would call {action_name}: {log_payload}. ScoreResult: {score_summary:?}");
    } else {
        let ais_labels = AisMetricLabels {
            source: "kafka",
            topic,
            entity_type,
            action: action_name,
        };
        match call_ais(ais_client, entity_id, action_name, payload, ais_labels).await {
            Ok(_resp) => {}
            Err(e) => {
                let status = if e.downcast_ref::<PermanentAisFailure>().is_some() {
                    "ais_failure"
                } else {
                    "error"
                };
                metrics::ENFORCEMENT_TOTAL
                    .with_label_values(&[
                        "kafka",
                        topic,
                        entity_type,
                        action_name,
                        status,
                        action_name,
                        &metric_label,
                        head,
                        model,
                    ])
                    .inc();
                return Err(e);
            }
        }
    }

    let status = if dry_run { "dry_run" } else { "success" };
    metrics::ENFORCEMENT_TOTAL
        .with_label_values(&[
            "kafka",
            topic,
            entity_type,
            "final",
            status,
            action_name,
            &metric_label,
            head,
            model,
        ])
        .inc();
    Ok(())
}


fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
}


#[derive(Debug)]
pub struct PermanentAisFailure(pub String);

impl std::fmt::Display for PermanentAisFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "permanent AIS rejection: {}", self.0)
    }
}

impl std::error::Error for PermanentAisFailure {}

const DETERMINISTIC_AIS_EXCEPTION_CLASSES: &[&str] = &[
    "com.twitter.gizmoduck.thriftscala.ValidationFailed",
];

pub fn classify_ais_failure(action_name: &str, message: &str, resp: &Value) -> anyhow::Error {
    if resp.pointer("/success/outcome/failure").is_some()
        || resp.pointer("/outcome/failure").is_some()
    {
        return anyhow::Error::new(PermanentAisFailure(format!(
            "AIS rejected {action_name}: {message}"
        )));
    }
    if let Some(class) = resp
        .pointer("/aisException/underlyingDetails/exceptionClass")
        .and_then(Value::as_str)
        && DETERMINISTIC_AIS_EXCEPTION_CLASSES.contains(&class)
    {
        let detail = resp
            .pointer("/aisException/underlyingDetails/message")
            .and_then(Value::as_str)
            .unwrap_or("");
        return anyhow::Error::new(PermanentAisFailure(format!(
            "AIS {action_name} raised deterministic {class}: {detail}"
        )));
    }
    anyhow::anyhow!("AIS failure on {action_name}: {message}")
}

pub fn parse_ais_response(user_id: i64, resp: &Value) -> UserActionResult {
    let outcome_success = resp
        .pointer("/success/outcome/success")
        .or_else(|| resp.pointer("/outcome/success"));
    if let Some(action_id) = outcome_success
        .and_then(|s| s.get("actionId"))
        .and_then(|id| id.as_str())
    {
        UserActionResult {
            user_id,
            success: true,
            message: format!("actionId={action_id}"),
        }
    } else {
        UserActionResult {
            user_id,
            success: false,
            message: format!("Unexpected AIS response: {resp}"),
        }
    }
}


fn rate_limit_entity_type(s: &str) -> Result<crate::facts::EntityType, (StatusCode, Json<Value>)> {
    crate::facts::EntityType::from_path(s).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("unknown entity_type `{s}` (expected `user` or `post`)")
            })),
        )
    })
}

#[utoipa::path(
    get,
    path = "/rate_limit/{entity_type}",
    tag = "debug",
    params(
        ("entity_type" = String, Path, description = "Which cap to report: `user` or `post`."),
    ),
    responses(
        (
            status = 200,
            description = "Current rate-limit counter stats.",
            body = crate::api_doc::RateLimitStats,
        ),
        (
            status = 503,
            description = "Limiter backend not configured or unavailable.",
        ),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_rate_limit(
    State(state): State<Arc<AppState>>,
    Path(entity_type): Path<String>,
) -> impl IntoResponse {
    let entity_type = match rate_limit_entity_type(&entity_type) {
        Ok(et) => et,
        Err(resp) => return resp,
    };
    match state.dynamic_config.rate_limit_stats(entity_type) {
        Some((used, cap, remaining)) => (
            StatusCode::OK,
            Json(json!({
                "entity_type": entity_type.as_str(),
                "used": used,
                "cap": cap,
                "remaining": remaining,
            })),
        ),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "limiter unavailable"})),
        ),
    }
}

#[utoipa::path(
    post,
    path = "/rate_limit/{entity_type}",
    tag = "debug",
    security(("BearerAuth" = [])),
    description = "[PROTECTED — bearer token] Consume one ticket from the rolling enforcement cap for the given entity_type. Side-effecting debug aid: it burns real budget, so exhausting it wedges enforcement until POST /rate_limit/{entity_type}/reset.",
    params(
        ("entity_type" = String, Path, description = "Which cap to increment: `user` or `post`."),
    ),
    responses(
        (
            status = 200,
            description = "Limiter increment response: {allowed, limit, remaining, reset_at, denial}.",
            body = Object,
        ),
        (
            status = 503,
            description = "Limiter backend not configured or unavailable.",
        ),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_rate_limit_increment(
    State(state): State<Arc<AppState>>,
    Path(entity_type): Path<String>,
) -> impl IntoResponse {
    let entity_type = match rate_limit_entity_type(&entity_type) {
        Ok(et) => et,
        Err(resp) => return resp,
    };
    match state
        .dynamic_config
        .record_enforcement_report(entity_type)
        .await
    {
        Some(response) => (StatusCode::OK, Json(response)),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "limiter unavailable"})),
        ),
    }
}

#[utoipa::path(
    post,
    path = "/rate_limit/{entity_type}/reset",
    tag = "debug",
    security(("BearerAuth" = [])),
    description = "[PROTECTED — bearer token] Reset the rolling enforcement cap for the given entity_type: clears stored snapshots and re-baselines used→~0. Recovers a wedged cap.",
    params(
        ("entity_type" = String, Path, description = "Which cap to reset: `user` or `post`."),
    ),
    responses(
        (
            status = 200,
            description = "Cap reset. Body: {reset: true, deleted_snapshots: <n>}.",
            body = Object,
        ),
        (
            status = 503,
            description = "Limiter backend not configured, or the reset scan failed.",
        ),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_rate_limit_reset(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Path(entity_type): Path<String>,
) -> impl IntoResponse {
    let entity_type = match rate_limit_entity_type(&entity_type) {
        Ok(et) => et,
        Err(resp) => return resp,
    };
    let before_json = state
        .dynamic_config
        .rate_limit_stats(entity_type)
        .map(|(used, cap, remaining)| {
            json!({"used": used, "cap": cap, "remaining": remaining}).to_string()
        })
        .unwrap_or_default();
    let deleted = state.dynamic_config.reset_rate_limit(entity_type).await;
    let (success, error) = match deleted {
        Some(_) => (true, String::new()),
        None => (false, "limiter unavailable".to_string()),
    };
    publish_admin_action_event(
        state.kafka_producers.admin_actions(),
        &AdminActionEvent {
            occurred_at_ms: crate::now_millis(),
            action: AdminAction::RateLimitReset as i32,
            actor_key_id: actor.0,
            environment: AUDIT_ENVIRONMENT.clone(),
            http_method: method.to_string(),
            http_path: uri.path().to_string(),
            success,
            error,
            entity_type: entity_type.as_str().to_string(),
            before_json,
            ..Default::default()
        },
    );
    match deleted {
        Some(deleted) => (
            StatusCode::OK,
            Json(json!({
                "reset": true,
                "entity_type": entity_type.as_str(),
                "deleted_snapshots": deleted,
            })),
        ),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "limiter unavailable"})),
        ),
    }
}

#[utoipa::path(
    get,
    path = "/dedup/{user_id}",
    tag = "debug",
    params(
        ("user_id" = i64, Path, description = "Numeric Twitter user ID"),
    ),
    responses(
        (status = 200, description = "Remaining TTL (seconds) + the decompressed dedup entry: `{ttl_secs, entry}` (entry is the decoded ScoreResult + audit map). ttl_secs is -1 if no expiry.", body = Object),
        (status = 404, description = "No dedup entry for this user_id"),
        (status = 503, description = "Dedup cache not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(user_id))]
pub async fn handle_dedup_get(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i64>,
) -> impl IntoResponse {
    let Some(cache) = &state.dedup_cache else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "dedup cache unavailable"})),
        );
    };
    let Some((raw, ttl_secs)) = cache
        .get_with_ttl(crate::facts::EntityType::User, user_id)
        .await
    else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"})));
    };
    let json_bytes = zstd::decode_all(raw.as_slice()).unwrap_or(raw);
    let entry = serde_json::from_slice::<Value>(&json_bytes)
        .unwrap_or_else(|_| json!({"raw": String::from_utf8_lossy(&json_bytes)}));
    (
        StatusCode::OK,
        Json(json!({"ttl_secs": ttl_secs, "entry": entry})),
    )
}

#[utoipa::path(
    get,
    path = "/dedup/{entity_type}/{entity_id}",
    tag = "debug",
    params(
        ("entity_type" = String, Path, description = "Entity type: `user` or `post`"),
        ("entity_id" = i64, Path, description = "Numeric entity id"),
    ),
    responses(
        (status = 200, description = "Remaining TTL (seconds) + decompressed dedup entry: `{ttl_secs, entry}`. ttl_secs is -1 if no expiry.", body = Object),
        (status = 400, description = "Unknown entity_type"),
        (status = 404, description = "No dedup entry for this entity"),
        (status = 503, description = "Dedup cache not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(entity_type, entity_id))]
pub async fn handle_dedup_get_entity(
    State(state): State<Arc<AppState>>,
    Path((entity_type, entity_id)): Path<(String, i64)>,
) -> impl IntoResponse {
    let Some(et) = crate::facts::EntityType::from_path(&entity_type) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error": format!("unknown entity_type `{entity_type}` (expected `user` or `post`)")}),
            ),
        );
    };
    let Some(cache) = &state.dedup_cache else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "dedup cache unavailable"})),
        );
    };
    let Some((raw, ttl_secs)) = cache.get_with_ttl(et, entity_id).await else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not found"})));
    };
    let json_bytes = zstd::decode_all(raw.as_slice()).unwrap_or(raw);
    let entry = serde_json::from_slice::<Value>(&json_bytes)
        .unwrap_or_else(|_| json!({"raw": String::from_utf8_lossy(&json_bytes)}));
    (
        StatusCode::OK,
        Json(json!({"ttl_secs": ttl_secs, "entry": entry})),
    )
}


use xai_abuse_enforcement_types::allowlist::{
    AllowlistAddRequest, AllowlistBulkEntry, AllowlistBulkRequest, BulkResponse, BulkRowResult,
    MAX_TTL_SECS,
};
use xai_abuse_enforcement_types::config::ConfigPatchRequest;

#[utoipa::path(
    put,
    path = "/config",
    tag = "config",
    security(("BearerAuth" = [])),
    request_body(
        content = Object,
        description = "Full replacement config value (must be a JSON object)",
    ),
    responses(
        (status = 200, description = "Stored config object echoed back", body = Object),
        (status = 400, description = "Malformed request body"),
        (status = 500, description = "GrowthBook write failed"),
        (status = 503, description = "GROWTHBOOK_ADMIN_API_KEY not configured at startup"),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_config_put(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Json(value): Json<Value>,
) -> impl IntoResponse {
    let Some(writer) = state.growthbook_writer.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "GrowthBook admin API not configured"})),
        );
    };
    info!(
        "PUT /api/config (feature={} env={})",
        state.growthbook_feature_key, state.growthbook_environment
    );
    let request_json = value.to_string();
    let before_json = state
        .dynamic_config
        .config()
        .map(|c| c.to_string())
        .unwrap_or_default();
    let result = writer
        .set_env_value(
            &state.growthbook_feature_key,
            &state.growthbook_environment,
            value,
        )
        .await;
    let (success, error) = match &result {
        Ok(_) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };
    publish_admin_action_event(
        state.kafka_producers.admin_actions(),
        &AdminActionEvent {
            occurred_at_ms: crate::now_millis(),
            action: AdminAction::ConfigReplace as i32,
            actor_key_id: actor.0,
            environment: AUDIT_ENVIRONMENT.clone(),
            http_method: method.to_string(),
            http_path: uri.path().to_string(),
            success,
            error,
            request_json,
            before_json,
            ..Default::default()
        },
    );
    match result {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err(e) => {
            error!("PUT /api/config failed: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        }
    }
}

#[utoipa::path(
    patch,
    path = "/config/{field}",
    tag = "config",
    security(("BearerAuth" = [])),
    params(
        ("field" = String, Path, description = "Top-level key inside the GrowthBook config object to overwrite"),
    ),
    request_body = ConfigPatchRequest,
    responses(
        (status = 200, description = "New full config object after the patch", body = Object),
        (status = 400, description = "Malformed request body"),
        (status = 500, description = "GrowthBook write failed"),
        (status = 503, description = "GROWTHBOOK_ADMIN_API_KEY not configured at startup"),
    ),
)]
#[tracing::instrument(skip_all, fields(field))]
pub async fn handle_config_patch_field(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Path(field): Path<String>,
    Json(body): Json<ConfigPatchRequest>,
) -> impl IntoResponse {
    let Some(writer) = state.growthbook_writer.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "GrowthBook admin API not configured"})),
        );
    };
    let span = tracing::Span::current();
    span.record("field", &field);
    info!("PATCH /api/config/{field} = {}", body.value);
    let request_json = serde_json::to_string(&body).unwrap_or_default();
    let before_json = state
        .dynamic_config
        .config()
        .and_then(|c| c.get(&field).cloned())
        .map(|v| v.to_string())
        .unwrap_or_default();
    let result = writer
        .set_env_field(
            &state.growthbook_feature_key,
            &state.growthbook_environment,
            &field,
            body.value,
        )
        .await;
    let (success, error) = match &result {
        Ok(_) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };
    publish_admin_action_event(
        state.kafka_producers.admin_actions(),
        &AdminActionEvent {
            occurred_at_ms: crate::now_millis(),
            action: AdminAction::ConfigPatchField as i32,
            actor_key_id: actor.0,
            environment: AUDIT_ENVIRONMENT.clone(),
            http_method: method.to_string(),
            http_path: uri.path().to_string(),
            success,
            error,
            config_field: field.clone(),
            request_json,
            before_json,
            ..Default::default()
        },
    );
    match result {
        Ok(new_config) => (StatusCode::OK, Json(new_config)),
        Err(e) => {
            error!("PATCH /api/config/{field} failed: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        }
    }
}

#[utoipa::path(
    get,
    path = "/rules",
    tag = "config",
    security(("BearerAuth" = [])),
    responses(
        (status = 200, description = "Active rule pipelines (user + post)", body = Object),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_rules_get(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let dc = &state.dynamic_config;
    let cache = &state.rules_cache;
    let user = cache.status(
        EntityType::User,
        dc.enforcement_rules_yaml(EntityType::User).as_ref(),
    );
    let post = cache.status(
        EntityType::Post,
        dc.enforcement_rules_yaml(EntityType::Post).as_ref(),
    );
    (StatusCode::OK, Json(json!({ "user": user, "post": post })))
}

#[utoipa::path(
    put,
    path = "/allowlist/{user_id}",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    params(
        ("user_id" = i64, Path, description = "Numeric Twitter user ID"),
    ),
    request_body = AllowlistAddRequest,
    responses(
        (status = 200, description = "Stored allowlist record echoed back", body = AllowlistRecord),
        (status = 400, description = "Invalid TTL (zero or exceeds 90-day max)"),
        (status = 500, description = "Manhattan write failed"),
        (status = 503, description = "Allowlist not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(user_id))]
pub async fn handle_allowlist_put(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Path(user_id): Path<i64>,
    Json(req): Json<AllowlistAddRequest>,
) -> impl IntoResponse {
    let Some(al) = state.allowlist.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        );
    };
    if req.ttl_secs == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "TTL must be > 0"})),
        );
    }
    if req.ttl_secs > MAX_TTL_SECS {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error": format!("TTL exceeds 90-day max ({MAX_TTL_SECS}s)"), "max_ttl_secs": MAX_TTL_SECS}),
            ),
        );
    }
    let request_json = serde_json::to_string(&req).unwrap_or_default();
    let ttl_secs = req.ttl_secs;
    let entry = AllowlistEntry {
        added_by: req.added_by,
        reason: req.reason,
        added_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    info!("PUT /api/allowlist/{user_id} ttl={ttl_secs}s");
    let result = al.add(user_id, ttl_secs, &entry).await;
    let (success, error) = match &result {
        Ok(()) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };
    publish_admin_action_event(
        state.kafka_producers.admin_actions(),
        &AdminActionEvent {
            occurred_at_ms: crate::now_millis(),
            action: AdminAction::AllowlistUpsert as i32,
            actor_key_id: actor.0,
            environment: AUDIT_ENVIRONMENT.clone(),
            http_method: method.to_string(),
            http_path: uri.path().to_string(),
            success,
            error,
            entity_type: EntityType::User.as_str().to_string(),
            entity_id: user_id,
            request_json,
            ..Default::default()
        },
    );
    match result {
        Ok(()) => {
            let record = AllowlistRecord {
                user_id,
                added_by: entry.added_by,
                reason: entry.reason,
                added_at: entry.added_at,
                ttl_secs: ttl_secs as i64,
            };
            (StatusCode::OK, Json(serde_json::to_value(record).unwrap()))
        }
        Err(e) => {
            error!("PUT /api/allowlist/{user_id} failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        }
    }
}

#[utoipa::path(
    delete,
    path = "/allowlist/{user_id}",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    params(
        ("user_id" = i64, Path, description = "Numeric Twitter user ID"),
    ),
    responses(
        (status = 200, description = "Delete attempted (idempotent; `removed=false` if key absent)", body = crate::api_doc::AllowlistDeleteResponse),
        (status = 500, description = "Manhattan delete failed"),
        (status = 503, description = "Allowlist not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(user_id))]
pub async fn handle_allowlist_delete(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Path(user_id): Path<i64>,
) -> impl IntoResponse {
    let Some(al) = state.allowlist.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        );
    };
    info!("DELETE /api/allowlist/{user_id}");
    let before_json = al
        .get(user_id)
        .await
        .map(|r| serde_json::to_string(&r).unwrap_or_default())
        .unwrap_or_default();
    let result = al.remove(user_id).await;
    let (success, error) = match &result {
        Ok(_) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };
    publish_admin_action_event(
        state.kafka_producers.admin_actions(),
        &AdminActionEvent {
            occurred_at_ms: crate::now_millis(),
            action: AdminAction::AllowlistRemove as i32,
            actor_key_id: actor.0,
            environment: AUDIT_ENVIRONMENT.clone(),
            http_method: method.to_string(),
            http_path: uri.path().to_string(),
            success,
            error,
            entity_type: EntityType::User.as_str().to_string(),
            entity_id: user_id,
            before_json,
            ..Default::default()
        },
    );
    match result {
        Ok(removed) => (
            StatusCode::OK,
            Json(json!({"user_id": user_id, "removed": removed})),
        ),
        Err(e) => {
            error!("DELETE /api/allowlist/{user_id} failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        }
    }
}

pub async fn bulk_allowlist_impl(
    al: &crate::allowlist::ManhattanAllowlist,
    entries: &[AllowlistBulkEntry],
    dry_run: bool,
) -> BulkResponse {
    let mut results: Vec<BulkRowResult> = Vec::with_capacity(entries.len());
    let mut would_add = 0usize;
    let mut would_update = 0usize;
    let mut errors = 0usize;

    for entry in entries {
        if entry.ttl_secs == 0 || entry.ttl_secs > MAX_TTL_SECS {
            errors += 1;
            results.push(BulkRowResult {
                user_id: entry.user_id,
                ok: false,
                mode: String::new(),
                error: Some(if entry.ttl_secs == 0 {
                    "TTL must be > 0".into()
                } else {
                    format!("TTL exceeds 90-day max ({}s)", MAX_TTL_SECS)
                }),
                existing_ttl_secs: None,
                ttl_secs: Some(entry.ttl_secs),
            });
            continue;
        }

        let existing_ttl = al.get(entry.user_id).await.map(|r| r.ttl_secs);
        let mode = if existing_ttl.is_some() {
            would_update += 1;
            "update"
        } else {
            would_add += 1;
            "add"
        };

        if dry_run {
            results.push(BulkRowResult {
                user_id: entry.user_id,
                ok: true,
                mode: mode.into(),
                error: None,
                existing_ttl_secs: existing_ttl,
                ttl_secs: Some(entry.ttl_secs),
            });
            continue;
        }

        let al_entry = AllowlistEntry {
            added_by: entry.added_by.clone(),
            reason: entry.reason.clone(),
            added_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        };
        match al.add(entry.user_id, entry.ttl_secs, &al_entry).await {
            Ok(()) => results.push(BulkRowResult {
                user_id: entry.user_id,
                ok: true,
                mode: mode.into(),
                error: None,
                existing_ttl_secs: None,
                ttl_secs: Some(entry.ttl_secs),
            }),
            Err(e) => {
                errors += 1;
                results.push(BulkRowResult {
                    user_id: entry.user_id,
                    ok: false,
                    mode: mode.into(),
                    error: Some(e.to_string()),
                    existing_ttl_secs: None,
                    ttl_secs: None,
                });
            }
        }
    }

    BulkResponse {
        dry_run,
        total: results.len(),
        would_add,
        would_update,
        errors,
        results,
    }
}

#[utoipa::path(
    post,
    path = "/allowlist/bulk",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    request_body = AllowlistBulkRequest,
    responses(
        (status = 200, description = "Per-row outcomes (always 200; failures are reported in `errors` + per-row `ok`)", body = BulkResponse),
        (status = 503, description = "Allowlist not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(count, dry_run))]
pub async fn handle_allowlist_bulk(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Json(req): Json<AllowlistBulkRequest>,
) -> impl IntoResponse {
    let Some(al) = state.allowlist.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        );
    };
    let span = tracing::Span::current();
    span.record("count", req.entries.len());
    span.record("dry_run", req.dry_run);
    info!(
        "POST /api/allowlist/bulk count={} dry_run={}",
        req.entries.len(),
        req.dry_run
    );

    let body = bulk_allowlist_impl(al, &req.entries, req.dry_run).await;
    if !req.dry_run {
        publish_admin_action_event(
            state.kafka_producers.admin_actions(),
            &AdminActionEvent {
                occurred_at_ms: crate::now_millis(),
                action: AdminAction::AllowlistBulkUpsert as i32,
                actor_key_id: actor.0,
                environment: AUDIT_ENVIRONMENT.clone(),
                http_method: method.to_string(),
                http_path: uri.path().to_string(),
                success: body.errors == 0,
                entity_type: EntityType::User.as_str().to_string(),
                affected_count: body.total.saturating_sub(body.errors) as i32,
                request_json: serde_json::to_string(&req).unwrap_or_default(),
                ..Default::default()
            },
        );
    }
    (StatusCode::OK, Json(serde_json::to_value(body).unwrap()))
}


#[utoipa::path(
    get,
    path = "/allowlist",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    responses(
        (
            status = 200,
            description = "All allowlist records with remaining TTLs.",
            body = Vec<AllowlistRecord>,
        ),
        (
            status = 503,
            description = "Allowlist not configured.",
        ),
    ),
)]
#[tracing::instrument(skip_all)]
pub async fn handle_allowlist_list(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match &state.allowlist {
        Some(al) => {
            let records = al.list_all().await;
            (
                StatusCode::OK,
                Json(serde_json::to_value(&records).unwrap_or_default()),
            )
        }
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        ),
    }
}

#[utoipa::path(
    get,
    path = "/allowlist/{user_id}",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    params(
        ("user_id" = i64, Path, description = "Numeric Twitter user ID"),
    ),
    responses(
        (status = 200, description = "Allowlist record for this user_id", body = AllowlistRecord),
        (status = 404, description = "No allowlist entry for this user_id"),
        (status = 503, description = "Allowlist not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(user_id))]
pub async fn handle_allowlist_get(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<i64>,
) -> impl IntoResponse {
    let Some(al) = &state.allowlist else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        );
    };
    match al.get(user_id).await {
        Some(record) => (
            StatusCode::OK,
            Json(serde_json::to_value(&record).unwrap_or_default()),
        ),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))),
    }
}


#[utoipa::path(
    get,
    path = "/allowlist/{entity_type}/{entity_id}",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    params(
        ("entity_type" = String, Path, description = "Entity type: `user` or `post`"),
        ("entity_id" = i64, Path, description = "Numeric entity id"),
    ),
    responses(
        (status = 200, description = "Allowlist record for this entity", body = Object),
        (status = 400, description = "Unknown entity_type"),
        (status = 404, description = "No allowlist entry for this entity"),
        (status = 503, description = "Allowlist not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(entity_type, entity_id))]
pub async fn handle_allowlist_get_entity(
    State(state): State<Arc<AppState>>,
    Path((entity_type, entity_id)): Path<(String, i64)>,
) -> impl IntoResponse {
    let Some(et) = crate::facts::EntityType::from_path(&entity_type) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error": format!("unknown entity_type `{entity_type}` (expected `user` or `post`)")}),
            ),
        );
    };
    let Some(al) = &state.allowlist else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        );
    };
    match al.get_entity(et, entity_id).await {
        Some((entry, ttl_secs)) => (
            StatusCode::OK,
            Json(json!({
                "entity_type": et.as_str(),
                "entity_id": entity_id,
                "added_by": entry.added_by,
                "reason": entry.reason,
                "added_at": entry.added_at,
                "ttl_secs": ttl_secs,
            })),
        ),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "not found"}))),
    }
}

#[utoipa::path(
    put,
    path = "/allowlist/{entity_type}/{entity_id}",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    params(
        ("entity_type" = String, Path, description = "Entity type: `user` or `post`"),
        ("entity_id" = i64, Path, description = "Numeric entity id"),
    ),
    request_body = AllowlistAddRequest,
    responses(
        (status = 200, description = "Stored allowlist record echoed back", body = Object),
        (status = 400, description = "Unknown entity_type, or invalid TTL (zero or exceeds 90-day max)"),
        (status = 500, description = "Manhattan write failed"),
        (status = 503, description = "Allowlist not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(entity_type, entity_id))]
pub async fn handle_allowlist_put_entity(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Path((entity_type, entity_id)): Path<(String, i64)>,
    Json(req): Json<AllowlistAddRequest>,
) -> impl IntoResponse {
    let Some(et) = crate::facts::EntityType::from_path(&entity_type) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error": format!("unknown entity_type `{entity_type}` (expected `user` or `post`)")}),
            ),
        );
    };
    let Some(al) = state.allowlist.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        );
    };
    if req.ttl_secs == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "TTL must be > 0"})),
        );
    }
    if req.ttl_secs > MAX_TTL_SECS {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error": format!("TTL exceeds 90-day max ({MAX_TTL_SECS}s)"), "max_ttl_secs": MAX_TTL_SECS}),
            ),
        );
    }
    let request_json = serde_json::to_string(&req).unwrap_or_default();
    let ttl_secs = req.ttl_secs;
    let entry = AllowlistEntry {
        added_by: req.added_by,
        reason: req.reason,
        added_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    };
    info!(
        "PUT /api/allowlist/{}/{entity_id} ttl={ttl_secs}s",
        et.as_str(),
    );
    let result = al.add_entity(et, entity_id, ttl_secs, &entry).await;
    let (success, error) = match &result {
        Ok(()) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };
    publish_admin_action_event(
        state.kafka_producers.admin_actions(),
        &AdminActionEvent {
            occurred_at_ms: crate::now_millis(),
            action: AdminAction::AllowlistUpsert as i32,
            actor_key_id: actor.0,
            environment: AUDIT_ENVIRONMENT.clone(),
            http_method: method.to_string(),
            http_path: uri.path().to_string(),
            success,
            error,
            entity_type: et.as_str().to_string(),
            entity_id,
            request_json,
            ..Default::default()
        },
    );
    match result {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({
                "entity_type": et.as_str(),
                "entity_id": entity_id,
                "added_by": entry.added_by,
                "reason": entry.reason,
                "added_at": entry.added_at,
                "ttl_secs": ttl_secs as i64,
            })),
        ),
        Err(e) => {
            error!("PUT /api/allowlist/{}/{entity_id} failed: {e}", et.as_str());
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        }
    }
}

#[utoipa::path(
    delete,
    path = "/allowlist/{entity_type}/{entity_id}",
    tag = "allowlist",
    security(("BearerAuth" = [])),
    params(
        ("entity_type" = String, Path, description = "Entity type: `user` or `post`"),
        ("entity_id" = i64, Path, description = "Numeric entity id"),
    ),
    responses(
        (status = 200, description = "Delete attempted (idempotent; `removed=false` if key absent)", body = Object),
        (status = 400, description = "Unknown entity_type"),
        (status = 500, description = "Manhattan delete failed"),
        (status = 503, description = "Allowlist not configured"),
    ),
)]
#[tracing::instrument(skip_all, fields(entity_type, entity_id))]
pub async fn handle_allowlist_delete_entity(
    State(state): State<Arc<AppState>>,
    Extension(actor): Extension<KeyId>,
    method: Method,
    uri: Uri,
    Path((entity_type, entity_id)): Path<(String, i64)>,
) -> impl IntoResponse {
    let Some(et) = crate::facts::EntityType::from_path(&entity_type) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error": format!("unknown entity_type `{entity_type}` (expected `user` or `post`)")}),
            ),
        );
    };
    let Some(al) = state.allowlist.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "allowlist not configured"})),
        );
    };
    info!("DELETE /api/allowlist/{}/{entity_id}", et.as_str());
    let before_json = al
        .get_entity(et, entity_id)
        .await
        .map(|(entry, ttl_secs)| {
            json!({
                "entity_type": et.as_str(),
                "entity_id": entity_id,
                "added_by": entry.added_by,
                "reason": entry.reason,
                "added_at": entry.added_at,
                "ttl_secs": ttl_secs,
            })
            .to_string()
        })
        .unwrap_or_default();
    let result = al.remove_entity(et, entity_id).await;
    let (success, error) = match &result {
        Ok(_) => (true, String::new()),
        Err(e) => (false, e.to_string()),
    };
    publish_admin_action_event(
        state.kafka_producers.admin_actions(),
        &AdminActionEvent {
            occurred_at_ms: crate::now_millis(),
            action: AdminAction::AllowlistRemove as i32,
            actor_key_id: actor.0,
            environment: AUDIT_ENVIRONMENT.clone(),
            http_method: method.to_string(),
            http_path: uri.path().to_string(),
            success,
            error,
            entity_type: et.as_str().to_string(),
            entity_id,
            before_json,
            ..Default::default()
        },
    );
    match result {
        Ok(removed) => (
            StatusCode::OK,
            Json(json!({
                "entity_type": et.as_str(),
                "entity_id": entity_id,
                "removed": removed,
            })),
        ),
        Err(e) => {
            error!(
                "DELETE /api/allowlist/{}/{entity_id} failed: {e}",
                et.as_str()
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_utf8_does_not_split_multibyte_chars() {
        let prefix = "a".repeat(999);
        let s = format!("{prefix}ـrest");
        assert!(s.is_char_boundary(999));
        assert!(!s.is_char_boundary(1000));
        let out = truncate_utf8(&s, 1000);
        assert!(out.ends_with("..."));
        assert_eq!(out, format!("{prefix}..."));
        assert_eq!(truncate_utf8("short", 1000), "short");
    }

    fn assert_audit_note_appended(info: &[Value], first_caller_entry: &str) {
        assert_eq!(
            info.first().and_then(|v| v.as_str()),
            Some(first_caller_entry)
        );
        assert!(
            info.last()
                .and_then(|v| v.as_str())
                .is_some_and(|s| s.starts_with("Automated enforcement by")),
        );
    }

    const POLICY_IN_VIOLATION: &str = "PlatformManipulation";
    const SPAM_HIGH_RECALL_LABEL: &str = "SpamHighRecall";
    const LABEL_EXPIRY_MSEC: i64 = 30 * 24 * 60 * 60 * 1000;

                                #[test]
    fn every_action_shape_encodes_against_the_ais_schema() {
        use xai_abuse_thrift_codec::{SCHEMA_AIS, Transcoder};

        let transcoder = Transcoder::from_schema_bytes(SCHEMA_AIS).expect("AIS schema loads");
        let method = transcoder
            .lookup_method(
                "com.twitter.agenttools.ais.thriftscala.ActionIntakeService",
                "intakeAction",
            )
            .expect("intakeAction in schema");

        let actions = [
            EnforcementAction::suspend_user(1, false, POLICY_IN_VIOLATION, vec!["n".to_owned()]),
            EnforcementAction::suspend_user(1, true, POLICY_IN_VIOLATION, vec![]),
            EnforcementAction::add_user_label(
                1,
                vec![SPAM_HIGH_RECALL_LABEL.to_owned()],
                Some(LABEL_EXPIRY_MSEC),
                vec!["n".to_owned()],
            ),
            EnforcementAction::add_user_label(
                1,
                vec![SPAM_HIGH_RECALL_LABEL.to_owned()],
                None,
                vec![],
            ),
            EnforcementAction::add_post_label(
                2,
                vec![SPAM_HIGH_RECALL_LABEL.to_owned()],
                Some(LABEL_EXPIRY_MSEC),
                vec![],
            ),
            EnforcementAction::bounce_arkose(1, vec!["n".to_owned()]),
            EnforcementAction::bounce_captcha(1, vec![]),
            EnforcementAction::spam_liveness_check(1, vec![]),
        ];

        for action in actions {
            let request = crate::ais_client::build_intake_request(
                action.actor_header(),
                action.to_json(),
                AIS_TOOL,
                Some(action.audit_note()),
            );
            transcoder
                .encode_call(
                    method.thrift_method_name,
                    method.arg_field_name,
                    method.arg_field_id,
                    method.request_type,
                    &request,
                    0,
                )
                .unwrap_or_else(|e| panic!("{} must encode: {e}", action.name()));
        }
    }

    #[test]
    fn ais_outcome_failure_is_detected_as_permanent() {
        let rejection = json!({"success": {"outcome": {"failure": {"businessLogic": {
            "reason": "Plugin(PluginHeader(xai-abuse-enforcement-service)) is not a supported SafetyLabelSource"
        }}}}});
        assert!(!parse_ais_response(1, &rejection).success);
        let err = classify_ais_failure("suspendUser", "rejected", &rejection);
        assert!(err.downcast_ref::<PermanentAisFailure>().is_some());

        let ok = json!({"success": {"outcome": {"success": {"actionId": "abc"}}}});
        assert!(parse_ais_response(1, &ok).success);

        let garbage = json!({"err": "timeout"});
        assert!(!parse_ais_response(1, &garbage).success);
        let err = classify_ais_failure("suspendUser", "unexpected", &garbage);
        assert!(err.downcast_ref::<PermanentAisFailure>().is_none());
    }

    #[test]
    fn deterministic_ais_exception_is_permanent_but_unknown_classes_retry() {
        let validation_failed = json!({"aisException": {
            "kind": "Internal",
            "message": "Processor chain: 'add_labels_v2_processor_chain' threw an unhandled exception (Throwable) with message: 'User could not be successfully validated.'",
            "underlyingDetails": {
                "exceptionClass": "com.twitter.gizmoduck.thriftscala.ValidationFailed",
                "message": "User could not be successfully validated.",
                "stackTrace": ["com.twitter.gizmoduck.thriftscala.ValidationFailed$.decodeInternal(ValidationFailed.scala:195)"],
            },
        }});
        let err =
            classify_ais_failure("addLabelsV2", "Unexpected AIS response", &validation_failed);
        let permanent = err
            .downcast_ref::<PermanentAisFailure>()
            .expect("ValidationFailed aisException must classify as permanent");
        assert!(permanent.0.contains("ValidationFailed"));
        assert!(
            permanent
                .0
                .contains("User could not be successfully validated.")
        );
        assert!(!permanent.0.contains("stackTrace"));

        let transient = json!({"aisException": {
            "kind": "Internal",
            "underlyingDetails": {
                "exceptionClass": "com.twitter.finagle.GlobalRequestTimeoutException",
                "message": "exceeded 5.seconds to gizmoduck while waiting for a response",
            },
        }});
        let err = classify_ais_failure("addLabelsV2", "Unexpected AIS response", &transient);
        assert!(err.downcast_ref::<PermanentAisFailure>().is_none());

        let bare = json!({"aisException": {"kind": "Internal", "message": "boom"}});
        let err = classify_ais_failure("addLabelsV2", "Unexpected AIS response", &bare);
        assert!(err.downcast_ref::<PermanentAisFailure>().is_none());
    }

    #[test]
    fn suspend_user_serializes_to_ais_shape() {
        let action = EnforcementAction::suspend_user(
            42,
            false,
            POLICY_IN_VIOLATION,
            vec!["model_version: v1".to_owned()],
        );
        let v = action.to_json();
        let inner = v.get("suspendUser").expect("suspendUser key");
        assert_eq!(inner["userId"], 42);
        assert_eq!(inner["perm"], false);
        assert_eq!(inner["policy"], POLICY_IN_VIOLATION);
        assert_audit_note_appended(
            inner["additionalInfo"].as_array().unwrap(),
            "model_version: v1",
        );
    }

    #[test]
    fn add_user_label_serializes_to_ais_shape() {
        let action = EnforcementAction::add_user_label(
            7,
            vec![SPAM_HIGH_RECALL_LABEL.into()],
            Some(LABEL_EXPIRY_MSEC),
            vec!["model_version: v1".to_owned(), "topic: foo".to_owned()],
        );
        let v = action.to_json();
        let inner = v.get("addLabelsV2").expect("addLabelsV2 key");

        let expected_entity = json!({
            "entity": {"entityType": "User", "entityId": {"longId": 7}}
        });
        assert_eq!(inner["entityId"], expected_entity);
        assert_eq!(inner["entityRef"], expected_entity);
        assert_eq!(inner["violatingUserIds"], json!([]));

        let labels = inner["labels"].as_array().expect("labels array");
        assert_eq!(labels.len(), 1);
        let label_ref_with_safety =
            &labels[0]["nonPolicyBasedLabel"]["label"]["labelReferenceWithSafetyLabel"];
        assert_eq!(
            label_ref_with_safety["labelReference"],
            json!({
                "entityType": "User",
                "labelIdType": {"nativeUserLabels": "SpamHighRecall"},
                "labelType": "Native",
            }),
        );
        let expires = label_ref_with_safety["safetyLabel"]["expiresAtMsec"]
            .as_i64()
            .expect("expiresAtMsec is i64");
        let now_msec = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        let delta = expires - now_msec;
        assert!(
            (LABEL_EXPIRY_MSEC - 60_000..=LABEL_EXPIRY_MSEC + 60_000).contains(&delta),
            "expiresAtMsec delta from now: {delta}",
        );

        assert!(inner.get("additionalInfo").is_none());
        assert!(inner.get("policy").is_none());

        let note = action.audit_note();
        assert!(note.starts_with("model_version: v1; topic: foo; "));
        assert!(note.contains("Automated enforcement by"));
    }

    #[test]
    fn add_user_label_without_ttl_emits_empty_safety_label() {
        let action =
            EnforcementAction::add_user_label(7, vec![SPAM_HIGH_RECALL_LABEL.into()], None, vec![]);
        let v = action.to_json();
        let safety_label = &v["addLabelsV2"]["labels"][0]["nonPolicyBasedLabel"]["label"]["labelReferenceWithSafetyLabel"]
            ["safetyLabel"];
        assert_eq!(*safety_label, json!({}));
        assert!(safety_label["expiresAtMsec"].is_null());
    }

    #[test]
    fn metric_label_is_joined_labels_for_add_labels_v2_and_empty_otherwise() {
        let single = EnforcementAction::add_user_label(
            7,
            vec![SPAM_HIGH_RECALL_LABEL.into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        assert_eq!(single.metric_label(), SPAM_HIGH_RECALL_LABEL);

        let multi = EnforcementAction::add_user_label(
            7,
            vec!["SpamHighRecall".into(), "LowQualityReply".into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        assert_eq!(multi.metric_label(), "SpamHighRecall,LowQualityReply");

        for action in [
            EnforcementAction::suspend_user(1, false, POLICY_IN_VIOLATION, vec![]),
            EnforcementAction::bounce_arkose(2, vec![]),
            EnforcementAction::bounce_captcha(3, vec![]),
            EnforcementAction::spam_liveness_check(4, vec![]),
        ] {
            assert_eq!(action.metric_label(), "", "action {}", action.name());
        }
    }

    #[test]
    fn enforcement_total_accepts_action_name_plus_metric_label_row() {
        let action = EnforcementAction::add_user_label(
            7,
            vec![SPAM_HIGH_RECALL_LABEL.into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        let counter = crate::metrics::ENFORCEMENT_TOTAL.with_label_values(&[
            "kafka",
            "test-topic",
            "user",
            "final",
            "success",
            action.name(),
            &action.metric_label(),
            "FollowBot",
            "v_test",
        ]);
        let before = counter.get();
        counter.inc();
        assert_eq!(counter.get(), before + 1);
    }

    #[test]
    fn kafka_message_latency_accepts_entity_type_row() {
        let histo = crate::metrics::KAFKA_MESSAGE_LATENCY.with_label_values(&[
            "test-topic",
            "post",
            "dedup_skipped",
            "false",
        ]);
        let before = histo.get_sample_count();
        histo.observe(0.001);
        assert_eq!(histo.get_sample_count(), before + 1);
    }

    #[test]
    fn add_user_label_audit_note_falls_back_to_audit_note_when_info_empty() {
        let action = EnforcementAction::add_user_label(
            7,
            vec![SPAM_HIGH_RECALL_LABEL.into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        assert_eq!(action.audit_note(), AUDIT_NOTE.as_str());
    }

    #[test]
    fn suspend_user_audit_note_is_just_the_audit_note() {
        let action = EnforcementAction::suspend_user(
            42,
            false,
            POLICY_IN_VIOLATION,
            vec!["model_version: v1".to_owned()],
        );
        assert_eq!(action.audit_note(), AUDIT_NOTE.as_str());
    }

    #[test]
    fn from_spec_dispatches_on_action_kind() {
        let suspend = ActionSpec::SuspendUser {
            perm: true,
            policy: "TestPolicy".into(),
        };
        let action = EnforcementAction::from_spec(&suspend, 1, 1, vec!["k: v".into()]);
        let v = action.to_json();
        assert_eq!(v["suspendUser"]["perm"], true);
        assert_eq!(v["suspendUser"]["policy"], "TestPolicy");

        let label = ActionSpec::AddLabelsV2 {
            labels: vec!["LowQualityReply".into()],
            ttl_msec: Some(7 * 24 * 60 * 60 * 1000),
        };
        let action = EnforcementAction::from_spec(&label, 2, 2, vec![]);
        let v = action.to_json();
        let label_id_type = &v["addLabelsV2"]["labels"][0]["nonPolicyBasedLabel"]["label"]["labelReferenceWithSafetyLabel"]
            ["labelReference"]["labelIdType"];
        assert_eq!(label_id_type["nativeUserLabels"], "LowQualityReply");

        let post_label = ActionSpec::AddPostLabelsV2 {
            labels: vec!["RiskyHighVizReply".into()],
            ttl_msec: Some(7 * 24 * 60 * 60 * 1000),
        };
        let action = EnforcementAction::from_spec(&post_label, 555, 999, vec![]);
        assert!(matches!(
            action,
            EnforcementAction::AddPostLabelsV2 { tweet_id: 555, .. }
        ));
        assert_eq!(action.name(), "addPostLabelsV2");

        let arkose = EnforcementAction::from_spec(&ActionSpec::Arkose, 3, 3, vec![]);
        assert!(matches!(
            arkose,
            EnforcementAction::BounceArkose { user_id: 3, .. }
        ));
        assert_eq!(arkose.name(), "bounceArkose");
        let captcha = EnforcementAction::from_spec(&ActionSpec::Captcha, 4, 4, vec![]);
        assert!(matches!(
            captcha,
            EnforcementAction::BounceCaptcha { user_id: 4, .. }
        ));
        assert_eq!(captcha.name(), "bounceCaptcha");
    }

    #[test]
    fn add_post_label_serializes_to_tweet_safety_shape() {
        let action = EnforcementAction::add_post_label(
            777,
            vec!["RiskyHighVizReply".into()],
            Some(7 * 24 * 60 * 60 * 1000),
            vec!["model_version: v51".to_owned()],
        );
        assert_eq!(action.name(), "addPostLabelsV2");
        assert_eq!(action.entity_id(), 777);
        let v = action.to_json();
        let inner = v.get("addLabelsV2").expect("addLabelsV2 key");

        let entity = json!({"entity": {"entityType": "Tweet", "entityId": {"longId": 777}}});
        assert_eq!(inner["entityId"], entity);
        assert_eq!(inner["entityRef"], entity);
        assert_eq!(inner["violatingUserIds"], json!([]));

        let label_ref = &inner["labels"][0]["nonPolicyBasedLabel"]["label"]["labelReferenceWithSafetyLabel"]
            ["labelReference"];
        assert_eq!(label_ref["entityType"], "Tweet");
        assert_eq!(label_ref["labelType"], "Safety");
        assert_eq!(
            label_ref["labelIdType"]["safetyTweetLabels"],
            "RiskyHighVizReply"
        );
        assert!(label_ref["labelIdType"]["nativeUserLabels"].is_null());

        let expires = inner["labels"][0]["nonPolicyBasedLabel"]["label"]
            ["labelReferenceWithSafetyLabel"]["safetyLabel"]["expiresAtMsec"]
            .as_i64()
            .expect("expiresAtMsec i64");
        assert!(expires > 0);

        let note = action.audit_note();
        assert!(note.starts_with("model_version: v51; "));
        assert!(note.contains("Automated enforcement by"));
    }

    #[test]
    fn add_post_label_without_ttl_emits_empty_safety_label() {
        let action =
            EnforcementAction::add_post_label(777, vec!["SpamHighRecall".into()], None, vec![]);
        let v = action.to_json();
        let safety_label = &v["addLabelsV2"]["labels"][0]["nonPolicyBasedLabel"]["label"]["labelReferenceWithSafetyLabel"]
            ["safetyLabel"];
        assert_eq!(*safety_label, json!({}));
        assert!(safety_label["expiresAtMsec"].is_null());
    }

    #[test]
    fn partition_contiguous_keeps_interleaved_order() {
        let plugin_a = EnforcementAction::add_user_label(
            1,
            vec![SPAM_HIGH_RECALL_LABEL.into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        let post = EnforcementAction::add_post_label(
            99,
            vec!["RiskyHighVizReply".into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        let plugin_b = EnforcementAction::bounce_captcha(1, vec![]);
        let groups = partition_contiguous_by_actor_header(vec![plugin_a, post, plugin_b]);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].1.len(), 1);
        assert_eq!(groups[0].1[0].name(), "addLabelsV2");
        assert_eq!(groups[1].1.len(), 1);
        assert_eq!(groups[1].1[0].name(), "addPostLabelsV2");
        assert_eq!(
            groups[1].0,
            json!({"botmakerBot": {"botId": STUB_BOTMAKER_BOT_ID}})
        );
        assert_eq!(groups[2].1.len(), 1);
        assert_eq!(groups[2].1[0].name(), "bounceCaptcha");
    }

    #[test]
    fn partition_contiguous_batches_adjacent_same_actor() {
        let plugin_a = EnforcementAction::add_user_label(
            1,
            vec![SPAM_HIGH_RECALL_LABEL.into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        let plugin_b = EnforcementAction::bounce_captcha(1, vec![]);
        let groups = partition_contiguous_by_actor_header(vec![plugin_a, plugin_b]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1.len(), 2);
        assert_eq!(groups[0].1[0].name(), "addLabelsV2");
        assert_eq!(groups[0].1[1].name(), "bounceCaptcha");
    }

    #[test]
    fn actor_header_is_stub_botmaker_for_post_labels_only() {
        let post_label = EnforcementAction::add_post_label(
            777,
            vec!["RiskyHighVizReply".into()],
            Some(7 * 24 * 60 * 60 * 1000),
            vec![],
        );
        assert_eq!(
            post_label.actor_header(),
            json!({"botmakerBot": {"botId": STUB_BOTMAKER_BOT_ID}})
        );
        let payload = build_intake_request(
            post_label.actor_header(),
            post_label.to_json(),
            AIS_TOOL,
            Some(post_label.audit_note()),
        );
        assert_eq!(payload["actorHeader"], json!({"botmakerBot": {"botId": 0}}));

        let plugin_header = json!({"plugin": {"pluginId": PLUGIN_ID}});
        let user_label = EnforcementAction::add_user_label(
            7,
            vec![SPAM_HIGH_RECALL_LABEL.into()],
            Some(LABEL_EXPIRY_MSEC),
            vec![],
        );
        assert_eq!(user_label.actor_header(), plugin_header);
        let suspend = EnforcementAction::suspend_user(42, false, POLICY_IN_VIOLATION, vec![]);
        assert_eq!(suspend.actor_header(), plugin_header);
        let arkose = EnforcementAction::bounce_arkose(42, vec![]);
        assert_eq!(arkose.actor_header(), plugin_header);
        let captcha = EnforcementAction::bounce_captcha(42, vec![]);
        assert_eq!(captcha.actor_header(), plugin_header);
        let liveness = EnforcementAction::spam_liveness_check(42, vec![]);
        assert_eq!(liveness.actor_header(), plugin_header);
    }


    #[test]
    fn bounce_arkose_serializes_to_ais_shape() {
        let action = EnforcementAction::bounce_arkose(42, vec!["model_version: v1".to_owned()]);
        let v = action.to_json();
        let inner = v.get("bounceViaSelection").expect("bounceViaSelection key");
        assert_eq!(inner["target"], json!({"user": {"userId": 42}}));
        assert_eq!(inner["tags"], json!([]));
        assert_eq!(inner["uncheckedTags"], json!(ARKOSE_BOUNCE_TAGS));
        assert_eq!(
            inner["bounceActor"],
            json!({"simpleService": {"serviceName": PLUGIN_ID}})
        );
        assert!(inner.get("note").is_none());

        let note = action.audit_note();
        assert!(note.starts_with("model_version: v1; "));
        assert!(note.contains("Automated enforcement by"));
    }

    #[test]
    fn bounce_captcha_uses_captcha_tag_set() {
        let action = EnforcementAction::bounce_captcha(7, vec![]);
        let v = action.to_json();
        let inner = v.get("bounceViaSelection").expect("bounceViaSelection key");
        assert_eq!(inner["target"], json!({"user": {"userId": 7}}));
        assert_eq!(inner["uncheckedTags"], json!(CAPTCHA_BOUNCE_TAGS));
    }


    #[test]
    fn spam_liveness_check_serializes_to_ais_shape() {
        let action =
            EnforcementAction::spam_liveness_check(42, vec!["model_version: v1".to_owned()]);
        assert_eq!(action.name(), "spamLivenessCheck");
        assert_eq!(action.entity_id(), 42);
        let v = action.to_json();
        let inner = v.get("bounceViaTemplate").expect("bounceViaTemplate key");
        assert_eq!(inner["userId"], 42);
        assert_eq!(inner["templateId"], SPAM_LIVENESS_CHECK_TEMPLATE_ID);
        assert_eq!(inner["policyTag"], SPAM_LIVENESS_CHECK_POLICY_TAG);
        assert_eq!(inner["additionalTags"], json!([]));
        assert_eq!(
            inner["bounceActor"],
            json!({"simpleService": {"serviceName": PLUGIN_ID}})
        );
        assert!(inner.get("note").is_none());

        let note = action.audit_note();
        assert!(note.starts_with("model_version: v1; "));
        assert!(note.contains("Automated enforcement by"));
    }

    #[test]
    fn from_spec_builds_spam_liveness_check() {
        let action = EnforcementAction::from_spec(&ActionSpec::SpamLivenessCheck, 9, 9, vec![]);
        assert!(matches!(
            action,
            EnforcementAction::SpamLivenessCheck { user_id: 9, .. }
        ));
        let v = action.to_json();
        assert_eq!(
            v["bounceViaTemplate"]["templateId"],
            SPAM_LIVENESS_CHECK_TEMPLATE_ID
        );
        assert_eq!(
            v["bounceViaTemplate"]["policyTag"],
            SPAM_LIVENESS_CHECK_POLICY_TAG
        );
    }

    #[test]
    fn bounce_audit_note_falls_back_to_audit_note_when_info_empty() {
        let arkose = EnforcementAction::bounce_arkose(1, vec![]);
        assert_eq!(arkose.audit_note(), AUDIT_NOTE.as_str());
        let captcha = EnforcementAction::bounce_captcha(1, vec![]);
        assert_eq!(captcha.audit_note(), AUDIT_NOTE.as_str());
    }

    #[test]
    fn audit_note_contains_pkg_version_and_git_stamp() {
        let note = AUDIT_NOTE.as_str();
        assert!(note.starts_with(&format!(
            "Automated enforcement by {} {} (",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION"),
        )));
        assert!(note.contains(" @ "));
        assert!(note.ends_with(']'));
        assert!(note.contains("[env: "));
    }
}
