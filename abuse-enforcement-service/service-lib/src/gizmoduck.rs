use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use tracing::info;
use xai_core_entities::entities::VerifiedType;
use xai_core_entities::gizmoduck_client::{GizmoduckClient, ProdGizmoduckClient, QueryFields};
use xai_core_entities::rpc_constants::{GizmoduckRpcConstants, RpcConstants};
use xai_strato::{StratoGrpc, StratoGrpcConfig};

use crate::facts::GizmoduckFacts;
use crate::gizmoduck_labels::label_ordinal_to_name;
use crate::metrics;

const QUERY_FIELDS: &[QueryFields] = &[QueryFields::SAFETY, QueryFields::LABELS];

#[derive(Debug, Clone)]
pub struct GizmoduckCoreClientConfig {
    pub zone: String,
    pub ca_cert_path: String,
    pub client_cert_path: String,
    pub client_key_path: String,
    pub client_id: String,
}

#[derive(Clone)]
pub struct GizmoduckCoreClient {
    inner: Arc<dyn GizmoduckClient + Send + Sync>,
}

impl GizmoduckCoreClient {
    #[cfg(test)]
    pub(crate) fn from_client(inner: Arc<dyn GizmoduckClient + Send + Sync>) -> Self {
        Self { inner }
    }

    pub async fn connect(cfg: &GizmoduckCoreClientConfig) -> Result<Self> {
        let service_url = format!("gizmoduck.gizmoduck.prod.{}.s2s.twttr.net", cfg.zone);
        let strato_cfg = StratoGrpcConfig {
            ca_cert_path: cfg.ca_cert_path.clone(),
            client_cert_path: cfg.client_cert_path.clone(),
            client_key_path: cfg.client_key_path.clone(),
            num_endpoints: Some(GizmoduckRpcConstants::num_endpoints()),
            max_batch_size: GizmoduckRpcConstants::max_batch_size(),
            connect_timeout_ms: GizmoduckRpcConstants::connect_timeout_msec(),
            request_timeout_ms: GizmoduckRpcConstants::request_timeout_msec(),
            client_id: Some(cfg.client_id.clone()),
            wily_ns_path: "/s/gizmoduck/gizmoduck:fed-grpc".to_string(),
            service_url,
            dns_probe_interval_ms: GizmoduckRpcConstants::dns_probe_interval_msec(),
            zone: cfg.zone.clone(),
            readiness_probe_port: Some(8081),
            filter_to_k8s_only: true,
            ..Default::default()
        };
        let grpc = StratoGrpc::new(strato_cfg)
            .await
            .context("failed to create gizmoduck StratoGrpc client")?;
        info!(
            zone = %cfg.zone,
            client_id = %cfg.client_id,
            "gizmoduck core client initialised (get-V2 fed-grpc)"
        );
        Ok(Self {
            inner: Arc::new(ProdGizmoduckClient {
                grpc_client: Arc::new(grpc),
            }),
        })
    }

    #[tracing::instrument(skip(self), fields(present, suspended, deactivated, label_count))]
    pub async fn fetch_user(&self, user_id: i64) -> Result<GizmoduckFacts> {
        let start = Instant::now();
        let mut results = self
            .inner
            .get_users_with_context(vec![user_id], None, QUERY_FIELDS)
            .await;
        let item = results
            .remove(&user_id)
            .ok_or_else(|| anyhow!("gizmoduck core: empty batch for user_id={user_id}"))?;

        let facts = match item {
            Ok(None) => GizmoduckFacts::default(),
            Ok(Some(result)) => match result.user {
                None => GizmoduckFacts::default(),
                Some(user) => {
                    let labels: Vec<String> = user
                        .labels
                        .labels
                        .iter()
                        .map(|l| label_ordinal_to_name(l.label_value))
                        .collect();
                    GizmoduckFacts {
                        present: true,
                        suspended: user.safety.suspended,
                        deactivated: user.safety.deactivated,
                        labels,
                        is_blue_verified: user.safety.is_blue_verified,
                        verified_type: user.safety.verified_type.map(verified_type_name),
                        is_affiliate: user
                            .safety
                            .verified_organization_details
                            .as_ref()
                            .map(|d| d.is_verified_organization_affiliate)
                            .unwrap_or(false),
                    }
                }
            },
            Err(e) => {
                metrics::GIZMODUCK_LATENCY
                    .with_label_values(&["error"])
                    .observe(start.elapsed().as_secs_f64());
                return Err(e.context(format!("gizmoduck get-V2 failed for user_id={user_id}")));
            }
        };

        metrics::GIZMODUCK_LATENCY
            .with_label_values(&["ok"])
            .observe(start.elapsed().as_secs_f64());

        let span = tracing::Span::current();
        if !facts.present {
            metrics::GIZMODUCK_CHECK_TOTAL
                .with_label_values(&["user_not_found"])
                .inc();
            span.record("present", false);
            return Ok(facts);
        }

        span.record("present", true);
        span.record("suspended", facts.suspended);
        span.record("deactivated", facts.deactivated);
        span.record("label_count", facts.labels.len());

        let state_label = if facts.suspended {
            "suspended"
        } else if facts.deactivated {
            "deactivated"
        } else {
            "active"
        };
        metrics::GIZMODUCK_CHECK_TOTAL
            .with_label_values(&[state_label])
            .inc();

        info!(
            state = state_label,
            has_spam_high_recall = facts.labels.iter().any(|l| l == "SpamHighRecall"),
            label_count = facts.labels.len(),
            "gizmoduck fetch ok",
        );

        Ok(facts)
    }
}

fn verified_type_name(v: VerifiedType) -> String {
    match v {
        VerifiedType::User => "User".to_owned(),
        VerifiedType::Notable => "Notable".to_owned(),
        VerifiedType::Business => "Business".to_owned(),
        VerifiedType::Government => "Government".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use crate::gizmoduck_labels::label_ordinal_to_name;

    #[test]
    fn spam_high_recall_maps_to_pascal_case() {
        assert_eq!(label_ordinal_to_name(23), "SpamHighRecall");
    }

    #[test]
    fn unmapped_ordinals_render_as_unknown_label() {
        assert_eq!(label_ordinal_to_name(3), "UnknownLabel(3)");
        assert_eq!(label_ordinal_to_name(9999), "UnknownLabel(9999)");
    }
}
