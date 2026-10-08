use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Duration, Instant};

use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use prost::Message;
use tonic::async_trait;
use tracing::warn;
use xai_ads_injection_proto::ads_injected_timeline::AuthorBrandSafetyHistory;
use xai_manhattan::{s2s::S2sConfig, Key, NativeManhattanClient, Tenant};
use xai_redis_client::{XdsRedisClient, XdsRedisConfig};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use crate::models::brand_safety::AuthorBrandSafetyFallback;

const MH_CLUSTER: &str = "nash";
const MH_APP_ID: &str = "brand_safety";
const MH_DATASET: &str = "author_brand_safety_history";
const MH_TIMEOUT: Duration = Duration::from_millis(100);
const MAX_MANHATTAN_KEYS_PER_SEC: NonZeroU32 = NonZeroU32::new(30).unwrap();

const CACHE_NAME: &str = "author-brand-safety-cache";
const CACHE_KEY_PREFIX: &str = "abs:1:";
const CACHE_TTL_SECS: u64 = 300;
const CACHE_TIMEOUT: Duration = Duration::from_millis(20);

const CACHE_METRIC_NAME: &str = "AuthorBrandSafetyClient.cache";
const MANHATTAN_METRIC_NAME: &str = "AuthorBrandSafetyClient.manhattan";

#[async_trait]
pub trait AuthorBrandSafetyClient: Send + Sync {
    async fn fetch(
        &self,
        author_ids: &[u64],
    ) -> Result<HashMap<u64, Option<AuthorBrandSafetyFallback>>, String>;
}

pub struct ProdAuthorBrandSafetyClient {
    cache: Arc<XdsRedisClient>,
    mh: NativeManhattanClient,
    manhattan_budget: DefaultDirectRateLimiter,
}

impl ProdAuthorBrandSafetyClient {
    pub async fn new(datacenter: &str) -> anyhow::Result<Self> {
        let eds_resource_name = eds_resource_name(datacenter);
        let cache = XdsRedisClient::new_with_response_timeout(
            XdsRedisConfig {
                eds_resource_name: eds_resource_name.clone(),
            },
            CACHE_TIMEOUT,
        )
        .await
        .map_err(|e| anyhow::anyhow!("ProdAuthorBrandSafetyClient cache build failed: {e}"))?;
        if !cache.is_ready() {
            warn!("{eds_resource_name} has no endpoints yet; reading Manhattan until it does");
        }

        let s2s = S2sConfig {
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            ca_cert_path: S2S_CHAIN_PATH.clone(),
        };
        let mh = NativeManhattanClient::builder_from_tenant_s2s(&tenant(), datacenter, s2s)
            .timeout(MH_TIMEOUT)
            .retries(0)
            .multiplexed()
            .no_batch()
            .build()
            .await
            .map_err(|e| {
                anyhow::anyhow!("ProdAuthorBrandSafetyClient Manhattan build failed: {e}")
            })?;

        Ok(Self {
            cache: Arc::new(cache),
            mh,
            manhattan_budget: RateLimiter::direct(Quota::per_second(MAX_MANHATTAN_KEYS_PER_SEC)),
        })
    }

    async fn cache_get(&self, author_ids: &[u64]) -> HashMap<u64, Vec<u8>> {
        let keys: Vec<Vec<u8>> = author_ids.iter().map(|&id| cache_key(id)).collect();
        let start = Instant::now();
        let found = self.cache.mget_raw_detailed(&keys, CACHE_TIMEOUT).await;
        observe_latency(CACHE_METRIC_NAME, start);
        let errors = found.failed.len();
        let rows: HashMap<u64, Vec<u8>> = author_ids
            .iter()
            .zip(found.values)
            .filter_map(|(&id, row)| Some((id, row?)))
            .collect();
        let misses = author_ids.len().saturating_sub(rows.len() + errors);
        record_keys(
            CACHE_METRIC_NAME,
            &[("hit", rows.len()), ("miss", misses), ("error", errors)],
        );
        rows
    }

    fn cache_put(&self, rows: Vec<(u64, Vec<u8>)>) {
        if rows.is_empty() {
            return;
        }
        let cache = Arc::clone(&self.cache);
        tokio::spawn(async move {
            let entries: Vec<(Vec<u8>, Vec<u8>)> = rows
                .into_iter()
                .map(|(id, row)| (cache_key(id), row))
                .collect();
            cache
                .mset_ex_raw(&entries, CACHE_TTL_SECS, CACHE_TIMEOUT)
                .await;
        });
    }

    async fn manhattan_get(&self, author_ids: &[u64]) -> HashMap<u64, Option<Vec<u8>>> {
        if author_ids.is_empty() {
            return HashMap::new();
        }
        let keys: Vec<(Key<'_>, Key<'_>)> = author_ids
            .iter()
            .map(|&id| (Key::from([manhattan_key(id)]), Key::new()))
            .collect();
        let start = Instant::now();
        let batch = self.mh.batch_get(tenant(), keys).await;
        observe_latency(MANHATTAN_METRIC_NAME, start);
        match batch {
            Ok(batch) => author_ids
                .iter()
                .zip(batch.items)
                .filter_map(|(&id, item)| {
                    Some((id, item.ok()?.map(|i| i.value().as_bytes().to_vec())))
                })
                .collect(),
            Err(e) => {
                warn!("author brand safety batch_get failed: {e}");
                HashMap::new()
            }
        }
    }
}

fn eds_resource_name(datacenter: &str) -> String {
    format!(
        "xdstp://discovery-{datacenter}/envoy.config.endpoint.v3.ClusterLoadAssignment/redis-global.{CACHE_NAME}.prod.cache:redis"
    )
}

fn tenant() -> Tenant {
    Tenant {
        cluster: MH_CLUSTER.to_string(),
        app_id: MH_APP_ID.to_string(),
        dataset: MH_DATASET.to_string(),
    }
}

fn cache_key(author_id: u64) -> Vec<u8> {
    format!("{CACHE_KEY_PREFIX}{author_id}").into_bytes()
}

fn manhattan_key(author_id: u64) -> Vec<u8> {
    (author_id as i64).to_be_bytes().to_vec()
}

fn decode_fallback(bytes: &[u8]) -> Option<AuthorBrandSafetyFallback> {
    let history = AuthorBrandSafetyHistory::decode(bytes).ok()?;
    AuthorBrandSafetyFallback::from_proto(history.fallback_label())
}

fn record_keys(metric: &str, counts: &[(&str, usize)]) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    for &(result, count) in counts {
        if count > 0 {
            receiver.incr(metric, &[("result", result)], count as u64);
        }
    }
}

fn observe_latency(metric: &str, start: Instant) {
    if let Some(receiver) = global_stats_receiver() {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        receiver.observe(metric, &[], latency_ms, HistogramBuckets::Bucket0To50);
    }
}

#[async_trait]
impl AuthorBrandSafetyClient for ProdAuthorBrandSafetyClient {
    async fn fetch(
        &self,
        author_ids: &[u64],
    ) -> Result<HashMap<u64, Option<AuthorBrandSafetyFallback>>, String> {
        let cached = self.cache_get(author_ids).await;
        let (allowed, capped): (Vec<u64>, Vec<u64>) = author_ids
            .iter()
            .copied()
            .filter(|id| !cached.contains_key(id))
            .partition(|_| self.manhattan_budget.check().is_ok());
        let rows = self.manhattan_get(&allowed).await;
        record_keys(
            MANHATTAN_METRIC_NAME,
            &[
                ("ok", rows.len()),
                ("failed", allowed.len() - rows.len()),
                ("capped", capped.len()),
            ],
        );

        let mut fallbacks: HashMap<u64, Option<AuthorBrandSafetyFallback>> = cached
            .iter()
            .map(|(&id, row)| (id, decode_fallback(row)))
            .collect();
        let mut write_back = Vec::with_capacity(rows.len());
        for (id, row) in rows {
            fallbacks.insert(id, row.as_deref().and_then(decode_fallback));
            write_back.push((id, row.unwrap_or_default()));
        }
        self.cache_put(write_back);
        Ok(fallbacks)
    }
}

#[derive(Default)]
pub struct MockAuthorBrandSafetyClient {
    pub fallbacks: HashMap<u64, AuthorBrandSafetyFallback>,
}

#[async_trait]
impl AuthorBrandSafetyClient for MockAuthorBrandSafetyClient {
    async fn fetch(
        &self,
        author_ids: &[u64],
    ) -> Result<HashMap<u64, Option<AuthorBrandSafetyFallback>>, String> {
        Ok(author_ids
            .iter()
            .map(|id| (*id, self.fallbacks.get(id).copied()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_ads_injection_proto::ads_injected_timeline::SafetyLabelType;

    #[test]
    fn decodes_fallback_label_from_processor_row() {
        let row = AuthorBrandSafetyHistory {
            fallback_label: SafetyLabelType::GrokNsfaLimited as i32,
            updated_at_ms: 1,
            ..Default::default()
        };
        assert_eq!(
            decode_fallback(&row.encode_to_vec()),
            Some(AuthorBrandSafetyFallback::NsfaLimited)
        );
        assert_eq!(decode_fallback(&[]), None, "cached no-row marker");
        assert_eq!(decode_fallback(b"not a proto"), None);
    }

    #[test]
    fn manhattan_key_matches_processor_encoding() {
        let author_id: i64 = 1_959_000_000_000_000_001;
        assert_eq!(
            manhattan_key(author_id as u64),
            author_id.to_be_bytes().to_vec()
        );
    }

    #[test]
    fn cache_key_is_versioned_author_id() {
        assert_eq!(
            cache_key(1_959_000_000_000_000_001),
            b"abs:1:1959000000000000001"
        );
    }
}
