use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use super::cached_value::{self, CacheLookup};
use super::lookup::TwemcacheLookup;
use super::metrics::{self, CacheResult, CacheTier, SourceOutcome};
use super::types::{FallbackReason, LabelSource, TwemcacheOutcome};
use tonic::async_trait;
use xai_cache::{CacheClient, KVCacheError, Key, Value};

const KEY_PREFIX: &str = "slm_";

type CacheOpResult<T> = std::result::Result<T, KVCacheError>;

#[async_trait]
pub(crate) trait CacheRead: Send + Sync + 'static {
    async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>>;
}

#[async_trait]
impl CacheRead for CacheClient {
    async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>> {
        match CacheClient::multi_get(self, keys).await {
            Ok(map) => map,
            Err(e) => keys.iter().cloned().map(|k| (k, Err(e.clone()))).collect(),
        }
    }
}

pub(crate) struct TwemcacheSource {
    cache: Arc<dyn CacheRead>,
}

#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "KVCacheError belongs to the shared xai-cache client; only its timeout and backpressure errors get their own FallbackReason, and every other error counts as Other, including any the client adds"
)]
fn fallback_reason(e: &KVCacheError) -> FallbackReason {
    match e {
        KVCacheError::Timeout(_) => FallbackReason::Timeout,
        KVCacheError::Backpressure => FallbackReason::Backpressure,
        _ => FallbackReason::Other,
    }
}

#[derive(Default)]
struct ItemCounts {
    hit: usize,
    not_found: usize,
    miss: usize,
    fallback: BTreeMap<FallbackReason, usize>,
}

impl ItemCounts {
    fn record_fallback(&mut self, reason: FallbackReason) {
        *self.fallback.entry(reason).or_default() += 1;
    }

    fn record_metrics(&self) {
        metrics::record_cache_keys(CacheTier::Twemcache, CacheResult::Hit, self.hit);
        metrics::record_cache_keys(CacheTier::Twemcache, CacheResult::NotFound, self.not_found);
        metrics::record_cache_keys(CacheTier::Twemcache, CacheResult::Miss, self.miss);
    }
}

impl TwemcacheSource {
    pub(crate) fn new(cache: Arc<CacheClient>) -> Self {
        Self { cache }
    }

    #[cfg(test)]
    pub(crate) fn with_cache(cache: Arc<dyn CacheRead>) -> Self {
        Self { cache }
    }
}

#[async_trait]
impl TwemcacheLookup for TwemcacheSource {
    async fn get(&self, ids: &[u64]) -> FxHashMap<u64, TwemcacheOutcome> {
        let mut results = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        if ids.is_empty() {
            return results;
        }

        let start = Instant::now();

        let keys: Vec<Key> = match ids
            .iter()
            .map(|&id| Key::new(format!("{KEY_PREFIX}{id}").into_bytes()))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(keys) => keys,
            Err(_) => {
                for &tweet_id in ids {
                    results.insert(
                        tweet_id,
                        TwemcacheOutcome::FallThrough(FallbackReason::Other),
                    );
                }
                metrics::record_source_request(
                    LabelSource::Twemcache,
                    SourceOutcome::Failure,
                    start.elapsed().as_secs_f64() * 1000.0,
                );
                return results;
            }
        };

        let mut counts = ItemCounts::default();

        let map = self.cache.multi_get(&keys).await;
        for (&tweet_id, key) in ids.iter().zip(&keys) {
            let outcome = match map.get(key) {
                Some(Ok(Some(bytes))) => match cached_value::decode(bytes) {
                    CacheLookup::Hit(label_map) => {
                        counts.hit += 1;
                        TwemcacheOutcome::Hit(label_map)
                    }
                    CacheLookup::NotFound => {
                        counts.not_found += 1;
                        TwemcacheOutcome::NotFound
                    }
                    CacheLookup::Miss => {
                        counts.miss += 1;
                        TwemcacheOutcome::Miss
                    }
                    CacheLookup::DecodeError | CacheLookup::DecodePanic => {
                        counts.record_fallback(FallbackReason::Decode);
                        TwemcacheOutcome::FallThrough(FallbackReason::Decode)
                    }
                },
                Some(Ok(None)) => {
                    counts.miss += 1;
                    TwemcacheOutcome::Miss
                }
                Some(Err(e)) => {
                    let reason = fallback_reason(e);
                    counts.record_fallback(reason);
                    TwemcacheOutcome::FallThrough(reason)
                }
                None => {
                    counts.record_fallback(FallbackReason::MissingResponse);
                    TwemcacheOutcome::FallThrough(FallbackReason::MissingResponse)
                }
            };
            results.insert(tweet_id, outcome);
        }
        metrics::record_source_request(
            LabelSource::Twemcache,
            SourceOutcome::Success,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        counts.record_metrics();

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum FakeTwemcacheMode {
        Empty,
        MissingResponse,
        PerKeyError(KVCacheError),
        WithData(Vec<u8>),
    }

    struct FakeTwemcache {
        mode: FakeTwemcacheMode,
    }

    impl FakeTwemcache {
        fn empty() -> Arc<Self> {
            Arc::new(Self {
                mode: FakeTwemcacheMode::Empty,
            })
        }

        fn with_missing_response() -> Arc<Self> {
            Arc::new(Self {
                mode: FakeTwemcacheMode::MissingResponse,
            })
        }

        fn with_per_key_error(err: KVCacheError) -> Arc<Self> {
            Arc::new(Self {
                mode: FakeTwemcacheMode::PerKeyError(err),
            })
        }

        fn with_data(data: Vec<u8>) -> Arc<Self> {
            Arc::new(Self {
                mode: FakeTwemcacheMode::WithData(data),
            })
        }
    }

    #[async_trait]
    impl CacheRead for FakeTwemcache {
        async fn multi_get(&self, keys: &[Key]) -> HashMap<Key, CacheOpResult<Option<Value>>> {
            match &self.mode {
                FakeTwemcacheMode::Empty => keys.iter().map(|k| (k.clone(), Ok(None))).collect(),
                FakeTwemcacheMode::MissingResponse => HashMap::new(),
                FakeTwemcacheMode::PerKeyError(err) => {
                    keys.iter().map(|k| (k.clone(), Err(err.clone()))).collect()
                }
                FakeTwemcacheMode::WithData(data) => keys
                    .iter()
                    .map(|k| (k.clone(), Ok(Some(data.clone()))))
                    .collect(),
            }
        }
    }

    async fn get_with_cache(cache: Arc<dyn CacheRead>) -> FxHashMap<u64, TwemcacheOutcome> {
        TwemcacheSource::with_cache(cache).get(&[42]).await
    }

    #[tokio::test]
    async fn get_cache_hit_returns_label_map() {
        let results = get_with_cache(FakeTwemcache::with_data(
            crate::safety_label_source::cached_value::tests::cached_value_found_mval(),
        ))
        .await;

        assert!(matches!(results.get(&42), Some(TwemcacheOutcome::Hit(_))));
    }

    #[tokio::test]
    async fn get_negative_cache_returns_empty_label_map() {
        let results = get_with_cache(FakeTwemcache::with_data(
            crate::safety_label_source::cached_value::tests::cached_value_not_found(),
        ))
        .await;

        assert!(matches!(results.get(&42), Some(TwemcacheOutcome::NotFound)));
    }

    #[tokio::test]
    async fn cache_transport_results_select_lookup_outcome() {
        let cases: Vec<(Arc<dyn CacheRead>, TwemcacheOutcome)> = vec![
            (FakeTwemcache::empty(), TwemcacheOutcome::Miss),
            (
                FakeTwemcache::with_per_key_error(KVCacheError::Io("conn refused".into())),
                TwemcacheOutcome::FallThrough(FallbackReason::Other),
            ),
            (
                FakeTwemcache::with_per_key_error(KVCacheError::Timeout("get".into())),
                TwemcacheOutcome::FallThrough(FallbackReason::Timeout),
            ),
            (
                FakeTwemcache::with_per_key_error(KVCacheError::Backpressure),
                TwemcacheOutcome::FallThrough(FallbackReason::Backpressure),
            ),
            (
                FakeTwemcache::with_missing_response(),
                TwemcacheOutcome::FallThrough(FallbackReason::MissingResponse),
            ),
        ];

        for (cache, expected) in cases {
            let results = get_with_cache(cache).await;
            assert_eq!(results.get(&42), Some(&expected));
        }
    }

    #[tokio::test]
    async fn get_decode_error_returns_fallback() {
        let results = get_with_cache(FakeTwemcache::with_data(vec![0xff; 16])).await;

        assert!(matches!(
            results.get(&42),
            Some(TwemcacheOutcome::FallThrough(FallbackReason::Decode))
        ));
    }
}
