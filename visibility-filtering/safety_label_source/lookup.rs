use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use tonic::async_trait;
use xai_visibility_filtering_proto as vf_pb;

use super::metrics::{self, BatchStage};
use super::types::{FailureKind, FallbackReason, LabelSource, ManhattanOutcome, TwemcacheOutcome};
use super::warmer::Warmer;

#[derive(Debug, Clone, thiserror::Error)]
#[error("{kind:?}: {message}")]
pub struct LookupError {
    pub(crate) kind: FailureKind,
    pub(crate) message: String,
}

impl LookupError {
    pub(crate) fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

pub(crate) type LookupResults = FxHashMap<u64, Result<vf_pb::SafetyLabelMap, LookupError>>;

#[async_trait]
pub(crate) trait TwemcacheLookup: Send + Sync {
    async fn get(&self, ids: &[u64]) -> FxHashMap<u64, TwemcacheOutcome>;
}

#[async_trait]
pub(crate) trait ManhattanLookup: Send + Sync {
    async fn get(&self, ids: &[u64]) -> FxHashMap<u64, ManhattanOutcome>;
}

pub(crate) struct RemoteSource {
    twemcache: Arc<dyn TwemcacheLookup>,
    manhattan: Arc<dyn ManhattanLookup>,
    warmer: Option<Arc<dyn Warmer>>,
}

impl RemoteSource {
    pub(crate) fn new<T, M>(twemcache: Arc<T>, manhattan: Arc<M>) -> Self
    where
        T: TwemcacheLookup + 'static,
        M: ManhattanLookup + 'static,
    {
        let twemcache: Arc<dyn TwemcacheLookup> = twemcache;
        let manhattan: Arc<dyn ManhattanLookup> = manhattan;
        Self {
            twemcache,
            manhattan,
            warmer: None,
        }
    }

    pub(crate) fn with_warmer(mut self, warmer: Arc<dyn Warmer>) -> Self {
        self.warmer = Some(warmer);
        self
    }

    pub(crate) async fn get(&self, ids: &[u64]) -> LookupResults {
        let mut twemcache_results = self.twemcache.get(ids).await;
        let mut results = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        let mut fallback_ids = Vec::new();
        let mut fallback_counts: BTreeMap<FallbackReason, usize> = BTreeMap::new();
        let mut warm_ids = Vec::new();

        for &tweet_id in ids {
            if results.contains_key(&tweet_id) || fallback_ids.contains(&tweet_id) {
                continue;
            }
            match twemcache_results.remove(&tweet_id) {
                Some(TwemcacheOutcome::Hit(label_map)) => {
                    results.insert(tweet_id, Ok(label_map));
                }
                Some(TwemcacheOutcome::NotFound) => {
                    results.insert(
                        tweet_id,
                        Ok(vf_pb::SafetyLabelMap {
                            labels: HashMap::new(),
                        }),
                    );
                }
                Some(TwemcacheOutcome::Miss) => {
                    fallback_ids.push(tweet_id);
                    if self.warmer.is_some() {
                        warm_ids.push(tweet_id);
                    }
                }
                Some(TwemcacheOutcome::FallThrough(reason)) => {
                    fallback_ids.push(tweet_id);
                    *fallback_counts.entry(reason).or_insert(0) += 1;
                }
                None => {
                    fallback_ids.push(tweet_id);
                    *fallback_counts
                        .entry(FallbackReason::MissingResponse)
                        .or_insert(0) += 1;
                }
            }
        }

        for (reason, count) in fallback_counts {
            metrics::record_cache_fallback_keys(LabelSource::Twemcache, reason, count);
        }

        if let Some(warmer) = &self.warmer
            && !warm_ids.is_empty()
        {
            warmer.warm(warm_ids);
        }

        metrics::record_batch_size(BatchStage::ManhattanFallback, fallback_ids.len());
        let mut manhattan_results = self.manhattan.get(&fallback_ids).await;
        for tweet_id in fallback_ids {
            match manhattan_results.remove(&tweet_id) {
                Some(ManhattanOutcome::Resolved(label_map)) => {
                    results.insert(tweet_id, Ok(label_map));
                }
                Some(ManhattanOutcome::Failure(failure)) => {
                    results.insert(tweet_id, Err(failure));
                }
                None => {
                    results.insert(
                        tweet_id,
                        Err(LookupError::new(
                            FailureKind::ManhattanFetch,
                            "missing from MH response",
                        )),
                    );
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct FakeTwemcache {
        results: Mutex<HashMap<u64, TwemcacheOutcome>>,
    }

    impl FakeTwemcache {
        fn new(results: HashMap<u64, TwemcacheOutcome>) -> Arc<Self> {
            Arc::new(Self {
                results: Mutex::new(results),
            })
        }
    }

    #[async_trait]
    impl TwemcacheLookup for FakeTwemcache {
        async fn get(&self, ids: &[u64]) -> FxHashMap<u64, TwemcacheOutcome> {
            let mut results = self.results.lock().unwrap();
            ids.iter()
                .filter_map(|id| results.remove(id).map(|result| (*id, result)))
                .collect()
        }
    }

    struct FakeManhattan {
        results: Mutex<HashMap<u64, ManhattanOutcome>>,
        calls: Mutex<Vec<Vec<u64>>>,
    }

    impl FakeManhattan {
        fn new(results: HashMap<u64, ManhattanOutcome>) -> Arc<Self> {
            Arc::new(Self {
                results: Mutex::new(results),
                calls: Mutex::new(Vec::new()),
            })
        }

        fn calls(&self) -> Vec<Vec<u64>> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ManhattanLookup for FakeManhattan {
        async fn get(&self, ids: &[u64]) -> FxHashMap<u64, ManhattanOutcome> {
            self.calls.lock().unwrap().push(ids.to_vec());
            let mut results = self.results.lock().unwrap();
            ids.iter()
                .filter_map(|id| results.remove(id).map(|result| (*id, result)))
                .collect()
        }
    }

    struct FakeWarmer {
        published: Mutex<Vec<Vec<u64>>>,
    }

    impl FakeWarmer {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                published: Mutex::new(Vec::new()),
            })
        }

        fn published(&self) -> Vec<Vec<u64>> {
            self.published.lock().unwrap().clone()
        }
    }

    impl Warmer for FakeWarmer {
        fn warm(&self, miss_ids: Vec<u64>) {
            self.published.lock().unwrap().push(miss_ids);
        }
    }

    fn empty_label_map() -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn duplicate_ids_resolve_once_without_a_phantom_miss() {
        let twemcache = FakeTwemcache::new(HashMap::from([(42, TwemcacheOutcome::Miss)]));
        let manhattan = FakeManhattan::new(HashMap::from([(
            42,
            ManhattanOutcome::Resolved(empty_label_map()),
        )]));
        let source = RemoteSource::new(Arc::clone(&twemcache), Arc::clone(&manhattan));

        let results = source.get(&[42, 42]).await;

        assert!(results.get(&42).unwrap().is_ok());
        assert_eq!(manhattan.calls(), vec![vec![42]]);
    }

    #[tokio::test]
    async fn get_missing_from_manhattan_is_fetch_error() {
        let twemcache = FakeTwemcache::new(HashMap::from([(42, TwemcacheOutcome::Miss)]));
        let manhattan = FakeManhattan::new(HashMap::new());
        let source = RemoteSource::new(Arc::clone(&twemcache), Arc::clone(&manhattan));

        let results = source.get(&[42]).await;

        assert!(matches!(
            results.get(&42).unwrap(),
            Err(failure) if failure.kind == FailureKind::ManhattanFetch
        ));
        assert_eq!(manhattan.calls(), vec![vec![42]]);
    }

    #[tokio::test]
    async fn plain_miss_publishes_to_warmer() {
        let twemcache = FakeTwemcache::new(HashMap::from([
            (1, TwemcacheOutcome::Hit(empty_label_map())),
            (2, TwemcacheOutcome::NotFound),
            (3, TwemcacheOutcome::Miss),
            (4, TwemcacheOutcome::FallThrough(FallbackReason::Timeout)),
        ]));
        let manhattan = FakeManhattan::new(HashMap::from([
            (3, ManhattanOutcome::Resolved(empty_label_map())),
            (4, ManhattanOutcome::Resolved(empty_label_map())),
            (5, ManhattanOutcome::Resolved(empty_label_map())),
        ]));
        let warmer = FakeWarmer::new();
        let source = RemoteSource::new(Arc::clone(&twemcache), Arc::clone(&manhattan))
            .with_warmer(Arc::<FakeWarmer>::clone(&warmer));

        let results = source.get(&[1, 2, 3, 4, 5]).await;

        assert_eq!(results.len(), 5);
        assert!(results.values().all(Result::is_ok));
        assert_eq!(manhattan.calls(), vec![vec![3, 4, 5]]);
        assert_eq!(warmer.published(), vec![vec![3]]);
    }

    #[tokio::test]
    async fn full_warm_channel_does_not_affect_fallback_result() {
        use super::super::warmer::CacheWarmer;

        let (warmer, _rx) = CacheWarmer::without_drain_task(1);
        let warmer = Arc::new(warmer);
        warmer.warm(vec![0]);

        let twemcache = FakeTwemcache::new(HashMap::from([(42, TwemcacheOutcome::Miss)]));
        let manhattan = FakeManhattan::new(HashMap::from([(
            42,
            ManhattanOutcome::Resolved(empty_label_map()),
        )]));
        let source =
            RemoteSource::new(Arc::clone(&twemcache), Arc::clone(&manhattan)).with_warmer(warmer);

        let results = source.get(&[42]).await;

        assert!(results.get(&42).unwrap().is_ok());
        assert_eq!(manhattan.calls(), vec![vec![42]]);
    }

    #[tokio::test]
    async fn get_mixed_results_merges_cache_and_manhattan() {
        let twemcache = FakeTwemcache::new(HashMap::from([
            (1, TwemcacheOutcome::Hit(empty_label_map())),
            (2, TwemcacheOutcome::Miss),
            (3, TwemcacheOutcome::FallThrough(FallbackReason::Decode)),
        ]));
        let manhattan = FakeManhattan::new(HashMap::from([
            (2, ManhattanOutcome::Resolved(empty_label_map())),
            (
                3,
                ManhattanOutcome::Failure(LookupError::new(FailureKind::ManhattanDecode, "decode")),
            ),
        ]));
        let source = RemoteSource::new(Arc::clone(&twemcache), Arc::clone(&manhattan));

        let results = source.get(&[1, 2, 3]).await;

        assert_eq!(results.len(), 3);
        assert!(results.get(&1).unwrap().is_ok());
        assert!(results.get(&2).unwrap().is_ok());
        assert!(results.get(&3).unwrap().is_err());
        assert_eq!(manhattan.calls(), vec![vec![2, 3]]);
    }
}
