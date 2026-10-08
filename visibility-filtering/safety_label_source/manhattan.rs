use rustc_hash::FxHashMap;
use std::sync::Arc;
use std::time::Instant;

use super::codec::{self, DecodeAttempt};
use super::lookup::{LookupError, ManhattanLookup};
use super::metrics::{self, ManhattanResult, SourceOutcome};
use super::mh_client::ManhattanLabelFetcher;
use super::types::{FailureKind, LabelSource, ManhattanOutcome};
use tonic::async_trait;

pub(crate) struct ManhattanSource {
    fetcher: Arc<dyn ManhattanLabelFetcher>,
}

impl ManhattanSource {
    pub(crate) fn new(fetcher: Arc<dyn ManhattanLabelFetcher>) -> Self {
        Self { fetcher }
    }
}

#[async_trait]
impl ManhattanLookup for ManhattanSource {
    async fn get(&self, ids: &[u64]) -> FxHashMap<u64, ManhattanOutcome> {
        let mut results = FxHashMap::with_capacity_and_hasher(ids.len(), Default::default());
        if ids.is_empty() {
            return results;
        }

        let start = Instant::now();
        let i64_ids: Vec<i64> = ids.iter().map(|&id| id.cast_signed()).collect();
        let mut successes = 0usize;
        let mut failures = 0usize;

        let request_outcome = match self.fetcher.fetch_labels(&i64_ids).await {
            Ok(fetch_results) => {
                for (i, &tweet_id) in ids.iter().enumerate() {
                    match fetch_results.get(i) {
                        Some(Ok(raw_items)) => match codec::decode_raw_labels(raw_items) {
                            DecodeAttempt::Success(label_map) => {
                                successes += 1;
                                results.insert(tweet_id, ManhattanOutcome::Resolved(label_map));
                            }
                            DecodeAttempt::Panic => {
                                failures += 1;
                                results.insert(
                                    tweet_id,
                                    ManhattanOutcome::Failure(LookupError::new(
                                        FailureKind::ManhattanDecode,
                                        "decode panic",
                                    )),
                                );
                            }
                        },
                        Some(Err(e)) => {
                            failures += 1;
                            results.insert(
                                tweet_id,
                                ManhattanOutcome::Failure(LookupError::new(
                                    FailureKind::ManhattanFetch,
                                    e.to_string(),
                                )),
                            );
                        }
                        None => {
                            failures += 1;
                            results.insert(
                                tweet_id,
                                ManhattanOutcome::Failure(LookupError::new(
                                    FailureKind::ManhattanFetch,
                                    "missing from MH response",
                                )),
                            );
                        }
                    }
                }
                SourceOutcome::Success
            }
            Err(e) => {
                for &tweet_id in ids {
                    failures += 1;
                    results.insert(
                        tweet_id,
                        ManhattanOutcome::Failure(LookupError::new(
                            FailureKind::ManhattanFetch,
                            e.to_string(),
                        )),
                    );
                }
                SourceOutcome::Failure
            }
        };

        metrics::record_source_request(
            LabelSource::Manhattan,
            request_outcome,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        metrics::record_manhattan_keys(ManhattanResult::Success, successes);
        metrics::record_manhattan_keys(ManhattanResult::Failed, failures);

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use xai_manhattan::ManhattanError;
    use xai_safety_label_store::types::encode_lkey;
    use xai_x_thrift::tweet_safety_label::SafetyLabelType;

    use crate::safety_label_source::codec::{LkeyBytes, MvalBytes, RawSafetyLabel};
    use crate::safety_label_source::mh_client::FetchResult;

    fn minimal_mval() -> Vec<u8> {
        vec![0x0C, 0x00, 0x04, 0x00, 0x00]
    }

    fn mval_with_score(score: f64) -> Vec<u8> {
        let mut buf = vec![0x0C, 0x00, 0x04];
        buf.extend_from_slice(&[0x04, 0x00, 0x01]);
        buf.extend_from_slice(&score.to_bits().to_be_bytes());
        buf.push(0x00);
        buf.push(0x00);
        buf
    }

    fn raw_label(lt: SafetyLabelType, mval: Vec<u8>) -> RawSafetyLabel {
        RawSafetyLabel {
            lkey: LkeyBytes(encode_lkey(lt)),
            mval: MvalBytes(mval),
        }
    }

    struct FakeLabelFetcher {
        items: HashMap<i64, Vec<RawSafetyLabel>>,
        per_key_errors: HashMap<i64, String>,
        batch_error: Mutex<Option<ManhattanError>>,
        truncate_to: Option<usize>,
    }

    impl FakeLabelFetcher {
        fn new() -> Self {
            Self {
                items: HashMap::new(),
                per_key_errors: HashMap::new(),
                batch_error: Mutex::new(None),
                truncate_to: None,
            }
        }

        fn with_label(mut self, tweet_id: i64, lt: SafetyLabelType, score: Option<f64>) -> Self {
            let mval = match score {
                Some(s) => mval_with_score(s),
                None => minimal_mval(),
            };
            self.items
                .entry(tweet_id)
                .or_default()
                .push(raw_label(lt, mval));
            self
        }

        fn with_per_key_error(mut self, tweet_id: i64, msg: &str) -> Self {
            self.per_key_errors.insert(tweet_id, msg.to_string());
            self
        }

        fn with_batch_error(self, msg: &str) -> Self {
            *self.batch_error.lock().unwrap() =
                Some(ManhattanError::NativeProtocol(msg.to_string()));
            self
        }

        fn with_truncated_response(mut self, len: usize) -> Self {
            self.truncate_to = Some(len);
            self
        }
    }

    #[async_trait]
    impl ManhattanLabelFetcher for FakeLabelFetcher {
        async fn fetch_labels(
            &self,
            tweet_ids: &[i64],
        ) -> Result<Vec<FetchResult>, ManhattanError> {
            if let Some(err) = self.batch_error.lock().unwrap().take() {
                return Err(err);
            }
            let mut out: Vec<FetchResult> = tweet_ids
                .iter()
                .map(|id| {
                    if let Some(msg) = self.per_key_errors.get(id) {
                        Err(ManhattanError::NativeProtocol(msg.clone()))
                    } else {
                        Ok(self.items.get(id).cloned().unwrap_or_default())
                    }
                })
                .collect();
            if let Some(len) = self.truncate_to {
                out.truncate(len);
            }
            Ok(out)
        }
    }

    async fn get_with_fetcher(
        fetcher: FakeLabelFetcher,
        ids: &[u64],
    ) -> FxHashMap<u64, ManhattanOutcome> {
        ManhattanSource::new(Arc::new(fetcher)).get(ids).await
    }

    #[tokio::test]
    async fn get_empty_label_list_returns_empty_label_map() {
        let results = get_with_fetcher(FakeLabelFetcher::new(), &[100]).await;

        match results.get(&100).unwrap() {
            ManhattanOutcome::Resolved(label_map) => assert!(label_map.labels.is_empty()),
            _ => panic!("expected hit"),
        }
    }

    #[tokio::test]
    async fn get_per_key_error_returns_error_for_that_id() {
        let results = get_with_fetcher(
            FakeLabelFetcher::new()
                .with_label(1, SafetyLabelType::SPAM, Some(0.5))
                .with_per_key_error(2, "key failed"),
            &[1, 2],
        )
        .await;

        assert!(matches!(
            results.get(&1).unwrap(),
            ManhattanOutcome::Resolved(label_map)
                if label_map.labels.contains_key(&i32::from(SafetyLabelType::SPAM))
        ));
        assert!(matches!(
            results.get(&2).unwrap(),
            ManhattanOutcome::Failure(failure)
                if failure.kind == FailureKind::ManhattanFetch
                    && failure.message.contains("key failed")
        ));
    }

    #[tokio::test]
    async fn get_batch_error_returns_errors_for_all_ids() {
        let results = get_with_fetcher(
            FakeLabelFetcher::new().with_batch_error("test error"),
            &[1, 2, 3],
        )
        .await;

        for id in [1, 2, 3] {
            assert!(matches!(
                results.get(&id).unwrap(),
                ManhattanOutcome::Failure(failure)
                    if failure.kind == FailureKind::ManhattanFetch
                        && failure.message.contains("test error")
            ));
        }
    }

    #[tokio::test]
    async fn get_short_response_marks_uncovered_ids_as_errors() {
        let results = get_with_fetcher(
            FakeLabelFetcher::new()
                .with_label(1, SafetyLabelType::SPAM, Some(0.5))
                .with_label(2, SafetyLabelType::SPAM, Some(0.5))
                .with_truncated_response(2),
            &[1, 2, 3],
        )
        .await;

        assert_eq!(results.len(), 3);
        assert!(matches!(
            results.get(&1).unwrap(),
            ManhattanOutcome::Resolved(_)
        ));
        assert!(matches!(
            results.get(&2).unwrap(),
            ManhattanOutcome::Resolved(_)
        ));
        assert!(matches!(
            results.get(&3).unwrap(),
            ManhattanOutcome::Failure(failure)
                if failure.kind == FailureKind::ManhattanFetch
                    && failure.message == "missing from MH response"
        ));
    }
}
