use crate::caller_identity::{self, Endpoint};
use crate::filter::{FilterOutcome, FilterRequest, FilterTweets};
use crate::models::{ClientCapability, RawCandidate, TweetId};
use crate::rules::metrics::{self as ft_metrics, RequestMetricsGuard, Rpc};
use crate::rules::SafetyLevel;
use crate::treatment;
use std::sync::Arc;
use std::time::Duration;
use tonic::metadata::MetadataMap;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub(crate) type FinishComparison = Box<dyn FnOnce(&[FilterOutcome]) + Send>;

pub(crate) trait Comparator: Send + Sync {
    fn begin(
        &self,
        viewer_id: Option<u64>,
        country_code: Option<String>,
        safety_level: SafetyLevel,
        tweet_ids: Vec<u64>,
    ) -> Option<FinishComparison>;
}

pub struct FilterTweetsEndpoint {
    filter_tweets: Arc<FilterTweets>,
    comparator: Option<Box<dyn Comparator>>,
}

impl FilterTweetsEndpoint {
    pub(crate) fn new(
        filter_tweets: Arc<FilterTweets>,
        comparator: Option<Box<dyn Comparator>>,
    ) -> Self {
        Self {
            filter_tweets,
            comparator,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::VisibilityFilterRequest>,
    ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
        let entered = tokio::time::Instant::now();
        let request_metrics = RequestMetricsGuard::new();
        let grpc_timeout = parse_grpc_timeout(request.metadata());
        let caller = caller_identity::record(Endpoint::FilterTweets, &request);
        let context = crate::hydration::request_context(entered, grpc_timeout);
        let req = request.into_inner();
        ft_metrics::record_batch_size(ft_metrics::BATCH_SIZE, req.tweets.len());
        let viewer_id = normalize_viewer_id(req.viewer_id);
        ft_metrics::record_viewer_state(req.viewer_id, viewer_id);

        let Some(safety_level) = SafetyLevel::from_proto(req.safety_level()) else {
            request_metrics.mark_failure();
            caller.mark_failure();
            return Err(Status::unimplemented("safety level has no Rust policy"));
        };
        let candidates: Vec<RawCandidate> = req
            .tweets
            .iter()
            .map(|t| RawCandidate {
                tweet_id: TweetId(t.tweet_id),
                request_author_id: t.author_id,
            })
            .collect();

        let finish_comparison = self.comparator.as_ref().and_then(|comparator| {
            comparator.begin(
                viewer_id,
                req.country_code.clone(),
                safety_level,
                req.tweets.iter().map(|t| t.tweet_id).collect(),
            )
        });

        let response = context
            .scope(self.filter_tweets.run(FilterRequest {
                viewer_id,
                country_code: req.country_code,
                client_capability: ClientCapability::default(),
                safety_level,
                candidates,
                rpc: Rpc::FilterTweets,
            }))
            .await;
        ft_metrics::record_verdicts(
            Rpc::FilterTweets,
            safety_level,
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.evaluation.verdict()),
        );
        ft_metrics::record_unresolved(
            Rpc::FilterTweets,
            safety_level,
            response.outcomes.iter().map(|outcome| &outcome.evaluation),
        );
        ft_metrics::record_fail_open_defaults(
            Rpc::FilterTweets,
            safety_level,
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.evaluation.fail_open_defaults()),
        );

        if let Some(finish_comparison) = finish_comparison {
            finish_comparison(&response.outcomes);
        }

        let results = response
            .outcomes
            .into_iter()
            .map(to_visibility_result)
            .collect();

        request_metrics.record_deadline(grpc_timeout);
        request_metrics.mark_success();
        caller.mark_success();
        Ok(Response::new(vf_pb::VisibilityFilterResponse { results }))
    }
}

pub(crate) fn normalize_viewer_id(raw: Option<u64>) -> Option<u64> {
    raw.filter(|&id| id.cast_signed() > 0)
}

pub(crate) fn parse_grpc_timeout(metadata: &MetadataMap) -> Option<Duration> {
    let raw = metadata.get("grpc-timeout")?.to_str().ok()?;
    let (digits, unit) = raw.split_at(raw.len().checked_sub(1)?);
    if digits.is_empty() || digits.len() > 8 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let value: u64 = digits.parse().ok()?;
    let nanos_per_unit = match unit {
        "H" => 3_600_000_000_000,
        "M" => 60_000_000_000,
        "S" => 1_000_000_000,
        "m" => 1_000_000,
        "u" => 1_000,
        "n" => 1,
        _ => return None,
    };
    Some(Duration::from_nanos(value.checked_mul(nanos_per_unit)?))
}

fn to_visibility_result(outcome: FilterOutcome) -> vf_pb::TweetVisibilityResult {
    let (action, filtered_reason) = treatment::proto_action(outcome.evaluation.into_verdict());
    vf_pb::TweetVisibilityResult {
        tweet_id: outcome.tweet_id.0,
        action: Some(action),
        filtered_reason,
        safety_labels: outcome.safety_labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::plan::Source;
    use crate::hydration::sources::InMemorySources;
    use crate::rules::RuleEngine;
    use std::sync::Mutex;

    async fn gizmoduck_calls(viewer_id: Option<u64>) -> usize {
        let sources = Arc::new(InMemorySources::default());
        let endpoint = FilterTweetsEndpoint::new(
            Arc::new(FilterTweets::new(
                Arc::<InMemorySources>::clone(&sources),
                RuleEngine::for_tests(),
            )),
            None,
        );
        let response = endpoint
            .handle(Request::new(vf_pb::VisibilityFilterRequest {
                safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                tweets: vec![vf_pb::TweetInput {
                    tweet_id: 2,
                    author_id: Some(20),
                }],
                viewer_id,
                country_code: None,
            }))
            .await
            .unwrap();
        assert_eq!(response.into_inner().results.len(), 1);
        sources
            .calls()
            .into_iter()
            .filter(|source| matches!(source, Source::GizmoduckViewer | Source::GizmoduckAuthor))
            .count()
    }

    #[tokio::test]
    async fn endpoint_treats_zero_viewer_id_as_logged_out() {
        let logged_out = gizmoduck_calls(None).await;
        assert_eq!(gizmoduck_calls(Some(0)).await, logged_out);
        assert_eq!(gizmoduck_calls(Some(u64::MAX)).await, logged_out);
        assert_eq!(gizmoduck_calls(Some(42)).await, logged_out + 1);
    }

    #[derive(Default)]
    struct ViewerIds(Arc<Mutex<Vec<Option<u64>>>>);

    impl Comparator for ViewerIds {
        fn begin(
            &self,
            viewer_id: Option<u64>,
            _country_code: Option<String>,
            _safety_level: SafetyLevel,
            _tweet_ids: Vec<u64>,
        ) -> Option<FinishComparison> {
            self.0.lock().unwrap().push(viewer_id);
            None
        }
    }

    #[tokio::test]
    async fn comparator_sees_the_normalized_viewer_id() {
        let viewer_ids = ViewerIds::default();
        let seen = Arc::clone(&viewer_ids.0);
        let endpoint = FilterTweetsEndpoint::new(
            Arc::new(FilterTweets::new(
                Arc::new(InMemorySources::default()),
                RuleEngine::for_tests(),
            )),
            Some(Box::new(viewer_ids)),
        );
        for viewer_id in [Some(0), Some(42)] {
            endpoint
                .handle(Request::new(vf_pb::VisibilityFilterRequest {
                    safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                    tweets: vec![vf_pb::TweetInput {
                        tweet_id: 2,
                        author_id: Some(20),
                    }],
                    viewer_id,
                    country_code: None,
                }))
                .await
                .unwrap();
        }
        assert_eq!(*seen.lock().unwrap(), vec![None, Some(42)]);
    }

    #[derive(Default)]
    struct FinishedResults(Arc<Mutex<Vec<Vec<vf_pb::TweetVisibilityResult>>>>);

    impl Comparator for FinishedResults {
        fn begin(
            &self,
            _viewer_id: Option<u64>,
            _country_code: Option<String>,
            _safety_level: SafetyLevel,
            _tweet_ids: Vec<u64>,
        ) -> Option<FinishComparison> {
            let finished = Arc::clone(&self.0);
            Some(Box::new(move |outcomes: &[FilterOutcome]| {
                finished
                    .lock()
                    .unwrap()
                    .push(outcomes.iter().cloned().map(to_visibility_result).collect());
            }))
        }
    }

    #[tokio::test]
    async fn comparator_finishes_with_the_served_outcomes() {
        let comparator = FinishedResults::default();
        let finished = Arc::clone(&comparator.0);
        let endpoint = FilterTweetsEndpoint::new(
            Arc::new(FilterTweets::new(
                Arc::new(InMemorySources::default()),
                RuleEngine::for_tests(),
            )),
            Some(Box::new(comparator)),
        );
        let served = endpoint
            .handle(Request::new(vf_pb::VisibilityFilterRequest {
                safety_level: vf_pb::SafetyLevel::TimelineHome.into(),
                tweets: [(2, 20), (3, 30)]
                    .map(|(tweet_id, author_id)| vf_pb::TweetInput {
                        tweet_id,
                        author_id: Some(author_id),
                    })
                    .into(),
                viewer_id: Some(42),
                country_code: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .results;
        assert_eq!(*finished.lock().unwrap(), vec![served]);
    }

    #[test]
    fn parse_grpc_timeout_units_and_garbage() {
        let parsed = |value: &str| {
            let mut metadata = MetadataMap::new();
            metadata.insert("grpc-timeout", value.parse().unwrap());
            parse_grpc_timeout(&metadata)
        };
        assert_eq!(parsed("400m"), Some(Duration::from_millis(400)));
        assert_eq!(parsed("1S"), Some(Duration::from_secs(1)));
        assert_eq!(parsed("2M"), Some(Duration::from_secs(120)));
        assert_eq!(parsed("500u"), Some(Duration::from_micros(500)));
        assert_eq!(parsed("3H"), Some(Duration::from_secs(10_800)));
        assert_eq!(parsed("9n"), Some(Duration::from_nanos(9)));
        assert_eq!(parsed("400"), None);
        assert_eq!(parsed("m"), None);
        assert_eq!(parsed("400x"), None);
        assert_eq!(parsed("+400m"), None);
        assert_eq!(parsed("12345678m"), Some(Duration::from_millis(12_345_678)));
        assert_eq!(parsed("123456789m"), None);
        assert_eq!(parse_grpc_timeout(&MetadataMap::new()), None);
    }
}
