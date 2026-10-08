use crate::discovery::{VfChannel, VfChannelError, VfChannelParams, VfDiscovery, build_vf_channel};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tonic::codec::CompressionEncoding;
use tonic::{Status, async_trait};
use tracing::warn;
use xai_safety_label_store::types::SafetyLabelMap;
use xai_stats_receiver::global_stats_receiver;
use xai_visibility_filtering_proto as vf_pb;
use xai_visibility_filtering_proto::visibility_filtering_service_client::VisibilityFilteringServiceClient;
use xai_x_rpc::balanced_channel::{LbPolicy, LoadBalancedChannel};
use xai_x_thrift::tweet_safety_label::{
    AgentTool, BotMakerAction, GrokAnnotationAction, GrokAnnotationSource, SafetyLabel,
    SafetyLabelSource, SafetyLabelType, ToolAction,
};

const DEFAULT_TIMEOUT_MS: u64 = 200;
const DEFAULT_MAX_BATCH_SIZE: usize = 50;

pub const LATENCY_VM_METRIC: &str = "vf_client_get_safety_labels_latency_ms_vm";
pub const BATCH_LATENCY_VM_METRIC: &str = "vf_client_get_safety_labels_batch_latency_ms_vm";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SafetyLabelFailure {
    LookupFailed,
    BatchFailed,
}

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelsBatch {
    pub labels: HashMap<u64, SafetyLabelMap>,
    pub failures: HashMap<u64, SafetyLabelFailure>,
}

#[async_trait]
pub trait TweetSafetyLabelClient: Send + Sync {
    async fn get_safety_labels(&self, tweet_ids: Vec<u64>) -> Result<SafetyLabelsBatch, Status>;
}

pub struct ProdTweetSafetyLabelClient {
    client: VfServiceClient,
    max_batch_size: usize,
    timeout: Duration,
}

#[derive(Clone)]
enum VfServiceClient {
    Wily(VisibilityFilteringServiceClient<tonic::transport::Channel>),
    Xds(VisibilityFilteringServiceClient<LoadBalancedChannel>),
}

impl VfServiceClient {
    async fn get_safety_labels(
        &mut self,
        request: tonic::Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<tonic::Response<vf_pb::GetSafetyLabelsResponse>, Status> {
        match self {
            Self::Wily(c) => c.get_safety_labels(request).await,
            Self::Xds(c) => c.get_safety_labels(request).await,
        }
    }
}

enum RequestStatus {
    Completed,
    Degraded,
    Cancelled,
}

struct RequestMetricsGuard {
    start: Instant,
    receiver: Option<Arc<dyn xai_stats_receiver::StatsReceiverExt>>,
    result: RequestStatus,
}

impl RequestMetricsGuard {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            receiver: global_stats_receiver(),
            result: RequestStatus::Cancelled,
        }
    }

        fn mark_completed(&mut self, failed_ids: usize) {
        self.result = if failed_ids > 0 {
            RequestStatus::Degraded
        } else {
            RequestStatus::Completed
        };
    }
}

impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        let Some(sr) = &self.receiver else {
            return;
        };

        let result = match self.result {
            RequestStatus::Completed => "completed",
            RequestStatus::Degraded => "degraded",
            RequestStatus::Cancelled => "cancelled",
        };

        sr.incr("vf_client_get_safety_labels", &[("requests", result)], 1);
        sr.observe_vm(
            LATENCY_VM_METRIC,
            &[],
            self.start.elapsed().as_secs_f64() * 1000.0,
        );
    }
}

const VF_SAFETY_LABELS_DISCOVERY_ENV: &str = "VF_SAFETY_LABELS_DISCOVERY";
const VF_SAFETY_LABELS_APERTURE_SIZE: usize = 16;

fn vf_channel_params(dc: &str) -> VfChannelParams<'_> {
    VfChannelParams {
        name: "vf-safety-labels",
        dc,
        discovery: VfDiscovery::from_env(VF_SAFETY_LABELS_DISCOVERY_ENV),
        aperture_size: VF_SAFETY_LABELS_APERTURE_SIZE,
        deterministic_aperture: std::env::var("APP_ENV").as_deref() == Ok("prod"),
        xds_lb_policy: LbPolicy::least_request(),
    }
}

fn channel_error_to_status(e: VfChannelError) -> Status {
    match e {
        VfChannelError::Config(e) => Status::internal(format!("{e:#}")),
        VfChannelError::Connect(e) => {
            Status::unavailable(format!("failed to connect to vf-service: {e:#}"))
        }
    }
}

fn make_client<T>(channel: T) -> VisibilityFilteringServiceClient<T>
where
    T: tonic::client::GrpcService<tonic::body::Body>,
    T::Error: Into<tonic::codegen::StdError>,
    T::ResponseBody: tonic::codegen::Body<Data = tonic::codegen::Bytes> + Send + 'static,
    <T::ResponseBody as tonic::codegen::Body>::Error: Into<tonic::codegen::StdError> + Send,
{
    VisibilityFilteringServiceClient::new(channel)
        .send_compressed(CompressionEncoding::Zstd)
        .accept_compressed(CompressionEncoding::Zstd)
}

impl ProdTweetSafetyLabelClient {
    pub async fn new(dc: &str) -> Result<Self, Status> {
        let params = vf_channel_params(dc);
        let client = match build_vf_channel(&params)
            .await
            .map_err(channel_error_to_status)?
        {
            VfChannel::Wily(ch) => VfServiceClient::Wily(make_client(ch)),
            VfChannel::Xds(ch) => VfServiceClient::Xds(make_client(ch)),
        };

        Ok(Self {
            client,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
        })
    }

    pub fn with_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout = if timeout_ms == 0 {
            Duration::from_millis(DEFAULT_TIMEOUT_MS)
        } else {
            Duration::from_millis(timeout_ms)
        };
        self
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = if size == 0 {
            DEFAULT_MAX_BATCH_SIZE
        } else {
            size
        };
        self
    }
}

fn proto_to_safety_label(label: &vf_pb::SafetyLabel) -> SafetyLabel {
    use xai_x_thrift::tweet_safety_label::PerspectivalUser;
    SafetyLabel {
        score: label.score.map(thrift::OrderedFloat::from),
        applicable_users: if label.applicable_users.is_empty() {
            None
        } else {
            Some(
                label
                    .applicable_users
                    .iter()
                    .map(|&uid| PerspectivalUser::new(uid))
                    .collect(),
            )
        },
        holdback_experiment: label.holdback_experiment,
        source: label.source.clone(),
        safety_label_source: label.safety_label_source.as_ref().map(|src| match src {
            vf_pb::safety_label::SafetyLabelSource::BotmakerAction(a) => {
                SafetyLabelSource::BotMakerAction(BotMakerAction::new(a.rule_id))
            }
            vf_pb::safety_label::SafetyLabelSource::ToolAction(a) => SafetyLabelSource::ToolAction(
                ToolAction::new(AgentTool::from(a.agent_tool), a.actor_ldap.clone()),
            ),
            vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(a) => {
                SafetyLabelSource::GrokAnnotationAction(GrokAnnotationAction::new(
                    GrokAnnotationSource::from(a.source),
                ))
            }
        }),
        created_at_msec: label.created_at_msec,
        expires_at_msec: label.expires_at_msec,
        applicable_countries: if label.applicable_countries.is_empty() {
            None
        } else {
            Some(label.applicable_countries.clone())
        },
    }
}

pub(crate) fn proto_to_safety_label_map(proto: &vf_pb::SafetyLabelMap) -> SafetyLabelMap {
    proto
        .labels
        .iter()
        .map(|(&type_id, label)| (SafetyLabelType::from(type_id), proto_to_safety_label(label)))
        .collect()
}

#[async_trait]
impl TweetSafetyLabelClient for ProdTweetSafetyLabelClient {
    async fn get_safety_labels(&self, tweet_ids: Vec<u64>) -> Result<SafetyLabelsBatch, Status> {
        if tweet_ids.is_empty() {
            return Ok(SafetyLabelsBatch {
                labels: HashMap::new(),
                failures: HashMap::new(),
            });
        }

        let timeout = self.timeout;
        let mut request_metrics = RequestMetricsGuard::new();

        let futures: Vec<_> = tweet_ids
            .chunks(self.max_batch_size)
            .map(|batch| {
                let batch_ids = batch.to_vec();
                let mut client = self.client.clone();
                async move {
                    let batch_start = Instant::now();
                    let mut request = tonic::Request::new(vf_pb::GetSafetyLabelsRequest {
                        tweet_ids: batch_ids.clone(),
                    });
                    request.set_timeout(timeout);
                    let result = client.get_safety_labels(request).await.map(|response| {
                        let response = response.into_inner();
                        let labels = response
                            .results
                            .iter()
                            .map(|(&id, proto)| (id, proto_to_safety_label_map(proto)))
                            .collect::<HashMap<u64, SafetyLabelMap>>();
                        let failures = response
                            .failed_ids
                            .into_iter()
                            .map(|id| (id, SafetyLabelFailure::LookupFailed))
                            .collect::<HashMap<u64, SafetyLabelFailure>>();

                        SafetyLabelsBatch { labels, failures }
                    });
                    (result, batch_start.elapsed())
                }
            })
            .collect();

        let responses = futures::future::join_all(futures).await;

        let mut labels = HashMap::with_capacity(tweet_ids.len());
        let mut failures = HashMap::new();

        for (i, (response, batch_elapsed)) in responses.into_iter().enumerate() {
            match response {
                Ok(batch) => {
                    labels.extend(batch.labels);
                    failures.extend(batch.failures);
                    if let Some(sr) = global_stats_receiver() {
                        sr.observe_vm(
                            BATCH_LATENCY_VM_METRIC,
                            &[("result", "success")],
                            batch_elapsed.as_secs_f64() * 1000.0,
                        );
                    }
                }
                Err(status) => {
                    warn!(batch = i, %status, "get_safety_labels batch failed");
                    let error_type = match status.code() {
                        tonic::Code::DeadlineExceeded => "deadline_exceeded",
                        tonic::Code::Cancelled => "cancelled",
                        tonic::Code::Unavailable => "unavailable",
                        _ => "other",
                    };
                    if let Some(sr) = global_stats_receiver() {
                        sr.incr(
                            "vf_client_get_safety_labels_batch_error",
                            &[("code", error_type)],
                            1,
                        );
                        sr.observe_vm(
                            BATCH_LATENCY_VM_METRIC,
                            &[("result", "error")],
                            batch_elapsed.as_secs_f64() * 1000.0,
                        );
                    }
                    let start = i * self.max_batch_size;
                    let end = usize::min(start + self.max_batch_size, tweet_ids.len());
                    for id in &tweet_ids[start..end] {
                        failures.insert(*id, SafetyLabelFailure::BatchFailed);
                    }
                }
            }
        }

        if !failures.is_empty()
            && let Some(sr) = global_stats_receiver()
        {
            sr.incr(
                "vf_client_get_safety_labels_failed_ids",
                &[],
                failures.len() as u64,
            );
        }

        request_metrics.mark_completed(failures.len());
        Ok(SafetyLabelsBatch { labels, failures })
    }
}

pub struct MockTweetSafetyLabelClient;

#[async_trait]
impl TweetSafetyLabelClient for MockTweetSafetyLabelClient {
    async fn get_safety_labels(&self, tweet_ids: Vec<u64>) -> Result<SafetyLabelsBatch, Status> {
        Ok(SafetyLabelsBatch {
            labels: tweet_ids
                .into_iter()
                .map(|id| (id, SafetyLabelMap::default()))
                .collect(),
            failures: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::RecordingReceiver;
    use std::collections::HashSet;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;
    use tonic::{Request, Response};
    use xai_visibility_filtering_proto::visibility_filtering_service_server::{
        VisibilityFilteringService, VisibilityFilteringServiceServer,
    };

    #[derive(Clone)]
    struct TestVisibilityFilteringService {
        requests: Arc<Mutex<Vec<Vec<u64>>>>,
        responses: Arc<Mutex<Vec<Result<vf_pb::GetSafetyLabelsResponse, Status>>>>,
    }

    impl Default for TestVisibilityFilteringService {
        fn default() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                responses: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[tonic::async_trait]
    impl VisibilityFilteringService for TestVisibilityFilteringService {
        async fn evaluate_tweets(
            &self,
            _: Request<vf_pb::EvaluateTweetsRequest>,
        ) -> Result<Response<vf_pb::EvaluateTweetsResponse>, Status> {
            Err(Status::unimplemented("unused"))
        }

        async fn filter_tweets(
            &self,
            _request: Request<vf_pb::VisibilityFilterRequest>,
        ) -> Result<Response<vf_pb::VisibilityFilterResponse>, Status> {
            unimplemented!()
        }

        async fn get_safety_labels(
            &self,
            request: Request<vf_pb::GetSafetyLabelsRequest>,
        ) -> Result<Response<vf_pb::GetSafetyLabelsResponse>, Status> {
            let tweet_ids = request.into_inner().tweet_ids;
            self.requests.lock().await.push(tweet_ids.clone());

            let mut responses = self.responses.lock().await;
            if !responses.is_empty() {
                return responses.remove(0).map(Response::new);
            }

            let results = tweet_ids
                .into_iter()
                .map(|id| (id, vf_pb::SafetyLabelMap::default()))
                .collect();
            Ok(Response::new(vf_pb::GetSafetyLabelsResponse {
                results,
                failed_ids: Vec::new(),
            }))
        }
    }

    async fn start_test_server(
        service: TestVisibilityFilteringService,
    ) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let incoming = TcpListenerStream::new(listener);
        let handle = tokio::spawn(async move {
            Server::builder()
                .add_service(VisibilityFilteringServiceServer::new(service))
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        (addr, handle)
    }

    async fn test_client(addr: SocketAddr) -> ProdTweetSafetyLabelClient {
        let endpoint = format!("http://{addr}");
        let channel = tonic::transport::Endpoint::from_shared(endpoint)
            .unwrap()
            .connect()
            .await
            .unwrap();
        ProdTweetSafetyLabelClient {
            client: VfServiceClient::Wily(VisibilityFilteringServiceClient::new(channel)),
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS),
        }
    }

    #[test]
    fn proto_botmaker_action_to_thrift() {
        let proto = vf_pb::SafetyLabel {
            safety_label_source: Some(vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                vf_pb::BotmakerAction { rule_id: 42 },
            )),
            ..Default::default()
        };
        let label = proto_to_safety_label(&proto);
        assert_eq!(
            label.safety_label_source,
            Some(SafetyLabelSource::BotMakerAction(BotMakerAction::new(42)))
        );
    }

    #[test]
    fn proto_tool_action_to_thrift() {
        let proto = vf_pb::SafetyLabel {
            safety_label_source: Some(vf_pb::safety_label::SafetyLabelSource::ToolAction(
                vf_pb::ToolAction {
                    agent_tool: 15,
                    actor_ldap: "someone".to_string(),
                },
            )),
            ..Default::default()
        };
        let label = proto_to_safety_label(&proto);
        assert_eq!(
            label.safety_label_source,
            Some(SafetyLabelSource::ToolAction(ToolAction::new(
                AgentTool::BME,
                "someone".to_string(),
            )))
        );
    }

    #[test]
    fn proto_grok_annotation_action_to_thrift() {
        let proto = vf_pb::SafetyLabel {
            safety_label_source: Some(
                vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(
                    vf_pb::GrokAnnotationAction { source: 1 },
                ),
            ),
            ..Default::default()
        };
        let label = proto_to_safety_label(&proto);
        assert_eq!(
            label.safety_label_source,
            Some(SafetyLabelSource::GrokAnnotationAction(
                GrokAnnotationAction::new(GrokAnnotationSource::GROX_PTOS)
            ))
        );
    }

    #[test]
    fn proto_no_source_to_thrift() {
        let proto = vf_pb::SafetyLabel::default();
        let label = proto_to_safety_label(&proto);
        assert_eq!(label.safety_label_source, None);
    }

    #[tokio::test]
    async fn get_safety_labels_returns_empty_map_for_empty_input() {
        let service = TestVisibilityFilteringService::default();
        let requests = service.requests.clone();
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await;

        let result = client.get_safety_labels(vec![]).await.unwrap();

        assert!(result.labels.is_empty());
        assert!(result.failures.is_empty());
        assert!(requests.lock().await.is_empty());
        handle.abort();
    }

    #[tokio::test]
    async fn get_safety_labels_uses_custom_batch_size() {
        let service = TestVisibilityFilteringService::default();
        let requests = service.requests.clone();
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await.with_max_batch_size(3);

        let tweet_ids: Vec<u64> = (1..=10).collect();
        let result = client.get_safety_labels(tweet_ids.clone()).await.unwrap();

        let requests = requests.lock().await.clone();
        let request_sets: HashSet<Vec<u64>> = requests.into_iter().collect();
        let expected_sets: HashSet<Vec<u64>> =
            [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9], vec![10]]
                .into_iter()
                .collect();

        assert_eq!(request_sets, expected_sets);
        assert_eq!(result.labels.len(), tweet_ids.len());
        assert!(result.failures.is_empty());
        for id in tweet_ids {
            assert!(result.labels.contains_key(&id));
        }
        handle.abort();
    }

    #[tokio::test]
    async fn with_max_batch_size_zero_falls_back_to_default_batch_size() {
        let service = TestVisibilityFilteringService::default();
        let requests = service.requests.clone();
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await.with_max_batch_size(0);

        let tweet_ids: Vec<u64> = (1..=(DEFAULT_MAX_BATCH_SIZE as u64 + 1)).collect();
        let result = client.get_safety_labels(tweet_ids.clone()).await.unwrap();

        let requests = requests.lock().await.clone();
        let request_sets: HashSet<Vec<u64>> = requests.into_iter().collect();
        let expected_sets: HashSet<Vec<u64>> = [
            (1..=DEFAULT_MAX_BATCH_SIZE as u64).collect(),
            vec![DEFAULT_MAX_BATCH_SIZE as u64 + 1],
        ]
        .into_iter()
        .collect();

        assert_eq!(request_sets, expected_sets);
        assert_eq!(result.labels.len(), tweet_ids.len());
        assert!(result.failures.is_empty());
        for id in tweet_ids {
            assert!(result.labels.contains_key(&id));
        }
        handle.abort();
    }

    #[tokio::test]
    async fn get_safety_labels_parses_failed_ids() {
        let service = TestVisibilityFilteringService {
            responses: Arc::new(Mutex::new(vec![Ok(vf_pb::GetSafetyLabelsResponse {
                results: HashMap::from([(1, vf_pb::SafetyLabelMap::default())]),
                failed_ids: vec![2],
            })])),
            ..Default::default()
        };
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await;

        let result = client.get_safety_labels(vec![1, 2]).await.unwrap();

        assert_eq!(result.labels.len(), 1);
        assert!(result.labels.contains_key(&1));
        assert_eq!(
            result.failures,
            HashMap::from([(2, SafetyLabelFailure::LookupFailed)])
        );
        handle.abort();
    }

    #[tokio::test]
    async fn get_safety_labels_marks_transport_failed_batch_ids() {
        let service = TestVisibilityFilteringService {
            responses: Arc::new(Mutex::new(vec![Err(Status::unavailable("vf down"))])),
            ..Default::default()
        };
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await;

        let result = client.get_safety_labels(vec![1, 2]).await.unwrap();

        assert!(result.labels.is_empty());
        assert_eq!(
            result.failures,
            HashMap::from([
                (1, SafetyLabelFailure::BatchFailed),
                (2, SafetyLabelFailure::BatchFailed),
            ])
        );
        handle.abort();
    }

    #[tokio::test]
    async fn get_safety_labels_merges_batches() {
        let service = TestVisibilityFilteringService {
            responses: Arc::new(Mutex::new(vec![
                Ok(vf_pb::GetSafetyLabelsResponse {
                    results: HashMap::from([(1, vf_pb::SafetyLabelMap::default())]),
                    failed_ids: vec![2],
                }),
                Ok(vf_pb::GetSafetyLabelsResponse {
                    results: HashMap::from([(3, vf_pb::SafetyLabelMap::default())]),
                    failed_ids: vec![4],
                }),
            ])),
            ..Default::default()
        };
        let (addr, handle) = start_test_server(service).await;
        let client = test_client(addr).await.with_max_batch_size(2);

        let result = client.get_safety_labels(vec![1, 2, 3, 4]).await.unwrap();

        assert_eq!(result.labels.len(), 2);
        assert!(result.labels.contains_key(&1));
        assert!(result.labels.contains_key(&3));
        assert_eq!(
            result.failures,
            HashMap::from([
                (2, SafetyLabelFailure::LookupFailed),
                (4, SafetyLabelFailure::LookupFailed),
            ])
        );
        handle.abort();
    }

    fn request_guard_with(receiver: Arc<RecordingReceiver>) -> RequestMetricsGuard {
        RequestMetricsGuard {
            start: Instant::now(),
            receiver: Some(receiver),
            result: RequestStatus::Cancelled,
        }
    }

    #[test]
    fn request_guard_counts_completed_degraded_or_cancelled() {
        let sr = Arc::new(RecordingReceiver::default());
        request_guard_with(sr.clone()).mark_completed(0);
        request_guard_with(sr.clone()).mark_completed(2);
        drop(request_guard_with(sr.clone()));
        for status in ["completed", "degraded", "cancelled"] {
            let key = format!("vf_client_get_safety_labels|requests={status}");
            assert_eq!(sr.counter(&key), 1, "{key}");
        }
    }
}
