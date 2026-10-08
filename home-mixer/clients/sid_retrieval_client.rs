use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{watch, OnceCell};
use tonic::{async_trait, Status};
use xai_candidate_pipeline::component_library::clients::XdsP2cConfig;
use xai_recsys_sid_retrieval_proto::sid_retrieval_service_client::SidRetrievalServiceClient;
use xai_recsys_sid_retrieval_proto::{RetrieveRequest, RetrieveResponse};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};
use xai_x_rpc::balanced_channel::{LbPolicy, LoadBalancedChannel};
use xai_x_rpc::grpc_client::insecure_tls_config;
use xai_x_rpc::service_probe::KeepAlive;
use xai_x_rpc::xds_endpoint_source::XdsEndpointSource;
use xai_xds_client::{ServiceState, StartFrom, XdsClient};

use super::rotating_channel::RotatingChannel;

const ENDPOINT: &str = "https://xai-recsys-sid-retrieval.prod.fou.twitter.biz:443";
const XDS_SERVICE: &str = "fou-xai-recsys-sid-retrieval.prod.prod:grpc-tls";
const READINESS_PORT: u16 = 29180;
const READINESS_PATH: &str = "/readyz";
pub const TIMEOUT_MS: u64 = 300;
const REFRESH_INTERVAL: Duration = Duration::from_secs(60);
const XDS_BUILD_TIMEOUT: Duration = Duration::from_secs(10);
const XDS_TRANSPORT_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_DECODING_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const METRIC_NAME: &str = "SidRetrievalClient.retrieve";
const XDS_BUILD_METRIC_NAME: &str = "SidRetrievalClient.xds_build";
const XDS_ENDPOINTS_METRIC_NAME: &str = "SidRetrievalClient.xds_endpoints";

#[async_trait]
pub trait SidRetrievalClient: Send + Sync {
    async fn retrieve(&self, request: RetrieveRequest) -> Result<RetrieveResponse, Status>;
}

type XdsServiceClient = SidRetrievalServiceClient<LoadBalancedChannel>;

pub struct ProdSidRetrievalClient {
    channel: OnceCell<Arc<RotatingChannel>>,
    xds_config: XdsP2cConfig,
    xds_build_started: AtomicBool,
    xds: Arc<OnceCell<Option<XdsServiceClient>>>,
}

pub fn xds_eds_resource(discovery_authority: &str) -> String {
    format!(
        "xdstp://discovery-{discovery_authority}/envoy.config.endpoint.v3.ClusterLoadAssignment/{XDS_SERVICE}"
    )
}

impl ProdSidRetrievalClient {
    pub fn new(xds_config: &XdsP2cConfig) -> Arc<Self> {
        let client = Arc::new(Self {
            channel: OnceCell::new(),
            xds_config: xds_config.clone(),
            xds_build_started: AtomicBool::new(false),
            xds: Arc::new(OnceCell::new()),
        });
        let warm = Arc::clone(&client);
        tokio::spawn(async move {
            if let Err(e) = warm.channel().await {
                tracing::warn!(error = %e, "SID retrieval channel warm-up failed");
            }
        });
        client
    }

    fn start_xds_build(&self) {
        if self.xds_build_started.swap(true, Ordering::Relaxed) {
            return;
        }
        let Some(policy) = self.xds_config.parse_lb_policy() else {
            let _ = self.xds.set(None);
            return;
        };
        let config = self.xds_config.clone();
        let xds = Arc::clone(&self.xds);
        tokio::spawn(async move {
            let built = build_xds_client(policy, &config).await;
            let _ = xds.set(built);
        });
    }

    async fn channel(&self) -> Result<Arc<RotatingChannel>, Status> {
        self.channel
            .get_or_try_init(|| {
                RotatingChannel::new(
                    ENDPOINT,
                    Duration::from_millis(TIMEOUT_MS),
                    REFRESH_INTERVAL,
                )
            })
            .await
            .cloned()
    }

    fn xds_client(&self) -> Option<XdsServiceClient> {
        self.start_xds_build();
        self.xds.get().and_then(|c| c.clone())
    }

    async fn call_dns(&self, request: RetrieveRequest) -> Result<RetrieveResponse, Status> {
        let channel = self.channel().await?;
        let client = SidRetrievalServiceClient::new((*channel.next()).clone());
        send(client, request).await
    }
}

async fn send<T>(
    client: SidRetrievalServiceClient<T>,
    request: RetrieveRequest,
) -> Result<RetrieveResponse, Status>
where
    T: tonic::client::GrpcService<tonic::body::Body>,
    T::Error: Into<tonic::codegen::StdError>,
    T::ResponseBody: tonic::codegen::Body<Data = tonic::codegen::Bytes> + Send + 'static,
    <T::ResponseBody as tonic::codegen::Body>::Error: Into<tonic::codegen::StdError> + Send,
{
    let mut client = client.max_decoding_message_size(MAX_DECODING_MESSAGE_BYTES);
    let mut request = tonic::Request::new(request);
    request.set_timeout(Duration::from_millis(TIMEOUT_MS));
    client.retrieve(request).await.map(|r| r.into_inner())
}

async fn build_xds_client(policy: LbPolicy, config: &XdsP2cConfig) -> Option<XdsServiceClient> {
    let eds_resource = xds_eds_resource(&config.discovery_authority);
    let build = async {
        let xds_client = XdsClient::from_bootstrap()
            .map_err(|e| format!("xDS bootstrap not available: {e}"))?
            .start_from(StartFrom::Eds(eds_resource.clone()))
            .build()
            .await
            .map_err(|e| format!("xDS client build failed: {e}"))?;
        spawn_endpoint_gauge(xds_client.subscribe());
        let channel = LoadBalancedChannel::builder(XdsEndpointSource::new(xds_client))
            .timeout(XDS_TRANSPORT_TIMEOUT)
            .connect_timeout(Duration::from_secs(1))
            .with_insecure_tls(insecure_tls_config())
            .http2_initial_windows(
                Some(config.h2_stream_window_bytes),
                Some(config.h2_connection_window_bytes),
            )
            .socket_buffer_bytes(Some(config.socket_buffer_bytes))
            .keep_alive(KeepAlive {
                interval: Some(Duration::from_secs(30)),
                timeout: Some(Duration::from_secs(10)),
                while_idle: true,
            })
            .aperture(config.aperture_size as usize)
            .deterministic()
            .lb_policy(policy)
            .readiness_probe_with_path(READINESS_PORT, READINESS_PATH.to_string())
            .channel()
            .await
            .map_err(|e| format!("xDS channel build failed: {e}"))?;
        Ok::<_, String>(SidRetrievalServiceClient::new(channel))
    };
    let outcome = match tokio::time::timeout(XDS_BUILD_TIMEOUT, build).await {
        Ok(result) => result,
        Err(_) => Err("xDS build timed out".to_string()),
    };
    let built = outcome.is_ok();
    if let Some(receiver) = global_stats_receiver() {
        receiver.incr(
            XDS_BUILD_METRIC_NAME,
            &[("result", if built { "built" } else { "skipped" })],
            1,
        );
    }
    match outcome {
        Ok(client) => {
            tracing::info!(eds = %eds_resource, "SidRetrievalClient: xDS P2C channel ready");
            Some(client)
        }
        Err(e) => {
            tracing::warn!(eds = %eds_resource, error = %e, "SidRetrievalClient: xDS unavailable; using DNS path");
            None
        }
    }
}

fn spawn_endpoint_gauge(mut state_rx: watch::Receiver<ServiceState>) {
    tokio::spawn(async move {
        loop {
            let count = state_rx
                .borrow_and_update()
                .endpoint_data
                .as_ref()
                .map(|cla| cla.num_endpoints())
                .unwrap_or(0);
            if let Some(receiver) = global_stats_receiver() {
                receiver.gauge(XDS_ENDPOINTS_METRIC_NAME, &[], count as f64);
            }
            if state_rx.changed().await.is_err() {
                break;
            }
        }
    });
}

#[async_trait]
impl SidRetrievalClient for ProdSidRetrievalClient {
    async fn retrieve(&self, request: RetrieveRequest) -> Result<RetrieveResponse, Status> {
        let start = Instant::now();
        let xds_client = self.xds_client();
        let path = if xds_client.is_some() { "xds" } else { "dns" };
        let call = async {
            match xds_client {
                Some(client) => send(client, request).await,
                None => self.call_dns(request).await,
            }
        };
        let result = tokio::time::timeout(Duration::from_millis(TIMEOUT_MS), call)
            .await
            .unwrap_or_else(|_| {
                Err(Status::deadline_exceeded(
                    "SidRetrievalClient: deadline exceeded",
                ))
            });
        if let Some(receiver) = global_stats_receiver() {
            let result_label = match &result {
                Ok(_) => "success",
                Err(status) if status.code() == tonic::Code::DeadlineExceeded => "timeout",
                Err(_) => "failure",
            };
            receiver.incr(METRIC_NAME, &[("result", result_label), ("path", path)], 1);
            receiver.observe(
                METRIC_NAME,
                &[("path", path)],
                start.elapsed().as_millis() as f64,
                HistogramBuckets::Bucket500To1000,
            );
        }
        result
    }
}

pub struct MockSidRetrievalClient;

#[async_trait]
impl SidRetrievalClient for MockSidRetrievalClient {
    async fn retrieve(&self, _request: RetrieveRequest) -> Result<RetrieveResponse, Status> {
        Ok(RetrieveResponse::default())
    }
}
