use crate::clients::about_this_account_client::ProdAboutThisAccountClient;
use crate::clients::article_client::ProdArticleClient;
use crate::clients::socialgraph_client::ProdSocialgraphClient;
use crate::clients::trusted_friends_client::ProdTrustedFriendsClient;
use crate::clients::user_location_client::ProdUserLocationClient;
use crate::clients::wingman_client::ProdWingmanClient;
use crate::evaluate_tweets::EvaluateTweetsEndpoint;
use crate::filter::{FilterRequest, FilterResponse, FilterTweets};
use crate::filter_tweets::{Comparator, FilterTweetsEndpoint};
use crate::get_safety_labels::GetSafetyLabelsEndpoint;
use crate::hydration::community_source::CommunitySource;
use crate::hydration::sources::ProdSources;
use crate::hydration::tweet_source::TweetSource;
use crate::hydration::{AuthorFallbackCache, Lookup, TweetFallbackCache};
use crate::limited_actions_copy::LimitedActionsCopy;
use crate::models::{ClientCapability, Evaluation, RawCandidate, TweetId};
use crate::params::ClientSwitches;
use crate::rules::metrics::Rpc;
use crate::rules::SafetyLevel;
use crate::safety_label_source::lookup::RemoteSource;
use crate::safety_label_source::manhattan::ManhattanSource;
use crate::safety_label_source::twemcache::TwemcacheSource;
use crate::safety_label_source::warmer::{CacheWarmer, StratoWarmFetcher, Warmer};
use crate::safety_label_source::{ManhattanLabelFetcher, MhLabelClient, SafetyLabelSource};
use crate::server::VFServer;
use anyhow::Context;
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tonic::metadata::MetadataMap;
use tracing::{error, info, warn};
use xai_cache::discovery::WilyDiscovery;
use xai_cache::{
    CacheClient, ClientBuilder, ClientConfig, HashAlgorithm, Key, Protocol, TlsConfig,
};
use xai_core_entities::gizmoduck_client::ProdGizmoduckClient;
use xai_core_entities::rpc_constants::{GizmoduckRpcConstants, RpcConstants, TESRpcConstants};
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CLIENT_ID, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_core_entities::tweet_entity_service_client::ProdTESClient;
use xai_strato::StratoGrpc;
use xai_x_rpc::balanced_channel::LbPolicy;
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};
use xai_x_rpc::retry::RetryConfig;
use xai_x_rpc::timed_buffer::DEFAULT_BUFFER_MAX_WAIT;
use xai_x_rpc::total_timeout::DEFAULT_TOTAL_TIMEOUT;
use xai_xds_client::StartFrom;

const CACHE_PATH: &str = "/s/cache/safety_label_store:twemcaches";
const READINESS_PROBE_PORT: u16 = 8081;

pub(crate) const CLIENT_INIT_RETRY_BUDGET: Duration = Duration::from_secs(240);
const CLIENT_INIT_MAX_BACKOFF: Duration = Duration::from_secs(15);
const CLIENT_INIT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) fn with_metadata(strato: StratoGrpc, metadata: Option<&MetadataMap>) -> StratoGrpc {
    match metadata {
        Some(metadata) => strato.with_default_metadata(metadata.clone()),
        None => strato,
    }
}

pub(crate) async fn init_client_with_retry<T, E, Fut>(
    client: &str,
    deadline: tokio::time::Instant,
    mut build: impl FnMut() -> Fut,
) -> Result<T, String>
where
    Fut: Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut backoff = Duration::from_secs(1);
    let mut attempt = 1u32;
    loop {
        if tokio::time::Instant::now() >= deadline {
            error!(
                client,
                attempt, "Client init skipped; init deadline exhausted"
            );
            return Err("init deadline exhausted before attempt".to_string());
        }
        let error = match tokio::time::timeout(CLIENT_INIT_ATTEMPT_TIMEOUT, build()).await {
            Ok(Ok(t)) => {
                if attempt > 1 {
                    info!(client, attempt, "Client init succeeded after retry");
                }
                return Ok(t);
            }
            Ok(Err(e)) => e.to_string(),
            Err(_) => format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ),
        };
        if tokio::time::Instant::now() + backoff >= deadline {
            error!(
                client,
                attempt,
                error = %error,
                "Client init failed; retry budget exhausted"
            );
            return Err(error);
        }
        warn!(
            client,
            attempt,
            backoff_secs = backoff.as_secs(),
            error = %error,
            "Client init failed; retrying"
        );
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(CLIENT_INIT_MAX_BACKOFF);
        attempt += 1;
    }
}

pub(crate) struct ServerDeps {
    pub(crate) init_deadline: tokio::time::Instant,
    pub(crate) filter_tweets: Arc<FilterTweets>,
    pub(crate) client_switches: ClientSwitches,
    limited_actions_copy: LimitedActionsCopy,
    safety_label_source: Arc<SafetyLabelSource>,
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
pub(crate) async fn build(datacenter: &str, metadata: Option<&MetadataMap>) -> ServerDeps {
    info!("Initializing prod clients for datacenter={}", datacenter);

    let (limited_actions_copy, copy_drift) = LimitedActionsCopy::load_baked(
        datacenter,
        xai_stats_receiver::global_stats_receiver().as_deref(),
    );
    let init_deadline = tokio::time::Instant::now() + CLIENT_INIT_RETRY_BUDGET;

    let deterministic_aperture = std::env::var("APP_ENV").as_deref() == Ok("prod");
    let author_cache_capacity = crate::config::author_cache_capacity();
    let author_cache = author_cache_capacity.map(crate::hydration::author_fallback_cache);
    let tweet_cache_capacity = crate::config::tweet_cache_capacity();
    let tweet_cache = tweet_cache_capacity.map(crate::hydration::tweet_fallback_cache);

    let (sources, safety_label_source) = prod_sources(
        datacenter,
        init_deadline,
        deterministic_aperture,
        author_cache,
        tweet_cache,
        metadata,
    )
    .await;
    let stats = xai_stats_receiver::global_stats_receiver();
    let mut switch_files = crate::params::SwitchFiles::beside(&crate::config::fs_path());
    let feature_switches = Arc::new(arc_swap::ArcSwap::from_pointee(
        switch_files
            .load(stats.as_deref())
            .expect("files that each built an engine build one together"),
    ));
    let country_lists = Arc::new(crate::params::CountryLists::starting_at_default());
    country_lists.refresh(&feature_switches.load());
    let client_switches = crate::params::ClientSwitches::new(Arc::clone(&feature_switches));
    crate::params::spawn_refresh(
        switch_files,
        feature_switches,
        Arc::clone(&country_lists),
        copy_drift,
        stats,
    );
    let rule_engine = crate::rules::RuleEngine::with_country_lists(country_lists);
    let (home_rule_count, recommendations_rule_count) = rule_engine.rule_counts();
    let filter_tweets = Arc::new(FilterTweets::new(Arc::new(sources), rule_engine));

    warm_filter_tweets(&filter_tweets).await;

    info!(
        hydrator_count = 5,
        author_cache_capacity = author_cache_capacity.unwrap_or(0),
        tweet_cache_capacity = tweet_cache_capacity.unwrap_or(0),
        home_rule_count,
        recommendations_rule_count,
        "VFServer initialized with prod clients"
    );

    ServerDeps {
        init_deadline,
        filter_tweets,
        client_switches,
        limited_actions_copy,
        safety_label_source,
    }
}

impl ServerDeps {
    pub(crate) fn into_server(self, comparator: Option<Box<dyn Comparator>>) -> VFServer {
        VFServer::from_endpoints(
            EvaluateTweetsEndpoint::new(
                Arc::clone(&self.filter_tweets),
                self.client_switches,
                self.limited_actions_copy,
            ),
            FilterTweetsEndpoint::new(self.filter_tweets, comparator),
            GetSafetyLabelsEndpoint::new(self.safety_label_source),
        )
    }
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
pub(crate) async fn prod_sources(
    datacenter: &str,
    init_deadline: tokio::time::Instant,
    deterministic_aperture: bool,
    author_cache: Option<AuthorFallbackCache>,
    tweet_cache: Option<TweetFallbackCache>,
    metadata: Option<&MetadataMap>,
) -> (ProdSources, Arc<SafetyLabelSource>) {
    let tes_client = Arc::new(
        init_client_with_retry("tes", init_deadline, || async move {
            let strato = build_xds_strato(
                XdsStratoParams {
                    name: "tes-xds",
                    xds_listener: "tweet-entity-service.prod.tweet-entity-service:fed-grpc",
                    tls_domain: format!(
                        "tweet-entity-service.tweet-entity-service.prod.{datacenter}.s2s.twttr.net"
                    ),
                    lb_policy: LbPolicy::penalized_peak_ewma(),
                    client_id: S2S_CLIENT_ID.clone(),
                    retry_config: Some(RetryConfig::for_idempotent()),
                    max_batch_size: TESRpcConstants::max_batch_size(),
                    metadata: metadata.cloned(),
                },
                deterministic_aperture,
            )
            .await?;
            anyhow::Ok(ProdTESClient {
                grpc_client: Arc::new(strato),
            })
        })
        .await
        .expect("Failed to initialize TES client"),
    );

    let gizmoduck_client_id = crate::config::gizmoduck_client_id();
    let gizmoduck_client: Arc<
        dyn xai_core_entities::gizmoduck_client::GizmoduckClient + Send + Sync,
    > = Arc::new(
        init_client_with_retry("gizmoduck", init_deadline, || {
            let client_id = gizmoduck_client_id.clone();
            async move {
                let strato = build_xds_strato(
                    XdsStratoParams {
                        name: "gizmoduck-xds",
                        xds_listener: "gizmoduck.prod.gizmoduck:fed-grpc",
                        tls_domain: format!("gizmoduck.gizmoduck.prod.{datacenter}.s2s.twttr.net"),
                        lb_policy: LbPolicy::least_request(),
                        client_id,
                        retry_config: None,
                        max_batch_size: GizmoduckRpcConstants::max_batch_size(),
                        metadata: metadata.cloned(),
                    },
                    deterministic_aperture,
                )
                .await?;
                anyhow::Ok(ProdGizmoduckClient {
                    grpc_client: Arc::new(strato),
                })
            }
        })
        .await
        .expect("Failed to initialize Gizmoduck client"),
    );
    info!(
        deterministic_aperture,
        "TES and Gizmoduck clients ready over xDS"
    );

    let sg_client: Arc<dyn crate::clients::socialgraph_client::SocialgraphClient + Send + Sync> =
        Arc::new(
            init_client_with_retry("socialgraph", init_deadline, || {
                ProdSocialgraphClient::new(
                    datacenter,
                    &S2S_CHAIN_PATH,
                    &S2S_CRT_PATH,
                    &S2S_KEY_PATH,
                    deterministic_aperture,
                    metadata.cloned(),
                )
            })
            .await
            .expect("Failed to initialize SocialGraph client"),
        );

    let stratoserver = init_client_with_retry("stratoserver", init_deadline, || {
        let config = xai_strato::StratoGrpcConfig {
            ca_cert_path: S2S_CHAIN_PATH.clone(),
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            aperture_size: Some(STRATO_APERTURE_SIZE),
            deterministic_aperture,
            connect_timeout_ms: u64::try_from(STRATO_CONNECT_TIMEOUT.as_millis())
                .unwrap_or(u64::MAX),
            request_timeout_ms: u64::try_from(STRATO_REQUEST_TIMEOUT.as_millis())
                .unwrap_or(u64::MAX),
            client_id: Some(S2S_CLIENT_ID.clone()),
            service_url: format!("stratostore.stratoserver.prod.{datacenter}.s2s.twttr.net"),
            zone: datacenter.to_string(),
            ..Default::default()
        };
        async move { anyhow::Ok(with_metadata(StratoGrpc::new(config).await?, metadata)) }
    })
    .await
    .expect("Failed to initialize the stratoserver client");
    let communities = CommunitySource {
        grpc_client: stratoserver.clone(),
    };
    let about_this_account_client = Arc::new(ProdAboutThisAccountClient::new(stratoserver.clone()));
    let trusted_friends_client = Arc::new(ProdTrustedFriendsClient::new(stratoserver.clone()));
    let user_location_client = Arc::new(ProdUserLocationClient::new(stratoserver.clone()));
    let article_client = Arc::new(ProdArticleClient::new(stratoserver));

    let wingman_client = Arc::new(
        init_client_with_retry("wingman", init_deadline, || {
            ProdWingmanClient::new(datacenter)
        })
        .await
        .expect("Failed to initialize Wingman client"),
    );

    let mh_label_client: Arc<dyn ManhattanLabelFetcher> = Arc::new(
        init_client_with_retry("manhattan", init_deadline, || {
            let s2s = xai_manhattan::s2s::S2sConfig {
                client_cert_path: S2S_CRT_PATH.clone(),
                client_key_path: S2S_KEY_PATH.clone(),
                ca_cert_path: S2S_CHAIN_PATH.clone(),
            };
            MhLabelClient::new(datacenter, s2s, deterministic_aperture)
        })
        .await
        .expect("Failed to initialize MhLabelClient"),
    );

    let twemcache_client_name = crate::config::twemcache_client_name();
    let twemcache = Arc::new(
        init_client_with_retry("twemcache", init_deadline, || {
            let name = twemcache_client_name.clone();
            let zone = datacenter.to_string();
            async move {
                let discovery = Arc::new(WilyDiscovery::new(CACHE_PATH, name, zone).await?);
                let config = ClientConfig::builder()
                    .request_timeout(Duration::from_millis(20))
                    .connect_timeout(Duration::from_secs(5))
                    .hash_algorithm(HashAlgorithm::FNV1)
                    .connections_per_endpoint(2)
                    .depth_cap(100)
                    .failure_accrual_enabled(false)
                    .build();
                ClientBuilder::new(Protocol::Memcached, discovery, config)
                    .with_tls(TlsConfig {
                        ca_cert_path: S2S_CHAIN_PATH.clone(),
                        client_cert_path: S2S_CRT_PATH.clone(),
                        client_key_path: S2S_KEY_PATH.clone(),
                    })
                    .build()
                    .await
            }
        })
        .await
        .expect("Failed to create twemcache client"),
    );
    let start = Instant::now();
    let server_count = twemcache.warm_up().await;
    info!(
        server_count,
        latency_ms = elapsed_ms(start),
        "Cache client connected to {CACHE_PATH}"
    );

    warm_cache(&twemcache).await;
    warm_manhattan(mh_label_client.as_ref()).await;

    let cache_warmer =
        build_cache_warmer(datacenter, init_deadline, deterministic_aperture, metadata).await;

    let twemcache_source = Arc::new(TwemcacheSource::new(twemcache));
    let manhattan_source = Arc::new(ManhattanSource::new(mh_label_client));
    let mut remote = RemoteSource::new(twemcache_source, manhattan_source);
    if let Some(warmer) = cache_warmer {
        remote = remote.with_warmer(warmer);
    }
    let remote = Arc::new(remote);
    let safety_label_source = Arc::new(SafetyLabelSource::new(
        remote,
        crate::config::safety_label_cache_capacity(),
    ));

    let tweet_source = TweetSource {
        grpc_client: Arc::clone(&tes_client.grpc_client),
    };
    let sources = ProdSources::new(
        tes_client,
        tweet_source,
        gizmoduck_client,
        sg_client,
        about_this_account_client,
        wingman_client,
        article_client,
        trusted_friends_client,
        user_location_client,
        Arc::clone(&safety_label_source),
        communities,
        author_cache,
        tweet_cache,
    );
    (sources, safety_label_source)
}

const CACHE_WARM_REQUEST_TIMEOUT_MS: u64 = 500;

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
async fn build_cache_warmer(
    datacenter: &str,
    init_deadline: tokio::time::Instant,
    deterministic_aperture: bool,
    metadata: Option<&MetadataMap>,
) -> Option<Arc<dyn Warmer>> {
    if !crate::config::cache_warm_enabled() {
        return None;
    }

    let client_id = format!(
        "visibility-filtering-service.{}",
        std::env::var("APP_ENV").unwrap_or_else(|_| "prod".to_string())
    );
    let grpc = init_client_with_retry("strato_cache_warm", init_deadline, || {
        let config = xai_strato::StratoGrpcConfig {
            ca_cert_path: S2S_CHAIN_PATH.clone(),
            client_cert_path: S2S_CRT_PATH.clone(),
            client_key_path: S2S_KEY_PATH.clone(),
            aperture_size: Some(STRATO_APERTURE_SIZE),
            deterministic_aperture,
            connect_timeout_ms: 400,
            request_timeout_ms: CACHE_WARM_REQUEST_TIMEOUT_MS,
            client_id: Some(client_id.clone()),
            service_url: format!("stratostore.stratoserver.prod.{datacenter}.s2s.twttr.net"),
            zone: datacenter.to_string(),
            ..Default::default()
        };
        async move { anyhow::Ok(with_metadata(StratoGrpc::new(config).await?, metadata)) }
    })
    .await
    .expect("Failed to initialize Strato cache-warm client");
    info!("L2 cache warmer enabled");
    Some(CacheWarmer::spawn(Arc::new(StratoWarmFetcher::new(grpc))))
}

const STRATO_REQUEST_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const STRATO_CONNECT_TIMEOUT: Duration = Duration::from_millis(400);
const STRATO_APERTURE_SIZE: usize = 12;
const XDS_EAGER_RESOLUTION_TIMEOUT: Duration = Duration::from_secs(5);

struct XdsStratoParams {
    name: &'static str,
    xds_listener: &'static str,
    tls_domain: String,
    lb_policy: LbPolicy,
    client_id: String,
    retry_config: Option<RetryConfig>,
    max_batch_size: usize,
    metadata: Option<MetadataMap>,
}

async fn build_xds_strato(
    params: XdsStratoParams,
    deterministic_aperture: bool,
) -> anyhow::Result<StratoGrpc> {
    let mut builder = ChannelBuilder::new(params.name)
        .tls(
            TlsMode::mtls_from_env()
                .context("S2S cert env vars required for mTLS")?
                .with_domain_override(params.tls_domain),
        )
        .request_timeout(STRATO_REQUEST_TIMEOUT)
        .connect_timeout(STRATO_CONNECT_TIMEOUT)
        .xds(StartFrom::Lds(params.xds_listener.to_string()))
        .await
        .with_context(|| format!("failed to initialize xDS for {}", params.xds_listener))?
        .eager_resolution(XDS_EAGER_RESOLUTION_TIMEOUT)
        .aperture(STRATO_APERTURE_SIZE)
        .readiness_probe(READINESS_PROBE_PORT, "/ready")
        .buffer_max_wait(DEFAULT_BUFFER_MAX_WAIT)
        .total_timeout(DEFAULT_TOTAL_TIMEOUT);
    if deterministic_aperture {
        builder = builder.deterministic();
    }

    let channel = builder
        .build_load_balanced(params.lb_policy)
        .await
        .with_context(|| {
            format!(
                "failed to build xDS LoadBalancedChannel for {}",
                params.name
            )
        })?;

    Ok(StratoGrpc::from_load_balanced_channel(
        channel,
        params.metadata,
        Some(params.client_id),
        params.retry_config,
        params.max_batch_size,
    ))
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

async fn warm_cache(twemcache: &CacheClient) {
    let start = Instant::now();
    let key = match Key::new(b"slm_warmup".to_vec()) {
        Ok(key) => key,
        Err(e) => {
            warn!(error = %e, "Cache warmup key construction failed (non-fatal)");
            return;
        }
    };
    match twemcache.multi_get(std::slice::from_ref(&key)).await {
        Ok(map) => match map.get(&key) {
            Some(Ok(_)) => info!(latency_ms = elapsed_ms(start), "Cache warmup succeeded"),
            Some(Err(e)) => warn!(
                latency_ms = elapsed_ms(start),
                error = %e,
                "Cache warmup failed (non-fatal)"
            ),
            None => warn!(
                latency_ms = elapsed_ms(start),
                "Cache warmup returned no response (non-fatal)"
            ),
        },
        Err(e) => warn!(
            latency_ms = elapsed_ms(start),
            error = %e,
            "Cache warmup failed (non-fatal)"
        ),
    }
}

const WARM_FILTER_TWEETS_TWEET_ID: u64 = 20;
const WARM_FILTER_TWEETS_CONSECUTIVE: u32 = 3;
const WARM_FILTER_TWEETS_DEADLINE: Duration = Duration::from_secs(20);
const WARM_FILTER_TWEETS_SLEEP: Duration = Duration::from_millis(500);

fn warm_filter_tweets_request() -> FilterRequest {
    FilterRequest {
        viewer_id: None,
        country_code: None,
        client_capability: ClientCapability::default(),
        safety_level: SafetyLevel::TimelineHomeRecommendations,
        candidates: vec![RawCandidate {
            tweet_id: TweetId(WARM_FILTER_TWEETS_TWEET_ID),
            request_author_id: None,
        }],
        rpc: Rpc::FilterTweets,
    }
}

fn filter_tweets_response_is_warm(response: &FilterResponse) -> bool {
    response
        .outcomes
        .iter()
        .all(|outcome| match outcome.evaluation {
            Evaluation::NotFound(lookup) | Evaluation::Failed(lookup) => match lookup {
                Lookup::Tweet => false,
                Lookup::Author | Lookup::SharedTweet | Lookup::SharedAuthor => true,
            },
            Evaluation::Complete { .. } | Evaluation::Partial { .. } => true,
        })
}

async fn warm_filter_tweets(filter_tweets: &FilterTweets) {
    let start = Instant::now();
    let mut attempts = 0u32;
    let mut consecutive_warm = 0u32;
    while start.elapsed() < WARM_FILTER_TWEETS_DEADLINE {
        attempts += 1;
        let response = filter_tweets.run(warm_filter_tweets_request()).await;
        if filter_tweets_response_is_warm(&response) {
            consecutive_warm += 1;
            if consecutive_warm >= WARM_FILTER_TWEETS_CONSECUTIVE {
                info!(
                    attempts,
                    elapsed_ms = elapsed_ms(start),
                    "filter_tweets warmup succeeded"
                );
                return;
            }
        } else {
            consecutive_warm = 0;
        }
        if start.elapsed() + WARM_FILTER_TWEETS_SLEEP < WARM_FILTER_TWEETS_DEADLINE {
            tokio::time::sleep(WARM_FILTER_TWEETS_SLEEP).await;
        } else {
            break;
        }
    }
    warn!(
        attempts,
        elapsed_ms = elapsed_ms(start),
        "filter_tweets warmup deadline expired (non-fatal)"
    );
}

async fn warm_manhattan(manhattan: &dyn ManhattanLabelFetcher) {
    let start = Instant::now();
    match manhattan.fetch_labels(&[0]).await {
        Ok(_) => info!(latency_ms = elapsed_ms(start), "Manhattan warmup succeeded"),
        Err(e) => warn!(
            latency_ms = elapsed_ms(start),
            error = %e,
            "Manhattan warmup failed (non-fatal)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn deadline_in(budget: Duration) -> tokio::time::Instant {
        tokio::time::Instant::now() + budget
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_recovers_from_transient_failures() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();

        let result = init_client_with_retry("test", deadline_in(Duration::from_secs(240)), || {
            attempts.set(attempts.get() + 1);
            let n = attempts.get();
            async move {
                if n < 3 {
                    Err("discovery timed out")
                } else {
                    Ok(n)
                }
            }
        })
        .await;

        assert_eq!(result, Ok(3));
        assert_eq!(attempts.get(), 3);
        assert_eq!(start.elapsed(), Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_gives_up_at_deadline() {
        let attempts = Cell::new(0u32);
        let start = tokio::time::Instant::now();
        let budget = Duration::from_secs(240);

        let result: Result<(), String> =
            init_client_with_retry("test", deadline_in(budget), || {
                attempts.set(attempts.get() + 1);
                async { Err("discovery timed out") }
            })
            .await;

        assert_eq!(result, Err("discovery timed out".to_string()));
        assert!(attempts.get() > 1);
        assert!(start.elapsed() < budget);
        assert!(start.elapsed() >= budget - CLIENT_INIT_MAX_BACKOFF);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_skips_attempt_when_deadline_already_passed() {
        let deadline = tokio::time::Instant::now();
        tokio::time::sleep(Duration::from_secs(1)).await;
        let attempts = Cell::new(0u32);

        let result: Result<(), String> = init_client_with_retry("test", deadline, || {
            attempts.set(attempts.get() + 1);
            async { Err("unreachable") }
        })
        .await;

        assert_eq!(
            result,
            Err("init deadline exhausted before attempt".to_string())
        );
        assert_eq!(attempts.get(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn init_retry_times_out_hanging_attempt() {
        let start = tokio::time::Instant::now();

        let result: Result<(), String> = init_client_with_retry(
            "test",
            deadline_in(CLIENT_INIT_ATTEMPT_TIMEOUT / 2),
            || async {
                std::future::pending::<()>().await;
                Err("unreachable")
            },
        )
        .await;

        assert_eq!(
            result,
            Err(format!(
                "init attempt timed out after {}s",
                CLIENT_INIT_ATTEMPT_TIMEOUT.as_secs()
            ))
        );
        assert_eq!(start.elapsed(), CLIENT_INIT_ATTEMPT_TIMEOUT);
    }
}
