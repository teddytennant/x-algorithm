pub mod following_candidate_pipeline;
pub mod for_you_candidate_pipeline;
pub mod phoenix_candidate_pipeline;
pub mod phoenix_scores_pipeline;
pub mod ranked_following_candidate_pipeline;
pub mod reverse_chron_posts_pipeline;

use std::sync::Arc;

use crate::clients::author_brand_safety_client::{
    AuthorBrandSafetyClient, ProdAuthorBrandSafetyClient,
};
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::PhoenixPredictionClient;
use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalClient;
use xai_candidate_pipeline::component_library::clients::{
    build_phoenix_xds_prediction_client, build_phoenix_xds_retrieval_client_boxed,
};

pub use xai_candidate_pipeline::component_library::clients::{PhoenixXdsConfig, XdsP2cConfig};

pub type VmRankerXdsConfig = XdsP2cConfig;

pub(crate) async fn build_phoenix_xds_client(
    config: &PhoenixXdsConfig,
) -> Option<Arc<dyn PhoenixPredictionClient + Send + Sync>> {
    static PHOENIX_XDS_CLIENT: tokio::sync::OnceCell<
        Option<Arc<dyn PhoenixPredictionClient + Send + Sync>>,
    > = tokio::sync::OnceCell::const_new();
    PHOENIX_XDS_CLIENT
        .get_or_init(|| build_phoenix_xds_prediction_client(config))
        .await
        .clone()
}

pub(crate) async fn build_phoenix_xds_retrieval_client(
    config: &PhoenixXdsConfig,
) -> Option<Arc<dyn PhoenixRetrievalClient + Send + Sync>> {
    static PHOENIX_XDS_RETRIEVAL_CLIENT: tokio::sync::OnceCell<
        Option<Arc<dyn PhoenixRetrievalClient + Send + Sync>>,
    > = tokio::sync::OnceCell::const_new();
    PHOENIX_XDS_RETRIEVAL_CLIENT
        .get_or_init(|| build_phoenix_xds_retrieval_client_boxed(config))
        .await
        .clone()
}

pub(crate) async fn build_vm_ranker_xds_client(
    config: &VmRankerXdsConfig,
) -> Option<Arc<dyn crate::clients::vm_ranker_client::VMRankerClient>> {
    static VM_RANKER_XDS_CLIENT: tokio::sync::OnceCell<
        Option<Arc<dyn crate::clients::vm_ranker_client::VMRankerClient>>,
    > = tokio::sync::OnceCell::const_new();
    VM_RANKER_XDS_CLIENT
        .get_or_init(|| build_vm_ranker_xds_client_uncached(config))
        .await
        .clone()
}

async fn build_vm_ranker_xds_client_uncached(
    config: &VmRankerXdsConfig,
) -> Option<Arc<dyn crate::clients::vm_ranker_client::VMRankerClient>> {
    use crate::clients::vm_ranker_client::{VMRankerClient, XdsVMRankerClient};
    let policy = config.parse_lb_policy();
    let health_probe = config.parse_health_probe();
    tracing::info!(
        vm_ranker_xds_built = policy.is_some(),
        aperture_size = config.aperture_size,
        health_probe = ?health_probe,
        discovery_authority = %config.discovery_authority,
        "VMRanker: resolved xDS ranking client (startup config)"
    );
    match policy {
        Some(policy) => {
            match XdsVMRankerClient::new(
                policy,
                config.h2_stream_window_bytes,
                config.h2_connection_window_bytes,
                config.socket_buffer_bytes,
                config.aperture_size as usize,
                health_probe,
                &config.discovery_authority,
            )
            .await
            {
                Ok(client) => Some(Arc::new(client) as Arc<dyn VMRankerClient>),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "VMRanker: failed to build xDS P2C client; falling back to DNS path"
                    );
                    None
                }
            }
        }
        None => None,
    }
}

pub(crate) async fn shared_author_brand_safety_client(
    datacenter: &str,
) -> Arc<dyn AuthorBrandSafetyClient> {
    static AUTHOR_BRAND_SAFETY_CLIENT: tokio::sync::OnceCell<Arc<dyn AuthorBrandSafetyClient>> =
        tokio::sync::OnceCell::const_new();
    AUTHOR_BRAND_SAFETY_CLIENT
        .get_or_init(|| async {
            Arc::new(
                ProdAuthorBrandSafetyClient::new(datacenter)
                    .await
                    .expect("Failed to create AuthorBrandSafety client"),
            ) as Arc<dyn AuthorBrandSafetyClient>
        })
        .await
        .clone()
}
