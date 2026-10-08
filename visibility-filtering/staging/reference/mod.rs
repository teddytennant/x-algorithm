use crate::config::{ENV_APP_ENV, ENV_REFERENCE, parse_env_flag};
use crate::filter::FilterTweets;
use crate::filter_tweets::{Comparator, FinishComparison};
use crate::params::ClientSwitches;
use crate::rules::SafetyLevel;
use crate::server_deps::{init_client_with_retry, with_metadata};
use crate::staging::reference_compare::{ReferenceCompareHarness, TweetVerdict};
use std::env;
use std::sync::Arc;
use strum::VariantNames;
use tokio::time::Instant;
use tonic::metadata::MetadataMap;
use tweetypie::TweetypieReference;
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_visibility_filtering::vf_client::{StratoVfClient, VfClient};

pub(crate) mod tweetypie;

const ENV_DUAL_CALL_HARNESS_ENABLED: &str = "VF_DUAL_CALL_HARNESS_ENABLED";
pub const ENV_IMAGE: &str = "VF_IMAGE";
pub const ENV_TWEETYPIE_XDS_LISTENER: &str = "VF_TWEETYPIE_XDS_LISTENER";
pub const ENV_TWEETYPIE_TLS_DOMAIN: &str = "VF_TWEETYPIE_TLS_DOMAIN";
pub const ENV_TWEETYPIE_CLIENT_ID: &str = "VF_TWEETYPIE_CLIENT_ID";

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::VariantNames)]
#[strum(serialize_all = "snake_case")]
pub enum Reference {
    None,
        VfService,
        Tweetypie,
}

impl Reference {
        pub fn resolve(
        configured: Option<&str>,
        legacy_harness_flag: bool,
        app_env: Option<&str>,
    ) -> Result<Self, String> {
        if legacy_harness_flag {
            return Err(format!(
                "{ENV_DUAL_CALL_HARNESS_ENABLED} is replaced by {ENV_REFERENCE}=vf_service"
            ));
        }
        let reference = configured.map_or(Ok(Self::None), |value| {
            value.parse().map_err(|_| {
                format!(
                    "{ENV_REFERENCE}={value:?} is not one of {}",
                    Self::VARIANTS.join(", ")
                )
            })
        })?;
        match (reference, app_env) {
            (Self::None, _) => Ok(Self::None),
            (_, Some("prod")) => Err(format!(
                "{ENV_REFERENCE} selects {reference:?} but {ENV_APP_ENV}=prod; reference comparisons are staging-only"
            )),
            (reference, _) => Ok(reference),
        }
    }
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
fn reference() -> Reference {
    Reference::resolve(
        env::var(ENV_REFERENCE).ok().as_deref(),
        parse_env_flag(env::var(ENV_DUAL_CALL_HARNESS_ENABLED).ok().as_deref()),
        env::var(ENV_APP_ENV).ok().as_deref(),
    )
    .unwrap_or_else(|misconfiguration| panic!("{misconfiguration}"))
}

pub(crate) async fn build(
    datacenter: &str,
    init_deadline: Instant,
    filter_tweets: &Arc<FilterTweets>,
    client_switches: &ClientSwitches,
    metadata: Option<&MetadataMap>,
) -> Option<Box<dyn Comparator>> {
    match reference() {
        Reference::None => None,
        Reference::VfService => Some(Box::new(
            build_vf_service(datacenter, init_deadline, metadata).await,
        )),
        Reference::Tweetypie => Some(Box::new(
            tweetypie::build(
                init_deadline,
                filter_tweets,
                client_switches.clone(),
                metadata,
            )
            .await,
        )),
    }
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
async fn build_vf_service(
    datacenter: &str,
    init_deadline: Instant,
    metadata: Option<&MetadataMap>,
) -> Arc<ReferenceCompareHarness> {
    let client_id = format!(
        "visibility-filtering-service.{}",
        env::var(ENV_APP_ENV).unwrap_or_else(|_| "staging".to_string())
    );
    let strato: Arc<dyn VfClient + Send + Sync> = Arc::new(
        init_client_with_retry("strato_vf", init_deadline, || {
            let client_id = client_id.clone();
            async move {
                let mut client = StratoVfClient::new(
                    S2S_CHAIN_PATH.clone(),
                    S2S_CRT_PATH.clone(),
                    S2S_KEY_PATH.clone(),
                    client_id,
                    datacenter.to_string(),
                )
                .await
                .map_err(|e| e.to_string())?;
                client.grpc_client = Arc::new(with_metadata(
                    Arc::unwrap_or_clone(client.grpc_client),
                    metadata,
                ));
                Ok::<_, String>(client)
            }
        })
        .await
        .expect("Failed to initialize Strato VF client (reference comparator)"),
    );
    Arc::new(ReferenceCompareHarness::new(strato, datacenter))
}

impl Comparator for Arc<ReferenceCompareHarness> {
    fn begin(
        &self,
        viewer_id: Option<u64>,
        country_code: Option<String>,
        safety_level: SafetyLevel,
        tweet_ids: Vec<u64>,
    ) -> Option<FinishComparison> {
        let verdicts = self.begin_compare(viewer_id, country_code, safety_level, tweet_ids)?;
        Some(Box::new(move |outcomes| {
            verdicts.send(
                outcomes
                    .iter()
                    .map(|outcome| TweetVerdict {
                        tweet_id: outcome.tweet_id.0,
                        verdict: outcome.evaluation.verdict().clone(),
                    })
                    .collect(),
            );
        }))
    }
}

impl Comparator for Arc<TweetypieReference> {
    fn begin(
        &self,
        viewer_id: Option<u64>,
        country_code: Option<String>,
        safety_level: SafetyLevel,
        _tweet_ids: Vec<u64>,
    ) -> Option<FinishComparison> {
        let reference = Arc::clone(self);
        Some(Box::new(move |outcomes| {
            reference.spawn(viewer_id, country_code, safety_level, outcomes);
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::Reference;

    #[test]
    fn reference_defaults_to_none_and_is_refused_in_prod() {
        assert_eq!(
            Reference::resolve(None, false, Some("staging")),
            Ok(Reference::None)
        );
        assert_eq!(
            Reference::resolve(Some("none"), false, Some("staging")),
            Ok(Reference::None)
        );
        assert_eq!(
            Reference::resolve(Some("vf_service"), false, Some("staging")),
            Ok(Reference::VfService)
        );
        assert_eq!(
            Reference::resolve(Some("tweetypie"), false, Some("staging")),
            Ok(Reference::Tweetypie)
        );
        assert!(Reference::resolve(Some("VfService"), false, Some("staging")).is_err());
        assert!(Reference::resolve(Some("vf_service"), false, Some("prod")).is_err());
        assert_eq!(
            Reference::resolve(Some("none"), false, Some("prod")),
            Ok(Reference::None)
        );
        assert!(Reference::resolve(None, true, Some("staging")).is_err());
    }
}
