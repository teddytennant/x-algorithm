use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use thrift::OrderedFloat;
use tonic::async_trait;
use xai_cache::discovery::WilyDiscovery;
use xai_cache::{
    CacheClient, ClientBuilder, ClientConfig, HashAlgorithm, Key, Protocol, TlsConfig,
};
use xai_x_thrift::simclusters_ann::SimClustersANNTweetCandidate;

use crate::clients::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};

const WILY_PATH: &str = "/s/cache/content_recommender_unified_v2:twemcaches";
const CLIENT_NAME: &str = "home-mixer";
const KEY_PREFIX: &str = "HOME:SANN:v1:";

const REQUEST_TIMEOUT: Duration = Duration::from_millis(50);
const CONNECTIONS_PER_ENDPOINT: usize = 2;
const RETRIES: usize = 0;
const FAILURE_THRESHOLD: u32 = 15;
const RECOVERY_INTERVAL: Duration = Duration::from_secs(5);

const TTL_SECS: u32 = 600;
const TTL_EARLY_EXPIRATION: f64 = 0.2;

const CANDIDATE_BYTES: usize = 16;

pub type CachedCandidates = Result<Option<Vec<SimClustersANNTweetCandidate>>, String>;

#[async_trait]
pub trait SimClustersAnnCacheClient: Send + Sync {
    async fn multi_get(
        &self,
        seed_ids: &[i64],
        max_candidate_age_hours: i32,
    ) -> HashMap<i64, CachedCandidates>;

    async fn set(
        &self,
        seed_id: i64,
        max_candidate_age_hours: i32,
        candidates: &[SimClustersANNTweetCandidate],
    ) -> Result<(), String>;
}

pub struct ProdSimClustersAnnCacheClient {
    client: Arc<CacheClient>,
}

impl ProdSimClustersAnnCacheClient {
    pub async fn new(datacenter: &str) -> anyhow::Result<Self> {
        let discovery = Arc::new(
            WilyDiscovery::new(WILY_PATH, CLIENT_NAME, datacenter)
                .await
                .map_err(|e| anyhow::anyhow!("SimClusters ANN cache discovery failed: {e}"))?,
        );

        let config = ClientConfig::builder()
            .request_timeout(REQUEST_TIMEOUT)
            .hash_algorithm(HashAlgorithm::FNV1)
            .connections_per_endpoint(CONNECTIONS_PER_ENDPOINT)
            .retries(RETRIES)
            .failure_threshold(FAILURE_THRESHOLD)
            .recovery_interval(RECOVERY_INTERVAL)
            .build();

        let client = ClientBuilder::new(Protocol::Memcached, discovery, config)
            .with_tls(TlsConfig {
                ca_cert_path: S2S_CHAIN_PATH.clone(),
                client_cert_path: S2S_CRT_PATH.clone(),
                client_key_path: S2S_KEY_PATH.clone(),
            })
            .build()
            .await
            .map_err(|e| anyhow::anyhow!("SimClusters ANN cache client build failed: {e}"))?;

        tracing::info!("SimClusters ANN cache client connected to {WILY_PATH}");
        Ok(Self {
            client: Arc::new(client),
        })
    }
}

#[async_trait]
impl SimClustersAnnCacheClient for ProdSimClustersAnnCacheClient {
    async fn multi_get(
        &self,
        seed_ids: &[i64],
        max_candidate_age_hours: i32,
    ) -> HashMap<i64, CachedCandidates> {
        let mut output = HashMap::with_capacity(seed_ids.len());
        let mut seen = HashSet::with_capacity(seed_ids.len());
        let mut keys_with_ids: Vec<(i64, Key)> = Vec::with_capacity(seed_ids.len());
        for &seed_id in seed_ids {
            if !seen.insert(seed_id) {
                continue;
            }
            match make_key(seed_id, max_candidate_age_hours) {
                Ok(key) => keys_with_ids.push((seed_id, key)),
                Err(e) => {
                    output.insert(seed_id, Err(e));
                }
            }
        }
        if keys_with_ids.is_empty() {
            return output;
        }

        let keys: Vec<Key> = keys_with_ids.iter().map(|(_, key)| key.clone()).collect();
        let results = match self.client.multi_get(&keys).await {
            Ok(results) => results,
            Err(e) => {
                let msg = format!("SimClusters ANN cache multi_get failed: {e}");
                for (seed_id, _) in keys_with_ids {
                    output.insert(seed_id, Err(msg.clone()));
                }
                return output;
            }
        };

        for (seed_id, key) in keys_with_ids {
            let entry = match results.get(&key) {
                Some(Err(e)) => Err(format!("SimClusters ANN cache read failed: {e}")),
                Some(Ok(Some(bytes))) => Ok(decode(bytes)),
                Some(Ok(None)) | None => Ok(None),
            };
            output.insert(seed_id, entry);
        }
        output
    }

    async fn set(
        &self,
        seed_id: i64,
        max_candidate_age_hours: i32,
        candidates: &[SimClustersANNTweetCandidate],
    ) -> Result<(), String> {
        let key = make_key(seed_id, max_candidate_age_hours)?;
        self.client
            .set(&key, encode(candidates), ttl_secs())
            .await
            .map_err(|e| format!("SimClusters ANN cache set failed: {e}"))
    }
}

pub struct MockSimClustersAnnCacheClient;

#[async_trait]
impl SimClustersAnnCacheClient for MockSimClustersAnnCacheClient {
    async fn multi_get(
        &self,
        seed_ids: &[i64],
        _max_candidate_age_hours: i32,
    ) -> HashMap<i64, CachedCandidates> {
        seed_ids
            .iter()
            .map(|&seed_id| (seed_id, Ok(None)))
            .collect()
    }

    async fn set(
        &self,
        _seed_id: i64,
        _max_candidate_age_hours: i32,
        _candidates: &[SimClustersANNTweetCandidate],
    ) -> Result<(), String> {
        Ok(())
    }
}

fn make_key(seed_id: i64, max_candidate_age_hours: i32) -> Result<Key, String> {
    Key::new(format!("{KEY_PREFIX}{max_candidate_age_hours}:{seed_id}").into_bytes())
        .map_err(|e| format!("invalid SimClusters ANN cache key: {e}"))
}

fn ttl_secs() -> u32 {
    let ttl = f64::from(TTL_SECS);
    (ttl - ttl * TTL_EARLY_EXPIRATION * rand::random::<f64>()) as u32
}

fn encode(candidates: &[SimClustersANNTweetCandidate]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(candidates.len() * CANDIDATE_BYTES);
    for candidate in candidates {
        bytes.extend_from_slice(&candidate.tweet_id.to_be_bytes());
        bytes.extend_from_slice(&candidate.score.0.to_be_bytes());
    }
    bytes
}

fn decode(bytes: &[u8]) -> Option<Vec<SimClustersANNTweetCandidate>> {
    if !bytes.len().is_multiple_of(CANDIDATE_BYTES) {
        return None;
    }
    Some(
        bytes
            .chunks_exact(CANDIDATE_BYTES)
            .map(|chunk| {
                let mut tweet_id = [0u8; 8];
                let mut score = [0u8; 8];
                tweet_id.copy_from_slice(&chunk[..8]);
                score.copy_from_slice(&chunk[8..]);
                SimClustersANNTweetCandidate::new(
                    i64::from_be_bytes(tweet_id),
                    OrderedFloat(f64::from_be_bytes(score)),
                )
            })
            .collect(),
    )
}
