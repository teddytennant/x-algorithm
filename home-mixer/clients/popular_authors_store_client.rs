use std::time::Duration;

use tonic::async_trait;
use xai_core_entities::s2s::{S2S_CHAIN_PATH, S2S_CRT_PATH, S2S_KEY_PATH};
use xai_manhattan::s2s::S2sConfig;
use xai_manhattan::{NativeManhattanClient, Tenant};

use crate::util::popular_authors::{
    decode_stored, encode_stored, PopularAuthorsStore, StoredPopularAuthors,
};
use crate::util::popular_posts::{self, PopularPostsStore, StoredPopularPosts};

const CLUSTER: &str = "omega";
const APP_ID: &str = "timelineservice_user_session_store";
const DATASET: &str = "tls_user_session_store";
const POPULAR_AUTHORS_PKEY: i64 = 0;
const POPULAR_AUTHORS_DATASET_ID: i32 = 1001;
pub const POPULAR_AUTHORS_VERSION: i32 = 2;
pub const POPULAR_POSTS_VERSION: i32 = 3;
const TIMEOUT: Duration = Duration::from_millis(500);

fn tenant() -> Tenant {
    Tenant {
        cluster: CLUSTER.to_string(),
        app_id: APP_ID.to_string(),
        dataset: DATASET.to_string(),
    }
}

fn pkey() -> Vec<Vec<u8>> {
    vec![POPULAR_AUTHORS_PKEY.to_be_bytes().to_vec()]
}

fn lkey(version: i32) -> Vec<Vec<u8>> {
    vec![
        POPULAR_AUTHORS_DATASET_ID.to_be_bytes().to_vec(),
        version.to_be_bytes().to_vec(),
    ]
}

async fn build_client(dc: &str) -> anyhow::Result<NativeManhattanClient> {
    let s2s = S2sConfig {
        client_cert_path: S2S_CRT_PATH.clone(),
        client_key_path: S2S_KEY_PATH.clone(),
        ca_cert_path: S2S_CHAIN_PATH.clone(),
    };
    NativeManhattanClient::builder_from_tenant_s2s(&tenant(), dc, s2s)
        .timeout(TIMEOUT)
        .retries(1)
        .no_batch()
        .build()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to create popular store Manhattan client: {e}"))
}

async fn get_bytes(
    client: &NativeManhattanClient,
    version: i32,
) -> Result<Option<Vec<u8>>, String> {
    let item = client
        .get(tenant(), pkey(), lkey(version))
        .await
        .map_err(|e| format!("failed to get popular store key {version}: {e}"))?;
    Ok(item.map(|it| it.value().as_bytes().to_vec()))
}

async fn put_bytes(
    client: &NativeManhattanClient,
    version: i32,
    bytes: Vec<u8>,
) -> Result<(), String> {
    client
        .put(tenant(), pkey(), lkey(version), bytes)
        .await
        .map_err(|e| format!("failed to put popular store key {version}: {e}"))
}

pub async fn read_raw(dc: &str, version: i32) -> anyhow::Result<Option<Vec<u8>>> {
    let client = build_client(dc).await?;
    get_bytes(&client, version)
        .await
        .map_err(anyhow::Error::msg)
}

pub struct ManhattanPopularAuthorsStore {
    client: NativeManhattanClient,
}

impl ManhattanPopularAuthorsStore {
    pub async fn new(dc: &str) -> anyhow::Result<Self> {
        Ok(Self {
            client: build_client(dc).await?,
        })
    }
}

#[async_trait]
impl PopularAuthorsStore for ManhattanPopularAuthorsStore {
    async fn load(&self) -> Result<Option<StoredPopularAuthors>, String> {
        get_bytes(&self.client, POPULAR_AUTHORS_VERSION)
            .await?
            .map(|bytes| decode_stored(&bytes))
            .transpose()
    }

    async fn save(&self, stored: &StoredPopularAuthors) -> Result<(), String> {
        put_bytes(&self.client, POPULAR_AUTHORS_VERSION, encode_stored(stored)).await
    }
}

pub struct ManhattanPopularPostsStore {
    client: NativeManhattanClient,
}

impl ManhattanPopularPostsStore {
    pub async fn new(dc: &str) -> anyhow::Result<Self> {
        Ok(Self {
            client: build_client(dc).await?,
        })
    }
}

#[async_trait]
impl PopularPostsStore for ManhattanPopularPostsStore {
    async fn load(&self) -> Result<Option<StoredPopularPosts>, String> {
        get_bytes(&self.client, POPULAR_POSTS_VERSION)
            .await?
            .map(|bytes| popular_posts::decode_stored(&bytes))
            .transpose()
    }

    async fn save(&self, stored: &StoredPopularPosts) -> Result<(), String> {
        put_bytes(
            &self.client,
            POPULAR_POSTS_VERSION,
            popular_posts::encode_stored(stored),
        )
        .await
    }
}
