use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use tonic::async_trait;

pub const REFRESH_INTERVAL_MS: i64 = 60 * 60 * 1000;
pub const EMPTY_REFRESH_INTERVAL_MS: i64 = 60 * 1000;
const STORE_FORMAT_VERSION: u8 = 1;
pub const TOP_POSTING_AUTHORS_FRACTION: f64 = 0.00005;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PopularAuthor {
    pub author_id: u64,
    pub follower_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoredPopularAuthors {
    pub updated_at_ms: i64,
    pub authors: Vec<PopularAuthor>,
}

pub fn select_top_posting_authors(
    by_followers_desc: &[PopularAuthor],
    active_posters_7d: u64,
    fraction: f64,
) -> Result<Vec<PopularAuthor>, String> {
    let k = (active_posters_7d as f64 * fraction).ceil() as usize;
    if k == 0 {
        return Err(format!(
            "no authors selected: {active_posters_7d} active posters x {fraction}"
        ));
    }
    if by_followers_desc.len() < k {
        return Err(format!(
            "need the top {k} posters by followers but the snapshot has {}",
            by_followers_desc.len()
        ));
    }
    Ok(by_followers_desc[..k].to_vec())
}

pub fn encode_stored(stored: &StoredPopularAuthors) -> Vec<u8> {
    let mut out = Vec::with_capacity(13 + 16 * stored.authors.len());
    out.push(STORE_FORMAT_VERSION);
    out.extend_from_slice(&stored.updated_at_ms.to_be_bytes());
    out.extend_from_slice(&(stored.authors.len() as u32).to_be_bytes());
    for a in &stored.authors {
        out.extend_from_slice(&a.author_id.to_be_bytes());
        out.extend_from_slice(&a.follower_count.to_be_bytes());
    }
    out
}

pub fn decode_stored(bytes: &[u8]) -> Result<StoredPopularAuthors, String> {
    let header = bytes
        .get(..13)
        .ok_or_else(|| format!("popular authors value too short: {} bytes", bytes.len()))?;
    if header[0] != STORE_FORMAT_VERSION {
        return Err(format!("unknown popular authors format {}", header[0]));
    }
    let updated_at_ms = i64::from_be_bytes(header[1..9].try_into().unwrap());
    let count = u32::from_be_bytes(header[9..13].try_into().unwrap()) as usize;
    let body = &bytes[13..];
    if body.len() != count * 16 {
        return Err(format!(
            "popular authors body is {} bytes, expected {}",
            body.len(),
            count * 16
        ));
    }
    let authors = body
        .chunks_exact(16)
        .map(|c| PopularAuthor {
            author_id: u64::from_be_bytes(c[..8].try_into().unwrap()),
            follower_count: u64::from_be_bytes(c[8..].try_into().unwrap()),
        })
        .collect();
    Ok(StoredPopularAuthors {
        updated_at_ms,
        authors,
    })
}

#[async_trait]
pub trait PopularAuthorsStore: Send + Sync {
    async fn load(&self) -> Result<Option<StoredPopularAuthors>, String>;
    async fn save(&self, stored: &StoredPopularAuthors) -> Result<(), String>;
}

#[derive(Default)]
pub struct InMemoryPopularAuthorsStore {
    value: Mutex<Option<StoredPopularAuthors>>,
}

#[async_trait]
impl PopularAuthorsStore for InMemoryPopularAuthorsStore {
    async fn load(&self) -> Result<Option<StoredPopularAuthors>, String> {
        Ok(self.value.lock().unwrap().clone())
    }

    async fn save(&self, stored: &StoredPopularAuthors) -> Result<(), String> {
        *self.value.lock().unwrap() = Some(stored.clone());
        Ok(())
    }
}

#[derive(Default)]
struct Snapshot {
    author_ids: HashSet<u64>,
    loaded_at_ms: Option<i64>,
}

pub struct PopularAuthorsCache {
    store: Arc<dyn PopularAuthorsStore>,
    snapshot: RwLock<Snapshot>,
    refreshing: AtomicBool,
}

impl PopularAuthorsCache {
    pub fn new(store: Arc<dyn PopularAuthorsStore>) -> Self {
        Self {
            store,
            snapshot: RwLock::new(Snapshot::default()),
            refreshing: AtomicBool::new(false),
        }
    }

    pub fn contains(&self, author_id: u64) -> bool {
        self.snapshot
            .read()
            .unwrap()
            .author_ids
            .contains(&author_id)
    }

    fn refresh_due(&self, now_ms: i64) -> bool {
        let snapshot = self.snapshot.read().unwrap();
        let interval = if snapshot.author_ids.is_empty() {
            EMPTY_REFRESH_INTERVAL_MS
        } else {
            REFRESH_INTERVAL_MS
        };
        snapshot.loaded_at_ms.is_none_or(|t| now_ms - t >= interval)
    }

    async fn refresh(&self, now_ms: i64) -> Result<usize, String> {
        let authors = self
            .store
            .load()
            .await?
            .map(|s| s.authors)
            .unwrap_or_default();
        let author_ids: HashSet<u64> = authors.iter().map(|a| a.author_id).collect();
        let loaded = author_ids.len();
        tracing::info!(
            loaded,
            min_followers = authors.iter().map(|a| a.follower_count).min().unwrap_or(0),
            "popular authors loaded"
        );
        *self.snapshot.write().unwrap() = Snapshot {
            author_ids,
            loaded_at_ms: Some(now_ms),
        };
        Ok(loaded)
    }

    pub fn maybe_spawn_refresh(self: &Arc<Self>, now_ms: i64) {
        if !self.refresh_due(now_ms) || self.refreshing.swap(true, Ordering::AcqRel) {
            return;
        }
        let cache = Arc::clone(self);
        tokio::spawn(async move {
            let result = cache.refresh(now_ms).await;
            if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
                match &result {
                    Ok(loaded) => {
                        receiver.incr("PopularAuthorsCache.Refresh", &[("result", "ok")], 1);
                        receiver.gauge("PopularAuthorsCache.LoadedAuthors", &[], *loaded as f64);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "popular authors refresh failed");
                        receiver.incr("PopularAuthorsCache.Refresh", &[("result", "error")], 1);
                    }
                }
            }
            cache.refreshing.store(false, Ordering::Release);
        });
    }
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}
