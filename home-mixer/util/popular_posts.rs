use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use tonic::async_trait;

pub const REFRESH_INTERVAL_MS: i64 = 60 * 1000;
const STORE_FORMAT_VERSION: u8 = 1;
const HEADER_BYTES: usize = 13;
const POST_BYTES: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostViews {
    pub post_id: u64,
    pub author_id: u64,
    pub age_hours: f64,
    pub views: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectedPost {
    pub post_id: u64,
    pub author_id: u64,
    pub quality: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionConfig {
    pub per_author: usize,
    pub budget: usize,
    pub half_life_hours: f64,
    pub floor_hours: f64,
    pub window_hours: f64,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            per_author: 5,
            budget: 500,
            half_life_hours: 8.0,
            floor_hours: 0.5,
            window_hours: 24.0,
        }
    }
}

pub fn quality(views: u64, age_hours: f64, config: &SelectionConfig) -> f64 {
    let lambda = std::f64::consts::LN_2 / config.half_life_hours;
    let saturation = 1.0 - (-lambda * config.window_hours).exp();
    let elapsed = 1.0 - (-lambda * age_hours.max(config.floor_hours)).exp();
    views as f64 * saturation / elapsed
}

fn by_quality_desc(a: &SelectedPost, b: &SelectedPost) -> std::cmp::Ordering {
    b.quality
        .total_cmp(&a.quality)
        .then(b.post_id.cmp(&a.post_id))
}

pub fn select_popular_posts(posts: &[PostViews], config: &SelectionConfig) -> Vec<SelectedPost> {
    let mut seen = HashSet::new();
    let mut by_author: HashMap<u64, Vec<SelectedPost>> = HashMap::new();
    for p in posts {
        if p.age_hours < 0.0 || p.age_hours > config.window_hours || !seen.insert(p.post_id) {
            continue;
        }
        by_author
            .entry(p.author_id)
            .or_default()
            .push(SelectedPost {
                post_id: p.post_id,
                author_id: p.author_id,
                quality: quality(p.views, p.age_hours, config),
            });
    }
    let mut pool: Vec<SelectedPost> = Vec::new();
    for mut author_posts in by_author.into_values() {
        author_posts.sort_by(by_quality_desc);
        author_posts.truncate(config.per_author);
        pool.extend(author_posts);
    }
    pool.sort_by(by_quality_desc);
    pool.truncate(config.budget);
    pool
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoredPopularPosts {
    pub generated_at_ms: i64,
    pub posts: Vec<SelectedPost>,
}

pub fn encode_stored(stored: &StoredPopularPosts) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES + POST_BYTES * stored.posts.len());
    out.push(STORE_FORMAT_VERSION);
    out.extend_from_slice(&stored.generated_at_ms.to_be_bytes());
    out.extend_from_slice(&(stored.posts.len() as u32).to_be_bytes());
    for p in &stored.posts {
        out.extend_from_slice(&p.post_id.to_be_bytes());
        out.extend_from_slice(&p.author_id.to_be_bytes());
        out.extend_from_slice(&p.quality.to_be_bytes());
    }
    out
}

pub fn decode_stored(bytes: &[u8]) -> Result<StoredPopularPosts, String> {
    let header = bytes
        .get(..HEADER_BYTES)
        .ok_or_else(|| format!("popular posts value too short: {} bytes", bytes.len()))?;
    if header[0] != STORE_FORMAT_VERSION {
        return Err(format!("unknown popular posts format {}", header[0]));
    }
    let generated_at_ms = i64::from_be_bytes(header[1..9].try_into().unwrap());
    let count = u32::from_be_bytes(header[9..13].try_into().unwrap()) as usize;
    let body = &bytes[HEADER_BYTES..];
    if body.len() != count * POST_BYTES {
        return Err(format!(
            "popular posts body is {} bytes, expected {}",
            body.len(),
            count * POST_BYTES
        ));
    }
    let posts = body
        .chunks_exact(POST_BYTES)
        .map(|c| SelectedPost {
            post_id: u64::from_be_bytes(c[..8].try_into().unwrap()),
            author_id: u64::from_be_bytes(c[8..16].try_into().unwrap()),
            quality: f64::from_be_bytes(c[16..].try_into().unwrap()),
        })
        .collect();
    Ok(StoredPopularPosts {
        generated_at_ms,
        posts,
    })
}

#[async_trait]
pub trait PopularPostsStore: Send + Sync {
    async fn load(&self) -> Result<Option<StoredPopularPosts>, String>;
    async fn save(&self, stored: &StoredPopularPosts) -> Result<(), String>;
}

#[derive(Default)]
pub struct InMemoryPopularPostsStore {
    value: Mutex<Option<StoredPopularPosts>>,
}

#[async_trait]
impl PopularPostsStore for InMemoryPopularPostsStore {
    async fn load(&self) -> Result<Option<StoredPopularPosts>, String> {
        Ok(self.value.lock().unwrap().clone())
    }

    async fn save(&self, stored: &StoredPopularPosts) -> Result<(), String> {
        *self.value.lock().unwrap() = Some(stored.clone());
        Ok(())
    }
}

#[derive(Default)]
struct Snapshot {
    stored: StoredPopularPosts,
    loaded_at_ms: Option<i64>,
}

pub struct PopularPostsCache {
    store: Arc<dyn PopularPostsStore>,
    snapshot: RwLock<Snapshot>,
    refreshing: AtomicBool,
}

impl PopularPostsCache {
    pub fn new(store: Arc<dyn PopularPostsStore>) -> Self {
        Self {
            store,
            snapshot: RwLock::new(Snapshot::default()),
            refreshing: AtomicBool::new(false),
        }
    }

    pub fn top_posts(&self, limit: usize) -> (i64, Vec<SelectedPost>) {
        let snapshot = self.snapshot.read().unwrap();
        (
            snapshot.stored.generated_at_ms,
            snapshot.stored.posts.iter().take(limit).copied().collect(),
        )
    }

    fn refresh_due(&self, now_ms: i64) -> bool {
        self.snapshot
            .read()
            .unwrap()
            .loaded_at_ms
            .is_none_or(|t| now_ms - t >= REFRESH_INTERVAL_MS)
    }

    async fn refresh(&self, now_ms: i64) -> Result<StoredPopularPosts, String> {
        let stored = self.store.load().await?.unwrap_or_default();
        *self.snapshot.write().unwrap() = Snapshot {
            stored: stored.clone(),
            loaded_at_ms: Some(now_ms),
        };
        Ok(stored)
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
                    Ok(stored) => {
                        receiver.incr("PopularPostsCache.Refresh", &[("result", "ok")], 1);
                        receiver.gauge(
                            "PopularPostsCache.LoadedPosts",
                            &[],
                            stored.posts.len() as f64,
                        );
                        if stored.generated_at_ms > 0 {
                            receiver.gauge(
                                "PopularPostsCache.ListAgeSecs",
                                &[],
                                (now_ms - stored.generated_at_ms) as f64 / 1000.0,
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "popular posts refresh failed");
                        receiver.incr("PopularPostsCache.Refresh", &[("result", "error")], 1);
                    }
                }
            }
            cache.refreshing.store(false, Ordering::Release);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn post(post_id: u64, author_id: u64, age_hours: f64, views: u64) -> PostViews {
        PostViews {
            post_id,
            author_id,
            age_hours,
            views,
        }
    }

    #[test]
    fn select_caps_per_author_and_budget_by_projected_views() {
        let config = SelectionConfig {
            per_author: 2,
            budget: 3,
            ..SelectionConfig::default()
        };
        let posts = [
            post(1, 10, 24.0, 1_000),
            post(2, 10, 0.25, 100),
            post(3, 10, 12.0, 900),
            post(4, 20, 6.0, 500),
            post(5, 20, 30.0, 1_000_000),
            post(6, 30, 2.0, 50),
        ];
        let selected = select_popular_posts(&posts, &config);
        let ids: Vec<u64> = selected.iter().map(|p| p.post_id).collect();
        assert_eq!(ids, vec![2, 3, 4]);
        assert!((selected[0].quality - 100.0 * 0.875 / 0.042_396_719).abs() < 1e-2);
        assert!((selected[1].quality - 900.0 * 0.875 / 0.646_446_609).abs() < 1e-3);
    }
}
