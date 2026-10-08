use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use xai_candidate_pipeline::component_library::clients::{
    ProdThunderCapiClient, ThunderCapiClient,
};
use xai_thunder_proto::{GetInNetworkPostsRequest, LightPost, PerAuthorLimits};

use crate::clients::engagement_counts_client::{
    EngagementCountsClient, ProdEngagementCountsClient,
};
use crate::clients::popular_authors_store_client::{
    ManhattanPopularAuthorsStore, ManhattanPopularPostsStore,
};
use crate::util::popular_authors::{now_ms, PopularAuthorsStore};
use crate::util::popular_posts::{
    select_popular_posts, PopularPostsStore, PostViews, SelectionConfig, StoredPopularPosts,
};

const METRIC_PREFIX: &str = "PopularPostsJob";
const AUTHORS_PER_THUNDER_REQUEST: usize = 20;
const THUNDER_POSTS_PER_AUTHOR: u32 = 50;
const THUNDER_MAX_RESULTS: u32 = 1200;
const VIEW_COUNT_BATCH: usize = 500;
const THUNDER_ATTEMPTS: usize = 5;
const THUNDER_RETRY_DELAY: Duration = Duration::from_secs(1);

pub struct JobConfig {
    pub datacenter: String,
    pub interval: Duration,
    pub selection: SelectionConfig,
    pub once: bool,
}

#[derive(Debug, Default)]
pub struct RunStats {
    pub authors: usize,
    pub thunder_posts: usize,
    pub candidate_posts: usize,
    pub posts_with_views: usize,
    pub selected_posts: usize,
    pub selected_authors: usize,
    pub generated_at_ms: i64,
}

struct Job {
    authors: ManhattanPopularAuthorsStore,
    posts: ManhattanPopularPostsStore,
    thunder: Arc<dyn ThunderCapiClient + Send + Sync>,
    views: Arc<dyn EngagementCountsClient>,
    selection: SelectionConfig,
}

fn gauge(name: &str, value: f64) {
    if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
        receiver.gauge(&format!("{METRIC_PREFIX}.{name}"), &[], value);
    }
}

fn count_run(result: &str) {
    if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
        receiver.incr(&format!("{METRIC_PREFIX}.Runs"), &[("result", result)], 1);
    }
}

fn is_original(post: &LightPost) -> bool {
    !post.is_reply && !post.is_retweet
}

impl Job {
    async fn fetch_posts(&self, author_ids: &[u64]) -> Result<Vec<LightPost>, String> {
        let mut posts = Vec::new();
        for chunk in author_ids.chunks(AUTHORS_PER_THUNDER_REQUEST) {
            let request = GetInNetworkPostsRequest {
                user_id: 0,
                following_user_ids: chunk.to_vec(),
                max_results: THUNDER_MAX_RESULTS,
                exclude_tweet_ids: Vec::new(),
                algorithm: String::new(),
                debug: false,
                is_video_request: false,
                per_author_limits: Some(PerAuthorLimits {
                    max_posts_per_author: Some(THUNDER_POSTS_PER_AUTHOR),
                    max_replies_reposts_per_author: Some(0),
                }),
            };
            let mut last_error = String::new();
            let mut fetched = None;
            for _ in 0..THUNDER_ATTEMPTS {
                match self.thunder.get_in_network_posts(request.clone()).await {
                    Ok(response) => {
                        fetched = Some(response.posts);
                        break;
                    }
                    Err(e) => {
                        last_error = e.to_string();
                        tokio::time::sleep(THUNDER_RETRY_DELAY).await;
                    }
                }
            }
            let chunk_posts = fetched
                .ok_or_else(|| format!("thunder failed for an author chunk: {last_error}"))?;
            posts.extend(chunk_posts.into_iter().filter(is_original));
        }
        Ok(posts)
    }

    async fn fetch_views(
        &self,
        posts: &[LightPost],
        now_ms: i64,
    ) -> Result<Vec<PostViews>, String> {
        let window_ms = (self.selection.window_hours * 3_600_000.0) as i64;
        let mut seen = HashSet::new();
        let recent: Vec<&LightPost> = posts
            .iter()
            .filter(|p| now_ms - p.created_at * 1000 <= window_ms && seen.insert(p.post_id))
            .collect();
        let mut out = Vec::with_capacity(recent.len());
        for chunk in recent.chunks(VIEW_COUNT_BATCH) {
            let ids: Vec<u64> = chunk.iter().map(|p| p.post_id as u64).collect();
            let counts = self.views.get_engagement_counts(&ids).await?;
            out.extend(chunk.iter().map(|p| PostViews {
                post_id: p.post_id as u64,
                author_id: p.author_id as u64,
                age_hours: (now_ms - p.created_at * 1000) as f64 / 3_600_000.0,
                views: counts.get(&(p.post_id as u64)).map_or(0, |c| c.view_count),
            }));
        }
        Ok(out)
    }

    async fn run_once(&self) -> Result<RunStats, String> {
        let authors = self
            .authors
            .load()
            .await?
            .map(|s| s.authors)
            .unwrap_or_default();
        if authors.is_empty() {
            return Err("popular posts job author list is empty".to_string());
        }
        let author_ids: Vec<u64> = authors.iter().map(|a| a.author_id).collect();
        let posts = self.fetch_posts(&author_ids).await?;
        let now = now_ms();
        let candidates = self.fetch_views(&posts, now).await?;
        let selected = select_popular_posts(&candidates, &self.selection);
        if selected.is_empty() {
            return Err("popular posts job selected no posts".to_string());
        }
        let stored = StoredPopularPosts {
            generated_at_ms: now,
            posts: selected,
        };
        self.posts.save(&stored).await?;
        let read_back = self
            .posts
            .load()
            .await?
            .ok_or_else(|| "popular posts list missing after write".to_string())?;
        if read_back.generated_at_ms != stored.generated_at_ms {
            return Err(format!(
                "popular posts read back generated_at {} != written {}",
                read_back.generated_at_ms, stored.generated_at_ms
            ));
        }
        let selected_authors: HashSet<u64> = stored.posts.iter().map(|p| p.author_id).collect();
        Ok(RunStats {
            authors: author_ids.len(),
            thunder_posts: posts.len(),
            candidate_posts: candidates.len(),
            posts_with_views: candidates.iter().filter(|p| p.views > 0).count(),
            selected_posts: stored.posts.len(),
            selected_authors: selected_authors.len(),
            generated_at_ms: stored.generated_at_ms,
        })
    }
}

pub async fn run(config: JobConfig) -> anyhow::Result<()> {
    let job = Job {
        authors: ManhattanPopularAuthorsStore::new(&config.datacenter).await?,
        posts: ManhattanPopularPostsStore::new(&config.datacenter).await?,
        thunder: Arc::new(
            ProdThunderCapiClient::new(&config.datacenter)
                .await
                .map_err(|e| anyhow::anyhow!("thunder client: {e}"))?,
        ),
        views: Arc::new(ProdEngagementCountsClient::new(&config.datacenter).await?),
        selection: config.selection,
    };
    tracing::info!(selection = ?job.selection, interval_secs = config.interval.as_secs(), "popular posts job started");
    loop {
        let started = Instant::now();
        match job.run_once().await {
            Ok(stats) => {
                count_run("ok");
                gauge("LastSuccessUnixSecs", stats.generated_at_ms as f64 / 1000.0);
                gauge("Authors", stats.authors as f64);
                gauge("ThunderPosts", stats.thunder_posts as f64);
                gauge("CandidatePosts", stats.candidate_posts as f64);
                gauge("PostsWithViews", stats.posts_with_views as f64);
                gauge("SelectedPosts", stats.selected_posts as f64);
                gauge("SelectedAuthors", stats.selected_authors as f64);
                gauge("RunSecs", started.elapsed().as_secs_f64());
                tracing::info!(
                    ?stats,
                    run_secs = started.elapsed().as_secs_f64(),
                    "popular posts job run ok"
                );
            }
            Err(e) => {
                count_run("error");
                tracing::error!(error = %e, "popular posts job run failed");
                if config.once {
                    anyhow::bail!(e);
                }
            }
        }
        if config.once {
            return Ok(());
        }
        tokio::time::sleep(config.interval.saturating_sub(started.elapsed())).await;
    }
}
