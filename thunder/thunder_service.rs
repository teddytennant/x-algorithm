use lazy_static::lazy_static;
use log::{debug, error, info, warn};
use std::cmp::Reverse;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tonic::{Request, Response, Status};

use governor::{clock::DefaultClock, state::InMemoryState, state::NotKeyed, Quota, RateLimiter};

use xai_thunder_proto::{
    in_network_posts_service_server::{InNetworkPostsService, InNetworkPostsServiceServer},
    GetInNetworkPostsRequest, GetInNetworkPostsResponse, LightPost,
};

use crate::config::{
    MAX_INPUT_LIST_SIZE, MAX_ORIGINAL_POSTS_PER_AUTHOR, MAX_POSTS_TO_RETURN,
    MAX_REPLY_POSTS_PER_AUTHOR, MAX_VIDEOS_TO_RETURN,
};
use crate::metrics::{
    Timer, GET_IN_NETWORK_POSTS_COUNT, GET_IN_NETWORK_POSTS_DURATION,
    GET_IN_NETWORK_POSTS_DURATION_WITHOUT_STRATO, GET_IN_NETWORK_POSTS_EXCLUDED_SIZE,
    GET_IN_NETWORK_POSTS_FOLLOWING_SIZE, GET_IN_NETWORK_POSTS_FOUND_FRESHNESS_SECONDS,
    GET_IN_NETWORK_POSTS_FOUND_POSTS_PER_AUTHOR, GET_IN_NETWORK_POSTS_FOUND_REPLY_RATIO,
    GET_IN_NETWORK_POSTS_FOUND_TIME_RANGE_SECONDS, GET_IN_NETWORK_POSTS_FOUND_UNIQUE_AUTHORS,
    GET_IN_NETWORK_POSTS_MAX_RESULTS, IN_FLIGHT_REQUESTS, REJECTED_REQUESTS,
};
use crate::posts::post_store::PostStore;
use crate::strato_client::StratoClient;

lazy_static! {
        static ref BLACKLISTED_USER_IDS: HashSet<i64> = {
        let mut set = HashSet::new();
        set.insert(1720665183188922368); // @grok bot account
        set.insert(1909390269533011968); // @gork bot account
        set.insert(1999031187604717568); // @products experimentation account
        set
    };
}

pub struct ThunderServiceImpl {
    post_store: Arc<PostStore>,
    strato_client: Arc<StratoClient>,
    rate_limiter: Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>,
}

impl ThunderServiceImpl {
    pub fn new(
        post_store: Arc<PostStore>,
        strato_client: Arc<StratoClient>,
        qps_limit: u32,
    ) -> Self {
        info!("Initializing ThunderService with qps_limit={}", qps_limit);
        let quota = Quota::per_second(NonZeroU32::new(qps_limit).expect("qps_limit must be > 0"));
        Self {
            post_store,
            strato_client,
            rate_limiter: Arc::new(RateLimiter::direct(quota)),
        }
    }

    pub fn server(self) -> InNetworkPostsServiceServer<Self> {
        InNetworkPostsServiceServer::new(self)
            .accept_compressed(tonic::codec::CompressionEncoding::Zstd)
            .send_compressed(tonic::codec::CompressionEncoding::Zstd)
    }

    fn analyze_and_report_post_statistics(posts: &[LightPost], stage: &str) {
        if posts.is_empty() {
            debug!("[{}] No posts found for analysis", stage);
            return;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let time_since_most_recent = posts
            .iter()
            .map(|post| post.created_at)
            .max()
            .map(|most_recent| now - most_recent);

        let time_since_oldest = posts
            .iter()
            .map(|post| post.created_at)
            .min()
            .map(|oldest| now - oldest);

        let reply_count = posts.iter().filter(|post| post.is_reply).count();
        let original_count = posts.len() - reply_count;

        let unique_authors: HashSet<_> = posts.iter().map(|post| post.author_id).collect();
        let unique_author_count = unique_authors.len();

        if let Some(freshness) = time_since_most_recent {
            GET_IN_NETWORK_POSTS_FOUND_FRESHNESS_SECONDS
                .with_label_values(&[stage])
                .observe(freshness as f64);
        }

        if let (Some(oldest), Some(newest)) = (time_since_oldest, time_since_most_recent) {
            let time_range = oldest - newest;
            GET_IN_NETWORK_POSTS_FOUND_TIME_RANGE_SECONDS
                .with_label_values(&[stage])
                .observe(time_range as f64);
        }

        let reply_ratio = reply_count as f64 / posts.len() as f64;
        GET_IN_NETWORK_POSTS_FOUND_REPLY_RATIO
            .with_label_values(&[stage])
            .observe(reply_ratio);

        GET_IN_NETWORK_POSTS_FOUND_UNIQUE_AUTHORS
            .with_label_values(&[stage])
            .observe(unique_author_count as f64);

        if unique_author_count > 0 {
            let posts_per_author = posts.len() as f64 / unique_author_count as f64;
            GET_IN_NETWORK_POSTS_FOUND_POSTS_PER_AUTHOR
                .with_label_values(&[stage])
                .observe(posts_per_author);
        }

        debug!(
            "[{}] Post statistics: total={}, original={}, replies={}, unique_authors={}, posts_per_author={:.2}, reply_ratio={:.2}, time_since_most_recent={:?}s, time_range={:?}s",
            stage,
            posts.len(),
            original_count,
            reply_count,
            unique_author_count,
            if unique_author_count > 0 {
                posts.len() as f64 / unique_author_count as f64
            } else {
                0.0
            },
            reply_ratio,
            time_since_most_recent,
            if let (Some(o), Some(n)) = (time_since_oldest, time_since_most_recent) {
                Some(o - n)
            } else {
                None
            }
        );
    }
}

#[tonic::async_trait]
impl InNetworkPostsService for ThunderServiceImpl {
    async fn get_in_network_posts(
        &self,
        request: Request<GetInNetworkPostsRequest>,
    ) -> Result<Response<GetInNetworkPostsResponse>, Status> {
        if self.rate_limiter.check().is_err() {
            REJECTED_REQUESTS.inc();
            return Err(Status::resource_exhausted(
                "QPS rate limit exceeded, please retry",
            ));
        }

        IN_FLIGHT_REQUESTS.inc();

        struct InFlightGuard;
        impl Drop for InFlightGuard {
            fn drop(&mut self) {
                IN_FLIGHT_REQUESTS.dec();
            }
        }
        let _in_flight_guard = InFlightGuard;

        let _total_timer = Timer::new(GET_IN_NETWORK_POSTS_DURATION.clone());

        let req = request.into_inner();

        if req.debug {
            info!(
                "Received GetInNetworkPosts request: user_id={}, following_count={}, exclude_tweet_ids={}, algorithm={}",
                req.user_id,
                req.following_user_ids.len(),
                req.exclude_tweet_ids.len(),
                req.algorithm
            );
        }

        let following_user_ids = if req.following_user_ids.is_empty() && req.debug {
            info!(
                "Following list is empty, fetching from Strato for user {}",
                req.user_id
            );

            match self
                .strato_client
                .fetch_following_list(req.user_id as i64, MAX_INPUT_LIST_SIZE as i32)
                .await
            {
                Ok(following_list) => {
                    info!(
                        "Fetched {} following users from Strato for user {}",
                        following_list.len(),
                        req.user_id
                    );
                    following_list.into_iter().map(|id| id as u64).collect()
                }
                Err(e) => {
                    warn!(
                        "Failed to fetch following list from Strato for user {}: {}",
                        req.user_id, e
                    );
                    return Err(Status::internal(format!(
                        "Failed to fetch following list: {}",
                        e
                    )));
                }
            }
        } else {
            req.following_user_ids
        };

        GET_IN_NETWORK_POSTS_FOLLOWING_SIZE.observe(following_user_ids.len() as f64);
        GET_IN_NETWORK_POSTS_EXCLUDED_SIZE.observe(req.exclude_tweet_ids.len() as f64);

        let _processing_timer = Timer::new(GET_IN_NETWORK_POSTS_DURATION_WITHOUT_STRATO.clone());

        let max_results = if req.max_results > 0 {
            req.max_results as usize
        } else if req.is_video_request {
            MAX_VIDEOS_TO_RETURN
        } else {
            MAX_POSTS_TO_RETURN
        };
        GET_IN_NETWORK_POSTS_MAX_RESULTS.observe(max_results as f64);

        let limits = req.per_author_limits.unwrap_or_default();
        let max_original_per_author = limits
            .max_posts_per_author
            .map_or(MAX_ORIGINAL_POSTS_PER_AUTHOR, |n| {
                (n as usize).min(MAX_ORIGINAL_POSTS_PER_AUTHOR)
            });
        let max_secondary_per_author = limits
            .max_replies_reposts_per_author
            .map_or(MAX_REPLY_POSTS_PER_AUTHOR, |n| {
                (n as usize).min(MAX_REPLY_POSTS_PER_AUTHOR)
            });

        let following_count = following_user_ids.len();
        if following_count > MAX_INPUT_LIST_SIZE {
            warn!(
                "Limiting following_user_ids from {} to {} entries for user {}",
                following_count, MAX_INPUT_LIST_SIZE, req.user_id
            );
        }
        let following_user_ids: Vec<u64> = following_user_ids
            .into_iter()
            .take(MAX_INPUT_LIST_SIZE)
            .collect();

        let exclude_count = req.exclude_tweet_ids.len();
        if exclude_count > MAX_INPUT_LIST_SIZE {
            warn!(
                "Limiting exclude_tweet_ids from {} to {} entries for user {}",
                exclude_count, MAX_INPUT_LIST_SIZE, req.user_id
            );
        }
        let exclude_tweet_ids: Vec<u64> = req
            .exclude_tweet_ids
            .into_iter()
            .take(MAX_INPUT_LIST_SIZE)
            .collect();

        let post_store = Arc::clone(&self.post_store);
        let request_user_id = req.user_id as i64;

        let proto_posts = tokio::task::spawn_blocking(move || {
            let following_user_ids: Vec<i64> = following_user_ids
                .iter()
                .map(|&id| id as i64)
                .filter(|id| !BLACKLISTED_USER_IDS.contains(id))
                .collect();

            let exclude_tweet_ids: HashSet<i64> =
                exclude_tweet_ids.iter().map(|&id| id as i64).collect();

            let start_time = Instant::now();

            let all_posts: Vec<LightPost> = if req.is_video_request {
                post_store.get_videos_by_users(
                    &following_user_ids,
                    &exclude_tweet_ids,
                    start_time,
                    request_user_id,
                )
            } else {
                post_store.get_all_posts_by_users(
                    &following_user_ids,
                    &exclude_tweet_ids,
                    start_time,
                    request_user_id,
                    max_original_per_author,
                    max_secondary_per_author,
                )
            };

            ThunderServiceImpl::analyze_and_report_post_statistics(&all_posts, "retrieved");

            let algorithm = req.algorithm.as_str();
            if !matches!(algorithm, "" | "recent" | "default" | "latest") {
                error!(
                    "Unsupported algorithm '{}' for user {}, falling back to 'recent'",
                    algorithm, request_user_id
                );
            }
            let scored_posts = score_recent(all_posts, max_results);

            ThunderServiceImpl::analyze_and_report_post_statistics(&scored_posts, "scored");

            scored_posts
        })
        .await
        .map_err(|e| Status::internal(format!("Failed to process posts: {}", e)))?;

        if req.debug {
            info!(
                "Returning {} posts for user {}",
                proto_posts.len(),
                req.user_id
            );
        }

        GET_IN_NETWORK_POSTS_COUNT.observe(proto_posts.len() as f64);

        let response = GetInNetworkPostsResponse { posts: proto_posts };

        Ok(Response::new(response))
    }
}

fn score_recent(mut light_posts: Vec<LightPost>, max_results: usize) -> Vec<LightPost> {
    light_posts.sort_unstable_by_key(|post| Reverse(post.created_at));

    light_posts.into_iter().take(max_results).collect()
}
