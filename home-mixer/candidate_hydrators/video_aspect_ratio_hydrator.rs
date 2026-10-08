use crate::clients::tweet_entity_service_client::TESClient;
use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::util::video_carousel;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{default_quick_cache, QuickCache};
use xai_candidate_pipeline::hydrator::{CacheStore, CachedHydrator};
use xai_core_entities::entities::MediaEntity;
use xai_x_thrift::tweet_media::MediaInfo;

pub struct VideoAspectRatioHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub cache: QuickCache<u64, Option<f32>>,
}

impl VideoAspectRatioHydrator {
    pub fn new(tes_client: Arc<dyn TESClient + Send + Sync>) -> Self {
        Self {
            tes_client,
            cache: default_quick_cache(),
        }
    }
}

#[async_trait]
impl CachedHydrator<ScoredPostsQuery, PostCandidate> for VideoAspectRatioHydrator {
    type CacheKey = u64;

    type CacheValue = Option<f32>;

    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query)
    }

    fn cache_store(&self) -> &dyn CacheStore<Self::CacheKey, Self::CacheValue> {
        &self.cache
    }

    fn cache_key(&self, candidate: &PostCandidate) -> Self::CacheKey {
        candidate.get_original_tweet_id()
    }

    fn cache_value(&self, hydrated: &PostCandidate) -> Self::CacheValue {
        hydrated.video_aspect_ratio
    }

    fn hydrate_from_cache(&self, value: Self::CacheValue) -> PostCandidate {
        PostCandidate {
            video_aspect_ratio: value,
            ..Default::default()
        }
    }

    fn already_hydrated(&self, candidate: &PostCandidate) -> bool {
        candidate.has_video != Some(true) || candidate.video_aspect_ratio.is_some()
    }

    async fn hydrate_from_client(
        &self,
        _query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let tweet_ids: Vec<u64> = candidates
            .iter()
            .map(|c| c.get_original_tweet_id())
            .collect();

        let media_entities = self
            .tes_client
            .get_tweet_media_entities(tweet_ids.clone())
            .await;

        tweet_ids
            .iter()
            .map(|tweet_id| match media_entities.get(tweet_id) {
                Some(Ok(entities)) => Ok(PostCandidate {
                    video_aspect_ratio: entities
                        .as_ref()
                        .and_then(|entities| entities.first())
                        .and_then(video_aspect_ratio),
                    ..Default::default()
                }),
                Some(Err(err)) => Err(err.to_string()),
                None => Ok(PostCandidate::default()),
            })
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.video_aspect_ratio = hydrated.video_aspect_ratio;
    }
}

fn video_aspect_ratio(entity: &MediaEntity) -> Option<f32> {
    let Some(MediaInfo::VideoInfo(info)) = &entity.media_info else {
        return None;
    };
    let ratio = info.aspect_ratio.as_ref()?;
    let aspect_ratio = f32::from(ratio.numerator?) / f32::from(ratio.denominator?);
    (aspect_ratio.is_finite() && aspect_ratio > 0.0).then_some(aspect_ratio)
}
