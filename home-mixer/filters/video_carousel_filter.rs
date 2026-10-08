use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::RESULT_SIZE;
use crate::util::video_carousel;
use xai_candidate_pipeline::filter::{Filter, FilterResult};

pub struct VideoCarouselFilter;

impl Filter<ScoredPostsQuery, PostCandidate> for VideoCarouselFilter {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        video_carousel::is_enabled(query)
    }

    fn filter(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<PostCandidate>,
    ) -> FilterResult<PostCandidate> {
        let (removed, kept): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|candidate| candidate.video_carousel_extra);
        let _ = query.video_carousel.set(video_carousel::select_videos(
            kept.iter().skip(RESULT_SIZE).chain(&removed),
        ));
        FilterResult { kept, removed }
    }
}
