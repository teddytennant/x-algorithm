use crate::models::candidate::PostCandidate;
use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::{EnableVideoCarousel, VIDEO_CAROUSEL_MAX_VIDEOS, VIDEO_CAROUSEL_MIN_VIDEOS};

const MAX_VERTICAL_ASPECT_RATIO: f32 = 1.0;

pub(crate) fn is_enabled(query: &ScoredPostsQuery) -> bool {
    query.request_type == RequestType::ForYou
        && query.params.get(EnableVideoCarousel)
        && query.video_carousel_eligible
}

pub(crate) fn select_videos<'a>(
    candidates: impl IntoIterator<Item = &'a PostCandidate>,
) -> Vec<PostCandidate> {
    let mut videos: Vec<&PostCandidate> = candidates
        .into_iter()
        .filter(|candidate| is_carousel_video(candidate) && is_vertical(candidate))
        .collect();
    videos.sort_by(|a, b| video_open_score(b).total_cmp(&video_open_score(a)));
    videos.truncate(VIDEO_CAROUSEL_MAX_VIDEOS);
    if videos.len() < VIDEO_CAROUSEL_MIN_VIDEOS {
        return Vec::new();
    }
    videos.into_iter().cloned().collect()
}

pub(crate) fn take_videos_by_video_open(
    candidates: &mut Vec<PostCandidate>,
    count: usize,
) -> Vec<PostCandidate> {
    let (mut videos, rest): (Vec<_>, Vec<_>) = std::mem::take(candidates)
        .into_iter()
        .partition(is_carousel_video);
    videos.sort_by(|a, b| video_open_score(b).total_cmp(&video_open_score(a)));
    let overflow = videos.split_off(count.min(videos.len()));
    *candidates = rest;
    candidates.extend(overflow);
    for video in &mut videos {
        video.video_carousel_extra = true;
    }
    videos
}

fn is_carousel_video(candidate: &PostCandidate) -> bool {
    candidate.has_video == Some(true)
        && candidate.in_reply_to_tweet_id.is_none()
        && candidate.phoenix_scores.video_open_score.is_some()
}

fn is_vertical(candidate: &PostCandidate) -> bool {
    candidate
        .video_aspect_ratio
        .is_some_and(|ratio| ratio < MAX_VERTICAL_ASPECT_RATIO)
}

fn video_open_score(candidate: &PostCandidate) -> f64 {
    candidate.phoenix_scores.video_open_score.unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::PhoenixScores;

    fn video(tweet_id: u64, aspect_ratio: f32, video_open: f64) -> PostCandidate {
        PostCandidate {
            tweet_id,
            has_video: Some(true),
            video_aspect_ratio: Some(aspect_ratio),
            phoenix_scores: PhoenixScores {
                video_open_score: Some(video_open),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn tweet_ids(candidates: &[PostCandidate]) -> Vec<u64> {
        candidates
            .iter()
            .map(|candidate| candidate.tweet_id)
            .collect()
    }

    #[test]
    fn keeps_vertical_videos_ranked_by_video_open() {
        let candidates = vec![
            video(1, 0.5625, 0.1),
            video(2, 1.7778, 0.9),
            video(3, 1.0, 0.8),
            video(4, 0.5625, 0.5),
            video(5, 0.75, 0.3),
            PostCandidate {
                in_reply_to_tweet_id: Some(9),
                ..video(6, 0.5625, 0.7)
            },
        ];

        assert_eq!(tweet_ids(&select_videos(&candidates)), vec![4, 5, 1]);
    }
}
