use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::util::tweet_type_metrics::*;
use crate::util::viewer_history;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::{Composition, duration_since_creation_opt};
use xai_candidate_pipeline::hydrator::Hydrator;

const THIRTY_MINUTES_MS: u64 = 30 * 60 * 1000;
const ONE_HOUR_MS: u64 = 60 * 60 * 1000;
const SIX_HOURS_MS: u64 = 6 * 60 * 60 * 1000;
const TWELVE_HOURS_MS: u64 = 12 * 60 * 60 * 1000;
const TWENTY_FOUR_HOURS_MS: u64 = 24 * 60 * 60 * 1000;

pub struct TweetTypeMetricsHydrator;

impl TweetTypeMetricsHydrator {
    pub fn new() -> Self {
        Self
    }

        pub fn create_tweet_type_bitset(
        candidate: &PostCandidate,
        query: &ScoredPostsQuery,
    ) -> HashSet<usize> {
        let mut true_tweet_types = HashSet::new();

        true_tweet_types.insert(ANY_CANDIDATE);

        if candidate.retweeted_tweet_id.is_some() {
            true_tweet_types.insert(RETWEET);
        }

        if candidate.in_reply_to_tweet_id.is_some() {
            true_tweet_types.insert(REPLY);
        }

        if candidate.subscription_author_id.is_some() {
            true_tweet_types.insert(SUBSCRIPTION_POST);
        }

        if let Some(score) = candidate.score
            && score != 0.0
        {
            true_tweet_types.insert(FULL_SCORING_SUCCEEDED);
        }

        if !candidate.ancestors.is_empty() {
            true_tweet_types.insert(HAS_ANCESTORS);
        }

        if candidate.in_network.unwrap_or(true) {
            true_tweet_types.insert(IN_NETWORK);
        }

        if let Some(followers) = candidate.author_followers_count {
            let followers_u32 = followers as u32;
            if followers_u32 < 100 {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_0_100);
            }
            if (100..1000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_100_1K);
            }
            if (1000..10000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_1K_10K);
            }
            if (10000..100000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_10K_100K);
            }
            if (100000..1000000).contains(&followers_u32) {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_100K_1M);
            }
            if followers_u32 >= 1000000 {
                true_tweet_types.insert(AUTHOR_FOLLOWERS_1M_PLUS);
            }
        }

        if candidate.min_video_duration_ms.is_some() {
            true_tweet_types.insert(VIDEO);
        }

        if let Some(duration_ms) = candidate.min_video_duration_ms {
            let duration_ms_u32 = duration_ms as u32;
            if duration_ms_u32 <= 10000 {
                true_tweet_types.insert(VIDEO_LTE_10_SEC);
            }
            if duration_ms_u32 > 10000 && duration_ms_u32 <= 60000 {
                true_tweet_types.insert(VIDEO_BT_10_60_SEC);
            }
            if duration_ms_u32 > 60000 {
                true_tweet_types.insert(VIDEO_GT_60_SEC);
            }
        }

        if let Some(age) = duration_since_creation_opt(candidate.tweet_id) {
            let age_ms = age.as_millis() as u64;

            if age_ms <= THIRTY_MINUTES_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_30_MINUTES);
            }
            if age_ms <= ONE_HOUR_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_1_HOUR);
            }
            if age_ms <= SIX_HOURS_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_6_HOURS);
            }
            if age_ms <= TWELVE_HOURS_MS {
                true_tweet_types.insert(TWEET_AGE_LTE_12_HOURS);
            }
            if age_ms >= TWENTY_FOUR_HOURS_MS {
                true_tweet_types.insert(TWEET_AGE_GTE_24_HOURS);
            }
        }

        let served_size = query.served_ids.len();
        if served_size == 0 {
            true_tweet_types.insert(EMPTY_REQUEST);
        }
        if served_size < 3 {
            true_tweet_types.insert(NEAR_EMPTY);
        }
        if served_size < 20 {
            true_tweet_types.insert(SERVED_SIZE_LESS_THAN_20);
        }
        if served_size < 10 {
            true_tweet_types.insert(SERVED_SIZE_LESS_THAN_10);
        }
        if served_size < 5 {
            true_tweet_types.insert(SERVED_SIZE_LESS_THAN_5);
        }

        true_tweet_types
    }

    pub fn author_diversity_bits(authors: &Composition) -> HashSet<usize> {
        let mut bits = HashSet::new();
        if authors.size == 0 {
            return bits;
        }
        if authors.unique_ratio() <= 0.5 {
            bits.insert(UNIQUE_AUTHOR_RATIO_LTE_50_PCT);
        }
        if authors.unique <= 5 {
            bits.insert(UNIQUE_AUTHOR_LTE_5);
        }
        if authors.unique <= 10 {
            bits.insert(UNIQUE_AUTHOR_LTE_10);
        }
        if authors.unique <= 15 {
            bits.insert(UNIQUE_AUTHOR_LTE_15);
        }
        if authors.max_share() >= 0.25 {
            bits.insert(SINGLE_AUTHOR_GTE_25_PCT);
        }
        if authors.max_share() >= 0.5 {
            bits.insert(SINGLE_AUTHOR_GTE_50_PCT);
        }
        bits
    }

    pub fn slate_position_bits(
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<HashSet<usize>> {
        let mut score_order: Vec<usize> = (0..candidates.len()).collect();
        score_order.sort_by(|&a, &b| {
            let score = |i: usize| candidates[i].score.unwrap_or(f64::NEG_INFINITY);
            score(b).partial_cmp(&score(a)).unwrap_or(Ordering::Equal)
        });

        let followed: HashSet<u64> = query
            .user_features
            .followed_user_ids
            .iter()
            .map(|&id| id as u64)
            .collect();
        let engaged_authors = query
            .columnar_scoring_sequence
            .as_ref()
            .and_then(viewer_history::positively_engaged_author_ids);

        let mut author_counts: HashMap<u64, usize> = HashMap::new();
        let mut seen_sid_l1: HashSet<&[i32]> = HashSet::new();
        let mut seen_sid_l2: HashSet<&[i32]> = HashSet::new();
        let mut bits = vec![HashSet::new(); candidates.len()];

        for idx in score_order {
            let candidate = &candidates[idx];
            let post_bits = &mut bits[idx];

            let prior_appearances = author_counts.entry(candidate.author_id).or_insert(0);
            if *prior_appearances >= 1 {
                post_bits.insert(AUTHOR_REPEAT_IN_SLATE);
            }
            if *prior_appearances >= 2 {
                post_bits.insert(AUTHOR_REPEAT_GTE_3_IN_SLATE);
            }
            *prior_appearances += 1;

            if let Some(engaged) = &engaged_authors
                && !followed.contains(&candidate.author_id)
                && !engaged.contains(&candidate.author_id)
            {
                post_bits.insert(AUTHOR_NOT_ENGAGED_BY_VIEWER);
            }

            if let Some(l1) = candidate.semantic_id_prefix(1) {
                post_bits.insert(HAS_SEMANTIC_IDS);
                if !seen_sid_l1.insert(l1) {
                    post_bits.insert(SID_L1_REPEAT_IN_SLATE);
                }
            }
            if let Some(l2) = candidate.semantic_id_prefix(2)
                && !seen_sid_l2.insert(l2)
            {
                post_bits.insert(SID_L2_REPEAT_IN_SLATE);
            }
        }
        bits
    }

        pub fn bitset_to_bytes(bits: &HashSet<usize>) -> Vec<u8> {
        if bits.is_empty() {
            return Vec::new();
        }

        let max_bit = bits.iter().max().copied().unwrap_or(0);
        let num_bytes = (max_bit / 8) + 1;
        let mut bytes = vec![0u8; num_bytes];

        for &bit_index in bits {
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            bytes[byte_index] |= 1u8 << bit_offset;
        }

        bytes
    }
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for TweetTypeMetricsHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let authors = Composition::from_keys(candidates.iter().map(|c| c.author_id));
        let author_diversity_bits = Self::author_diversity_bits(&authors);
        let slate_position_bits = Self::slate_position_bits(query, candidates);

        let mut hydrated_candidates = Vec::with_capacity(candidates.len());
        for (candidate, position_bits) in candidates.iter().zip(slate_position_bits) {
            let mut true_tweet_types = Self::create_tweet_type_bitset(candidate, query);
            true_tweet_types.extend(&author_diversity_bits);
            true_tweet_types.extend(position_bits);

            let tweet_type_metrics = Some(Self::bitset_to_bytes(&true_tweet_types));

            let hydrated = PostCandidate {
                tweet_type_metrics,
                ..Default::default()
            };
            hydrated_candidates.push(Ok(hydrated));
        }

        hydrated_candidates
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.tweet_type_metrics = hydrated.tweet_type_metrics;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::PostCandidate;
    use crate::models::query::ScoredPostsQuery;
    use std::collections::HashSet;

    #[test]
    fn test_bitset_to_bytes_empty() {
        let bits = HashSet::new();
        let bytes = TweetTypeMetricsHydrator::bitset_to_bytes(&bits);
        assert_eq!(bytes, Vec::<u8>::new());
    }

    #[test]
    fn test_bitset_to_bytes_multiple_bits_same_byte() {
        let mut bits = HashSet::new();
        bits.insert(0);
        bits.insert(2);
        bits.insert(7);
        let bytes = TweetTypeMetricsHydrator::bitset_to_bytes(&bits);
        assert_eq!(bytes, vec![0b10000101]);
    }

    #[test]
    fn test_bitset_to_bytes_multiple_bytes() {
        let mut bits = HashSet::new();
        bits.insert(0);
        bits.insert(8);
        bits.insert(15);
        let bytes = TweetTypeMetricsHydrator::bitset_to_bytes(&bits);
        assert_eq!(bytes, vec![0b00000001, 0b10000001]);
    }

    #[test]
    fn test_bitset_to_bytes_large_bit_index() {
        let mut bits = HashSet::new();
        bits.insert(314);
        let bytes = TweetTypeMetricsHydrator::bitset_to_bytes(&bits);
        assert_eq!(bytes.len(), 40);
        assert_eq!(bytes[39], 0b00000100);
    }

    #[tokio::test]
    async fn test_hydrate_multiple_candidates() {
        let hydrator = TweetTypeMetricsHydrator::new();
        let candidates = vec![
            PostCandidate {
                tweet_id: 1234567890123456789,
                retweeted_tweet_id: Some(456),
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 1234567890123456790,
                in_reply_to_tweet_id: Some(789),
                ..Default::default()
            },
        ];
        let query = ScoredPostsQuery::default();

        let hydrated = hydrator.hydrate(&query, &candidates).await;
        assert_eq!(hydrated.len(), 2);
        assert!(hydrated[0].as_ref().unwrap().tweet_type_metrics.is_some());
        assert!(hydrated[1].as_ref().unwrap().tweet_type_metrics.is_some());
    }
}
