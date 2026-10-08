use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{AuthorServedMetricsAuthorIds, EnableAuthorServedMetricsExperimentBucket};
use crate::util::popular_authors::{now_ms, PopularAuthorsCache};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_stats_receiver::global_stats_receiver;

const SERVED_METRIC: &str = "AuthorServedMetrics.Served";
const SCORED_METRIC: &str = "AuthorServedMetrics.Scored";
const SERVED_TOP10_METRIC: &str = "AuthorServedMetrics.ServedTop10";
const TOP10: usize = 10;
const REQUESTS_METRIC: &str = "AuthorServedMetrics.Requests";
const PAGE_SLOTS_METRIC: &str = "AuthorServedMetrics.PageSlots";
const POST_UNEXPLORED_METRIC: &str = "AuthorServedMetrics.PostUnexplored";
const POST_UNEXPLORED_MILLI_METRIC: &str = "AuthorServedMetrics.PostUnexploredMilli";
const POPULAR_AUTHOR_METRIC: &str = "AuthorServedMetrics.PopularAuthorSlots";
const POST_UNEXPLORED_THRESHOLD: f64 = 0.5;
const PAGE_RANGES: [(&str, usize); 3] = [("top10", 10), ("top35", 35), ("total", usize::MAX)];
const UNBUCKETED: &str = "all";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PostType {
    Original,
    Reply,
    Retweet,
}

impl PostType {
    fn classify(candidate: &PostCandidate) -> Self {
        if candidate.retweeted_tweet_id.is_some() {
            PostType::Retweet
        } else if candidate.in_reply_to_tweet_id.is_some() {
            PostType::Reply
        } else {
            PostType::Original
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            PostType::Original => "original",
            PostType::Reply => "reply",
            PostType::Retweet => "retweet",
        }
    }
}

pub struct AuthorServedMetricsSideEffect {
    pub popular_authors: Arc<PopularAuthorsCache>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RangeStats {
    pub slots: u64,
    pub post_unexplored: u64,
    pub post_unexplored_milli: u64,
    pub popular_in_network: u64,
    pub popular_out_of_network: u64,
}

pub fn served_page_stats(
    selected: &[PostCandidate],
    is_popular_author: impl Fn(u64) -> bool,
) -> [RangeStats; 3] {
    let mut page: Vec<&PostCandidate> = selected.iter().collect();
    page.sort_by(|a, b| {
        b.score
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&a.score.unwrap_or(f64::NEG_INFINITY))
    });
    PAGE_RANGES.map(|(_, n)| {
        let mut stats = RangeStats::default();
        for c in page.iter().take(n) {
            let p = c.phoenix_scores.post_unexplored_score.unwrap_or(0.0);
            stats.slots += 1;
            stats.post_unexplored += u64::from(p >= POST_UNEXPLORED_THRESHOLD);
            stats.post_unexplored_milli += (p.clamp(0.0, 1.0) * 1000.0).round() as u64;
            if is_popular_author(c.author_id) {
                if c.in_network == Some(true) {
                    stats.popular_in_network += 1;
                } else {
                    stats.popular_out_of_network += 1;
                }
            }
        }
        stats
    })
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for AuthorServedMetricsSideEffect {
    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let Some(receiver) = global_stats_receiver() else {
            return Ok(());
        };

        let params = &input.query.params;
        let mut buckets: Vec<(String, String)> = params
            .experiment_buckets(EnableAuthorServedMetricsExperimentBucket)
            .into_iter()
            .map(|b| (b.experiment.clone(), b.bucket.clone()))
            .collect();
        if buckets.is_empty() && params.get(EnableAuthorServedMetricsExperimentBucket) {
            buckets.push((UNBUCKETED.to_string(), UNBUCKETED.to_string()));
        }
        if buckets.is_empty() {
            return Ok(());
        }

        for (ddg, bucket) in &buckets {
            receiver.incr(REQUESTS_METRIC, &[("ddg", ddg), ("bucket", bucket)], 1);
        }

        self.popular_authors.maybe_spawn_refresh(now_ms());
        let page = served_page_stats(&input.selected_candidates, |author_id| {
            self.popular_authors.contains(author_id)
        });
        for ((range, _), stats) in PAGE_RANGES.iter().zip(page) {
            for (ddg, bucket) in &buckets {
                let labels = [
                    ("ddg", ddg.as_str()),
                    ("bucket", bucket.as_str()),
                    ("range", range),
                ];
                receiver.incr(PAGE_SLOTS_METRIC, &labels, stats.slots);
                receiver.incr(POST_UNEXPLORED_METRIC, &labels, stats.post_unexplored);
                receiver.incr(
                    POST_UNEXPLORED_MILLI_METRIC,
                    &labels,
                    stats.post_unexplored_milli,
                );
                for (network, count) in [
                    ("in", stats.popular_in_network),
                    ("oon", stats.popular_out_of_network),
                ] {
                    receiver.incr(
                        POPULAR_AUTHOR_METRIC,
                        &[
                            ("ddg", ddg.as_str()),
                            ("bucket", bucket.as_str()),
                            ("range", range),
                            ("network", network),
                        ],
                        count,
                    );
                }
            }
        }

        let tracked: HashSet<u64> = params
            .get(AuthorServedMetricsAuthorIds)
            .into_iter()
            .collect();
        if tracked.is_empty() {
            return Ok(());
        }
        let served = aggregate_counts(&input.selected_candidates, &tracked);
        let served_top10 = aggregate_counts(
            &input.selected_candidates[..input.selected_candidates.len().min(TOP10)],
            &tracked,
        );
        let mut scored = aggregate_counts(&input.non_selected_candidates, &tracked);
        for (key, count) in &served {
            *scored.entry(*key).or_insert(0) += count;
        }
        for (metric, counts) in [
            (SERVED_METRIC, served),
            (SERVED_TOP10_METRIC, served_top10),
            (SCORED_METRIC, scored),
        ] {
            for ((author_id, post_type), count) in counts {
                let author_str = author_id.to_string();
                for (ddg, bucket) in &buckets {
                    receiver.incr(
                        metric,
                        &[
                            ("type", post_type.as_str()),
                            ("author_id", &author_str),
                            ("ddg", ddg),
                            ("bucket", bucket),
                        ],
                        count,
                    );
                }
            }
        }

        Ok(())
    }
}

fn aggregate_counts(
    candidates: &[PostCandidate],
    tracked: &HashSet<u64>,
) -> HashMap<(u64, PostType), u64> {
    let mut counts: HashMap<(u64, PostType), u64> = HashMap::new();
    for candidate in candidates {
        if tracked.contains(&candidate.author_id) {
            *counts
                .entry((candidate.author_id, PostType::classify(candidate)))
                .or_insert(0) += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original(author_id: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            ..Default::default()
        }
    }

    fn reply(author_id: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            in_reply_to_tweet_id: Some(999),
            ..Default::default()
        }
    }

    fn retweet(author_id: u64) -> PostCandidate {
        PostCandidate {
            author_id,
            retweeted_tweet_id: Some(999),
            retweeted_user_id: Some(7),
            ..Default::default()
        }
    }

    #[test]
    fn classify_distinguishes_types() {
        assert_eq!(PostType::classify(&original(1)), PostType::Original);
        assert_eq!(PostType::classify(&reply(1)), PostType::Reply);
        assert_eq!(PostType::classify(&retweet(1)), PostType::Retweet);
    }

    #[test]
    fn retweet_takes_precedence_over_reply() {
        let mut candidate = retweet(1);
        candidate.in_reply_to_tweet_id = Some(123);
        assert_eq!(PostType::classify(&candidate), PostType::Retweet);
    }

    #[test]
    fn aggregate_counts_filters_untracked_and_buckets_by_type() {
        let tracked: HashSet<u64> = [10, 20].into_iter().collect();
        let candidates = vec![
            original(10),
            original(10),
            reply(10),
            retweet(20),
            original(30),
            reply(99),
        ];

        let counts = aggregate_counts(&candidates, &tracked);

        assert_eq!(counts.get(&(10, PostType::Original)), Some(&2));
        assert_eq!(counts.get(&(10, PostType::Reply)), Some(&1));
        assert_eq!(counts.get(&(20, PostType::Retweet)), Some(&1));
        assert_eq!(counts.get(&(30, PostType::Original)), None);
        assert_eq!(counts.get(&(99, PostType::Reply)), None);
        assert_eq!(counts.values().sum::<u64>(), 4);
    }

    fn served(score: f64, author_id: u64, post_unexplored: f64, in_network: bool) -> PostCandidate {
        PostCandidate {
            author_id,
            score: Some(score),
            in_network: Some(in_network),
            phoenix_scores: crate::models::candidate::PhoenixScores {
                post_unexplored_score: Some(post_unexplored),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn served_page_stats_counts_post_unexplored_share_per_range() {
        let mut page: Vec<PostCandidate> = (0..40u64)
            .map(|r| {
                let p = if r % 2 == 0 { 0.9 } else { 0.001 };
                let (author, in_net) = match r {
                    3 => (7, true),
                    12 => (8, false),
                    37 => (8, false),
                    _ => (100 + r, false),
                };
                served(100.0 - r as f64, author, p, in_net)
            })
            .collect();
        page.reverse();
        let [top10, top35, total] = served_page_stats(&page, |a| a == 7 || a == 8);

        assert_eq!(top10.slots, 10);
        assert_eq!(top10.post_unexplored, 5);
        assert_eq!(top10.post_unexplored_milli, 5 * 900 + 5);
        assert_eq!(
            (top10.popular_in_network, top10.popular_out_of_network),
            (1, 0)
        );

        assert_eq!(top35.slots, 35);
        assert_eq!(top35.post_unexplored, 18);
        assert_eq!(top35.post_unexplored_milli, 18 * 900 + 17);
        assert_eq!(
            (top35.popular_in_network, top35.popular_out_of_network),
            (1, 1)
        );

        assert_eq!(total.slots, 40);
        assert_eq!(total.post_unexplored, 20);
        assert_eq!(
            (total.popular_in_network, total.popular_out_of_network),
            (1, 2)
        );
    }

    #[test]
    fn aggregate_counts_empty_when_no_tracked_match() {
        let tracked: HashSet<u64> = [1].into_iter().collect();
        let candidates = vec![original(2), reply(3), retweet(4)];
        assert!(aggregate_counts(&candidates, &tracked).is_empty());
    }
}
