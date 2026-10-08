use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::EnableResponseDiversityStatsExperimentBucket;
use rand::random;
use std::cmp::Ordering;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::utils::Composition;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_feature_switches::ExperimentBucket;
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets, StatsReceiverExt};

const METRIC_PREFIX: &str = "ResponseDiversity";
const DEFAULT_SAMPLING_RATE: f64 = 0.05;
const TOP_POSITIONS: usize = 10;

const STAGE_FINAL: &str = "final";
const STAGE_TOP10: &str = "top10";
const STAGE_PRE_HEURISTIC: &str = "pre_heuristic";

pub struct ResponseDiversityStatsSideEffect;

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for ResponseDiversityStatsSideEffect {
    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let Some(receiver) = global_stats_receiver() else {
            return Ok(());
        };
        let selected = &input.selected_candidates;
        if selected.is_empty() {
            return Ok(());
        }

        let experiment_buckets = input
            .query
            .params
            .experiment_buckets(EnableResponseDiversityStatsExperimentBucket);
        if experiment_buckets.is_empty() && random::<f64>() >= DEFAULT_SAMPLING_RATE {
            return Ok(());
        }
        let unbucketed = ExperimentBucket::new("", "");
        let buckets: Vec<&ExperimentBucket> = if experiment_buckets.is_empty() {
            vec![&unbucketed]
        } else {
            experiment_buckets
        };

        let final_order = sorted_desc(selected.iter(), |c| c.score);
        let top = &final_order[..TOP_POSITIONS.min(final_order.len())];
        let pre_heuristic =
            pre_heuristic_top_k(selected, &input.non_selected_candidates, selected.len());

        for bucket in &buckets {
            record_stage(receiver.as_ref(), STAGE_FINAL, &final_order, bucket);
            record_stage(receiver.as_ref(), STAGE_TOP10, top, bucket);
            record_stage(
                receiver.as_ref(),
                STAGE_PRE_HEURISTIC,
                &pre_heuristic,
                bucket,
            );
        }

        Ok(())
    }
}

fn sorted_desc<'a>(
    candidates: impl Iterator<Item = &'a PostCandidate>,
    key: impl Fn(&PostCandidate) -> Option<f64>,
) -> Vec<&'a PostCandidate> {
    let mut sorted: Vec<&PostCandidate> = candidates.collect();
    sorted.sort_by(|a, b| {
        key(b)
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&key(a).unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(Ordering::Equal)
    });
    sorted
}

fn pre_heuristic_top_k<'a>(
    selected: &'a [PostCandidate],
    non_selected: &'a [PostCandidate],
    k: usize,
) -> Vec<&'a PostCandidate> {
    let mut ranked = sorted_desc(selected.iter().chain(non_selected.iter()), |c| {
        c.weighted_score
    });
    ranked.truncate(k);
    ranked
}

fn record_stage(
    receiver: &dyn StatsReceiverExt,
    stage: &str,
    posts: &[&PostCandidate],
    bucket: &ExperimentBucket,
) {
    if posts.is_empty() {
        return;
    }
    let scopes: [(&str, &str); 3] = [
        ("stage", stage),
        ("ddg", &bucket.experiment),
        ("bucket", &bucket.bucket),
    ];

    let authors = Composition::from_keys(posts.iter().map(|c| c.author_id));
    let sources = Composition::from_keys(posts.iter().map(|c| c.served_type));
    let sid_l1 = Composition::from_keys(posts.iter().filter_map(|c| c.semantic_id_prefix(1)));
    let sid_l2 = Composition::from_keys(posts.iter().filter_map(|c| c.semantic_id_prefix(2)));
    let in_network =
        posts.iter().filter(|c| c.in_network == Some(true)).count() as f64 / posts.len() as f64;
    let sid_coverage = sid_l1.size as f64 / posts.len() as f64;

    let observe = |name: &str, value: f64, buckets: HistogramBuckets| {
        receiver.observe(&format!("{METRIC_PREFIX}.{name}"), &scopes, value, buckets);
    };
    observe("Size", posts.len() as f64, HistogramBuckets::Bucket0To50);
    observe(
        "UniqueAuthors",
        authors.unique as f64,
        HistogramBuckets::Bucket0To50,
    );
    observe(
        "UniqueAuthorRatio",
        authors.unique_ratio(),
        HistogramBuckets::Bucket0To1,
    );
    observe(
        "MaxAuthorShare",
        authors.max_share(),
        HistogramBuckets::Bucket0To1,
    );
    observe("AuthorHhi", authors.hhi, HistogramBuckets::Bucket0To1);
    observe(
        "AuthorEntropyNorm",
        authors.entropy_norm,
        HistogramBuckets::Bucket0To1,
    );
    observe(
        "UniqueSources",
        sources.unique as f64,
        HistogramBuckets::Bucket0To50,
    );
    observe(
        "SourceEntropyNorm",
        sources.entropy_norm,
        HistogramBuckets::Bucket0To1,
    );
    observe("InNetworkShare", in_network, HistogramBuckets::Bucket0To1);
    observe("SidCoverage", sid_coverage, HistogramBuckets::Bucket0To1);
    if sid_l1.size > 0 {
        observe(
            "UniqueSidL1",
            sid_l1.unique as f64,
            HistogramBuckets::Bucket0To50,
        );
        observe(
            "MaxSidL1Share",
            sid_l1.max_share(),
            HistogramBuckets::Bucket0To1,
        );
        observe(
            "SidL1EntropyNorm",
            sid_l1.entropy_norm,
            HistogramBuckets::Bucket0To1,
        );
    }
    if sid_l2.size > 0 {
        observe(
            "UniqueSidL2",
            sid_l2.unique as f64,
            HistogramBuckets::Bucket0To50,
        );
        observe(
            "SidL2EntropyNorm",
            sid_l2.entropy_norm,
            HistogramBuckets::Bucket0To1,
        );
    }
}
