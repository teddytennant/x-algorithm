use crate::models::candidate::CandidateHelpers;
use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnableHomeExcursionScores, PhoenixInferenceClusterId, PhoenixRankerNewUserHistoryThreshold,
    PhoenixRankerNewUserInferenceClusterId, RerankerHeadTag,
};
use crate::util::egress::PredictionDispatch;
use crate::util::phoenix_request::build_prediction_request;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PredictNextActions,
};

use xai_candidate_pipeline::component_library::models::PhoenixScores;
use xai_candidate_pipeline::component_library::utils::current_timestamp_millis;
use xai_candidate_pipeline::scorer::Scorer;
use xai_recsys_proto::{ContinuousActionName, ProductSurface};

pub const PHOENIX_RANKER_KILL_SWITCH_DECIDER: &str = "disable_home_mixer_phoenix_ranker";

pub struct PhoenixScorer {
    pub dispatch: PredictionDispatch,
}

impl PhoenixScorer {
    pub(crate) fn resolve_cluster(query: &ScoredPostsQuery) -> PhoenixCluster {
        let configured_cluster =
            PhoenixCluster::parse(&query.params.get(PhoenixInferenceClusterId));

        let threshold: u64 = query.params.get(PhoenixRankerNewUserHistoryThreshold);
        if threshold > 0 {
            let action_count = query
                .scoring_sequence
                .as_ref()
                .and_then(|s| s.metadata.as_ref())
                .map(|m| m.length)
                .unwrap_or(0);

            if action_count < threshold {
                return PhoenixCluster::parse(
                    &query.params.get(PhoenixRankerNewUserInferenceClusterId),
                );
            }
        }

        if let Some(decider) = &query.decider {
            let is_prod = matches!(
                configured_cluster,
                PhoenixCluster::Experiment1Fou | PhoenixCluster::Experiment2Fou
            );
            if is_prod {
                if decider.enabled("override_qf_use_experiment2_fou") {
                    return PhoenixCluster::Experiment2Fou;
                }
                if decider.enabled("override_qf_use_experiment1_fou") {
                    return PhoenixCluster::Experiment1Fou;
                }
            }
        }

        configured_cluster
    }
}

#[async_trait]
impl Scorer<ScoredPostsQuery, PostCandidate> for PhoenixScorer {
    fn enable(&self, query: &ScoredPostsQuery) -> bool {
        if query.has_cached_posts {
            return false;
        }
        let killed = query
            .decider
            .as_ref()
            .is_some_and(|d| d.enabled(PHOENIX_RANKER_KILL_SWITCH_DECIDER));
        !killed
    }

    async fn score(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let last_scored_at_ms = current_timestamp_millis();
        let product_surface = if query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };

        if query.scoring_sequence.is_none() {
            return vec![Ok(PostCandidate::default()); candidates.len()];
        };

        let cluster = Self::resolve_cluster(query);
        let request = build_prediction_request(query, candidates, product_surface);

        let predictions = self
            .dispatch
            .predict_with_fallback(query, cluster, request)
            .await
            .map_err(|e| format!("Phoenix prediction failed: {}", e));

        let predictions = match predictions {
            Ok(predictions) => predictions,
            Err(err) => return vec![Err(err); candidates.len()],
        };

        let home_excursion_scores = query.params.get(EnableHomeExcursionScores);
        candidates
            .iter()
            .map(|c| PostCandidate {
                phoenix_scores: if home_excursion_scores {
                    with_home_excursion_heads(&predictions, c.get_original_tweet_id())
                } else {
                    predictions.candidate_scores(&c.get_original_tweet_id())
                },
                backbone_scores: predictions.candidate_backbone_scores(&c.get_original_tweet_id()),
                served_slate_context: predictions
                    .candidate_slate_context(&c.get_original_tweet_id())
                    .map(Into::into),
                prediction_request_id: Some(query.prediction_id),
                last_scored_at_ms,
                reranker_head_tag: Some(query.params.get(RerankerHeadTag) as u32),
                ..Default::default()
            })
            .map(Ok)
            .collect()
    }

    fn update(&self, candidate: &mut PostCandidate, scored: PostCandidate) {
        candidate.phoenix_scores = scored.phoenix_scores;
        candidate.backbone_scores = scored.backbone_scores;
        candidate.served_slate_context = scored.served_slate_context;
        candidate.prediction_request_id = scored.prediction_request_id;
        candidate.last_scored_at_ms = scored.last_scored_at_ms;
        candidate.reranker_head_tag = scored.reranker_head_tag;
    }
}

fn with_home_excursion_heads(predictions: &PredictNextActions, tweet_id: u64) -> PhoenixScores {
    PhoenixScores {
        home_video_continuation_secs: predictions.candidate_continuous_value(
            &tweet_id,
            ContinuousActionName::HomeVideoContinuationSecs as usize,
        ),
        home_profile_visit_secs: predictions.candidate_continuous_value(
            &tweet_id,
            ContinuousActionName::HomeProfileVisitSecs as usize,
        ),
        ..predictions.candidate_scores(&tweet_id)
    }
}
