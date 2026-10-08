pub mod dpp_model;
pub mod value_model;

use std::sync::Arc;

use log::error;

use xai_feature_switches::Params;
use xai_vm_ranker_proto::{RankRequest, RankedCandidate};

use crate::dpp::DppConfig;
use crate::embedding_store::EmbeddingStore;
use crate::metrics::PARAMS_RESOLVED;
use crate::params::*;
use crate::ranking_config::RankingConfig;
use dpp_model::EmbeddingMemo;
use value_model::ValueModelOutput;

#[derive(Clone)]
pub struct DppContext {
    pub store: Arc<EmbeddingStore>,
    pub config: DppConfig,
}

#[derive(Clone, Default)]
pub struct RankContext {
    pub dpp: Option<DppContext>,
    pub config: Option<Arc<RankingConfig>>,
}

pub async fn rank(req: RankRequest, ctx: RankContext) -> Result<Vec<RankedCandidate>, String> {
    let params = resolve_params(&req, ctx.config.as_deref());
    let compute_value_model =
        req.compute_value_model || params.as_ref().is_some_and(|p| p.get(ComputeValueModel));
    let dpp_enabled = params.as_ref().is_none_or(|p| p.get(DppEnabled));

    if let (Some(dpp), true) = (ctx.dpp.clone(), dpp_enabled) {
        if req.value_model_id != "dpp" {
            error!(
                "DPP context provided but value_model_id='{}' is not 'dpp'",
                req.value_model_id
            );
        }
        let mut dpp = dpp;

        match (&params, &req.dpp_params) {
            (Some(p), _) => {
                dpp.config.theta = p.get(DppTheta);
                dpp.config.max_selected_rank = p.get(DppMaxSelectedRank) as usize;
            }
            (None, Some(overrides)) => {
                if overrides.theta != 0.0 {
                    dpp.config.theta = overrides.theta;
                }
                if overrides.max_selected_rank != 0 {
                    dpp.config.max_selected_rank = overrides.max_selected_rank as usize;
                }
            }
            (None, None) => {}
        }

        let config = ctx.config.clone();
        return tokio::task::spawn_blocking(move || {
            let value_model = value_model::compute(
                &req,
                params.as_ref(),
                config.as_deref().map(RankingConfig::author_rules),
                compute_value_model,
            );
            let served_pre_dpp = served_pre_dpp_scores(&req, value_model.as_ref());
            let mut memo = EmbeddingMemo::default();
            let served = dpp_model::rank_with_memo(&req, &served_pre_dpp, &dpp, &mut memo);
            ranked_candidates(&req, served, value_model.as_ref())
        })
        .await
        .map_err(|e| format!("DPP spawn_blocking failed: {e}"));
    }

    let value_model = value_model::compute(
        &req,
        params.as_ref(),
        ctx.config.as_deref().map(RankingConfig::author_rules),
        compute_value_model,
    );
    let served = served_pre_dpp_scores(&req, value_model.as_ref())
        .iter()
        .map(|s| s.unwrap_or(0.0))
        .collect();
    Ok(ranked_candidates(&req, served, value_model.as_ref()))
}

fn resolve_params(req: &RankRequest, config: Option<&RankingConfig>) -> Option<Params> {
    let outcome = match (req.viewer.as_ref(), config) {
        (Some(_), Some(_)) => "resolved",
        (None, _) => "no_viewer_context",
        (_, None) => "no_config",
    };
    PARAMS_RESOLVED.with_label_values(&[outcome]).inc();
    let config = config?;
    Some(match req.viewer.as_ref() {
        Some(viewer) => config.resolve(viewer),
        None => config.resolve_anonymous(),
    })
}

fn served_pre_dpp_scores(
    req: &RankRequest,
    value_model: Option<&ValueModelOutput>,
) -> Vec<Option<f64>> {
    match value_model {
        Some(vm) => vm.scores.iter().copied().map(Some).collect(),
        None => req.candidates.iter().map(|c| c.score).collect(),
    }
}

fn ranked_candidates(
    req: &RankRequest,
    final_scores: Vec<f64>,
    value_model: Option<&ValueModelOutput>,
) -> Vec<RankedCandidate> {
    let weighted = value_model;
    req.candidates
        .iter()
        .zip(final_scores)
        .enumerate()
        .map(|(i, (c, score))| RankedCandidate {
            tweet_id: c.tweet_id,
            score,
            weighted_score: weighted.map(|vm| vm.weighted[i]),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_feature_switches::{
        BucketMembership, ExperimentBucket, ExperimentBucketsChooser, FeatureSwitches,
        NullBucketImpressor, Recipient,
    };
    use xai_value_model::{
        compute_value_scores, CandidateScoringInputs, QueryScoringContext, ValueModelWeights,
    };
    use xai_vm_ranker_proto::{PhoenixScores, RankCandidate, ViewerContext};

    const FEATURES_YAML: &str = r#"
rust_home_mixer:
  description: "test"
  owner: "test@example.com"
  parameters:
    rust_home_mixer_favorite_weight:
      type: double
      default: 2.0
    rust_home_mixer_reply_weight:
      type: double
      default: 4.0
    rust_home_mixer_report_weight:
      type: double
      default: -100.0
    rust_home_mixer_oon_weight_factor:
      type: double
      default: 0.75
  rules: []
"#;

    fn candidate(
        tweet_id: u64,
        author_id: u64,
        fav: f64,
        reply: Option<f64>,
        in_network: bool,
        upstream_score: f64,
    ) -> RankCandidate {
        RankCandidate {
            tweet_id,
            author_id,
            in_network,
            score: Some(upstream_score),
            phoenix_scores: Some(PhoenixScores {
                favorite_score: Some(fav),
                reply_score: reply,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn weights() -> ValueModelWeights {
        let params = config().resolve(&viewer("US"));
        value_model::weights_from_params(&params, 7)
    }

    fn scoring_context() -> QueryScoringContext {
        QueryScoringContext {
            effective_oon_weight: 0.75,
        }
    }

    fn config() -> Arc<RankingConfig> {
        Arc::new(RankingConfig::from_yaml(FEATURES_YAML).unwrap())
    }

    fn context(config: Arc<RankingConfig>) -> RankContext {
        RankContext {
            dpp: None,
            config: Some(config),
        }
    }

    fn viewer(country: &str) -> ViewerContext {
        ViewerContext {
            user_id: 7,
            country_code: country.to_string(),
            product: "ForYou".to_string(),
            now_ms: 1_700_000_000_000,
            ..Default::default()
        }
    }

    fn request(candidates: Vec<RankCandidate>, country: &str) -> RankRequest {
        RankRequest {
            viewer_id: 7,
            request_timestamp_ms: 1_700_000_000_000,
            candidates,
            value_model_id: "dpp".to_string(),
            viewer: Some(viewer(country)),
            compute_value_model: true,
            ..Default::default()
        }
    }

    fn inputs(candidates: &[RankCandidate]) -> Vec<CandidateScoringInputs> {
        candidates
            .iter()
            .map(|c| CandidateScoringInputs::from_rank_candidate(c, 10_000))
            .collect()
    }

    fn order_by_score_desc(scores: &[f64]) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        idx
    }

    #[tokio::test]
    async fn value_model_scores_match_shared_crate_and_passthrough_keeps_upstream() {
        let candidates = vec![
            candidate(1, 10, 0.5, None, true, 1.0),
            candidate(2, 20, 0.4, Some(0.2), false, 2.0),
            candidate(3, 10, 0.45, None, true, 3.0),
        ];
        let req = request(candidates.clone(), "US");
        let weights = weights();
        assert_eq!(weights.favorite, 2.0);
        assert_eq!(weights.retweet, 1.0);

        let ranked = rank(req.clone(), context(config())).await.unwrap();

        let expected = compute_value_scores(&weights, &scoring_context(), &inputs(&candidates));
        assert_eq!(ranked.len(), 3);
        for (i, r) in ranked.iter().enumerate() {
            assert_eq!(r.tweet_id, candidates[i].tweet_id);
            assert_eq!(r.score, expected.scores[i]);
        }
        let ranked_scores: Vec<f64> = ranked.iter().map(|r| r.score).collect();
        assert_eq!(
            order_by_score_desc(&ranked_scores),
            order_by_score_desc(&expected.scores)
        );
        assert_eq!(order_by_score_desc(&expected.scores), vec![1, 0, 2]);

        let passthrough = rank(
            RankRequest {
                compute_value_model: false,
                ..req.clone()
            },
            context(config()),
        )
        .await
        .unwrap();
        for (r, c) in passthrough.iter().zip(&candidates) {
            assert_eq!(r.tweet_id, c.tweet_id);
            assert_eq!(r.score, c.score.unwrap());
        }
    }

    #[derive(Debug)]
    struct TreatmentAuthor(u64);

    impl ExperimentBucketsChooser for TreatmentAuthor {
        fn choose_buckets(&self, _recipient: &dyn Recipient) -> BucketMembership {
            BucketMembership::new()
        }

        fn choose_bucket_without_overrides(
            &self,
            experiment_key: &str,
            recipient: &dyn Recipient,
        ) -> Option<ExperimentBucket> {
            (recipient.user_id()? == self.0)
                .then(|| ExperimentBucket::new(experiment_key, "treatment").with_version(1))
        }
    }

    #[tokio::test]
    async fn author_exploration_bonus_boosts_treatment_authors() {
        let yaml = format!(
            "{FEATURES_YAML}\n{}",
            r#"
rust_home_mixer_author_exploration:
  parameters:
    rust_home_mixer_author_exploration_bonus:
      type: double
      default: 0.0
  rules:
    - query: "[author_exploration author_bucket_membership treatment]"
      values:
        rust_home_mixer_author_exploration_bonus: 0.2
"#
        );
        let fs = FeatureSwitches::with_options(
            xai_feature_switches::load_yaml_string(&yaml).unwrap(),
            Arc::new(TreatmentAuthor(10)),
            Arc::new(NullBucketImpressor),
            None,
            false,
        )
        .unwrap();
        let author_config = Arc::new(RankingConfig::new(Arc::new(fs)));
        let req = request(
            vec![
                candidate(1, 10, 0.5, None, true, 0.0),
                candidate(2, 20, 0.5, None, true, 0.0),
            ],
            "US",
        );

        for pre_offset in [false, true] {
            let mut req = req.clone();
            req.viewer.as_mut().unwrap().fs_overrides.insert(
                "rust_home_mixer_multiplier_pre_offset".to_string(),
                pre_offset.to_string(),
            );
            let base = rank(req.clone(), context(config())).await.unwrap();
            let boosted = rank(req, context(Arc::clone(&author_config)))
                .await
                .unwrap();

            assert!((boosted[0].score - (base[0].score + 0.2)).abs() < 1e-12);
            assert_eq!(boosted[1].score, base[1].score);
            assert_eq!(boosted[0].weighted_score, base[0].weighted_score);
        }
    }
}
