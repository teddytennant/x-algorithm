use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::util::phoenix_request::build_prediction_request;
use crate::util::shadow::is_shadow_sampled_for_cluster;
use crate::util::xds::use_xds_for_cluster;
use futures::future::join_all;
use prost::Message;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use strum::VariantArray;
use thrift::OrderedFloat;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::KafkaPublisherClient;
use xai_candidate_pipeline::component_library::clients::phoenix_prediction_client::{
    PhoenixCluster, PhoenixPredictionClient,
};
use xai_candidate_pipeline::component_library::models::PhoenixScores;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_proto::recsys_logging::LoggedScoredCandidate;
use xai_recsys_logging_thrift::{serialize_to_bytes_binary, PredictionScore, ScoredCandidate};
use xai_recsys_proto::{language_code_string_to_enum, ProductSurface};

pub struct PhoenixExperimentsSideEffect {
    phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
    xds_client: Option<Arc<dyn PhoenixPredictionClient + Send + Sync>>,
    kafka_client: Arc<dyn KafkaPublisherClient>,
    logged_scored_candidates_kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl PhoenixExperimentsSideEffect {
    pub fn new(
        phoenix_client: Arc<dyn PhoenixPredictionClient + Send + Sync>,
        xds_client: Option<Arc<dyn PhoenixPredictionClient + Send + Sync>>,
        kafka_client: Arc<dyn KafkaPublisherClient>,
        logged_scored_candidates_kafka_client: Arc<dyn KafkaPublisherClient>,
    ) -> Self {
        Self {
            phoenix_client,
            xds_client,
            kafka_client,
            logged_scored_candidates_kafka_client,
        }
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for PhoenixExperimentsSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        query.is_shadow_traffic
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        if input.query.scoring_sequence.is_none() {
            return Ok(());
        };

        let request_time_ms = input.query.request_time_ms;

        let product_surface = if input.query.in_network_only {
            ProductSurface::HomeTimelineRankedFollowing
        } else {
            ProductSurface::HomeTimelineRanking
        };

        let user_id = input.query.user_id;

        let base_request =
            build_prediction_request(&input.query, &input.selected_candidates, product_surface);

        let futures = PhoenixCluster::VARIANTS
            .iter()
            .filter(|&&c| {
                is_shadow_sampled_for_cluster(&input.query.params, input.query.request_id, c)
            })
            .map(|&cluster_id| {
                let cluster_name = format!("{cluster_id:?}");
                let client = if use_xds_for_cluster(&input.query, &cluster_name) {
                    self.xds_client
                        .as_ref()
                        .map(Arc::clone)
                        .unwrap_or_else(|| Arc::clone(&self.phoenix_client))
                } else {
                    Arc::clone(&self.phoenix_client)
                };
                let request = base_request.clone();
                async move {
                    let result: Result<_, tonic::Status> =
                        client.predict(cluster_id, request, 0).await;
                    if let Err(ref err) = result {
                        tracing::error!(
                            "Phoenix experiment {:?} request failed: {}",
                            cluster_id,
                            err
                        );
                    }
                    (cluster_id, result)
                }
            });
        let experiment_results: Vec<_> = join_all(futures).await;

        for candidate in &input.selected_candidates {
            let mut prediction_scores = HashMap::new();

            for (cluster_id, result) in &experiment_results {
                let Ok(predictions) = result else {
                    continue;
                };
                let cluster_name = format!("{:?}", cluster_id);
                let scores = predictions.candidate_scores(&candidate.get_original_tweet_id());
                insert_cluster_scores(&mut prediction_scores, &cluster_name, &scores);
            }

            let thrift_scores: BTreeSet<Box<PredictionScore>> = prediction_scores
                .iter()
                .map(|(name, value)| {
                    Box::new(PredictionScore::new(
                        Some(name.clone()),
                        Some(OrderedFloat::from(*value)),
                    ))
                })
                .collect();

            let source_tweet_id = candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id);
            let scored = ScoredCandidate {
                tweet_id: candidate.tweet_id as i64,
                viewer_id: Some(user_id as i64),
                author_id: Some(candidate.author_id as i64),
                request_join_id: Some(input.query.request_id as i64),
                score: None,
                suggest_type: None,
                is_in_network: candidate.in_network,
                in_reply_to_tweet_id: candidate.in_reply_to_tweet_id.map(|id| id as i64),
                quoted_tweet_id: candidate.quoted_tweet_id.map(|id| id as i64),
                quoted_user_id: candidate.quoted_user_id.map(|id| id as i64),
                request_time_ms: Some(request_time_ms),
                source_tweet_id: Some(source_tweet_id as i64),
                prediction_scores: Some(thrift_scores),
                fav_count: candidate.fav_count,
                reply_count: candidate.reply_count,
                retweet_count: candidate.repost_count,
                quote_count: candidate.quote_count,
                has_media: candidate.has_media,
                language_code: candidate
                    .language_code
                    .as_deref()
                    .map(|lc| language_code_string_to_enum(lc) as i32),
                video_duration_ms: candidate.min_video_duration_ms,
                raw_query: None,
            };

            let thrift_bytes = serialize_to_bytes_binary(&scored);
            let proto_bytes = build_logged_scored_candidate(
                candidate,
                user_id,
                input.query.request_id,
                request_time_ms,
                prediction_scores,
            )
            .encode_to_vec();
            let (thrift_result, proto_result) = tokio::join!(
                async {
                    match thrift_bytes {
                        Ok(bytes) => self.kafka_client.send(&bytes).await,
                        Err(e) => {
                            tracing::error!("Failed to serialize scored candidate: {e}");
                            Ok(())
                        }
                    }
                },
                self.logged_scored_candidates_kafka_client
                    .send(&proto_bytes)
            );
            if let Err(e) = thrift_result {
                tracing::error!("Failed to publish scored candidate to Kafka: {e}");
            }
            if let Err(e) = proto_result {
                tracing::error!("Failed to publish logged scored candidate to Kafka: {e}");
            }
        }

        Ok(())
    }
}

fn insert_cluster_scores(
    prediction_scores: &mut HashMap<String, f64>,
    cluster_name: &str,
    s: &PhoenixScores,
) {
    let score_fields: &[(&str, Option<f64>)] = &[
        ("favorite", s.favorite_score),
        ("reply", s.reply_score),
        ("retweet", s.retweet_score),
        ("photo_expand", s.photo_expand_score),
        ("video_open", s.video_open_score),
        ("click", s.click_score),
        ("open_link", s.open_link_score),
        ("profile_click", s.profile_click_score),
        ("vqv", s.vqv_score),
        ("share", s.share_score),
        ("share_via_dm", s.share_via_dm_score),
        ("share_via_copy_link", s.share_via_copy_link_score),
        ("dwell", s.dwell_score),
        ("quote", s.quote_score),
        ("quoted_click", s.quoted_click_score),
        ("quoted_vqv", s.quoted_vqv_score),
        ("follow_author", s.follow_author_score),
        ("not_interested", s.not_interested_score),
        ("block_author", s.block_author_score),
        ("mute_author", s.mute_author_score),
        ("report", s.report_score),
        ("not_dwelled", s.not_dwelled_score),
        ("post_unexplored", s.post_unexplored_score),
        ("pdwell", s.post_unexplored_score),
        ("dwell_time", s.dwell_time),
        ("click_dwell_time", s.click_dwell_time),
    ];
    for (name, score) in score_fields {
        if let Some(value) = score {
            prediction_scores.insert(format!("phoenix.{cluster_name}.{name}"), *value);
        }
    }
}

fn build_logged_scored_candidate(
    candidate: &PostCandidate,
    user_id: u64,
    request_id: u64,
    request_time_ms: i64,
    prediction_scores: HashMap<String, f64>,
) -> LoggedScoredCandidate {
    let source_post_id = candidate.retweeted_tweet_id.unwrap_or(candidate.tweet_id);
    LoggedScoredCandidate {
        post_id: candidate.tweet_id as i64,
        viewer_id: Some(user_id as i64),
        author_id: Some(candidate.author_id as i64),
        request_join_id: Some(request_id as i64),
        request_time_ms: Some(request_time_ms),
        source_post_id: Some(source_post_id as i64),
        is_in_network: candidate.in_network,
        in_reply_to_post_id: candidate.in_reply_to_tweet_id.map(|id| id as i64),
        quoted_post_id: candidate.quoted_tweet_id.map(|id| id as i64),
        quoted_user_id: candidate.quoted_user_id.map(|id| id as i64),
        prediction_scores,
        fav_count: candidate.fav_count,
        reply_count: candidate.reply_count,
        retweet_count: candidate.repost_count,
        quote_count: candidate.quote_count,
        has_media: candidate.has_media,
        language_code: candidate
            .language_code
            .as_deref()
            .map(|lc| language_code_string_to_enum(lc) as i32),
        video_duration_ms: candidate.min_video_duration_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_present_heads_with_cluster() {
        let mut prediction_scores = HashMap::new();
        insert_cluster_scores(
            &mut prediction_scores,
            "Experiment1",
            &PhoenixScores {
                favorite_score: Some(0.12),
                reply_score: None,
                post_unexplored_score: Some(0.4),
                ..Default::default()
            },
        );
        assert_eq!(
            prediction_scores.get("phoenix.Experiment1.favorite"),
            Some(&0.12)
        );
        assert_eq!(
            prediction_scores.get("phoenix.Experiment1.pdwell"),
            Some(&0.4)
        );
        assert_eq!(
            prediction_scores.get("phoenix.Experiment1.post_unexplored"),
            Some(&0.4)
        );
        assert!(!prediction_scores.contains_key("phoenix.Experiment1.reply"));
    }

    #[test]
    fn copies_candidate_identity_and_features() {
        let candidate = PostCandidate {
            tweet_id: 11,
            author_id: 22,
            retweeted_tweet_id: Some(33),
            in_reply_to_tweet_id: Some(44),
            quoted_tweet_id: Some(55),
            quoted_user_id: Some(66),
            in_network: Some(true),
            fav_count: Some(1),
            reply_count: Some(2),
            repost_count: Some(3),
            quote_count: Some(4),
            has_media: Some(true),
            language_code: Some("en".to_string()),
            min_video_duration_ms: Some(1500),
            ..Default::default()
        };
        let logged = build_logged_scored_candidate(
            &candidate,
            99,
            123,
            1_700_000_000_000,
            HashMap::from([("phoenix.Prod.favorite".to_string(), 0.5)]),
        );
        assert_eq!(logged.post_id, 11);
        assert_eq!(logged.viewer_id, Some(99));
        assert_eq!(logged.author_id, Some(22));
        assert_eq!(logged.source_post_id, Some(33));
        assert_eq!(logged.in_reply_to_post_id, Some(44));
        assert_eq!(logged.quoted_post_id, Some(55));
        assert_eq!(logged.quoted_user_id, Some(66));
        assert_eq!(logged.request_join_id, Some(123));
        assert_eq!(logged.request_time_ms, Some(1_700_000_000_000));
        assert_eq!(logged.is_in_network, Some(true));
        assert_eq!(logged.fav_count, Some(1));
        assert_eq!(logged.reply_count, Some(2));
        assert_eq!(logged.retweet_count, Some(3));
        assert_eq!(logged.quote_count, Some(4));
        assert_eq!(logged.has_media, Some(true));
        assert_eq!(logged.video_duration_ms, Some(1500));
        assert_eq!(
            logged.prediction_scores.get("phoenix.Prod.favorite"),
            Some(&0.5)
        );
        assert!(logged.language_code.is_some());
    }
}
