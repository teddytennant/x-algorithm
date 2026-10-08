use crate::models::candidate::{CandidateHelpers, PostCandidate};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    EnablePhoenixRetrievalStatsExperimentBucket, RetrievalCandidatesKafkaMaxCandidates,
    RetrievalCandidatesKafkaSamplePercent,
};
use crate::scorers::phoenix_scorer::PhoenixScorer;
use prost::Message;
use std::cmp::Ordering;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::KafkaPublisherClient;
use xai_candidate_pipeline::component_library::utils::{is_prod, is_sampled};
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto as pb;

pub struct RetrievalCandidatesKafkaSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl RetrievalCandidatesKafkaSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }
}

fn candidate_record(
    candidate: &PostCandidate,
    status: pb::RetrievalCandidateStatus,
    rank: u32,
) -> pb::RetrievalCandidateRecord {
    pb::RetrievalCandidateRecord {
        tweet_id: candidate.tweet_id,
        author_id: candidate.author_id,
        original_tweet_id: candidate.get_original_tweet_id(),
        original_author_id: candidate.get_original_author_id(),
        credited_served_type: candidate.served_type.map(i32::from),
        sources: candidate
            .retrieval_sources
            .iter()
            .map(|source| pb::RetrievalSourceMembership {
                served_type: source.served_type.into(),
                cluster: source.cluster.map(|cluster| format!("{cluster:?}")),
                score: source.score,
                position: source.position,
            })
            .collect(),
        score: candidate.score,
        status: status.into(),
        rank: Some(rank),
        weighted_score: candidate.weighted_score,
        in_network: candidate.in_network,
    }
}

fn build_batch(
    input: &SideEffectInput<ScoredPostsQuery, PostCandidate>,
) -> pb::RetrievalCandidateBatch {
    let query = &input.query;
    let mut non_selected: Vec<&PostCandidate> = input.non_selected_candidates.iter().collect();
    non_selected.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                b.weighted_score
                    .partial_cmp(&a.weighted_score)
                    .unwrap_or(Ordering::Equal)
            })
    });
    let max_candidates = query.params.get(RetrievalCandidatesKafkaMaxCandidates) as usize;
    let candidates = input
        .selected_candidates
        .iter()
        .map(|c| (c, pb::RetrievalCandidateStatus::RetrievalCandidateSelected))
        .chain(non_selected.into_iter().map(|c| {
            (
                c,
                pb::RetrievalCandidateStatus::RetrievalCandidateNotSelected,
            )
        }))
        .take(max_candidates)
        .enumerate()
        .map(|(index, (candidate, status))| candidate_record(candidate, status, index as u32 + 1))
        .collect();

    pb::RetrievalCandidateBatch {
        request_id: query.request_id,
        viewer_id: Some(query.user_id),
        request_time_ms: query.request_time_ms,
        sample_rate: query.params.get(RetrievalCandidatesKafkaSamplePercent) / 100.0,
        ranker_cluster: Some(format!("{:?}", PhoenixScorer::resolve_cluster(query))),
        experiment_buckets: query
            .params
            .experiment_buckets(EnablePhoenixRetrievalStatsExperimentBucket)
            .into_iter()
            .map(|bucket| pb::ExperimentBucket {
                experiment: bucket.experiment.clone(),
                bucket: bucket.bucket.clone(),
            })
            .collect(),
        candidates,
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, PostCandidate> for RetrievalCandidatesKafkaSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod()
            && !query.in_network_only
            && !query.has_cached_posts
            && is_sampled(
                query.request_id,
                query.params.get(RetrievalCandidatesKafkaSamplePercent),
            )
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, PostCandidate>>,
    ) -> Result<(), String> {
        let batch = build_batch(&input);
        if batch.candidates.is_empty() {
            return Ok(());
        }
        self.kafka_client
            .send(&batch.encode_to_vec())
            .await
            .map_err(|e| format!("Kafka publish failed: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::candidate::RetrievalSource;
    use pb::{RetrievalCandidateStatus, ServedType};
    use xai_candidate_pipeline::component_library::clients::phoenix_retrieval_client::PhoenixRetrievalCluster;

    fn post(tweet_id: u64, served_type: ServedType, score: Option<f64>) -> PostCandidate {
        PostCandidate {
            tweet_id,
            score,
            served_type: Some(served_type),
            retrieval_sources: vec![RetrievalSource::from_served_type(served_type)],
            ..Default::default()
        }
    }

    #[test]
    fn batch_covers_selected_and_non_selected_candidates() {
        let selected = PostCandidate {
            author_id: 10,
            retweeted_tweet_id: Some(100),
            retweeted_user_id: Some(1000),
            weighted_score: Some(1.2),
            in_network: Some(true),
            retrieval_sources: vec![
                RetrievalSource::from_served_type(ServedType::ForYouInNetwork),
                RetrievalSource {
                    served_type: ServedType::ForYouPhoenixRetrieval,
                    cluster: Some(PhoenixRetrievalCluster::Experiment1Fou),
                    score: Some(0.5),
                    position: Some(12),
                },
            ],
            ..post(1, ServedType::ForYouInNetwork, Some(0.9))
        };

        let batch = build_batch(&SideEffectInput {
            query: Arc::new(ScoredPostsQuery::default()),
            selected_candidates: vec![selected],
            non_selected_candidates: vec![post(2, ServedType::ForYouPhoenixRetrieval, None)],
        });

        let outcomes: Vec<(u64, i32, Option<u32>)> = batch
            .candidates
            .iter()
            .map(|c| (c.tweet_id, c.status, c.rank))
            .collect();
        assert_eq!(
            vec![
                (
                    1,
                    RetrievalCandidateStatus::RetrievalCandidateSelected.into(),
                    Some(1)
                ),
                (
                    2,
                    RetrievalCandidateStatus::RetrievalCandidateNotSelected.into(),
                    Some(2)
                ),
            ],
            outcomes
        );
        assert_eq!(
            pb::RetrievalCandidateRecord {
                tweet_id: 1,
                author_id: 10,
                original_tweet_id: 100,
                original_author_id: 1000,
                credited_served_type: Some(ServedType::ForYouInNetwork.into()),
                sources: vec![
                    pb::RetrievalSourceMembership {
                        served_type: ServedType::ForYouInNetwork.into(),
                        ..Default::default()
                    },
                    pb::RetrievalSourceMembership {
                        served_type: ServedType::ForYouPhoenixRetrieval.into(),
                        cluster: Some("Experiment1Fou".to_owned()),
                        score: Some(0.5),
                        position: Some(12),
                    },
                ],
                score: Some(0.9),
                status: RetrievalCandidateStatus::RetrievalCandidateSelected.into(),
                rank: Some(1),
                weighted_score: Some(1.2),
                in_network: Some(true),
            },
            batch.candidates[0]
        );
    }
}
