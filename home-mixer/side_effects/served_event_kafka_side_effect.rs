use crate::models::query::ScoredPostsQuery;
use crate::params::EnableUrtMigrationComponents;
use crate::side_effects::client_events_kafka_side_effect::build_served_log_events;
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::client_event::publish_served_events;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaCluster, KafkaPublisherClient, ProdKafkaPublisherClient, XAI_SERVED_EVENT_TOPIC,
};
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::FeedItem;
use xai_proto::served_event::ServedEvent;

const ENABLE_HOME_SERVED_EVENT_KAFKA: &str = "enable_home_served_event_kafka";

pub struct ServedEventKafkaSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl ServedEventKafkaSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }

    pub async fn prod() -> Self {
        Self::new(Arc::new(
            ProdKafkaPublisherClient::new(XAI_SERVED_EVENT_TOPIC, KafkaCluster::CoreData).await,
        ))
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for ServedEventKafkaSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod()
            && query.params.get(EnableUrtMigrationComponents)
            && query
                .decider
                .as_ref()
                .is_some_and(|d| d.enabled(ENABLE_HOME_SERVED_EVENT_KAFKA))
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let base = ServedEvent {
            request_id: query.request_id as i64,
            prediction_id: query.prediction_id as i64,
            user_id: query.user_id as i64,
            client_app_id: query.client_app_id.into(),
            request_time_ms: query.request_time_ms,
            producer: "home_mixer".into(),
            ..Default::default()
        };
        publish_served_events(
            Arc::clone(&self.kafka_client),
            &base,
            &build_served_log_events(query, &input.selected_candidates),
        )
        .await
    }
}
