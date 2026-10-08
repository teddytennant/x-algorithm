use crate::models::query::{RequestType, ScoredPostsQuery};
use crate::params::EnableUrtMigrationComponents;
use std::sync::Arc;
use thrift::OrderedFloat;
use tonic::async_trait;
use xai_candidate_pipeline::component_library::clients::kafka_publisher_client::{
    KafkaCluster, KafkaPublisherClient, ProdKafkaPublisherClient, SERVED_CANDIDATES_TOPIC,
};
use xai_candidate_pipeline::component_library::utils::client_utils::RequestContext;
use xai_candidate_pipeline::component_library::utils::is_prod;
use xai_candidate_pipeline::side_effect::{SideEffect, SideEffectInput};
use xai_home_mixer_proto::{feed_item, FeedItem};
use xai_x_thrift::serialize_compact;
use xai_x_thrift::timeline_logging::{
    EntryInfo, EntryType, ItemDetails, PromotedTweetDetails, QueryType, RequestInfo,
    RequestProvenance, ServedEntry, TimelineType, TweetDetails, WhoToFollowDetails,
};

pub struct ServedCandidatesKafkaSideEffect {
    kafka_client: Arc<dyn KafkaPublisherClient>,
}

impl ServedCandidatesKafkaSideEffect {
    pub fn new(kafka_client: Arc<dyn KafkaPublisherClient>) -> Self {
        Self { kafka_client }
    }

    pub async fn prod() -> Self {
        Self::new(Arc::new(
            ProdKafkaPublisherClient::new(SERVED_CANDIDATES_TOPIC, KafkaCluster::Phoenix).await,
        ))
    }
}

#[async_trait]
impl SideEffect<ScoredPostsQuery, FeedItem> for ServedCandidatesKafkaSideEffect {
    fn enable(&self, query: Arc<ScoredPostsQuery>) -> bool {
        is_prod() && query.is_shadow_traffic && query.params.get(EnableUrtMigrationComponents)
    }

    async fn side_effect(
        &self,
        input: Arc<SideEffectInput<ScoredPostsQuery, FeedItem>>,
    ) -> Result<(), String> {
        let query = &input.query;
        let items = &input.selected_candidates;
        if items.is_empty() {
            return Ok(());
        }

        let now_ms = query.request_time_ms;

        let request_provenance = match RequestContext::parse(&query.request_context) {
            RequestContext::Foreground => Some(Box::new(RequestProvenance::FOREGROUND)),
            RequestContext::Launch => Some(Box::new(RequestProvenance::LAUNCH)),
            RequestContext::PullToRefresh => Some(Box::new(RequestProvenance::PTR)),
            RequestContext::ViewportAwareRefresh => {
                Some(Box::new(RequestProvenance::VIEWPORT_AWARE_REFRESH))
            }
            _ => Some(Box::new(RequestProvenance::OTHER)),
        };

        let query_type = if query.is_bottom_request {
            Some(Box::new(QueryType::GET_OLDER))
        } else if query.is_top_request {
            Some(Box::new(QueryType::GET_NEWER))
        } else if query.cursor.is_none() {
            Some(Box::new(QueryType::GET_INITIAL))
        } else {
            Some(Box::new(QueryType::OTHER))
        };

        let timeline_type = match query.request_type {
            RequestType::RankedFollowing => TimelineType::HOME_RANKED_FOLLOWING,
            RequestType::Following => TimelineType::HOME_LATEST,
            _ => TimelineType::HOME,
        };

        let request_info = Box::new(RequestInfo {
            request_time_ms: now_ms,
            trace_id: query.request_id as i64,
            user_id: Some(query.user_id as i64),
            client_app_id: Some(query.client_app_id as i64),
            timeline_type: Some(Box::new(timeline_type)),
            ip_address: Some(query.ip_address.clone()),
            user_agent: Some(query.user_agent.clone()),
            query_type,
            request_provenance,
            language_code: Some(query.language_code.clone()),
            country_code: Some(query.country_code.clone()),
            request_end_time_ms: Some(now_ms),
            request_join_id: Some(query.request_id as i64),
        });

        let mut payloads = Vec::with_capacity(items.len());
        for item in items {
            if let Some(entry_info) = build_entry_info(item) {
                let served = ServedEntry {
                    request: request_info.clone(),
                    entry: Some(Box::new(entry_info)),
                };
                let bytes = serialize_compact(&served)
                    .map_err(|e| format!("Thrift serialization failed: {e}"))?;
                payloads.push(bytes);
            }
        }

        let futs: Vec<_> = payloads
            .iter()
            .map(|bytes| self.kafka_client.send(bytes))
            .collect();
        let results = futures::future::join_all(futs).await;

        if let Some(err) = results.into_iter().find_map(|r| r.err()) {
            return Err(format!("Served-candidates Kafka publish failed: {err}"));
        }

        Ok(())
    }
}

fn build_entry_info(item: &FeedItem) -> Option<EntryInfo> {
    match &item.item {
        Some(feed_item::Item::Post(post)) => {
            let source_tweet_id = if post.retweeted_tweet_id != 0 {
                Some(post.retweeted_tweet_id as i64)
            } else {
                None
            };
            Some(EntryInfo {
                id: post.tweet_id as i64,
                position: item.position as i16,
                entry_id: format!("tweet-{}", post.tweet_id),
                entry_type: Box::new(EntryType::TWEET),
                vertical_size: Some(1),
                sort_index: None,
                display_type: None,
                score: Some(OrderedFloat::from(post.score as f64)),
                details: Some(Box::new(ItemDetails::TweetDetails(Box::new(
                    TweetDetails {
                        source_tweet_id,
                        author_id: Some(post.author_id as i64),
                    },
                )))),
                prediction_scores: None,
            })
        }
        Some(feed_item::Item::Ad(ad)) => Some(EntryInfo {
            id: ad.post_id,
            position: item.position as i16,
            entry_id: format!("promotedTweet-{}", ad.post_id),
            entry_type: Box::new(EntryType::PROMOTED_TWEET),
            vertical_size: Some(1),
            sort_index: None,
            display_type: None,
            score: None,
            details: Some(Box::new(ItemDetails::PromotedTweetDetails(Box::new(
                PromotedTweetDetails {
                    advertiser_id: Some(ad.account_id),
                    insert_position: Some(ad.insert_position),
                    impression_id: Some(ad.impression_id.to_string()),
                },
            )))),
            prediction_scores: None,
        }),
        Some(feed_item::Item::WhoToFollow(_)) => Some(EntryInfo {
            id: 0,
            position: item.position as i16,
            entry_id: "who-to-follow".to_string(),
            entry_type: Box::new(EntryType::WHO_TO_FOLLOW_MODULE),
            vertical_size: None,
            sort_index: None,
            display_type: None,
            score: None,
            details: Some(Box::new(ItemDetails::WhoToFollowDetails(Box::new(
                WhoToFollowDetails {
                    advertiser_id: None,
                },
            )))),
            prediction_scores: None,
        }),
        Some(feed_item::Item::PushToHome(pth)) => Some(EntryInfo {
            id: pth.tweet_id as i64,
            position: item.position as i16,
            entry_id: format!("tweet-{}", pth.tweet_id),
            entry_type: Box::new(EntryType::TWEET),
            vertical_size: Some(1),
            sort_index: None,
            display_type: None,
            score: None,
            details: Some(Box::new(ItemDetails::TweetDetails(Box::new(
                TweetDetails {
                    source_tweet_id: None,
                    author_id: Some(pth.author_id as i64),
                },
            )))),
            prediction_scores: None,
        }),
        Some(feed_item::Item::Prompt(_)) => Some(EntryInfo {
            id: 0,
            position: item.position as i16,
            entry_id: "prompt".to_string(),
            entry_type: Box::new(EntryType::PROMPT),
            vertical_size: None,
            sort_index: None,
            display_type: None,
            score: None,
            details: None,
            prediction_scores: None,
        }),
        Some(feed_item::Item::Frame(_)) => None,
        Some(feed_item::Item::FeedSurvey(_)) => None,
        Some(feed_item::Item::VideoCarousel(_)) => None,
        None => None,
    }
}
