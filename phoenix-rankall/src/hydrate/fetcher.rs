use std::sync::Arc;

use anyhow::Result;
use futures::future::try_join_all;

use super::features::PostFeatures;
use super::metrics::HydrateMetrics;
use super::post_metadata::{self, PostMetadata};
use super::strato::{CallLimiter, StratoColumn};
use super::tweet_fields::{self, TweetFields};
use super::Hydrator;
use crate::processor::record::PostId;

pub struct PostFeatureFetcher {
    hydrators: Vec<Box<dyn Hydrator>>,
}

impl PostFeatureFetcher {
    pub fn new(hydrators: Vec<Box<dyn Hydrator>>) -> Self {
        Self { hydrators }
    }

    pub fn standard(
        endpoint: &str,
        client: &reqwest::Client,
        limiter: &CallLimiter,
        metrics: &HydrateMetrics,
    ) -> Self {
        let column = |name: &str| {
            StratoColumn::new(
                name,
                endpoint,
                client.clone(),
                Arc::clone(limiter),
                metrics.clone(),
            )
        };
        Self::new(vec![
            Box::new(PostMetadata::new(column(post_metadata::COLUMN))),
            Box::new(TweetFields::new(column(tweet_fields::COLUMN))),
        ])
    }

    pub async fn fetch(&self, post_id: PostId) -> Result<PostFeatures> {
        let responses = try_join_all(self.hydrators.iter().map(|h| h.fetch(post_id))).await?;
        let mut features = PostFeatures::stamped(chrono::Utc::now().timestamp());
        for (hydrator, response) in self.hydrators.iter().zip(&responses) {
            hydrator.apply(response, &mut features);
        }
        Ok(features)
    }
}
