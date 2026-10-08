pub mod columns;
pub mod fav_counts;
pub mod features;
pub mod fetcher;
pub mod metrics;
pub mod post_metadata;
pub mod scheduler;
pub mod strato;
pub mod tweet_fields;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::processor::record::PostId;
use features::PostFeatures;

#[async_trait]
pub trait Hydrator: Send + Sync {
    async fn fetch(&self, post_id: PostId) -> Result<Value>;

    fn apply(&self, response: &Value, features: &mut PostFeatures);
}
