use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use super::features::PostFeatures;
use super::strato::{json_i64, StratoColumn};
use super::Hydrator;
use crate::processor::record::PostId;

pub const COLUMN: &str = "hydra/phoenix_rank_all/getPostPhoenixRankAllMetadata";

pub struct PostMetadata {
    column: StratoColumn,
}

impl PostMetadata {
    pub fn new(column: StratoColumn) -> Self {
        Self { column }
    }
}

#[async_trait]
impl Hydrator for PostMetadata {
    async fn fetch(&self, post_id: PostId) -> Result<Value> {
        self.column.execute(&post_id).await
    }

    fn apply(&self, response: &Value, features: &mut PostFeatures) {
        apply_metadata(response, features);
    }
}

fn apply_metadata(metadata: &Value, features: &mut PostFeatures) {
    let engagement = &metadata["engagementCount"];
    let count = |field: &str| json_i64(&engagement[field]).max(0);
    features.author_id = json_i64(&metadata["authorId"]);
    features.fav_count = count("favoriteCount");
    features.reply_count = count("replyCount");
    features.repost_count = count("retweetCount");
    features.quote_count = count("quoteCount");
    features.bookmark_count = count("bookmarkCount");
    features.view_count = count("viewCount");
    features.author_followers_count = json_i64(&metadata["authorFollowersCount"]).max(0);
    features.has_image = metadata["hasImage"].as_bool().unwrap_or(false);
    features.has_video = metadata["hasVideo"].as_bool().unwrap_or(false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_response() {
        let response = json!({
            "postId": "101",
            "authorId": "202",
            "hasVideo": false,
            "hasImage": true,
            "authorFollowersCount": "1000",
            "engagementCount": {
                "retweetCount": "5",
                "replyCount": "4",
                "favoriteCount": "12",
                "quoteCount": "3",
                "bookmarkCount": "2",
                "viewCount": "300",
                "notInterestedInCount": "1",
                "blockCount": "1"
            }
        });
        let mut features = PostFeatures::stamped(7);
        apply_metadata(&response, &mut features);
        assert_eq!(features.author_id, 202);
        assert_eq!(features.fav_count, 12);
        assert_eq!(features.reply_count, 4);
        assert_eq!(features.repost_count, 5);
        assert_eq!(features.quote_count, 3);
        assert_eq!(features.bookmark_count, 2);
        assert_eq!(features.view_count, 300);
        assert_eq!(features.author_followers_count, 1000);
        assert!(features.has_image);
        assert!(!features.has_video);
        assert_eq!(features.features_ts, 7);
    }

    #[test]
    fn missing_post_keeps_stamped_defaults() {
        let mut features = PostFeatures::stamped(7);
        apply_metadata(&Value::Null, &mut features);
        assert_eq!(features, PostFeatures::stamped(7));
    }
}
