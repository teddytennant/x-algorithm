use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

use super::features::{Language, PostFeatures};
use super::strato::StratoColumn;
use super::Hydrator;
use crate::processor::record::PostId;

pub const COLUMN: &str = "tweetypie/getTweetFields.Tweet";

const CORE_DATA_FIELD_ID: i32 = 2;
const QUOTED_TWEET_FIELD_ID: i32 = 11;
const LANGUAGE_FIELD_ID: i32 = 18;

pub struct TweetFields {
    column: StratoColumn,
    view: Value,
}

impl TweetFields {
    pub fn new(column: StratoColumn) -> Self {
        let includes: Vec<Value> = [CORE_DATA_FIELD_ID, QUOTED_TWEET_FIELD_ID, LANGUAGE_FIELD_ID]
            .into_iter()
            .map(|id| json!({ "tweetFieldId": id }))
            .collect();
        Self {
            column,
            view: json!({ "tweetIncludes": includes }),
        }
    }
}

#[async_trait]
impl Hydrator for TweetFields {
    async fn fetch(&self, post_id: PostId) -> Result<Value> {
        let mut response = self.column.fetch(&post_id, &self.view).await?;
        Ok(response["v"].take())
    }

    fn apply(&self, response: &Value, features: &mut PostFeatures) {
        apply_tweet_fields(response, features);
    }
}

fn apply_tweet_fields(tweet_fields: &Value, features: &mut PostFeatures) {
    let tweet = &tweet_fields["tweetResult"]["found"]["tweet"];
    let core = &tweet["coreData"];
    features.is_reply = !core["reply"].is_null();
    features.is_quote = !tweet["quotedTweet"].is_null();
    features.has_media = core["hasMedia"]
        .as_bool()
        .unwrap_or(features.has_image || features.has_video);
    features.author_nsfw_user = core["nsfwUser"].as_bool().unwrap_or(false);
    features.author_nsfw_admin = core["nsfwAdmin"].as_bool().unwrap_or(false);
    features.language = Language::new(tweet["language"]["language"].as_str().unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_found_tweet() {
        let response = json!({
            "tweetId": "101",
            "tweetResult": {"found": {"tweet": {
                "id": "101",
                "coreData": {
                    "user": "202",
                    "createdAtSecs": "1700000000",
                    "hasTakedown": false,
                    "nsfwUser": false,
                    "nsfwAdmin": true,
                    "nullcast": false,
                    "conversation": "101",
                    "hasMedia": true
                },
                "language": {"language": "en", "rightToLeft": false, "confidence": 0.66}
            }}}
        });
        let mut features = PostFeatures::default();
        apply_tweet_fields(&response, &mut features);
        assert!(!features.is_reply);
        assert!(!features.is_quote);
        assert!(features.has_media);
        assert!(!features.author_nsfw_user);
        assert!(features.author_nsfw_admin);
        assert_eq!(features.language.as_str(), "en");
    }

    #[test]
    fn reply_and_quote_presence() {
        let response = json!({"tweetResult": {"found": {"tweet": {
            "coreData": {"reply": {"inReplyToUserId": "1"}},
            "quotedTweet": {"tweetId": "2", "userId": "3"}
        }}}});
        let mut features = PostFeatures::default();
        apply_tweet_fields(&response, &mut features);
        assert!(features.is_reply);
        assert!(features.is_quote);
    }

    #[test]
    fn not_found_falls_back_to_metadata_media() {
        let response = json!({
            "tweetId": "103",
            "tweetResult": {"notFound": {"deleted": false, "bounceDeleted": false}}
        });
        let mut features = PostFeatures {
            has_video: true,
            ..PostFeatures::default()
        };
        apply_tweet_fields(&response, &mut features);
        assert!(features.has_media);
        assert!(!features.is_reply);
        assert_eq!(features.language.as_str(), "");
    }
}
