pub mod author;
pub mod conversation_control;
pub mod region;
pub mod safety_labels;
pub mod tweet;
pub mod verdict;
pub mod viewer;

pub use author::{AuthorFeatures, AuthorLabel, AuthorLabelSet};
pub use conversation_control::ConversationControlFeatures;
pub use safety_labels::{SafetyLabelMap, SafetyLabelType};
pub use tweet::{ArticleLifecycle, CommunityModeration, MediaFeature, NsfwFeature, TweetFeatures};
pub use verdict::{
    Decided, DropReason, Evaluation, FosnrReason, LimitedEngagement, LimitedEngagementReason,
    MediaInterstitial, MediaRestriction, Notice, NsfwViewerDropReason, TombstoneReason, Verdict,
    Withholding,
};
pub use viewer::{
    ClientCapability, VerifyBlurSupport, Viewer, ViewerAge, ViewerFeatures, ViewerProfile,
};

use crate::hydration::Hydrators;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TweetId(pub u64);

pub fn tweet_timestamp_ms(tweet_id: u64) -> u64 {
    const SNOWFLAKE_EPOCH_MS: u64 = 1288834974657;
    const SNOWFLAKE_TIMESTAMP_SHIFT: u32 = 22;
    (tweet_id >> SNOWFLAKE_TIMESTAMP_SHIFT) + SNOWFLAKE_EPOCH_MS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AuthorId(pub u64);

impl AuthorId {
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PureCore {
    pub author_id: AuthorId,
    pub source_tweet_id: Option<TweetId>,
    pub source_author_id: Option<AuthorId>,
    pub direct_reply_root_author_id: Option<AuthorId>,
}

#[derive(Clone, Copy, Debug)]
pub struct RawCandidate {
    pub tweet_id: TweetId,
    pub request_author_id: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct HydratedTweetCandidate {
    pub tweet_id: u64,
    pub author_id: u64,
    pub source_tweet_id: Option<u64>,
    pub tweet_features: TweetFeatures,
    pub author_features: AuthorFeatures,
    pub author_labels: AuthorLabelSet,
    pub safety_labels: SafetyLabelMap,
    pub edges: Hydrators,
    pub conversation_control: Option<ConversationControlFeatures>,
    pub community_moderation: CommunityModeration,
    pub viewer_is_community_moderator: Option<bool>,
    pub viewer_is_removed_from_community: bool,
    pub article_lifecycle: Option<ArticleLifecycle>,
    pub failed: Hydrators,
}
