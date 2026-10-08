pub mod batch;
pub(crate) mod community_source;
mod decode;
mod execute;
pub(crate) mod fallback_cache;
mod fetcher;
pub mod metrics;
pub(crate) mod plan;
pub(crate) mod sources;
mod store;
pub mod tweet_source;

use crate::models::{
    ClientCapability, HydratedTweetCandidate, RawCandidate, TweetId, ViewerFeatures,
};
pub(crate) use decode::author::{AuthorFallbackCache, fallback_cache as author_fallback_cache};
pub(crate) use decode::tweet::{TweetFallbackCache, tweet_fallback_cache};
pub(crate) use plan::HydrationPlan;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) const HYDRATION_TIMEOUT: Duration = Duration::from_secs(1);
pub(crate) const INBOUND_ALLOWANCE: Duration = Duration::from_millis(10);

pub(crate) fn request_context(
    entered: tokio::time::Instant,
    grpc_timeout: Option<Duration>,
) -> xai_x_rpc::CallContext {
    xai_x_rpc::CallContext {
        deadline: Some(
            entered
                + grpc_timeout
                    .unwrap_or(HYDRATION_TIMEOUT)
                    .saturating_sub(INBOUND_ALLOWANCE),
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
#[repr(u8)]
pub enum Hydrator {
    PureCore,
    Tweet,
    ConversationControl,
    TweetSafetyLabels,
    ViewerProfile,
    ViewerLabels,
    AuthorSafety,
    AuthorLabels,
    Follows,
    Blocks,
    Mutes,
    MuteRetweets,
    BlockedByAuthor,
    BlockedByReplyRoot,
    SuperFollowsExclusive,
    RootFollowsViewer,
    RootFollowsViewerSecondDegree,
    SuperFollowsRoot,
    ViewerCountry,
    CommunityModeration,
    CommunityModerator,
    CommunityViewerRemoved,
    ArticleLifecycle,
    TrustedFriends,
    OutsideNarrowcastPlace,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Hydrators(u32);

impl Hydrators {
    pub const fn empty() -> Self {
        Self(0)
    }

    #[cfg(test)]
    pub fn all() -> Self {
        Self(u32::MAX >> (u32::BITS as usize - <Hydrator as strum::VariantArray>::VARIANTS.len()))
    }

    pub const fn of(hydrator: Hydrator) -> Self {
        Self(1 << hydrator as u8)
    }

    pub const fn with(self, hydrator: Hydrator) -> Self {
        self.union(Self::of(hydrator))
    }

    #[cfg(test)]
    pub const fn without(self, hydrator: Hydrator) -> Self {
        Self(self.0 & !Self::of(hydrator).0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, hydrator: Hydrator) -> bool {
        self.0 & Self::of(hydrator).0 != 0
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

pub(crate) struct HydrationRequest<'a> {
    viewer_id: Option<u64>,
    country_code: Option<String>,
    client_capability: ClientCapability,
    raw_candidates: &'a [RawCandidate],
    is_expanding_retweet_sources: bool,
}

impl<'a> HydrationRequest<'a> {
    pub(crate) fn new(
        viewer_id: Option<u64>,
        country_code: Option<String>,
        client_capability: ClientCapability,
        raw_candidates: &'a [RawCandidate],
    ) -> Self {
        Self {
            viewer_id,
            country_code,
            client_capability,
            raw_candidates,
            is_expanding_retweet_sources: false,
        }
    }

    pub(crate) fn with_retweet_sources(self, is_expanding_retweet_sources: bool) -> Self {
        Self {
            is_expanding_retweet_sources,
            ..self
        }
    }
}

pub(crate) fn candidate_count_by_key<K: Eq + Hash>(
    keys: impl Iterator<Item = K>,
) -> FxHashMap<K, usize> {
    let mut candidate_count_by_key =
        FxHashMap::with_capacity_and_hasher(keys.size_hint().0, Default::default());
    for key in keys {
        *candidate_count_by_key.entry(key).or_default() += 1;
    }
    candidate_count_by_key
}

pub(crate) struct Hydration {
    viewer: ViewerFeatures,
    tweets: FxHashMap<TweetId, HydratedTweet>,
    has_fetched_sources: bool,
}

impl Hydration {
    pub(crate) fn viewer(&self) -> &ViewerFeatures {
        &self.viewer
    }

    pub(crate) fn tweet(&self, id: TweetId) -> Option<&HydratedTweet> {
        self.tweets.get(&id)
    }

    pub(crate) fn has_fetched_sources(&self) -> bool {
        self.has_fetched_sources
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "nearly every tweet resolves, so a box would cost an allocation per tweet"
)]
pub(crate) enum HydratedTweet {
    Resolved {
        candidate: HydratedTweetCandidate,
        has_failed_node: bool,
        source_tweet_id: Option<TweetId>,
        safety_labels: Option<Arc<vf_pb::SafetyLabelMap>>,
    },
    Unresolved {
        reason: Unresolved,
        safety_labels: Option<Arc<vf_pb::SafetyLabelMap>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Unresolved {
    pub(crate) lookup: Lookup,
    pub(crate) cause: Cause,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Lookup {
    Tweet,
    Author,
    SharedTweet,
    SharedAuthor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Cause {
    NotFound,
    Failed,
}

impl HydratedTweet {
    pub(crate) fn candidate(&self) -> Option<&HydratedTweetCandidate> {
        match self {
            Self::Resolved { candidate, .. } => Some(candidate),
            Self::Unresolved { .. } => None,
        }
    }

    pub(crate) fn source_tweet_id(&self) -> Option<TweetId> {
        match self {
            Self::Resolved {
                source_tweet_id, ..
            } => *source_tweet_id,
            Self::Unresolved { .. } => None,
        }
    }

    pub(crate) fn source_to_merge(&self) -> Option<TweetId> {
        match self {
            Self::Resolved {
                has_failed_node: false,
                source_tweet_id,
                ..
            } => *source_tweet_id,
            Self::Resolved {
                has_failed_node: true,
                ..
            }
            | Self::Unresolved { .. } => None,
        }
    }

    pub(crate) fn safety_labels(&self) -> Option<&Arc<vf_pb::SafetyLabelMap>> {
        match self {
            Self::Resolved { safety_labels, .. } | Self::Unresolved { safety_labels, .. } => {
                safety_labels.as_ref()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_context_budget_is_header_or_hang_guard_minus_allowance() {
        let entered = tokio::time::Instant::now();

        let header = Duration::from_millis(400);
        assert_eq!(
            request_context(entered, Some(header)).deadline,
            Some(entered + header - INBOUND_ALLOWANCE)
        );
        assert_eq!(
            request_context(entered, None).deadline,
            Some(entered + HYDRATION_TIMEOUT - INBOUND_ALLOWANCE)
        );
    }
}
