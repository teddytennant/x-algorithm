use super::builders::tweet_candidate;
use super::{Role, Row};
use crate::models::{
    AuthorFeatures, ClientCapability, HydratedTweetCandidate, LimitedEngagementReason,
    TweetFeatures, ViewerFeatures,
};
use crate::rules::fixtures::{allow, candidate, dropped, limited, viewer, VIEWER_ID};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use std::num::NonZeroU64;
use xai_core_entities::entities::{EditControl, EditControlInitial};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "nullcast",
            post: tweet_candidate(|t| t.is_nullcast = true),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::TweetIsNullcast, "nullcasted_tweet/drop"),
                ),
                (
                    TimelineHome,
                    Role::Author,
                    dropped(FilteredReason::TweetIsNullcast, "nullcasted_tweet/drop"),
                ),
            ],
        },
        Row {
            name: "nullcast_retweet",
            post: {
                let features = TweetFeatures {
                    is_nullcast: true,
                    ..Default::default()
                };
                candidate()
                    .with_tweet_features(features)
                    .retweet_of(2)
                    .build()
            },
            expect: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "nullcast_community",
            post: tweet_candidate(|t| {
                t.is_nullcast = true;
                t.community_id = NonZeroU64::new(500);
            }),
            expect: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "protected_author_community_post",
            post: candidate()
                .with_author_features(AuthorFeatures {
                    is_protected: true,
                    ..Default::default()
                })
                .with_tweet_features(TweetFeatures {
                    community_id: NonZeroU64::new(500),
                    ..Default::default()
                })
                .build(),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "protected_community_tweet/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
                ),
                (TimelineHomeHydration, Role::Author, allow()),
            ],
        },
        Row {
            name: "stale_edit",
            post: stale_candidate(),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "stale_tweet/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    limited(
                        LimitedEngagementReason::StaleTweet,
                        "stale_tweet/limited_engagement",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::Author,
                    limited(
                        LimitedEngagementReason::StaleTweet,
                        "stale_tweet/limited_engagement",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "client_without_stale_tweet_limits",
                        ViewerFeatures {
                            client_capability: ClientCapability {
                                stale_tweet_limits: false,
                                ..ClientCapability::default()
                            },
                            ..viewer(VIEWER_ID)
                        },
                    ),
                    allow(),
                ),
            ],
        },
        Row {
            name: "stale_edit_retweet",
            post: {
                let mut retweet = stale_candidate();
                retweet.source_tweet_id = Some(2);
                retweet
            },
            expect: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "current_edit",
            post: tweet_candidate(|t| {
                t.edit_control = Some(EditControl::Initial(EditControlInitial {
                    edit_tweet_ids: vec![1],
                    ..Default::default()
                }))
            }),
            expect: vec![(TimelineHome, Role::NonFollower, allow())],
        },
    ]
}

fn stale_candidate() -> HydratedTweetCandidate {
    tweet_candidate(|t| {
        t.edit_control = Some(EditControl::Initial(EditControlInitial {
            edit_tweet_ids: vec![1, 2],
            ..Default::default()
        }))
    })
}
