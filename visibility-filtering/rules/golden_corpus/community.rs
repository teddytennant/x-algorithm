use super::builders::on_client;
use super::{Role, Row};
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    ClientCapability, CommunityModeration, HydratedTweetCandidate, LimitedEngagementReason,
    TweetFeatures, Verdict, ViewerFeatures,
};
use crate::rules::fixtures::{
    allow, author_viewer, candidate, dropped, limited, viewer, VIEWER_ID,
};
use crate::rules::SafetyLevel::TimelineHomeHydration;
use std::num::NonZeroU64;
use xai_visibility_filtering::models::FilteredReason;

const HIDDEN: CommunityModeration = CommunityModeration {
    is_hidden: true,
    is_author_removed: false,
};

const AUTHOR_REMOVED: CommunityModeration = CommunityModeration {
    is_hidden: false,
    is_author_removed: true,
};

fn community_post(
    moderation: CommunityModeration,
    viewer_is_moderator: Option<bool>,
) -> HydratedTweetCandidate {
    let mut post = candidate()
        .with_tweet_features(TweetFeatures {
            community_id: NonZeroU64::new(500),
            ..Default::default()
        })
        .build();
    post.community_moderation = moderation;
    post.viewer_is_community_moderator = viewer_is_moderator;
    post
}

fn hidden_drop() -> Verdict {
    dropped(
        FilteredReason::UnspecifiedReason,
        "hidden_community_tweet/drop/unspecified",
    )
}

fn author_removed_drop() -> Verdict {
    dropped(
        FilteredReason::UnspecifiedReason,
        "author_removed_community_tweet/drop/unspecified",
    )
}

fn viewer_removed_post(viewer_is_removed: bool) -> HydratedTweetCandidate {
    HydratedTweetCandidate {
        viewer_is_removed_from_community: viewer_is_removed,
        ..community_post(CommunityModeration::default(), Some(false))
    }
}

fn viewer_removed_limit() -> Verdict {
    limited(
        LimitedEngagementReason::CommunityTweetViewerRemoved,
        "community_tweet_viewer_removed/limited_engagement",
    )
}

fn ios_before_viewer_removed_limits() -> Role {
    let ios = on_client("ios_current", "us", viewer(VIEWER_ID));
    Role::As(
        "ios_10_15",
        ViewerFeatures {
            client_capability: ClientCapability {
                community_viewer_removed_limits: false,
                ..ios.client_capability
            },
            ..ios
        },
    )
}

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "hidden_community_post",
            post: community_post(HIDDEN, Some(false)),
            expect: vec![
                (TimelineHomeHydration, Role::NonFollower, hidden_drop()),
                (TimelineHomeHydration, Role::Follower, hidden_drop()),
                (TimelineHomeHydration, Role::LoggedOut, hidden_drop()),
            ],
        },
        Row {
            name: "hidden_community_post_moderator_lookup_failed",
            post: community_post(HIDDEN, None),
            expect: vec![
                (TimelineHomeHydration, Role::NonFollower, allow()),
                (TimelineHomeHydration, Role::LoggedOut, hidden_drop()),
            ],
        },
        Row {
            name: "author_removed_community_post",
            post: community_post(AUTHOR_REMOVED, Some(false)),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    author_removed_drop(),
                ),
                (TimelineHomeHydration, Role::Follower, author_removed_drop()),
                (
                    TimelineHomeHydration,
                    Role::LoggedOut,
                    author_removed_drop(),
                ),
            ],
        },
        Row {
            name: "hidden_author_removed_community_post",
            post: community_post(
                CommunityModeration {
                    is_hidden: true,
                    is_author_removed: true,
                },
                Some(false),
            ),
            expect: vec![(TimelineHomeHydration, Role::NonFollower, hidden_drop())],
        },
        Row {
            name: "unmoderated_community_post",
            post: community_post(CommunityModeration::default(), None),
            expect: vec![
                (TimelineHomeHydration, Role::NonFollower, allow()),
                (TimelineHomeHydration, Role::LoggedOut, allow()),
            ],
        },
        Row {
            name: "community_post_viewer_removed",
            post: viewer_removed_post(true),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::As("web", on_client("web", "us", viewer(VIEWER_ID))),
                    viewer_removed_limit(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "ios_current",
                        on_client("ios_current", "us", viewer(VIEWER_ID)),
                    ),
                    viewer_removed_limit(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "android_current",
                        on_client("android_current", "us", viewer(VIEWER_ID)),
                    ),
                    viewer_removed_limit(),
                ),
                (
                    TimelineHomeHydration,
                    ios_before_viewer_removed_limits(),
                    allow(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As("mac_app", on_client("mac_app", "us", viewer(VIEWER_ID))),
                    allow(),
                ),
            ],
        },
        Row {
            name: "community_post_author_removed_from_its_community",
            post: viewer_removed_post(true),
            expect: vec![(
                TimelineHomeHydration,
                Role::As("author_on_web", on_client("web", "us", author_viewer())),
                viewer_removed_limit(),
            )],
        },
        Row {
            name: "community_post_viewer_not_removed",
            post: viewer_removed_post(false),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::As("web", on_client("web", "us", viewer(VIEWER_ID))),
                    allow(),
                ),
                (
                    TimelineHomeHydration,
                    Role::As(
                        "ios_current",
                        on_client("ios_current", "us", viewer(VIEWER_ID)),
                    ),
                    allow(),
                ),
                (TimelineHomeHydration, Role::LoggedOut, allow()),
            ],
        },
        Row {
            name: "community_post_viewer_removal_lookup_failed",
            post: HydratedTweetCandidate {
                failed: Hydrators::of(Hydrator::CommunityViewerRemoved),
                ..viewer_removed_post(false)
            },
            expect: vec![(
                TimelineHomeHydration,
                Role::As("web", on_client("web", "us", viewer(VIEWER_ID))),
                allow(),
            )],
        },
    ]
}
