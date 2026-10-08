use super::builders::author_candidate;
use super::{Role, Row};
use crate::models::{AuthorFeatures, SafetyLabelType};
use crate::rules::fixtures::{allow, candidate, dropped};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "suspended_author",
            post: author_candidate(|a| a.is_suspended = true),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Follower,
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
                (TimelineHomeHydration, Role::Author, allow()),
            ],
        },
        Row {
            name: "suspended_author_nsfw_media",
            post: candidate()
                .with_author_features(AuthorFeatures {
                    is_suspended: true,
                    ..Default::default()
                })
                .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                .with_media()
                .build(),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
            ],
        },
        Row {
            name: "suspended_author_bounced_post",
            post: candidate()
                .with_author_features(AuthorFeatures {
                    is_suspended: true,
                    ..Default::default()
                })
                .with_label(SafetyLabelType::BOUNCE)
                .build(),
            expect: vec![(
                TimelineHomeHydration,
                Role::NonFollower,
                dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
            )],
        },
        Row {
            name: "deactivated_author",
            post: author_candidate(|a| a.is_deactivated = true),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::AuthorIsDeactivated,
                        "deactivated_author/drop",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::AuthorIsDeactivated,
                        "deactivated_author/drop",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Follower,
                    dropped(
                        FilteredReason::AuthorIsDeactivated,
                        "deactivated_author/drop",
                    ),
                ),
            ],
        },
        Row {
            name: "suspended_and_deactivated_author",
            post: author_candidate(|a| {
                a.is_suspended = true;
                a.is_deactivated = true;
            }),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsSuspended, "suspended_author/drop"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::AuthorIsDeactivated,
                        "deactivated_author/drop",
                    ),
                ),
            ],
        },
        Row {
            name: "erased_author",
            post: author_candidate(|a| a.is_erased = true),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "erased_author/drop/inactive",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "erased_author/drop/inactive",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Follower,
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "erased_author/drop/inactive",
                    ),
                ),
            ],
        },
        Row {
            name: "offboarded_author",
            post: author_candidate(|a| a.is_offboarded = true),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "offboarded_author/drop/inactive",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "offboarded_author/drop/inactive",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Follower,
                    dropped(
                        FilteredReason::AuthorAccountIsInactive,
                        "offboarded_author/drop/inactive",
                    ),
                ),
            ],
        },
        Row {
            name: "protected_author",
            post: author_candidate(|a| a.is_protected = true),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
                ),
                (TimelineHome, Role::Follower, allow()),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHome,
                    Role::LoggedOut,
                    dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
                ),
                (TimelineHomeHydration, Role::Follower, allow()),
                (TimelineHomeHydration, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::LoggedOut,
                    dropped(FilteredReason::AuthorIsProtected, "protected_author/drop"),
                ),
            ],
        },
    ]
}
