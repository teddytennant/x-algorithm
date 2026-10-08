use super::{Role, Row};
use crate::models::{AuthorLabel, HydratedTweetCandidate};
use crate::rules::fixtures::{allow, candidate, dropped};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeRecommendations};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "nsfw_high_recall_user_label",
            post: user_labeled(AuthorLabel::NsfwHighRecall),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (TimelineHome, Role::NonFollower, allow()),
            ],
        },
        Row {
            name: "nsfw_high_precision_user_label",
            post: user_labeled(AuthorLabel::NsfwHighPrecision),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_high_precision_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_high_precision_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "spam_high_recall_user_label",
            post: user_labeled(AuthorLabel::SpamHighRecall),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "spam_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "spam_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "compromised_user_label",
            post: user_labeled(AuthorLabel::Compromised),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "compromised_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "compromised_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "read_only_user_label",
            post: user_labeled(AuthorLabel::ReadOnly),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "read_only_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "read_only_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "impersonation_user_label",
            post: user_labeled(AuthorLabel::ImpersonationHighPrecision),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "impersonation_high_precision_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "impersonation_high_precision_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "nsfw_avatar_user_label",
            post: user_labeled(AuthorLabel::NsfwAvatarImage),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_avatar_image_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_avatar_image_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "nsfw_banner_user_label",
            post: user_labeled(AuthorLabel::NsfwBannerImage),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_banner_image_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_banner_image_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "abusive_high_recall_user_label",
            post: user_labeled(AuthorLabel::AbusiveHighRecall),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "abusive_high_recall_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Follower, allow()),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "nsfw_near_perfect_user_label",
            post: user_labeled(AuthorLabel::NsfwNearPerfect),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_near_perfect_user_label/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "nsfw_near_perfect_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (TimelineHome, Role::NonFollower, allow()),
            ],
        },
        Row {
            name: "do_not_amplify_user_label",
            post: user_labeled(AuthorLabel::DoNotAmplify),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "do_not_amplify_user_label/drop/unspecified",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Follower, allow()),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
    ]
}

fn user_labeled(label: AuthorLabel) -> HydratedTweetCandidate {
    candidate().with_author_user_label(label).build()
}
