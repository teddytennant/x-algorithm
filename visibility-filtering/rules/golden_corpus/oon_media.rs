use super::builders::{author_candidate, tweet_candidate, viewer_in_country, viewer_with_age};
use super::{Role, Row};
use crate::models::{HydratedTweetCandidate, ViewerAge, ViewerFeatures};
use crate::rules::fixtures::{allow, author_viewer, dropped};
use crate::rules::SafetyLevel::{
    ImmersiveExpandedRecommendations, TimelineHome, TimelineHomeRecommendations,
};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "dmca_media",
            post: tweet_candidate(|t| t.media.has_dmca_media = true),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "dmca_media/drop/unspecified",
                    ),
                ),
                (TimelineHome, Role::NonFollower, allow()),
            ],
        },
        Row {
            name: "nsfw_user_flag",
            post: tweet_candidate(|t| t.nsfw.user = true),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_user_tweet_flag/drop/nsfw_media",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Author,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_user_tweet_flag/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_admin_flag",
            post: tweet_candidate(|t| t.nsfw.admin = true),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_admin_tweet_flag/drop/nsfw_media",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Author,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_admin_tweet_flag/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_user_author",
            post: author_candidate(|a| a.is_nsfw_user = true),
            expect: vec![
                (TimelineHome, Role::NonFollower, allow()),
                (
                    TimelineHome,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    allow(),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_user_author/drop/nsfw_media",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_user_author/drop/nsfw_media",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
                (
                    ImmersiveExpandedRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_sensitive_viewer_user/drop/nsfw_media",
                    ),
                ),
                (ImmersiveExpandedRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "nsfw_admin_author",
            post: author_candidate(|a| a.is_nsfw_admin = true),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_admin_author/drop/nsfw_media",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::Follower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_admin_author/drop/nsfw_media",
                    ),
                ),
                (TimelineHomeRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "geo_denied_media_de",
            post: tweet_candidate(|t| t.media.geo_deny_list = vec!["de".to_string()]),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::As("in_de", viewer_in_country("de")),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "geo_restricted_media/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::As("in_us", viewer_in_country("us")),
                    allow(),
                ),
                (TimelineHomeRecommendations, Role::NonFollower, allow()),
                (
                    TimelineHomeRecommendations,
                    Role::As(
                        "author_in_de",
                        ViewerFeatures {
                            country_code: Some("de".to_string()),
                            ..author_viewer()
                        },
                    ),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "geo_restricted_media/drop/unspecified",
                    ),
                ),
            ],
        },
        Row {
            name: "geo_allow_listed_media_us",
            post: tweet_candidate(|t| t.media.geo_allow_list = vec!["us".to_string()]),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "geo_restricted_media/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::As("in_us", viewer_in_country("us")),
                    allow(),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::As("in_de", viewer_in_country("de")),
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "geo_restricted_media/drop/unspecified",
                    ),
                ),
            ],
        },
        Row {
            name: "geo_allow_listed_media_uppercase",
            post: tweet_candidate(|t| t.media.geo_allow_list = vec!["US".to_string()]),
            expect: vec![(
                TimelineHomeRecommendations,
                Role::As("in_us", viewer_in_country("us")),
                allow(),
            )],
        },
        Row {
            name: "geo_denied_media_uppercase",
            post: tweet_candidate(|t| t.media.geo_deny_list = vec!["DE".to_string()]),
            expect: vec![(
                TimelineHomeRecommendations,
                Role::As("in_de", viewer_in_country("de")),
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "geo_restricted_media/drop/unspecified",
                ),
            )],
        },
        Row {
            name: "geo_denied_media_worldwide",
            post: tweet_candidate(|t| t.media.geo_deny_list = vec!["xx".to_string()]),
            expect: vec![(
                TimelineHomeRecommendations,
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "geo_restricted_media/drop/unspecified",
                ),
            )],
        },
        Row {
            name: "geo_denied_media_retweet",
            post: HydratedTweetCandidate {
                source_tweet_id: Some(2),
                ..tweet_candidate(|t| t.media.geo_deny_list = vec!["de".to_string()])
            },
            expect: vec![(
                TimelineHomeRecommendations,
                Role::As("in_de", viewer_in_country("de")),
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "geo_restricted_media/drop/unspecified",
                ),
            )],
        },
    ]
}
