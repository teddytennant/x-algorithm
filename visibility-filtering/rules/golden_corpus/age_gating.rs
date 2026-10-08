use super::builders::{
    labeled, no_stated_age_viewer, tweet_candidate, viewer_in_country, viewer_with_age,
};
use super::{Role, Row};
use crate::models::{
    HydratedTweetCandidate, NsfwViewerDropReason, SafetyLabelType, Viewer, ViewerAge,
    ViewerFeatures, ViewerProfile,
};
use crate::rules::fixtures::{
    allow, blurred, candidate, dropped, legacy_interstitial, nsfw_viewer_dropped,
    sensitive_opt_in_viewer, viewer_with_profile, AUTHOR_ID,
};
use crate::rules::SafetyLevel::{
    ImmersiveExpandedRecommendations, TimelineHome, TimelineHomeHydration,
    TimelineHomeRecommendations,
};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::InterstitialReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "unflagged_media",
            post: candidate().with_media().build(),
            expect: vec![
                (
                    TimelineHome,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    allow(),
                ),
                (TimelineHome, Role::LoggedOut, allow()),
                (
                    TimelineHome,
                    Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                    allow(),
                ),
            ],
        },
        Row {
            name: "nsfw_high_recall_media",
            post: labeled_media(SafetyLabelType::NSFW_HIGH_RECALL),
            expect: vec![
                (
                    TimelineHome,
                    Role::LoggedOut,
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::LoggedOut,
                        "sensitive_viewer_logged_out/drop",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::IsUnderage,
                        "sensitive_viewer_underage/drop",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As("no_stated_age_in_us", no_stated_age_viewer("us")),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As("adult", viewer_with_age(ViewerAge::Known(30))),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::HasNoStatedAge,
                        "sensitive_viewer_no_stated_age/drop",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As("unknown_age_in_gb", viewer_in_country("gb")),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "underage_opted_in",
                        viewer_with_profile(ViewerProfile {
                            viewer_age: ViewerAge::Known(17),
                            allows_sensitive_media: true,
                            ..ViewerProfile::default()
                        }),
                    ),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::IsUnderage,
                        "sensitive_viewer_underage/drop",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "underage_author",
                        ViewerFeatures {
                            viewer: Viewer::LoggedIn {
                                id: AUTHOR_ID,
                                profile: ViewerProfile {
                                    viewer_age: ViewerAge::Known(17),
                                    ..ViewerProfile::default()
                                },
                                has_age_verified_18_label: false,
                            },
                            ..ViewerFeatures::default()
                        },
                    ),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "no_stated_age_without_country",
                        viewer_with_age(ViewerAge::NotStated),
                    ),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "no_stated_age_in_de",
                        ViewerFeatures {
                            country_code: Some("de".to_string()),
                            ..viewer_with_age(ViewerAge::NotStated)
                        },
                    ),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::HasNoStatedAge,
                        "sensitive_viewer_no_stated_age/drop",
                    ),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "no_stated_age_us_account_in_de",
                        ViewerFeatures {
                            country_code: Some("de".to_string()),
                            ..no_stated_age_viewer("us")
                        },
                    ),
                    allow(),
                ),
                (
                    TimelineHome,
                    Role::As(
                        "no_stated_age_kr_account_in_us",
                        ViewerFeatures {
                            country_code: Some("us".to_string()),
                            ..no_stated_age_viewer("kr")
                        },
                    ),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::HasNoStatedAge,
                        "sensitive_viewer_no_stated_age/drop",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_high_precision_media",
            post: labeled_media(SafetyLabelType::NSFW_HIGH_PRECISION),
            expect: vec![
                (
                    TimelineHome,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::IsUnderage,
                        "sensitive_viewer_underage/drop",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    legacy_interstitial("nsfw_high_precision/legacy_interstitial"),
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::As("sensitive_opt_in", sensitive_opt_in_viewer()),
                    allow(),
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "nsfw_sensitive_viewer_tweet/drop/nsfw_media",
                    ),
                ),
                (ImmersiveExpandedRecommendations, Role::Author, allow()),
                (
                    ImmersiveExpandedRecommendations,
                    Role::LoggedOut,
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::LoggedOut,
                        "sensitive_viewer_logged_out/drop",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_text_label",
            post: labeled(SafetyLabelType::NSFW_TEXT),
            expect: vec![
                (
                    TimelineHomeRecommendations,
                    Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::IsUnderage,
                        "sensitive_viewer_underage/drop",
                    ),
                ),
                (
                    TimelineHomeRecommendations,
                    Role::As("adult", viewer_with_age(ViewerAge::Known(30))),
                    allow(),
                ),
            ],
        },
        Row {
            name: "gore_and_violence_media",
            post: labeled_media(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
            expect: vec![
                (
                    TimelineHome,
                    Role::As("no_stated_age_in_gb", no_stated_age_viewer("gb")),
                    nsfw_viewer_dropped(
                        NsfwViewerDropReason::HasNoStatedAge,
                        "sensitive_viewer_no_stated_age/drop",
                    ),
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "gore_and_violence_high_precision/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "nsfw_user_flag_media_retweet",
            post: HydratedTweetCandidate {
                source_tweet_id: Some(2),
                ..tweet_candidate(|t| {
                    t.nsfw.user = true;
                    t.media.has_media = true;
                })
            },
            expect: vec![(
                TimelineHome,
                Role::As("underage", viewer_with_age(ViewerAge::Known(17))),
                blurred(
                    InterstitialReason::SensitiveUser(true),
                    "nsfw_user/blur/sensitive_user",
                ),
            )],
        },
    ]
}

fn labeled_media(label: SafetyLabelType) -> HydratedTweetCandidate {
    candidate().with_label(label).with_media().build()
}
