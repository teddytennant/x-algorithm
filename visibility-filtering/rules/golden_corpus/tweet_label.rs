use super::builders::labeled;
use super::{Role, Row};
use crate::models::SafetyLabelType;
use crate::rules::fixtures::{allow, appealed, candidate, dropped};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_visibility_filtering::models::{
    Action, DropReason, FilteredReason, SafetyResult, SafetyResultReason,
};
use xai_x_thrift::action::AppealablePolicy;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "for_emergency_use_only_label",
            post: labeled(SafetyLabelType::FOR_EMERGENCY_USE_ONLY),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "for_emergency_use_only/drop/unspecified",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "for_emergency_use_only/drop/unspecified",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Author,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "for_emergency_use_only/drop/unspecified",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::Author,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "for_emergency_use_only/drop/unspecified",
                    ),
                ),
            ],
        },
        Row {
            name: "pdna_label",
            post: labeled(SafetyLabelType::PDNA),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(nsfw_high_precision_reason(), "pdna/drop/safety_result"),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(nsfw_high_precision_reason(), "pdna/drop/safety_result"),
                ),
                (TimelineHome, Role::Author, allow()),
            ],
        },
        Row {
            name: "bounce_label",
            post: labeled(SafetyLabelType::BOUNCE),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
                (TimelineHome, Role::Author, allow()),
            ],
        },
        Row {
            name: "spam_label",
            post: labeled(SafetyLabelType::SPAM),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::PossiblyUndesirable, "spam/drop/undesirable"),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(FilteredReason::PossiblyUndesirable, "spam/drop/undesirable"),
                ),
                (TimelineHome, Role::Author, allow()),
            ],
        },
        Row {
            name: "spam_and_bounce_labels",
            post: candidate()
                .with_label(SafetyLabelType::SPAM)
                .with_label(SafetyLabelType::BOUNCE)
                .build(),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(FilteredReason::PossiblyUndesirable, "spam/drop/undesirable"),
                ),
            ],
        },
        Row {
            name: "fosnr_hateful_conduct_label",
            post: labeled(SafetyLabelType::FOSNR_HATEFUL_CONDUCT),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_hateful_conduct/drop/undesirable",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_hateful_conduct/drop/undesirable",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Author,
                    appealed(
                        AppealablePolicy::HATEFUL_CONDUCT,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_violent_speech_label",
            post: labeled(SafetyLabelType::FOSNR_VIOLENT_SPEECH),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_violent_speech/drop/undesirable",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_violent_speech/drop/undesirable",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Author,
                    appealed(
                        AppealablePolicy::VIOLENT_SPEECH,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_abuse_label",
            post: labeled(SafetyLabelType::FOSNR_ABUSE),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse/drop/undesirable",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse/drop/undesirable",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Author,
                    appealed(
                        AppealablePolicy::ABUSE,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_civic_integrity_label",
            post: labeled(SafetyLabelType::FOSNR_CIVIC_INTEGRITY),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_civic_integrity/drop/undesirable",
                    ),
                ),
                (
                    TimelineHome,
                    Role::Follower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_civic_integrity/drop/undesirable",
                    ),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeHydration,
                    Role::Author,
                    appealed(
                        AppealablePolicy::CIVIC_INTEGRITY,
                        3,
                        true,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_level_3_and_level_1_labels",
            post: candidate()
                .with_label(SafetyLabelType::FOSNR_VIOLENT_SPEECH)
                .with_label(SafetyLabelType::FOSNR_ABUSE_INSULTS)
                .with_agent_label(SafetyLabelType::FOSNR_CIVIC_INTEGRITY)
                .build(),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::Author,
                    appealed(
                        AppealablePolicy::CIVIC_INTEGRITY,
                        3,
                        false,
                        false,
                        "fosnr_author/appealable",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_violent_speech/drop/undesirable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_hateful_conduct_and_bounce_labels",
            post: candidate()
                .with_label(SafetyLabelType::FOSNR_HATEFUL_CONDUCT)
                .with_label(SafetyLabelType::BOUNCE)
                .build(),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_hateful_conduct/drop/undesirable",
                    ),
                ),
            ],
        },
        Row {
            name: "fosnr_abuse_insults_and_bounce_labels",
            post: candidate()
                .with_label(SafetyLabelType::FOSNR_ABUSE_INSULTS)
                .with_label(SafetyLabelType::BOUNCE)
                .build(),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::PossiblyUndesirable,
                        "fosnr_abuse_insults_non_follower/drop/undesirable",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::Follower,
                    dropped(FilteredReason::TweetIsBounced, "bounce/drop/bounced"),
                ),
            ],
        },
    ]
}

fn nsfw_high_precision_reason() -> FilteredReason {
    FilteredReason::SafetyResult(SafetyResult {
        reason: Some(SafetyResultReason::NsfwHighPrecision),
        action: Action::Drop(DropReason {}),
    })
}
