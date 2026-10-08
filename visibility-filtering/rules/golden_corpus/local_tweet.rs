use super::builders::{controlled_root, on_client};
use super::{Role, Row};
use crate::hydration::Hydrator::OutsideNarrowcastPlace;
use crate::hydration::{Hydrator, Hydrators};
use crate::models::{
    ConversationControlFeatures, HydratedTweetCandidate, LimitedEngagementReason, SafetyLabelType,
    TweetFeatures,
};
use crate::rules::fixtures::{
    allow, blurred, blurred_and_limited, candidate, conversation_control, limited, limited_for,
    viewer, CandidateBuilder, VIEWER_ID,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_core_entities::entities::ConversationControl;
use xai_core_entities::entities::ConversationControlArm::{ByInvitation, Local};
use xai_x_thrift::action::InterstitialReason;

const PLACE_ID: u64 = 0xa000_0000_0000_0001;
const REPLY_ROOT_AUTHOR_ID: u64 = 4242;

pub(super) fn rows() -> Vec<Row> {
    let local = || {
        limited(
            LimitedEngagementReason::LocalTweet,
            "local_tweet/limited_engagement",
        )
    };
    let hydration = |role, verdict| (TimelineHomeHydration, role, verdict);
    vec![
        Row {
            name: "local_post_viewer_outside_place",
            post: local_post().with_edge(OutsideNarrowcastPlace).build(),
            expect: vec![
                hydration(Role::NonFollower, local()),
                hydration(Role::Author, allow()),
                hydration(Role::LoggedOut, allow()),
                (TimelineHome, Role::NonFollower, allow()),
            ],
        },
        Row {
            name: "local_post_viewer_in_place",
            post: local_post().build(),
            expect: vec![hydration(Role::NonFollower, allow())],
        },
        Row {
            name: "local_post_invited_viewer",
            post: local_post()
                .with_edge(OutsideNarrowcastPlace)
                .with_conversation_control(ConversationControlFeatures {
                    control: ConversationControl {
                        invited_user_ids: vec![VIEWER_ID],
                        ..controlled_root(Local).control
                    },
                    ..controlled_root(Local)
                })
                .build(),
            expect: vec![hydration(Role::NonFollower, allow())],
        },
        Row {
            name: "local_reply_author_outside_place",
            post: local_post()
                .with_edge(OutsideNarrowcastPlace)
                .with_conversation_control(conversation_control(Local, REPLY_ROOT_AUTHOR_ID))
                .build(),
            expect: vec![hydration(Role::Author, local())],
        },
        Row {
            name: "local_control_without_place",
            post: candidate()
                .with_edge(OutsideNarrowcastPlace)
                .with_conversation_control(controlled_root(Local))
                .build(),
            expect: vec![hydration(Role::NonFollower, allow())],
        },
        Row {
            name: "local_post_location_lookup_failed",
            post: HydratedTweetCandidate {
                failed: Hydrators::of(OutsideNarrowcastPlace),
                ..local_post().build()
            },
            expect: vec![hydration(Role::NonFollower, allow())],
        },
        Row {
            name: "local_post_conversation_control_failed",
            post: HydratedTweetCandidate {
                failed: Hydrators::of(Hydrator::ConversationControl),
                ..candidate()
                    .with_tweet_features(local_place())
                    .with_edge(OutsideNarrowcastPlace)
                    .build()
            },
            expect: vec![hydration(Role::Author, local())],
        },
        Row {
            name: "nsfw_local_post",
            post: local_post()
                .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                .with_media()
                .with_edge(OutsideNarrowcastPlace)
                .build(),
            expect: vec![hydration(
                Role::As("web_in_us", on_client("web", "us", viewer(VIEWER_ID))),
                blurred_and_limited(
                    blurred(
                        InterstitialReason::Sensitive(true),
                        "nsfw_high_precision/blur/sensitive",
                    ),
                    local(),
                ),
            )],
        },
        Row {
            name: "local_post_under_by_invitation_control",
            post: local_post()
                .with_edge(OutsideNarrowcastPlace)
                .with_conversation_control(controlled_root(ByInvitation))
                .build(),
            expect: vec![hydration(
                Role::NonFollower,
                limited_for(
                    &[
                        LimitedEngagementReason::ConversationControl,
                        LimitedEngagementReason::LocalTweet,
                    ],
                    "limit_replies_by_invitation/limited_engagement/conversation_control",
                ),
            )],
        },
    ]
}

fn local_place() -> TweetFeatures {
    TweetFeatures {
        narrowcast_place_id: Some(PLACE_ID),
        ..TweetFeatures::default()
    }
}

fn local_post() -> CandidateBuilder {
    candidate()
        .with_tweet_features(local_place())
        .with_conversation_control(controlled_root(Local))
}
