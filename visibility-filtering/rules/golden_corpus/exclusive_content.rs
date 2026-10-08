use super::{Role, Row};
use crate::hydration::Hydrator;
use crate::models::{HydratedTweetCandidate, SafetyLabelType};
use crate::rules::fixtures::{allow, candidate, dropped, CandidateBuilder};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "exclusive",
            post: exclusive_candidate(candidate()),
            expect: vec![
                (
                    TimelineHome,
                    Role::NonFollower,
                    dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
                ),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHome,
                    Role::LoggedOut,
                    dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
                ),
            ],
        },
        Row {
            name: "exclusive_super_followed",
            post: exclusive_candidate(candidate().with_edge(Hydrator::SuperFollowsExclusive)),
            expect: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "exclusive_retweet",
            post: exclusive_candidate(candidate().retweet_of(2)),
            expect: vec![(
                TimelineHome,
                Role::Author,
                dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
            )],
        },
        Row {
            name: "exclusive_nsfw_high_precision_media",
            post: exclusive_candidate(
                candidate()
                    .with_media()
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION),
            ),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "creator_tweet_nsfw/drop/nsfw_media",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::LoggedOut,
                    dropped(
                        FilteredReason::ContainNsfwMedia,
                        "creator_tweet_nsfw/drop/nsfw_media",
                    ),
                ),
            ],
        },
        Row {
            name: "exclusive_nsfw_high_precision_media_super_followed",
            post: exclusive_candidate(
                candidate()
                    .with_media()
                    .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
                    .with_edge(Hydrator::SuperFollowsExclusive),
            ),
            expect: vec![(
                TimelineHomeHydration,
                Role::NonFollower,
                dropped(
                    FilteredReason::ContainNsfwMedia,
                    "creator_tweet_nsfw/drop/nsfw_media",
                ),
            )],
        },
        Row {
            name: "exclusive_super_followed_author_blocks_viewer",
            post: exclusive_candidate(
                candidate()
                    .with_edge(Hydrator::SuperFollowsExclusive)
                    .with_edge(Hydrator::BlockedByAuthor),
            ),
            expect: vec![
                (
                    TimelineHomeHydration,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "author_blocks_viewer_exclusive_content/drop/unspecified",
                    ),
                ),
                (TimelineHomeHydration, Role::Author, allow()),
            ],
        },
        Row {
            name: "exclusive_author_blocks_viewer",
            post: exclusive_candidate(candidate().with_edge(Hydrator::BlockedByAuthor)),
            expect: vec![(
                TimelineHomeHydration,
                Role::NonFollower,
                dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
            )],
        },
    ]
}

fn exclusive_candidate(builder: CandidateBuilder) -> HydratedTweetCandidate {
    let mut c = builder.build();
    c.tweet_features.exclusive_conversation_author_id = Some(42);
    c
}
