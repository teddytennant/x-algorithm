use super::{Role, Row};
use crate::hydration::Hydrator::TrustedFriends;
use crate::models::TweetFeatures;
use crate::rules::fixtures::{allow, candidate, dropped, CandidateBuilder};
use crate::rules::SafetyLevel::{
    ImmersiveExpandedRecommendations, TimelineHome, TimelineHomeHydration,
    TimelineHomeRecommendations,
};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    let trusted_friends = || {
        dropped(
            FilteredReason::UnspecifiedReason,
            "trusted_friends_tweet/drop/unspecified",
        )
    };
    vec![
        Row {
            name: "trusted_friends",
            post: trusted_friends_post().build(),
            expect: vec![
                (TimelineHomeHydration, Role::NonFollower, trusted_friends()),
                (TimelineHomeHydration, Role::Follower, trusted_friends()),
                (TimelineHomeHydration, Role::LoggedOut, trusted_friends()),
                (TimelineHomeHydration, Role::Author, allow()),
                (TimelineHome, Role::Follower, trusted_friends()),
                (TimelineHome, Role::Author, allow()),
                (
                    TimelineHomeRecommendations,
                    Role::NonFollower,
                    trusted_friends(),
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::Follower,
                    trusted_friends(),
                ),
                (
                    ImmersiveExpandedRecommendations,
                    Role::LoggedOut,
                    trusted_friends(),
                ),
                (ImmersiveExpandedRecommendations, Role::Author, allow()),
            ],
        },
        Row {
            name: "trusted_friends_list_member_or_owner",
            post: trusted_friends_post().with_edge(TrustedFriends).build(),
            expect: vec![
                (TimelineHomeHydration, Role::NonFollower, allow()),
                (ImmersiveExpandedRecommendations, Role::NonFollower, allow()),
            ],
        },
    ]
}

fn trusted_friends_post() -> CandidateBuilder {
    candidate().with_tweet_features(TweetFeatures {
        trusted_friends_list_id: Some(7),
        ..TweetFeatures::default()
    })
}
