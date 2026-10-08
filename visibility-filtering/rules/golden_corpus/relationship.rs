use super::builders::controlled_root;
use super::{Role, Row};
use crate::hydration::Hydrator::{
    BlockedByAuthor, BlockedByReplyRoot, Blocks, MuteRetweets, Mutes,
};
use crate::models::LimitedEngagementReason;
use crate::rules::fixtures::{allow, candidate, dropped, limited, limited_for};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_core_entities::entities::ConversationControlArm;
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "viewer_blocks_author",
            post: candidate().with_edge(Blocks).build(),
            expect: vec![(
                TimelineHome,
                Role::NonFollower,
                dropped(
                    FilteredReason::ViewerBlocksAuthor,
                    "viewer_blocks_author/drop",
                ),
            )],
        },
        Row {
            name: "viewer_mutes_author",
            post: candidate().with_edge(Mutes).build(),
            expect: vec![(
                TimelineHome,
                Role::NonFollower,
                dropped(
                    FilteredReason::ViewerMutesAuthor,
                    "viewer_mutes_author/drop",
                ),
            )],
        },
        Row {
            name: "viewer_blocks_and_mutes_author",
            post: candidate().with_edge(Blocks).with_edge(Mutes).build(),
            expect: vec![(
                TimelineHome,
                Role::NonFollower,
                dropped(
                    FilteredReason::ViewerBlocksAuthor,
                    "viewer_blocks_author/drop",
                ),
            )],
        },
        Row {
            name: "muted_retweets_retweet",
            post: candidate().with_edge(MuteRetweets).retweet_of(2).build(),
            expect: vec![(
                TimelineHome,
                Role::NonFollower,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "viewer_mutes_retweets/drop/unspecified",
                ),
            )],
        },
        Row {
            name: "muted_retweets_original",
            post: candidate().with_edge(MuteRetweets).build(),
            expect: vec![(TimelineHome, Role::NonFollower, allow())],
        },
        Row {
            name: "author_and_root_author_block",
            post: candidate()
                .with_edge(BlockedByAuthor)
                .with_edge(BlockedByReplyRoot)
                .build(),
            expect: vec![(
                TimelineHomeHydration,
                Role::NonFollower,
                limited_for(
                    &[
                        LimitedEngagementReason::BlockedViewer,
                        LimitedEngagementReason::RootAuthorBlockedViewer,
                    ],
                    "blocked_viewer/limited_engagement",
                ),
            )],
        },
        Row {
            name: "author_block",
            post: candidate().with_edge(BlockedByAuthor).build(),
            expect: vec![(TimelineHomeHydration, Role::Author, allow())],
        },
        Row {
            name: "root_author_block",
            post: candidate().with_edge(BlockedByReplyRoot).build(),
            expect: vec![(
                TimelineHomeHydration,
                Role::Author,
                limited(
                    LimitedEngagementReason::RootAuthorBlockedViewer,
                    "blocked_viewer/limited_engagement/root_author_blocked_viewer",
                ),
            )],
        },
        Row {
            name: "root_author_block_community_conversation",
            post: candidate()
                .with_edge(BlockedByReplyRoot)
                .with_conversation_control(controlled_root(ConversationControlArm::Community))
                .build(),
            expect: vec![(
                TimelineHomeHydration,
                Role::NonFollower,
                limited_for(
                    &[
                        LimitedEngagementReason::RootAuthorBlockedViewer,
                        LimitedEngagementReason::ConversationControl,
                    ],
                    "blocked_viewer/limited_engagement/root_author_blocked_viewer",
                ),
            )],
        },
    ]
}
