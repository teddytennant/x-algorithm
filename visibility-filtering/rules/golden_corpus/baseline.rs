use super::builders::{labeled, read_only_viewer};
use super::{Role, Row};
use crate::models::{LimitedEngagementReason, SafetyLabelType};
use crate::rules::fixtures::{allow, candidate, dropped, limited, AUTHOR_ID, VIEWER_ID};
use crate::rules::SafetyLevel::{
    FilterAll, TimelineHome, TimelineHomeHydration, TimelineHomeRecommendations,
};
use xai_visibility_filtering::models::FilteredReason;

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "pristine",
            post: candidate().build(),
            expect: vec![
                (
                    FilterAll,
                    Role::NonFollower,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "filter_all/drop/unspecified",
                    ),
                ),
                (
                    FilterAll,
                    Role::Author,
                    dropped(
                        FilteredReason::UnspecifiedReason,
                        "filter_all/drop/unspecified",
                    ),
                ),
                (TimelineHome, Role::NonFollower, allow()),
                (TimelineHome, Role::LoggedOut, allow()),
                (TimelineHomeRecommendations, Role::NonFollower, allow()),
                (
                    TimelineHomeHydration,
                    Role::As("read_only", read_only_viewer(VIEWER_ID)),
                    limited(
                        LimitedEngagementReason::ReadonlyViewer,
                        "read_only_viewer/limited_engagement",
                    ),
                ),
                (
                    TimelineHomeHydration,
                    Role::As("read_only_author", read_only_viewer(AUTHOR_ID)),
                    limited(
                        LimitedEngagementReason::ReadonlyViewer,
                        "read_only_viewer/limited_engagement",
                    ),
                ),
            ],
        },
        Row {
            name: "egregious_nsfw_label",
            post: labeled(SafetyLabelType::EGREGIOUS_NSFW),
            expect: vec![
                (TimelineHome, Role::NonFollower, allow()),
                (TimelineHomeRecommendations, Role::NonFollower, allow()),
            ],
        },
    ]
}
