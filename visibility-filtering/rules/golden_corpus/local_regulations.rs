use super::builders::{nsfw_high_precision_media, on_client, viewer_in_country};
use super::{Role, Row};
use crate::models::TombstoneReason;
use crate::rules::fixtures::{
    author_viewer, blurred, sensitive_opt_in_viewer, tombstoned, viewer, VIEWER_ID,
};
use crate::rules::SafetyLevel::{TimelineHome, TimelineHomeHydration};
use xai_x_thrift::action::InterstitialReason;

fn web_in_in() -> Role {
    Role::As("web_in_in", on_client("web", "in", viewer(VIEWER_ID)))
}

pub(super) fn rows() -> Vec<Row> {
    let high_precision_tombstone = || {
        tombstoned(
            TombstoneReason::LocalRegulations,
            "nsfw_high_precision/tombstone/local_regulations",
        )
    };
    let plain_high_precision = || {
        blurred(
            InterstitialReason::Sensitive(true),
            "nsfw_high_precision/blur/sensitive",
        )
    };
    vec![Row {
        name: "nsfw_high_precision_media_in_a_local_regulations_country",
        post: nsfw_high_precision_media(),
        expect: vec![
            (
                TimelineHomeHydration,
                web_in_in(),
                high_precision_tombstone(),
            ),
            (
                TimelineHomeHydration,
                Role::As(
                    "sensitive_opt_in_web_in_in",
                    on_client("web", "in", sensitive_opt_in_viewer()),
                ),
                high_precision_tombstone(),
            ),
            (
                TimelineHomeHydration,
                Role::As("no_client_context_in_in", viewer_in_country("in")),
                high_precision_tombstone(),
            ),
            (
                TimelineHomeHydration,
                Role::As("author_web_in_in", on_client("web", "in", author_viewer())),
                plain_high_precision(),
            ),
            (
                TimelineHomeHydration,
                Role::As("web_in_us", on_client("web", "us", viewer(VIEWER_ID))),
                plain_high_precision(),
            ),
            (TimelineHome, web_in_in(), plain_high_precision()),
        ],
    }]
}
