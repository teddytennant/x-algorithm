use super::builders::tweet_candidate;
use super::{Role, Row};
use crate::models::{ArticleLifecycle, HydratedTweetCandidate, Verdict};
use crate::rules::fixtures::{allow, dropped};
use crate::rules::SafetyLevel::{self, TimelineHomeHydration};
use std::num::NonZeroU64;
use xai_visibility_filtering::models::FilteredReason;

fn article_post(lifecycle: Option<ArticleLifecycle>) -> HydratedTweetCandidate {
    HydratedTweetCandidate {
        article_lifecycle: lifecycle,
        ..tweet_candidate(|t| t.article_id = NonZeroU64::new(1))
    }
}

fn home_hydration_only(verdict: fn() -> Verdict) -> Vec<(SafetyLevel, Role, Verdict)> {
    vec![
        (TimelineHomeHydration, Role::NonFollower, verdict()),
        (TimelineHomeHydration, Role::Follower, verdict()),
        (TimelineHomeHydration, Role::LoggedOut, verdict()),
        (TimelineHomeHydration, Role::Author, verdict()),
    ]
}

fn unpublished() -> Verdict {
    dropped(
        FilteredReason::UnspecifiedReason,
        "article_tweet_content/drop/unspecified",
    )
}

pub(super) fn rows() -> Vec<Row> {
    vec![
        Row {
            name: "article_draft",
            post: article_post(Some(ArticleLifecycle::Draft)),
            expect: home_hydration_only(unpublished),
        },
        Row {
            name: "article_soft_deleted",
            post: article_post(Some(ArticleLifecycle::SoftDeleted)),
            expect: home_hydration_only(unpublished),
        },
        Row {
            name: "article_without_lifecycle",
            post: article_post(None),
            expect: home_hydration_only(unpublished),
        },
        Row {
            name: "article_published",
            post: article_post(Some(ArticleLifecycle::Published)),
            expect: home_hydration_only(allow),
        },
        Row {
            name: "no_article",
            post: tweet_candidate(|_| {}),
            expect: home_hydration_only(allow),
        },
    ]
}
