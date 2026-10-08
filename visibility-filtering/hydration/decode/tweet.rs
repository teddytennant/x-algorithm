use crate::hydration::fallback_cache::{Column, FallbackCache};
use crate::models::{AuthorId, MediaFeature, NsfwFeature, PureCore, TweetFeatures, TweetId};
use std::num::NonZeroU64;
use xai_core_entities::entities::{EditControl, EditControlInitial, PureCoreData, TakedownReason};

#[derive(Clone)]
pub(crate) struct CachedTweet {
    pure_core: Option<PureCore>,
    row: Option<CachedTweetRow>,
}

pub(crate) type TweetFallbackCache = FallbackCache<CachedTweet>;

pub(crate) fn tweet_fallback_cache(capacity: usize) -> TweetFallbackCache {
    FallbackCache::new("tweet", capacity)
}

pub(crate) struct PureCoreColumn;

impl Column for PureCoreColumn {
    type Entry = CachedTweet;
    type Value = PureCore;
    type Stored = PureCore;
    const NAME: &'static str = "pure_core";

    fn store(value: &PureCore) -> PureCore {
        *value
    }

    fn new_entry(stored: PureCore) -> CachedTweet {
        CachedTweet {
            pure_core: Some(stored),
            row: None,
        }
    }

    fn replace(entry: &mut CachedTweet, stored: PureCore) -> Option<PureCore> {
        entry.pure_core.replace(stored)
    }

    fn get(entry: &CachedTweet) -> Option<PureCore> {
        entry.pure_core
    }

    fn holds(entry: &CachedTweet) -> bool {
        entry.pure_core.is_some()
    }

    fn others_hold(entry: &CachedTweet) -> bool {
        entry.row.is_some()
    }

    fn clear(entry: &mut CachedTweet) -> Option<PureCore> {
        entry.pure_core.take()
    }
}

pub(crate) struct TweetRowColumn;

impl Column for TweetRowColumn {
    type Entry = CachedTweet;
    type Value = TweetFeatures;
    type Stored = CachedTweetRow;
    const NAME: &'static str = "tweet_row";

    fn store(value: &TweetFeatures) -> CachedTweetRow {
        CachedTweetRow::new(value)
    }

    fn new_entry(stored: CachedTweetRow) -> CachedTweet {
        CachedTweet {
            pure_core: None,
            row: Some(stored),
        }
    }

    fn replace(entry: &mut CachedTweet, stored: CachedTweetRow) -> Option<CachedTweetRow> {
        entry.row.replace(stored)
    }

    fn get(entry: &CachedTweet) -> Option<TweetFeatures> {
        entry.row.as_ref().map(CachedTweetRow::features)
    }

    fn holds(entry: &CachedTweet) -> bool {
        entry.row.is_some()
    }

    fn others_hold(entry: &CachedTweet) -> bool {
        entry.pure_core.is_some()
    }

    fn clear(entry: &mut CachedTweet) -> Option<CachedTweetRow> {
        entry.row.take()
    }
}

#[derive(Clone)]
pub(crate) struct CachedTweetRow {
    has_media: bool,
    has_uploaded_media: bool,
    has_dmca_media: bool,
    nsfw_user: bool,
    nsfw_admin: bool,
    is_nullcast: bool,
    is_trusted_friends_tweet: bool,
    trusted_friends_list_id: u64,
    community_id: Option<NonZeroU64>,
    latest_edit_tweet_id: Option<NonZeroU64>,
    exclusive_conversation_author_id: Option<NonZeroU64>,
    article_id: Option<NonZeroU64>,
    restrictions: Option<Box<Restrictions>>,
}

#[derive(Clone, Default)]
struct Restrictions {
    geo_allow_list: Vec<String>,
    geo_deny_list: Vec<String>,
    takedown_reasons: Vec<TakedownReason>,
    narrowcast_place_id: Option<u64>,
}

impl CachedTweetRow {
    fn new(tweet: &TweetFeatures) -> Self {
        let TweetFeatures {
            media:
                MediaFeature {
                    has_media,
                    has_uploaded_media,
                    has_dmca_media,
                    geo_allow_list,
                    geo_deny_list,
                },
            takedown_reasons,
            nsfw: NsfwFeature { user, admin },
            is_nullcast,
            community_id,
            trusted_friends_list_id,
            edit_control: _,
            exclusive_conversation_author_id,
            article_id,
            narrowcast_place_id,
        } = tweet;
        let restricted = !(geo_allow_list.is_empty()
            && geo_deny_list.is_empty()
            && takedown_reasons.is_empty()
            && narrowcast_place_id.is_none());
        Self {
            has_media: *has_media,
            has_uploaded_media: *has_uploaded_media,
            has_dmca_media: *has_dmca_media,
            nsfw_user: *user,
            nsfw_admin: *admin,
            is_nullcast: *is_nullcast,
            is_trusted_friends_tweet: trusted_friends_list_id.is_some(),
            trusted_friends_list_id: trusted_friends_list_id.unwrap_or_default(),
            community_id: *community_id,
            latest_edit_tweet_id: tweet.latest_edit_tweet_id().map(stored_id),
            exclusive_conversation_author_id: exclusive_conversation_author_id.map(stored_id),
            article_id: *article_id,
            restrictions: restricted.then(|| {
                Box::new(Restrictions {
                    geo_allow_list: geo_allow_list.clone(),
                    geo_deny_list: geo_deny_list.clone(),
                    takedown_reasons: takedown_reasons.clone(),
                    narrowcast_place_id: *narrowcast_place_id,
                })
            }),
        }
    }

    fn features(&self) -> TweetFeatures {
        let Restrictions {
            geo_allow_list,
            geo_deny_list,
            takedown_reasons,
            narrowcast_place_id,
        } = self.restrictions.as_deref().cloned().unwrap_or_default();
        TweetFeatures {
            media: MediaFeature {
                has_media: self.has_media,
                has_uploaded_media: self.has_uploaded_media,
                has_dmca_media: self.has_dmca_media,
                geo_allow_list,
                geo_deny_list,
            },
            takedown_reasons,
            nsfw: NsfwFeature {
                user: self.nsfw_user,
                admin: self.nsfw_admin,
            },
            is_nullcast: self.is_nullcast,
            community_id: self.community_id,
            trusted_friends_list_id: self
                .is_trusted_friends_tweet
                .then_some(self.trusted_friends_list_id),
            edit_control: self.latest_edit_tweet_id.map(|latest| {
                EditControl::Initial(EditControlInitial {
                    edit_tweet_ids: vec![read_id(latest)],
                    ..Default::default()
                })
            }),
            exclusive_conversation_author_id: self.exclusive_conversation_author_id.map(read_id),
            article_id: self.article_id,
            narrowcast_place_id,
        }
    }
}

fn stored_id(id: u64) -> NonZeroU64 {
    NonZeroU64::new(id).unwrap_or(NonZeroU64::MAX)
}

fn read_id(id: NonZeroU64) -> u64 {
    if id == NonZeroU64::MAX {
        0
    } else {
        id.get()
    }
}

pub(crate) fn pure_core(core: &PureCoreData) -> PureCore {
    PureCore {
        author_id: AuthorId(core.author_id),
        source_tweet_id: core.source_tweet_id.map(TweetId),
        source_author_id: core.source_user_id.filter(|&id| id != 0).map(AuthorId),
        direct_reply_root_author_id: direct_reply_root_author(core),
    }
}

fn direct_reply_root_author(core: &PureCoreData) -> Option<AuthorId> {
    core.in_reply_to_tweet_id
        .filter(|&replied_to| core.conversation_id == Some(replied_to))
        .and(core.in_reply_to_user_id)
        .map(AuthorId)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::fixtures::{candidate, dropped, logged_out_viewer};
    use crate::rules::{RuleEngine, SafetyLevel};
    use xai_core_entities::entities::EditControlEdit;
    use xai_visibility_filtering::models::FilteredReason;

    #[test]
    fn a_share_without_its_user_id_leaves_the_source_author_unknown() {
        for source_user_id in [Some(0), None] {
            let core = PureCoreData {
                author_id: 10,
                source_tweet_id: Some(5),
                source_user_id,
                ..Default::default()
            };
            assert_eq!(pure_core(&core).source_author_id, None);
        }
    }

    #[test]
    fn a_cached_row_fits_in_56_bytes_and_keeps_a_circle_list_id_of_zero() {
        let circle = TweetFeatures {
            trusted_friends_list_id: Some(0),
            ..Default::default()
        };
        assert_eq!(CachedTweetRow::new(&circle).features(), circle);
        assert_eq!(size_of::<CachedTweetRow>(), 56);
    }

    #[test]
    fn a_cached_row_keeps_the_place_of_a_local_post_with_no_other_restriction() {
        let local = TweetFeatures {
            narrowcast_place_id: Some(0xa000_0000_0000_0001),
            ..Default::default()
        };
        assert_eq!(CachedTweetRow::new(&local).features(), local);
    }

    #[test]
    fn a_cached_row_drops_a_zero_id_tweet_as_its_fresh_read_does() {
        let exclusive = TweetFeatures {
            exclusive_conversation_author_id: Some(0),
            ..Default::default()
        };
        let stale = TweetFeatures {
            edit_control: Some(EditControl::Initial(EditControlInitial {
                edit_tweet_ids: vec![0],
                ..Default::default()
            })),
            ..Default::default()
        };
        let engine = RuleEngine::for_tests();
        for (tweet, expected) in [
            (
                exclusive,
                dropped(FilteredReason::ExclusiveTweet, "exclusive_tweet/drop"),
            ),
            (
                stale,
                dropped(
                    FilteredReason::UnspecifiedReason,
                    "stale_tweet/drop/unspecified",
                ),
            ),
        ] {
            let cached = CachedTweetRow::new(&tweet).features();
            assert_eq!(cached, tweet);
            let post = candidate().with_tweet_features(cached).build();
            assert_eq!(
                engine
                    .evaluate(SafetyLevel::TimelineHome, &logged_out_viewer(), &post)
                    .into_verdict(),
                expected
            );
        }
    }

    #[test]
    fn a_cached_row_reads_as_its_tweet_with_only_the_latest_edit() {
        let tweet = TweetFeatures {
            media: MediaFeature {
                has_media: true,
                has_uploaded_media: true,
                has_dmca_media: true,
                geo_allow_list: vec!["us".to_string()],
                geo_deny_list: vec!["de".to_string()],
            },
            takedown_reasons: vec![TakedownReason::Dmca],
            nsfw: NsfwFeature {
                user: true,
                admin: true,
            },
            is_nullcast: true,
            community_id: NonZeroU64::new(500),
            trusted_friends_list_id: Some(8),
            edit_control: Some(EditControl::Edit(EditControlEdit {
                initial_tweet_id: 10,
                edit_control_initial: Some(EditControlInitial {
                    edit_tweet_ids: vec![10, 20, 30],
                    ..Default::default()
                }),
            })),
            exclusive_conversation_author_id: Some(7),
            article_id: NonZeroU64::new(8),
            narrowcast_place_id: Some(0xa000_0000_0000_0001),
        };

        let cached = CachedTweetRow::new(&tweet).features();

        assert_eq!(
            cached,
            TweetFeatures {
                edit_control: Some(EditControl::Initial(EditControlInitial {
                    edit_tweet_ids: vec![30],
                    ..Default::default()
                })),
                ..tweet.clone()
            }
        );
        for id in [10, 20, 30] {
            assert_eq!(cached.is_superseded_edit(id), tweet.is_superseded_edit(id));
        }
        assert_eq!(
            CachedTweetRow::new(&TweetFeatures::default()).features(),
            TweetFeatures::default()
        );
    }
}
