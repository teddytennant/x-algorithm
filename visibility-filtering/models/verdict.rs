use crate::hydration::{Hydrators, Lookup};
use crate::params::LimitedActionType;
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::{AppealablePolicy, InterstitialAction, InterstitialReason};

#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    Withheld(Decided<Withholding>),
    Shown {
        notice: Option<Decided<Notice>>,
        media: Option<Decided<MediaRestriction>>,
        engagement: Option<Decided<LimitedEngagement>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Notice {
    SoftIntervention(FosnrReason),
    Appealable {
        reason: FosnrReason,
        limited_actions: &'static [LimitedActionType],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FosnrReason {
    pub policy: AppealablePolicy,
    pub level: i8,
    pub proactive: bool,
    pub appeal_submitted: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decided<T> {
    pub value: T,
    pub by: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Withholding {
    Drop(DropReason),
    Tombstone(TombstoneReason),
}

#[derive(Clone, Debug, PartialEq)]
pub enum DropReason {
    Legacy(FilteredReason),
    NsfwViewer(NsfwViewerDropReason),
}

impl DropReason {
    pub fn legacy(&self) -> &FilteredReason {
        match self {
            Self::Legacy(reason) => reason,
            Self::NsfwViewer(_) => &FilteredReason::ContainNsfwMedia,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum NsfwViewerDropReason {
    IsUnderage,
    HasNoStatedAge,
    LoggedOut,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MediaRestriction {
    MediaInterstitial(MediaInterstitial),
    NsfwInterstitial,
}

impl MediaRestriction {
    pub fn legacy(&self) -> &FilteredReason {
        match self {
            Self::MediaInterstitial(blur) => &blur.legacy,
            Self::NsfwInterstitial => &FilteredReason::ContainNsfwMedia,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MediaInterstitial {
    pub legacy: FilteredReason,
    pub reason: InterstitialReason,
    pub prompt: Option<InterstitialAction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LimitedEngagement {
    first: LimitedEngagementReason,
    rest: Vec<LimitedEngagementReason>,
}

impl LimitedEngagement {
    pub fn new(reason: LimitedEngagementReason) -> Self {
        Self {
            first: reason,
            rest: Vec::new(),
        }
    }

    pub fn add(&mut self, reason: LimitedEngagementReason) {
        if !self.reasons().any(|held| held == reason) {
            self.rest.push(reason);
        }
    }

    pub fn reason(&self) -> LimitedEngagementReason {
        self.first
    }

    pub fn reasons(&self) -> impl Iterator<Item = LimitedEngagementReason> + '_ {
        std::iter::once(self.first).chain(self.rest.iter().copied())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum TombstoneReason {
    SensitiveViewerAgeVerification,
    UpdateAppIos,
    UpdateAppAndroid,
    LocalRegulations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum LimitedEngagementReason {
    ConversationControl,
    ReadonlyViewer,
    BlockedViewer,
    RootAuthorBlockedViewer,
    StaleTweet,
    CommunityTweetHidden,
    CommunityTweetMemberRemoved,
    CommunityTweetCommunityNotFound,
    CommunityTweetCommunityDeleted,
    CommunityTweetCommunitySuspended,
    CommunityTweetViewerRemoved,
    LocalTweet,
}

impl LimitedEngagementReason {
    pub const fn limited_actions_string(self) -> &'static str {
        match self {
            Self::ConversationControl => "limited_replies",
            Self::ReadonlyViewer => "readonly_viewer",
            Self::BlockedViewer => "blocked_viewer",
            Self::RootAuthorBlockedViewer => "root_author_blocked_viewer",
            Self::StaleTweet => "stale_tweet",
            Self::CommunityTweetHidden => "community_tweet_hidden",
            Self::CommunityTweetMemberRemoved => "community_tweet_member_removed",
            Self::CommunityTweetCommunityNotFound => "community_tweet_community_not_found",
            Self::CommunityTweetCommunityDeleted => "community_tweet_community_deleted",
            Self::CommunityTweetCommunitySuspended => "community_tweet_community_suspended",
            Self::CommunityTweetViewerRemoved => "community_tweet_viewer_removed",
            Self::LocalTweet => "local_tweet",
        }
    }
}

impl Verdict {
    pub const fn not_found() -> Self {
        Self::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::UnspecifiedReason)),
            by: "not_found",
        })
    }

    pub const fn lookup_failed() -> Self {
        Self::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::UnspecifiedReason)),
            by: "lookup_failed",
        })
    }
}

static NOT_FOUND: Verdict = Verdict::not_found();
static LOOKUP_FAILED: Verdict = Verdict::lookup_failed();

#[derive(Clone, Debug, PartialEq)]
pub enum Evaluation {
    Complete {
        verdict: Verdict,
    },
    Partial {
        verdict: Verdict,
        fail_open_defaults: Hydrators,
    },
    NotFound(Lookup),
    Failed(Lookup),
}

impl Evaluation {
    pub fn verdict(&self) -> &Verdict {
        match self {
            Self::Complete { verdict } | Self::Partial { verdict, .. } => verdict,
            Self::NotFound(_) => &NOT_FOUND,
            Self::Failed(_) => &LOOKUP_FAILED,
        }
    }

    pub fn into_verdict(self) -> Verdict {
        match self {
            Self::Complete { verdict } | Self::Partial { verdict, .. } => verdict,
            Self::NotFound(_) => Verdict::not_found(),
            Self::Failed(_) => Verdict::lookup_failed(),
        }
    }

    pub fn fail_open_defaults(&self) -> Hydrators {
        match self {
            Self::Partial {
                fail_open_defaults, ..
            } => *fail_open_defaults,
            Self::Complete { .. } | Self::NotFound(_) | Self::Failed(_) => Hydrators::empty(),
        }
    }
}
