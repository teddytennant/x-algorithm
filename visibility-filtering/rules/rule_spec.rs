use crate::hydration::{Hydrator, Hydrators};
use crate::models::region::allows_country;
use crate::models::{
    ArticleLifecycle, AuthorLabel, DropReason, LimitedEngagementReason, MediaInterstitial,
    MediaRestriction, NsfwViewerDropReason, SafetyLabelType, TombstoneReason, VerifyBlurSupport,
    ViewerProfile,
};
use crate::params::{CountryList, LimitedActionType};
use crate::rules::context::CoreFacts;
use crate::rules::RuleContext;
use std::ops::Not;
use xai_core_entities::entities::ConversationControlArm;
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::{AppealablePolicy, InterstitialAction, InterstitialReason};

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(super) enum RuleId {
    FilterAll,

    SuspendedAuthor,
    DeactivatedAuthor,
    ErasedAuthor,
    OffboardedAuthor,
    ProtectedAuthor,
    ViewerBlocksAuthor,
    ViewerMutesAuthor,
    ViewerMutesRetweets,

    Pdna,
    Bounce,
    Spam,
    ForEmergencyUseOnly,
    FosnrHatefulConduct,
    FosnrViolentSpeech,
    FosnrAbuse,
    FosnrCivicIntegrity,
    FosnrAbuseInsults,
    FosnrAbuseInsultsNonFollower,
    FosnrAbuseInsultsFollower,
    FosnrAuthor,
    FosnrFallback,
    NullcastedTweet,
    StaleTweet,
    LegalTakedown,
    LocalLawsTakedown,
    ArticleTweetContent,
    ProtectedCommunityTweet,
    HiddenCommunityTweet,
    AuthorRemovedCommunityTweet,

    SensitiveViewerLoggedOut,
    SensitiveViewerUnderage,
    SensitiveViewerNoStatedAge,
    NsfwSensitiveViewerTweet,
    NsfwSensitiveViewerUser,

    ExclusiveTweet,
    AuthorBlocksViewerExclusiveContent,
    CreatorTweetNsfw,
    TrustedFriendsTweet,

    NsfwHighPrecision,
    GoreAndViolenceHighPrecision,
    GoreAndViolenceIgnoringSettings,
    NsfwCardImage,
    NsfwAdmin,
    NsfwUser,
    NsfwAccount,
    NsfwReportedHeuristics,
    GoreAndViolenceReportedHeuristics,

    DmcaMedia,
    GeoRestrictedMedia,
    NsfwUserAuthor,
    NsfwAdminAuthor,
    NsfwUserTweetFlag,
    NsfwAdminTweetFlag,
    NsfwHighRecall,
    DoNotAmplify,
    MaliciousUrl,
    SpamHighRecall,
    BrazilElectionLegal,

    NsfwHighRecallUserLabel,
    NsfwHighPrecisionUserLabel,
    NsfwAvatarImageUserLabel,
    NsfwBannerImageUserLabel,
    NsfwNearPerfectUserLabel,
    SpamHighRecallUserLabel,
    CompromisedUserLabel,
    ReadOnlyUserLabel,
    ImpersonationHighPrecisionUserLabel,
    AbusiveHighRecallUserLabel,
    DoNotAmplifyUserLabel,

    BlockedViewer,
    LimitRepliesByInvitation,
    LimitRepliesCommunity,
    LimitRepliesSubscribers,
    LimitRepliesVerified,
    LimitRepliesMyNetwork,
    LimitRepliesCo,
    CommunityTweetViewerRemoved,
    LocalTweet,
    ReadOnlyViewer,
}

#[derive(Clone, PartialEq)]
pub(super) struct RuleClause {
    pub(super) id: RuleId,
    pub(super) when: Vec<Condition>,
    pub(super) applies_to: Audience,
    pub(super) action: ActionSpec,
}

impl RuleClause {
    pub(super) fn name(&self) -> String {
        let id: &str = self.id.into();
        let (kind, reason, prompt) = match &self.action {
            ActionSpec::Drop(DropReason::Legacy(reason)) => {
                let reason = match reason {
                    FilteredReason::UnspecifiedReason => "unspecified".to_owned(),
                    FilteredReason::PossiblyUndesirable => "undesirable".to_owned(),
                    FilteredReason::ContainNsfwMedia => "nsfw_media".to_owned(),
                    FilteredReason::AuthorAccountIsInactive => "inactive".to_owned(),
                    reason @ (FilteredReason::AuthorBlockViewer
                    | FilteredReason::AuthorIsProtected
                    | FilteredReason::AuthorIsUnsafe
                    | FilteredReason::ReportedTweet
                    | FilteredReason::TweetMatchesViewerMutedKeyword(_)
                    | FilteredReason::TweetIsBounced
                    | FilteredReason::SafetyResult(_)
                    | FilteredReason::AuthorIsDeactivated
                    | FilteredReason::AuthorIsSuspended
                    | FilteredReason::ViewerMutesAuthor
                    | FilteredReason::TweetIsNullcast
                    | FilteredReason::ExclusiveTweet
                    | FilteredReason::ViewerBlocksAuthor) => snake(reason),
                };
                ("drop", Some(reason), None)
            }
            ActionSpec::Drop(DropReason::NsfwViewer(reason)) => {
                ("drop", Some(<&str>::from(reason).to_owned()), None)
            }
            ActionSpec::Tombstone(TombstoneReason::SensitiveViewerAgeVerification) => {
                ("tombstone", Some("age_verification".to_owned()), None)
            }
            ActionSpec::Tombstone(reason) => {
                ("tombstone", Some(<&str>::from(reason).to_owned()), None)
            }
            ActionSpec::MediaRestriction(MediaRestriction::MediaInterstitial(blur)) => {
                #[expect(
                    clippy::wildcard_enum_match_arm,
                    reason = "InterstitialReason is a generated Thrift union; every variant but PossiblyUndesirable takes its snake-cased name, including any the IDL adds"
                )]
                let reason = match &blur.reason {
                    InterstitialReason::PossiblyUndesirable(_) => "undesirable".to_owned(),
                    reason => snake(reason),
                };
                let prompt = blur.prompt.map(|prompt| match prompt {
                    InterstitialAction::AGE_VERIFICATION_PROMPT => "age_prompt".to_owned(),
                    prompt => prompt.0.to_string(),
                });
                ("blur", Some(reason), prompt)
            }
            ActionSpec::MediaRestriction(MediaRestriction::NsfwInterstitial) => {
                ("legacy_interstitial", None, None)
            }
            ActionSpec::LimitedEngagement(reason) => (
                "limited_engagement",
                Some(<&str>::from(reason).to_owned()),
                None,
            ),
            ActionSpec::SoftIntervention { policy, .. } => {
                let policy = match *policy {
                    AppealablePolicy::ABUSE => "abuse".to_owned(),
                    policy => policy.0.to_string(),
                };
                ("soft_intervention", Some(policy), None)
            }
            ActionSpec::Appealable(_) => ("appealable", None, None),
        };
        let keeps_reason = matches!(self.action, ActionSpec::SoftIntervention { .. });
        let squashed = |name: &str| name.replace('_', "");
        let reason = reason
            .map(|reason| {
                ["author_is_", "tweet_is_", "is_", "has_"]
                    .iter()
                    .find_map(|prefix| reason.strip_prefix(prefix))
                    .unwrap_or(&reason)
                    .to_owned()
            })
            .filter(|reason| keeps_reason || !squashed(id).contains(&squashed(reason)));
        let mut name = format!("{id}/{kind}");
        for part in [reason, prompt].into_iter().flatten() {
            name.push('/');
            name.push_str(&part);
        }
        name
    }
}

fn snake(value: &impl std::fmt::Debug) -> String {
    let debug = format!("{value:?}");
    let variant = debug.split('(').next().unwrap_or_default();
    let mut snake = String::new();
    for (index, char) in variant.chars().enumerate() {
        if char.is_uppercase() && index > 0 {
            snake.push('_');
        }
        snake.push(char.to_ascii_lowercase());
    }
    snake
}

pub(super) struct Clause {
    when: Vec<Condition>,
    applies_to: Audience,
    action: ActionSpec,
}

pub(super) fn except_author(
    when: impl IntoIterator<Item = Condition>,
    action: ActionSpec,
) -> Clause {
    Clause {
        when: when.into_iter().collect(),
        applies_to: Audience::ExceptAuthor,
        action,
    }
}

pub(super) fn only_author(when: impl IntoIterator<Item = Condition>, action: ActionSpec) -> Clause {
    Clause {
        when: when.into_iter().collect(),
        applies_to: Audience::OnlyAuthor,
        action,
    }
}

pub(super) fn everyone(when: impl IntoIterator<Item = Condition>, action: ActionSpec) -> Clause {
    Clause {
        when: when.into_iter().collect(),
        applies_to: Audience::Everyone,
        action,
    }
}

pub(super) fn only_when(
    condition: Condition,
    clauses: impl IntoIterator<Item = Clause>,
) -> impl Iterator<Item = Clause> {
    clauses.into_iter().map(move |mut clause| {
        clause.when.insert(0, condition);
        clause
    })
}

pub(super) struct Family {
    id: RuleId,
    when: Vec<Condition>,
    clauses: Vec<Clause>,
}

pub(super) fn family(id: RuleId) -> Family {
    Family {
        id,
        when: Vec::new(),
        clauses: Vec::new(),
    }
}

pub(super) fn rule(id: RuleId, clause: Clause) -> Vec<RuleClause> {
    family(id).clause(clause).into()
}

impl Family {
    pub(super) fn when(mut self, when: impl IntoIterator<Item = Condition>) -> Self {
        self.when.extend(when);
        self
    }

    pub(super) fn clause(mut self, clause: Clause) -> Self {
        self.clauses.push(clause);
        self
    }

    pub(super) fn clauses(mut self, clauses: impl IntoIterator<Item = Clause>) -> Self {
        self.clauses.extend(clauses);
        self
    }
}

impl From<Family> for Vec<RuleClause> {
    fn from(family: Family) -> Self {
        family
            .clauses
            .into_iter()
            .map(|clause| RuleClause {
                id: family.id,
                when: family.when.iter().copied().chain(clause.when).collect(),
                applies_to: clause.applies_to,
                action: clause.action,
            })
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub(super) enum Condition {
    Holds(Predicate),
    Not(Predicate),
    AnyOf(&'static [Predicate]),
}

pub(super) const fn tweet(leaf: TweetPredicate) -> Condition {
    Condition::Holds(Predicate::Tweet(leaf))
}

pub(super) const fn label(label: SafetyLabelType) -> Condition {
    Condition::Holds(has_tweet_label(label))
}

pub(super) const fn author(leaf: AuthorPredicate) -> Condition {
    Condition::Holds(Predicate::Author(leaf))
}

pub(super) const fn viewer(leaf: ViewerPredicate) -> Condition {
    Condition::Holds(Predicate::Viewer(leaf))
}

pub(super) const fn relationship(leaf: RelationshipPredicate) -> Condition {
    Condition::Holds(Predicate::Relationship(leaf))
}

#[expect(clippy::panic, reason = "a malformed rule fails at startup")]
pub(super) const fn not(condition: Condition) -> Condition {
    match condition {
        Condition::Holds(leaf) => Condition::Not(leaf),
        Condition::Not(_) | Condition::AnyOf(_) => panic!("`not` negates one leaf"),
    }
}

pub(super) const fn has_tweet_label(label: SafetyLabelType) -> Predicate {
    Predicate::Tweet(TweetPredicate::HasSafetyLabel(label))
}

pub(super) const fn has_user_label(label: AuthorLabel) -> Predicate {
    Predicate::Author(AuthorPredicate::HasUserLabel(label))
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub(super) enum Predicate {
    Tweet(TweetPredicate),
    Author(AuthorPredicate),
    Viewer(ViewerPredicate),
    Relationship(RelationshipPredicate),
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub(super) enum TweetPredicate {
    HasSafetyLabel(SafetyLabelType),
    CreatedAfter(u64),
    NsfwUserFlag,
    NsfwAdminFlag,
    HasMedia,
    HasDmcaMedia,
    IsRetweet,
    IsSupersededEdit,
    LegalTakedownInRequestCountry,
    LocalLawsTakedownInRequestCountry,
    MediaGeoRestrictedInRequestCountry,
    IsNullcast,
    IsCommunityTweet,
    CommunityTweetIsHidden,
    CommunityTweetAuthorIsRemoved,
    HasExclusiveContent,
    IsTrustedFriendsTweet,
    HasArticle,
    ArticleIsPublished,
    HasConversationControl(ConversationControlArm),
    HasNarrowcastPlace,
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub(super) enum AuthorPredicate {
    HasUserLabel(AuthorLabel),
    IsSuspended,
    IsDeactivated,
    IsErased,
    IsOffboarded,
    IsProtected,
    IsNsfwUser,
    IsNsfwAdmin,
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub(super) enum ViewerPredicate {
    LoggedOut,
    Underage,
    NoStatedAge,
    AllowsSensitiveMedia,
    RequestCountryIn(CountryList),
    RequestCountryIs(&'static str),
    AccountOrRequestCountryIn(CountryList),
    AgeVerified,
    ClientVerifyBlurSupportIs(VerifyBlurSupport),
    ClientHasModernBlur,
    ClientHasStaleTweetLimits,
    ClientHasCommunityViewerRemovedLimits,
    ClientBlursGoreIgnoringSettings,
    ClientHasFosnrRules,
    ClientNeedsFosnrFallbackDrops,
    HasVerifiedBadge,
    ReadOnly,
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
#[expect(
    clippy::enum_variant_names,
    reason = "the Viewer prefix identifies the acting subject of each relationship"
)]
pub(super) enum RelationshipPredicate {
    ViewerFollowsAuthor,
    ViewerBlocksAuthor,
    ViewerMutesAuthor,
    ViewerMutesRetweetsFromAuthor,
    ViewerIsConversationAuthor,
    ViewerSuperFollowsAuthor,
    ViewerIsConversationRootAuthor,
    ViewerIsInvitedToConversation,
    ViewerIsFollowedByConversationRootAuthor,
    ViewerIsInConversationRootAuthorNetwork,
    ViewerSuperFollowsConversationRootAuthor,
    ViewerIsBlockedByAuthor,
    ViewerIsBlockedByConversationRootAuthor,
    ViewerIsInAllowedCountry,
    ViewerIsCommunityModerator,
    ViewerIsRemovedFromCommunity,
    ViewerIsTrustedFriendsListMemberOrOwner,
    ViewerIsOutsideNarrowcastPlace,
}

#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(test, derive(Debug))]
pub(super) enum Audience {
    Everyone,
    ExceptAuthor,
    OnlyAuthor,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FosnrViolation {
    pub(super) label: SafetyLabelType,
    pub(super) policy: AppealablePolicy,
    pub(super) level: i8,
    pub(super) limited_actions: &'static [LimitedActionType],
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum ActionSpec {
    Drop(DropReason),
    Tombstone(TombstoneReason),
    MediaRestriction(MediaRestriction),
    SoftIntervention {
        label: SafetyLabelType,
        policy: AppealablePolicy,
        level: i8,
    },
    LimitedEngagement(LimitedEngagementReason),
    Appealable(&'static [FosnrViolation]),
}

pub(super) fn drop_post(reason: FilteredReason) -> ActionSpec {
    ActionSpec::Drop(DropReason::Legacy(reason))
}

pub(super) fn nsfw_viewer_drop(reason: NsfwViewerDropReason) -> ActionSpec {
    ActionSpec::Drop(DropReason::NsfwViewer(reason))
}

pub(super) fn tombstone(reason: TombstoneReason) -> ActionSpec {
    ActionSpec::Tombstone(reason)
}

pub(super) fn limit(reason: LimitedEngagementReason) -> ActionSpec {
    ActionSpec::LimitedEngagement(reason)
}

pub(super) fn soft_intervention(
    label: SafetyLabelType,
    policy: AppealablePolicy,
    level: i8,
) -> ActionSpec {
    ActionSpec::SoftIntervention {
        label,
        policy,
        level,
    }
}

pub(super) const fn appealable(violations: &'static [FosnrViolation]) -> ActionSpec {
    ActionSpec::Appealable(violations)
}

pub(super) fn blur(reason: InterstitialReason) -> ActionSpec {
    media_interstitial(reason, None)
}

pub(super) fn blur_with_age_prompt(reason: InterstitialReason) -> ActionSpec {
    media_interstitial(reason, Some(InterstitialAction::AGE_VERIFICATION_PROMPT))
}

fn media_interstitial(
    reason: InterstitialReason,
    prompt: Option<InterstitialAction>,
) -> ActionSpec {
    ActionSpec::MediaRestriction(MediaRestriction::MediaInterstitial(MediaInterstitial {
        legacy: FilteredReason::ContainNsfwMedia,
        reason,
        prompt,
    }))
}

pub(super) const LEGACY_NSFW_INTERSTITIAL: ActionSpec =
    ActionSpec::MediaRestriction(MediaRestriction::NsfwInterstitial);

impl ActionSpec {
    pub(super) const fn severity(&self) -> u8 {
        match self {
            Self::Appealable(_) => 18,
            Self::Drop(_) => 17,
            Self::Tombstone(_) => 16,
            Self::MediaRestriction(_) => 10,
            Self::SoftIntervention { .. } => 8,
            Self::LimitedEngagement(_) => 6,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Truth {
    True,
    False,
    Unknown { default: bool, failed: Hydrators },
}

impl Truth {
    #[inline]
    pub(super) fn resolves_true(self) -> bool {
        matches!(self, Truth::True | Truth::Unknown { default: true, .. })
    }

    #[inline]
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Truth::False, _) | (_, Truth::False) => Truth::False,
            (Truth::True, other) | (other, Truth::True) => other,
            (
                Truth::Unknown {
                    default: a,
                    failed: x,
                },
                Truth::Unknown {
                    default: b,
                    failed: y,
                },
            ) => Truth::Unknown {
                default: a && b,
                failed: x.union(y),
            },
        }
    }

    #[inline]
    fn or(self, other: Self) -> Self {
        match (self, other) {
            (Truth::True, _) | (_, Truth::True) => Truth::True,
            (Truth::False, other) | (other, Truth::False) => other,
            (
                Truth::Unknown {
                    default: a,
                    failed: x,
                },
                Truth::Unknown {
                    default: b,
                    failed: y,
                },
            ) => Truth::Unknown {
                default: a || b,
                failed: x.union(y),
            },
        }
    }
}

impl Not for Truth {
    type Output = Self;

    #[inline]
    fn not(self) -> Self {
        match self {
            Truth::True => Truth::False,
            Truth::False => Truth::True,
            Truth::Unknown { default, failed } => Truth::Unknown {
                default: !default,
                failed,
            },
        }
    }
}

impl RuleClause {
    pub(super) fn applies(&self, context: &RuleContext<'_>) -> Truth {
        if !self.applies_to.admits(context.facts()) {
            return Truth::False;
        }
        let mut truth = Truth::True;
        for condition in &self.when {
            truth = truth.and(condition.truth(context));
            if truth == Truth::False {
                break;
            }
        }
        truth
    }

    pub(super) fn hydrators(&self) -> Hydrators {
        self.when
            .iter()
            .fold(Hydrators::empty(), |hydrators, condition| {
                hydrators.union(condition.hydrators())
            })
    }
}

impl Condition {
    pub(super) const fn hydrators(&self) -> Hydrators {
        match self {
            Condition::Holds(leaf) | Condition::Not(leaf) => leaf.hydrators(),
            Condition::AnyOf(leaves) => {
                let mut hydrators = Hydrators::empty();
                let mut rest = *leaves;
                while let [leaf, tail @ ..] = rest {
                    hydrators = hydrators.union(leaf.hydrators());
                    rest = tail;
                }
                hydrators
            }
        }
    }
}

impl Predicate {
    pub(super) const fn hydrators(self) -> Hydrators {
        match self {
            Predicate::Tweet(fact) => fact.hydrators(),
            Predicate::Author(fact) => fact.hydrators(),
            Predicate::Viewer(fact) => fact.hydrators(),
            Predicate::Relationship(fact) => fact.hydrators(),
        }
    }
}

impl Audience {
    #[inline]
    pub(super) fn admits(self, facts: CoreFacts<'_>) -> bool {
        match self {
            Audience::Everyone => true,
            Audience::ExceptAuthor => !facts.is_author_viewer(),
            Audience::OnlyAuthor => facts.is_author_viewer(),
        }
    }
}

impl Condition {
    #[inline]
    fn truth(&self, context: &RuleContext<'_>) -> Truth {
        match self {
            Condition::Holds(leaf) => leaf.truth(context),
            Condition::Not(leaf) => !leaf.truth(context),
            Condition::AnyOf(leaves) => {
                let mut truth = Truth::False;
                for leaf in *leaves {
                    truth = truth.or(leaf.truth(context));
                    if truth == Truth::True {
                        break;
                    }
                }
                truth
            }
        }
    }
}

impl Predicate {
    #[inline]
    fn truth(self, context: &RuleContext<'_>) -> Truth {
        let value = self.holds(context);
        let failed = context.failed();
        let failed = if failed.is_empty() {
            failed
        } else {
            failed.intersection(self.hydrators())
        };
        match (failed.is_empty(), value) {
            (false, default) => Truth::Unknown { default, failed },
            (true, true) => Truth::True,
            (true, false) => Truth::False,
        }
    }

    #[inline]
    pub(super) fn holds(self, context: &RuleContext<'_>) -> bool {
        match self {
            Predicate::Tweet(fact) => fact.holds(context),
            Predicate::Author(fact) => fact.holds(context),
            Predicate::Viewer(fact) => fact.holds(context),
            Predicate::Relationship(fact) => fact.holds(context),
        }
    }
}

macro_rules! predicates {
    ($(
        $predicate:ident {
            $($variant:ident $(($($arg:ident),*))? reads $reads:tt
                => |$facts:pat_param, $value:pat_param| $body:expr),+ $(,)?
        }
    )+) => {$(
        impl $predicate {
            const fn hydrators(self) -> Hydrators {
                match self {
                    $(Self::$variant { .. } => predicates!(@declare $reads),)+
                }
            }

            #[inline]
            fn holds(self, context: &RuleContext<'_>) -> bool {
                match self {
                    $(Self::$variant $(($($arg),*))? => {
                        let $facts = context.facts();
                        let $value = predicates!(@read context $reads);
                        $body
                    })+
                }
            }
        }
    )+};
    (@declare ()) => { Hydrators::empty() };
    (@declare ($($node:ident),+)) => { Hydrators::empty()$(.with(Hydrator::$node))+ };
    (@declare $node:ident) => { Hydrators::of(Hydrator::$node) };
    (@read $context:ident ()) => { () };
    (@read $context:ident ($($node:ident),+)) => { ($(predicates!(@read $context $node)),+) };
    (@read $context:ident PureCore) => { $context.source_tweet_id() };
    (@read $context:ident Tweet) => { $context.tweet_features() };
    (@read $context:ident ConversationControl) => { $context.conversation_control() };
    (@read $context:ident TweetSafetyLabels) => { $context.tweet_safety_labels() };
    (@read $context:ident ViewerProfile) => { $context.viewer_profile() };
    (@read $context:ident ViewerLabels) => { $context.viewer_has_age_verified_18_label() };
    (@read $context:ident AuthorSafety) => { $context.author_features() };
    (@read $context:ident AuthorLabels) => { $context.author_labels() };
    (@read $context:ident ViewerCountry) => { $context.viewer_country() };
    (@read $context:ident CommunityModeration) => { $context.community_moderation() };
    (@read $context:ident CommunityModerator) => { $context.viewer_is_community_moderator() };
    (@read $context:ident CommunityViewerRemoved) => {
        $context.viewer_is_removed_from_community()
    };
    (@read $context:ident ArticleLifecycle) => { $context.article_lifecycle() };
    (@read $context:ident $edge:ident) => {{
        const { assert!(Hydrator::$edge.is_edge()) };
        $context.edge(Hydrator::$edge)
    }};
}

predicates! {
    TweetPredicate {
        HasSafetyLabel(label) reads TweetSafetyLabels => |_, labels| labels.has_label(label),
        CreatedAfter(unix_ms) reads () => |facts, ()| facts.created_after(unix_ms),
        NsfwUserFlag reads Tweet => |_, tweet| tweet.nsfw.user,
        NsfwAdminFlag reads Tweet => |_, tweet| tweet.nsfw.admin,
        HasMedia reads Tweet => |_, tweet| tweet.has_media(),
        HasDmcaMedia reads Tweet => |_, tweet| tweet.has_dmca_media(),
        IsRetweet reads PureCore => |_, source_tweet_id| source_tweet_id.is_some(),
        IsSupersededEdit reads Tweet => |facts, tweet| tweet.is_superseded_edit(facts.tweet_id()),
        LegalTakedownInRequestCountry reads Tweet
            => |facts, tweet| tweet.legal_takedown_in(facts.request_country()),
        LocalLawsTakedownInRequestCountry reads Tweet
            => |facts, tweet| tweet.local_laws_takedown_in(facts.request_country()),
        MediaGeoRestrictedInRequestCountry reads Tweet
            => |facts, tweet| tweet.media_restricted_in(facts.request_country()),
        IsNullcast reads Tweet => |_, tweet| tweet.is_nullcast,
        IsCommunityTweet reads Tweet => |_, tweet| tweet.community_id.is_some(),
        CommunityTweetIsHidden reads CommunityModeration => |_, moderation| moderation.is_hidden,
        CommunityTweetAuthorIsRemoved reads CommunityModeration
            => |_, moderation| moderation.is_author_removed,
        HasExclusiveContent reads Tweet
            => |_, tweet| tweet.exclusive_conversation_author_id.is_some(),
        IsTrustedFriendsTweet reads Tweet => |_, tweet| tweet.trusted_friends_list_id.is_some(),
        HasArticle reads Tweet => |_, tweet| tweet.article_id.is_some(),
        ArticleIsPublished reads ArticleLifecycle
            => |_, lifecycle| lifecycle == Some(ArticleLifecycle::Published),
        HasConversationControl(arm) reads ConversationControl
            => |_, control| control.is_some_and(|control| control.arm == arm),
        HasNarrowcastPlace reads Tweet => |_, tweet| tweet.narrowcast_place_id.is_some(),
    }

    AuthorPredicate {
        HasUserLabel(label) reads AuthorLabels => |_, labels| labels.has_label(label),
        IsSuspended reads AuthorSafety => |_, author| author.is_suspended,
        IsDeactivated reads AuthorSafety => |_, author| author.is_deactivated,
        IsErased reads AuthorSafety => |_, author| author.is_erased,
        IsOffboarded reads AuthorSafety => |_, author| author.is_offboarded,
        IsProtected reads AuthorSafety => |_, author| author.is_protected,
        IsNsfwUser reads AuthorSafety => |_, author| author.is_nsfw_user,
        IsNsfwAdmin reads AuthorSafety => |_, author| author.is_nsfw_admin,
    }

    ViewerPredicate {
        LoggedOut reads () => |facts, ()| facts.viewer_id().is_none(),
        Underage reads ViewerProfile => |_, profile| profile.is_some_and(ViewerProfile::is_underage),
        NoStatedAge reads ViewerProfile
            => |_, profile| profile.is_some_and(ViewerProfile::has_no_stated_age),
        AllowsSensitiveMedia reads ViewerProfile
            => |_, profile| profile.is_some_and(|profile| profile.allows_sensitive_media),
        RequestCountryIn(list) reads () => |facts, ()| {
            facts
                .request_country()
                .is_some_and(|country| facts.in_country_list(list, country))
        },
        RequestCountryIs(country) reads () => |facts, ()| facts.request_country() == Some(country),
        AccountOrRequestCountryIn(list) reads ViewerProfile => |facts, profile| {
            profile
                .and_then(|profile| profile.account_country_code.as_deref())
                .or(facts.request_country())
                .is_some_and(|country| facts.in_country_list(list, country))
        },
        AgeVerified reads (ViewerProfile, ViewerLabels) => |_, (profile, age_verified_18)| {
            age_verified_18 || profile.is_some_and(|profile| profile.has_idv_premium)
        },
        ClientVerifyBlurSupportIs(support) reads ()
            => |facts, ()| facts.client_capability().verify_blur_support == Some(support),
        ClientHasModernBlur reads () => |facts, ()| facts.client_capability().modern_blur,
        ClientHasStaleTweetLimits reads ()
            => |facts, ()| facts.client_capability().stale_tweet_limits,
        ClientHasCommunityViewerRemovedLimits reads ()
            => |facts, ()| facts.client_capability().community_viewer_removed_limits,
        ClientBlursGoreIgnoringSettings reads ()
            => |facts, ()| facts.client_capability().gore_blur_ignores_settings,
        ClientHasFosnrRules reads () => |facts, ()| facts.client_capability().fosnr_rules,
        ClientNeedsFosnrFallbackDrops reads ()
            => |facts, ()| facts.client_capability().fosnr_fallback_drops,
        HasVerifiedBadge reads ViewerProfile
            => |_, profile| profile.is_some_and(|profile| profile.has_verified_badge),
        ReadOnly reads ViewerProfile
            => |_, profile| profile.is_some_and(|profile| profile.is_read_only),
    }

    RelationshipPredicate {
        ViewerFollowsAuthor reads Follows => |_, follows| follows,
        ViewerBlocksAuthor reads Blocks => |_, blocks| blocks,
        ViewerMutesAuthor reads Mutes => |_, mutes| mutes,
        ViewerMutesRetweetsFromAuthor reads MuteRetweets => |_, mutes| mutes,
        ViewerIsConversationAuthor reads Tweet => |facts, tweet| {
            tweet
                .exclusive_conversation_author_id
                .is_some_and(|author| facts.viewer_id() == Some(author))
        },
        ViewerSuperFollowsAuthor reads (Tweet, SuperFollowsExclusive)
            => |_, (tweet, super_follows)| {
                tweet.exclusive_conversation_author_id.is_some() && super_follows
            },
        ViewerIsConversationRootAuthor reads ConversationControl => |facts, control| {
            control
                .zip(facts.viewer_id())
                .is_some_and(|(control, viewer_id)| viewer_id == control.conversation_tweet_author_id)
        },
        ViewerIsInvitedToConversation reads ConversationControl => |facts, control| {
            control
                .zip(facts.viewer_id())
                .is_some_and(|(control, viewer_id)| control.invited_user_ids.contains(&viewer_id))
        },
        ViewerIsFollowedByConversationRootAuthor reads RootFollowsViewer
            => |_, follows| follows,
        ViewerIsInConversationRootAuthorNetwork reads (RootFollowsViewer, RootFollowsViewerSecondDegree)
            => |_, (first, second)| first || second,
        ViewerSuperFollowsConversationRootAuthor reads SuperFollowsRoot
            => |_, super_follows| super_follows,
        ViewerIsBlockedByAuthor reads BlockedByAuthor => |_, blocked| blocked,
        ViewerIsBlockedByConversationRootAuthor reads BlockedByReplyRoot => |_, blocked| blocked,
        ViewerIsInAllowedCountry reads (ConversationControl, ViewerCountry)
            => |_, (control, country)| {
                control.zip(country).is_some_and(|(control, country)| {
                    allows_country(&control.allowed_country_codes, country)
                })
            },
        ViewerIsCommunityModerator reads CommunityModerator
            => |facts, is_moderator| facts.viewer_id().is_some() && is_moderator.unwrap_or(true),
        ViewerIsRemovedFromCommunity reads CommunityViewerRemoved => |_, removed| removed,
        ViewerIsTrustedFriendsListMemberOrOwner reads TrustedFriends => |_, holds| holds,
        ViewerIsOutsideNarrowcastPlace reads OutsideNarrowcastPlace => |_, outside| outside,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        ClientCapability, ConversationControlFeatures, HydratedTweetCandidate, Viewer,
        ViewerFeatures,
    };
    use crate::rules::fixtures::{
        candidate, logged_out_viewer, viewer, viewer_with_profile, VIEWER_ID,
    };
    use crate::rules::{holds_narrowed, test_context};
    use xai_core_entities::entities::ConversationControl;

    const ROOT_AUTHOR_ID: u64 = 4242;

    fn controlled(
        arm: ConversationControlArm,
        invited_user_ids: Vec<u64>,
        edges: &[Hydrator],
    ) -> HydratedTweetCandidate {
        edges
            .iter()
            .fold(candidate(), |candidate, &edge| candidate.with_edge(edge))
            .with_conversation_control(ConversationControlFeatures {
                control: ConversationControl {
                    arm,
                    conversation_tweet_author_id: ROOT_AUTHOR_ID,
                    invited_user_ids,
                    invite_via_mention: None,
                    allowed_country_codes: vec![],
                },
                viewer_country: None,
            })
            .build()
    }

    #[test]
    fn conversation_control_predicates_read_the_root_keyed_features() {
        use ConversationControlArm::{ByInvitation, Community, Subscribers};
        use RelationshipPredicate::{
            ViewerIsConversationRootAuthor, ViewerIsFollowedByConversationRootAuthor,
            ViewerIsInvitedToConversation, ViewerSuperFollowsConversationRootAuthor,
        };
        let community = controlled(Community, vec![], &[Hydrator::RootFollowsViewer]);
        let subscribers = controlled(Subscribers, vec![], &[Hydrator::SuperFollowsRoot]);
        let invitation = controlled(ByInvitation, vec![VIEWER_ID], &[]);
        let unrelated = controlled(Community, vec![], &[]);
        let uncontrolled = candidate().build();
        let root_author = viewer(ROOT_AUTHOR_ID);
        let viewer = viewer(VIEWER_ID);
        let logged_out = logged_out_viewer();
        for (predicate, viewer, candidate, expected) in [
            (
                Predicate::Tweet(TweetPredicate::HasConversationControl(Community)),
                &viewer,
                &community,
                true,
            ),
            (
                Predicate::Tweet(TweetPredicate::HasConversationControl(Community)),
                &viewer,
                &uncontrolled,
                false,
            ),
            (
                Predicate::Relationship(ViewerIsConversationRootAuthor),
                &root_author,
                &community,
                true,
            ),
            (
                Predicate::Relationship(ViewerIsConversationRootAuthor),
                &logged_out,
                &community,
                false,
            ),
            (
                Predicate::Relationship(ViewerIsInvitedToConversation),
                &viewer,
                &invitation,
                true,
            ),
            (
                Predicate::Relationship(ViewerIsFollowedByConversationRootAuthor),
                &viewer,
                &community,
                true,
            ),
            (
                Predicate::Relationship(ViewerIsFollowedByConversationRootAuthor),
                &viewer,
                &unrelated,
                false,
            ),
            (
                Predicate::Relationship(ViewerSuperFollowsConversationRootAuthor),
                &viewer,
                &subscribers,
                true,
            ),
            (
                Predicate::Relationship(ViewerSuperFollowsConversationRootAuthor),
                &viewer,
                &unrelated,
                false,
            ),
            (
                Predicate::Relationship(ViewerSuperFollowsConversationRootAuthor),
                &viewer,
                &uncontrolled,
                false,
            ),
        ] {
            assert_eq!(holds_narrowed(predicate, viewer, candidate), expected);
        }
    }

    #[test]
    fn created_after_is_strict_at_the_unix_millisecond_boundary() {
        const CUTOFF_MS: u64 = 1705536000000;
        const AT_CUTOFF: u64 = (CUTOFF_MS - 1288834974657) << 22;
        let viewer = ViewerFeatures::default();
        for (tweet_id, expected) in [
            (AT_CUTOFF - 1, false),
            (AT_CUTOFF, false),
            (AT_CUTOFF + (1 << 22) - 1, false),
            (AT_CUTOFF + (1 << 22), true),
            (0, false),
        ] {
            let candidate = HydratedTweetCandidate {
                tweet_id,
                ..Default::default()
            };
            let context = test_context(&viewer, &candidate);
            assert_eq!(
                TweetPredicate::CreatedAfter(CUTOFF_MS).holds(&context),
                expected,
                "{tweet_id}"
            );
        }
    }

    #[test]
    fn age_verified_is_the_label_or_idv_premium_and_the_request_country_is_never_unknown() {
        use ViewerPredicate::{AgeVerified, RequestCountryIn};
        let in_country = |country: Option<&str>, profile: ViewerProfile| ViewerFeatures {
            country_code: country.map(str::to_string),
            ..viewer_with_profile(profile)
        };
        let labeled = ViewerFeatures {
            viewer: Viewer::LoggedIn {
                id: VIEWER_ID,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: true,
            },
            ..ViewerFeatures::default()
        };
        let idv_premium = ViewerProfile {
            has_idv_premium: true,
            ..ViewerProfile::default()
        };
        let account_in_fr = ViewerProfile {
            account_country_code: Some("fr".into()),
            ..ViewerProfile::default()
        };
        let request_country_in = RequestCountryIn(CountryList::AgeVerification);
        let candidate = candidate().build();
        for (predicate, viewer, expected) in [
            (AgeVerified, labeled, true),
            (AgeVerified, in_country(None, idv_premium), true),
            (
                AgeVerified,
                in_country(None, ViewerProfile::default()),
                false,
            ),
            (AgeVerified, logged_out_viewer(), false),
            (
                request_country_in,
                in_country(Some("fr"), ViewerProfile::default()),
                true,
            ),
            (
                request_country_in,
                in_country(Some("us"), account_in_fr),
                false,
            ),
            (
                request_country_in,
                in_country(None, ViewerProfile::default()),
                false,
            ),
        ] {
            assert_eq!(
                holds_narrowed(Predicate::Viewer(predicate), &viewer, &candidate),
                expected
            );
        }

        let failed = HydratedTweetCandidate {
            failed: Hydrators::all(),
            ..candidate
        };
        let viewer = in_country(Some("fr"), ViewerProfile::default());
        let context = test_context(&viewer, &failed);
        assert_eq!(
            Predicate::Viewer(AgeVerified).truth(&context),
            Truth::Unknown {
                default: false,
                failed: Hydrators::of(Hydrator::ViewerProfile).with(Hydrator::ViewerLabels),
            }
        );
        assert_eq!(
            Predicate::Viewer(request_country_in).truth(&context),
            Truth::True
        );
    }

    #[test]
    fn the_client_checks_read_the_resolved_capability_and_are_never_unknown() {
        use VerifyBlurSupport::{IosNeedsUpdate, Supported};
        let viewer = ViewerFeatures {
            client_capability: ClientCapability {
                verify_blur_support: Some(IosNeedsUpdate),
                modern_blur: true,
                stale_tweet_limits: true,
                community_viewer_removed_limits: true,
                gore_blur_ignores_settings: true,
                fosnr_rules: true,
                fosnr_fallback_drops: true,
            },
            ..viewer(VIEWER_ID)
        };
        let failed = HydratedTweetCandidate {
            failed: Hydrators::all(),
            ..candidate().build()
        };
        let context = test_context(&viewer, &failed);
        for (support, expected) in [(IosNeedsUpdate, Truth::True), (Supported, Truth::False)] {
            let check = Predicate::Viewer(ViewerPredicate::ClientVerifyBlurSupportIs(support));
            assert_eq!(check.truth(&context), expected, "{support:?}");
        }
        for check in [
            ViewerPredicate::ClientHasModernBlur,
            ViewerPredicate::ClientHasCommunityViewerRemovedLimits,
        ] {
            assert_eq!(
                Predicate::Viewer(check).truth(&context),
                Truth::True,
                "{check:?}"
            );
        }
    }
}
