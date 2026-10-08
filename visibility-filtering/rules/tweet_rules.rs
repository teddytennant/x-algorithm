use crate::models::{
    AuthorLabel, LimitedEngagementReason, NsfwViewerDropReason, SafetyLabelType, TombstoneReason,
    VerifyBlurSupport,
};
use crate::params::{CountryList, LimitedActionType};
use crate::rules::rule_spec::{
    appealable, author, blur, blur_with_age_prompt, drop_post, everyone, except_author, family,
    has_tweet_label, has_user_label, label, limit, not, nsfw_viewer_drop, only_author, only_when,
    relationship, rule, soft_intervention, tombstone, tweet, viewer, AuthorPredicate, Clause,
    Condition, FosnrViolation, Predicate, RelationshipPredicate, RuleClause, RuleId,
    TweetPredicate, ViewerPredicate, LEGACY_NSFW_INTERSTITIAL,
};
use xai_core_entities::entities::ConversationControlArm;
use xai_visibility_filtering::models::{Action, FilteredReason, SafetyResult, SafetyResultReason};
use xai_x_thrift::action::{AppealablePolicy, InterstitialReason};

pub(super) const NSFW_HIGH_PRECISION_REASON: FilteredReason =
    FilteredReason::SafetyResult(SafetyResult {
        reason: Some(SafetyResultReason::NsfwHighPrecision),
        action: Action::Drop(xai_visibility_filtering::models::DropReason {}),
    });

const HAS_MEDIA: Condition = tweet(TweetPredicate::HasMedia);
const NOT_RETWEET: Condition = not(tweet(TweetPredicate::IsRetweet));
const SENSITIVE_MEDIA_DISABLED: Condition = not(viewer(ViewerPredicate::AllowsSensitiveMedia));
const LOGGED_OUT: Condition = viewer(ViewerPredicate::LoggedOut);
const UNDERAGE: Condition = viewer(ViewerPredicate::Underage);
const NO_STATED_AGE: Condition = viewer(ViewerPredicate::NoStatedAge);
const IN_NSFW_GATING_COUNTRY: Condition = viewer(ViewerPredicate::AccountOrRequestCountryIn(
    CountryList::NsfwGating,
));
const NSFW_MEDIA_LABEL: Condition = Condition::AnyOf(&[
    has_tweet_label(SafetyLabelType::NSFW_HIGH_PRECISION),
    has_tweet_label(SafetyLabelType::NSFW_HIGH_RECALL),
    has_tweet_label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION),
]);
const NSFW_FLAGGED: Condition = Condition::AnyOf(&[
    Predicate::Author(AuthorPredicate::IsNsfwUser),
    Predicate::Author(AuthorPredicate::IsNsfwAdmin),
    Predicate::Tweet(TweetPredicate::NsfwUserFlag),
    Predicate::Tweet(TweetPredicate::NsfwAdminFlag),
]);
const NSFW_TEXT_OR_CARD_LABEL: Condition = Condition::AnyOf(&[
    has_tweet_label(SafetyLabelType::NSFW_TEXT),
    has_tweet_label(SafetyLabelType::NSFW_CARD_IMAGE),
]);
const HAS_EXCLUSIVE_CONTENT: Condition = tweet(TweetPredicate::HasExclusiveContent);
const NOT_CONVERSATION_AUTHOR: Condition = not(relationship(
    RelationshipPredicate::ViewerIsConversationAuthor,
));
const NOT_SUPER_FOLLOWER: Condition = not(relationship(
    RelationshipPredicate::ViewerSuperFollowsAuthor,
));
const NOT_LOGGED_OUT: Condition = not(viewer(ViewerPredicate::LoggedOut));
const NOT_CONVERSATION_ROOT_AUTHOR: Condition = not(relationship(
    RelationshipPredicate::ViewerIsConversationRootAuthor,
));
const NOT_INVITED_TO_CONVERSATION: Condition = not(relationship(
    RelationshipPredicate::ViewerIsInvitedToConversation,
));

pub(super) fn protected_community_tweet_drop() -> Vec<RuleClause> {
    rule(
        RuleId::ProtectedCommunityTweet,
        except_author(
            [
                tweet(TweetPredicate::IsCommunityTweet),
                author(AuthorPredicate::IsProtected),
            ],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

pub(super) fn hidden_community_tweet_drop() -> Vec<RuleClause> {
    moderated_community_tweet_drop(
        RuleId::HiddenCommunityTweet,
        TweetPredicate::CommunityTweetIsHidden,
    )
}

pub(super) fn author_removed_community_tweet_drop() -> Vec<RuleClause> {
    moderated_community_tweet_drop(
        RuleId::AuthorRemovedCommunityTweet,
        TweetPredicate::CommunityTweetAuthorIsRemoved,
    )
}

fn moderated_community_tweet_drop(id: RuleId, moderated: TweetPredicate) -> Vec<RuleClause> {
    rule(
        id,
        except_author(
            [
                tweet(TweetPredicate::IsCommunityTweet),
                tweet(moderated),
                not(relationship(
                    RelationshipPredicate::ViewerIsCommunityModerator,
                )),
            ],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

fn pdna_tweet_label_drop() -> Vec<RuleClause> {
    rule(
        RuleId::Pdna,
        except_author(
            [label(SafetyLabelType::PDNA)],
            drop_post(NSFW_HIGH_PRECISION_REASON),
        ),
    )
}

fn bounce_tweet_label_drop() -> Vec<RuleClause> {
    rule(
        RuleId::Bounce,
        except_author(
            [label(SafetyLabelType::BOUNCE)],
            drop_post(FilteredReason::TweetIsBounced),
        ),
    )
}

fn spam_tweet_label_drop() -> Vec<RuleClause> {
    rule(
        RuleId::Spam,
        except_author(
            [label(SafetyLabelType::SPAM)],
            drop_post(FilteredReason::PossiblyUndesirable),
        ),
    )
}

fn for_emergency_use_only_drop() -> Vec<RuleClause> {
    rule(
        RuleId::ForEmergencyUseOnly,
        everyone(
            [label(SafetyLabelType::FOR_EMERGENCY_USE_ONLY)],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

pub(super) fn tweet_label_drops() -> Vec<RuleClause> {
    [
        pdna_tweet_label_drop(),
        bounce_tweet_label_drop(),
        spam_tweet_label_drop(),
        for_emergency_use_only_drop(),
    ]
    .concat()
}

pub(super) fn home_hydration_tweet_label_drops() -> Vec<RuleClause> {
    [
        spam_tweet_label_drop(),
        pdna_tweet_label_drop(),
        bounce_tweet_label_drop(),
        for_emergency_use_only_drop(),
    ]
    .concat()
}

fn label_drop(id: RuleId, label_type: SafetyLabelType, reason: FilteredReason) -> Vec<RuleClause> {
    rule(id, except_author([label(label_type)], drop_post(reason)))
}

struct FosnrLevel3 {
    id: RuleId,
    label: SafetyLabelType,
    policy: AppealablePolicy,
}

const FOSNR_LEVEL_3: [FosnrLevel3; 4] = [
    FosnrLevel3 {
        id: RuleId::FosnrHatefulConduct,
        label: SafetyLabelType::FOSNR_HATEFUL_CONDUCT,
        policy: AppealablePolicy::HATEFUL_CONDUCT,
    },
    FosnrLevel3 {
        id: RuleId::FosnrViolentSpeech,
        label: SafetyLabelType::FOSNR_VIOLENT_SPEECH,
        policy: AppealablePolicy::VIOLENT_SPEECH,
    },
    FosnrLevel3 {
        id: RuleId::FosnrAbuse,
        label: SafetyLabelType::FOSNR_ABUSE,
        policy: AppealablePolicy::ABUSE,
    },
    FosnrLevel3 {
        id: RuleId::FosnrCivicIntegrity,
        label: SafetyLabelType::FOSNR_CIVIC_INTEGRITY,
        policy: AppealablePolicy::CIVIC_INTEGRITY,
    },
];

const FOSNR_LEVEL_1: SafetyLabelType = SafetyLabelType::FOSNR_ABUSE_INSULTS;

const FOSNR: [Predicate; 5] = {
    let [hateful_conduct, violent_speech, abuse, civic_integrity] = FOSNR_LEVEL_3;
    [
        has_tweet_label(hateful_conduct.label),
        has_tweet_label(violent_speech.label),
        has_tweet_label(abuse.label),
        has_tweet_label(civic_integrity.label),
        has_tweet_label(FOSNR_LEVEL_1),
    ]
};

pub(super) fn fosnr_level_3_drops() -> Vec<RuleClause> {
    FOSNR_LEVEL_3
        .into_iter()
        .flat_map(|FosnrLevel3 { id, label, .. }| {
            label_drop(id, label, FilteredReason::PossiblyUndesirable)
        })
        .collect()
}

const NSFW_HIGH_PRECISION_CHANGED_AT: u64 = 1705536000000;

const NSFW_HIGH_PRECISION: Condition = label(SafetyLabelType::NSFW_HIGH_PRECISION);
const CREATED_AFTER_NSFW_HIGH_PRECISION_CHANGE: Condition =
    tweet(TweetPredicate::CreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT));
const CREATED_BEFORE_NSFW_HIGH_PRECISION_CHANGE: Condition = not(tweet(
    TweetPredicate::CreatedAfter(NSFW_HIGH_PRECISION_CHANGED_AT),
));
const GORE_AND_VIOLENCE_HIGH_PRECISION: Condition =
    label(SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION);
const NSFW_CARD_IMAGE: Condition = label(SafetyLabelType::NSFW_CARD_IMAGE);
const NSFW_REPORTED_HEURISTICS: Condition = label(SafetyLabelType::NSFW_REPORTED_HEURISTICS);
const GORE_AND_VIOLENCE_REPORTED_HEURISTICS: Condition =
    label(SafetyLabelType::GORE_AND_VIOLENCE_REPORTED_HEURISTICS);
const NSFW_ADMIN: Condition = Condition::AnyOf(&[
    Predicate::Author(AuthorPredicate::IsNsfwAdmin),
    Predicate::Tweet(TweetPredicate::NsfwAdminFlag),
]);
const NSFW_USER: Condition = Condition::AnyOf(&[
    Predicate::Author(AuthorPredicate::IsNsfwUser),
    Predicate::Tweet(TweetPredicate::NsfwUserFlag),
]);
const TWEET_NSFW_ADMIN: Condition = tweet(TweetPredicate::NsfwAdminFlag);
const TWEET_NSFW_USER: Condition = tweet(TweetPredicate::NsfwUserFlag);
const TWEET_NSFW_FLAGGED: Condition = Condition::AnyOf(&[
    Predicate::Tweet(TweetPredicate::NsfwAdminFlag),
    Predicate::Tweet(TweetPredicate::NsfwUserFlag),
]);

const CLIENT_HAS_VERIFY_BLUR: Condition = viewer(ViewerPredicate::ClientVerifyBlurSupportIs(
    VerifyBlurSupport::Supported,
));
const REQUEST_IS_FROM_AGE_VERIFICATION_COUNTRIES: Condition = viewer(
    ViewerPredicate::RequestCountryIn(CountryList::AgeVerification),
);
const VIEWER_IS_NOT_AGE_VERIFIED: Condition = not(viewer(ViewerPredicate::AgeVerified));
const REQUEST_IS_FROM_LOCAL_REGULATIONS_COUNTRIES: Condition = viewer(
    ViewerPredicate::RequestCountryIn(CountryList::LocalRegulations),
);
const MODERN_BLUR_CLIENT: Condition = viewer(ViewerPredicate::ClientHasModernBlur);
const LEGACY_INTERSTITIAL_CLIENT: Condition = not(viewer(ViewerPredicate::ClientHasModernBlur));
const GORE_BLUR_IGNORES_SETTINGS_CLIENT: Condition =
    viewer(ViewerPredicate::ClientBlursGoreIgnoringSettings);

fn sensitive_media_blur(reason: InterstitialReason) -> Clause {
    except_author([SENSITIVE_MEDIA_DISABLED], blur(reason))
}

fn modern_blur(reason: InterstitialReason) -> Clause {
    everyone([SENSITIVE_MEDIA_DISABLED, MODERN_BLUR_CLIENT], blur(reason))
}

fn age_prompt_blur(reason: InterstitialReason) -> Clause {
    except_author(
        [
            CLIENT_HAS_VERIFY_BLUR,
            REQUEST_IS_FROM_AGE_VERIFICATION_COUNTRIES,
            VIEWER_IS_NOT_AGE_VERIFIED,
        ],
        blur_with_age_prompt(reason),
    )
}

fn legacy_client_interstitial() -> Clause {
    everyone([LEGACY_INTERSTITIAL_CLIENT], LEGACY_NSFW_INTERSTITIAL)
}

fn age_gated_blurs(reason: InterstitialReason) -> [Clause; 3] {
    [
        age_prompt_blur(reason.clone()),
        modern_blur(reason),
        legacy_client_interstitial(),
    ]
}

#[derive(Clone, Copy)]
enum Rungs {
    All,
    UpdateApp,
}

fn age_gate_ladder(countries: CountryList, rungs: Rungs) -> Vec<Clause> {
    use VerifyBlurSupport::{AndroidNeedsUpdate, IosNeedsUpdate, Unsupported};
    let generic = match rungs {
        Rungs::All => Some((Unsupported, TombstoneReason::SensitiveViewerAgeVerification)),
        Rungs::UpdateApp => None,
    };
    generic
        .into_iter()
        .chain([
            (IosNeedsUpdate, TombstoneReason::UpdateAppIos),
            (AndroidNeedsUpdate, TombstoneReason::UpdateAppAndroid),
        ])
        .map(|(client, reason)| {
            except_author(
                [
                    viewer(ViewerPredicate::ClientVerifyBlurSupportIs(client)),
                    viewer(ViewerPredicate::RequestCountryIn(countries)),
                    VIEWER_IS_NOT_AGE_VERIFIED,
                ],
                tombstone(reason),
            )
        })
        .collect()
}

pub(super) mod nsfw_high_precision {
    use super::*;

    pub(in crate::rules) fn blurs() -> Vec<RuleClause> {
        use InterstitialReason::{Nudity, Sensitive};
        family(RuleId::NsfwHighPrecision)
            .when([NSFW_HIGH_PRECISION])
            .clauses(only_when(
                CREATED_AFTER_NSFW_HIGH_PRECISION_CHANGE,
                [sensitive_media_blur(Nudity(true))],
            ))
            .clauses(only_when(
                CREATED_BEFORE_NSFW_HIGH_PRECISION_CHANGE,
                [sensitive_media_blur(Sensitive(true))],
            ))
            .into()
    }

    pub(in crate::rules) fn oon_drop() -> Vec<RuleClause> {
        rule(
            RuleId::NsfwHighPrecision,
            except_author(
                [NSFW_HIGH_PRECISION],
                drop_post(FilteredReason::ContainNsfwMedia),
            ),
        )
    }

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        use InterstitialReason::{Nudity, Sensitive};
        family(RuleId::NsfwHighPrecision)
            .when([NSFW_HIGH_PRECISION])
            .clauses(only_when(
                CREATED_BEFORE_NSFW_HIGH_PRECISION_CHANGE,
                [
                    age_prompt_blur(Sensitive(true)),
                    modern_blur(Sensitive(true)),
                ],
            ))
            .clauses(only_when(
                CREATED_AFTER_NSFW_HIGH_PRECISION_CHANGE,
                [age_prompt_blur(Nudity(true)), modern_blur(Nudity(true))],
            ))
            .clause(except_author(
                [REQUEST_IS_FROM_LOCAL_REGULATIONS_COUNTRIES],
                tombstone(TombstoneReason::LocalRegulations),
            ))
            .clause(legacy_client_interstitial())
            .clauses(age_gate_ladder(CountryList::Tombstone, Rungs::All))
            .into()
    }
}

pub(super) mod nsfw_admin {
    use super::*;

    pub(in crate::rules) fn blurs() -> Vec<RuleClause> {
        family(RuleId::NsfwAdmin)
            .when([NSFW_ADMIN, HAS_MEDIA])
            .clause(sensitive_media_blur(InterstitialReason::Sensitive(true)))
            .into()
    }

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        family(RuleId::NsfwAdmin)
            .when([TWEET_NSFW_ADMIN, HAS_MEDIA])
            .clause(age_prompt_blur(InterstitialReason::Sensitive(true)))
            .clause(modern_blur(InterstitialReason::Sensitive(true)))
            .into()
    }
}

pub(super) mod nsfw_user {
    use super::*;

    pub(in crate::rules) fn blurs() -> Vec<RuleClause> {
        family(RuleId::NsfwUser)
            .when([NSFW_USER, HAS_MEDIA])
            .clause(sensitive_media_blur(InterstitialReason::SensitiveUser(
                true,
            )))
            .into()
    }

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        family(RuleId::NsfwUser)
            .when([TWEET_NSFW_USER, HAS_MEDIA])
            .clause(age_prompt_blur(InterstitialReason::SensitiveUser(true)))
            .clause(modern_blur(InterstitialReason::SensitiveUser(true)))
            .into()
    }
}

pub(super) mod nsfw_account {
    use super::*;

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        family(RuleId::NsfwAccount)
            .when([TWEET_NSFW_FLAGGED, HAS_MEDIA])
            .clause(everyone(
                [SENSITIVE_MEDIA_DISABLED, LEGACY_INTERSTITIAL_CLIENT],
                LEGACY_NSFW_INTERSTITIAL,
            ))
            .clauses(age_gate_ladder(CountryList::AgeVerification, Rungs::All))
            .into()
    }
}

pub(super) mod gore_and_violence_high_precision {
    use super::*;

    pub(in crate::rules) fn blurs() -> Vec<RuleClause> {
        family(RuleId::GoreAndViolenceHighPrecision)
            .when([GORE_AND_VIOLENCE_HIGH_PRECISION])
            .clause(sensitive_media_blur(InterstitialReason::Violence(true)))
            .into()
    }

    pub(in crate::rules) fn oon_drop() -> Vec<RuleClause> {
        rule(
            RuleId::GoreAndViolenceHighPrecision,
            except_author(
                [GORE_AND_VIOLENCE_HIGH_PRECISION],
                drop_post(FilteredReason::ContainNsfwMedia),
            ),
        )
    }

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        let gore = || {
            family(RuleId::GoreAndViolenceHighPrecision).when([GORE_AND_VIOLENCE_HIGH_PRECISION])
        };
        [
            gore()
                .clause(modern_blur(InterstitialReason::Violence(true)))
                .into(),
            rule(
                RuleId::GoreAndViolenceIgnoringSettings,
                except_author(
                    [
                        GORE_AND_VIOLENCE_HIGH_PRECISION,
                        GORE_BLUR_IGNORES_SETTINGS_CLIENT,
                    ],
                    blur(InterstitialReason::Violence(true)),
                ),
            ),
            gore().clause(legacy_client_interstitial()).into(),
        ]
        .concat()
    }

    pub(in crate::rules) fn all_users_age_verification() -> Vec<RuleClause> {
        family(RuleId::GoreAndViolenceHighPrecision)
            .when([GORE_AND_VIOLENCE_HIGH_PRECISION])
            .clause(age_prompt_blur(InterstitialReason::Violence(true)))
            .clauses(age_gate_ladder(CountryList::Tombstone, Rungs::UpdateApp))
            .into()
    }
}

pub(super) mod nsfw_reported_heuristics {
    use super::*;

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        family(RuleId::NsfwReportedHeuristics)
            .when([NSFW_REPORTED_HEURISTICS])
            .clauses(age_gated_blurs(InterstitialReason::Sensitive(true)))
            .clauses(age_gate_ladder(CountryList::AgeVerification, Rungs::All))
            .into()
    }
}

pub(super) mod gore_and_violence_reported_heuristics {
    use super::*;

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        family(RuleId::GoreAndViolenceReportedHeuristics)
            .when([GORE_AND_VIOLENCE_REPORTED_HEURISTICS])
            .clause(modern_blur(InterstitialReason::Sensitive(true)))
            .clause(legacy_client_interstitial())
            .into()
    }
}

pub(super) mod nsfw_card_image {
    use super::*;

    pub(in crate::rules) fn blurs() -> Vec<RuleClause> {
        family(RuleId::NsfwCardImage)
            .when([NSFW_CARD_IMAGE])
            .clause(sensitive_media_blur(InterstitialReason::Sensitive(true)))
            .into()
    }

    pub(in crate::rules) fn oon_drop() -> Vec<RuleClause> {
        rule(
            RuleId::NsfwCardImage,
            except_author(
                [NSFW_CARD_IMAGE],
                drop_post(FilteredReason::ContainNsfwMedia),
            ),
        )
    }

    pub(in crate::rules) fn all_users() -> Vec<RuleClause> {
        family(RuleId::NsfwCardImage)
            .when([NSFW_CARD_IMAGE])
            .clauses(age_gated_blurs(InterstitialReason::Sensitive(true)))
            .clauses(age_gate_ladder(CountryList::AgeVerification, Rungs::All))
            .into()
    }
}

pub(super) fn nsfw_media_interstitials() -> Vec<RuleClause> {
    [
        nsfw_high_precision::blurs(),
        gore_and_violence_high_precision::blurs(),
        nsfw_card_image::blurs(),
    ]
    .concat()
}

pub(super) fn nsfw_author_interstitials() -> Vec<RuleClause> {
    [nsfw_admin::blurs(), nsfw_user::blurs()].concat()
}

pub(super) fn home_hydration_nsfw_rules() -> Vec<RuleClause> {
    [
        nsfw_high_precision::all_users(),
        nsfw_admin::all_users(),
        nsfw_user::all_users(),
        nsfw_account::all_users(),
        gore_and_violence_high_precision::all_users(),
        nsfw_reported_heuristics::all_users(),
        gore_and_violence_reported_heuristics::all_users(),
        nsfw_card_image::all_users(),
        gore_and_violence_high_precision::all_users_age_verification(),
    ]
    .concat()
}

pub(super) fn oon_tweet_flag_drops() -> Vec<RuleClause> {
    let nsfw_media = || drop_post(FilteredReason::ContainNsfwMedia);
    [
        rule(
            RuleId::NsfwUserTweetFlag,
            everyone([tweet(TweetPredicate::NsfwUserFlag)], nsfw_media()),
        ),
        rule(
            RuleId::NsfwAdminTweetFlag,
            everyone([tweet(TweetPredicate::NsfwAdminFlag)], nsfw_media()),
        ),
    ]
    .concat()
}

pub(super) fn oon_nsfw_media_label_drops() -> Vec<RuleClause> {
    [
        label_drop(
            RuleId::NsfwHighRecall,
            SafetyLabelType::NSFW_HIGH_RECALL,
            FilteredReason::ContainNsfwMedia,
        ),
        nsfw_high_precision::oon_drop(),
        nsfw_card_image::oon_drop(),
    ]
    .concat()
}

pub(super) fn oon_low_quality_tweet_label_drops() -> Vec<RuleClause> {
    use SafetyLabelType as L;
    let undesirable = FilteredReason::PossiblyUndesirable;
    [
        label_drop(RuleId::DoNotAmplify, L::DO_NOT_AMPLIFY, undesirable.clone()),
        label_drop(RuleId::MaliciousUrl, L::MALICIOUS_URL, undesirable.clone()),
        label_drop(RuleId::SpamHighRecall, L::SPAM_HIGH_RECALL, undesirable),
        rule(
            RuleId::BrazilElectionLegal,
            except_author(
                [
                    label(L::BRAZIL_ELECTION_LEGAL),
                    viewer(ViewerPredicate::RequestCountryIs("br")),
                ],
                drop_post(FilteredReason::PossiblyUndesirable),
            ),
        ),
    ]
    .concat()
}

pub(super) fn oon_text_label_drops() -> Vec<RuleClause> {
    label_drop(
        RuleId::FosnrAbuseInsults,
        SafetyLabelType::FOSNR_ABUSE_INSULTS,
        FilteredReason::PossiblyUndesirable,
    )
}

pub(super) fn fosnr_level_1_non_follower_drop() -> Vec<RuleClause> {
    rule(
        RuleId::FosnrAbuseInsultsNonFollower,
        except_author(
            [
                label(FOSNR_LEVEL_1),
                not(relationship(RelationshipPredicate::ViewerFollowsAuthor)),
                viewer(ViewerPredicate::ClientHasFosnrRules),
            ],
            drop_post(FilteredReason::PossiblyUndesirable),
        ),
    )
}

pub(super) fn fosnr_level_1_follower_soft_intervention() -> Vec<RuleClause> {
    rule(
        RuleId::FosnrAbuseInsultsFollower,
        except_author(
            [
                label(FOSNR_LEVEL_1),
                relationship(RelationshipPredicate::ViewerFollowsAuthor),
                viewer(ViewerPredicate::ClientHasFosnrRules),
            ],
            soft_intervention(FOSNR_LEVEL_1, AppealablePolicy::ABUSE, 1),
        ),
    )
}

const AUTHOR_LEVEL_3_LIMITED_ACTIONS: [LimitedActionType; 13] = {
    use LimitedActionType as A;
    [
        A::Like,
        A::Reply,
        A::Retweet,
        A::QuoteTweet,
        A::ShareTweetVia,
        A::AddToBookmarks,
        A::PinToProfile,
        A::CopyLink,
        A::SendViaDm,
        A::EditTweet,
        A::Highlight,
        A::Embed,
        A::ListsAddRemove,
    ]
};

const fn author_level_3(FosnrLevel3 { label, policy, .. }: FosnrLevel3) -> FosnrViolation {
    FosnrViolation {
        label,
        policy,
        level: 3,
        limited_actions: &AUTHOR_LEVEL_3_LIMITED_ACTIONS,
    }
}

const FOSNR_AUTHOR_VIOLATIONS: [FosnrViolation; 5] = {
    let [hateful_conduct, violent_speech, abuse, civic_integrity] = FOSNR_LEVEL_3;
    [
        author_level_3(abuse),
        author_level_3(civic_integrity),
        author_level_3(hateful_conduct),
        author_level_3(violent_speech),
        FosnrViolation {
            label: FOSNR_LEVEL_1,
            policy: AppealablePolicy::ABUSE,
            level: 1,
            limited_actions: &[LimitedActionType::EditTweet],
        },
    ]
};

pub(super) fn fosnr_author_appealable() -> Vec<RuleClause> {
    rule(
        RuleId::FosnrAuthor,
        only_author(
            [
                viewer(ViewerPredicate::ClientHasFosnrRules),
                Condition::AnyOf(&FOSNR),
            ],
            appealable(&FOSNR_AUTHOR_VIOLATIONS),
        ),
    )
}

pub(super) fn fosnr_fallback_drop() -> Vec<RuleClause> {
    rule(
        RuleId::FosnrFallback,
        everyone(
            [
                viewer(ViewerPredicate::ClientNeedsFosnrFallbackDrops),
                Condition::AnyOf(&FOSNR),
            ],
            drop_post(FilteredReason::PossiblyUndesirable),
        ),
    )
}

pub(super) fn creator_tweet_nsfw_drop() -> Vec<RuleClause> {
    rule(
        RuleId::CreatorTweetNsfw,
        except_author(
            [HAS_MEDIA, NSFW_HIGH_PRECISION, HAS_EXCLUSIVE_CONTENT],
            drop_post(FilteredReason::ContainNsfwMedia),
        ),
    )
}

pub(super) fn exclusive_tweet_drop() -> Vec<RuleClause> {
    let exclusive = || drop_post(FilteredReason::ExclusiveTweet);
    family(RuleId::ExclusiveTweet)
        .when([HAS_EXCLUSIVE_CONTENT])
        .clause(everyone([LOGGED_OUT], exclusive()))
        .clause(everyone(
            [
                NOT_CONVERSATION_AUTHOR,
                NOT_SUPER_FOLLOWER,
                tweet(TweetPredicate::IsRetweet),
            ],
            exclusive(),
        ))
        .clause(except_author(
            [NOT_CONVERSATION_AUTHOR, NOT_SUPER_FOLLOWER],
            exclusive(),
        ))
        .into()
}

pub(super) fn trusted_friends_tweet_drop() -> Vec<RuleClause> {
    rule(
        RuleId::TrustedFriendsTweet,
        except_author(
            [
                tweet(TweetPredicate::IsTrustedFriendsTweet),
                not(relationship(
                    RelationshipPredicate::ViewerIsTrustedFriendsListMemberOrOwner,
                )),
            ],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

pub(super) fn author_blocks_viewer_exclusive_content_drop() -> Vec<RuleClause> {
    rule(
        RuleId::AuthorBlocksViewerExclusiveContent,
        except_author(
            [
                relationship(RelationshipPredicate::ViewerIsBlockedByAuthor),
                HAS_EXCLUSIVE_CONTENT,
            ],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

fn limit_replies(
    id: RuleId,
    arm: ConversationControlArm,
    exemption: impl IntoIterator<Item = Condition>,
) -> Vec<RuleClause> {
    family(id)
        .when([
            tweet(TweetPredicate::HasConversationControl(arm)),
            NOT_LOGGED_OUT,
            NOT_RETWEET,
            NOT_CONVERSATION_ROOT_AUTHOR,
            NOT_INVITED_TO_CONVERSATION,
        ])
        .clause(everyone(
            exemption,
            limit(LimitedEngagementReason::ConversationControl),
        ))
        .into()
}

fn limit_replies_conversation_rules() -> Vec<RuleClause> {
    use ConversationControlArm::{ByInvitation, Co, Community, MyNetwork, Subscribers, Verified};
    use RelationshipPredicate::{
        ViewerIsFollowedByConversationRootAuthor, ViewerIsInAllowedCountry,
        ViewerIsInConversationRootAuthorNetwork, ViewerSuperFollowsConversationRootAuthor,
    };
    let unless = |leaf| [not(relationship(leaf))];
    [
        limit_replies(RuleId::LimitRepliesByInvitation, ByInvitation, []),
        limit_replies(
            RuleId::LimitRepliesCommunity,
            Community,
            unless(ViewerIsFollowedByConversationRootAuthor),
        ),
        limit_replies(
            RuleId::LimitRepliesSubscribers,
            Subscribers,
            unless(ViewerSuperFollowsConversationRootAuthor),
        ),
        limit_replies(
            RuleId::LimitRepliesVerified,
            Verified,
            [not(viewer(ViewerPredicate::HasVerifiedBadge))],
        ),
        limit_replies(
            RuleId::LimitRepliesMyNetwork,
            MyNetwork,
            unless(ViewerIsInConversationRootAuthorNetwork),
        ),
        limit_replies(RuleId::LimitRepliesCo, Co, unless(ViewerIsInAllowedCountry)),
    ]
    .concat()
}

fn blocked_viewer_limited_actions() -> Vec<RuleClause> {
    family(RuleId::BlockedViewer)
        .clause(except_author(
            [relationship(RelationshipPredicate::ViewerIsBlockedByAuthor)],
            limit(LimitedEngagementReason::BlockedViewer),
        ))
        .clause(everyone(
            [relationship(
                RelationshipPredicate::ViewerIsBlockedByConversationRootAuthor,
            )],
            limit(LimitedEngagementReason::RootAuthorBlockedViewer),
        ))
        .into()
}

fn read_only_viewer_limited_actions() -> Vec<RuleClause> {
    rule(
        RuleId::ReadOnlyViewer,
        everyone(
            [viewer(ViewerPredicate::ReadOnly)],
            limit(LimitedEngagementReason::ReadonlyViewer),
        ),
    )
}

fn community_tweet_limited_actions() -> Vec<RuleClause> {
    rule(
        RuleId::CommunityTweetViewerRemoved,
        everyone(
            [
                tweet(TweetPredicate::IsCommunityTweet),
                relationship(RelationshipPredicate::ViewerIsRemovedFromCommunity),
                viewer(ViewerPredicate::ClientHasCommunityViewerRemovedLimits),
            ],
            limit(LimitedEngagementReason::CommunityTweetViewerRemoved),
        ),
    )
}

fn local_tweet_limited_actions() -> Vec<RuleClause> {
    rule(
        RuleId::LocalTweet,
        everyone(
            [
                tweet(TweetPredicate::HasNarrowcastPlace),
                NOT_LOGGED_OUT,
                NOT_CONVERSATION_ROOT_AUTHOR,
                NOT_INVITED_TO_CONVERSATION,
                relationship(RelationshipPredicate::ViewerIsOutsideNarrowcastPlace),
            ],
            limit(LimitedEngagementReason::LocalTweet),
        ),
    )
}

pub(super) fn limited_engagement_rules() -> Vec<RuleClause> {
    [
        blocked_viewer_limited_actions(),
        stale_tweet_limited_actions(),
        limit_replies_conversation_rules(),
        community_tweet_limited_actions(),
        local_tweet_limited_actions(),
        read_only_viewer_limited_actions(),
    ]
    .concat()
}

fn sensitive_viewer_drop(
    id: RuleId,
    reason: NsfwViewerDropReason,
    viewer_class: impl IntoIterator<Item = Condition>,
) -> Vec<RuleClause> {
    family(id)
        .when(viewer_class)
        .clause(except_author(
            [HAS_MEDIA, NSFW_MEDIA_LABEL],
            nsfw_viewer_drop(reason),
        ))
        .clause(except_author(
            [HAS_MEDIA, NOT_RETWEET, NSFW_FLAGGED],
            nsfw_viewer_drop(reason),
        ))
        .clause(except_author(
            [NSFW_TEXT_OR_CARD_LABEL],
            nsfw_viewer_drop(reason),
        ))
        .into()
}

pub(super) fn sensitive_viewer_drops() -> Vec<RuleClause> {
    [
        sensitive_viewer_drop(
            RuleId::SensitiveViewerLoggedOut,
            NsfwViewerDropReason::LoggedOut,
            [LOGGED_OUT],
        ),
        sensitive_viewer_drop(
            RuleId::SensitiveViewerUnderage,
            NsfwViewerDropReason::IsUnderage,
            [UNDERAGE],
        ),
        sensitive_viewer_drop(
            RuleId::SensitiveViewerNoStatedAge,
            NsfwViewerDropReason::HasNoStatedAge,
            [NO_STATED_AGE, IN_NSFW_GATING_COUNTRY],
        ),
    ]
    .concat()
}

const NSFW_SENSITIVE_TWEET: Condition = Condition::AnyOf(&[
    has_tweet_label(SafetyLabelType::NSFW_HIGH_PRECISION),
    has_tweet_label(SafetyLabelType::NSFW_HIGH_RECALL),
    has_tweet_label(SafetyLabelType::NSFW_TEXT),
    has_tweet_label(SafetyLabelType::NSFW_TEXT_HIGH_PRECISION),
    has_tweet_label(SafetyLabelType::NSFW_VIDEO),
    Predicate::Tweet(TweetPredicate::NsfwAdminFlag),
    Predicate::Tweet(TweetPredicate::NsfwUserFlag),
]);

const NSFW_SENSITIVE_AUTHOR: Condition = Condition::AnyOf(&[
    has_user_label(AuthorLabel::NsfwAvatarImage),
    has_user_label(AuthorLabel::NsfwBannerImage),
    has_user_label(AuthorLabel::NsfwHighPrecision),
    has_user_label(AuthorLabel::NsfwHighRecall),
    has_user_label(AuthorLabel::NsfwNearPerfect),
    Predicate::Author(AuthorPredicate::IsNsfwAdmin),
    Predicate::Author(AuthorPredicate::IsNsfwUser),
]);

pub(super) fn sensitive_media_opt_out_drops() -> Vec<RuleClause> {
    let nsfw_media = || drop_post(FilteredReason::ContainNsfwMedia);
    [
        rule(
            RuleId::NsfwSensitiveViewerTweet,
            except_author(
                [SENSITIVE_MEDIA_DISABLED, NSFW_SENSITIVE_TWEET],
                nsfw_media(),
            ),
        ),
        rule(
            RuleId::NsfwSensitiveViewerUser,
            except_author(
                [SENSITIVE_MEDIA_DISABLED, NSFW_SENSITIVE_AUTHOR],
                nsfw_media(),
            ),
        ),
    ]
    .concat()
}

pub(super) fn nullcast_drop() -> Vec<RuleClause> {
    rule(
        RuleId::NullcastedTweet,
        everyone(
            [
                tweet(TweetPredicate::IsNullcast),
                NOT_RETWEET,
                not(tweet(TweetPredicate::IsCommunityTweet)),
            ],
            drop_post(FilteredReason::TweetIsNullcast),
        ),
    )
}

pub(super) fn stale_tweet_drop() -> Vec<RuleClause> {
    rule(
        RuleId::StaleTweet,
        everyone(
            [tweet(TweetPredicate::IsSupersededEdit), NOT_RETWEET],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

fn stale_tweet_limited_actions() -> Vec<RuleClause> {
    rule(
        RuleId::StaleTweet,
        everyone(
            [
                tweet(TweetPredicate::IsSupersededEdit),
                viewer(ViewerPredicate::ClientHasStaleTweetLimits),
            ],
            limit(LimitedEngagementReason::StaleTweet),
        ),
    )
}

pub(super) fn takedown_drops() -> Vec<RuleClause> {
    let unspecified = || drop_post(FilteredReason::UnspecifiedReason);
    [
        rule(
            RuleId::LegalTakedown,
            except_author(
                [tweet(TweetPredicate::LegalTakedownInRequestCountry)],
                unspecified(),
            ),
        ),
        rule(
            RuleId::LocalLawsTakedown,
            except_author(
                [tweet(TweetPredicate::LocalLawsTakedownInRequestCountry)],
                unspecified(),
            ),
        ),
    ]
    .concat()
}

pub(super) fn article_tweet_content_drop() -> Vec<RuleClause> {
    rule(
        RuleId::ArticleTweetContent,
        everyone(
            [
                tweet(TweetPredicate::HasArticle),
                not(tweet(TweetPredicate::ArticleIsPublished)),
            ],
            drop_post(FilteredReason::UnspecifiedReason),
        ),
    )
}

pub(super) fn filter_all() -> Vec<RuleClause> {
    rule(
        RuleId::FilterAll,
        everyone([], drop_post(FilteredReason::UnspecifiedReason)),
    )
}

pub(super) fn recs_media_drops() -> Vec<RuleClause> {
    let unspecified = || drop_post(FilteredReason::UnspecifiedReason);
    [
        rule(
            RuleId::DmcaMedia,
            everyone([tweet(TweetPredicate::HasDmcaMedia)], unspecified()),
        ),
        rule(
            RuleId::GeoRestrictedMedia,
            everyone(
                [tweet(TweetPredicate::MediaGeoRestrictedInRequestCountry)],
                unspecified(),
            ),
        ),
    ]
    .concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_moderated_community_drops_exempt_the_author_and_the_community_moderators() {
        use crate::models::{CommunityModeration, Decided, TweetFeatures, Verdict};
        use crate::rules::fixtures::{author_viewer, candidate, viewer};
        use crate::rules::{RuleEngine, SafetyLevel};
        use std::num::NonZeroU64;
        let engine = RuleEngine::for_tests();
        for (moderation, rule) in [
            (
                CommunityModeration {
                    is_hidden: true,
                    is_author_removed: false,
                },
                "hidden_community_tweet/drop/unspecified",
            ),
            (
                CommunityModeration {
                    is_hidden: false,
                    is_author_removed: true,
                },
                "author_removed_community_tweet/drop/unspecified",
            ),
        ] {
            for (viewer, is_moderator, drops) in [
                (viewer(1), Some(false), true),
                (viewer(1), Some(true), false),
                (author_viewer(), Some(false), false),
            ] {
                let mut post = candidate()
                    .with_tweet_features(TweetFeatures {
                        community_id: NonZeroU64::new(500),
                        ..Default::default()
                    })
                    .build();
                post.community_moderation = moderation;
                post.viewer_is_community_moderator = is_moderator;
                let verdict = engine
                    .evaluate(SafetyLevel::TimelineHomeHydration, &viewer, &post)
                    .into_verdict();
                assert_eq!(
                    matches!(verdict, Verdict::Withheld(Decided { by, .. }) if by == rule),
                    drops,
                    "{rule}, moderator {is_moderator:?}: {verdict:?}"
                );
            }
        }
    }

    #[test]
    fn home_hydration_label_blurs_add_the_modern_blur_gate_to_the_timeline_home_conditions() {
        let home_hydration = home_hydration_nsfw_rules();
        for home_blur in nsfw_media_interstitials() {
            let clause = home_hydration
                .iter()
                .find(|clause| clause.id == home_blur.id && clause.action == home_blur.action)
                .unwrap_or_else(|| {
                    panic!("{} is not wired at TimelineHomeHydration", home_blur.name())
                });
            let mut gated = home_blur.when.clone();
            gated.push(MODERN_BLUR_CLIENT);
            assert_eq!(clause.when, gated, "{}", home_blur.name());
        }
    }
}
