use crate::limited_actions_copy::{Prompt, PromptKind, LEARN_MORE_PLACEHOLDER};
use crate::models::{
    Decided, DropReason, FosnrReason, LimitedEngagement, LimitedEngagementReason,
    MediaInterstitial, MediaRestriction, Notice, NsfwViewerDropReason, TombstoneReason, Verdict,
    Withholding,
};
use crate::params::{LimitedActionType, LimitedActionsPolicies};
use crate::rules::SafetyLevel;
use xai_visibility_filtering::models::FilteredReason;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::action::{
    self, Action, AgeVerificationOption, AnyInterstitial, Appealable, AppealablePolicy,
    AppealableReason, Avoid, BasicLimitedActionPrompt, BlurredImageInterstitial,
    BrandSafetyRiskLevel, ComposedMediaVisibilityActions, CtaLimitedActionPrompt, Interstitial,
    InterstitialAction, InterstitialReason, LimitedAction, LimitedActionCtaType,
    LimitedActionPrompt, LimitedActionsPolicy, LimitedEngagements, LocalizedMessage,
    LocalizedMessageLimitedActionPrompt, MediaInterstitial as ThriftMediaInterstitial, MessageLink,
    SoftIntervention as ThriftSoftIntervention, SoftInterventionDisplayType,
    SoftInterventionReason, Tombstone, TweetInterstitial,
};
use xai_x_thrift::safety_result::{
    FilteredReason as ThriftFilteredReason, SafetyResult as ThriftSafetyResult, SafetyResultReason,
};
use xai_x_thrift::tweet_service::{
    TweetFieldsResultFiltered, TweetFieldsResultFound, TweetFieldsResultState,
};

pub(crate) fn thrift_action(
    verdict: &Verdict,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> Action {
    match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(reason),
            ..
        }) => Action::Drop(action::Drop::new(drop_reason(reason, level), None)),
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(reason),
            ..
        }) => Action::Tombstone(Tombstone::new(Some(tombstone_reason(*reason)), None)),
        Verdict::Shown {
            notice:
                Some(Decided {
                    value: Notice::SoftIntervention(reason),
                    ..
                }),
            ..
        } => Action::TweetInterstitial(TweetInterstitial {
            soft_intervention: Some(soft_intervention(reason)),
            avoid: Some(normal_avoid()),
            ..TweetInterstitial::default()
        }),
        Verdict::Shown {
            notice:
                Some(Decided {
                    value:
                        Notice::Appealable {
                            reason,
                            limited_actions,
                        },
                    ..
                }),
            ..
        } => Action::TweetInterstitial(TweetInterstitial {
            limited_engagements: Some(fosnr_limited_engagements(reason, limited_actions)),
            avoid: Some(normal_avoid()),
            appealable: Some(Appealable::new(appealable_reason(reason), None)),
            ..TweetInterstitial::default()
        }),
        Verdict::Shown {
            notice: None,
            media: None,
            engagement: None,
        } => Action::Allow(action::Allow::new()),
        Verdict::Shown {
            notice: None,
            media: None,
            engagement: Some(Decided { value, .. }),
        } => Action::LimitedEngagements(limited_engagements(value, policies)),
        Verdict::Shown {
            notice: None,
            media: Some(Decided { value, .. }),
            engagement: None,
        } => match value {
            MediaRestriction::MediaInterstitial(blur) => {
                Action::ComposedMediaVisibilityResults(blurred_media(blur))
            }
            MediaRestriction::NsfwInterstitial => Action::Interstitial(nsfw_interstitial()),
        },
        Verdict::Shown {
            notice: None,
            media: Some(media),
            engagement: Some(limit),
        } => {
            let (interstitial, all_media_visibility_results) = match &media.value {
                MediaRestriction::MediaInterstitial(blur) => (None, Some(blurred_media(blur))),
                MediaRestriction::NsfwInterstitial => (
                    Some(AnyInterstitial::Interstitial(nsfw_interstitial())),
                    None,
                ),
            };
            Action::TweetInterstitial(TweetInterstitial {
                interstitial,
                limited_engagements: Some(limited_engagements(&limit.value, policies)),
                all_media_visibility_results,
                ..TweetInterstitial::default()
            })
        }
    }
}

pub(crate) fn thrift_result_state(
    verdict: &Verdict,
    level: SafetyLevel,
    policies: &LimitedActionsPolicies,
) -> TweetFieldsResultState {
    let reason = match verdict {
        Verdict::Shown {
            notice:
                Some(Decided {
                    value: Notice::Appealable { reason, .. },
                    ..
                }),
            ..
        } => Some(fosnr_safety_result_reason(reason)),
        Verdict::Withheld(_) | Verdict::Shown { .. } => None,
    };
    let safety_result = || {
        ThriftFilteredReason::SafetyResult(ThriftSafetyResult::new(
            reason,
            thrift_action(verdict, level, policies),
        ))
    };
    let found = |reason| TweetFieldsResultState::Found(TweetFieldsResultFound::new(reason));
    let filtered =
        |reason| TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason));
    match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(reason)),
            ..
        }) => filtered(legacy_drop_reason(reason).unwrap_or_else(safety_result)),
        Verdict::Withheld(_) => filtered(safety_result()),
        Verdict::Shown {
            notice: None,
            media: None,
            engagement: None,
        } => found(None),
        Verdict::Shown {
            notice: None,
            media:
                Some(Decided {
                    value: MediaRestriction::NsfwInterstitial,
                    ..
                }),
            engagement: None,
        } => found(Some(ThriftFilteredReason::ContainNsfwMedia(true))),
        Verdict::Shown { .. } => found(Some(safety_result())),
    }
}

fn legacy_drop_reason(reason: &FilteredReason) -> Option<ThriftFilteredReason> {
    use ThriftFilteredReason as T;
    Some(match reason {
        FilteredReason::TweetIsBounced => T::TweetIsBounced(true),
        FilteredReason::AuthorBlockViewer => T::AuthorBlockViewer(true),
        FilteredReason::AuthorIsProtected => T::AuthorIsProtected(true),
        FilteredReason::AuthorIsSuspended => T::AuthorIsSuspended(true),
        FilteredReason::ContainNsfwMedia
        | FilteredReason::PossiblyUndesirable
        | FilteredReason::UnspecifiedReason
        | FilteredReason::AuthorAccountIsInactive
        | FilteredReason::AuthorIsUnsafe
        | FilteredReason::ReportedTweet
        | FilteredReason::TweetMatchesViewerMutedKeyword(_)
        | FilteredReason::SafetyResult(_)
        | FilteredReason::AuthorIsDeactivated
        | FilteredReason::ViewerMutesAuthor
        | FilteredReason::TweetIsNullcast
        | FilteredReason::ExclusiveTweet
        | FilteredReason::ViewerBlocksAuthor => return None,
    })
}

fn normal_avoid() -> Avoid {
    Avoid::new(None, Some(BrandSafetyRiskLevel::NORMAL), None)
}

fn appealable_reason(reason: &FosnrReason) -> AppealableReason {
    AppealableReason::new(
        reason.level,
        Box::new(reason.policy),
        reason.proactive,
        reason.appeal_submitted,
    )
}

fn fosnr_safety_result_reason(reason: &FosnrReason) -> SafetyResultReason {
    match reason.policy {
        AppealablePolicy::HATEFUL_CONDUCT => SafetyResultReason::FOSNR_HATEFUL_CONDUCT,
        AppealablePolicy::ABUSE => SafetyResultReason::FOSNR_ABUSE,
        AppealablePolicy::VIOLENT_SPEECH => SafetyResultReason::FOSNR_VIOLENT_SPEECH,
        AppealablePolicy::CIVIC_INTEGRITY => SafetyResultReason::FOSNR_CIVIC_INTEGRITY,
        AppealablePolicy::SYNTHETIC_AND_MANIPULATED_MEDIA => {
            SafetyResultReason::FOSNR_SYNTHETIC_AND_MANIPULATED_MEDIA
        }
        _ => SafetyResultReason::FOSNR_UNSPECIFIED,
    }
}

fn fosnr_limited_engagements(
    reason: &FosnrReason,
    limited_actions: &[LimitedActionType],
) -> LimitedEngagements {
    LimitedEngagements::new(
        action::LimitedEngagementReason::FosnrReason(appealable_reason(reason)),
        LimitedActionsPolicy::new(
            limited_actions
                .iter()
                .map(|&action_type| LimitedAction::new(limited_action_type(action_type), None))
                .collect(),
        ),
        (reason.level != 1).then(|| "freedom_of_speech_not_reach".to_string()),
    )
}

fn soft_intervention(reason: &FosnrReason) -> ThriftSoftIntervention {
    ThriftSoftIntervention {
        soft_intervention_reason: Some(SoftInterventionReason::Fosnr(Box::new(appealable_reason(
            reason,
        )))),
        engagement_nudge: Some(false),
        suppress_autoplay: Some(true),
        warning: None,
        details_url: None,
        display_type: Some(SoftInterventionDisplayType::FOSNR),
    }
}

fn nsfw_interstitial() -> Interstitial {
    Interstitial::new(InterstitialReason::ContainsNsfwMedia(true), None)
}

fn limited_engagements(
    limit: &LimitedEngagement,
    policies: &LimitedActionsPolicies,
) -> LimitedEngagements {
    let mut actions: Vec<LimitedAction> = Vec::new();
    for action in limit.reasons().flat_map(|reason| policies.actions(reason)) {
        let action_type = limited_action_type(action.action_type);
        if !actions
            .iter()
            .any(|kept| kept.limited_action_type == action_type)
        {
            let prompt = action.prompt.as_ref().map(limited_action_prompt);
            actions.push(LimitedAction::new(action_type, prompt));
        }
    }
    LimitedEngagements::new(
        limited_engagement_reason(limit.reason()),
        (!actions.is_empty()).then(|| LimitedActionsPolicy::new(actions)),
        limit.reason().limited_actions_string().to_string(),
    )
}

fn limited_action_prompt(prompt: &Prompt) -> LimitedActionPrompt {
    let headline = prompt.headline.clone();
    let subtext = prompt.subtext.clone();
    match &prompt.kind {
        PromptKind::Basic => LimitedActionPrompt::BasicLimitedActionPrompt(
            BasicLimitedActionPrompt::new(headline, subtext),
        ),
        PromptKind::SeeConversation => LimitedActionPrompt::CtaLimitedActionPrompt(
            CtaLimitedActionPrompt::new(headline, subtext, LimitedActionCtaType::SEE_CONVERSATION),
        ),
        PromptKind::LearnMore {
            language,
            link_text,
            url,
        } => LimitedActionPrompt::LocalizedMessageLimitedActionPrompt(
            LocalizedMessageLimitedActionPrompt::new(
                LocalizedMessage::new(headline, language.clone(), Vec::new()),
                LocalizedMessage::new(
                    subtext,
                    language.clone(),
                    vec![MessageLink::new(
                        LEARN_MORE_PLACEHOLDER.to_string(),
                        link_text.to_string(),
                        url.clone(),
                    )],
                ),
            ),
        ),
    }
}

fn limited_action_type(action_type: LimitedActionType) -> action::LimitedActionType {
    use action::LimitedActionType as T;
    use LimitedActionType as L;
    match action_type {
        L::Reply => T::REPLY,
        L::Retweet => T::RETWEET,
        L::QuoteTweet => T::QUOTE_TWEET,
        L::Like => T::LIKE,
        L::React => T::REACT,
        L::SendViaDm => T::SEND_VIA_DM,
        L::AddToBookmarks => T::ADD_TO_BOOKMARKS,
        L::AddToMoment => T::ADD_TO_MOMENT,
        L::PinToProfile => T::PIN_TO_PROFILE,
        L::ViewTweetActivity => T::VIEW_TWEET_ACTIVITY,
        L::ShareTweetVia => T::SHARE_TWEET_VIA,
        L::Follow => T::FOLLOW,
        L::ListsAddRemove => T::LISTS_ADD_REMOVE,
        L::MuteConversation => T::MUTE_CONVERSATION,
        L::Embed => T::EMBED,
        L::ViewHiddenReplies => T::VIEW_HIDDEN_REPLIES,
        L::HideCommunityTweet => T::HIDE_COMMUNITY_TWEET,
        L::CopyLink => T::COPY_LINK,
        L::VoteOnPoll => T::VOTE_ON_POLL,
        L::RemoveFromCommunity => T::REMOVE_FROM_COMMUNITY,
        L::ShowRetweetActionMenu => T::SHOW_RETWEET_ACTION_MENU,
        L::ReplyDownVote => T::REPLY_DOWN_VOTE,
        L::Autoplay => T::AUTOPLAY,
        L::EditTweet => T::EDIT_TWEET,
        L::Highlight => T::HIGHLIGHT,
        L::ViewPostEngagements => T::VIEW_POST_ENGAGEMENTS,
    }
}

const AGE_VERIFICATION_OPTIONS: [AgeVerificationOption; 2] = [
    AgeVerificationOption::SELFIE,
    AgeVerificationOption::PERSONA,
];

fn blurred_media(blur: &MediaInterstitial) -> ComposedMediaVisibilityActions {
    let verification_options = (blur.prompt == Some(InterstitialAction::AGE_VERIFICATION_PROMPT))
        .then(|| AGE_VERIFICATION_OPTIONS.to_vec());
    ComposedMediaVisibilityActions {
        media_interstitial: Some(Box::new(ThriftMediaInterstitial::BlurredImageInterstitial(
            BlurredImageInterstitial {
                reason: Some(blur.reason.clone()),
                opacity: Some(0.8.into()),
                interstitial_action: blur.prompt,
                available_verification_options: verification_options,
            },
        ))),
    }
}

fn drop_reason(reason: &DropReason, level: SafetyLevel) -> Option<action::DropReason> {
    let legacy = match reason {
        DropReason::Legacy(legacy) => legacy,
        DropReason::NsfwViewer(reason) => {
            return Some(match reason {
                NsfwViewerDropReason::IsUnderage => action::DropReason::NsfwViewerIsUnderage(true),
                NsfwViewerDropReason::HasNoStatedAge => {
                    action::DropReason::NsfwViewerHasNoStatedAge(true)
                }
                NsfwViewerDropReason::LoggedOut => action::DropReason::NsfwLoggedOut(true),
            });
        }
    };
    Some(match legacy {
        FilteredReason::AuthorIsProtected => action::DropReason::ProtectedAuthor(true),
        FilteredReason::AuthorIsSuspended => action::DropReason::SuspendedAuthor(true),
        FilteredReason::AuthorBlockViewer => action::DropReason::AuthorBlocksViewer(true),
        FilteredReason::ViewerBlocksAuthor => action::DropReason::ViewerBlocksAuthor(true),
        FilteredReason::ViewerMutesAuthor => action::DropReason::ViewerMutesAuthor(true),
        FilteredReason::ExclusiveTweet => action::DropReason::ExclusiveTweet(true),
        FilteredReason::UnspecifiedReason if level.spec().unspecified_drop_is_exact => {
            action::DropReason::Unspecified(true)
        }
        FilteredReason::UnspecifiedReason
        | FilteredReason::ContainNsfwMedia
        | FilteredReason::PossiblyUndesirable
        | FilteredReason::AuthorAccountIsInactive
        | FilteredReason::AuthorIsUnsafe
        | FilteredReason::ReportedTweet
        | FilteredReason::TweetMatchesViewerMutedKeyword(_)
        | FilteredReason::TweetIsBounced
        | FilteredReason::SafetyResult(_)
        | FilteredReason::AuthorIsDeactivated
        | FilteredReason::TweetIsNullcast => return None,
    })
}

fn tombstone_reason(reason: TombstoneReason) -> action::TombstoneReason {
    match reason {
        TombstoneReason::SensitiveViewerAgeVerification => {
            action::TombstoneReason::SENSITIVE_VIEWER_AGE_VERIFICATION
        }
        TombstoneReason::UpdateAppIos => action::TombstoneReason::UPDATE_APP_IOS,
        TombstoneReason::UpdateAppAndroid => action::TombstoneReason::UPDATE_APP_ANDROID,
        TombstoneReason::LocalRegulations => action::TombstoneReason::LOCAL_REGULATIONS,
    }
}

fn limited_engagement_reason(reason: LimitedEngagementReason) -> action::LimitedEngagementReason {
    match reason {
        LimitedEngagementReason::ConversationControl => {
            action::LimitedEngagementReason::ConversationControl(action::ConversationControl::new())
        }
        LimitedEngagementReason::ReadonlyViewer => {
            action::LimitedEngagementReason::ReadonlyViewer(action::ReadonlyViewer::new())
        }
        LimitedEngagementReason::BlockedViewer => {
            action::LimitedEngagementReason::BlockedViewer(action::BlockedViewer::new())
        }
        LimitedEngagementReason::RootAuthorBlockedViewer => {
            action::LimitedEngagementReason::RootAuthorBlockedViewer(
                action::RootAuthorBlockedViewer::new(),
            )
        }
        LimitedEngagementReason::StaleTweet => {
            action::LimitedEngagementReason::StaleTweet(action::StaleTweet::new())
        }
        LimitedEngagementReason::CommunityTweetHidden => {
            action::LimitedEngagementReason::CommunityTweetHidden(
                action::CommunityTweetHidden::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetMemberRemoved => {
            action::LimitedEngagementReason::CommunityTweetMemberRemoved(
                action::CommunityTweetMemberRemoved::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetCommunityNotFound => {
            action::LimitedEngagementReason::CommunityTweetCommunityNotFound(
                action::CommunityTweetCommunityNotFound::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetCommunityDeleted => {
            action::LimitedEngagementReason::CommunityTweetCommunityDeleted(
                action::CommunityTweetCommunityDeleted::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetCommunitySuspended => {
            action::LimitedEngagementReason::CommunityTweetCommunitySuspended(
                action::CommunityTweetCommunitySuspended::new(),
            )
        }
        LimitedEngagementReason::CommunityTweetViewerRemoved => {
            action::LimitedEngagementReason::CommunityTweetViewerRemoved(
                action::CommunityTweetViewerRemoved::new(),
            )
        }
        LimitedEngagementReason::LocalTweet => {
            action::LimitedEngagementReason::LocalTweet(action::LocalTweet::new())
        }
    }
}

pub(crate) fn proto_action(verdict: Verdict) -> (vf_pb::Action, Option<vf_pb::FilteredReason>) {
    let (kind, filtered_reason) = match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(reason),
            ..
        }) => (
            vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            Some(reason.legacy().clone().into()),
        ),
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(_),
            ..
        }) => (
            vf_pb::action::Kind::Drop(vf_pb::DropReason {}),
            Some(FilteredReason::UnspecifiedReason.into()),
        ),
        Verdict::Shown {
            notice: None | Some(_),
            media: Some(Decided { value, .. }),
            engagement: None | Some(_),
        } => (
            vf_pb::action::Kind::Interstitial(true),
            Some(value.legacy().clone().into()),
        ),
        Verdict::Shown {
            notice: None | Some(_),
            media: None,
            engagement: None | Some(_),
        } => (vf_pb::action::Kind::Allow(true), None),
    };
    (vf_pb::Action { kind: Some(kind) }, filtered_reason)
}

pub(crate) fn metric_label(verdict: &Verdict) -> &'static str {
    match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(_),
            ..
        }) => "drop",
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(_),
            ..
        }) => "tombstone",
        Verdict::Shown {
            notice: Some(notice),
            ..
        } => notice_label(&notice.value),
        Verdict::Shown {
            notice: None,
            media: None,
            engagement: None,
        } => "allow",
        Verdict::Shown {
            notice: None,
            media: Some(_),
            engagement: None,
        } => "interstitial",
        Verdict::Shown {
            notice: None,
            media: None,
            engagement: Some(_),
        } => "limited_engagement",
        Verdict::Shown {
            notice: None,
            media: Some(_),
            engagement: Some(_),
        } => "tweet_interstitial",
    }
}

const fn notice_label(notice: &Notice) -> &'static str {
    match notice {
        Notice::SoftIntervention(_) => "soft_intervention",
        Notice::Appealable { .. } => "appealable",
    }
}

pub(crate) fn decided_rows(
    verdict: &Verdict,
) -> impl Iterator<Item = (&'static str, &'static str)> {
    let (notice, first, second) = match verdict {
        Verdict::Withheld(decided) => (None, Some((decided.by, metric_label(verdict))), None),
        Verdict::Shown {
            notice,
            media,
            engagement,
        } => (
            notice
                .as_ref()
                .map(|notice| (notice.by, notice_label(&notice.value))),
            media.as_ref().map(|blur| (blur.by, "interstitial")),
            engagement
                .as_ref()
                .map(|limit| (limit.by, "limited_engagement")),
        ),
    };
    notice.into_iter().chain(first).chain(second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::fixtures::{appealed, limited_for, noticed};
    use crate::rules::metrics::Rpc;
    use vf_pb::action::Kind;
    use xai_visibility_filtering::graphql_results::resolve_blurred_image_interstitial;
    use xai_visibility_filtering::models::{KeywordMatch, SafetyResult};
    use xai_x_thrift::safety_result::SafetyResult as ThriftSafetyResult;
    use SafetyLevel::{FilterAll, TimelineHome};

    fn policies() -> LimitedActionsPolicies {
        LimitedActionsPolicies::for_tests(vec![(
            LimitedEngagementReason::ConversationControl,
            vec![LimitedActionType::Reply],
        )])
    }

    fn dropped(reason: FilteredReason) -> Verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(reason)),
            by: "rule",
        })
    }

    fn tombstoned(reason: TombstoneReason) -> Verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(reason),
            by: "rule",
        })
    }

    fn blur_with(
        reason: InterstitialReason,
        prompt: Option<InterstitialAction>,
    ) -> Decided<MediaRestriction> {
        Decided {
            value: MediaRestriction::MediaInterstitial(MediaInterstitial {
                legacy: FilteredReason::ContainNsfwMedia,
                reason,
                prompt,
            }),
            by: "blur_rule",
        }
    }

    fn blur(reason: InterstitialReason) -> Decided<MediaRestriction> {
        blur_with(reason, None)
    }

    fn legacy_interstitial() -> Decided<MediaRestriction> {
        Decided {
            value: MediaRestriction::NsfwInterstitial,
            by: "fallback_rule",
        }
    }

    fn thrift_media(reason: InterstitialReason) -> ComposedMediaVisibilityActions {
        ComposedMediaVisibilityActions {
            media_interstitial: Some(Box::new(ThriftMediaInterstitial::BlurredImageInterstitial(
                BlurredImageInterstitial {
                    reason: Some(reason),
                    opacity: Some(0.8.into()),
                    interstitial_action: None,
                    available_verification_options: None,
                },
            ))),
        }
    }

    fn thrift_blur(reason: InterstitialReason) -> Action {
        Action::ComposedMediaVisibilityResults(thrift_media(reason))
    }

    fn thrift_limit_for(
        reason: action::LimitedEngagementReason,
        policy: Option<LimitedActionsPolicy>,
        limited_actions: &str,
    ) -> Action {
        Action::LimitedEngagements(LimitedEngagements::new(
            reason,
            policy,
            limited_actions.to_string(),
        ))
    }

    fn thrift_limit() -> LimitedEngagements {
        LimitedEngagements::new(
            Some(action::LimitedEngagementReason::ConversationControl(
                action::ConversationControl::new(),
            )),
            LimitedActionsPolicy::new(vec![LimitedAction::new(
                action::LimitedActionType::REPLY,
                None,
            )]),
            "limited_replies".to_string(),
        )
    }

    fn limit() -> Decided<LimitedEngagement> {
        Decided {
            value: LimitedEngagement::new(LimitedEngagementReason::ConversationControl),
            by: "limit_rule",
        }
    }

    fn shown(
        media: Option<Decided<MediaRestriction>>,
        engagement: Option<Decided<LimitedEngagement>>,
    ) -> Verdict {
        Verdict::Shown {
            notice: None,
            media,
            engagement,
        }
    }

    fn thrift_drop(reason: Option<action::DropReason>) -> Action {
        Action::Drop(action::Drop::new(reason, None))
    }

    struct Projected {
        thrift: Action,
        proto: Kind,
        reason: Option<FilteredReason>,
        label: &'static str,
        rows: &'static [(&'static str, &'static str)],
    }

    fn verdict_cases() -> [(Verdict, Projected); 18] {
        use action::LimitedActionType as T;
        let proto_drop = Kind::Drop(vf_pb::DropReason {});
        let tombstone = |reason, code| {
            (
                tombstoned(reason),
                Projected {
                    thrift: Action::Tombstone(Tombstone::new(
                        Some(action::TombstoneReason(code)),
                        None,
                    )),
                    proto: proto_drop,
                    reason: Some(FilteredReason::UnspecifiedReason),
                    label: "tombstone",
                    rows: &[("rule", "tombstone")],
                },
            )
        };
        let nsfw_viewer_drop = |reason, thrift| {
            (
                Verdict::Withheld(Decided {
                    value: Withholding::Drop(DropReason::NsfwViewer(reason)),
                    by: "rule",
                }),
                Projected {
                    thrift: thrift_drop(Some(thrift)),
                    proto: proto_drop,
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "drop",
                    rows: &[("rule", "drop")],
                },
            )
        };
        [
            (
                shown(None, None),
                Projected {
                    thrift: Action::Allow(action::Allow::new()),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "allow",
                    rows: &[],
                },
            ),
            (
                dropped(FilteredReason::AuthorIsSuspended),
                Projected {
                    thrift: thrift_drop(Some(action::DropReason::SuspendedAuthor(true))),
                    proto: proto_drop,
                    reason: Some(FilteredReason::AuthorIsSuspended),
                    label: "drop",
                    rows: &[("rule", "drop")],
                },
            ),
            nsfw_viewer_drop(
                NsfwViewerDropReason::IsUnderage,
                action::DropReason::NsfwViewerIsUnderage(true),
            ),
            nsfw_viewer_drop(
                NsfwViewerDropReason::HasNoStatedAge,
                action::DropReason::NsfwViewerHasNoStatedAge(true),
            ),
            nsfw_viewer_drop(
                NsfwViewerDropReason::LoggedOut,
                action::DropReason::NsfwLoggedOut(true),
            ),
            tombstone(TombstoneReason::SensitiveViewerAgeVerification, 30),
            tombstone(TombstoneReason::UpdateAppIos, 31),
            tombstone(TombstoneReason::UpdateAppAndroid, 32),
            tombstone(TombstoneReason::LocalRegulations, 33),
            (
                shown(Some(blur(InterstitialReason::Sensitive(true))), None),
                Projected {
                    thrift: thrift_blur(InterstitialReason::Sensitive(true)),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "interstitial",
                    rows: &[("blur_rule", "interstitial")],
                },
            ),
            (
                shown(None, Some(limit())),
                Projected {
                    thrift: Action::LimitedEngagements(thrift_limit()),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "limited_engagement",
                    rows: &[("limit_rule", "limited_engagement")],
                },
            ),
            (
                limited_for(&[LimitedEngagementReason::StaleTweet], "limit_rule"),
                Projected {
                    thrift: thrift_limit_for(
                        action::LimitedEngagementReason::StaleTweet(action::StaleTweet::new()),
                        None,
                        "stale_tweet",
                    ),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "limited_engagement",
                    rows: &[("limit_rule", "limited_engagement")],
                },
            ),
            (
                shown(
                    Some(blur(InterstitialReason::Sensitive(true))),
                    Some(limit()),
                ),
                Projected {
                    thrift: Action::TweetInterstitial(TweetInterstitial {
                        limited_engagements: Some(thrift_limit()),
                        all_media_visibility_results: Some(thrift_media(
                            InterstitialReason::Sensitive(true),
                        )),
                        ..TweetInterstitial::default()
                    }),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "tweet_interstitial",
                    rows: &[
                        ("blur_rule", "interstitial"),
                        ("limit_rule", "limited_engagement"),
                    ],
                },
            ),
            (
                shown(Some(legacy_interstitial()), None),
                Projected {
                    thrift: Action::Interstitial(Interstitial::new(
                        InterstitialReason::ContainsNsfwMedia(true),
                        None,
                    )),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "interstitial",
                    rows: &[("fallback_rule", "interstitial")],
                },
            ),
            (
                shown(Some(legacy_interstitial()), Some(limit())),
                Projected {
                    thrift: Action::TweetInterstitial(TweetInterstitial {
                        interstitial: Some(AnyInterstitial::Interstitial(Interstitial::new(
                            InterstitialReason::ContainsNsfwMedia(true),
                            None,
                        ))),
                        limited_engagements: Some(thrift_limit()),
                        ..TweetInterstitial::default()
                    }),
                    proto: Kind::Interstitial(true),
                    reason: Some(FilteredReason::ContainNsfwMedia),
                    label: "tweet_interstitial",
                    rows: &[
                        ("fallback_rule", "interstitial"),
                        ("limit_rule", "limited_engagement"),
                    ],
                },
            ),
            (
                noticed(true, false, "fosnr_rule"),
                Projected {
                    thrift: Action::TweetInterstitial(TweetInterstitial {
                        soft_intervention: Some(ThriftSoftIntervention {
                            soft_intervention_reason: Some(SoftInterventionReason::Fosnr(
                                Box::new(AppealableReason::new(
                                    1,
                                    Box::new(action::AppealablePolicy::ABUSE),
                                    true,
                                    false,
                                )),
                            )),
                            engagement_nudge: Some(false),
                            suppress_autoplay: Some(true),
                            warning: None,
                            details_url: None,
                            display_type: Some(SoftInterventionDisplayType::FOSNR),
                        }),
                        avoid: Some(Avoid::new(None, Some(BrandSafetyRiskLevel::NORMAL), None)),
                        ..TweetInterstitial::default()
                    }),
                    proto: Kind::Allow(true),
                    reason: None,
                    label: "soft_intervention",
                    rows: &[("fosnr_rule", "soft_intervention")],
                },
            ),
            appeal(
                AppealablePolicy::VIOLENT_SPEECH,
                3,
                &[
                    T::LIKE,
                    T::REPLY,
                    T::RETWEET,
                    T::QUOTE_TWEET,
                    T::SHARE_TWEET_VIA,
                    T::ADD_TO_BOOKMARKS,
                    T::PIN_TO_PROFILE,
                    T::COPY_LINK,
                    T::SEND_VIA_DM,
                    T::EDIT_TWEET,
                    T::HIGHLIGHT,
                    T::EMBED,
                    T::LISTS_ADD_REMOVE,
                ],
                Some("freedom_of_speech_not_reach"),
            ),
            appeal(AppealablePolicy::ABUSE, 1, &[T::EDIT_TWEET], None),
        ]
    }

    fn appeal(
        policy: AppealablePolicy,
        level: i8,
        limited: &[action::LimitedActionType],
        legacy_limited_actions: Option<&str>,
    ) -> (Verdict, Projected) {
        let reason = AppealableReason::new(level, Box::new(policy), false, true);
        let limited = limited
            .iter()
            .map(|&action_type| LimitedAction::new(action_type, None))
            .collect();
        (
            appealed(policy, level, false, true, "fosnr_author"),
            Projected {
                thrift: Action::TweetInterstitial(TweetInterstitial {
                    limited_engagements: Some(LimitedEngagements::new(
                        action::LimitedEngagementReason::FosnrReason(reason.clone()),
                        LimitedActionsPolicy::new(limited),
                        legacy_limited_actions.map(str::to_string),
                    )),
                    avoid: Some(Avoid::new(None, Some(BrandSafetyRiskLevel::NORMAL), None)),
                    appealable: Some(Appealable::new(reason, None)),
                    ..TweetInterstitial::default()
                }),
                proto: Kind::Allow(true),
                reason: None,
                label: "appealable",
                rows: &[("fosnr_author", "appealable")],
            },
        )
    }

    #[test]
    fn every_verdict_case_projects_per_the_table() {
        for (verdict, expected) in verdict_cases() {
            let name = format!("{verdict:?}");
            assert_eq!(
                thrift_action(&verdict, TimelineHome, &policies()),
                expected.thrift,
                "{name}"
            );
            assert_eq!(metric_label(&verdict), expected.label, "{name}");
            assert_eq!(
                decided_rows(&verdict).collect::<Vec<_>>(),
                expected.rows,
                "{name}"
            );
            let (proto, reason) = proto_action(verdict);
            assert_eq!(proto.kind, Some(expected.proto), "{name}");
            assert_eq!(reason, expected.reason.map(Into::into), "{name}");
        }
    }

    #[test]
    fn a_limit_sends_its_first_reason_and_the_union_of_its_reasons_policies() {
        use LimitedActionType::{Like, Reply, Retweet};
        use LimitedEngagementReason::{BlockedViewer, ConversationControl, StaleTweet};
        let policies = LimitedActionsPolicies::for_tests(vec![
            (BlockedViewer, vec![Retweet, Reply, Like]),
            (ConversationControl, vec![Reply]),
        ]);
        let verdict = limited_for(
            &[StaleTweet, ConversationControl, BlockedViewer],
            "limit_rule",
        );
        let action = |action_type| LimitedAction::new(action_type, None);
        assert_eq!(
            thrift_action(&verdict, TimelineHome, &policies),
            thrift_limit_for(
                action::LimitedEngagementReason::StaleTweet(action::StaleTweet::new()),
                Some(LimitedActionsPolicy::new(vec![
                    action(action::LimitedActionType::REPLY),
                    action(action::LimitedActionType::RETWEET),
                    action(action::LimitedActionType::LIKE),
                ])),
                "stale_tweet",
            )
        );
        assert_eq!(metric_label(&verdict), "limited_engagement");
        assert_eq!(
            decided_rows(&verdict).collect::<Vec<_>>(),
            [("limit_rule", "limited_engagement")]
        );
    }

    #[test]
    fn each_limited_engagement_reason_sends_its_thrift_arm_and_legacy_string() {
        use action::LimitedEngagementReason as T;
        use LimitedEngagementReason as R;
        for (reason, arm, legacy) in [
            (
                R::ConversationControl,
                T::ConversationControl(action::ConversationControl::new()),
                "limited_replies",
            ),
            (
                R::ReadonlyViewer,
                T::ReadonlyViewer(action::ReadonlyViewer::new()),
                "readonly_viewer",
            ),
            (
                R::BlockedViewer,
                T::BlockedViewer(action::BlockedViewer::new()),
                "blocked_viewer",
            ),
            (
                R::RootAuthorBlockedViewer,
                T::RootAuthorBlockedViewer(action::RootAuthorBlockedViewer::new()),
                "root_author_blocked_viewer",
            ),
            (
                R::StaleTweet,
                T::StaleTweet(action::StaleTweet::new()),
                "stale_tweet",
            ),
            (
                R::CommunityTweetHidden,
                T::CommunityTweetHidden(action::CommunityTweetHidden::new()),
                "community_tweet_hidden",
            ),
            (
                R::CommunityTweetMemberRemoved,
                T::CommunityTweetMemberRemoved(action::CommunityTweetMemberRemoved::new()),
                "community_tweet_member_removed",
            ),
            (
                R::CommunityTweetCommunityNotFound,
                T::CommunityTweetCommunityNotFound(action::CommunityTweetCommunityNotFound::new()),
                "community_tweet_community_not_found",
            ),
            (
                R::CommunityTweetCommunityDeleted,
                T::CommunityTweetCommunityDeleted(action::CommunityTweetCommunityDeleted::new()),
                "community_tweet_community_deleted",
            ),
            (
                R::CommunityTweetCommunitySuspended,
                T::CommunityTweetCommunitySuspended(action::CommunityTweetCommunitySuspended::new()),
                "community_tweet_community_suspended",
            ),
            (
                R::CommunityTweetViewerRemoved,
                T::CommunityTweetViewerRemoved(action::CommunityTweetViewerRemoved::new()),
                "community_tweet_viewer_removed",
            ),
            (
                R::LocalTweet,
                T::LocalTweet(action::LocalTweet::new()),
                "local_tweet",
            ),
        ] {
            assert_eq!(
                thrift_action(
                    &limited_for(&[reason], "limit_rule"),
                    TimelineHome,
                    &LimitedActionsPolicies::default(),
                ),
                thrift_limit_for(arm, None, legacy),
                "{reason:?}"
            );
        }
    }

    #[test]
    fn entity_mixer_resolves_the_blur_and_its_prompt_from_both_media_arms() {
        let reason = InterstitialReason::Nudity(true);
        let verify = blur_with(
            reason.clone(),
            Some(InterstitialAction::AGE_VERIFICATION_PROMPT),
        );
        for engagement in [None, Some(limit())] {
            let verdict = shown(Some(verify.clone()), engagement);
            let rendered = resolve_blurred_image_interstitial(&ThriftSafetyResult::new(
                None,
                thrift_action(&verdict, TimelineHome, &policies()),
            ));
            assert_eq!(
                rendered,
                Some(BlurredImageInterstitial {
                    reason: Some(reason.clone()),
                    opacity: Some(0.8.into()),
                    interstitial_action: Some(InterstitialAction::AGE_VERIFICATION_PROMPT),
                    available_verification_options: Some(vec![
                        AgeVerificationOption::SELFIE,
                        AgeVerificationOption::PERSONA,
                    ]),
                }),
                "{verdict:?}"
            );
        }
    }

    #[test]
    fn drop_reasons_without_a_canonical_form_drop_without_a_reason() {
        assert_eq!(
            thrift_action(
                &dropped(FilteredReason::AuthorIsProtected),
                TimelineHome,
                &policies()
            ),
            thrift_drop(Some(action::DropReason::ProtectedAuthor(true)))
        );
        assert_eq!(
            thrift_action(
                &dropped(FilteredReason::UnspecifiedReason),
                FilterAll,
                &policies()
            ),
            thrift_drop(Some(action::DropReason::Unspecified(true)))
        );
        let lossy = [
            FilteredReason::UnspecifiedReason,
            FilteredReason::ContainNsfwMedia,
            FilteredReason::PossiblyUndesirable,
            FilteredReason::AuthorAccountIsInactive,
            FilteredReason::AuthorIsUnsafe,
            FilteredReason::ReportedTweet,
            FilteredReason::TweetMatchesViewerMutedKeyword(KeywordMatch {
                keyword: "kw".into(),
            }),
            FilteredReason::TweetIsBounced,
            FilteredReason::SafetyResult(SafetyResult::default()),
            FilteredReason::AuthorIsDeactivated,
            FilteredReason::TweetIsNullcast,
        ];
        for reason in lossy {
            let verdict = dropped(reason);
            assert_eq!(
                thrift_action(&verdict, TimelineHome, &policies()),
                thrift_drop(None),
                "{verdict:?}"
            );
        }
    }

    #[test]
    fn thrift_result_state_uses_a_legacy_arm_only_where_it_renders_differently() {
        use ThriftFilteredReason as T;
        let found = |reason| TweetFieldsResultState::Found(TweetFieldsResultFound::new(reason));
        let filtered =
            |reason| TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason));
        let safety_result_with = |reason, verdict: &Verdict| {
            T::SafetyResult(ThriftSafetyResult::new(
                reason,
                thrift_action(verdict, TimelineHome, &policies()),
            ))
        };
        let safety_result = |verdict: &Verdict| safety_result_with(None, verdict);
        let shown = |media, engagement| Verdict::Shown {
            notice: None,
            media,
            engagement,
        };
        let limited = Decided {
            value: LimitedEngagement::new(LimitedEngagementReason::ConversationControl),
            by: "limit_rule",
        };
        let nsfw = Decided {
            value: MediaRestriction::NsfwInterstitial,
            by: "nsfw_rule",
        };
        let mut cases = vec![
            (
                dropped(FilteredReason::TweetIsBounced),
                filtered(T::TweetIsBounced(true)),
            ),
            (
                dropped(FilteredReason::AuthorBlockViewer),
                filtered(T::AuthorBlockViewer(true)),
            ),
            (
                dropped(FilteredReason::AuthorIsProtected),
                filtered(T::AuthorIsProtected(true)),
            ),
            (
                dropped(FilteredReason::AuthorIsSuspended),
                filtered(T::AuthorIsSuspended(true)),
            ),
            (shown(None, None), found(None)),
            (
                shown(Some(nsfw.clone()), None),
                found(Some(T::ContainNsfwMedia(true))),
            ),
        ];
        for verdict in [
            dropped(FilteredReason::AuthorIsDeactivated),
            dropped(FilteredReason::ExclusiveTweet),
            Verdict::Withheld(Decided {
                value: Withholding::Drop(DropReason::NsfwViewer(NsfwViewerDropReason::IsUnderage)),
                by: "rule",
            }),
            tombstoned(TombstoneReason::LocalRegulations),
        ] {
            cases.push((verdict.clone(), filtered(safety_result(&verdict))));
        }
        for verdict in [
            shown(None, Some(limited.clone())),
            shown(Some(nsfw), Some(limited)),
            noticed(true, false, "fosnr_rule"),
        ] {
            cases.push((verdict.clone(), found(Some(safety_result(&verdict)))));
        }
        for (policy, reason) in [
            (
                AppealablePolicy::HATEFUL_CONDUCT,
                SafetyResultReason::FOSNR_HATEFUL_CONDUCT,
            ),
            (AppealablePolicy::ABUSE, SafetyResultReason::FOSNR_ABUSE),
            (
                AppealablePolicy::VIOLENT_SPEECH,
                SafetyResultReason::FOSNR_VIOLENT_SPEECH,
            ),
            (
                AppealablePolicy::CIVIC_INTEGRITY,
                SafetyResultReason::FOSNR_CIVIC_INTEGRITY,
            ),
        ] {
            let verdict = appealed(policy, 3, true, false, "fosnr_author");
            let expected = found(Some(safety_result_with(Some(reason), &verdict)));
            cases.push((verdict, expected));
        }
        for (verdict, expected) in cases {
            assert_eq!(
                thrift_result_state(&verdict, TimelineHome, &policies()),
                expected,
                "{verdict:?}"
            );
        }
    }

    #[test]
    fn dashboard_generator_pins_the_rpc_and_verdict_mix_labels() {
        let cargo = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/dashboard.py");
        let ws = "crates/x-product/xai-visibility-filtering-service/scripts/dashboard.py";
        let path = if std::path::Path::new(cargo).exists() {
            cargo
        } else {
            ws
        };
        let dashboard =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let filter_tweets = <&str>::from(Rpc::FilterTweets);
        assert!(dashboard.contains(&format!("FT_RPC_FILTER = 'rpc=~\"{filter_tweets}|\"'")));
        let actions = dashboard
            .split_once("FT_VERDICT_ACTIONS = (")
            .and_then(|(_, rest)| rest.split_once(')'))
            .map_or("", |(tuple, _)| tuple);
        for (_, expected) in verdict_cases() {
            let label = expected.label;
            assert!(actions.contains(&format!("\"{label}\",")), "{label}");
        }
    }
}
