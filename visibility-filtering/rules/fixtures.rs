use crate::hydration::Hydrator;
use crate::models::{
    AuthorFeatures, AuthorLabel, ClientCapability, ConversationControlFeatures, Decided,
    DropReason, FosnrReason, HydratedTweetCandidate, LimitedEngagement, LimitedEngagementReason,
    MediaInterstitial, MediaRestriction, Notice, NsfwViewerDropReason, SafetyLabelMap,
    SafetyLabelType, TombstoneReason, TweetFeatures, Verdict, VerifyBlurSupport, Viewer,
    ViewerFeatures, ViewerProfile, Withholding,
};
use crate::params::LimitedActionType;
use std::collections::HashSet;
use xai_core_entities::entities::{ConversationControl, ConversationControlArm};
use xai_visibility_filtering::models::FilteredReason;
use xai_x_thrift::action::{AppealablePolicy, InterstitialAction, InterstitialReason};

const TWEET_ID: u64 = 1;
pub(super) const AUTHOR_ID: u64 = 100;
pub(crate) const VIEWER_ID: u64 = 999;

#[derive(Clone, Copy)]
pub(crate) struct ClientClass {
    pub(crate) name: &'static str,
    pub(crate) app_id: i64,
    pub(crate) user_agent: &'static str,
    pub(crate) capability: ClientCapability,
}

pub(crate) const CLIENT_CLASSES: [ClientClass; 8] = {
    use VerifyBlurSupport::{AndroidNeedsUpdate, IosNeedsUpdate, Supported, Unsupported};
    const RWEB: i64 = 3033300;
    const IPHONE: i64 = 129032;
    const ANDROID: i64 = 258901;
    const MAC: i64 = 557701;
    const fn class(
        name: &'static str,
        app_id: i64,
        user_agent: &'static str,
        verify_blur_support: VerifyBlurSupport,
        modern_blur: bool,
        gore_blur_ignores_settings: bool,
    ) -> ClientClass {
        ClientClass {
            name,
            app_id,
            user_agent,
            capability: ClientCapability {
                verify_blur_support: Some(verify_blur_support),
                modern_blur,
                stale_tweet_limits: true,
                community_viewer_removed_limits: matches!(app_id, RWEB | IPHONE | ANDROID),
                gore_blur_ignores_settings,
                fosnr_rules: true,
                fosnr_fallback_drops: false,
            },
        }
    }
    const fn android_9_82(class: ClientClass) -> ClientClass {
        ClientClass {
            capability: ClientCapability {
                fosnr_rules: false,
                fosnr_fallback_drops: true,
                community_viewer_removed_limits: false,
                ..class.capability
            },
            ..class
        }
    }
    [
        class(
            "web",
            RWEB,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
             (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            Supported,
            true,
            true,
        ),
        class(
            "ios_current",
            IPHONE,
            "Twitter-iPhone/11.11.5 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)",
            Supported,
            true,
            true,
        ),
        class(
            "ios_outdated",
            IPHONE,
            "Twitter-iPhone/11.11.4 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)",
            IosNeedsUpdate,
            true,
            true,
        ),
        class(
            "android_current",
            ANDROID,
            "TwitterAndroid/11.11.0-release.00 (311110000-r-0) Pixel 7/14 \
             (Google;panther;google;panther;0;;1;2022)",
            Supported,
            true,
            true,
        ),
        class(
            "android_outdated",
            ANDROID,
            "TwitterAndroid/11.10.9-release.00 (311109000-r-0) Pixel 7/14 \
             (Google;panther;google;panther;0;;1;2022)",
            AndroidNeedsUpdate,
            true,
            true,
        ),
        android_9_82(class(
            "android_without_fosnr",
            ANDROID,
            "TwitterAndroid/9.82.0-release.00 (29820000-r-0) Pixel 7/14 \
             (Google;panther;google;panther;0;;1;2022)",
            AndroidNeedsUpdate,
            false,
            false,
        )),
        class(
            "mac_app",
            MAC,
            "Twitter-Mac/11.11.5 macOS/14.0 (Apple;Mac14,2)",
            Unsupported,
            false,
            false,
        ),
        class(
            "third_party_client",
            0,
            "ThirdPartyClient/2.0",
            Unsupported,
            false,
            false,
        ),
    ]
};

pub(crate) fn allow() -> Verdict {
    Verdict::Shown {
        notice: None,
        media: None,
        engagement: None,
    }
}

pub(crate) fn noticed(proactive: bool, appeal_submitted: bool, by: &'static str) -> Verdict {
    Verdict::Shown {
        notice: Some(Decided {
            value: Notice::SoftIntervention(FosnrReason {
                policy: AppealablePolicy::ABUSE,
                level: 1,
                proactive,
                appeal_submitted,
            }),
            by,
        }),
        media: None,
        engagement: None,
    }
}

pub(crate) fn appealed(
    policy: AppealablePolicy,
    level: i8,
    proactive: bool,
    appeal_submitted: bool,
    by: &'static str,
) -> Verdict {
    use LimitedActionType as A;
    let limited_actions: &'static [LimitedActionType] = if level == 1 {
        &[A::EditTweet]
    } else {
        &[
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
    Verdict::Shown {
        notice: Some(Decided {
            value: Notice::Appealable {
                reason: FosnrReason {
                    policy,
                    level,
                    proactive,
                    appeal_submitted,
                },
                limited_actions,
            },
            by,
        }),
        media: None,
        engagement: None,
    }
}

pub(crate) fn dropped(reason: FilteredReason, by: &'static str) -> Verdict {
    Verdict::Withheld(Decided {
        value: Withholding::Drop(DropReason::Legacy(reason)),
        by,
    })
}

pub(crate) fn nsfw_viewer_dropped(reason: NsfwViewerDropReason, by: &'static str) -> Verdict {
    Verdict::Withheld(Decided {
        value: Withholding::Drop(DropReason::NsfwViewer(reason)),
        by,
    })
}

pub(crate) fn tombstoned(reason: TombstoneReason, by: &'static str) -> Verdict {
    Verdict::Withheld(Decided {
        value: Withholding::Tombstone(reason),
        by,
    })
}

pub(crate) fn blurred(reason: InterstitialReason, by: &'static str) -> Verdict {
    media_blurred(reason, None, by)
}

pub(crate) fn verify_blurred(reason: InterstitialReason, by: &'static str) -> Verdict {
    media_blurred(
        reason,
        Some(InterstitialAction::AGE_VERIFICATION_PROMPT),
        by,
    )
}

fn media_blurred(
    reason: InterstitialReason,
    prompt: Option<InterstitialAction>,
    by: &'static str,
) -> Verdict {
    Verdict::Shown {
        notice: None,
        media: Some(Decided {
            value: MediaRestriction::MediaInterstitial(MediaInterstitial {
                legacy: FilteredReason::ContainNsfwMedia,
                reason,
                prompt,
            }),
            by,
        }),
        engagement: None,
    }
}

pub(crate) fn legacy_interstitial(by: &'static str) -> Verdict {
    Verdict::Shown {
        notice: None,
        media: Some(Decided {
            value: MediaRestriction::NsfwInterstitial,
            by,
        }),
        engagement: None,
    }
}

pub(crate) fn limited(reason: LimitedEngagementReason, by: &'static str) -> Verdict {
    limited_for(&[reason], by)
}

pub(crate) fn limited_for(reasons: &[LimitedEngagementReason], by: &'static str) -> Verdict {
    let (&first, rest) = reasons.split_first().unwrap();
    let mut value = LimitedEngagement::new(first);
    for &reason in rest {
        value.add(reason);
    }
    Verdict::Shown {
        notice: None,
        media: None,
        engagement: Some(Decided { value, by }),
    }
}

pub(crate) fn blurred_and_limited(blur: Verdict, limit: Verdict) -> Verdict {
    match (blur, limit) {
        (Verdict::Shown { media, .. }, Verdict::Shown { engagement, .. }) => Verdict::Shown {
            notice: None,
            media,
            engagement,
        },
        (blur, limit) => panic!("expected two Shown verdicts, got {blur:?} and {limit:?}"),
    }
}

pub(crate) fn viewer(id: u64) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn {
            id,
            profile: ViewerProfile::default(),
            has_age_verified_18_label: false,
        },
        ..Default::default()
    }
}

pub(crate) fn viewer_with_profile(profile: ViewerProfile) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn {
            id: VIEWER_ID,
            profile,
            has_age_verified_18_label: false,
        },
        ..Default::default()
    }
}

pub(crate) fn author_viewer() -> ViewerFeatures {
    viewer(AUTHOR_ID)
}

pub(crate) fn logged_out_viewer() -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedOut,
        ..Default::default()
    }
}

pub(crate) fn sensitive_opt_in_viewer() -> ViewerFeatures {
    viewer_with_profile(ViewerProfile {
        allows_sensitive_media: true,
        ..ViewerProfile::default()
    })
}

pub(super) fn conversation_control(
    arm: ConversationControlArm,
    root_author_id: u64,
) -> ConversationControlFeatures {
    ConversationControlFeatures {
        control: ConversationControl {
            arm,
            conversation_tweet_author_id: root_author_id,
            invited_user_ids: vec![],
            invite_via_mention: None,
            allowed_country_codes: vec![],
        },
        viewer_country: None,
    }
}

pub(crate) fn candidate() -> CandidateBuilder {
    CandidateBuilder {
        candidate: HydratedTweetCandidate {
            tweet_id: TWEET_ID,
            author_id: AUTHOR_ID,
            ..Default::default()
        },
        labels: HashSet::new(),
        agent_labels: HashSet::new(),
    }
}

pub(crate) struct CandidateBuilder {
    candidate: HydratedTweetCandidate,
    labels: HashSet<SafetyLabelType>,
    agent_labels: HashSet<SafetyLabelType>,
}

impl CandidateBuilder {
    pub(crate) fn tweet_id(mut self, id: u64) -> Self {
        self.candidate.tweet_id = id;
        self
    }

    pub(crate) fn with_label(mut self, label: SafetyLabelType) -> Self {
        self.labels.insert(label);
        self
    }

    pub(crate) fn with_agent_label(mut self, label: SafetyLabelType) -> Self {
        self.agent_labels.insert(label);
        self.with_label(label)
    }

    pub(crate) fn with_author_user_label(mut self, label: AuthorLabel) -> Self {
        self.candidate.author_labels.insert(label);
        self
    }

    pub(crate) fn with_tweet_features(mut self, features: TweetFeatures) -> Self {
        self.candidate.tweet_features = features;
        self
    }

    pub(crate) fn with_author_features(mut self, features: AuthorFeatures) -> Self {
        self.candidate.author_features = features;
        self
    }

    pub(crate) fn with_edge(mut self, edge: Hydrator) -> Self {
        debug_assert!(edge.is_edge(), "{edge:?} is not an edge node");
        self.candidate.edges = self.candidate.edges.with(edge);
        self
    }

    pub(crate) fn with_media(mut self) -> Self {
        self.candidate.tweet_features.media.has_media = true;
        self
    }

    pub(crate) fn with_conversation_control(
        mut self,
        features: ConversationControlFeatures,
    ) -> Self {
        self.candidate.conversation_control = Some(features);
        self
    }

    pub(crate) fn retweet_of(mut self, source_tweet_id: u64) -> Self {
        self.candidate.source_tweet_id = Some(source_tweet_id);
        self
    }

    pub(crate) fn build(self) -> HydratedTweetCandidate {
        let mut candidate = self.candidate;
        if !self.labels.is_empty() {
            candidate.safety_labels = self.agent_labels.into_iter().fold(
                SafetyLabelMap::new(self.labels),
                SafetyLabelMap::assigned_by_agent,
            );
        }
        candidate
    }
}
