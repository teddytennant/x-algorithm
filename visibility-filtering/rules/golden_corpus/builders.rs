use super::Role;
use crate::models::{
    AuthorFeatures, ConversationControlFeatures, HydratedTweetCandidate, SafetyLabelType,
    TweetFeatures, Verdict, VerifyBlurSupport, Viewer, ViewerAge, ViewerFeatures, ViewerProfile,
};
use crate::rules::fixtures::{
    candidate, conversation_control, viewer, viewer_with_profile, AUTHOR_ID, CLIENT_CLASSES,
    VIEWER_ID,
};
use crate::rules::SafetyLevel;
use xai_core_entities::entities::ConversationControlArm;

pub(super) fn author_candidate(set: fn(&mut AuthorFeatures)) -> HydratedTweetCandidate {
    let mut features = AuthorFeatures::default();
    set(&mut features);
    candidate().with_author_features(features).build()
}

pub(super) fn tweet_candidate(set: fn(&mut TweetFeatures)) -> HydratedTweetCandidate {
    let mut features = TweetFeatures::default();
    set(&mut features);
    candidate().with_tweet_features(features).build()
}

pub(super) fn labeled(label: SafetyLabelType) -> HydratedTweetCandidate {
    candidate().with_label(label).build()
}

pub(super) fn controlled_root(arm: ConversationControlArm) -> ConversationControlFeatures {
    conversation_control(arm, AUTHOR_ID)
}

pub(super) fn viewer_in_country(code: &str) -> ViewerFeatures {
    ViewerFeatures {
        country_code: Some(code.to_string()),
        ..viewer(VIEWER_ID)
    }
}

pub(super) fn per_client_class(
    verdict: impl Fn(VerifyBlurSupport) -> Verdict,
) -> Vec<(SafetyLevel, Role, Verdict)> {
    CLIENT_CLASSES
        .into_iter()
        .map(|class| {
            let support = class
                .capability
                .verify_blur_support
                .expect("every client class resolves a Verify blur support");
            let viewer = ViewerFeatures {
                client_capability: class.capability,
                ..viewer_in_country("fr")
            };
            (
                SafetyLevel::TimelineHomeHydration,
                Role::As(class.name, viewer),
                verdict(support),
            )
        })
        .collect()
}

pub(super) fn on_client(class_name: &str, country: &str, base: ViewerFeatures) -> ViewerFeatures {
    let class = CLIENT_CLASSES
        .iter()
        .find(|class| class.name == class_name)
        .unwrap_or_else(|| panic!("no client class {class_name}"));
    ViewerFeatures {
        country_code: Some(country.to_string()),
        client_capability: class.capability,
        ..base
    }
}

pub(super) fn nsfw_high_precision_media() -> HydratedTweetCandidate {
    candidate()
        .with_label(SafetyLabelType::NSFW_HIGH_PRECISION)
        .with_media()
        .build()
}

pub(super) fn viewer_with_age(age: ViewerAge) -> ViewerFeatures {
    viewer_with_profile(ViewerProfile {
        viewer_age: age,
        ..ViewerProfile::default()
    })
}

pub(super) fn no_stated_age_viewer(account_country_code: &str) -> ViewerFeatures {
    viewer_with_profile(ViewerProfile {
        viewer_age: ViewerAge::NotStated,
        account_country_code: Some(account_country_code.to_string()),
        ..ViewerProfile::default()
    })
}

pub(super) fn read_only_viewer(id: u64) -> ViewerFeatures {
    ViewerFeatures {
        viewer: Viewer::LoggedIn {
            id,
            profile: ViewerProfile {
                is_read_only: true,
                ..ViewerProfile::default()
            },
            has_age_verified_18_label: false,
        },
        ..ViewerFeatures::default()
    }
}
