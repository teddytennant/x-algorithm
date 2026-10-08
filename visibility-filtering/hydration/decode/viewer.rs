use crate::models::{ViewerAge, ViewerProfile};
use xai_core_entities::entities::{
    AccessPolicy, GizmoduckUser, Labels, SubscriptionLevel, VerifiedType,
};
use xai_core_entities::gizmoduck_client::{QueryFields, ViewerData};
use xai_x_thrift::user_labels::LabelValue;

fn has_verified_badge(data: &ViewerData) -> bool {
    matches!(
        data.verified_type,
        Some(VerifiedType::Business | VerifiedType::Government)
    ) || data.is_blue_verified
}

fn has_idv_premium(data: &ViewerData) -> bool {
    matches!(
        data.subscription_level,
        Some(SubscriptionLevel::Premium | SubscriptionLevel::PremiumPlus)
    ) || matches!(
        data.verified_type,
        Some(VerifiedType::Business | VerifiedType::Government)
    )
}

fn has_age_verified_18_label(labels: &Labels) -> bool {
    labels
        .labels
        .iter()
        .any(|label| LabelValue(label.label_value) == LabelValue::AGE_VERIFIED_18)
}

#[derive(Default)]
pub(crate) struct DecodedViewer {
    pub(crate) profile: ViewerProfile,
    pub(crate) has_age_verified_18_label: bool,
}

pub(crate) fn decode_viewer(user: Option<&GizmoduckUser>, fields: &[QueryFields]) -> DecodedViewer {
    match user {
        Some(user) => DecodedViewer {
            profile: profile(ViewerData::from_user(user, fields)),
            has_age_verified_18_label: has_age_verified_18_label(&user.labels),
        },
        None => DecodedViewer::default(),
    }
}

fn profile(data: ViewerData) -> ViewerProfile {
    let viewer_age = match data.age_in_years {
        Some(age) => ViewerAge::Known(age),
        None if data.user_exists => ViewerAge::NotStated,
        None => ViewerAge::Unknown,
    };
    ViewerProfile {
        allows_sensitive_media: data.nsfw_view.unwrap_or(false),
        viewer_age,
        has_verified_badge: has_verified_badge(&data),
        has_idv_premium: has_idv_premium(&data),
        is_read_only: data.access_policy == AccessPolicy::BounceAllPublicWrites,
        account_country_code: data.account_country_code.map(|c| c.to_ascii_lowercase()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::Label;

    #[test]
    fn viewer_existence_and_preference_determine_age_and_sensitive_media() {
        for (data, expected_age, expected_sensitive_media) in [
            (
                ViewerData {
                    user_exists: true,
                    nsfw_view: Some(false),
                    age_in_years: None,
                    ..Default::default()
                },
                ViewerAge::NotStated,
                false,
            ),
            (
                ViewerData {
                    user_exists: true,
                    nsfw_view: Some(true),
                    age_in_years: None,
                    ..Default::default()
                },
                ViewerAge::NotStated,
                true,
            ),
            (ViewerData::default(), ViewerAge::Unknown, false),
        ] {
            let viewer = profile(data);
            assert_eq!(viewer.viewer_age, expected_age);
            assert_eq!(viewer.allows_sensitive_media, expected_sensitive_media);
        }
    }

    #[test]
    fn verified_badge_requires_org_type_or_blue_check() {
        let cases = [
            (Some(VerifiedType::Business), false, true),
            (Some(VerifiedType::Government), false, true),
            (None, true, true),
            (Some(VerifiedType::User), false, false),
            (Some(VerifiedType::Notable), false, false),
            (None, false, false),
        ];
        for (verified_type, is_blue_verified, expected) in cases {
            let data = ViewerData {
                verified_type,
                is_blue_verified,
                ..Default::default()
            };
            assert_eq!(
                profile(data).has_verified_badge,
                expected,
                "{verified_type:?} blue={is_blue_verified}"
            );
        }
    }

    #[test]
    fn only_bounce_all_public_writes_makes_the_viewer_read_only() {
        for (access_policy, expected) in [
            (AccessPolicy::Normal, false),
            (AccessPolicy::BounceAll, false),
            (AccessPolicy::BounceAllWritesAndNpci, false),
            (AccessPolicy::BounceAllPublicWrites, true),
            (AccessPolicy::BounceOnUnsuspension, false),
        ] {
            let viewer = profile(ViewerData {
                access_policy,
                ..Default::default()
            });
            assert_eq!(viewer.is_read_only, expected, "{access_policy:?}");
        }
    }

    #[test]
    fn idv_premium_is_a_premium_tier_or_an_org_verified_type() {
        use SubscriptionLevel::{Basic, Premium, PremiumPlus};
        use VerifiedType::{Business, Government, Notable, User};
        let cases = [
            (Some(Premium), None, true),
            (Some(PremiumPlus), None, true),
            (Some(Basic), Some(Business), true),
            (None, Some(Government), true),
            (Some(Basic), Some(User), false),
            (None, Some(Notable), false),
            (None, None, false),
        ];
        for (subscription_level, verified_type, expected) in cases {
            let data = ViewerData {
                subscription_level,
                verified_type,
                ..Default::default()
            };
            assert_eq!(
                profile(data).has_idv_premium,
                expected,
                "{subscription_level:?} {verified_type:?}"
            );
        }
    }

    #[test]
    fn only_the_age_verified_18_label_marks_the_viewer() {
        let labels = |values: &[LabelValue]| Labels {
            labels: values
                .iter()
                .map(|value| Label {
                    label_value: value.0,
                    created_at_msec: 0,
                })
                .collect(),
        };
        for (values, expected) in [
            (
                &[LabelValue::AGE_VERIFIED_16, LabelValue::AGE_VERIFIED_18][..],
                true,
            ),
            (&[LabelValue::AGE_VERIFIED_16], false),
            (&[], false),
        ] {
            let user = GizmoduckUser {
                labels: labels(values),
                ..Default::default()
            };
            let fields = [QueryFields::SAFETY, QueryFields::LABELS];
            assert_eq!(
                decode_viewer(Some(&user), &fields).has_age_verified_18_label,
                expected,
                "{values:?}"
            );
        }
    }
}
