use crate::hydration::batch::{Hydrated, HydrationBatch, HydrationError, RawHydrationBatch};
use crate::hydration::fallback_cache::{Column, FallbackCache};
use crate::hydration::metrics::record_author_labels;
use crate::models::{AuthorFeatures, AuthorLabel, AuthorLabelSet};
use strum::VariantArray;
use xai_core_entities::entities::{GizmoduckUserResult, UserResponseState};
use xai_x_thrift::user_labels::LabelValue;

pub(crate) type DecodedAuthor = (AuthorFeatures, AuthorLabelSet);
pub(crate) type AuthorFallbackCache = FallbackCache<DecodedAuthor>;

pub(crate) fn fallback_cache(capacity: usize) -> AuthorFallbackCache {
    FallbackCache::new("author", capacity)
}

pub(crate) struct AuthorColumn;

impl Column for AuthorColumn {
    type Entry = DecodedAuthor;
    type Value = DecodedAuthor;
    type Stored = DecodedAuthor;
    const NAME: &'static str = "author";

    fn store(value: &DecodedAuthor) -> DecodedAuthor {
        *value
    }

    fn new_entry(stored: DecodedAuthor) -> DecodedAuthor {
        stored
    }

    fn replace(entry: &mut DecodedAuthor, stored: DecodedAuthor) -> Option<DecodedAuthor> {
        Some(std::mem::replace(entry, stored))
    }

    fn get(entry: &DecodedAuthor) -> Option<DecodedAuthor> {
        Some(*entry)
    }

    fn holds(_: &DecodedAuthor) -> bool {
        true
    }

    fn others_hold(_: &DecodedAuthor) -> bool {
        false
    }

    fn clear(_: &mut DecodedAuthor) -> Option<DecodedAuthor> {
        None
    }
}

pub(crate) fn author_batch(
    users: RawHydrationBatch<GizmoduckUserResult>,
) -> RawHydrationBatch<DecodedAuthor> {
    let mut label_counts = LabelCounts::default();
    let authors = users
        .into_hydrated()
        .into_iter()
        .map(|(author, user)| {
            let decoded = match user {
                Hydrated::Found(result) | Hydrated::Partial(result) => {
                    evaluable_author_features(result, &mut label_counts)
                }
                Hydrated::NotFound => Hydrated::NotFound,
                Hydrated::Failed(error) => Hydrated::Failed(error),
            };
            (author, decoded)
        })
        .collect();
    record_author_labels(label_counts.mapped, label_counts.unmapped);
    HydrationBatch::from_hydrated(authors)
}

#[derive(Default)]
struct LabelCounts {
    mapped: usize,
    unmapped: usize,
}

fn evaluable_author_features(
    result: GizmoduckUserResult,
    counts: &mut LabelCounts,
) -> Hydrated<DecodedAuthor> {
    match result.response_state {
        Some(UserResponseState::Failed | UserResponseState::Filtered) => {
            Hydrated::Failed(HydrationError::Error)
        }
        Some(
            UserResponseState::NotFound
            | UserResponseState::SoftUser
            | UserResponseState::PeriscopeUser
            | UserResponseState::NoScreenNameUser,
        ) => Hydrated::NotFound,
        None | Some(UserResponseState::Found | UserResponseState::Partial)
            if result.user.is_none() =>
        {
            Hydrated::NotFound
        }
        None | Some(UserResponseState::Partial) => {
            Hydrated::Partial(author_features(result, counts))
        }
        Some(
            UserResponseState::Found
            | UserResponseState::DeactivatedUser
            | UserResponseState::SuspendedUser
            | UserResponseState::ProtectedUser
            | UserResponseState::ErasedUser
            | UserResponseState::UnsafeUser
            | UserResponseState::OffboardedUser,
        ) => Hydrated::Found(author_features(result, counts)),
    }
}

fn author_features(user_result: GizmoduckUserResult, counts: &mut LabelCounts) -> DecodedAuthor {
    user_result
        .user
        .map(|user| {
            let mut user_labels = AuthorLabelSet::default();
            for label in &user.labels.labels {
                match author_label(LabelValue(label.label_value)) {
                    Some(modeled) => {
                        user_labels.insert(modeled);
                        counts.mapped += 1;
                    }
                    None => counts.unmapped += 1,
                }
            }
            let features = AuthorFeatures {
                is_suspended: user.safety.suspended,
                is_deactivated: user.safety.deactivated,
                is_protected: user.safety.is_protected,
                is_nsfw_user: user.safety.nsfw_user,
                is_nsfw_admin: user.safety.nsfw_admin,
                is_erased: user.safety.erased,
                is_offboarded: user.safety.offboarded,
            };
            (features, user_labels)
        })
        .unwrap_or_default()
}

fn author_label(value: LabelValue) -> Option<AuthorLabel> {
    AuthorLabel::VARIANTS
        .iter()
        .copied()
        .find(|&label| gizmoduck_label(label) == value)
}

const fn gizmoduck_label(label: AuthorLabel) -> LabelValue {
    match label {
        AuthorLabel::NsfwHighRecall => LabelValue::NSFW_HIGH_RECALL,
        AuthorLabel::NsfwHighPrecision => LabelValue::NSFW_HIGH_PRECISION,
        AuthorLabel::NsfwNearPerfect => LabelValue::NSFW_NEAR_PERFECT,
        AuthorLabel::NsfwAvatarImage => LabelValue::NSFW_AVATAR_IMAGE,
        AuthorLabel::NsfwBannerImage => LabelValue::NSFW_BANNER_IMAGE,
        AuthorLabel::SpamHighRecall => LabelValue::SPAM_HIGH_RECALL,
        AuthorLabel::AbusiveHighRecall => LabelValue::ABUSIVE_HIGH_RECALL,
        AuthorLabel::Compromised => LabelValue::COMPROMISED,
        AuthorLabel::ReadOnly => LabelValue::READ_ONLY,
        AuthorLabel::ImpersonationHighPrecision => LabelValue::IMPERSONATION_HIGH_PRECISION,
        AuthorLabel::DoNotAmplify => LabelValue::DO_NOT_AMPLIFY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_core_entities::entities::{GizmoduckUser, Label, Labels, Safety};

    #[test]
    fn response_states_decode_as_tweetypie_reads_them() {
        let suspended = GizmoduckUserResult {
            user: Some(GizmoduckUser {
                safety: Safety {
                    suspended: true,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        };
        let found = Hydrated::Found(());
        for (state, expected) in [
            (Some(UserResponseState::Found), found.clone()),
            (Some(UserResponseState::DeactivatedUser), found.clone()),
            (Some(UserResponseState::SuspendedUser), found.clone()),
            (Some(UserResponseState::ProtectedUser), found.clone()),
            (Some(UserResponseState::ErasedUser), found.clone()),
            (Some(UserResponseState::OffboardedUser), found.clone()),
            (Some(UserResponseState::UnsafeUser), found),
            (Some(UserResponseState::Partial), Hydrated::Partial(())),
            (None, Hydrated::Partial(())),
            (Some(UserResponseState::NotFound), Hydrated::NotFound),
            (Some(UserResponseState::SoftUser), Hydrated::NotFound),
            (Some(UserResponseState::PeriscopeUser), Hydrated::NotFound),
            (
                Some(UserResponseState::NoScreenNameUser),
                Hydrated::NotFound,
            ),
            (
                Some(UserResponseState::Failed),
                Hydrated::Failed(HydrationError::Error),
            ),
            (
                Some(UserResponseState::Filtered),
                Hydrated::Failed(HydrationError::Error),
            ),
        ] {
            let author = evaluable_author_features(
                GizmoduckUserResult {
                    response_state: state,
                    ..suspended.clone()
                },
                &mut LabelCounts::default(),
            );
            assert_eq!(shape(&author), expected, "{state:?}");
            if let Some((features, _)) = author.value() {
                assert!(features.is_suspended, "{state:?}");
            }
        }
    }

    #[test]
    fn a_success_state_without_a_user_is_not_found() {
        for state in [
            None,
            Some(UserResponseState::Found),
            Some(UserResponseState::Partial),
        ] {
            let author = evaluable_author_features(
                GizmoduckUserResult {
                    response_state: state,
                    user: None,
                },
                &mut LabelCounts::default(),
            );
            assert_eq!(shape(&author), Hydrated::NotFound, "{state:?}");
        }
    }

    fn shape(author: &Hydrated<DecodedAuthor>) -> Hydrated<()> {
        match author {
            Hydrated::Found(_) => Hydrated::Found(()),
            Hydrated::Partial(_) => Hydrated::Partial(()),
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed(error) => Hydrated::Failed(error.clone()),
        }
    }

    fn user_with_labels(label_values: &[i32]) -> GizmoduckUserResult {
        GizmoduckUserResult {
            user: Some(GizmoduckUser {
                user_id: 1,
                labels: Labels {
                    labels: label_values
                        .iter()
                        .map(|&label_value| Label {
                            label_value,
                            created_at_msec: 0,
                        })
                        .collect(),
                },
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn every_author_label_variant_round_trips_from_its_thrift_constant() {
        for (thrift, variant) in [
            (LabelValue::NSFW_HIGH_RECALL, AuthorLabel::NsfwHighRecall),
            (
                LabelValue::NSFW_HIGH_PRECISION,
                AuthorLabel::NsfwHighPrecision,
            ),
            (LabelValue::NSFW_NEAR_PERFECT, AuthorLabel::NsfwNearPerfect),
            (LabelValue::NSFW_AVATAR_IMAGE, AuthorLabel::NsfwAvatarImage),
            (LabelValue::NSFW_BANNER_IMAGE, AuthorLabel::NsfwBannerImage),
            (LabelValue::SPAM_HIGH_RECALL, AuthorLabel::SpamHighRecall),
            (
                LabelValue::ABUSIVE_HIGH_RECALL,
                AuthorLabel::AbusiveHighRecall,
            ),
            (LabelValue::COMPROMISED, AuthorLabel::Compromised),
            (LabelValue::READ_ONLY, AuthorLabel::ReadOnly),
            (
                LabelValue::IMPERSONATION_HIGH_PRECISION,
                AuthorLabel::ImpersonationHighPrecision,
            ),
            (LabelValue::DO_NOT_AMPLIFY, AuthorLabel::DoNotAmplify),
        ] {
            let mut counts = LabelCounts::default();
            let (_, labels) = author_features(user_with_labels(&[thrift.0]), &mut counts);
            assert!(labels.has_label(variant), "{thrift:?}");
            assert_eq!((counts.mapped, counts.unmapped), (1, 0), "{thrift:?}");
        }

        for unmodelled in [
            LabelValue::EGREGIOUS_NSFW,
            LabelValue::RECOMMENDATIONS_BLACKLIST,
        ] {
            let mut counts = LabelCounts::default();
            let (_, labels) = author_features(
                user_with_labels(&[unmodelled.0, LabelValue::SPAM_HIGH_RECALL.0]),
                &mut counts,
            );
            let mut expected = AuthorLabelSet::default();
            expected.insert(AuthorLabel::SpamHighRecall);
            assert_eq!(labels, expected, "{unmodelled:?}");
            assert_eq!((counts.mapped, counts.unmapped), (1, 1), "{unmodelled:?}");
        }
    }
}
