use crate::rules::registry::{
    immersive_expanded_recommendations, timeline_home_hydration, timeline_home_recommendations,
    timeline_home_shared,
};
use crate::rules::rule_spec::RuleClause;
use crate::rules::tweet_rules;
use strum::VariantArray;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "snake_case")]
pub enum SafetyLevel {
    FilterAll,
    TimelineHome,
    TimelineHomeRecommendations,
    TimelineHomeHydration,
    ImmersiveExpandedRecommendations,
}

pub(crate) struct LevelSpec {
    proto: Option<vf_pb::SafetyLevel>,
    thrift: Option<ThriftLevel>,
    pub(super) rules: fn() -> Vec<RuleClause>,
    pub(crate) unspecified_drop_is_exact: bool,
}

impl SafetyLevel {
    pub(crate) const fn spec(self) -> LevelSpec {
        match self {
            SafetyLevel::FilterAll => LevelSpec {
                proto: Some(vf_pb::SafetyLevel::FilterAll),
                thrift: Some(ThriftLevel::FILTER_ALL),
                rules: tweet_rules::filter_all,
                unspecified_drop_is_exact: true,
            },
            SafetyLevel::TimelineHome => LevelSpec {
                proto: Some(vf_pb::SafetyLevel::TimelineHome),
                thrift: Some(ThriftLevel::TIMELINE_HOME),
                rules: timeline_home_shared,
                unspecified_drop_is_exact: false,
            },
            SafetyLevel::TimelineHomeRecommendations => LevelSpec {
                proto: Some(vf_pb::SafetyLevel::TimelineHomeRecommendations),
                thrift: Some(ThriftLevel::TIMELINE_HOME_RECOMMENDATIONS),
                rules: timeline_home_recommendations,
                unspecified_drop_is_exact: false,
            },
            SafetyLevel::TimelineHomeHydration => LevelSpec {
                proto: None,
                thrift: Some(ThriftLevel::TIMELINE_HOME_HYDRATION),
                rules: timeline_home_hydration,
                unspecified_drop_is_exact: false,
            },
            SafetyLevel::ImmersiveExpandedRecommendations => LevelSpec {
                proto: Some(vf_pb::SafetyLevel::ImmersiveExpandedRecommendations),
                thrift: None,
                rules: immersive_expanded_recommendations,
                unspecified_drop_is_exact: false,
            },
        }
    }

    pub(crate) fn from_proto(level: vf_pb::SafetyLevel) -> Option<Self> {
        Self::VARIANTS
            .iter()
            .copied()
            .find(|variant| variant.spec().proto == Some(level))
    }

    pub(crate) fn from_thrift(level: ThriftLevel) -> Option<Self> {
        Self::VARIANTS
            .iter()
            .copied()
            .find(|variant| variant.spec().thrift == Some(level))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_filter_tweets_level_reaches_its_level() {
        let reached: Vec<_> = (0..=255)
            .filter_map(|number| vf_pb::SafetyLevel::try_from(number).ok())
            .map(|proto| (proto, SafetyLevel::from_proto(proto)))
            .collect();
        assert_eq!(
            reached,
            [
                (vf_pb::SafetyLevel::FilterAll, Some(SafetyLevel::FilterAll)),
                (
                    vf_pb::SafetyLevel::TimelineHome,
                    Some(SafetyLevel::TimelineHome)
                ),
                (
                    vf_pb::SafetyLevel::TimelineHomeRecommendations,
                    Some(SafetyLevel::TimelineHomeRecommendations)
                ),
                (
                    vf_pb::SafetyLevel::ImmersiveExpandedRecommendations,
                    Some(SafetyLevel::ImmersiveExpandedRecommendations)
                ),
            ]
        );
    }

    #[test]
    fn evaluate_tweets_serves_these_thrift_levels() {
        let served: Vec<_> = ThriftLevel::ENUM_VALUES
            .iter()
            .filter_map(|&thrift| SafetyLevel::from_thrift(thrift).map(|level| (thrift, level)))
            .collect();
        assert_eq!(
            served,
            [
                (ThriftLevel::TIMELINE_HOME, SafetyLevel::TimelineHome),
                (ThriftLevel::FILTER_ALL, SafetyLevel::FilterAll),
                (
                    ThriftLevel::TIMELINE_HOME_RECOMMENDATIONS,
                    SafetyLevel::TimelineHomeRecommendations
                ),
                (
                    ThriftLevel::TIMELINE_HOME_HYDRATION,
                    SafetyLevel::TimelineHomeHydration
                ),
            ]
        );
    }

    #[test]
    fn each_level_has_rules_an_ingress_and_wire_ids_of_its_own() {
        for &level in SafetyLevel::VARIANTS {
            let spec = level.spec();
            assert!(!(spec.rules)().is_empty(), "{level:?} wires no rule");
            assert!(
                spec.proto.is_some() || spec.thrift.is_some(),
                "{level:?} has no ingress"
            );
            for &other in SafetyLevel::VARIANTS
                .iter()
                .filter(|&&other| other != level)
            {
                let other_spec = other.spec();
                assert!(
                    spec.proto.is_none() || spec.proto != other_spec.proto,
                    "{level:?} and {other:?} share a proto level"
                );
                assert!(
                    spec.thrift.is_none() || spec.thrift != other_spec.thrift,
                    "{level:?} and {other:?} share a Thrift level"
                );
            }
        }
    }
}
