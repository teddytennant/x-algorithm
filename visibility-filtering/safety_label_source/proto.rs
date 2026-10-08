use std::collections::HashMap;

use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelSource};

pub(crate) fn label_to_proto(label: SafetyLabel) -> vf_pb::SafetyLabel {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "SafetyLabelSource is a generated Thrift union; vf_pb::SafetyLabel carries only the BotMaker, tool and Grok annotation sources, so it omits every other source, including any the IDL adds"
    )]
    let safety_label_source = label.safety_label_source.and_then(|src| match src {
        SafetyLabelSource::BotMakerAction(a) => {
            Some(vf_pb::safety_label::SafetyLabelSource::BotmakerAction(
                vf_pb::BotmakerAction { rule_id: a.rule_id },
            ))
        }
        SafetyLabelSource::ToolAction(a) => Some(
            vf_pb::safety_label::SafetyLabelSource::ToolAction(vf_pb::ToolAction {
                agent_tool: i32::from(a.agent_tool),
                actor_ldap: a.actor_ldap,
            }),
        ),
        SafetyLabelSource::GrokAnnotationAction(a) => Some(
            vf_pb::safety_label::SafetyLabelSource::GrokAnnotationAction(
                vf_pb::GrokAnnotationAction {
                    source: i32::from(a.grok_annotation_source),
                },
            ),
        ),
        _ => None,
    });

    vf_pb::SafetyLabel {
        score: label.score.map(|f| f.into_inner()),
        applicable_users: label.applicable_users.map_or_else(Vec::new, |users| {
            users.into_iter().map(|u| u.user_id).collect()
        }),
        holdback_experiment: label.holdback_experiment,
        source: label.source,
        created_at_msec: label.created_at_msec,
        expires_at_msec: label.expires_at_msec,
        applicable_countries: label.applicable_countries.unwrap_or_default(),
        safety_label_source,
    }
}

pub(crate) fn label_map_to_proto(
    labels: xai_safety_label_store::types::SafetyLabelMap,
) -> vf_pb::SafetyLabelMap {
    let map: HashMap<i32, vf_pb::SafetyLabel> = labels
        .into_iter()
        .map(|(lt, label)| (i32::from(lt), label_to_proto(label)))
        .collect();
    vf_pb::SafetyLabelMap { labels: map }
}
