pub use xai_x_thrift::tweet_safety_label::SafetyLabelType;

use std::collections::HashSet;
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelMap {
    types: HashSet<SafetyLabelType>,
    by_agent: HashSet<SafetyLabelType>,
}

impl SafetyLabelMap {
    #[cfg(test)]
    pub fn new(label_types: HashSet<SafetyLabelType>) -> Self {
        Self {
            types: label_types,
            by_agent: HashSet::new(),
        }
    }

    #[cfg(test)]
    pub fn assigned_by_agent(mut self, label_type: SafetyLabelType) -> Self {
        self.by_agent.insert(label_type);
        self
    }

    pub fn from_proto_label_types(proto: &vf_pb::SafetyLabelMap) -> Self {
        let mut map = Self::default();
        for (&label_type, label) in &proto.labels {
            let label_type = SafetyLabelType(label_type);
            map.types.insert(label_type);
            if matches!(
                label.safety_label_source,
                Some(vf_pb::safety_label::SafetyLabelSource::ToolAction(_))
            ) {
                map.by_agent.insert(label_type);
            }
        }
        map
    }

    #[inline]
    pub fn has_label(&self, label_type: SafetyLabelType) -> bool {
        self.types.contains(&label_type)
    }

    #[inline]
    pub fn is_by_agent(&self, label_type: SafetyLabelType) -> bool {
        self.by_agent.contains(&label_type)
    }
}
