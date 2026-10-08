use std::sync::Arc;
use xai_core_entities::entities::ConversationControl;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConversationControlFeatures {
    pub control: ConversationControl,
    pub viewer_country: Option<Arc<str>>,
}
