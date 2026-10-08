use crate::phoenix_scores::PhoenixScores;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CandidateScoringInputs {
    pub phoenix_scores: PhoenixScores,
    pub author_id: u64,
    pub in_network: Option<bool>,
    pub is_reply: bool,
    pub is_retweet: bool,
    pub is_mutual_follow_author: bool,
    pub vqv_eligible: bool,
    pub quoted_vqv_eligible: bool,
    pub cold_start_lift_to_rank: Option<u32>,
    pub weighted_score: Option<f64>,
    pub author_exploration_bonus: f64,
    pub user_video_continuation_secs: Option<f64>,
}

impl CandidateScoringInputs {
    pub fn bidirectional_boost_eligible(&self) -> bool {
        !self.is_reply && !self.is_retweet && self.is_mutual_follow_author
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryScoringContext {
    pub effective_oon_weight: f64,
}

impl Default for QueryScoringContext {
    fn default() -> Self {
        Self {
            effective_oon_weight: 1.0,
        }
    }
}
