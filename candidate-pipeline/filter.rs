use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
use crate::util;
use crate::SPAN_LEVEL;
use std::any::{type_name_of_val, Any};
use tracing::{field::Empty, Span};
use xai_stats_receiver::global_stats_receiver;

pub struct FilterResult<C> {
    pub kept: Vec<C>,
    pub removed: Vec<C>,
}

pub trait Filter<Q, C>: Any + Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency=Bucket0To50)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "filter", fields(
        name = self.name(),
        input_count = candidates.len(),
        kept_count = Empty,
        removed_count = Empty,
        filter_rate = Empty,
    ))]
    fn run(&self, query: &Q, candidates: Vec<C>, stage: PipelineStage) -> FilterResult<C> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let result = self.filter(query, candidates);
        stats.finish_filter(result.kept.len(), result.removed.len());
        let total = result.kept.len() + result.removed.len();
        let rate = if total > 0 {
            result.removed.len() as f64 / total as f64
        } else {
            0.0
        };
        let span = Span::current();
        span.record("kept_count", result.kept.len());
        span.record("removed_count", result.removed.len());
        span.record("filter_rate", format!("{:.3}", rate).as_str());
        self.stat(&result, stage);
        result
    }

    fn filter(&self, query: &Q, candidates: Vec<C>) -> FilterResult<C>;

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat(&self, result: &FilterResult<C>, stage: PipelineStage) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.run", self.name());
            receiver.incr(
                metric_name.as_str(),
                &stage.stat_labels(self.name(), "kept"),
                result.kept.len() as u64,
            );
            receiver.incr(
                metric_name.as_str(),
                &stage.stat_labels(self.name(), "removed"),
                result.removed.len() as u64,
            );
        }
    }
}
