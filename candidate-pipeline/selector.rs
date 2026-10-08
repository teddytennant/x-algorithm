use crate::candidate_pipeline::{PipelineCandidate, PipelineQuery, PipelineStage};
use crate::pipeline_summary::ComponentStats;
use crate::util;
use crate::SPAN_LEVEL;
use std::any::type_name_of_val;
use tracing::{field::Empty, Span};
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

pub struct SelectResult<C> {
    pub selected: Vec<C>,
    pub non_selected: Vec<C>,
}

impl<C> SelectResult<C> {
    pub fn len(&self) -> usize {
        self.selected.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty() && self.non_selected.is_empty()
    }
}

pub trait Selector<Q, C>: Send + Sync
where
    Q: PipelineQuery,
    C: PipelineCandidate,
{
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    #[xai_stats_macro::receive_stats(latency=Bucket0To50)]
    #[tracing::instrument(level = SPAN_LEVEL, skip_all, name = "selector", fields(
        name = self.name(),
        input_count = candidates.len(),
        selected_count = Empty,
        non_selected_count = Empty,
    ))]
    fn run(&self, query: &Q, candidates: Vec<C>, stage: PipelineStage) -> SelectResult<C> {
        let stats = ComponentStats::begin(stage, self.name(), type_name_of_val(self));
        let input_count = candidates.len();
        let result = self.select(query, candidates);
        stats.finish_with_input(input_count);
        let span = Span::current();
        span.record("selected_count", result.selected.len());
        span.record("non_selected_count", result.non_selected.len());
        self.stat(&result, stage);
        result
    }

    fn select(&self, _query: &Q, candidates: Vec<C>) -> SelectResult<C> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() {
            let non_selected = sorted.split_off(limit.min(sorted.len()));
            SelectResult {
                selected: sorted,
                non_selected,
            }
        } else {
            SelectResult {
                selected: sorted,
                non_selected: vec![],
            }
        }
    }

    fn score(&self, candidate: &C) -> f64;

    fn sort(&self, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| {
            self.score(b)
                .partial_cmp(&self.score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    fn size(&self) -> Option<usize> {
        None
    }

    fn name(&self) -> &'static str {
        util::short_type_name(type_name_of_val(self))
    }

    fn stat(&self, result: &SelectResult<C>, stage: PipelineStage) {
        if let Some(receiver) = global_stats_receiver() {
            let metric_name = format!("{}.run", self.name());
            let result_size = result.len() as f64;
            receiver.observe(
                metric_name.as_str(),
                &stage.stat_labels(self.name(), "result_size"),
                result_size,
                HistogramBuckets::Bucket0To50,
            );
            if result_size == 0.0 {
                receiver.incr(
                    metric_name.as_str(),
                    &stage.stat_labels(self.name(), "result_empty"),
                    1u64,
                );
            }
        }
    }
}
