use std::cell::Cell;
use std::time::Instant;

use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

use super::types::{FailureKind, FallbackReason, LabelSource};

const REQUESTS: &str = "safety_labels_lookup_requests";
const LATENCY_MS: &str = "safety_labels_lookup_latency_ms_vm";
const LOOKUP_TWEET_IDS: &str = "safety_labels_lookup_tweet_ids";
const FAILURES: &str = "safety_labels_lookup_failures";
const SOURCE_REQUESTS: &str = "safety_labels_source_requests";
const SOURCE_LATENCY_MS: &str = "safety_labels_source_latency_ms";
const CACHE_KEYS: &str = "safety_labels_cache_keys";
const MANHATTAN_KEYS: &str = "safety_labels_manhattan_keys";
const CACHE_FALLBACK_KEYS: &str = "safety_labels_cache_fallback_keys";
const BATCH_SIZE: &str = "safety_labels_lookup_batch_size";
const CACHE_WARM_KEYS: &str = "safety_labels_cache_warm_keys";

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum RequestOutcome {
    Started,
    Cancelled,
    Success,
    Failure,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum SourceOutcome {
    Success,
    Failure,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum CacheTier {
    Local,
    Twemcache,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum CacheResult {
    Hit,
    NotFound,
    Miss,
    Expired,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum ManhattanResult {
    Success,
    Failed,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum BatchStage {
    Request,
    LocalMiss,
    ManhattanFallback,
}

pub(crate) struct RequestMetricsGuard {
    start: Instant,
    outcome: Cell<RequestOutcome>,
}

impl RequestMetricsGuard {
    pub(crate) fn new() -> Self {
        incr(REQUESTS, &[("outcome", RequestOutcome::Started.into())], 1);
        Self {
            start: Instant::now(),
            outcome: Cell::new(RequestOutcome::Cancelled),
        }
    }

    pub(crate) fn mark_success(&self) {
        self.outcome.set(RequestOutcome::Success);
    }

    pub(crate) fn mark_failure(&self) {
        self.outcome.set(RequestOutcome::Failure);
    }
}

impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        let outcome = <&str>::from(self.outcome.get());
        incr(REQUESTS, &[("outcome", outcome)], 1);
        observe_vm(LATENCY_MS, &[], self.start.elapsed().as_secs_f64() * 1000.0);
    }
}

pub(crate) fn record_lookup_tweet_ids(success: usize, failed: usize) {
    incr_nonzero(LOOKUP_TWEET_IDS, &[("result", "success")], success as u64);
    incr_nonzero(LOOKUP_TWEET_IDS, &[("result", "failed")], failed as u64);
}

pub(crate) fn record_lookup_failures(kind: FailureKind, count: usize) {
    incr_nonzero(FAILURES, &[("kind", kind.into())], count as u64);
}

pub(crate) fn record_source_request(source: LabelSource, outcome: SourceOutcome, elapsed_ms: f64) {
    incr(
        SOURCE_REQUESTS,
        &[("source", source.into()), ("outcome", outcome.into())],
        1,
    );
    observe_vm(SOURCE_LATENCY_MS, &[("source", source.into())], elapsed_ms);
}

pub(crate) fn record_cache_keys(tier: CacheTier, result: CacheResult, count: usize) {
    incr_nonzero(
        CACHE_KEYS,
        &[("tier", tier.into()), ("result", result.into())],
        count as u64,
    );
}

pub(crate) fn record_manhattan_keys(result: ManhattanResult, count: usize) {
    incr_nonzero(MANHATTAN_KEYS, &[("result", result.into())], count as u64);
}

pub(crate) fn record_cache_fallback_keys(
    source: LabelSource,
    reason: FallbackReason,
    count: usize,
) {
    incr_nonzero(
        CACHE_FALLBACK_KEYS,
        &[("source", source.into()), ("reason", reason.into())],
        count as u64,
    );
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum WarmKeyResult {
    EligibleMiss,
    Enqueued,
    DroppedChannelFull,
    FetchIssued,
    FetchFailed,
}

pub(crate) fn record_cache_warm_keys(result: WarmKeyResult, count: usize) {
    incr_nonzero(CACHE_WARM_KEYS, &[("result", result.into())], count as u64);
}

pub(crate) fn record_batch_size(stage: BatchStage, size: usize) {
    observe(
        BATCH_SIZE,
        &[("stage", stage.into())],
        size as f64,
        HistogramBuckets::Bucket50To500,
    );
}

fn incr_nonzero(metric: &str, labels: &[(&str, &str)], count: u64) {
    if count > 0 {
        incr(metric, labels, count);
    }
}

fn incr(metric: &str, labels: &[(&str, &str)], count: u64) {
    if let Some(sr) = global_stats_receiver() {
        sr.incr(metric, labels, count);
    }
}

fn observe(metric: &str, labels: &[(&str, &str)], value: f64, buckets: HistogramBuckets) {
    if let Some(sr) = global_stats_receiver() {
        sr.observe(metric, labels, value, buckets);
    }
}

fn observe_vm(metric: &str, labels: &[(&str, &str)], value: f64) {
    if let Some(sr) = global_stats_receiver() {
        sr.observe_vm(metric, labels, value);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use xai_visibility_filtering::tweet_safety_label::{
        BATCH_LATENCY_VM_METRIC, LATENCY_VM_METRIC,
    };

    use super::LATENCY_MS;

    #[test]
    fn dashboard_generator_pins_the_get_safety_labels_vm_latency_metrics() {
        let cargo = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/dashboard.py");
        let ws = "crates/x-product/xai-visibility-filtering-service/scripts/dashboard.py";
        let path = if Path::new(cargo).exists() { cargo } else { ws };
        let dashboard = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        for pin in [
            format!("LOOKUP_LATENCY_VM_METRIC = \"{LATENCY_MS}\""),
            format!("VF_CLIENT_LATENCY_VM_METRIC = \"{LATENCY_VM_METRIC}\""),
            format!("VF_CLIENT_BATCH_LATENCY_VM_METRIC = \"{BATCH_LATENCY_VM_METRIC}\""),
        ] {
            assert!(dashboard.contains(&pin), "dashboard.py is missing `{pin}`");
        }
    }
}
