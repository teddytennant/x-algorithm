use prometheus::{CounterVec, GaugeVec, HistogramOpts, HistogramVec, Opts, Registry};

pub const OUTCOME_OK: &str = "ok";
pub const OUTCOME_EMPTY: &str = "empty";
pub const OUTCOME_ERROR: &str = "error";
pub const OUTCOME_RETRY: &str = "retry";

pub const DECISION_ENQUEUED: &str = "enqueued";
pub const DECISION_RECENTLY_FETCHED: &str = "recently_fetched";
pub const DECISION_ALREADY_QUEUED: &str = "already_queued";
pub const DECISION_QUEUE_FULL: &str = "queue_full";

pub const QUEUE_IN_FLIGHT: &str = "in_flight";

#[derive(Clone)]
pub struct HydrateMetrics {
    pub fetch_total: CounterVec,
    pub sightings_total: CounterVec,
    pub queue_size: GaugeVec,
    pub coverage: GaugeVec,
    pub unfetched_rows: GaugeVec,
    pub strato_requests_total: CounterVec,
    pub strato_latency_seconds: HistogramVec,
}

impl HydrateMetrics {
    pub fn new(registry: &Registry) -> anyhow::Result<Self> {
        let fetch_total = CounterVec::new(
            Opts::new(
                "phoenix_post_feature_fetch_total",
                "Post feature fetches by trigger and outcome",
            ),
            &["reason", "outcome"],
        )?;
        registry.register(Box::new(fetch_total.clone()))?;

        let sightings_total = CounterVec::new(
            Opts::new(
                "phoenix_post_feature_sightings_total",
                "Hydration requests by queueing decision",
            ),
            &["queue", "decision"],
        )?;
        registry.register(Box::new(sightings_total.clone()))?;

        let queue_size = GaugeVec::new(
            Opts::new(
                "phoenix_post_feature_queue_size",
                "Post ids waiting for a fetch",
            ),
            &["queue"],
        )?;
        registry.register(Box::new(queue_size.clone()))?;

        let coverage = GaugeVec::new(
            Opts::new(
                "phoenix_post_feature_coverage",
                "Fraction of store rows fetched at the current features version",
            ),
            &["window"],
        )?;
        registry.register(Box::new(coverage.clone()))?;

        let unfetched_rows = GaugeVec::new(
            Opts::new(
                "phoenix_post_feature_unfetched_rows",
                "Store rows not fetched at the current features version",
            ),
            &["window"],
        )?;
        registry.register(Box::new(unfetched_rows.clone()))?;

        let strato_requests_total = CounterVec::new(
            Opts::new(
                "strato_client_requests_total",
                "Strato HTTP attempts by column and outcome",
            ),
            &["column", "outcome"],
        )?;
        registry.register(Box::new(strato_requests_total.clone()))?;

        let strato_latency_seconds = HistogramVec::new(
            HistogramOpts::new(
                "strato_client_request_latency_seconds",
                "Latency of successful Strato HTTP attempts",
            )
            .buckets(vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
            &["column"],
        )?;
        registry.register(Box::new(strato_latency_seconds.clone()))?;

        Ok(Self {
            fetch_total,
            sightings_total,
            queue_size,
            coverage,
            unfetched_rows,
            strato_requests_total,
            strato_latency_seconds,
        })
    }

    #[cfg(test)]
    pub fn for_tests() -> Self {
        Self::new(&Registry::new()).expect("metrics for tests")
    }
}
