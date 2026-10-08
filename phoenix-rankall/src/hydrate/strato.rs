use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use governor::{DefaultDirectRateLimiter, Quota, RateLimiter};
use log::warn;
use serde::Serialize;
use serde_json::Value;
#[allow(deprecated)]
use xai_strato::Strato;

use super::metrics::{HydrateMetrics, OUTCOME_ERROR, OUTCOME_OK, OUTCOME_RETRY};

const MAX_ATTEMPTS: u32 = 3;
const BACKOFF_BASE: Duration = Duration::from_millis(200);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const IDLE_TIMEOUT: Duration = Duration::from_secs(50);
const BURST_FRACTION: u32 = 20;
const ERROR_LOG_EVERY: u64 = 1000;

pub type CallLimiter = Arc<DefaultDirectRateLimiter>;

pub fn call_limiter(calls_per_sec: u32) -> Result<CallLimiter> {
    let rate = NonZeroU32::new(calls_per_sec).context("Strato calls/s cap must be > 0")?;
    let burst = NonZeroU32::new(calls_per_sec / BURST_FRACTION).unwrap_or(NonZeroU32::MIN);
    Ok(Arc::new(RateLimiter::direct(
        Quota::per_second(rate).allow_burst(burst),
    )))
}

pub fn http_client(max_idle_connections: usize) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .http1_only()
        .pool_max_idle_per_host(max_idle_connections)
        .pool_idle_timeout(IDLE_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("failed to build Strato HTTP client")
}

pub fn json_i64(value: &Value) -> i64 {
    match value {
        Value::String(s) => s.parse().unwrap_or(0),
        Value::Number(n) => n.as_i64().unwrap_or(0),
        Value::Bool(b) => i64::from(*b),
        _ => 0,
    }
}

#[derive(Clone)]
#[allow(deprecated)]
pub struct StratoColumn {
    strato: Strato,
    limiter: CallLimiter,
    metrics: HydrateMetrics,
    errors: Arc<AtomicU64>,
}

impl StratoColumn {
    #[allow(deprecated)]
    pub fn new(
        column: &str,
        endpoint: &str,
        client: reqwest::Client,
        limiter: CallLimiter,
        metrics: HydrateMetrics,
    ) -> Self {
        Self {
            strato: Strato::with_client(column, endpoint, client),
            limiter,
            metrics,
            errors: Arc::default(),
        }
    }

    #[allow(deprecated)]
    pub fn name(&self) -> &str {
        &self.strato.column
    }

    pub async fn execute<A: Serialize + Sync>(&self, arg: &A) -> Result<Value> {
        self.call(|| self.strato.execute(arg)).await
    }

    pub async fn fetch<K: Serialize + Sync, V: Serialize + Sync>(
        &self,
        key: &K,
        view: &V,
    ) -> Result<Value> {
        self.call(|| self.strato.fetch(key, Some(view))).await
    }

    async fn call<F, Fut>(&self, op: F) -> Result<Value>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Value>>,
    {
        let mut attempt = 1;
        loop {
            self.limiter.until_ready().await;
            let start = Instant::now();
            match op().await {
                Ok(value) => {
                    self.count(OUTCOME_OK);
                    self.metrics
                        .strato_latency_seconds
                        .with_label_values(&[self.name()])
                        .observe(start.elapsed().as_secs_f64());
                    return Ok(value);
                }
                Err(_) if attempt < MAX_ATTEMPTS => {
                    self.count(OUTCOME_RETRY);
                    tokio::time::sleep(BACKOFF_BASE * 2u32.pow(attempt - 1)).await;
                    attempt += 1;
                }
                Err(e) => {
                    self.count(OUTCOME_ERROR);
                    if self
                        .errors
                        .fetch_add(1, Ordering::Relaxed)
                        .is_multiple_of(ERROR_LOG_EVERY)
                    {
                        warn!(
                            "strato {} failed after {attempt} attempts: {e:#}",
                            self.name()
                        );
                    }
                    return Err(e);
                }
            }
        }
    }

    fn count(&self, outcome: &str) {
        self.metrics
            .strato_requests_total
            .with_label_values(&[self.name(), outcome])
            .inc();
    }
}
