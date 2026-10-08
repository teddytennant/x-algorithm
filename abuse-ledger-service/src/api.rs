// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::codes::{ErrorBody, ErrorCode};
use crate::metrics;
use crate::pg::{PgError, QUERY_DEADLINE_SLACK};
use crate::repo::{Hold, HoldRepo};

pub const LOOKUP_PATH: &str = "/v1/holds/lookup";
pub const DEADLINE_HEADER: &str = "x-request-deadline-ms";
pub const MAX_USER_IDS: usize = 100;
pub const MAX_LABELS: usize = 32;
const ERROR_LOG_EVERY: u64 = 1_000;

pub struct AppState {
    pub repo: Arc<dyn HoldRepo>,
    pub ready: Arc<AtomicBool>,
    pub statement_timeout: Duration,
    errors_seen: AtomicU64,
}

impl AppState {
    pub fn new(
        repo: Arc<dyn HoldRepo>,
        ready: Arc<AtomicBool>,
        statement_timeout: Duration,
    ) -> Self {
        Self {
            repo,
            ready,
            statement_timeout,
            errors_seen: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LookupRequest {
    pub user_ids: Vec<i64>,
    #[serde(default)]
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LookupResponse {
    pub holds: Vec<Hold>,
    pub db_ms: u64,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(LOOKUP_PATH, post(lookup))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

pub fn validate(req: &LookupRequest) -> Result<(), ErrorCode> {
    if req.user_ids.is_empty() || req.user_ids.len() > MAX_USER_IDS {
        return Err(ErrorCode::BadRequest);
    }
    if req.labels.len() > MAX_LABELS || req.labels.iter().any(|l| l.trim().is_empty()) {
        return Err(ErrorCode::BadRequest);
    }
    Ok(())
}

pub fn budget(deadline: Option<Duration>, statement_timeout: Duration) -> Duration {
    match deadline {
        Some(d) if d < statement_timeout => d,
        _ => statement_timeout + QUERY_DEADLINE_SLACK,
    }
}

pub fn parse_deadline(headers: &HeaderMap) -> Result<Option<Duration>, ErrorCode> {
    let Some(raw) = headers.get(DEADLINE_HEADER) else {
        return Ok(None);
    };
    let ms: u64 = raw
        .to_str()
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|ms| *ms > 0)
        .ok_or(ErrorCode::BadRequest)?;
    Ok(Some(Duration::from_millis(ms)))
}

fn error_response(status: StatusCode, body: ErrorBody) -> Response {
    (status, Json(body)).into_response()
}

async fn lookup(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let started = Instant::now();
    let resp = lookup_inner(&state, &headers, &body).await;
    metrics::LOOKUP_SECONDS.observe(started.elapsed().as_secs_f64());
    resp
}

async fn lookup_inner(state: &AppState, headers: &HeaderMap, body: &Bytes) -> Response {
    let req: LookupRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(_) => return reject(ErrorCode::BadRequest),
    };
    if let Err(code) = validate(&req) {
        return reject(code);
    }
    let deadline = match parse_deadline(headers) {
        Ok(d) => d,
        Err(code) => return reject(code),
    };
    let budget = budget(deadline, state.statement_timeout);

    let db_started = Instant::now();
    let result = state.repo.lookup(&req.user_ids, &req.labels, budget).await;
    let db_ms = db_started.elapsed().as_millis() as u64;
    if let Some(pool) = state.repo.pool_status() {
        metrics::set_pool(pool);
    }
    match result {
        Ok(holds) => {
            metrics::LOOKUP_TOTAL.with_label_values(&["ok", ""]).inc();
            metrics::HOLDS_RETURNED.observe(holds.len() as f64);
            (StatusCode::OK, Json(LookupResponse { holds, db_ms })).into_response()
        }
        Err(e) => {
            let code = e.code();
            metrics::LOOKUP_TOTAL
                .with_label_values(&["error", code.as_str()])
                .inc();
            log_error(state, &e, &req, budget);
            error_response(
                e.status(),
                ErrorBody {
                    code,
                    retryable: e.retryable(),
                },
            )
        }
    }
}

fn reject(code: ErrorCode) -> Response {
    metrics::LOOKUP_TOTAL
        .with_label_values(&["error", code.as_str()])
        .inc();
    error_response(code.status(), ErrorBody::new(code))
}

fn log_error(state: &AppState, e: &PgError, req: &LookupRequest, budget: Duration) {
    let n = state.errors_seen.fetch_add(1, Ordering::Relaxed);
    if n == 0 || n.is_multiple_of(ERROR_LOG_EVERY) {
        warn!(
            code = e.code().as_str(),
            status = e.status().as_u16(),
            error = %e,
            user_ids = req.user_ids.len(),
            labels = req.labels.len(),
            budget_ms = budget.as_millis() as u64,
            seen = n + 1,
            "lookup failed (sampled: first, then every {ERROR_LOG_EVERY})"
        );
    }
}

async fn healthz() -> &'static str {
    "ok"
}

async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    if state.ready.load(Ordering::Acquire) {
        (StatusCode::OK, "ready").into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "not ready: boot contract check has not passed",
        )
            .into_response()
    }
}

async fn metrics_handler() -> String {
    metrics::render()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(ids: usize, labels: &[&str]) -> LookupRequest {
        LookupRequest {
            user_ids: (1..=ids as i64).collect(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn validation_bounds() {
        assert_eq!(validate(&req(1, &[])), Ok(()));
        assert_eq!(validate(&req(100, &["A"])), Ok(()));
        assert_eq!(validate(&req(0, &[])), Err(ErrorCode::BadRequest));
        assert_eq!(validate(&req(101, &[])), Err(ErrorCode::BadRequest));
        let many: Vec<String> = (0..32).map(|i| format!("L{i}")).collect();
        let many_ref: Vec<&str> = many.iter().map(String::as_str).collect();
        assert_eq!(validate(&req(1, &many_ref)), Ok(()));
        let too_many: Vec<String> = (0..33).map(|i| format!("L{i}")).collect();
        let too_many_ref: Vec<&str> = too_many.iter().map(String::as_str).collect();
        assert_eq!(validate(&req(1, &too_many_ref)), Err(ErrorCode::BadRequest));
        assert_eq!(validate(&req(1, &[""])), Err(ErrorCode::BadRequest));
        assert_eq!(validate(&req(1, &["  "])), Err(ErrorCode::BadRequest));
        assert_eq!(validate(&req(1, &["ok", " "])), Err(ErrorCode::BadRequest));
    }

    #[test]
    fn budget_clamps_to_the_shorter_of_deadline_and_statement_timeout() {
        let st = Duration::from_millis(150);
        assert_eq!(budget(None, st), st + QUERY_DEADLINE_SLACK);
        assert_eq!(
            budget(Some(Duration::from_millis(5_000)), st),
            st + QUERY_DEADLINE_SLACK
        );
        assert_eq!(budget(Some(st), st), st + QUERY_DEADLINE_SLACK);
        assert_eq!(
            budget(Some(Duration::from_millis(40)), st),
            Duration::from_millis(40)
        );
    }

    #[test]
    fn deadline_header_parses_or_rejects() {
        let mut h = HeaderMap::new();
        assert_eq!(parse_deadline(&h), Ok(None));
        h.insert(DEADLINE_HEADER, "200".parse().unwrap());
        assert_eq!(parse_deadline(&h), Ok(Some(Duration::from_millis(200))));
        h.insert(DEADLINE_HEADER, " 75 ".parse().unwrap());
        assert_eq!(parse_deadline(&h), Ok(Some(Duration::from_millis(75))));
        for bad in ["0", "-1", "abc", "1.5", ""] {
            h.insert(DEADLINE_HEADER, bad.parse().unwrap());
            assert_eq!(parse_deadline(&h), Err(ErrorCode::BadRequest), "{bad:?}");
        }
    }

    #[test]
    fn request_ignores_unknown_fields_and_defaults_labels() {
        let r: LookupRequest = serde_json::from_str(r#"{"user_ids":[1,2],"future":true}"#).unwrap();
        assert_eq!(r.user_ids, vec![1, 2]);
        assert!(r.labels.is_empty());
    }
}
