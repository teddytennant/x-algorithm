// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use lazy_static::lazy_static;
use prometheus::{
    register_histogram, register_int_counter_vec, register_int_gauge, register_int_gauge_vec,
    Histogram, IntCounterVec, IntGauge, IntGaugeVec,
};

use crate::codes::ErrorCode;
use crate::repo::PoolStatus;

lazy_static! {
    pub static ref LOOKUP_TOTAL: IntCounterVec = register_int_counter_vec!(
        "xai_abuse_ledger_lookup_total",
        "POST /v1/holds/lookup requests by result (ok|error) and error code",
        &["result", "code"]
    )
    .unwrap();
    pub static ref LOOKUP_SECONDS: Histogram = register_histogram!(
        "xai_abuse_ledger_lookup_seconds",
        "POST /v1/holds/lookup handler latency",
        vec![0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5]
    )
    .unwrap();
    pub static ref DB_SECONDS: Histogram = register_histogram!(
        "xai_abuse_ledger_db_seconds",
        "ledger.enforcement_holds query latency (session already checked out)",
        vec![0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5]
    )
    .unwrap();
    pub static ref HOLDS_RETURNED: Histogram = register_histogram!(
        "xai_abuse_ledger_holds_returned",
        "Active holds returned per successful lookup",
        vec![0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    )
    .unwrap();
    pub static ref POOL: IntGaugeVec = register_int_gauge_vec!(
        "xai_abuse_ledger_pool",
        "deadpool-postgres pool state (size|available|waiting|max_size)",
        &["state"]
    )
    .unwrap();
    pub static ref DB_UP: IntGauge = register_int_gauge!(
        "xai_abuse_ledger_db_up",
        "Ledger reachability (1 = last round trip succeeded)"
    )
    .unwrap();
    pub static ref CONTRACT_OK: IntGauge = register_int_gauge!(
        "xai_abuse_ledger_contract_ok",
        "Boot contract check against ledger.enforcement_holds passed (1) or not yet (0)"
    )
    .unwrap();
    pub static ref BUILD_INFO: IntGaugeVec = register_int_gauge_vec!(
        "xai_abuse_ledger_build_info",
        "Build / deployment identity (always 1)",
        &["env", "commit"]
    )
    .unwrap();
}

pub fn init(env: &str, commit: &str) {
    for code in ErrorCode::ALL {
        LOOKUP_TOTAL
            .with_label_values(&["error", code.as_str()])
            .reset();
    }
    LOOKUP_TOTAL.with_label_values(&["ok", ""]).reset();
    let _ = &*LOOKUP_SECONDS;
    let _ = &*DB_SECONDS;
    let _ = &*HOLDS_RETURNED;
    set_pool(PoolStatus::default());
    DB_UP.set(0);
    CONTRACT_OK.set(0);
    BUILD_INFO.with_label_values(&[env, commit]).set(1);
}

pub fn set_pool(s: PoolStatus) {
    POOL.with_label_values(&["size"]).set(s.size as i64);
    POOL.with_label_values(&["available"])
        .set(s.available as i64);
    POOL.with_label_values(&["waiting"]).set(s.waiting as i64);
    POOL.with_label_values(&["max_size"]).set(s.max_size as i64);
}

pub fn render() -> String {
    use prometheus::{Encoder, TextEncoder};
    let encoder = TextEncoder::new();
    let mut buf = Vec::new();
    if let Err(e) = encoder.encode(&prometheus::gather(), &mut buf) {
        tracing::error!("failed to encode metrics: {e}");
    }
    String::from_utf8(buf).unwrap_or_default()
}
