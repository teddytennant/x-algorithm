// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use lazy_static::lazy_static;
use prometheus::{
    Gauge, HistogramVec, IntCounterVec, IntGauge, register_gauge, register_histogram_vec,
    register_int_counter_vec, register_int_gauge,
};

use crate::index::SidIndex;

lazy_static! {
    pub static ref SNAPSHOT_TIMESTAMP: IntGauge = register_int_gauge!(
        "sid_retrieval_snapshot_timestamp_seconds",
        "Timestamp of the loaded SID corpus snapshot"
    )
    .unwrap();
    pub static ref INDEX_POSTS: IntGauge =
        register_int_gauge!("sid_retrieval_index_posts", "Posts in the loaded SID index").unwrap();
    pub static ref INDEX_SKIPPED_POSTS: IntGauge = register_int_gauge!(
        "sid_retrieval_index_skipped_posts",
        "Snapshot rows skipped for missing or invalid SID codes"
    )
    .unwrap();
    pub static ref INDEX_LOAD_SECONDS: Gauge = register_gauge!(
        "sid_retrieval_index_load_seconds",
        "Duration of the last SID index load"
    )
    .unwrap();
    pub static ref INDEX_LOADS: IntCounterVec = register_int_counter_vec!(
        "sid_retrieval_index_loads_total",
        "SID index load attempts",
        &["result"]
    )
    .unwrap();
    pub static ref REQUESTS: IntCounterVec = register_int_counter_vec!(
        "sid_retrieval_requests_total",
        "Retrieve requests",
        &["caller", "result"]
    )
    .unwrap();
    pub static ref REQUEST_LATENCY: HistogramVec = register_histogram_vec!(
        "sid_retrieval_request_latency_seconds",
        "Retrieve latency",
        &["caller"],
        vec![
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0
        ]
    )
    .unwrap();
    pub static ref CANDIDATES_PER_REQUEST: HistogramVec = register_histogram_vec!(
        "sid_retrieval_candidates_per_request",
        "Candidates returned per request",
        &["caller"],
        vec![
            0.0, 1.0, 10.0, 50.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0
        ]
    )
    .unwrap();
    pub static ref SEEDS: IntCounterVec = register_int_counter_vec!(
        "sid_retrieval_seeds_total",
        "Seeds by SID source",
        &["source"]
    )
    .unwrap();
    pub static ref SID_SERVICE_LOOKUPS: IntCounterVec = register_int_counter_vec!(
        "sid_retrieval_sid_service_lookups_total",
        "Seed SID lookups against the SID service",
        &["result"]
    )
    .unwrap();
}

pub fn record_index_load(index: &SidIndex, load_secs: f64) {
    INDEX_LOADS.with_label_values(&["ok"]).inc();
    SNAPSHOT_TIMESTAMP.set(index.snapshot_timestamp_secs());
    INDEX_POSTS.set(index.len() as i64);
    INDEX_SKIPPED_POSTS.set(index.skipped_posts() as i64);
    INDEX_LOAD_SECONDS.set(load_secs);
}
