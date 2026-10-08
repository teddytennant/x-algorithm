// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use xai_recsys_sid_retrieval_proto::{RetrieveRequest, SeedSidSource};

use crate::service::SidRetrievalServiceImpl;

#[derive(Deserialize)]
struct RetrieveBody {
    seed_post_ids: Vec<i64>,
    #[serde(default)]
    max_results: u32,
    #[serde(default)]
    max_per_seed: u32,
    #[serde(default)]
    min_prefix_depth: u32,
    #[serde(default)]
    max_prefix_depth: u32,
    #[serde(default)]
    caller: String,
}

#[derive(Serialize)]
struct CandidateJson {
    post_id: i64,
    author_id: i64,
    seed_post_id: i64,
    shared_prefix_depth: u32,
}

#[derive(Serialize)]
struct SeedJson {
    post_id: i64,
    codes: Vec<i32>,
    sid_source: String,
    num_candidates: u32,
}

#[derive(Serialize)]
struct RetrieveJson {
    candidates: Vec<CandidateJson>,
    seeds: Vec<SeedJson>,
    snapshot_timestamp_secs: i64,
    index_posts: u64,
}

fn sid_source_name(value: i32) -> String {
    SeedSidSource::try_from(value)
        .map(|s| s.as_str_name().to_string())
        .unwrap_or_else(|_| "SEED_SID_SOURCE_UNSPECIFIED".to_string())
}

async fn retrieve(
    State(service): State<Arc<SidRetrievalServiceImpl>>,
    Json(body): Json<RetrieveBody>,
) -> Result<Json<RetrieveJson>, (StatusCode, String)> {
    let response = service
        .handle(RetrieveRequest {
            seed_post_ids: body.seed_post_ids,
            max_results: body.max_results,
            max_per_seed: body.max_per_seed,
            min_prefix_depth: body.min_prefix_depth,
            max_prefix_depth: body.max_prefix_depth,
            caller: body.caller,
        })
        .await
        .map_err(|status| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                status.message().to_string(),
            )
        })?;
    Ok(Json(RetrieveJson {
        candidates: response
            .candidates
            .into_iter()
            .map(|c| CandidateJson {
                post_id: c.post_id,
                author_id: c.author_id,
                seed_post_id: c.seed_post_id,
                shared_prefix_depth: c.shared_prefix_depth,
            })
            .collect(),
        seeds: response
            .seeds
            .into_iter()
            .map(|s| SeedJson {
                post_id: s.post_id,
                codes: s.codes,
                sid_source: sid_source_name(s.sid_source),
                num_candidates: s.num_candidates,
            })
            .collect(),
        snapshot_timestamp_secs: response.snapshot_timestamp_secs,
        index_posts: response.index_posts,
    }))
}

pub fn router(service: Arc<SidRetrievalServiceImpl>) -> Router {
    Router::new()
        .route("/v1/retrieve", post(retrieve))
        .with_state(service)
}
