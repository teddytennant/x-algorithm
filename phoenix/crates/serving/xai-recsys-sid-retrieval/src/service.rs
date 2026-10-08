// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use arc_swap::ArcSwapOption;
use tonic::{Request, Response, Status};
use xai_recsys_sid_retrieval_proto::sid_retrieval_service_server::SidRetrievalService;
use xai_recsys_sid_retrieval_proto::{
    Candidate as CandidateProto, RetrieveRequest, RetrieveResponse, Seed, SeedSidSource,
};

use crate::index::{RetrieveParams, SID_LEVELS, SeedKey, SidIndex, codes_from_key, seed_key};
use crate::metrics::{CANDIDATES_PER_REQUEST, REQUEST_LATENCY, REQUESTS, SEEDS};
use crate::seed_sids::SeedSidLookup;

const MAX_RESULTS_CAP: usize = 5_000;
const MAX_PER_SEED_CAP: usize = 2_000;
const MAX_CALLER_LEN: usize = 64;

pub struct SidRetrievalServiceImpl {
    index: Arc<ArcSwapOption<SidIndex>>,
    seed_lookup: Arc<SeedSidLookup>,
    defaults: RetrieveParams,
    max_seeds: usize,
}

struct ResolvedSeed {
    post_id: i64,
    codes: Vec<i32>,
    key: Option<SeedKey>,
    source: SeedSidSource,
}

impl SidRetrievalServiceImpl {
    pub fn new(
        index: Arc<ArcSwapOption<SidIndex>>,
        seed_lookup: Arc<SeedSidLookup>,
        defaults: RetrieveParams,
        max_seeds: usize,
    ) -> Self {
        Self {
            index,
            seed_lookup,
            defaults,
            max_seeds,
        }
    }

    fn params(&self, request: &RetrieveRequest) -> RetrieveParams {
        let pick = |value: u32, default: usize| {
            if value == 0 { default } else { value as usize }
        };
        let min_prefix_depth =
            pick(request.min_prefix_depth, self.defaults.min_prefix_depth).clamp(1, SID_LEVELS);
        let max_prefix_depth = pick(request.max_prefix_depth, self.defaults.max_prefix_depth)
            .clamp(min_prefix_depth, SID_LEVELS);
        RetrieveParams {
            max_results: pick(request.max_results, self.defaults.max_results).min(MAX_RESULTS_CAP),
            max_per_seed: pick(request.max_per_seed, self.defaults.max_per_seed)
                .min(MAX_PER_SEED_CAP),
            min_prefix_depth,
            max_prefix_depth,
        }
    }

    async fn resolve_seeds(&self, index: &SidIndex, seed_post_ids: &[i64]) -> Vec<ResolvedSeed> {
        let mut seen = HashSet::new();
        let distinct: Vec<i64> = seed_post_ids
            .iter()
            .copied()
            .filter(|&id| id > 0 && seen.insert(id))
            .take(self.max_seeds)
            .collect();
        let missing: Vec<i64> = distinct
            .iter()
            .copied()
            .filter(|&id| index.lookup(id).is_none())
            .collect();
        let looked_up = self.seed_lookup.lookup(&missing).await;
        distinct
            .into_iter()
            .map(|post_id| {
                if let Some(key) = index.lookup(post_id) {
                    return ResolvedSeed {
                        post_id,
                        codes: codes_from_key(key),
                        key: Some(SeedKey {
                            key,
                            valid_depth: SID_LEVELS,
                        }),
                        source: SeedSidSource::Index,
                    };
                }
                match looked_up.get(&post_id) {
                    Some(codes) => ResolvedSeed {
                        post_id,
                        codes: codes.clone(),
                        key: seed_key(codes),
                        source: SeedSidSource::SidService,
                    },
                    None => ResolvedSeed {
                        post_id,
                        codes: vec![],
                        key: None,
                        source: SeedSidSource::Missing,
                    },
                }
            })
            .collect()
    }
}

fn caller_label(caller: &str) -> String {
    let label: String = caller
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(MAX_CALLER_LEN)
        .collect();
    if label.is_empty() {
        "unknown".to_string()
    } else {
        label
    }
}

fn source_label(source: SeedSidSource) -> &'static str {
    match source {
        SeedSidSource::Index => "index",
        SeedSidSource::SidService => "sid_service",
        SeedSidSource::Missing | SeedSidSource::Unspecified => "missing",
    }
}

impl SidRetrievalServiceImpl {
    pub async fn handle(&self, request: RetrieveRequest) -> Result<RetrieveResponse, Status> {
        let started = Instant::now();
        let caller = caller_label(&request.caller);
        let Some(index) = self.index.load_full() else {
            REQUESTS
                .with_label_values(&[caller.as_str(), "unavailable"])
                .inc();
            return Err(Status::unavailable("SID index not loaded yet"));
        };

        let params = self.params(&request);
        let seeds = self.resolve_seeds(&index, &request.seed_post_ids).await;
        let keyed: Vec<(i64, SeedKey)> = seeds
            .iter()
            .filter_map(|s| s.key.map(|key| (s.post_id, key)))
            .collect();
        let (candidates, counts) = index.retrieve(&keyed, &params);

        let mut count_by_seed = counts.into_iter();
        let seeds: Vec<Seed> = seeds
            .into_iter()
            .map(|s| {
                SEEDS.with_label_values(&[source_label(s.source)]).inc();
                let num_candidates = if s.key.is_some() {
                    count_by_seed.next().unwrap_or(0) as u32
                } else {
                    0
                };
                Seed {
                    post_id: s.post_id,
                    codes: s.codes,
                    sid_source: s.source as i32,
                    num_candidates,
                }
            })
            .collect();

        let result = if candidates.is_empty() { "empty" } else { "ok" };
        REQUESTS.with_label_values(&[caller.as_str(), result]).inc();
        CANDIDATES_PER_REQUEST
            .with_label_values(&[caller.as_str()])
            .observe(candidates.len() as f64);
        REQUEST_LATENCY
            .with_label_values(&[caller.as_str()])
            .observe(started.elapsed().as_secs_f64());

        Ok(RetrieveResponse {
            candidates: candidates
                .into_iter()
                .map(|c| CandidateProto {
                    post_id: c.post_id,
                    author_id: c.author_id,
                    seed_post_id: c.seed_post_id,
                    shared_prefix_depth: c.shared_prefix_depth,
                })
                .collect(),
            seeds,
            snapshot_timestamp_secs: index.snapshot_timestamp_secs(),
            index_posts: index.len() as u64,
        })
    }
}

#[tonic::async_trait]
impl SidRetrievalService for SidRetrievalServiceImpl {
    async fn retrieve(
        &self,
        request: Request<RetrieveRequest>,
    ) -> Result<Response<RetrieveResponse>, Status> {
        self.handle(request.into_inner()).await.map(Response::new)
    }
}
