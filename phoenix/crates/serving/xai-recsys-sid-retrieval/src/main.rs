// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
mod http_api;
mod index;
mod loader;
mod metrics;
mod seed_sids;
mod service;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use clap::Parser;
use tokio::sync::watch;
use tonic::service::Routes;
use tracing::info;
use xai_recsys_server::{CancellationToken, GrpcConfig, HttpServer};
use xai_recsys_sid_retrieval_proto::sid_retrieval_service_server::SidRetrievalServiceServer;

use crate::index::RetrieveParams;
use crate::seed_sids::SeedSidLookup;
use crate::service::SidRetrievalServiceImpl;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 9090)]
    grpc_port: u16,
    #[arg(long, default_value_t = 8080)]
    http_port: u16,
    #[arg(long, default_value = "post_sid_snapshot.parquet")]
    snapshot_path: PathBuf,
    #[arg(long, default_value_t = 30)]
    snapshot_poll_secs: u64,
    #[arg(long, default_value = "http://localhost:50051")]
    sid_endpoint: String,
    #[arg(long, default_value_t = 200)]
    sid_timeout_ms: u64,
    #[arg(long, default_value_t = 2_000_000)]
    sid_cache_size: u64,
    #[arg(long, default_value_t = 200)]
    max_seeds: usize,
    #[arg(long, default_value_t = 800)]
    default_max_results: usize,
    #[arg(long, default_value_t = 100)]
    default_max_per_seed: usize,
    #[arg(long, default_value_t = 3)]
    default_min_prefix_depth: usize,
    #[arg(long, default_value_t = 6)]
    default_max_prefix_depth: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    info!(?args, "starting xai-recsys-sid-retrieval");

    let index = Arc::new(ArcSwapOption::empty());
    let (ready_tx, mut ready_rx) = watch::channel(false);
    tokio::spawn(loader::watch_snapshots(
        args.snapshot_path.clone(),
        Duration::from_secs(args.snapshot_poll_secs),
        Arc::clone(&index),
        ready_tx,
    ));

    let seed_lookup = Arc::new(SeedSidLookup::new(
        &args.sid_endpoint,
        Duration::from_millis(args.sid_timeout_ms),
        args.sid_cache_size,
    )?);
    let warm_up_lookup = Arc::clone(&seed_lookup);
    tokio::spawn(async move { warm_up_lookup.warm_up().await });
    let service = SidRetrievalServiceImpl::new(
        index,
        seed_lookup,
        RetrieveParams {
            max_results: args.default_max_results,
            max_per_seed: args.default_max_per_seed,
            min_prefix_depth: args.default_min_prefix_depth,
            max_prefix_depth: args.default_max_prefix_depth,
        },
        args.max_seeds,
    );

    let service = Arc::new(service);
    let routes = Routes::new(SidRetrievalServiceServer::from_arc(Arc::clone(&service)));
    let mut http_server = HttpServer::builder(
        args.http_port,
        http_api::router(service),
        CancellationToken::new(),
        Duration::from_secs(10),
    )
    .with_grpc(GrpcConfig::new(args.grpc_port, routes))
    .build()
    .await?;
    info!(
        grpc_port = args.grpc_port,
        http_port = args.http_port,
        "servers started"
    );

    if ready_rx.wait_for(|ready| *ready).await.is_ok() {
        http_server.set_readiness(true);
    }
    http_server.wait_for_termination().await;
    Ok(())
}
