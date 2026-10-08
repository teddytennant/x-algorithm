// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::Context;
use clap::Parser;
use tracing::{error, info};

use xai_abuse_ledger_service::api::{self, AppState};
use xai_abuse_ledger_service::config::Config;
use xai_abuse_ledger_service::pg::PgHoldRepo;
use xai_abuse_ledger_service::repo::HoldRepo;
use xai_abuse_ledger_service::{boot, metrics};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();

    let config = Config::parse();
    config.validate()?;
    info!(?config, "starting xai-abuse-ledger-service");
    metrics::init(&config.ledger_env, &config.commit);

    let repo = PgHoldRepo::new(
        config.database_url.expose(),
        &config.ledger_env,
        config.statement_timeout(),
        config.pool_max,
    )
    .context("DATABASE_URL is unusable")?;
    let repo: Arc<dyn HoldRepo> = Arc::new(repo);
    info!(endpoint = %repo.endpoint(), "ledger repo built (nothing connected yet)");

    let ready = Arc::new(AtomicBool::new(false));
    let contract = tokio::spawn(boot::contract_check_until_ok(repo.clone(), ready.clone()));
    let db_up = tokio::spawn(boot::db_up_loop(repo.clone()));

    let state = Arc::new(AppState::new(repo, ready, config.statement_timeout()));
    let app = api::router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    info!(%addr, path = api::LOOKUP_PATH, "listening");

    let result = axum::serve(listener, app)
        .with_graceful_shutdown(wait_for_shutdown())
        .await;
    contract.abort();
    db_up.abort();
    if let Err(e) = &result {
        error!("server error: {e}");
    }
    result.context("server exited with an error")
}

async fn wait_for_shutdown() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                error!("failed to install SIGTERM handler: {e}");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
    info!("shutdown signal received; draining");
}
