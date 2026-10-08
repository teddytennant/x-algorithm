#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::sync::atomic::Ordering;
use std::{path::PathBuf, time::Duration};

use axum::{http::StatusCode, routing::get, Router};
use clap::Parser;
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer};
use tracing::{info, Level};
use xai_service_runner::ServerBuilder;

use xai_abuse_enforcement_service::auth::ApiKeys;
use xai_abuse_enforcement_service::config::Config;
use xai_abuse_enforcement_service::{
    build_api_router, build_docs_router, build_state, init_runtime_globals, shutdown_kafka,
    spawn_config_refresh, start_kafka_consumers,
};

fn admin_ui_public_dir() -> PathBuf {
    if let Ok(dir) =
        std::env::var("DIOXUS_PUBLIC_PATH").or_else(|_| std::env::var("ADMIN_UI_PUBLIC_DIR"))
    {
        return PathBuf::from(dir);
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("public")))
        .unwrap_or_else(|| PathBuf::from("public"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_runtime_globals();

    let cfg = Config::parse();
    let (state, cfg) = build_state(cfg).await?;

    let api_keys = ApiKeys::from_config(&cfg)?;
    if api_keys.is_empty() {
        tracing::warn!(
            "API_KEYS_JSON is empty — every request to the protected admin \
             endpoints (/api/action, /api/config*, /api/allowlist*) will return \
             401. The admin UI calls those routes directly from the browser, so \
             every UI write will also be rejected with 401 until a key map is set."
        );
    } else {
        info!(
            key_ids = ?api_keys.configured_key_ids(),
            "api-key auth configured for protected admin endpoints"
        );
    }

    let api_router = build_api_router(state.clone(), api_keys);
    let docs_router = build_docs_router();

    let public_dir = admin_ui_public_dir();
    info!(public_dir = %public_dir.display(), "serving admin UI static bundle");
    let index_html = public_dir.join("index.html");
    let admin_ui = ServeDir::new(&public_dir).fallback(ServeFile::new(index_html));

    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
        .on_response(DefaultOnResponse::new().level(Level::INFO));

    let kafka_ready = state.kafka_ready.clone();

    let app: Router = Router::new()
        .nest("/api", api_router)
        .merge(docs_router)
        .route(
            "/kafka_ready",
            get(move || {
                let kafka_ready = kafka_ready.clone();
                async move {
                    if kafka_ready.load(Ordering::Relaxed) {
                        StatusCode::OK
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE
                    }
                }
            }),
        )
        .fallback_service(admin_ui)
        .layer(trace_layer);

    let kafka_consumers = start_kafka_consumers(state.clone(), &cfg).await?;

    let server = ServerBuilder::new(cfg.port)
        .merge(app)
        .drain_period(Duration::from_secs(cfg.drain_period_secs))
        .on_ready(|| async {
            info!("Server ready and accepting requests (api=/api, docs=/api/docs, admin UI=/)");
        });
    spawn_config_refresh(&state, server.shutdown_signal());
    let served = server.run().await;

    shutdown_kafka(&state, kafka_consumers).await;
    served?;
    info!("Server shutdown complete");
    Ok(())
}
