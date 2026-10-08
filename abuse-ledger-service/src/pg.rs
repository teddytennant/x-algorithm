// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use deadpool_postgres::{
    Manager, ManagerConfig, Pool, PoolError, RecyclingMethod, Runtime, TimeoutType,
};
use tracing::{info, warn};

use crate::codes::ErrorCode;
use crate::metrics;
use crate::repo::{Hold, HoldRepo, PoolStatus};
use crate::sql::{CONTRACT_SQL, PING_SQL, PROBE_SQL};

pub const POOL_WAIT_TIMEOUT: Duration = Duration::from_millis(1_000);
pub const POOL_CREATE_TIMEOUT: Duration = Duration::from_millis(1_500);
pub const POOL_RECYCLE_TIMEOUT: Duration = Duration::from_millis(500);
pub const QUERY_DEADLINE_SLACK: Duration = Duration::from_millis(100);

pub const SQLSTATE_QUERY_CANCELED: &str = "57014";
pub const SQLSTATE_AUTH: [&str; 2] = ["28P01", "28000"];

const APPLICATION_NAME_MAX_BYTES: usize = 63;
const APPLICATION_NAME_PREFIX: &str = "xai-abuse-ledger-service";

#[derive(Debug, Clone, thiserror::Error)]
pub enum PgError {
    #[error("bad DSN: {0}")]
    BadDsn(String),
    #[error("connect ({}): {message}", kind.as_str())]
    Connect {
        kind: ConnectFailure,
        message: String,
    },
    #[error("pool saturated: no free session within {0:?}")]
    Saturated(Duration),
    #[error("transport: {0}")]
    Transport(String),
    #[error("query [{sqlstate}]: {message}")]
    Query { sqlstate: String, message: String },
    #[error("row decode: {0}")]
    Decode(String),
    #[error("timed out after {0:?}")]
    Timeout(Duration),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectFailure {
    Refused,
    Auth,
    Tls,
    Timeout,
    Other,
}

impl ConnectFailure {
    pub fn as_str(self) -> &'static str {
        match self {
            ConnectFailure::Refused => "refused",
            ConnectFailure::Auth => "auth",
            ConnectFailure::Tls => "tls",
            ConnectFailure::Timeout => "timeout",
            ConnectFailure::Other => "other",
        }
    }
}

impl PgError {
    fn connect(kind: ConnectFailure, message: impl Into<String>) -> Self {
        Self::Connect {
            kind,
            message: message.into(),
        }
    }

    pub fn code(&self) -> ErrorCode {
        match self {
            PgError::BadDsn(_) => ErrorCode::Other,
            PgError::Connect { kind, .. } => match kind {
                ConnectFailure::Refused => ErrorCode::ConnectRefused,
                ConnectFailure::Auth => ErrorCode::AuthFailed,
                ConnectFailure::Tls => ErrorCode::Tls,
                ConnectFailure::Timeout => ErrorCode::Timeout,
                ConnectFailure::Other => ErrorCode::Other,
            },
            PgError::Saturated(_) => ErrorCode::PoolWait,
            PgError::Transport(_) => ErrorCode::Other,
            PgError::Query { sqlstate, .. } if sqlstate == SQLSTATE_QUERY_CANCELED => {
                ErrorCode::StatementTimeout
            }
            PgError::Query { sqlstate, .. } if sqlstate_is_unhealthy(sqlstate) => {
                ErrorCode::ServerUnhealthy
            }
            PgError::Query { .. } => ErrorCode::Other,
            PgError::Decode(_) => ErrorCode::Decode,
            PgError::Timeout(_) => ErrorCode::Timeout,
        }
    }

    pub fn status(&self) -> axum::http::StatusCode {
        match self {
            PgError::Transport(_) => axum::http::StatusCode::SERVICE_UNAVAILABLE,
            other => other.code().status(),
        }
    }

    pub fn retryable(&self) -> bool {
        match self {
            PgError::Transport(_) => true,
            other => other.code().retryable(),
        }
    }
}

pub fn sqlstate_is_unhealthy(sqlstate: &str) -> bool {
    sqlstate == SQLSTATE_QUERY_CANCELED || sqlstate.starts_with("57P") || sqlstate.starts_with("53")
}

pub fn render_error_chain(e: &dyn std::error::Error) -> String {
    let mut out = e.to_string();
    let mut source = e.source();
    while let Some(err) = source {
        out.push_str(": ");
        out.push_str(&err.to_string());
        source = err.source();
    }
    out
}

pub fn classify_pg_error(e: &tokio_postgres::Error) -> PgError {
    match e.as_db_error() {
        Some(db) => PgError::Query {
            sqlstate: db.code().code().to_owned(),
            message: db.message().to_owned(),
        },
        None => PgError::Transport(render_error_chain(e)),
    }
}

pub fn classify_connect_error(e: &tokio_postgres::Error) -> PgError {
    let message = render_error_chain(e);
    if e.as_db_error()
        .is_some_and(|db| SQLSTATE_AUTH.contains(&db.code().code()))
    {
        return PgError::connect(ConnectFailure::Auth, message);
    }
    if message.starts_with("error performing TLS handshake") {
        return PgError::connect(ConnectFailure::Tls, message);
    }
    let mut source = std::error::Error::source(e);
    while let Some(err) = source {
        if err
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::ConnectionRefused)
        {
            return PgError::connect(ConnectFailure::Refused, message);
        }
        source = err.source();
    }
    PgError::connect(ConnectFailure::Other, message)
}

fn map_pool_error(e: PoolError) -> PgError {
    match e {
        PoolError::Timeout(TimeoutType::Wait) => PgError::Saturated(POOL_WAIT_TIMEOUT),
        PoolError::Timeout(TimeoutType::Create) => PgError::connect(
            ConnectFailure::Timeout,
            format!("pool create timeout ({POOL_CREATE_TIMEOUT:?})"),
        ),
        PoolError::Timeout(kind) => {
            PgError::connect(ConnectFailure::Other, format!("pool timeout ({kind:?})"))
        }
        PoolError::Backend(e) => classify_connect_error(&e),
        other => PgError::connect(ConnectFailure::Other, render_error_chain(&other)),
    }
}

pub fn application_name(env: &str) -> String {
    let mut name = String::with_capacity(APPLICATION_NAME_MAX_BYTES);
    name.push_str(APPLICATION_NAME_PREFIX);
    name.push('-');
    let mut last_dash = true;
    for ch in env.chars().flat_map(|c| c.to_lowercase()) {
        let ch = if ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            ch
        } else {
            '-'
        };
        if ch == '-' && last_dash {
            continue;
        }
        last_dash = ch == '-';
        name.push(ch);
        if name.len() >= APPLICATION_NAME_MAX_BYTES {
            break;
        }
    }
    let trimmed = name.trim_end_matches('-');
    if trimmed.len() == APPLICATION_NAME_PREFIX.len() {
        format!("{APPLICATION_NAME_PREFIX}-unknown")
    } else {
        trimmed.to_owned()
    }
}

pub fn parse_dsn(
    dsn: &str,
    app_name: &str,
    statement_timeout: Duration,
) -> Result<tokio_postgres::Config, String> {
    use tokio_postgres::config::{ChannelBinding, Host, SslMode};
    let mut cfg: tokio_postgres::Config =
        dsn.parse().map_err(|e| format!("does not parse: {e}"))?;
    cfg.application_name(app_name);
    match cfg.get_ssl_mode() {
        SslMode::Disable => {
            return Err(
                "sslmode=disable refused: the lookup must not cross the network in \
                        plaintext (use sslmode=require, the default)"
                    .to_owned(),
            );
        }
        SslMode::Prefer => {
            cfg.ssl_mode(SslMode::Require);
        }
        _ => {}
    }
    cfg.channel_binding(ChannelBinding::Require);
    for host in cfg.get_hosts() {
        if let Host::Tcp(h) = host {
            let first_label = h.split('.').next().unwrap_or("");
            if first_label.ends_with("-ro") || first_label.ends_with("-r") {
                return Err(format!(
                    "host {first_label:?} is a read-only replica service; the lookup must read \
                     the primary (`-rw`)"
                ));
            }
            let local = h == "localhost" || h.parse::<std::net::IpAddr>().is_ok();
            if !first_label.ends_with("-rw") && !local {
                warn!(
                    host = first_label,
                    "DSN host is not a `-rw` (read-write) service; fine for a local DB, wrong \
                     for the cluster ledger"
                );
            }
        }
    }
    cfg.options(format!(
        "-c statement_timeout={}",
        statement_timeout.as_millis()
    ));
    Ok(cfg)
}

pub fn make_tls() -> crate::tls::MakeLedgerTls {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("aws-lc-rs supports TLS 1.3")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(ChannelBindingOnlyVerifier { provider }))
        .with_no_client_auth();
    crate::tls::MakeLedgerTls::new(config)
}

/// The server's certificate chain and hostname are not verified: the database
/// certificate is issued by an internal CA this service does not carry. The
/// server is authenticated by SCRAM-SHA-256-PLUS channel binding, which
/// `parse_dsn` requires, so a party presenting a different certificate cannot
/// complete the login. Handshake signatures are still verified.
#[derive(Debug)]
struct ChannelBindingOnlyVerifier {
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl rustls::client::danger::ServerCertVerifier for ChannelBindingOnlyVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer,
        _intermediates: &[rustls::pki_types::CertificateDer],
        _server_name: &rustls::pki_types::ServerName,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn row_to_hold(r: &tokio_postgres::Row) -> Result<Hold, tokio_postgres::Error> {
    Ok(Hold {
        user_id: r.try_get::<_, i64>(0)?,
        hold_id: r.try_get::<_, i64>(1)?,
        head: r.try_get::<_, Option<String>>(2)?,
        expires_at: r.try_get::<_, DateTime<Utc>>(3)?,
        case_group_id: r.try_get::<_, Option<i32>>(4)?,
        reason: r.try_get::<_, String>(5)?,
        action_kind: r.try_get::<_, String>(6)?,
        label: r.try_get::<_, Option<String>>(7)?,
    })
}

pub struct PgHoldRepo {
    config: tokio_postgres::Config,
    pool: Pool,
    statement_timeout: Duration,
}

impl PgHoldRepo {
    pub fn new(
        dsn: &str,
        env: &str,
        statement_timeout: Duration,
        pool_max: usize,
    ) -> Result<Self, PgError> {
        let app_name = application_name(env);
        let config = parse_dsn(dsn, &app_name, statement_timeout).map_err(PgError::BadDsn)?;
        let manager = Manager::from_config(
            config.clone(),
            make_tls(),
            ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            },
        );
        let pool = Pool::builder(manager)
            .max_size(pool_max)
            .runtime(Runtime::Tokio1)
            .wait_timeout(Some(POOL_WAIT_TIMEOUT))
            .create_timeout(Some(POOL_CREATE_TIMEOUT))
            .recycle_timeout(Some(POOL_RECYCLE_TIMEOUT))
            .build()
            .map_err(|e| PgError::BadDsn(format!("pool: {e}")))?;
        info!(
            application_name = app_name,
            statement_timeout_ms = statement_timeout.as_millis() as u64,
            max_size = pool_max,
            recycling = "fast",
            wait_timeout_ms = POOL_WAIT_TIMEOUT.as_millis() as u64,
            create_timeout_ms = POOL_CREATE_TIMEOUT.as_millis() as u64,
            "ledger pool built (lazy; first session opens on the first checkout)"
        );
        Ok(Self {
            config,
            pool,
            statement_timeout,
        })
    }

    pub fn config(&self) -> &tokio_postgres::Config {
        &self.config
    }

    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    async fn checkout(&self) -> Result<deadpool_postgres::Object, PgError> {
        self.pool.get().await.map_err(|e| {
            if !matches!(e, PoolError::Timeout(TimeoutType::Wait)) {
                metrics::DB_UP.set(0);
            }
            map_pool_error(e)
        })
    }

    async fn lookup_once(
        &self,
        user_ids: &[i64],
        labels: &[String],
        budget: Duration,
    ) -> Result<Vec<Hold>, (PgError, bool)> {
        let client = self.checkout().await.map_err(|e| (e, false))?;
        let started = Instant::now();
        let run = async {
            let stmt = client.prepare_cached(PROBE_SQL).await?;
            client.query(&stmt, &[&user_ids, &labels]).await
        };
        let res = tokio::time::timeout(budget, run).await;
        metrics::DB_SECONDS.observe(started.elapsed().as_secs_f64());
        match res {
            Ok(Ok(rows)) => {
                metrics::DB_UP.set(1);
                rows.iter()
                    .map(row_to_hold)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| (PgError::Decode(e.to_string()), false))
            }
            Ok(Err(e)) => {
                let err = classify_pg_error(&e);
                let dead = matches!(err, PgError::Transport(_));
                if dead {
                    drop(deadpool_postgres::Object::take(client));
                    metrics::DB_UP.set(0);
                }
                Err((err, dead))
            }
            Err(_) => {
                drop(deadpool_postgres::Object::take(client));
                Err((PgError::Timeout(budget), false))
            }
        }
    }

    async fn run_simple(&self, sql: &str, budget: Duration) -> Result<(), PgError> {
        let client = self.checkout().await?;
        let run = async {
            let stmt = client.prepare(sql).await?;
            client.query(&stmt, &[]).await.map(|_| ())
        };
        match tokio::time::timeout(budget, run).await {
            Ok(Ok(())) => {
                metrics::DB_UP.set(1);
                Ok(())
            }
            Ok(Err(e)) => {
                let err = classify_pg_error(&e);
                if matches!(err, PgError::Transport(_)) {
                    drop(deadpool_postgres::Object::take(client));
                    metrics::DB_UP.set(0);
                }
                Err(err)
            }
            Err(_) => {
                drop(deadpool_postgres::Object::take(client));
                Err(PgError::Timeout(budget))
            }
        }
    }
}

#[async_trait]
impl HoldRepo for PgHoldRepo {
    async fn lookup(
        &self,
        user_ids: &[i64],
        labels: &[String],
        budget: Duration,
    ) -> Result<Vec<Hold>, PgError> {
        match self.lookup_once(user_ids, labels, budget).await {
            Ok(rows) => Ok(rows),
            Err((e, true)) => {
                info!(error = %e, "lookup hit a dead session; retrying once on a fresh one");
                self.lookup_once(user_ids, labels, budget)
                    .await
                    .map_err(|(e, _)| e)
            }
            Err((e, false)) => Err(e),
        }
    }

    async fn contract_check(&self, budget: Duration) -> Result<(), PgError> {
        self.run_simple(CONTRACT_SQL, budget).await?;
        let client = self.checkout().await?;
        let run = client.prepare_cached(PROBE_SQL);
        match tokio::time::timeout(budget, run).await {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(classify_pg_error(&e)),
            Err(_) => Err(PgError::Timeout(budget)),
        }
    }

    async fn ping(&self, budget: Duration) -> Result<(), PgError> {
        self.run_simple(PING_SQL, budget).await
    }

    fn pool_status(&self) -> Option<PoolStatus> {
        let s = self.pool.status();
        Some(PoolStatus {
            size: s.size,
            available: s.available,
            waiting: s.waiting,
            max_size: s.max_size,
        })
    }

    fn endpoint(&self) -> String {
        let cfg = &self.config;
        let hosts: Vec<String> = cfg
            .get_hosts()
            .iter()
            .map(|h| match h {
                tokio_postgres::config::Host::Tcp(s) => s.clone(),
                #[cfg(unix)]
                tokio_postgres::config::Host::Unix(p) => p.display().to_string(),
            })
            .collect();
        let ports: Vec<String> = cfg.get_ports().iter().map(u16::to_string).collect();
        let tls = match cfg.get_ssl_mode() {
            tokio_postgres::config::SslMode::Require => "require",
            tokio_postgres::config::SslMode::Prefer => "prefer",
            tokio_postgres::config::SslMode::Disable => "disable",
            _ => "other",
        };
        let cb = match cfg.get_channel_binding() {
            tokio_postgres::config::ChannelBinding::Require => "require",
            tokio_postgres::config::ChannelBinding::Prefer => "prefer",
            tokio_postgres::config::ChannelBinding::Disable => "disable",
            _ => "other",
        };
        format!(
            "host={} port={} db={} user={} tls={tls} channel_binding={cb} statement_timeout_ms={}",
            hosts.join(","),
            ports.join(","),
            cfg.get_dbname().unwrap_or(""),
            cfg.get_user().unwrap_or(""),
            self.statement_timeout.as_millis()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T150: Duration = Duration::from_millis(150);

    #[test]
    fn application_name_is_env_suffixed_sanitised_and_bounded() {
        assert_eq!(application_name("prod"), "xai-abuse-ledger-service-prod");
        assert_eq!(
            application_name("staging"),
            "xai-abuse-ledger-service-staging"
        );
        assert_eq!(
            application_name("Team_A/Dev.1"),
            "xai-abuse-ledger-service-team-a-dev-1"
        );
        assert_eq!(application_name(""), "xai-abuse-ledger-service-unknown");
        assert_eq!(application_name("___"), "xai-abuse-ledger-service-unknown");
        let long = application_name(&"x".repeat(200));
        assert_eq!(long.len(), 63);
        assert!(long.starts_with("xai-abuse-ledger-service-xxx"));
        for n in [
            application_name("prod"),
            application_name("Team_A/Dev.1"),
            long.clone(),
        ] {
            assert!(
                n.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
                "{n}"
            );
            assert!(!n.ends_with('-'));
        }
    }

    #[test]
    fn dsn_is_pinned_with_app_name_timeout_sslmode_and_channel_binding() {
        use tokio_postgres::config::{ChannelBinding, SslMode};
        let c = parse_dsn(
            "postgresql://u:p@ledger-rw.example:5432/ledger",
            "xai-abuse-ledger-service-prod",
            T150,
        )
        .expect("valid DSN");
        assert_eq!(
            c.get_application_name(),
            Some("xai-abuse-ledger-service-prod")
        );
        assert_eq!(c.get_options(), Some("-c statement_timeout=150"));
        assert_eq!(c.get_ssl_mode(), SslMode::Require, "prefer -> require");
        assert_eq!(c.get_channel_binding(), ChannelBinding::Require);
        let c = parse_dsn(
            "postgresql://u:p@ledger-rw.example:5432/ledger?sslmode=require&channel_binding=prefer",
            "app",
            Duration::from_millis(350),
        )
        .expect("valid DSN");
        assert_eq!(c.get_ssl_mode(), SslMode::Require);
        assert_eq!(c.get_channel_binding(), ChannelBinding::Require);
        assert_eq!(c.get_options(), Some("-c statement_timeout=350"));
    }

    #[test]
    fn bad_plaintext_or_readonly_dsn_is_refused() {
        for (dsn, why) in [
            ("not a dsn", "does not parse"),
            (
                "postgresql://u:p@ledger-rw.example:5432/ledger?sslmode=disable",
                "sslmode=disable refused",
            ),
            (
                "postgresql://u:p@ledger-ro.ns.svc:5432/ledger",
                "read-only replica service",
            ),
            (
                "postgresql://u:p@ledger-r.ns.svc:5432/ledger",
                "read-only replica service",
            ),
        ] {
            let reason = parse_dsn(dsn, "app", T150).expect_err("bad DSN");
            assert!(reason.contains(why), "{dsn}: {reason}");
            assert!(!reason.contains(":p@"), "reason never carries the password");
            match PgHoldRepo::new(dsn, "prod", T150, 8) {
                Err(PgError::BadDsn(r)) => assert!(r.contains(why), "{dsn}: {r}"),
                Err(other) => panic!("{dsn}: expected BadDsn, got {other:?}"),
                Ok(_) => panic!("{dsn}: expected BadDsn, got a repo"),
            }
        }
    }

    #[tokio::test]
    async fn pool_is_lazy_and_bounded() {
        let repo = PgHoldRepo::new(
            "postgresql://u:p@ledger-rw.invalid:5432/ledger",
            "staging",
            T150,
            8,
        )
        .expect("repo");
        let status = repo.pool_status().unwrap();
        assert_eq!(status.max_size, 8);
        assert_eq!(status.size, 0, "no connection at build time");
        let t = repo.pool().timeouts();
        assert_eq!(t.wait, Some(POOL_WAIT_TIMEOUT));
        assert_eq!(t.create, Some(POOL_CREATE_TIMEOUT));
        assert_eq!(t.recycle, Some(POOL_RECYCLE_TIMEOUT));
        let ep = repo.endpoint();
        assert_eq!(
            ep,
            "host=ledger-rw.invalid port=5432 db=ledger user=u tls=require \
             channel_binding=require statement_timeout_ms=150"
        );
        assert!(!ep.contains(":p@"));
        assert_eq!(
            repo.config().get_application_name(),
            Some("xai-abuse-ledger-service-staging")
        );
    }

    #[test]
    fn pg_errors_map_to_closed_codes_and_statuses() {
        use ErrorCode::*;
        let cases: Vec<(PgError, ErrorCode, u16, bool)> = vec![
            (
                PgError::connect(ConnectFailure::Refused, "x"),
                ConnectRefused,
                503,
                true,
            ),
            (
                PgError::connect(ConnectFailure::Auth, "x"),
                AuthFailed,
                503,
                true,
            ),
            (PgError::connect(ConnectFailure::Tls, "x"), Tls, 503, true),
            (
                PgError::connect(ConnectFailure::Timeout, "x"),
                Timeout,
                504,
                true,
            ),
            (
                PgError::connect(ConnectFailure::Other, "x"),
                Other,
                500,
                false,
            ),
            (PgError::Saturated(POOL_WAIT_TIMEOUT), PoolWait, 503, true),
            (PgError::Transport("x".into()), Other, 503, true),
            (
                PgError::Query {
                    sqlstate: "57014".into(),
                    message: "canceling statement due to statement timeout".into(),
                },
                StatementTimeout,
                504,
                true,
            ),
            (
                PgError::Query {
                    sqlstate: "42501".into(),
                    message: "permission denied for user \"app\"".into(),
                },
                Other,
                500,
                false,
            ),
            (
                PgError::Query {
                    sqlstate: "57P01".into(),
                    message: "terminating connection due to administrator command".into(),
                },
                ServerUnhealthy,
                503,
                true,
            ),
            (
                PgError::Query {
                    sqlstate: "57P03".into(),
                    message: "the database system is starting up".into(),
                },
                ServerUnhealthy,
                503,
                true,
            ),
            (
                PgError::Query {
                    sqlstate: "53300".into(),
                    message: "too many connections".into(),
                },
                ServerUnhealthy,
                503,
                true,
            ),
            (PgError::Decode("x".into()), Decode, 500, false),
            (
                PgError::Timeout(Duration::from_millis(1)),
                Timeout,
                504,
                true,
            ),
        ];
        for (err, code, status, retryable) in cases {
            assert_eq!(err.code(), code, "{err}");
            assert_eq!(err.status().as_u16(), status, "{err}");
            assert_eq!(err.retryable(), retryable, "{err}");
        }
        assert_eq!(
            PgError::connect(
                ConnectFailure::Auth,
                "FATAL: password authentication failed"
            )
            .to_string(),
            "connect (auth): FATAL: password authentication failed"
        );
        assert!(sqlstate_is_unhealthy("57014"));
        assert!(sqlstate_is_unhealthy("57P02"));
        assert!(sqlstate_is_unhealthy("53200"));
        assert!(!sqlstate_is_unhealthy("42501"));
        assert!(!sqlstate_is_unhealthy("42703"));
        assert_eq!(SQLSTATE_AUTH, ["28P01", "28000"]);
        assert_eq!(SQLSTATE_QUERY_CANCELED, "57014");
    }

    #[tokio::test]
    async fn connect_errors_are_classified_from_the_typed_error() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let cfg: tokio_postgres::Config = format!("postgresql://u:p@127.0.0.1:{port}/db")
            .parse()
            .unwrap();
        let refused = cfg.connect(tokio_postgres::NoTls).await.err().unwrap();
        let err = classify_connect_error(&refused);
        match &err {
            PgError::Connect { kind, message } => {
                assert_eq!(*kind, ConnectFailure::Refused, "{message}");
            }
            other => panic!("expected Connect, got {other:?}"),
        }
        assert_eq!(err.code(), ErrorCode::ConnectRefused);
        assert_eq!(err.status().as_u16(), 503);
    }

    #[tokio::test]
    async fn connect_error_message_carries_the_source_chain() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let cfg: tokio_postgres::Config = format!("postgresql://u:p@127.0.0.1:{port}/db")
            .parse()
            .unwrap();
        let refused = cfg.connect(tokio_postgres::NoTls).await.err().unwrap();
        let rendered = render_error_chain(&refused);
        assert!(
            rendered.starts_with("error connecting to server: "),
            "{rendered}"
        );
        assert!(
            rendered.to_lowercase().contains("connection refused"),
            "{rendered}"
        );
        match classify_connect_error(&refused) {
            PgError::Connect { message, .. } => assert_eq!(message, rendered),
            other => panic!("expected Connect, got {other:?}"),
        }
        assert_eq!(refused.to_string(), "error connecting to server");
    }

    #[tokio::test]
    async fn require_without_a_binding_is_the_staging_authentication_error() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut len = [0u8; 4];
            sock.read_exact(&mut len).await.unwrap();
            let mut body = vec![0u8; u32::from_be_bytes(len) as usize - 4];
            sock.read_exact(&mut body).await.unwrap();
            assert_eq!(&body[..4], &[0, 3, 0, 0], "protocol 3.0");
            let mechanisms = b"SCRAM-SHA-256-PLUS\0SCRAM-SHA-256\0\0";
            let mut msg = vec![b'R'];
            msg.extend_from_slice(&((4 + 4 + mechanisms.len()) as u32).to_be_bytes());
            msg.extend_from_slice(&10u32.to_be_bytes());
            msg.extend_from_slice(mechanisms);
            sock.write_all(&msg).await.unwrap();
            sock.flush().await.unwrap();
            let mut sink = [0u8; 64];
            let _ = sock.read(&mut sink).await;
        });
        let mut cfg: tokio_postgres::Config = format!("postgresql://u:p@127.0.0.1:{port}/db")
            .parse()
            .unwrap();
        cfg.channel_binding(tokio_postgres::config::ChannelBinding::Require);
        let err = tokio::time::timeout(Duration::from_secs(5), cfg.connect(tokio_postgres::NoTls))
            .await
            .expect("server answers")
            .err()
            .expect("channel_binding=require without a binding must fail");
        server.abort();
        assert_eq!(err.to_string(), "authentication error");
        assert_eq!(
            render_error_chain(&err),
            "authentication error: server did not use channel binding"
        );
        let classified = classify_connect_error(&err);
        match &classified {
            PgError::Connect { kind, message } => {
                assert_eq!(*kind, ConnectFailure::Other);
                assert_eq!(
                    message,
                    "authentication error: server did not use channel binding"
                );
            }
            other => panic!("expected Connect, got {other:?}"),
        }
        assert_eq!(
            classified.to_string(),
            "connect (other): authentication error: server did not use channel binding"
        );
    }

    #[tokio::test]
    async fn lookup_against_a_closed_port_is_connect_refused() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let repo = PgHoldRepo::new(
            &format!("postgresql://u:p@127.0.0.1:{port}/db"),
            "test",
            T150,
            2,
        )
        .unwrap();
        let err = repo
            .lookup(&[1], &[], Duration::from_millis(250))
            .await
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::ConnectRefused, "{err}");
        let err = repo.ping(Duration::from_millis(250)).await.unwrap_err();
        assert_eq!(err.code(), ErrorCode::ConnectRefused, "{err}");
        let err = repo
            .contract_check(Duration::from_millis(250))
            .await
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::ConnectRefused, "{err}");
    }
}
