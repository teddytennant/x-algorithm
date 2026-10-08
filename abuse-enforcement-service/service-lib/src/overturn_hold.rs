//! Overturn-hold gate: the last precheck before a suspend (or a gated
//! label) is dispatched.
//! It exists so we do not re-apply an enforcement that a human reviewer
//! already overturned on appeal.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use moka::ops::compute::Op;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::metrics;

pub(crate) const CONFIG_KEY: &str = "overturn_hold_gate";

pub const STATUS_HOLD_OVERTURNED: &str = "hold_overturned";

pub const STATUS_HOLD_LOOKUP_FAILED: &str = "hold_lookup_failed";

pub(crate) const LOOKUP_PATH: &str = "/v1/holds/lookup";
pub(crate) const READY_PATH: &str = "/readyz";
pub(crate) const DEADLINE_HEADER: &str = "x-request-deadline-ms";
pub(crate) const CONNECT_TIMEOUT: Duration = Duration::from_millis(200);
pub(crate) const MAX_BODY_BYTES: usize = 64 * 1024;

const DEFAULT_TIMEOUT_MS: u64 = 200;
const DEFAULT_POSITIVE_CACHE_SECS: u64 = 60;
const MIN_TIMEOUT_MS: u64 = 10;
const MAX_TIMEOUT_MS: u64 = 5_000;
const MAX_POSITIVE_CACHE_SECS: u64 = 3_600;
const STARTUP_PROBE_TIMEOUT: Duration = Duration::from_secs(5);
pub(crate) const BREAKER_FAILURES: u32 = 5;
pub(crate) const DEFAULT_BREAKER_OPEN: Duration = Duration::from_secs(10);
pub(crate) const REACHABLE_ERRORS_BEFORE_BACKOFF: u32 = 5;
pub(crate) const REACHABLE_ERROR_BACKOFF: Duration = Duration::from_secs(5);
const HALF_OPEN_STALE_SLACK: Duration = Duration::from_millis(1_000);

fn half_open_stale_after(timeout_ms: u64) -> Duration {
    CONNECT_TIMEOUT + Duration::from_millis(timeout_ms) + HALF_OPEN_STALE_SLACK
}
const MAX_LOGGED_BAD_VALUES: usize = 32;
pub(crate) const CACHE_MAX_WEIGHT: u64 = 100_000;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateMode {
        #[default]
    Off,
        Shadow,
        Enforce,
}

impl GateMode {
            pub fn as_str(self) -> &'static str {
        match self {
            GateMode::Off => "off",
            GateMode::Shadow => "shadow",
            GateMode::Enforce => "enforce",
        }
    }

            fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Some(GateMode::Off),
            "shadow" => Some(GateMode::Shadow),
            "enforce" => Some(GateMode::Enforce),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnProbeError {
        #[default]
    Allow,
            Skip,
}

impl OnProbeError {
            pub fn as_str(self) -> &'static str {
        match self {
            OnProbeError::Allow => "allow",
            OnProbeError::Skip => "skip",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "allow" => Some(OnProbeError::Allow),
            "skip" => Some(OnProbeError::Skip),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKind {
        Suspend,
            Label,
}

impl GateKind {
        pub fn as_str(self) -> &'static str {
        match self {
            GateKind::Suspend => HOLD_KIND_SUSPEND,
            GateKind::Label => HOLD_KIND_LABEL,
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            HOLD_KIND_SUSPEND => Some(GateKind::Suspend),
            HOLD_KIND_LABEL => Some(GateKind::Label),
            _ => None,
        }
    }
}

pub(crate) const HOLD_KIND_SUSPEND: &str = "suspend";
pub(crate) const HOLD_KIND_LABEL: &str = "label";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatedAction {
                            Suspend { perm: bool },
                Label { name: String },
}

impl GatedAction {
        pub const TEMPORARY_SUSPEND: Self = Self::Suspend { perm: false };
        pub const PERMANENT_SUSPEND: Self = Self::Suspend { perm: true };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldGateConfig {
            pub mode: GateMode,
        pub on_probe_error: OnProbeError,
                        pub topics: Vec<String>,
                            pub kinds: Vec<GateKind>,
                                                            pub labels: Vec<String>,
            pub timeout_ms: u64,
                    pub positive_cache_secs: u64,
                                                pub perm_suspend_case_groups: Vec<i32>,
}

impl Default for HoldGateConfig {
    fn default() -> Self {
        Self {
            mode: GateMode::Off,
            on_probe_error: OnProbeError::Allow,
            topics: Vec::new(),
            kinds: vec![GateKind::Suspend],
            labels: Vec::new(),
            timeout_ms: DEFAULT_TIMEOUT_MS,
            positive_cache_secs: DEFAULT_POSITIVE_CACHE_SECS,
            perm_suspend_case_groups: Vec::new(),
        }
    }
}

static LOGGED_BAD_CONFIG: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);

fn first_sighting(field: &str, raw: &Value) -> bool {
    let mut guard = LOGGED_BAD_CONFIG.lock().unwrap_or_else(|p| p.into_inner());
    first_sighting_in(guard.get_or_insert_with(Default::default), field, raw)
}

fn first_sighting_in(
    seen: &mut std::collections::HashSet<String>,
    field: &str,
    raw: &Value,
) -> bool {
    let key = format!("{field}={raw}");
    if seen.contains(&key) {
        return false;
    }
    if seen.len() >= MAX_LOGGED_BAD_VALUES {
        seen.clear();
    }
    seen.insert(key);
    true
}

fn log_bad_config_once(what: &str, raw: &Value) {
    if !first_sighting(what, raw) {
        return;
    }
    error!(
        key = CONFIG_KEY,
        field = what,
        value = %raw,
        "overturn_hold_gate config is not recognised; treating the gate as OFF \
         (a typo must never crash the consumer or silently turn shadow into enforce)"
    );
}

fn log_bad_kinds_once(raw: &Value, offending: &Value) {
    if !first_sighting("kinds", raw) {
        return;
    }
    error!(
        key = CONFIG_KEY,
        field = "kinds",
        value = %raw,
        rejected = %offending,
        accepted = "[\"suspend\", \"label\"] (non-empty array; case-insensitive; duplicates collapse)",
        "overturn_hold_gate.kinds is not recognised → the WHOLE gate is OFF (suspend gating \
         included) until the paste is fixed; accepted values are \"suspend\" and \"label\""
    );
}

fn log_ignored_field_once(what: &str, raw: &Value, using: &str) {
    if !first_sighting(what, raw) {
        return;
    }
    warn!(
        key = CONFIG_KEY,
        field = what,
        value = %raw,
        using,
        "overturn_hold_gate field not recognised; using the default"
    );
}

fn string_list(arr: &[Value]) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::with_capacity(arr.len());
    for entry in arr {
        let s = entry.as_str()?.trim();
        if !s.is_empty() && !out.iter().any(|t| t == s) {
            out.push(s.to_owned());
        }
    }
    if !arr.is_empty() && out.is_empty() {
        return None;
    }
    Some(out)
}

impl HoldGateConfig {
                                                                                                                    pub fn from_config(config: Option<&Value>) -> Self {
        let Some(obj) = config.and_then(|c| c.get(CONFIG_KEY)) else {
            return Self::default();
        };
        let Some(map) = obj.as_object() else {
            log_bad_config_once("object", obj);
            return Self::default();
        };
        let mut cfg = Self::default();

        match map.get("mode") {
            None | Some(Value::Null) => return cfg,
            Some(v) => match v.as_str().and_then(GateMode::parse) {
                Some(GateMode::Off) => return cfg,
                Some(mode) => cfg.mode = mode,
                None => {
                    log_bad_config_once("mode", v);
                    return Self::default();
                }
            },
        }

        match map.get("on_probe_error") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_str().and_then(OnProbeError::parse) {
                Some(p) => cfg.on_probe_error = p,
                None => log_ignored_field_once("on_probe_error", v, "allow"),
            },
        }

        match map.get("topics") {
            None | Some(Value::Null) => {}
            Some(raw @ Value::Array(arr)) => {
                let mut topics = Vec::with_capacity(arr.len());
                for entry in arr {
                    let Some(s) = entry.as_str() else {
                        log_bad_config_once("topics", raw);
                        return Self::default();
                    };
                    let s = s.trim();
                    if !s.is_empty() {
                        topics.push(s.to_owned());
                    }
                }
                if !arr.is_empty() && topics.is_empty() {
                    log_bad_config_once("topics", raw);
                    return Self::default();
                }
                cfg.topics = topics;
            }
            Some(v) => {
                log_bad_config_once("topics", v);
                return Self::default();
            }
        }

        match map.get("kinds") {
            None | Some(Value::Null) => {}
            Some(raw @ Value::Array(arr)) => {
                let mut kinds: Vec<GateKind> = Vec::with_capacity(arr.len());
                for entry in arr {
                    let Some(kind) = entry.as_str().and_then(GateKind::parse) else {
                        log_bad_kinds_once(raw, entry);
                        return Self::default();
                    };
                    if !kinds.contains(&kind) {
                        kinds.push(kind);
                    }
                }
                if kinds.is_empty() {
                    log_bad_kinds_once(raw, raw);
                    return Self::default();
                }
                cfg.kinds = kinds;
            }
            Some(v) => {
                log_bad_kinds_once(v, v);
                return Self::default();
            }
        }

        match map.get("labels") {
            None | Some(Value::Null) => {}
            Some(raw @ Value::Array(arr)) => match string_list(arr) {
                Some(labels) => cfg.labels = labels,
                None => {
                    log_bad_config_once("labels", raw);
                    return Self::default();
                }
            },
            Some(v) => {
                log_bad_config_once("labels", v);
                return Self::default();
            }
        }
        if cfg.kinds.contains(&GateKind::Label)
            && cfg.labels.is_empty()
            && first_sighting(
                "kinds+labels",
                &Value::String("label kind without labels".into()),
            )
        {
            warn!(
                key = CONFIG_KEY,
                kinds = ?cfg.kinds,
                "overturn_hold_gate: `kinds` names `label` but `labels` is empty; label gating is \
                 inert until `labels` names at least one label (suspend gating unaffected)"
            );
        }

        match map.get("perm_suspend_case_groups") {
            None | Some(Value::Null) => {}
            Some(raw @ Value::Array(arr)) => {
                let mut groups = Vec::with_capacity(arr.len());
                for entry in arr {
                    let Some(id) = entry.as_i64().and_then(|n| i32::try_from(n).ok()) else {
                        log_bad_config_once("perm_suspend_case_groups", raw);
                        return Self::default();
                    };
                    if !groups.contains(&id) {
                        groups.push(id);
                    }
                }
                cfg.perm_suspend_case_groups = groups;
            }
            Some(v) => {
                log_bad_config_once("perm_suspend_case_groups", v);
                return Self::default();
            }
        }

        match map.get("timeout_ms") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_u64() {
                Some(ms) => cfg.timeout_ms = ms.clamp(MIN_TIMEOUT_MS, MAX_TIMEOUT_MS),
                None => log_ignored_field_once("timeout_ms", v, "200"),
            },
        }
        match map.get("positive_cache_secs") {
            None | Some(Value::Null) => {}
            Some(v) => match v.as_u64() {
                Some(secs) => cfg.positive_cache_secs = secs.min(MAX_POSITIVE_CACHE_SECS),
                None => log_ignored_field_once("positive_cache_secs", v, "60"),
            },
        }
        cfg
    }

        pub(crate) fn covers_topic(&self, topic: &str) -> bool {
        self.topics.is_empty() || self.topics.iter().any(|t| t == topic)
    }

        pub fn enabled(&self) -> bool {
        self.mode != GateMode::Off
    }

        pub fn gates_suspend(&self) -> bool {
        self.kinds.contains(&GateKind::Suspend)
    }

            pub fn gates_label(&self, name: &str) -> bool {
        self.kinds.contains(&GateKind::Label) && self.labels.iter().any(|l| l == name)
    }

                fn hold_applies(&self, hold: &Hold, perm: bool) -> bool {
        !perm
            || self.perm_suspend_case_groups.is_empty()
            || hold.reason == HOLD_REASON_MANUAL
            || hold
                .case_group_id
                .is_some_and(|cg| self.perm_suspend_case_groups.contains(&cg))
    }
}

const HOLD_REASON_MANUAL: &str = "manual";

pub fn startup_probe_requested(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}


#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct Hold {
        pub(crate) user_id: i64,
            pub(crate) hold_id: i64,
        #[serde(default)]
    pub(crate) head: Option<String>,
        pub(crate) expires_at: DateTime<Utc>,
            #[serde(default)]
    pub(crate) case_group_id: Option<i32>,
            pub(crate) reason: String,
                    pub(crate) action_kind: String,
            #[serde(default)]
    pub(crate) label: Option<String>,
}

impl Hold {
        fn is_suspend(&self) -> bool {
        self.action_kind == HOLD_KIND_SUSPEND
    }

        fn withholds_label(&self, name: &str) -> bool {
        self.action_kind == HOLD_KIND_LABEL && self.label.as_deref() == Some(name)
    }
}

#[derive(Debug, Serialize)]
struct LookupRequest<'a> {
        user_ids: &'a [i64],
        labels: &'a [String],
}

#[derive(Debug, Deserialize)]
struct LookupResponse {
    holds: Vec<Hold>,
}

#[derive(Debug, Deserialize)]
struct ErrorBody {
    code: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LedgerCode {
        ConnectRefused,
        AuthFailed,
        Tls,
        StatementTimeout,
            ServerUnhealthy,
            PoolWait,
        Decode,
        Timeout,
                Other,
}

impl LedgerCode {
                fn parse(raw: &str) -> Self {
        match raw.trim() {
            "connect_refused" => LedgerCode::ConnectRefused,
            "auth_failed" => LedgerCode::AuthFailed,
            "tls" => LedgerCode::Tls,
            "statement_timeout" => LedgerCode::StatementTimeout,
            "server_unhealthy" => LedgerCode::ServerUnhealthy,
            "pool_wait" => LedgerCode::PoolWait,
            "decode" => LedgerCode::Decode,
            "timeout" => LedgerCode::Timeout,
            _ => LedgerCode::Other,
        }
    }

        fn probe_code(self) -> ProbeErrorCode {
        match self {
            LedgerCode::ConnectRefused => ProbeErrorCode::ConnectRefused,
            LedgerCode::AuthFailed => ProbeErrorCode::AuthFailed,
            LedgerCode::Tls => ProbeErrorCode::Tls,
            LedgerCode::StatementTimeout => ProbeErrorCode::StatementTimeout,
            LedgerCode::ServerUnhealthy => ProbeErrorCode::ServerUnhealthy,
            LedgerCode::PoolWait => ProbeErrorCode::PoolWait,
            LedgerCode::Decode => ProbeErrorCode::Decode,
            LedgerCode::Timeout => ProbeErrorCode::Timeout,
            LedgerCode::Other => ProbeErrorCode::Other,
        }
    }

        fn breaker_signal(self) -> BreakerSignal {
        match self {
            LedgerCode::ConnectRefused
            | LedgerCode::AuthFailed
            | LedgerCode::Tls
            | LedgerCode::StatementTimeout
            | LedgerCode::ServerUnhealthy
            | LedgerCode::Timeout => BreakerSignal::Failure,
            LedgerCode::PoolWait | LedgerCode::Decode => BreakerSignal::Neutral,
            LedgerCode::Other => BreakerSignal::Reachable,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum HoldStoreError {
        #[error("no hold store configured (OVERTURN_HOLD_LEDGER_URL unset)")]
    NotConfigured,
                    #[error("bad URL: {0}")]
    BadUrl(String),
                #[error("ledger service unreachable: {0}")]
    Unreachable(String),
            #[error("timed out after {0:?}")]
    Timeout(Duration),
                    #[error("ledger service answered {status}{}", code.map_or_else(String::new, |c| format!(" code={}", c.probe_code().as_str())))]
    Upstream {
        status: u16,
        code: Option<LedgerCode>,
    },
                    #[error("response decode: {0}")]
    Decode(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProbeErrorCode {
        ConnectRefused,
        AuthFailed,
        StatementTimeout,
                    ServerUnhealthy,
        PoolWait,
        BreakerOpen,
                ReachableBackoff,
        BadUrl,
        NotConfigured,
            Timeout,
        Decode,
        Tls,
            LedgerUnreachable,
        LedgerStatus,
            Other,
}

impl ProbeErrorCode {
        pub(crate) fn as_str(self) -> &'static str {
        match self {
            ProbeErrorCode::ConnectRefused => "connect_refused",
            ProbeErrorCode::AuthFailed => "auth_failed",
            ProbeErrorCode::StatementTimeout => "statement_timeout",
            ProbeErrorCode::ServerUnhealthy => "server_unhealthy",
            ProbeErrorCode::PoolWait => "pool_wait",
            ProbeErrorCode::BreakerOpen => "breaker_open",
            ProbeErrorCode::ReachableBackoff => "reachable_backoff",
            ProbeErrorCode::BadUrl => "bad_url",
            ProbeErrorCode::NotConfigured => "not_configured",
            ProbeErrorCode::Timeout => "timeout",
            ProbeErrorCode::Decode => "decode",
            ProbeErrorCode::Tls => "tls",
            ProbeErrorCode::LedgerUnreachable => "ledger_unreachable",
            ProbeErrorCode::LedgerStatus => "ledger_status",
            ProbeErrorCode::Other => "other",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeError {
        pub(crate) code: ProbeErrorCode,
        pub(crate) detail: String,
}

impl ProbeError {
    fn breaker_open(retry_in_secs: u64) -> Self {
        Self {
            code: ProbeErrorCode::BreakerOpen,
            detail: format!("breaker open: ledger probe skipped (next trial in {retry_in_secs}s)"),
        }
    }

    fn reachable_backoff(retry_in_secs: u64, streak: u32, last: Option<ProbeErrorCode>) -> Self {
        Self {
            code: ProbeErrorCode::ReachableBackoff,
            detail: format!(
                "reachable-error backoff: the ledger service answered but could not serve \
                 {streak} consecutive probes (last: {}); probe skipped (next in {retry_in_secs}s)",
                last.map_or("?", ProbeErrorCode::as_str)
            ),
        }
    }
}

impl From<&HoldStoreError> for ProbeError {
    fn from(e: &HoldStoreError) -> Self {
        Self {
            code: e.code(),
            detail: e.to_string(),
        }
    }
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.detail)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BreakerSignal {
            Reachable,
        Failure,
            Neutral,
}

const HTTP_GATEWAY_TIMEOUT: u16 = 504;

impl HoldStoreError {
        pub(crate) fn code(&self) -> ProbeErrorCode {
        match self {
            HoldStoreError::NotConfigured => ProbeErrorCode::NotConfigured,
            HoldStoreError::BadUrl(_) => ProbeErrorCode::BadUrl,
            HoldStoreError::Unreachable(_) => ProbeErrorCode::LedgerUnreachable,
            HoldStoreError::Timeout(_) => ProbeErrorCode::Timeout,
            HoldStoreError::Upstream {
                code: Some(code), ..
            } => code.probe_code(),
            HoldStoreError::Upstream {
                status: HTTP_GATEWAY_TIMEOUT,
                code: None,
            } => ProbeErrorCode::Timeout,
            HoldStoreError::Upstream { code: None, .. } => ProbeErrorCode::LedgerStatus,
            HoldStoreError::Decode(_) => ProbeErrorCode::Decode,
        }
    }

                            pub(crate) fn ledger_answered(&self) -> Option<bool> {
        match self {
            HoldStoreError::Upstream { .. } | HoldStoreError::Decode(_) => Some(true),
            HoldStoreError::Unreachable(_) | HoldStoreError::Timeout(_) => Some(false),
            HoldStoreError::BadUrl(_) | HoldStoreError::NotConfigured => None,
        }
    }

        pub(crate) fn breaker_signal(&self) -> BreakerSignal {
        match self {
            HoldStoreError::Unreachable(_) | HoldStoreError::Timeout(_) => BreakerSignal::Failure,
            HoldStoreError::Upstream {
                code: Some(code), ..
            } => code.breaker_signal(),
            HoldStoreError::Upstream { status, code: None } => {
                if *status >= 500 {
                    BreakerSignal::Failure
                } else {
                    BreakerSignal::Reachable
                }
            }
            HoldStoreError::Decode(_)
            | HoldStoreError::BadUrl(_)
            | HoldStoreError::NotConfigured => BreakerSignal::Neutral,
        }
    }
}

#[async_trait]
pub(crate) trait HoldStore: Send + Sync {
                        async fn active_holds(
        &self,
        user_ids: &[i64],
        labels: &[String],
        timeout_ms: u64,
    ) -> Result<Vec<Hold>, HoldStoreError>;

            async fn ping(&self, budget: Duration) -> Result<(), HoldStoreError>;

            fn endpoint(&self) -> String;
}

async fn spawn_probe(
    store: Arc<dyn HoldStore>,
    user_id: i64,
    labels: Vec<String>,
    timeout_ms: u64,
) -> Result<(Result<Vec<Hold>, HoldStoreError>, Duration), HoldStoreError> {
    use tracing::Instrument as _;
    let mut task = AbortOnDrop(tokio::spawn(
        async move {
            let started = Instant::now();
            let res = store.active_holds(&[user_id], &labels, timeout_ms).await;
            (res, started.elapsed())
        }
        .instrument(tracing::Span::current()),
    ));
    (&mut task.0)
        .await
        .map_err(|e| HoldStoreError::Unreachable(format!("probe task: {e}")))
}

struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

pub(crate) struct HttpHoldStore {
                ready: Result<HttpEndpoints, String>,
        base: String,
}

#[derive(Debug)]
struct HttpEndpoints {
    client: reqwest::Client,
    lookup_url: reqwest::Url,
    ready_url: reqwest::Url,
}

impl HttpHoldStore {
                            pub(crate) fn new(base: &str, env: Option<&str>) -> Self {
        let base = base.trim().trim_end_matches('/').to_owned();
        let ready = Self::endpoints(&base, env);
        if let Err(reason) = &ready {
            error!(
                reason,
                "overturn-hold gate: OVERTURN_HOLD_LEDGER_URL is unusable; every probe will be \
                 a `bad_url` probe error under on_probe_error (nothing will be sent)"
            );
        }
        Self { ready, base }
    }

    fn endpoints(base: &str, env: Option<&str>) -> Result<HttpEndpoints, String> {
        let base_url = reqwest::Url::parse(base).map_err(|e| format!("does not parse: {e}"))?;
        if !matches!(base_url.scheme(), "http" | "https") {
            return Err(format!(
                "scheme must be http or https, got `{}`",
                base_url.scheme()
            ));
        }
        if base_url.host_str().is_none_or(str::is_empty) {
            return Err("no host".to_owned());
        }
        let parse = |path: &str| -> Result<reqwest::Url, String> {
            reqwest::Url::parse(&format!("{base}{path}"))
                .map_err(|e| format!("does not parse: {e}"))
        };
        let lookup_url = parse(LOOKUP_PATH)?;
        let ready_url = parse(READY_PATH)?;
        let client = reqwest::Client::builder()
            .user_agent(user_agent(env))
            .connect_timeout(CONNECT_TIMEOUT)
            .tcp_keepalive(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        Ok(HttpEndpoints {
            client,
            lookup_url,
            ready_url,
        })
    }

    fn ready(&self) -> Result<&HttpEndpoints, HoldStoreError> {
        self.ready
            .as_ref()
            .map_err(|reason| HoldStoreError::BadUrl(reason.clone()))
    }
}

fn user_agent(env: Option<&str>) -> String {
    format!(
        "xai-abuse-enforcement-service/{} env={}",
        env!("CARGO_PKG_VERSION"),
        env.map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("unset")
    )
}

fn classify_reqwest_error(e: reqwest::Error, budget: Duration) -> HoldStoreError {
    if e.is_connect() {
        return HoldStoreError::Unreachable(e.to_string());
    }
    if e.is_timeout() {
        return HoldStoreError::Timeout(budget);
    }
    HoldStoreError::Unreachable(e.to_string())
}

enum BodyRead {
        Oversize(usize),
        Failed(reqwest::Error),
}

async fn read_body_capped(mut response: reqwest::Response) -> Result<Vec<u8>, BodyRead> {
    if let Some(len) = response.content_length()
        && len > MAX_BODY_BYTES as u64
    {
        return Err(BodyRead::Oversize(len as usize));
    }
    let mut body: Vec<u8> = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(BodyRead::Failed)? {
        if body.len() + chunk.len() > MAX_BODY_BYTES {
            return Err(BodyRead::Oversize(body.len() + chunk.len()));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[async_trait]
impl HoldStore for HttpHoldStore {
    async fn active_holds(
        &self,
        user_ids: &[i64],
        labels: &[String],
        timeout_ms: u64,
    ) -> Result<Vec<Hold>, HoldStoreError> {
        let ep = self.ready()?;
        let budget = Duration::from_millis(timeout_ms);
        let sent = ep
            .client
            .post(ep.lookup_url.clone())
            .timeout(budget)
            .header(DEADLINE_HEADER, timeout_ms)
            .json(&LookupRequest { user_ids, labels })
            .send()
            .await;
        let response = sent.map_err(|e| classify_reqwest_error(e, budget))?;
        let status = response.status().as_u16();
        let body = read_body_capped(response).await;
        if (200..300).contains(&status) {
            let body = match body {
                Ok(b) => b,
                Err(BodyRead::Oversize(n)) => {
                    return Err(HoldStoreError::Decode(format!(
                        "200 body of {n} bytes exceeds the {MAX_BODY_BYTES}-byte cap"
                    )));
                }
                Err(BodyRead::Failed(e)) => return Err(classify_reqwest_error(e, budget)),
            };
            let parsed: LookupResponse = serde_json::from_slice(&body)
                .map_err(|e| HoldStoreError::Decode(format!("200 body: {e}")))?;
            Ok(parsed.holds)
        } else {
            let code = match body {
                Ok(b) => serde_json::from_slice::<ErrorBody>(&b)
                    .ok()
                    .map(|e| LedgerCode::parse(&e.code)),
                Err(_) => None,
            };
            Err(HoldStoreError::Upstream { status, code })
        }
    }

    async fn ping(&self, budget: Duration) -> Result<(), HoldStoreError> {
        let ep = self.ready()?;
        let response = ep
            .client
            .get(ep.ready_url.clone())
            .timeout(budget)
            .send()
            .await
            .map_err(|e| classify_reqwest_error(e, budget))?;
        let status = response.status().as_u16();
        if (200..300).contains(&status) {
            return Ok(());
        }
        let code = match read_body_capped(response).await {
            Ok(b) => serde_json::from_slice::<ErrorBody>(&b)
                .ok()
                .map(|e| LedgerCode::parse(&e.code)),
            Err(_) => None,
        };
        Err(HoldStoreError::Upstream { status, code })
    }

    fn endpoint(&self) -> String {
        match &self.ready {
            Ok(_) => format!("url={}", self.base),
            Err(_) => "url=<bad URL>".to_owned(),
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GateVerdict {
        NotHeld,
        Held(Hold),
            HeldIgnoredByPolicy(Hold),
                ProbeError(ProbeError),
}

enum Lookup {
    Disabled,
                Holds(Vec<Hold>),
    ProbeError(ProbeError),
}

impl Lookup {
                #[cfg(test)]
    fn verdict(self, cfg: &HoldGateConfig, perm: bool) -> Option<GateVerdict> {
        match self {
            Lookup::Disabled => None,
            Lookup::ProbeError(e) => Some(GateVerdict::ProbeError(e)),
            Lookup::Holds(holds) => Some(suspend_verdict(&holds, cfg, perm)),
        }
    }
}

fn longest<'a>(it: impl Iterator<Item = &'a Hold>) -> Option<Hold> {
    it.max_by_key(|h| h.expires_at).cloned()
}

fn suspend_verdict(holds: &[Hold], cfg: &HoldGateConfig, perm: bool) -> GateVerdict {
    let suspends = || holds.iter().filter(|h| h.is_suspend());
    match longest(suspends().filter(|h| cfg.hold_applies(h, perm))) {
        Some(hold) => GateVerdict::Held(hold),
        None => match longest(suspends()) {
            Some(hold) => GateVerdict::HeldIgnoredByPolicy(hold),
            None => GateVerdict::NotHeld,
        },
    }
}

fn label_hold(holds: &[Hold], name: &str) -> Option<Hold> {
    longest(holds.iter().filter(|h| h.withholds_label(name)))
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GateOutcome {
            pub skip_status: Option<&'static str>,
            pub info: BTreeMap<String, String>,
                        pub strip_labels: Vec<String>,
                pub(crate) strip_hold_ids: Vec<i64>,
}

impl GateOutcome {
                                            pub fn mark_label_collapse(&mut self) {
        let (Some(label), Some(hold_id)) = (self.strip_labels.first(), self.strip_hold_ids.first())
        else {
            return;
        };
        self.info
            .insert("overturn_hold_kind".into(), HOLD_KIND_LABEL.into());
        self.info
            .insert("overturn_hold_id".into(), hold_id.to_string());
        self.info
            .insert("overturn_hold_label".into(), label.clone());
    }
}

struct CachedHolds {
    holds: Vec<Hold>,
                        probed_labels: Vec<String>,
                        ttl: Duration,
                                    written_at: Instant,
}

impl CachedHolds {
        fn covers(&self, labels: &[String]) -> bool {
        labels.iter().all(|l| self.probed_labels.contains(l))
    }
}

type PositiveCache = moka::sync::Cache<i64, Arc<CachedHolds>>;

struct HoldExpiry;

impl moka::Expiry<i64, Arc<CachedHolds>> for HoldExpiry {
    fn expire_after_create(
        &self,
        _key: &i64,
        value: &Arc<CachedHolds>,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }

    fn expire_after_update(
        &self,
        _key: &i64,
        value: &Arc<CachedHolds>,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

fn positive_cache() -> PositiveCache {
    moka::sync::Cache::builder()
        .max_capacity(CACHE_MAX_WEIGHT)
        .weigher(|_key: &i64, value: &Arc<CachedHolds>| {
            u32::try_from(value.holds.len()).unwrap_or(u32::MAX).max(1)
        })
        .expire_after(HoldExpiry)
        .build()
}

#[derive(Debug, Default)]
struct Breaker {
        consecutive_failures: u32,
                open_until: Option<Instant>,
                half_open: Option<HalfOpenSlot>,
                    trial_seq: u64,
                        generation: u64,
                    reachable_errors: u32,
                backoff_until: Option<Instant>,
        last_reachable_error: Option<ProbeErrorCode>,
}

#[derive(Debug)]
struct HalfOpenSlot {
    seq: u64,
    since: Instant,
}

struct HalfOpenTrial<'a> {
    gate: &'a HoldGate,
    seq: u64,
    armed: bool,
}

impl Drop for HalfOpenTrial<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut b = self.gate.breaker.lock().unwrap_or_else(|p| p.into_inner());
        if b.half_open.as_ref().is_some_and(|s| s.seq == self.seq) {
            b.half_open = None;
            warn!(
                trial = self.seq,
                "overturn-hold gate: half-open trial dropped before it reported; slot freed for \
                 the next trial"
            );
        }
    }
}

enum BreakerDecision<'a> {
                        Probe {
        trial: Option<HalfOpenTrial<'a>>,
        generation: u64,
    },
                Skip(ProbeError),
}

pub(crate) trait Clock: Send + Sync {
            fn now(&self) -> Instant;
        fn now_utc(&self) -> DateTime<Utc>;
}

struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn now_utc(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

#[cfg(test)]
pub(crate) struct ManualClock {
    now: Mutex<(Instant, DateTime<Utc>)>,
}

#[cfg(test)]
impl ManualClock {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            now: Mutex::new((Instant::now(), Utc::now())),
        })
    }

        pub(crate) fn advance(&self, d: Duration) {
        let mut now = self.now.lock().unwrap_or_else(|p| p.into_inner());
        now.0 += d;
        now.1 += chrono::Duration::from_std(d).expect("test durations are small");
    }
}

#[cfg(test)]
impl Clock for ManualClock {
    fn now(&self) -> Instant {
        self.now.lock().unwrap_or_else(|p| p.into_inner()).0
    }

    fn now_utc(&self) -> DateTime<Utc> {
        self.now.lock().unwrap_or_else(|p| p.into_inner()).1
    }
}

pub struct HoldGate {
    store: Option<Arc<dyn HoldStore>>,
    clock: Arc<dyn Clock>,
        cache: PositiveCache,
            env: Option<String>,
    no_store_logged: AtomicBool,
    breaker: Mutex<Breaker>,
    breaker_failures: u32,
    breaker_open: Duration,
                                        probe_failures: AtomicU64,
                    expired_holds_seen: AtomicU64,
                    last_published: Mutex<Option<HoldGateConfig>>,
}

const PROBE_FAILURE_LOG_EVERY: u64 = 1_000;

fn sampled(counter: &AtomicU64) -> (u64, bool) {
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    (n, n == 1 || n.is_multiple_of(PROBE_FAILURE_LOG_EVERY))
}

fn config_info_key(cfg: &HoldGateConfig) -> (&'static str, &'static str, String) {
    (
        cfg.mode.as_str(),
        cfg.on_probe_error.as_str(),
        cfg.kinds
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>()
            .join(","),
    )
}

impl HoldGate {
                pub(crate) fn new(store: Option<Arc<dyn HoldStore>>) -> Self {
        Self::with_breaker_and_clock(
            store,
            BREAKER_FAILURES,
            DEFAULT_BREAKER_OPEN,
            Arc::new(SystemClock),
        )
    }

        #[cfg(test)]
    pub(crate) fn with_breaker(
        store: Option<Arc<dyn HoldStore>>,
        breaker_failures: u32,
        breaker_open: Duration,
    ) -> Self {
        Self::with_breaker_and_clock(store, breaker_failures, breaker_open, Arc::new(SystemClock))
    }

        #[cfg(test)]
    pub(crate) fn with_clock(store: Option<Arc<dyn HoldStore>>, clock: Arc<dyn Clock>) -> Self {
        Self::with_breaker_and_clock(store, BREAKER_FAILURES, DEFAULT_BREAKER_OPEN, clock)
    }

        fn with_breaker_and_clock(
        store: Option<Arc<dyn HoldStore>>,
        breaker_failures: u32,
        breaker_open: Duration,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            store,
            clock,
            cache: positive_cache(),
            env: None,
            no_store_logged: AtomicBool::new(false),
            breaker: Mutex::new(Breaker::default()),
            breaker_failures: breaker_failures.max(1),
            breaker_open,
            probe_failures: AtomicU64::new(0),
            expired_holds_seen: AtomicU64::new(0),
            last_published: Mutex::new(None),
        }
    }

                                            pub fn publish_config(&self, cfg: &HoldGateConfig) {
        self.publish_config_changed(cfg);
    }

            fn publish_config_changed(&self, cfg: &HoldGateConfig) -> bool {
        for mode in [GateMode::Off, GateMode::Shadow, GateMode::Enforce] {
            metrics::HOLD_GATE_MODE
                .with_label_values(&[mode.as_str()])
                .set(i64::from(mode == cfg.mode));
        }
        let previous = {
            let mut last = self
                .last_published
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            last.replace(cfg.clone())
        };
        let previous_key = previous.as_ref().map(config_info_key);
        let current_key = config_info_key(cfg);
        if previous_key.as_ref() != Some(&current_key) {
            if let Some((old_mode, old_policy, old_kinds)) = previous_key {
                metrics::HOLD_GATE_CONFIG_INFO
                    .with_label_values(&[old_mode, old_policy, &old_kinds])
                    .set(0);
            }
            metrics::HOLD_GATE_CONFIG_INFO
                .with_label_values(&[current_key.0, current_key.1, &current_key.2])
                .set(1);
        }
        let changed = previous.as_ref() != Some(cfg);
        if changed {
            info!(
                old_mode = previous
                    .as_ref()
                    .map(|p| p.mode.as_str())
                    .unwrap_or("<boot>"),
                mode = cfg.mode.as_str(),
                on_probe_error = cfg.on_probe_error.as_str(),
                topics = ?cfg.topics,
                kinds = %current_key.2,
                labels = ?cfg.labels,
                perm_suspend_case_groups = ?cfg.perm_suspend_case_groups,
                timeout_ms = cfg.timeout_ms,
                positive_cache_secs = cfg.positive_cache_secs,
                "overturn-hold gate config changed"
            );
        }
        changed
    }

        #[cfg(test)]
    fn breaker_open(&self) -> bool {
        let b = self.breaker.lock().unwrap_or_else(|p| p.into_inner());
        b.open_until.is_some()
    }

        #[cfg(test)]
    fn half_open_inflight(&self) -> bool {
        let b = self.breaker.lock().unwrap_or_else(|p| p.into_inner());
        b.half_open.is_some()
    }

        #[cfg(test)]
    fn breaker_snapshot(&self) -> (u32, Option<Instant>, u64) {
        let b = self.breaker.lock().unwrap_or_else(|p| p.into_inner());
        (b.consecutive_failures, b.open_until, b.generation)
    }

            #[cfg(test)]
    fn probe_failures(&self) -> u64 {
        self.probe_failures.load(Ordering::Relaxed)
    }

        #[cfg(test)]
    fn backoff_snapshot(&self) -> (u32, bool) {
        let b = self.breaker.lock().unwrap_or_else(|p| p.into_inner());
        (b.reachable_errors, b.backoff_until.is_some())
    }

            fn breaker_before_probe(&self, timeout_ms: u64) -> BreakerDecision<'_> {
        let mut b = self.breaker.lock().unwrap_or_else(|p| p.into_inner());
        let generation = b.generation;
        let Some(until) = b.open_until else {
            if let Some(backoff_until) = b.backoff_until {
                let now = self.clock.now();
                if now < backoff_until {
                    return BreakerDecision::Skip(ProbeError::reachable_backoff(
                        backoff_until.duration_since(now).as_secs().max(1),
                        b.reachable_errors,
                        b.last_reachable_error,
                    ));
                }
                b.backoff_until = None;
            }
            return BreakerDecision::Probe {
                trial: None,
                generation,
            };
        };
        let now = self.clock.now();
        if now < until {
            return BreakerDecision::Skip(ProbeError::breaker_open(
                until.duration_since(now).as_secs().max(1),
            ));
        }
        if let Some(slot) = b.half_open.as_ref() {
            let age = now.saturating_duration_since(slot.since);
            if age < half_open_stale_after(timeout_ms) {
                return BreakerDecision::Skip(ProbeError::breaker_open(1));
            }
            warn!(
                trial = slot.seq,
                age_ms = age.as_millis() as u64,
                "overturn-hold gate: half-open trial slot is stale; starting a new trial"
            );
        }
        b.trial_seq = b.trial_seq.wrapping_add(1);
        let seq = b.trial_seq;
        b.half_open = Some(HalfOpenSlot { seq, since: now });
        BreakerDecision::Probe {
            trial: Some(HalfOpenTrial {
                gate: self,
                seq,
                armed: true,
            }),
            generation,
        }
    }

                        fn breaker_after_probe(
        &self,
        trial: Option<HalfOpenTrial<'_>>,
        generation: u64,
        signal: BreakerSignal,
        error: Option<ProbeErrorCode>,
    ) {
        let trial_seq = trial.map(|mut t| {
            t.armed = false;
            t.seq
        });
        let mut b = self.breaker.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(seq) = trial_seq
            && b.half_open.as_ref().is_some_and(|s| s.seq == seq)
        {
            b.half_open = None;
        }
        if generation != b.generation {
            debug!(
                probe_generation = generation,
                breaker_generation = b.generation,
                ?signal,
                "overturn-hold gate: ignoring a probe result from before the breaker opened"
            );
            return;
        }
        let was_open = b.open_until.is_some();
        match signal {
            BreakerSignal::Reachable => {
                if was_open {
                    info!("overturn-hold gate: ledger reachable again; breaker closed");
                }
                *b = Breaker {
                    trial_seq: b.trial_seq,
                    generation: b.generation,
                    reachable_errors: b.reachable_errors,
                    backoff_until: b.backoff_until,
                    last_reachable_error: b.last_reachable_error,
                    ..Breaker::default()
                };
                metrics::HOLD_GATE_BREAKER_OPEN.set(0);
                match error {
                    None => {
                        if b.reachable_errors >= REACHABLE_ERRORS_BEFORE_BACKOFF {
                            info!(
                                reachable_errors = b.reachable_errors,
                                "overturn-hold gate: ledger service serving again; reachable-error \
                                 backoff disarmed"
                            );
                        }
                        b.reachable_errors = 0;
                        b.backoff_until = None;
                        b.last_reachable_error = None;
                    }
                    Some(code) => {
                        b.reachable_errors = b.reachable_errors.saturating_add(1);
                        b.last_reachable_error = Some(code);
                        if b.reachable_errors >= REACHABLE_ERRORS_BEFORE_BACKOFF {
                            b.backoff_until = Some(self.clock.now() + REACHABLE_ERROR_BACKOFF);
                            if b.reachable_errors == REACHABLE_ERRORS_BEFORE_BACKOFF {
                                warn!(
                                    reachable_errors = b.reachable_errors,
                                    code = code.as_str(),
                                    backoff_secs = REACHABLE_ERROR_BACKOFF.as_secs(),
                                    "overturn-hold gate: the ledger service answers but cannot \
                                     serve (persistent reachable error, e.g. wrong path or \
                                     missing grant); probes skipped for {}s at a time and counted \
                                     as probe errors (code reachable_backoff) until a probe \
                                     succeeds — the breaker stays closed",
                                    REACHABLE_ERROR_BACKOFF.as_secs()
                                );
                            } else {
                                debug!(
                                    reachable_errors = b.reachable_errors,
                                    code = code.as_str(),
                                    "overturn-hold gate: reachable-error backoff re-armed"
                                );
                            }
                        }
                    }
                }
            }
            BreakerSignal::Neutral => {
            }
            BreakerSignal::Failure => {
                b.reachable_errors = 0;
                b.backoff_until = None;
                b.last_reachable_error = None;
                b.consecutive_failures = b.consecutive_failures.saturating_add(1);
                if was_open || b.consecutive_failures >= self.breaker_failures {
                    b.open_until = Some(self.clock.now() + self.breaker_open);
                    b.generation = b.generation.wrapping_add(1);
                    metrics::HOLD_GATE_BREAKER_OPEN.set(1);
                    warn!(
                        consecutive_failures = b.consecutive_failures,
                        open_secs = self.breaker_open.as_secs(),
                        generation = b.generation,
                        "overturn-hold gate: ledger unreachable; breaker {} — probes skipped and \
                         counted as probe errors until the next trial",
                        if was_open { "stays open" } else { "opened" }
                    );
                }
            }
        }
    }

                                        pub fn from_url(url: Option<&str>, startup_probe: bool, env: Option<&str>) -> Self {
        let env = env
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        let mut gate = match url.map(str::trim).filter(|s| !s.is_empty()) {
            Some(url) => Self::new(Some(Arc::new(HttpHoldStore::new(url, env.as_deref())))),
            None => Self::new(None),
        };
        gate.env = env;
        info!(
            env = gate.env.as_deref().unwrap_or("<unset>"),
            store = %gate
                .store
                .as_ref()
                .map(|s| s.endpoint())
                .unwrap_or_else(|| "<none: OVERTURN_HOLD_LEDGER_URL unset>".to_owned()),
            startup_probe,
            connect_timeout_ms = CONNECT_TIMEOUT.as_millis() as u64,
            breaker = %format!(
                "{}/{}s",
                gate.breaker_failures,
                gate.breaker_open.as_secs()
            ),
            "overturn-hold gate: built (no request until `overturn_hold_gate.mode` is \
             shadow/enforce; without a store every probe would be a probe error under \
             on_probe_error)"
        );
        gate
    }

        #[cfg(test)]
    fn has_store(&self) -> bool {
        self.store.is_some()
    }

                                        pub async fn startup_probe(&self) -> bool {
        let Some(store) = self.store.as_ref() else {
            warn!(
                result = "fail",
                endpoint = "url=-",
                ms = 0u64,
                "hold_gate_probe fail: OVERTURN_HOLD_STARTUP_PROBE set but OVERTURN_HOLD_LEDGER_URL is unset"
            );
            metrics::HOLD_GATE_DB_CONNECTED.set(0);
            return false;
        };
        let endpoint = store.endpoint();
        let started = Instant::now();
        let res = store.ping(STARTUP_PROBE_TIMEOUT).await;
        let ms = started.elapsed().as_millis() as u64;
        match res {
            Ok(()) => {
                metrics::HOLD_GATE_DB_CONNECTED.set(1);
                info!(result = "ok", endpoint = %endpoint, ms, "hold_gate_probe ok");
                true
            }
            Err(e) => {
                metrics::HOLD_GATE_DB_CONNECTED.set(0);
                warn!(result = "fail", endpoint = %endpoint, ms, error = %e, "hold_gate_probe fail");
                false
            }
        }
    }

                    #[cfg(test)]
    async fn check(&self, cfg: &HoldGateConfig, user_id: i64, topic: &str) -> Option<GateVerdict> {
        self.check_timed(cfg, user_id, topic, &[])
            .await
            .0
            .verdict(cfg, false)
    }

                    async fn check_timed(
        &self,
        cfg: &HoldGateConfig,
        user_id: i64,
        topic: &str,
        labels: &[String],
    ) -> (Lookup, u64, bool) {
        if !cfg.enabled() || !cfg.covers_topic(topic) {
            return (Lookup::Disabled, 0, false);
        }
        if let Some(holds) = self.cached(user_id, labels) {
            return (Lookup::Holds(holds), 0, true);
        }
        let Some(store) = self.store.as_ref() else {
            if !self.no_store_logged.swap(true, Ordering::Relaxed) {
                error!(
                    mode = cfg.mode.as_str(),
                    "overturn-hold gate is {} but OVERTURN_HOLD_LEDGER_URL is unset: every probe \
                     fails and follows on_probe_error={} (logged once)",
                    cfg.mode.as_str(),
                    cfg.on_probe_error.as_str()
                );
            }
            return (
                Lookup::ProbeError(ProbeError::from(&HoldStoreError::NotConfigured)),
                0,
                false,
            );
        };

        let (trial, generation) = match self.breaker_before_probe(cfg.timeout_ms) {
            BreakerDecision::Probe { trial, generation } => (trial, generation),
            BreakerDecision::Skip(err) => {
                match err.code {
                    ProbeErrorCode::ReachableBackoff => {
                        metrics::HOLD_GATE_BACKOFF_SKIP_TOTAL.inc();
                    }
                    _ => metrics::HOLD_GATE_BREAKER_SKIP_TOTAL.inc(),
                }
                return (Lookup::ProbeError(err), 0, false);
            }
        };

        let probe_started = self.clock.now();
        let started = Instant::now();
        let (res, elapsed) =
            match spawn_probe(Arc::clone(store), user_id, labels.to_vec(), cfg.timeout_ms).await {
                Ok((res, elapsed)) => (res, elapsed),
                Err(e) => (Err(e), started.elapsed()),
            };
        metrics::HOLD_GATE_PROBE_SECONDS.observe(elapsed.as_secs_f64());
        metrics::HOLD_GATE_PROBE_SCHED_DELAY_SECONDS
            .observe(started.elapsed().saturating_sub(elapsed).as_secs_f64());
        let probe_ms = elapsed.as_millis() as u64;
        let answered = match &res {
            Ok(_) => Some(true),
            Err(e) => e.ledger_answered(),
        };
        if let Some(answered) = answered {
            metrics::HOLD_GATE_DB_CONNECTED.set(i64::from(answered));
        }
        let (signal, error) = match &res {
            Ok(_) => (BreakerSignal::Reachable, None),
            Err(e) => (e.breaker_signal(), Some(e.code())),
        };
        if error.is_none() {
            self.probe_failures.store(0, Ordering::Relaxed);
        }
        self.breaker_after_probe(trial, generation, signal, error);

        match res {
            Ok(holds) => {
                let holds = self.drop_expired(
                    user_id,
                    holds.into_iter().filter(|h| h.user_id == user_id).collect(),
                );
                if holds.is_empty() {
                    self.forget_if_older(user_id, probe_started);
                } else {
                    self.remember(
                        user_id,
                        holds.clone(),
                        labels,
                        cfg.positive_cache_secs,
                        probe_started,
                    );
                }
                (Lookup::Holds(holds), probe_ms, false)
            }
            Err(e) => (Lookup::ProbeError(ProbeError::from(&e)), probe_ms, false),
        }
    }

                                                                                                pub async fn evaluate(
        &self,
        cfg: &HoldGateConfig,
        user_id: i64,
        topic: &str,
        actions: &[GatedAction],
    ) -> GateOutcome {
        let mut out = GateOutcome::default();
        if actions.is_empty() {
            return out;
        }
        let suspend_perm = actions.iter().find_map(|a| match a {
            GatedAction::Suspend { perm } => Some(*perm),
            GatedAction::Label { .. } => None,
        });
        let mut labels: Vec<String> = actions
            .iter()
            .filter_map(|a| match a {
                GatedAction::Label { name } => Some(name.clone()),
                GatedAction::Suspend { .. } => None,
            })
            .collect();
        labels.sort();
        labels.dedup();

        let (lookup, probe_ms, cache_hit) = self.check_timed(cfg, user_id, topic, &labels).await;
        let mode = cfg.mode.as_str();
        let (verdict, holds) = match lookup {
            Lookup::Disabled => return out,
            Lookup::ProbeError(e) => (GateVerdict::ProbeError(e), Vec::new()),
            Lookup::Holds(holds) => {
                let verdict = match suspend_perm {
                    Some(perm) => suspend_verdict(&holds, cfg, perm),
                    None => GateVerdict::NotHeld,
                };
                (verdict, holds)
            }
        };

        if let Some(env) = &self.env {
            out.info.insert("overturn_hold_env".into(), env.clone());
        }
        if cache_hit {
            metrics::HOLD_GATE_TOTAL
                .with_label_values(&[mode, "cache_hit", topic])
                .inc();
        }
        out.info
            .insert("overturn_hold_mode".into(), mode.to_owned());
        out.info
            .insert("overturn_hold_probe_ms".into(), probe_ms.to_string());

        let mut disposition_emitted = true;
        match verdict {
            GateVerdict::NotHeld => disposition_emitted = false,
            GateVerdict::HeldIgnoredByPolicy(hold) => {
                out.info.insert(
                    "overturn_hold_ignored_case_group_id".into(),
                    hold.case_group_id
                        .map_or_else(|| "none".to_owned(), |cg| cg.to_string()),
                );
                out.info.insert(
                    "overturn_hold_ignored_hold_id".into(),
                    hold.hold_id.to_string(),
                );
                metrics::HOLD_GATE_TOTAL
                    .with_label_values(&[mode, "held_ignored_policy", topic])
                    .inc();
                info!(
                    user_id,
                    topic,
                    hold_id = hold.hold_id,
                    case_group_id = ?hold.case_group_id,
                    reason = %hold.reason,
                    perm_suspend_case_groups = ?cfg.perm_suspend_case_groups,
                    probe_ms,
                    cache_hit,
                    "overturn-hold gate: active suspend hold ignored for a permanent suspend \
                     (case group outside perm_suspend_case_groups); acting"
                );
            }
            GateVerdict::Held(hold) => {
                out.info
                    .insert("overturn_hold_id".into(), hold.hold_id.to_string());
                out.info.insert(
                    "overturn_hold_head".into(),
                    hold.head.clone().unwrap_or_default(),
                );
                out.info.insert(
                    "overturn_hold_expires_at".into(),
                    hold.expires_at.to_rfc3339(),
                );
                if let Some(cg) = hold.case_group_id {
                    out.info
                        .insert("overturn_hold_case_group_id".into(), cg.to_string());
                }
                match cfg.mode {
                    GateMode::Shadow => {
                        out.info
                            .insert("overturn_hold_would_block".into(), "true".into());
                        metrics::HOLD_GATE_TOTAL
                            .with_label_values(&[mode, "held_shadow", topic])
                            .inc();
                        info!(
                            user_id,
                            topic,
                            hold_id = hold.hold_id,
                            head = hold.head.as_deref().unwrap_or(""),
                            expires_at = %hold.expires_at,
                            probe_ms,
                            cache_hit,
                            "overturn-hold gate (shadow): active suspend hold; would block, acting as today"
                        );
                    }
                    GateMode::Enforce => {
                        out.skip_status = Some(STATUS_HOLD_OVERTURNED);
                        out.info
                            .insert("overturn_hold_kind".into(), HOLD_KIND_SUSPEND.into());
                        metrics::HOLD_GATE_TOTAL
                            .with_label_values(&[mode, "held_blocked", topic])
                            .inc();
                        info!(
                            user_id,
                            topic,
                            hold_id = hold.hold_id,
                            head = hold.head.as_deref().unwrap_or(""),
                            expires_at = %hold.expires_at,
                            probe_ms,
                            cache_hit,
                            "overturn-hold gate (enforce): active suspend hold; skipping decision"
                        );
                    }
                    GateMode::Off => unreachable!("Off returns Disabled above"),
                }
            }
            GateVerdict::ProbeError(err) => {
                out.info.insert(
                    "overturn_hold_probe_error".into(),
                    err.code.as_str().to_owned(),
                );
                let fail_closed =
                    cfg.mode == GateMode::Enforce && cfg.on_probe_error == OnProbeError::Skip;
                let outcome = if fail_closed {
                    out.skip_status = Some(STATUS_HOLD_LOOKUP_FAILED);
                    "probe_error_skip"
                } else {
                    "probe_error_open"
                };
                metrics::HOLD_GATE_TOTAL
                    .with_label_values(&[mode, outcome, topic])
                    .inc();
                let disposition = if fail_closed {
                    "skipping decision"
                } else {
                    "fail-open, acting"
                };
                if matches!(
                    err.code,
                    ProbeErrorCode::BreakerOpen | ProbeErrorCode::ReachableBackoff
                ) {
                    debug!(
                        user_id,
                        topic,
                        mode,
                        on_probe_error = cfg.on_probe_error.as_str(),
                        code = err.code.as_str(),
                        "overturn-hold gate: probe skipped ({disposition})"
                    );
                } else {
                    let (n, log) = sampled(&self.probe_failures);
                    if log {
                        warn!(
                            user_id,
                            topic,
                            mode,
                            on_probe_error = cfg.on_probe_error.as_str(),
                            probe_ms,
                            code = err.code.as_str(),
                            error = %err.detail,
                            failures_so_far = n,
                            "overturn-hold gate: probe failed ({disposition}); logged for the \
                             first and every {PROBE_FAILURE_LOG_EVERY}th failure of this run"
                        );
                    } else {
                        debug!(
                            user_id,
                            topic,
                            mode,
                            probe_ms,
                            code = err.code.as_str(),
                            error = %err.detail,
                            "overturn-hold gate: probe failed ({disposition})"
                        );
                    }
                }
            }
        }

        if out.skip_status.is_none() && !labels.is_empty() {
            let held: Vec<(&String, Hold)> = labels
                .iter()
                .filter_map(|name| label_hold(&holds, name).map(|h| (name, h)))
                .collect();
            if !held.is_empty() {
                disposition_emitted = true;
                let names = held
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                let hold_ids = held
                    .iter()
                    .map(|(_, h)| h.hold_id.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                out.info
                    .insert("overturn_hold_label_hold_id".into(), hold_ids.clone());
                match cfg.mode {
                    GateMode::Shadow => {
                        out.info
                            .insert("overturn_hold_label_would_strip".into(), names.clone());
                        metrics::HOLD_GATE_TOTAL
                            .with_label_values(&[mode, "held_label_shadow", topic])
                            .inc();
                        info!(
                            user_id,
                            topic,
                            labels = %names,
                            hold_ids = %hold_ids,
                            probe_ms,
                            cache_hit,
                            "overturn-hold gate (shadow): active label hold(s); would strip, acting as today"
                        );
                    }
                    GateMode::Enforce => {
                        out.info
                            .insert("overturn_hold_labels_stripped".into(), names.clone());
                        out.strip_hold_ids = held.iter().map(|(_, h)| h.hold_id).collect();
                        out.strip_labels = held.into_iter().map(|(n, _)| n.clone()).collect();
                        metrics::HOLD_GATE_TOTAL
                            .with_label_values(&[mode, "held_label_stripped", topic])
                            .inc();
                        info!(
                            user_id,
                            topic,
                            labels = %names,
                            hold_ids = %hold_ids,
                            probe_ms,
                            cache_hit,
                            "overturn-hold gate (enforce): active label hold(s); stripping the label(s), \
                             remaining actions proceed"
                        );
                    }
                    GateMode::Off => unreachable!("Off returns Disabled above"),
                }
            }
        }

        if !disposition_emitted {
            metrics::HOLD_GATE_TOTAL
                .with_label_values(&[mode, "not_held", topic])
                .inc();
        }
        out
    }

                            fn drop_expired(&self, user_id: i64, holds: Vec<Hold>) -> Vec<Hold> {
        let now_utc = self.clock.now_utc();
        if holds.iter().all(|h| h.expires_at > now_utc) {
            return holds;
        }
        let (live, expired): (Vec<Hold>, Vec<Hold>) =
            holds.into_iter().partition(|h| h.expires_at > now_utc);
        metrics::HOLD_GATE_EXPIRED_HOLD_TOTAL.inc_by(expired.len() as u64);
        for hold in &expired {
            let (n, log) = sampled(&self.expired_holds_seen);
            if log {
                warn!(
                    user_id,
                    hold_id = hold.hold_id,
                    action_kind = %hold.action_kind,
                    expires_at = %hold.expires_at,
                    now = %now_utc,
                    expired_holds_so_far = n,
                    "overturn-hold gate: the ledger service returned a hold that is already \
                     past expires_at; dropped (clock skew or a ledger-service bug — it filters \
                     on expires_at > now()); logged for the first and every \
                     {PROBE_FAILURE_LOG_EVERY}th"
                );
            } else {
                debug!(
                    user_id,
                    hold_id = hold.hold_id,
                    expires_at = %hold.expires_at,
                    "overturn-hold gate: expired hold in a fresh probe result; dropped"
                );
            }
        }
        live
    }

                                    fn cached(&self, user_id: i64, labels: &[String]) -> Option<Vec<Hold>> {
        let c = self.cache.get(&user_id)?;
        if !c.covers(labels) {
            return None;
        }
        let now_utc = self.clock.now_utc();
        let live: Vec<Hold> = c
            .holds
            .iter()
            .filter(|h| h.expires_at > now_utc)
            .cloned()
            .collect();
        if live.is_empty() {
            self.cache.invalidate(&user_id);
            return None;
        }
        Some(live)
    }

                                                                                fn remember(
        &self,
        user_id: i64,
        holds: Vec<Hold>,
        probed_labels: &[String],
        ttl_secs: u64,
        probe_started: Instant,
    ) {
        if ttl_secs == 0 {
            return;
        }
        let Some(earliest) = holds.iter().map(|h| h.expires_at).min() else {
            return;
        };
        let remaining = (earliest - self.clock.now_utc())
            .to_std()
            .unwrap_or(Duration::ZERO);
        let ttl = Duration::from_secs(ttl_secs).min(remaining);
        if ttl.is_zero() {
            return;
        }
        let entry = Arc::new(CachedHolds {
            holds,
            probed_labels: probed_labels.to_vec(),
            ttl,
            written_at: self.clock.now(),
        });
        self.cache
            .entry(user_id)
            .and_compute_with(|current| match current {
                Some(cur) if cur.value().written_at > probe_started => Op::Nop,
                _ => Op::Put(entry),
            });
    }

                                fn forget_if_older(&self, user_id: i64, probe_started: Instant) {
        self.cache
            .entry(user_id)
            .and_compute_with(|current| match current {
                Some(cur) if cur.value().written_at <= probe_started => Op::Remove,
                _ => Op::Nop,
            });
    }

        #[cfg(test)]
    fn cached_snapshot(&self, user_id: i64) -> Option<(Vec<i64>, Vec<String>)> {
        self.cache.get(&user_id).map(|c| {
            (
                c.holds.iter().map(|h| h.hold_id).collect(),
                c.probed_labels.clone(),
            )
        })
    }

        #[cfg(test)]
    fn cached_ttl(&self, user_id: i64) -> Option<Duration> {
        self.cache.get(&user_id).map(|c| c.ttl)
    }

                #[cfg(test)]
    fn cached_len(&self) -> usize {
        self.cache.run_pending_tasks();
        let now_utc = self.clock.now_utc();
        self.cache
            .iter()
            .filter(|(_, c)| c.holds.iter().any(|h| h.expires_at > now_utc))
            .count()
    }
}


#[cfg(test)]
pub(crate) struct FakeHoldStore {
    holds: Mutex<Vec<Hold>>,
        fail: Mutex<FakeFailure>,
                hang: AtomicBool,
                    hang_only: Mutex<Option<i64>>,
    release: tokio::sync::Notify,
    calls: std::sync::atomic::AtomicUsize,
    pings: std::sync::atomic::AtomicUsize,
            label_requests: Mutex<Vec<Vec<String>>>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum FakeFailure {
    #[default]
    None,
            Connect,
            Unreachable,
            SqlPermission,
        SqlStatementTimeout,
            Decode,
            Saturated,
            NotFound,
}

#[cfg(test)]
impl FakeHoldStore {
    pub fn new(holds: Vec<Hold>) -> Arc<Self> {
        Arc::new(Self {
            holds: Mutex::new(holds),
            fail: Mutex::new(FakeFailure::None),
            hang: AtomicBool::new(false),
            hang_only: Mutex::new(None),
            release: tokio::sync::Notify::new(),
            calls: std::sync::atomic::AtomicUsize::new(0),
            pings: std::sync::atomic::AtomicUsize::new(0),
            label_requests: Mutex::new(Vec::new()),
        })
    }

        pub fn set_holds(&self, holds: Vec<Hold>) {
        *self.holds.lock().unwrap_or_else(|p| p.into_inner()) = holds;
    }

        pub fn label_requests(&self) -> Vec<Vec<String>> {
        self.label_requests
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    pub fn set_failure(&self, f: FakeFailure) {
        *self.fail.lock().unwrap_or_else(|p| p.into_inner()) = f;
    }

        pub fn set_fail(&self, fail: bool) {
        self.set_failure(if fail {
            FakeFailure::Connect
        } else {
            FakeFailure::None
        });
    }

        pub fn set_fail_query(&self, fail: bool) {
        self.set_failure(if fail {
            FakeFailure::SqlPermission
        } else {
            FakeFailure::None
        });
    }

            pub fn set_hang(&self, hang: bool) {
        self.hang.store(hang, Ordering::SeqCst);
        if !hang {
            self.release.notify_waiters();
        }
    }

            pub fn set_hang_only(&self, user_id: Option<i64>) {
        *self.hang_only.lock().unwrap_or_else(|p| p.into_inner()) = user_id;
    }

    fn failure(&self) -> Option<HoldStoreError> {
        match *self.fail.lock().unwrap_or_else(|p| p.into_inner()) {
            FakeFailure::None => None,
            FakeFailure::Connect => Some(HoldStoreError::Upstream {
                status: 503,
                code: Some(LedgerCode::ConnectRefused),
            }),
            FakeFailure::Unreachable => Some(HoldStoreError::Unreachable(
                "fake: connection refused".into(),
            )),
            FakeFailure::SqlPermission => Some(HoldStoreError::Upstream {
                status: 500,
                code: Some(LedgerCode::Other),
            }),
            FakeFailure::SqlStatementTimeout => Some(HoldStoreError::Upstream {
                status: 504,
                code: Some(LedgerCode::StatementTimeout),
            }),
            FakeFailure::Decode => Some(HoldStoreError::Decode(
                "fake: missing field `hold_id`".into(),
            )),
            FakeFailure::Saturated => Some(HoldStoreError::Upstream {
                status: 503,
                code: Some(LedgerCode::PoolWait),
            }),
            FakeFailure::NotFound => Some(HoldStoreError::Upstream {
                status: 404,
                code: None,
            }),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn pings(&self) -> usize {
        self.pings.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
#[async_trait]
impl HoldStore for FakeHoldStore {
    async fn active_holds(
        &self,
        user_ids: &[i64],
        labels: &[String],
        _timeout_ms: u64,
    ) -> Result<Vec<Hold>, HoldStoreError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.label_requests
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(labels.to_vec());
        let parks = self
            .hang_only
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_none_or(|only| user_ids.contains(&only));
        while parks && self.hang.load(Ordering::SeqCst) {
            let mut released = std::pin::pin!(self.release.notified());
            released.as_mut().enable();
            if !self.hang.load(Ordering::SeqCst) {
                break;
            }
            released.await;
        }
        if let Some(e) = self.failure() {
            return Err(e);
        }
        let holds = self.holds.lock().unwrap_or_else(|p| p.into_inner());
        Ok(holds
            .iter()
            .filter(|h| user_ids.contains(&h.user_id))
            .filter(|h| {
                h.is_suspend()
                    || (h.action_kind == HOLD_KIND_LABEL
                        && h.label.as_ref().is_some_and(|l| labels.contains(l)))
            })
            .cloned()
            .collect())
    }

    async fn ping(&self, _budget: Duration) -> Result<(), HoldStoreError> {
        self.pings.fetch_add(1, Ordering::SeqCst);
        if let Some(e) = self.failure() {
            return Err(e);
        }
        Ok(())
    }

    fn endpoint(&self) -> String {
        "url=fake".to_owned()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

        fn hold(user_id: i64, hold_id: i64, days: i64) -> Hold {
        Hold {
            user_id,
            hold_id,
            head: Some("FollowBot".into()),
            expires_at: Utc::now() + chrono::Duration::days(days),
            case_group_id: Some(63),
            reason: "appeal_overturned".into(),
            action_kind: HOLD_KIND_SUSPEND.into(),
            label: None,
        }
    }

        fn label_hold_row(user_id: i64, hold_id: i64, days: i64, label: &str) -> Hold {
        Hold {
            head: Some("SpamEmbeddingMajorityPoster".into()),
            case_group_id: Some(64),
            action_kind: HOLD_KIND_LABEL.into(),
            label: Some(label.into()),
            ..hold(user_id, hold_id, days)
        }
    }

    fn cfg(mode: GateMode) -> HoldGateConfig {
        HoldGateConfig {
            mode,
            ..HoldGateConfig::default()
        }
    }

        fn cfg_labels(mode: GateMode, labels: &[&str]) -> HoldGateConfig {
        HoldGateConfig {
            mode,
            kinds: vec![GateKind::Suspend, GateKind::Label],
            labels: labels.iter().map(|s| (*s).to_owned()).collect(),
            ..HoldGateConfig::default()
        }
    }

    fn counter(mode: &str, outcome: &str) -> u64 {
        metrics::HOLD_GATE_TOTAL
            .with_label_values(&[mode, outcome, TOPIC])
            .get()
    }

    const TOPIC: &str = "abuse.embeddings.user_decisions";
    const TEMP: GatedAction = GatedAction::TEMPORARY_SUSPEND;
    const PERM: GatedAction = GatedAction::PERMANENT_SUSPEND;
    const SHR: &str = "SpamHighRecall";

    fn label_action(name: &str) -> GatedAction {
        GatedAction::Label { name: name.into() }
    }

                    pub(crate) static METRICS_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());


    #[test]
    fn missing_key_is_off() {
        assert_eq!(HoldGateConfig::from_config(None), HoldGateConfig::default());
        let c = json!({"dry_run": true});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.mode, GateMode::Off);
        assert!(!parsed.enabled());
    }

        #[test]
    fn garbage_mode_is_off() {
        for bad in [
            json!({"overturn_hold_gate": {"mode": "garbage"}}),
            json!({"overturn_hold_gate": {"mode": 1}}),
            json!({"overturn_hold_gate": {"mode": true}}),
            json!({"overturn_hold_gate": {"mode": "ON"}}),
            json!({"overturn_hold_gate": "enforce"}),
            json!({"overturn_hold_gate": ["enforce"]}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&bad));
            assert_eq!(parsed.mode, GateMode::Off, "{bad}");
        }
    }

    #[test]
    fn parses_shadow_and_enforce_with_options() {
        let c = json!({"overturn_hold_gate": {
            "mode": " Shadow ",
            "on_probe_error": "skip",
            "topics": ["a.topic", "", "  b.topic "],
            "timeout_ms": 350,
            "positive_cache_secs": 5,
        }});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.mode, GateMode::Shadow);
        assert_eq!(parsed.on_probe_error, OnProbeError::Skip);
        assert_eq!(
            parsed.topics,
            vec!["a.topic".to_owned(), "b.topic".to_owned()]
        );
        assert_eq!(parsed.timeout_ms, 350);
        assert_eq!(parsed.positive_cache_secs, 5);
        assert!(parsed.covers_topic("a.topic"));
        assert!(!parsed.covers_topic("c.topic"));

        let c = json!({"overturn_hold_gate": {"mode": "enforce"}});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.mode, GateMode::Enforce);
        assert_eq!(
            parsed.on_probe_error,
            OnProbeError::Allow,
            "default is fail-open"
        );
        assert!(parsed.topics.is_empty());
        assert!(parsed.covers_topic("anything"));
        assert_eq!(parsed.timeout_ms, DEFAULT_TIMEOUT_MS);
        assert_eq!(parsed.positive_cache_secs, DEFAULT_POSITIVE_CACHE_SECS);
    }

                        #[test]
    fn non_array_topics_turns_the_gate_off() {
        for bad in [
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": "abuse.v3.score_results"}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": {"a": 1}}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": 7}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": true}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": [1, 2]}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "topics": ["a.topic", null]}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "topics": ["a.topic", ["nested"]]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": ["", " "]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": [""]}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "topics": ["   "]}}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&bad));
            assert_eq!(parsed, HoldGateConfig::default(), "{bad}");
            assert_eq!(parsed.mode, GateMode::Off, "{bad}");
            assert!(!parsed.enabled(), "{bad}");
        }

        for all in [
            json!({"overturn_hold_gate": {"mode": "enforce"}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": null}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "topics": []}}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&all));
            assert_eq!(parsed.mode, GateMode::Enforce, "{all}");
            assert!(parsed.topics.is_empty(), "{all}");
            assert!(parsed.covers_topic("anything"), "{all}");
        }
        let c = json!({"overturn_hold_gate": {"mode": "off", "topics": "x"}});
        assert_eq!(
            HoldGateConfig::from_config(Some(&c)),
            HoldGateConfig::default()
        );
    }

    #[test]
    fn bad_on_probe_error_defaults_to_allow_and_timeouts_are_clamped() {
        let c = json!({"overturn_hold_gate": {
            "mode": "enforce",
            "on_probe_error": "explode",
            "timeout_ms": 0,
            "positive_cache_secs": 999999,
        }});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.mode, GateMode::Enforce);
        assert_eq!(parsed.on_probe_error, OnProbeError::Allow);
        assert_eq!(parsed.timeout_ms, MIN_TIMEOUT_MS);
        assert_eq!(parsed.positive_cache_secs, MAX_POSITIVE_CACHE_SECS);
        let c = json!({"overturn_hold_gate": {"mode": "enforce", "timeout_ms": 60000}});
        assert_eq!(
            HoldGateConfig::from_config(Some(&c)).timeout_ms,
            MAX_TIMEOUT_MS
        );
    }

                #[test]
    fn off_parses_nothing_else_and_bad_numbers_fall_back_to_defaults() {
        let c = json!({"overturn_hold_gate": {
            "mode": "off",
            "on_probe_error": "explode",
            "timeout_ms": "200",
            "positive_cache_secs": 60.5,
        }});
        assert_eq!(
            HoldGateConfig::from_config(Some(&c)),
            HoldGateConfig::default()
        );

        let c = json!({"overturn_hold_gate": {
            "mode": "enforce",
            "timeout_ms": 200.0,
            "positive_cache_secs": -5,
        }});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.mode, GateMode::Enforce);
        assert_eq!(parsed.timeout_ms, DEFAULT_TIMEOUT_MS);
        assert_eq!(parsed.positive_cache_secs, DEFAULT_POSITIVE_CACHE_SECS);

        let mut seen = std::collections::HashSet::new();
        let v = json!("dedupe-me-please");
        assert!(first_sighting_in(&mut seen, "test_field", &v));
        assert!(!first_sighting_in(&mut seen, "test_field", &v));
        assert!(!first_sighting_in(&mut seen, "test_field", &v));
        assert!(
            first_sighting_in(&mut seen, "other_field", &v),
            "keyed by field AND value"
        );
        for i in 0..MAX_LOGGED_BAD_VALUES {
            first_sighting_in(&mut seen, "fill", &json!(i));
        }
        assert!(seen.len() <= MAX_LOGGED_BAD_VALUES);
        assert!(
            first_sighting_in(&mut seen, "test_field", &v),
            "after a clear the value is a first sighting again"
        );
    }

    #[test]
    fn startup_probe_env_parse() {
        for on in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(startup_probe_requested(Some(on)), "{on:?}");
        }
        for off in ["", "0", "false", "no", "off", "garbage", "  "] {
            assert!(!startup_probe_requested(Some(off)), "{off:?}");
        }
        assert!(!startup_probe_requested(None));
    }


    #[tokio::test]
    async fn off_does_nothing_and_never_calls_the_store() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let out = gate.evaluate(&cfg(GateMode::Off), 1, TOPIC, &[TEMP]).await;
        assert_eq!(out, GateOutcome::default());
        assert_eq!(gate.check(&cfg(GateMode::Off), 1, TOPIC).await, None);
        assert_eq!(store.calls(), 0);
        assert_eq!(store.pings(), 0);
    }

    #[tokio::test]
    async fn startup_probe_pings_once_in_any_mode_and_never_probes_holds() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        assert!(gate.startup_probe().await);
        assert_eq!(store.pings(), 1);
        assert_eq!(store.calls(), 0, "GET /readyz only; no hold lookup");
        assert_eq!(metrics::HOLD_GATE_DB_CONNECTED.get(), 1);
        store.set_fail(true);
        assert!(!gate.startup_probe().await);
        assert_eq!(store.pings(), 2);
        assert_eq!(metrics::HOLD_GATE_DB_CONNECTED.get(), 0);
        assert!(!HoldGate::new(None).startup_probe().await);
    }

    #[tokio::test]
    async fn topic_allow_list_disables_other_topics() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = HoldGateConfig {
            mode: GateMode::Enforce,
            topics: vec!["only.this".into()],
            ..HoldGateConfig::default()
        };
        assert_eq!(gate.check(&c, 1, TOPIC).await, None);
        assert_eq!(
            gate.evaluate(&c, 1, TOPIC, &[TEMP]).await,
            GateOutcome::default()
        );
        assert_eq!(store.calls(), 0);
        let out = gate.evaluate(&c, 1, "only.this", &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(store.calls(), 1);
    }

    #[tokio::test]
    async fn shadow_held_acts_with_info_and_counter() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(7, 70, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let before = counter("shadow", "held_shadow");
        let out = gate
            .evaluate(&cfg(GateMode::Shadow), 7, TOPIC, &[TEMP])
            .await;
        assert_eq!(out.skip_status, None, "shadow never changes the decision");
        assert_eq!(out.info["overturn_hold_would_block"], "true");
        assert_eq!(out.info["overturn_hold_id"], "70");
        assert_eq!(out.info["overturn_hold_head"], "FollowBot");
        assert_eq!(out.info["overturn_hold_mode"], "shadow");
        assert!(
            !out.info.contains_key("overturn_hold_kind"),
            "kind is stamped on collapsed skips only"
        );
        assert!(out.info.contains_key("overturn_hold_expires_at"));
        assert!(out.info.contains_key("overturn_hold_probe_ms"));
        assert_eq!(counter("shadow", "held_shadow"), before + 1);
        assert_eq!(store.calls(), 1);
    }

    #[tokio::test]
    async fn enforce_held_skips_with_hold_overturned() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(8, 80, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let before = counter("enforce", "held_blocked");
        let out = gate
            .evaluate(&cfg(GateMode::Enforce), 8, TOPIC, &[TEMP])
            .await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_id"], "80");
        assert_eq!(out.info["overturn_hold_head"], "FollowBot");
        assert_eq!(out.info["overturn_hold_mode"], "enforce");
        assert!(out.info.contains_key("overturn_hold_expires_at"));
        assert!(out.info.contains_key("overturn_hold_probe_ms"));
        assert!(!out.info.contains_key("overturn_hold_would_block"));
        assert_eq!(out.info["overturn_hold_kind"], "suspend");
        assert!(!out.info.contains_key("overturn_hold_label"));
        assert_eq!(counter("enforce", "held_blocked"), before + 1);
        let mut marked = out.clone();
        marked.mark_label_collapse();
        assert_eq!(marked, out);
    }

    #[tokio::test]
    async fn not_held_proceeds_with_mode_and_probe_ms() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let before = counter("enforce", "not_held");
        let out = gate
            .evaluate(&cfg(GateMode::Enforce), 2, TOPIC, &[TEMP])
            .await;
        assert_eq!(out.skip_status, None);
        assert_eq!(out.info["overturn_hold_mode"], "enforce");
        assert!(out.info.contains_key("overturn_hold_probe_ms"));
        assert!(!out.info.contains_key("overturn_hold_id"));
        assert_eq!(counter("enforce", "not_held"), before + 1);
    }

    #[tokio::test]
    async fn enforce_probe_error_allow_acts_and_counts_open() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(3, 30, 30)]);
        store.set_fail(true);
        let gate = HoldGate::new(Some(store.clone()));
        let before = counter("enforce", "probe_error_open");
        let out = gate
            .evaluate(&cfg(GateMode::Enforce), 3, TOPIC, &[TEMP])
            .await;
        assert_eq!(
            out.skip_status, None,
            "default on_probe_error=allow is fail-open"
        );
        assert_eq!(out.info["overturn_hold_probe_error"], "connect_refused");
        assert_eq!(counter("enforce", "probe_error_open"), before + 1);
    }

    #[tokio::test]
    async fn enforce_probe_error_skip_emits_hold_lookup_failed() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        store.set_fail(true);
        let gate = HoldGate::new(Some(store.clone()));
        let c = HoldGateConfig {
            mode: GateMode::Enforce,
            on_probe_error: OnProbeError::Skip,
            ..HoldGateConfig::default()
        };
        let before = counter("enforce", "probe_error_skip");
        let out = gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_LOOKUP_FAILED));
        assert_eq!(counter("enforce", "probe_error_skip"), before + 1);
    }

    #[tokio::test]
    async fn shadow_probe_error_is_always_fail_open() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        store.set_fail(true);
        let gate = HoldGate::new(Some(store.clone()));
        let c = HoldGateConfig {
            mode: GateMode::Shadow,
            on_probe_error: OnProbeError::Skip,
            ..HoldGateConfig::default()
        };
        let before = counter("shadow", "probe_error_open");
        let out = gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, None);
        assert_eq!(counter("shadow", "probe_error_open"), before + 1);
    }

    #[tokio::test]
    async fn no_store_is_a_probe_error() {
        let _serial = METRICS_LOCK.lock().await;
        let gate = HoldGate::new(None);
        assert!(!gate.has_store());
        match gate.check(&cfg(GateMode::Enforce), 5, TOPIC).await {
            Some(GateVerdict::ProbeError(e)) => {
                assert_eq!(e.code, ProbeErrorCode::NotConfigured);
                assert!(e.detail.contains("OVERTURN_HOLD_LEDGER_URL"));
            }
            other => panic!("expected ProbeError, got {other:?}"),
        }
        let out = gate
            .evaluate(&cfg(GateMode::Enforce), 5, TOPIC, &[TEMP])
            .await;
        assert_eq!(out.skip_status, None);
        let c = HoldGateConfig {
            mode: GateMode::Enforce,
            on_probe_error: OnProbeError::Skip,
            ..HoldGateConfig::default()
        };
        let out = gate.evaluate(&c, 5, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_LOOKUP_FAILED));
        assert_eq!(
            gate.evaluate(&cfg(GateMode::Off), 5, TOPIC, &[TEMP]).await,
            GateOutcome::default()
        );
    }

            #[tokio::test]
    async fn env_is_stamped_on_gated_decisions_only() {
        let _serial = METRICS_LOCK.lock().await;
        let gate = HoldGate::from_url(None, false, Some(" staging "));
        let out = gate
            .evaluate(&cfg(GateMode::Enforce), 1, TOPIC, &[TEMP])
            .await;
        assert_eq!(out.info["overturn_hold_env"], "staging", "trimmed");
        assert_eq!(out.info["overturn_hold_probe_error"], "not_configured");
        assert_eq!(
            gate.evaluate(&cfg(GateMode::Off), 1, TOPIC, &[TEMP]).await,
            GateOutcome::default(),
            "off: no keys at all"
        );
        for blank in [None, Some(""), Some("   ")] {
            let gate = HoldGate::from_url(None, false, blank);
            let out = gate
                .evaluate(&cfg(GateMode::Enforce), 1, TOPIC, &[TEMP])
                .await;
            assert!(!out.info.contains_key("overturn_hold_env"), "{blank:?}");
        }
        let gate = HoldGate::from_url(Some("not a url"), false, Some("prod"));
        let out = gate
            .evaluate(&cfg(GateMode::Enforce), 1, TOPIC, &[TEMP])
            .await;
        assert_eq!(out.info["overturn_hold_env"], "prod");
        assert_eq!(out.info["overturn_hold_probe_error"], "bad_url");
    }

    #[tokio::test]
    async fn positive_cached_negative_not() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(11, 110, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg(GateMode::Enforce);

        let before_hit = counter("enforce", "cache_hit");
        let a = gate.evaluate(&c, 11, TOPIC, &[TEMP]).await;
        let b = gate.evaluate(&c, 11, TOPIC, &[TEMP]).await;
        assert_eq!(a.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(b.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(store.calls(), 1, "positive answer must be cached");
        assert_eq!(counter("enforce", "cache_hit"), before_hit + 1);
        assert_eq!(b.info["overturn_hold_probe_ms"], "0");
        assert_eq!(gate.cached_len(), 1);

        gate.evaluate(&c, 12, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 12, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 3, "negative answers are never cached");
        assert_eq!(gate.cached_len(), 1);
    }

    #[tokio::test]
    async fn positive_cache_can_be_disabled() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(11, 110, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = HoldGateConfig {
            mode: GateMode::Enforce,
            positive_cache_secs: 0,
            ..HoldGateConfig::default()
        };
        gate.evaluate(&c, 11, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 11, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 2);
        assert_eq!(gate.cached_len(), 0);
    }

                            #[tokio::test]
    async fn positive_cache_is_clamped_to_the_holds_expiry() {
        let _serial = METRICS_LOCK.lock().await;
        let clock = ManualClock::new();
        let mut soon = hold(31, 310, 1);
        soon.expires_at = clock.now_utc() + chrono::Duration::seconds(30);
        let store = FakeHoldStore::new(vec![soon]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        let c = HoldGateConfig {
            mode: GateMode::Enforce,
            positive_cache_secs: 3_600,
            ..HoldGateConfig::default()
        };
        assert_eq!(
            gate.evaluate(&c, 31, TOPIC, &[TEMP]).await.skip_status,
            Some(STATUS_HOLD_OVERTURNED)
        );
        assert_eq!(gate.cached_len(), 1);
        assert_eq!(
            gate.cached_ttl(31),
            Some(Duration::from_secs(30)),
            "TTL clamped to the hold, not the 3600 s config"
        );
        clock.advance(Duration::from_secs(29));
        gate.evaluate(&c, 31, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 1, "served from cache while the hold is live");
        clock.advance(Duration::from_secs(2));
        assert_eq!(
            gate.cached_len(),
            0,
            "the entry is dead with the hold (read-path view)"
        );
        let out = gate.evaluate(&c, 31, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 2, "re-probed once the hold expired");
        assert_eq!(
            out.skip_status, None,
            "the re-probe's (expired) hold is dropped too"
        );
        assert_eq!(gate.cached_snapshot(31), None, "…and nothing is cached");

        let mut past = hold(32, 320, 1);
        past.expires_at = clock.now_utc() - chrono::Duration::seconds(1);
        let store = FakeHoldStore::new(vec![past]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        gate.evaluate(&c, 32, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 32, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 2);
        assert_eq!(gate.cached_len(), 0);

        let store = FakeHoldStore::new(vec![hold(33, 330, 30)]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        let c60 = cfg(GateMode::Enforce);
        assert_eq!(c60.positive_cache_secs, 60);
        gate.evaluate(&c60, 33, TOPIC, &[TEMP]).await;
        assert_eq!(gate.cached_ttl(33), Some(Duration::from_secs(60)));

        let mut short = hold(34, 340, 30);
        short.expires_at = clock.now_utc() + chrono::Duration::seconds(90);
        let store = FakeHoldStore::new(vec![hold(34, 341, 30), short]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        gate.evaluate(&c, 34, TOPIC, &[TEMP]).await;
        assert_eq!(gate.cached_ttl(34), Some(Duration::from_secs(90)));
        assert_eq!(gate.cached_snapshot(34), Some((vec![341, 340], vec![])));
        assert_eq!(CACHE_MAX_WEIGHT, 100_000);
    }

                    #[tokio::test]
    async fn positive_cache_entries_expire_by_ttl() {
        let _serial = METRICS_LOCK.lock().await;
        let mut one_sec = hold(41, 410, 30);
        one_sec.expires_at = Utc::now() + chrono::Duration::seconds(1);
        let store = FakeHoldStore::new(vec![one_sec, hold(42, 420, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c_long = HoldGateConfig {
            positive_cache_secs: 3_600,
            ..cfg(GateMode::Enforce)
        };
        let c_1s = HoldGateConfig {
            positive_cache_secs: 1,
            ..cfg(GateMode::Enforce)
        };
        assert_eq!(
            gate.evaluate(&c_long, 41, TOPIC, &[TEMP]).await.skip_status,
            Some(STATUS_HOLD_OVERTURNED)
        );
        assert_eq!(
            gate.evaluate(&c_1s, 42, TOPIC, &[TEMP]).await.skip_status,
            Some(STATUS_HOLD_OVERTURNED)
        );
        assert_eq!(store.calls(), 2);
        assert!(
            gate.cached_ttl(41)
                .is_some_and(|t| t <= Duration::from_secs(1))
        );
        assert_eq!(gate.cached_ttl(42), Some(Duration::from_secs(1)));
        gate.evaluate(&c_long, 41, TOPIC, &[TEMP]).await;
        gate.evaluate(&c_1s, 42, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 2, "served from the cache inside the TTL");

        tokio::time::sleep(Duration::from_millis(1_500)).await;
        assert_eq!(
            gate.cached_snapshot(41),
            None,
            "moka expired the hold-clamped entry"
        );
        assert_eq!(
            gate.cached_snapshot(42),
            None,
            "moka expired the config-TTL entry"
        );
        gate.evaluate(&c_long, 41, TOPIC, &[TEMP]).await;
        gate.evaluate(&c_1s, 42, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 4, "both re-probed after the TTL");
        assert_eq!(
            gate.cached_snapshot(42),
            Some((vec![420], vec![])),
            "re-cached"
        );
        assert_eq!(
            gate.cached_snapshot(41),
            None,
            "the 1 s hold has lapsed: dropped, not cached"
        );
    }

                                #[tokio::test]
    async fn expired_holds_in_a_fresh_probe_result_are_dropped_and_counted() {
        let _serial = METRICS_LOCK.lock().await;
        let clock = ManualClock::new();
        let mut expired = hold(1, 10, 1);
        expired.expires_at = clock.now_utc() - chrono::Duration::seconds(5);
        let mut expired_label = label_hold_row(1, 11, 1, SHR);
        expired_label.expires_at = clock.now_utc() - chrono::Duration::seconds(1);
        let store = FakeHoldStore::new(vec![expired.clone(), expired_label]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        let c = cfg_labels(GateMode::Enforce, &[SHR]);
        let before = metrics::HOLD_GATE_EXPIRED_HOLD_TOTAL.get();
        let before_not_held = counter("enforce", "not_held");

        let out = gate
            .evaluate(&c, 1, TOPIC, &[TEMP, label_action(SHR)])
            .await;
        assert_eq!(
            out.skip_status, None,
            "an expired suspend hold is not a hold"
        );
        assert!(
            out.strip_labels.is_empty(),
            "an expired label hold is not a hold"
        );
        assert!(!out.info.contains_key("overturn_hold_id"));
        assert!(
            !out.info.contains_key("overturn_hold_probe_error"),
            "not an error"
        );
        assert_eq!(counter("enforce", "not_held"), before_not_held + 1);
        assert_eq!(metrics::HOLD_GATE_EXPIRED_HOLD_TOTAL.get(), before + 2);
        assert_eq!(gate.cached_len(), 0, "nothing live to cache");
        assert_eq!(gate.expired_holds_seen.load(Ordering::Relaxed), 2);

        let live = hold(1, 12, 30);
        store.set_holds(vec![expired, live]);
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_id"], "12");
        assert_eq!(metrics::HOLD_GATE_EXPIRED_HOLD_TOTAL.get(), before + 3);
        assert_eq!(gate.cached_snapshot(1), Some((vec![12], Vec::new())));
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.info["overturn_hold_id"], "12");
        assert_eq!(out.info["overturn_hold_probe_ms"], "0", "cache hit");
        assert_eq!(store.calls(), 2);
        assert_eq!(
            metrics::HOLD_GATE_EXPIRED_HOLD_TOTAL.get(),
            before + 3,
            "cache hits count nothing"
        );

        let mut edge = hold(2, 20, 1);
        edge.expires_at = clock.now_utc();
        let mut soon = hold(2, 21, 1);
        soon.expires_at = clock.now_utc() + chrono::Duration::seconds(1);
        store.set_holds(vec![edge, soon]);
        let out = gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(out.info["overturn_hold_id"], "21");
        assert_eq!(metrics::HOLD_GATE_EXPIRED_HOLD_TOTAL.get(), before + 4);
        assert_eq!(sampled(&AtomicU64::new(0)), (1, true));
        assert_eq!(sampled(&AtomicU64::new(1)), (2, false));
        assert_eq!(sampled(&AtomicU64::new(999)), (1_000, true));
        assert_eq!(sampled(&AtomicU64::new(1_000)), (1_001, false));
    }


            fn breaker_gate(
        store: Arc<FakeHoldStore>,
        failures: u32,
        open: Duration,
    ) -> (HoldGate, Arc<ManualClock>) {
        let clock = ManualClock::new();
        let gate = HoldGate::with_breaker_and_clock(Some(store), failures, open, clock.clone());
        (gate, clock)
    }

        const PAST_OPEN: Duration = Duration::from_secs(DEFAULT_BREAKER_OPEN.as_secs() + 1);

                #[tokio::test]
    async fn breaker_opens_after_m_failures_half_opens_after_n_and_closes_on_success() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        store.set_fail(true);
        let (gate, clock) = breaker_gate(store.clone(), 3, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);

        for _ in 0..2 {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.skip_status, None, "allow = fail-open");
        }
        assert!(!gate.breaker_open());
        assert_eq!(store.calls(), 2);

        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open());
        assert_eq!(metrics::HOLD_GATE_BREAKER_OPEN.get(), 1);
        assert_eq!(store.calls(), 3);

        let before_open = counter("enforce", "probe_error_open");
        let before_skips = metrics::HOLD_GATE_BREAKER_SKIP_TOTAL.get();
        let out = gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 3, "no probe while open");
        assert_eq!(out.info["overturn_hold_probe_error"], "breaker_open");
        assert_eq!(out.skip_status, None);
        assert_eq!(counter("enforce", "probe_error_open"), before_open + 1);
        assert_eq!(
            metrics::HOLD_GATE_BREAKER_SKIP_TOTAL.get(),
            before_skips + 1
        );
        let c_skip = HoldGateConfig {
            mode: GateMode::Enforce,
            on_probe_error: OnProbeError::Skip,
            ..HoldGateConfig::default()
        };
        let before_skip = counter("enforce", "probe_error_skip");
        let out = gate.evaluate(&c_skip, 2, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_LOOKUP_FAILED));
        assert_eq!(counter("enforce", "probe_error_skip"), before_skip + 1);
        assert_eq!(store.calls(), 3);

        clock.advance(DEFAULT_BREAKER_OPEN - Duration::from_secs(1));
        gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 3, "still inside the open window");

        clock.advance(Duration::from_secs(2));
        gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 4, "half-open trial probed");
        assert!(gate.breaker_open(), "failed trial re-opens");
        gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 4, "re-opened: skipped again");

        store.set_fail(false);
        clock.advance(PAST_OPEN);
        let out = gate.evaluate(&c, 4, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 5);
        assert!(!gate.breaker_open());
        assert_eq!(metrics::HOLD_GATE_BREAKER_OPEN.get(), 0);
        assert!(!out.info.contains_key("overturn_hold_probe_error"));
        gate.evaluate(&c, 4, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 6);
    }

                #[tokio::test]
    async fn breaker_treats_benign_sql_errors_as_reachable() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let gate = HoldGate::with_breaker(Some(store.clone()), 3, Duration::from_secs(60));
        let c = cfg(GateMode::Enforce);

        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        store.set_fail_query(true);
        for _ in 0..4 {
            gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        }
        assert!(!gate.breaker_open(), "SQL errors never open the breaker");
        assert_eq!(
            gate.backoff_snapshot(),
            (4, false),
            "one short of the backoff"
        );
        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(!gate.breaker_open(), "count was reset by the SQL error");
        assert_eq!(
            gate.backoff_snapshot(),
            (0, false),
            "a failure resets the reachable run"
        );
        store.set_fail(false);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(!gate.breaker_open(), "count was reset by the success");
        assert_eq!(store.calls(), 11, "every call reached the store");
        assert_eq!(BREAKER_FAILURES, 5);
        assert_eq!(DEFAULT_BREAKER_OPEN, Duration::from_secs(10));
    }

                    #[test]
    fn every_store_error_has_a_breaker_class() {
        use BreakerSignal::*;
        let fwd = |status: u16, code: LedgerCode| HoldStoreError::Upstream {
            status,
            code: Some(code),
        };
        let bare = |status: u16| HoldStoreError::Upstream { status, code: None };
        let cases = [
            (
                HoldStoreError::Unreachable("connection refused".into()),
                Failure,
            ),
            (HoldStoreError::Timeout(Duration::from_millis(300)), Failure),
            (fwd(503, LedgerCode::ConnectRefused), Failure),
            (fwd(503, LedgerCode::AuthFailed), Failure),
            (fwd(503, LedgerCode::Tls), Failure),
            (fwd(504, LedgerCode::StatementTimeout), Failure),
            (fwd(503, LedgerCode::ServerUnhealthy), Failure),
            (fwd(504, LedgerCode::Timeout), Failure),
            (fwd(503, LedgerCode::PoolWait), Neutral),
            (fwd(500, LedgerCode::Decode), Neutral),
            (fwd(500, LedgerCode::Other), Reachable),
            (fwd(400, LedgerCode::Other), Reachable),
            (bare(502), Failure),
            (bare(503), Failure),
            (bare(504), Failure),
            (bare(500), Failure),
            (bare(404), Reachable),
            (bare(400), Reachable),
            (bare(429), Reachable),
            (HoldStoreError::Decode("missing field".into()), Neutral),
            (HoldStoreError::BadUrl("does not parse".into()), Neutral),
            (HoldStoreError::NotConfigured, Neutral),
        ];
        for (err, want) in cases {
            assert_eq!(err.breaker_signal(), want, "{err}");
        }
        assert_eq!(fwd(200, LedgerCode::PoolWait).breaker_signal(), Neutral);
        assert_eq!(fwd(418, LedgerCode::AuthFailed).breaker_signal(), Failure);
        assert_eq!(
            fwd(504, LedgerCode::StatementTimeout).to_string(),
            "ledger service answered 504 code=statement_timeout"
        );
        assert_eq!(bare(502).to_string(), "ledger service answered 502");
        assert!(
            HoldStoreError::Unreachable("x".into())
                .to_string()
                .starts_with("ledger service unreachable:")
        );
        assert!(
            HoldStoreError::Decode("x".into())
                .to_string()
                .starts_with("response decode:")
        );
    }

                            #[tokio::test]
    async fn db_connected_gauge_follows_the_probe_class_in_the_gate() {
        let _serial = METRICS_LOCK.lock().await;
        let fwd = |status: u16, code: LedgerCode| HoldStoreError::Upstream {
            status,
            code: Some(code),
        };
        for (err, want) in [
            (HoldStoreError::Unreachable("refused".into()), Some(false)),
            (
                HoldStoreError::Timeout(Duration::from_millis(1)),
                Some(false),
            ),
            (fwd(503, LedgerCode::ConnectRefused), Some(true)),
            (fwd(500, LedgerCode::Other), Some(true)),
            (
                HoldStoreError::Upstream {
                    status: 404,
                    code: None,
                },
                Some(true),
            ),
            (HoldStoreError::Decode("x".into()), Some(true)),
            (HoldStoreError::BadUrl("x".into()), None),
            (HoldStoreError::NotConfigured, None),
        ] {
            assert_eq!(err.ledger_answered(), want, "{err}");
        }

        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::with_breaker(Some(store.clone()), 100, Duration::from_secs(60));
        let c = cfg(GateMode::Enforce);
        metrics::HOLD_GATE_DB_CONNECTED.set(0);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(metrics::HOLD_GATE_DB_CONNECTED.get(), 1);
        store.set_failure(FakeFailure::Unreachable);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(metrics::HOLD_GATE_DB_CONNECTED.get(), 0, "hop failed");
        store.set_failure(FakeFailure::SqlPermission);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(
            metrics::HOLD_GATE_DB_CONNECTED.get(),
            1,
            "a 500 with a code is an answer"
        );
        store.set_failure(FakeFailure::Unreachable);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(metrics::HOLD_GATE_DB_CONNECTED.get(), 0);
        store.set_failure(FakeFailure::Decode);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(
            metrics::HOLD_GATE_DB_CONNECTED.get(),
            1,
            "an undecodable 200 is an answer"
        );
        let no_store = HoldGate::new(None);
        no_store.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(
            metrics::HOLD_GATE_DB_CONNECTED.get(),
            1,
            "not_configured leaves it"
        );
        let bad = HoldGate::from_url(Some("not a url"), false, None);
        bad.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(
            metrics::HOLD_GATE_DB_CONNECTED.get(),
            1,
            "bad_url leaves it"
        );
        store.set_failure(FakeFailure::None);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        metrics::HOLD_GATE_DB_CONNECTED.set(0);
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.info["overturn_hold_probe_ms"], "0", "cache hit");
        assert_eq!(
            metrics::HOLD_GATE_DB_CONNECTED.get(),
            0,
            "no probe, no gauge write"
        );
    }

                        #[tokio::test]
    async fn pool_saturation_never_opens_the_breaker() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let gate = HoldGate::with_breaker(Some(store.clone()), 3, Duration::from_secs(60));
        let c = cfg(GateMode::Enforce);
        store.set_failure(FakeFailure::Saturated);
        let before = counter("enforce", "probe_error_open");
        metrics::HOLD_GATE_BREAKER_OPEN.set(0);
        for _ in 0..20 {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.skip_status, None, "allow: fail-open");
            assert_eq!(out.info["overturn_hold_probe_error"], "pool_wait");
        }
        assert!(!gate.breaker_open(), "20 saturations: still closed");
        assert_eq!(store.calls(), 20, "every probe reached the store");
        assert_eq!(counter("enforce", "probe_error_open"), before + 20);
        assert_eq!(
            metrics::HOLD_GATE_BREAKER_OPEN.get(),
            0,
            "neutral never sets it"
        );

        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        store.set_failure(FakeFailure::Saturated);
        for _ in 0..5 {
            gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        }
        assert!(!gate.breaker_open());
        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open(), "3rd real failure opens");
    }

                                #[tokio::test]
    async fn probe_failure_warn_sampler_resets_on_success_only() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let gate = HoldGate::with_breaker(Some(store.clone()), 100, Duration::from_secs(60));
        let c = cfg(GateMode::Enforce);
        assert_eq!(gate.probe_failures(), 0);

        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 1, "first failure is warn-eligible");
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 3);

        store.set_fail(false);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 0, "a success resets the sampler");

        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 1, "first failure of the new run");

        store.set_fail_query(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(
            gate.probe_failures(),
            2,
            "reachable error: counted, not reset"
        );
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 4, "…and keeps counting");

        store.set_failure(FakeFailure::Saturated);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 5, "neutral: counted, not reset");
        assert!(!gate.breaker_open());

        store.set_failure(FakeFailure::None);
        store.set_holds(vec![hold(1, 10, 30)]);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 0, "held = success = reset");
    }

            #[derive(Clone, Default)]
    struct LogBuf(Arc<Mutex<Vec<u8>>>);

    impl LogBuf {
        fn lines_containing(&self, needle: &str) -> usize {
            let bytes = self.0.lock().unwrap_or_else(|p| p.into_inner());
            String::from_utf8_lossy(&bytes)
                .lines()
                .filter(|l| l.contains(needle))
                .count()
        }
    }

    impl std::io::Write for LogBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuf {
        type Writer = LogBuf;

        fn make_writer(&'a self) -> LogBuf {
            self.clone()
        }
    }

            fn capture_warns() -> (tracing::subscriber::DefaultGuard, LogBuf) {
        let buf = LogBuf::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .with_writer(buf.clone())
            .finish();
        (tracing::subscriber::set_default(subscriber), buf)
    }

                                    #[tokio::test]
    async fn persistent_reachable_errors_log_one_warn_per_run_not_per_decision() {
        let _serial = METRICS_LOCK.lock().await;
        let (_guard, logs) = capture_warns();
        let store = FakeHoldStore::new(vec![]);
        let (gate, clock) = breaker_gate(store.clone(), BREAKER_FAILURES, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);
        store.set_fail_query(true);
        const N: usize = 50;
        let past_backoff = REACHABLE_ERROR_BACKOFF + Duration::from_secs(1);
        for _ in 0..N {
            clock.advance(past_backoff);
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.info["overturn_hold_probe_error"], "other");
        }
        assert_eq!(store.calls(), N, "every call was a real probe");
        assert_eq!(gate.probe_failures(), N as u64, "counted, never reset");
        assert_eq!(
            logs.lines_containing("probe failed"),
            1,
            "one sampled warn for {N} consecutive reachable errors:\n{}",
            String::from_utf8_lossy(&logs.0.lock().unwrap())
        );
        assert_eq!(
            logs.lines_containing("answers but cannot serve"),
            1,
            "the backoff arms (and logs) once per run"
        );
        assert!(
            !gate.breaker_open(),
            "reachable errors never open the breaker"
        );

        store.set_failure(FakeFailure::None);
        clock.advance(past_backoff);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.probe_failures(), 0);
        store.set_failure(FakeFailure::NotFound);
        for _ in 0..N {
            clock.advance(past_backoff);
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.info["overturn_hold_probe_error"], "ledger_status");
        }
        assert_eq!(
            logs.lines_containing("probe failed"),
            2,
            "one more for the new run"
        );
        assert_eq!(logs.lines_containing("answers but cannot serve"), 2);
        let before = logs.lines_containing("probe failed");
        for _ in 0..20 {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.info["overturn_hold_probe_error"], "reachable_backoff");
        }
        assert_eq!(logs.lines_containing("probe failed"), before);
        assert_eq!(logs.lines_containing("probe skipped"), 0);
    }

                                            #[tokio::test]
    async fn persistent_reachable_errors_back_off_without_opening_the_breaker() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(9, 90, 30)]);
        let (gate, clock) = breaker_gate(store.clone(), BREAKER_FAILURES, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);
        assert_eq!(REACHABLE_ERRORS_BEFORE_BACKOFF, 5);
        assert_eq!(REACHABLE_ERROR_BACKOFF, Duration::from_secs(5));
        metrics::HOLD_GATE_BREAKER_OPEN.set(0);

        assert_eq!(
            gate.evaluate(&c, 9, TOPIC, &[TEMP]).await.skip_status,
            Some(STATUS_HOLD_OVERTURNED)
        );
        assert_eq!(store.calls(), 1);

        store.set_failure(FakeFailure::NotFound);
        for i in 1..REACHABLE_ERRORS_BEFORE_BACKOFF {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.info["overturn_hold_probe_error"], "ledger_status");
            assert_eq!(gate.backoff_snapshot(), (i, false), "not yet");
        }
        assert_eq!(store.calls(), 5);
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.info["overturn_hold_probe_error"], "ledger_status");
        assert_eq!(gate.backoff_snapshot(), (5, true), "armed");
        assert_eq!(store.calls(), 6);
        assert!(
            !gate.breaker_open(),
            "a 404 is reachable: the breaker never opens"
        );
        assert_eq!(metrics::HOLD_GATE_BREAKER_OPEN.get(), 0);

        let before_open = counter("enforce", "probe_error_open");
        let before_backoff = metrics::HOLD_GATE_BACKOFF_SKIP_TOTAL.get();
        let before_breaker = metrics::HOLD_GATE_BREAKER_SKIP_TOTAL.get();
        let out = gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(
            out.skip_status, None,
            "allow: fail-open, exactly like a breaker skip"
        );
        assert_eq!(out.info["overturn_hold_probe_error"], "reachable_backoff");
        assert_eq!(out.info["overturn_hold_probe_ms"], "0");
        assert_eq!(store.calls(), 6, "no round trip");
        assert_eq!(counter("enforce", "probe_error_open"), before_open + 1);
        assert_eq!(
            metrics::HOLD_GATE_BACKOFF_SKIP_TOTAL.get(),
            before_backoff + 1
        );
        assert_eq!(metrics::HOLD_GATE_BREAKER_SKIP_TOTAL.get(), before_breaker);
        let c_skip = HoldGateConfig {
            on_probe_error: OnProbeError::Skip,
            ..cfg(GateMode::Enforce)
        };
        let out = gate.evaluate(&c_skip, 2, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_LOOKUP_FAILED));
        assert_eq!(out.info["overturn_hold_probe_error"], "reachable_backoff");
        let out = gate
            .evaluate(&cfg(GateMode::Shadow), 2, TOPIC, &[TEMP])
            .await;
        assert_eq!(out.skip_status, None);
        assert_eq!(store.calls(), 6);
        let out = gate.evaluate(&c, 9, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_probe_ms"], "0", "cache hit");
        assert_eq!(store.calls(), 6);

        clock.advance(REACHABLE_ERROR_BACKOFF - Duration::from_secs(1));
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 6, "inside the window");
        clock.advance(Duration::from_secs(2));
        let out = gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(
            out.info["overturn_hold_probe_error"], "ledger_status",
            "real probe"
        );
        assert_eq!(store.calls(), 7);
        assert_eq!(gate.backoff_snapshot(), (6, true), "re-armed");
        let out = gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(out.info["overturn_hold_probe_error"], "reachable_backoff");
        assert_eq!(store.calls(), 7, "one round trip per window per pod");

        clock.advance(REACHABLE_ERROR_BACKOFF + Duration::from_secs(1));
        store.set_failure(FakeFailure::None);
        let out = gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert!(!out.info.contains_key("overturn_hold_probe_error"));
        assert_eq!(gate.backoff_snapshot(), (0, false), "disarmed");
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 10, "every call probes");

        store.set_failure(FakeFailure::SqlPermission);
        for _ in 0..REACHABLE_ERRORS_BEFORE_BACKOFF {
            assert_eq!(
                gate.evaluate(&c, 3, TOPIC, &[TEMP]).await.info["overturn_hold_probe_error"],
                "other"
            );
        }
        assert_eq!(gate.backoff_snapshot(), (5, true));
        assert_eq!(
            gate.evaluate(&c, 3, TOPIC, &[TEMP]).await.info["overturn_hold_probe_error"],
            "reachable_backoff"
        );
        assert_eq!(store.calls(), 15);

        clock.advance(REACHABLE_ERROR_BACKOFF + Duration::from_secs(1));
        store.set_failure(FakeFailure::Unreachable);
        let out = gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(out.info["overturn_hold_probe_error"], "ledger_unreachable");
        assert_eq!(
            gate.backoff_snapshot(),
            (0, false),
            "failure resets the reachable run"
        );
        assert_eq!(gate.breaker_snapshot().0, 1);
        assert!(!gate.breaker_open());
        store.set_failure(FakeFailure::NotFound);
        gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(gate.backoff_snapshot(), (1, false));
        assert_eq!(
            gate.breaker_snapshot().0,
            0,
            "reachable resets the breaker count"
        );

        store.set_failure(FakeFailure::Saturated);
        for _ in 0..3 {
            gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        }
        assert_eq!(gate.backoff_snapshot(), (1, false), "neutral: untouched");
        store.set_failure(FakeFailure::NotFound);
        for _ in 0..4 {
            gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        }
        assert_eq!(
            gate.backoff_snapshot(),
            (5, true),
            "armed across the neutral errors"
        );
        assert!(!gate.breaker_open());
        assert_eq!(metrics::HOLD_GATE_BREAKER_OPEN.get(), 0, "never touched");
    }

                    #[tokio::test]
    async fn stragglers_from_before_the_open_are_ignored() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let (gate, clock) = breaker_gate(store.clone(), 2, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);

        store.set_hang(true);
        let mut straggler = Box::pin(gate.check(&c, 1, TOPIC));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut straggler)
                .await
                .is_err()
        );
        assert_eq!(store.calls(), 1);
        assert_eq!(gate.breaker_snapshot().2, 0, "generation 0");

        store.set_hang(false);
        store.set_fail(true);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open());
        let (failures, open_until, generation) = gate.breaker_snapshot();
        assert_eq!((failures, generation), (2, 1));

        clock.advance(Duration::from_secs(3));
        match straggler.await {
            Some(GateVerdict::ProbeError(e)) => {
                assert_eq!(e.code, ProbeErrorCode::ConnectRefused, "{e}")
            }
            other => panic!("expected the straggler to fail, got {other:?}"),
        }
        assert_eq!(store.calls(), 3);
        let (failures2, open_until2, generation2) = gate.breaker_snapshot();
        assert_eq!(failures2, failures, "stale failure not counted");
        assert_eq!(
            open_until2, open_until,
            "stale failure did not extend open_until"
        );
        assert_eq!(generation2, generation, "stale failure did not re-arm");

        clock.advance(PAST_OPEN);
        store.set_hang_only(Some(3));
        store.set_hang(true);
        store.set_fail(false);
        let mut trial = Box::pin(gate.check(&c, 3, TOPIC));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut trial)
                .await
                .is_err()
        );
        assert!(
            gate.half_open_inflight(),
            "the straggler holds the trial slot"
        );
        clock.advance(half_open_stale_after(c.timeout_ms) + Duration::from_secs(1));
        store.set_fail(true);
        gate.evaluate(&c, 4, TOPIC, &[TEMP]).await;
        assert_eq!(
            gate.breaker_snapshot().2,
            2,
            "failing trial re-armed → gen 2"
        );
        assert!(gate.breaker_open());
        let open_until_g2 = gate.breaker_snapshot().1;
        store.set_fail(false);
        store.set_hang(false);
        store.set_hang_only(None);
        assert_eq!(
            trial.await,
            Some(GateVerdict::NotHeld),
            "the gen-1 straggler succeeded"
        );
        assert!(
            gate.breaker_open(),
            "stale success did not close the breaker"
        );
        assert_eq!(gate.breaker_snapshot().1, open_until_g2);
        assert!(!gate.half_open_inflight(), "…but its own slot is freed");
    }

                #[tokio::test]
    async fn ledger_service_unreachable_trips_breaker() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let gate = HoldGate::with_breaker(Some(store.clone()), 3, Duration::from_secs(60));
        let c = cfg(GateMode::Enforce);
        store.set_failure(FakeFailure::Unreachable);
        for _ in 0..2 {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.info["overturn_hold_probe_error"], "ledger_unreachable");
        }
        assert!(!gate.breaker_open());
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open(), "3rd consecutive unreachable opens it");
        assert_eq!(store.calls(), 3);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 3, "open: skipped");
    }

                #[tokio::test]
    async fn statement_timeout_sqlstate_trips_breaker_but_permission_error_does_not() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let gate = HoldGate::with_breaker(Some(store.clone()), 2, Duration::from_secs(60));
        let c = cfg(GateMode::Enforce);
        store.set_failure(FakeFailure::SqlPermission);
        for _ in 0..4 {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.info["overturn_hold_probe_error"], "other");
        }
        assert!(!gate.breaker_open(), "benign SQLSTATE never opens");
        store.set_failure(FakeFailure::SqlStatementTimeout);
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.info["overturn_hold_probe_error"], "statement_timeout");
        assert!(!gate.breaker_open(), "1 of 2");
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(
            gate.breaker_open(),
            "2 consecutive statement_timeouts open it"
        );
        assert_eq!(store.calls(), 6);
    }

                #[tokio::test]
    async fn half_open_trial_dying_on_unreachable_reopens_breaker() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let (gate, clock) = breaker_gate(store.clone(), 1, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);
        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open());
        assert_eq!(store.calls(), 1);

        clock.advance(PAST_OPEN);
        store.set_failure(FakeFailure::Unreachable);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 2, "trial reached the store");
        assert!(gate.breaker_open(), "unreachable on the trial re-opens");
        assert!(!gate.half_open_inflight(), "slot released");
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 2, "re-opened: skipped for another window");

        clock.advance(PAST_OPEN);
        store.set_failure(FakeFailure::SqlPermission);
        gate.evaluate(&c, 3, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 3);
        assert!(!gate.breaker_open(), "service answered: reachable, closed");
    }

                #[tokio::test]
    async fn decode_errors_are_neutral_for_the_breaker() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let (gate, clock) = breaker_gate(store.clone(), 2, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);
        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        store.set_failure(FakeFailure::Decode);
        for _ in 0..5 {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.info["overturn_hold_probe_error"], "decode");
        }
        assert!(!gate.breaker_open(), "decode errors never open");
        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open(), "count survived the decode errors");
        assert_eq!(store.calls(), 7);

        clock.advance(PAST_OPEN);
        store.set_failure(FakeFailure::Decode);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 8, "trial ran");
        assert!(gate.breaker_open(), "neutral: not closed");
        assert!(!gate.half_open_inflight(), "neutral: slot freed");
        store.set_fail(false);
        gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 9, "next request trials immediately");
        assert!(!gate.breaker_open());
    }

                #[tokio::test]
    async fn dropped_half_open_trial_frees_the_slot() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let (gate, clock) = breaker_gate(store.clone(), 1, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);
        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open());
        clock.advance(PAST_OPEN);

        store.set_hang(true);
        let dropped =
            tokio::time::timeout(Duration::from_millis(20), gate.check(&c, 2, TOPIC)).await;
        assert!(dropped.is_err(), "the trial was dropped mid-flight");
        assert_eq!(store.calls(), 2, "the trial had reached the store");
        assert!(gate.breaker_open(), "a dropped trial is not a success");
        assert!(!gate.half_open_inflight(), "guard freed the slot on drop");

        store.set_hang(false);
        store.set_fail(false);
        assert_eq!(gate.check(&c, 3, TOPIC).await, Some(GateVerdict::NotHeld));
        assert_eq!(store.calls(), 3, "next request trialled");
        assert!(!gate.breaker_open());

        store.set_fail(true);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert!(gate.breaker_open());
        clock.advance(PAST_OPEN);
        store.set_hang(true);
        let mut inflight = Box::pin(gate.check(&c, 4, TOPIC));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut inflight)
                .await
                .is_err(),
            "trial parked"
        );
        assert!(gate.half_open_inflight());
        let calls = store.calls();
        match gate.check(&c, 5, TOPIC).await {
            Some(GateVerdict::ProbeError(e)) => {
                assert_eq!(e.code, ProbeErrorCode::BreakerOpen, "{e}")
            }
            other => panic!("expected skip while the trial is in flight, got {other:?}"),
        }
        assert_eq!(
            store.calls(),
            calls,
            "second request skipped, not a second trial"
        );
        drop(inflight);
        assert!(
            !gate.half_open_inflight(),
            "dropping the parked trial frees the slot"
        );
    }

                        #[tokio::test]
    async fn dropped_probe_aborts_its_task() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg(GateMode::Enforce);
        let baseline = Arc::strong_count(&store);

        store.set_hang(true);
        let dropped =
            tokio::time::timeout(Duration::from_millis(20), gate.check(&c, 1, TOPIC)).await;
        assert!(dropped.is_err(), "the probe was dropped mid-flight");
        assert_eq!(store.calls(), 1, "the probe had reached the store");
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            Arc::strong_count(&store),
            baseline,
            "the aborted probe task released its store handle"
        );

        store.set_hang(false);
        assert_eq!(gate.check(&c, 2, TOPIC).await, Some(GateVerdict::NotHeld));
        assert_eq!(store.calls(), 2);
    }

                #[tokio::test]
    async fn stale_half_open_slot_is_taken_over() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![]);
        let (gate, clock) = breaker_gate(store.clone(), 1, DEFAULT_BREAKER_OPEN);
        let c = cfg(GateMode::Enforce);
        let bound = half_open_stale_after(c.timeout_ms);
        assert_eq!(bound, Duration::from_millis(200 + 200 + 1_000));

        let plant = |age: Duration| {
            let mut b = gate.breaker.lock().unwrap();
            b.open_until = Some(clock.now() - Duration::from_secs(1));
            b.trial_seq = 41;
            b.half_open = Some(HalfOpenSlot {
                seq: 41,
                since: clock.now() - age,
            });
        };
        plant(Duration::from_millis(0));
        match gate.check(&c, 1, TOPIC).await {
            Some(GateVerdict::ProbeError(e)) => {
                assert_eq!(e.code, ProbeErrorCode::BreakerOpen, "{e}")
            }
            other => panic!("expected skip, got {other:?}"),
        }
        assert_eq!(store.calls(), 0);
        plant(bound + Duration::from_millis(1));
        assert_eq!(gate.check(&c, 1, TOPIC).await, Some(GateVerdict::NotHeld));
        assert_eq!(store.calls(), 1);
        assert!(!gate.breaker_open());
        assert!(!gate.half_open_inflight());
        plant(Duration::from_millis(0));
        {
            let mut b = gate.breaker.lock().unwrap();
            b.trial_seq = 42;
            b.half_open.as_mut().unwrap().seq = 42;
        }
        drop(HalfOpenTrial {
            gate: &gate,
            seq: 41,
            armed: true,
        });
        assert!(
            gate.half_open_inflight(),
            "stale guard left the newer slot alone"
        );
    }


    fn perm_policy(groups: &[i32]) -> HoldGateConfig {
        HoldGateConfig {
            mode: GateMode::Enforce,
            perm_suspend_case_groups: groups.to_vec(),
            ..HoldGateConfig::default()
        }
    }

            #[tokio::test]
    async fn empty_case_group_list_holds_every_shape() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg(GateMode::Enforce);
        assert!(c.perm_suspend_case_groups.is_empty());
        let before = counter("enforce", "held_ignored_policy");
        for shape in [TEMP, PERM] {
            let out = gate
                .evaluate(&c, 1, TOPIC, std::slice::from_ref(&shape))
                .await;
            assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED), "{shape:?}");
            assert_eq!(out.info["overturn_hold_case_group_id"], "63");
            assert!(!out.info.contains_key("overturn_hold_ignored_case_group_id"));
        }
        assert_eq!(counter("enforce", "held_ignored_policy"), before);
    }

            #[tokio::test]
    async fn listed_case_group_or_manual_hold_holds_a_perm_suspend() {
        let _serial = METRICS_LOCK.lock().await;
        let mut manual = hold(2, 20, 30);
        manual.case_group_id = None;
        manual.reason = "manual".into();
        let mut cse = hold(3, 30, 30);
        cse.case_group_id = Some(54);
        let store = FakeHoldStore::new(vec![hold(1, 10, 30), manual, cse]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = perm_policy(&[54, 53]);

        let out = gate.evaluate(&c, 3, TOPIC, &[PERM]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_case_group_id"], "54");
        let out = gate.evaluate(&c, 2, TOPIC, &[PERM]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert!(!out.info.contains_key("overturn_hold_case_group_id"));
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_case_group_id"], "63");
    }

                #[tokio::test]
    async fn unlisted_case_group_is_ignored_for_a_perm_suspend() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = perm_policy(&[54]);
        let before_ignored = counter("enforce", "held_ignored_policy");
        let before_blocked = counter("enforce", "held_blocked");
        let out = gate.evaluate(&c, 1, TOPIC, &[PERM]).await;
        assert_eq!(out.skip_status, None, "perm suspend proceeds");
        assert_eq!(out.info["overturn_hold_ignored_case_group_id"], "63");
        assert_eq!(out.info["overturn_hold_ignored_hold_id"], "10");
        assert!(!out.info.contains_key("overturn_hold_id"));
        assert!(!out.info.contains_key("overturn_hold_would_block"));
        assert_eq!(out.info["overturn_hold_mode"], "enforce");
        assert_eq!(
            counter("enforce", "held_ignored_policy"),
            before_ignored + 1
        );
        assert_eq!(counter("enforce", "held_blocked"), before_blocked);
        match gate.check(&c, 1, TOPIC).await {
            Some(GateVerdict::Held(h)) => assert_eq!(h.hold_id, 10),
            other => panic!("expected Held, got {other:?}"),
        }
        let c_shadow = HoldGateConfig {
            mode: GateMode::Shadow,
            ..perm_policy(&[54])
        };
        let out = gate.evaluate(&c_shadow, 1, TOPIC, &[PERM]).await;
        assert_eq!(out.skip_status, None);
        assert_eq!(out.info["overturn_hold_ignored_case_group_id"], "63");
        assert!(!out.info.contains_key("overturn_hold_would_block"));
        let mut orphan = hold(4, 40, 30);
        orphan.case_group_id = None;
        orphan.reason = "backfill_bq".into();
        let store = FakeHoldStore::new(vec![orphan]);
        let gate = HoldGate::new(Some(store));
        let out = gate.evaluate(&c, 4, TOPIC, &[PERM]).await;
        assert_eq!(out.skip_status, None);
        assert_eq!(out.info["overturn_hold_ignored_case_group_id"], "none");
    }

                #[tokio::test]
    async fn policy_filters_before_longest_wins_and_cache_keeps_all_holds() {
        let _serial = METRICS_LOCK.lock().await;
        let long_spam = hold(5, 50, 80);
        let mut short_cse = hold(5, 51, 5);
        short_cse.case_group_id = Some(54);
        let store = FakeHoldStore::new(vec![long_spam, short_cse]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = perm_policy(&[54]);

        let out = gate.evaluate(&c, 5, TOPIC, &[PERM]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_id"], "51");
        assert_eq!(out.info["overturn_hold_case_group_id"], "54");
        assert_eq!(store.calls(), 1);
        let out = gate.evaluate(&c, 5, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_id"], "50");
        assert_eq!(out.info["overturn_hold_probe_ms"], "0", "cache hit");
        assert_eq!(store.calls(), 1, "no re-probe");
        let out = gate.evaluate(&c, 5, TOPIC, &[PERM]).await;
        assert_eq!(out.info["overturn_hold_id"], "51");
        assert_eq!(store.calls(), 1);
    }

            #[test]
    fn perm_suspend_case_groups_parse() {
        let c = json!({"overturn_hold_gate": {
            "mode": "enforce", "perm_suspend_case_groups": [54, 53, 54]
        }});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.mode, GateMode::Enforce);
        assert_eq!(parsed.perm_suspend_case_groups, vec![54, 53]);
        for all in [
            json!({"overturn_hold_gate": {"mode": "enforce"}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": null}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": []}}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&all));
            assert_eq!(parsed.mode, GateMode::Enforce, "{all}");
            assert!(parsed.perm_suspend_case_groups.is_empty(), "{all}");
        }
        for bad in [
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": "54"}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": 54}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": ["54"]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": [54.5]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": [54, null]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "perm_suspend_case_groups": [99999999999_i64]}}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&bad));
            assert_eq!(parsed, HoldGateConfig::default(), "{bad}");
        }
        let c = perm_policy(&[54]);
        let mut h = hold(1, 1, 1);
        assert!(c.hold_applies(&h, false));
        assert!(!c.hold_applies(&h, true), "cg 63 ∉ [54]");
        h.case_group_id = Some(54);
        assert!(c.hold_applies(&h, true));
        h.case_group_id = None;
        assert!(!c.hold_applies(&h, true));
        h.reason = "manual".into();
        assert!(c.hold_applies(&h, true));
        assert!(
            cfg(GateMode::Enforce).hold_applies(&hold(1, 1, 1), true),
            "empty list"
        );
    }


                    #[test]
    fn kinds_and_labels_parse() {
        let c = json!({"overturn_hold_gate": {"mode": "enforce"}});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.kinds, vec![GateKind::Suspend]);
        assert!(parsed.labels.is_empty());
        assert!(parsed.gates_suspend());
        assert!(!parsed.gates_label("SpamHighRecall"));
        for all in [
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": null, "labels": null}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "labels": []}}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&all));
            assert_eq!(parsed.kinds, vec![GateKind::Suspend], "{all}");
            assert!(parsed.labels.is_empty(), "{all}");
        }

        let c = json!({"overturn_hold_gate": {
            "mode": "shadow",
            "kinds": ["suspend", "label"],
            "labels": ["SpamHighRecall"],
        }});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.mode, GateMode::Shadow);
        assert_eq!(parsed.kinds, vec![GateKind::Suspend, GateKind::Label]);
        assert_eq!(parsed.labels, vec!["SpamHighRecall".to_owned()]);
        assert!(parsed.gates_suspend());
        assert!(parsed.gates_label("SpamHighRecall"));
        assert!(!parsed.gates_label("SpamMediumRecall"));

        let c = json!({"overturn_hold_gate": {
            "mode": "enforce",
            "kinds": [" Label ", "SUSPEND", "label"],
            "labels": ["SpamHighRecall", "", " SpamHighRecall ", "  Other  "],
        }});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.kinds, vec![GateKind::Label, GateKind::Suspend]);
        assert_eq!(
            parsed.labels,
            vec!["SpamHighRecall".to_owned(), "Other".to_owned()]
        );

        let c = json!({"overturn_hold_gate": {
            "mode": "enforce", "kinds": ["label"], "labels": ["SpamHighRecall"]
        }});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert!(!parsed.gates_suspend());
        assert!(parsed.gates_label("SpamHighRecall"));

        for inert in [
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": ["suspend", "label"]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": ["suspend", "label"], "labels": []}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": ["label"], "labels": null}}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&inert));
            assert_eq!(parsed.mode, GateMode::Enforce, "{inert}");
            assert!(parsed.kinds.contains(&GateKind::Label), "{inert}");
            assert!(parsed.labels.is_empty(), "{inert}");
            assert!(!parsed.gates_label("SpamHighRecall"), "{inert}");
        }
        let c = json!({"overturn_hold_gate": {"mode": "enforce", "labels": ["SpamHighRecall"]}});
        let parsed = HoldGateConfig::from_config(Some(&c));
        assert_eq!(parsed.labels, vec!["SpamHighRecall".to_owned()]);
        assert!(!parsed.gates_label("SpamHighRecall"));

        for bad in [
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": []}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": "suspend"}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": ["suspend", "bounce"]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": ["labels"]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": ["suspend", 1]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": ["suspend", null]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": [""]}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": {"label": true}}}),
            json!({"overturn_hold_gate": {"mode": "enforce", "kinds": 7}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "labels": "SpamHighRecall"}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "labels": ["SpamHighRecall", 1]}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "labels": ["SpamHighRecall", null]}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "labels": [["SpamHighRecall"]]}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "labels": {"a": 1}}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "labels": ["", " "]}}),
            json!({"overturn_hold_gate": {"mode": "shadow", "labels": [""]}}),
        ] {
            let parsed = HoldGateConfig::from_config(Some(&bad));
            assert_eq!(parsed, HoldGateConfig::default(), "{bad}");
            assert_eq!(parsed.mode, GateMode::Off, "{bad}");
        }
        let c = json!({"overturn_hold_gate": {"mode": "off", "kinds": [], "labels": 1}});
        assert_eq!(
            HoldGateConfig::from_config(Some(&c)),
            HoldGateConfig::default()
        );
    }

                        #[tokio::test]
    async fn enforce_held_label_is_stripped_and_the_rest_proceeds() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![label_hold_row(1, 100, 30, SHR)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg_labels(GateMode::Enforce, &[SHR]);
        let before = counter("enforce", "held_label_stripped");
        let before_not_held = counter("enforce", "not_held");
        let before_blocked = counter("enforce", "held_blocked");
        let out = gate
            .evaluate(&c, 1, TOPIC, &[TEMP, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, None, "a label hold never blocks a suspend");
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(out.info["overturn_hold_labels_stripped"], SHR);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "100");
        assert_eq!(out.info["overturn_hold_mode"], "enforce");
        assert!(out.info.contains_key("overturn_hold_probe_ms"));
        assert!(
            !out.info.contains_key("overturn_hold_id"),
            "no suspend hold"
        );
        assert!(!out.info.contains_key("overturn_hold_would_block"));
        assert!(!out.info.contains_key("overturn_hold_label_would_strip"));
        assert_eq!(counter("enforce", "held_label_stripped"), before + 1);
        assert_eq!(
            counter("enforce", "not_held"),
            before_not_held,
            "label row instead"
        );
        assert_eq!(counter("enforce", "held_blocked"), before_blocked);
        assert_eq!(store.label_requests(), vec![vec![SHR.to_owned()]]);

        let mut out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.skip_status, None);
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(out.strip_hold_ids, vec![100]);
        assert_eq!(out.info["overturn_hold_labels_stripped"], SHR);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "100");
        assert!(
            !out.info.contains_key("overturn_hold_kind"),
            "not collapsed yet"
        );
        assert!(!out.info.contains_key("overturn_hold_id"));
        assert_eq!(store.calls(), 1, "second lookup served from the cache");
        out.mark_label_collapse();
        assert_eq!(out.info["overturn_hold_kind"], "label");
        assert_eq!(out.info["overturn_hold_id"], "100");
        assert_eq!(out.info["overturn_hold_label"], SHR);
        assert_eq!(out.info["overturn_hold_labels_stripped"], SHR);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "100");

        let store = FakeHoldStore::new(vec![
            label_hold_row(2, 200, 30, SHR),
            label_hold_row(2, 201, 30, "NotGated"),
        ]);
        let gate = HoldGate::new(Some(store.clone()));
        let c2 = cfg_labels(GateMode::Enforce, &[SHR, "SpamMediumRecall"]);
        let out = gate
            .evaluate(
                &c2,
                2,
                TOPIC,
                &[label_action("SpamMediumRecall"), label_action(SHR)],
            )
            .await;
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(out.info["overturn_hold_labels_stripped"], SHR);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "200");
        assert_eq!(
            store.label_requests(),
            vec![vec![SHR.to_owned(), "SpamMediumRecall".to_owned()]],
            "sorted, deduplicated request set"
        );
        let store = FakeHoldStore::new(vec![
            label_hold_row(3, 300, 30, SHR),
            label_hold_row(3, 301, 30, "SpamMediumRecall"),
        ]);
        let gate = HoldGate::new(Some(store));
        let out = gate
            .evaluate(
                &c2,
                3,
                TOPIC,
                &[label_action(SHR), label_action("SpamMediumRecall")],
            )
            .await;
        assert_eq!(
            out.strip_labels,
            vec![SHR.to_owned(), "SpamMediumRecall".to_owned()]
        );
        assert_eq!(
            out.info["overturn_hold_labels_stripped"],
            "SpamHighRecall,SpamMediumRecall"
        );
        assert_eq!(out.info["overturn_hold_label_hold_id"], "300,301");
        assert_eq!(out.strip_hold_ids, vec![300, 301]);
        let mut out = out;
        out.mark_label_collapse();
        assert_eq!(out.info["overturn_hold_kind"], "label");
        assert_eq!(out.info["overturn_hold_id"], "300");
        assert_eq!(out.info["overturn_hold_label"], SHR);
        assert_eq!(
            out.info["overturn_hold_labels_stripped"],
            "SpamHighRecall,SpamMediumRecall"
        );
        assert_eq!(out.info["overturn_hold_label_hold_id"], "300,301");
    }

                #[tokio::test]
    async fn shadow_held_label_tags_without_stripping() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![label_hold_row(1, 100, 30, SHR)]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg_labels(GateMode::Shadow, &[SHR]);
        let before = counter("shadow", "held_label_shadow");
        let before_stripped = counter("shadow", "held_label_stripped");
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.skip_status, None);
        assert!(
            out.strip_labels.is_empty(),
            "shadow never changes the decision"
        );
        assert_eq!(out.info["overturn_hold_label_would_strip"], SHR);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "100");
        assert_eq!(out.info["overturn_hold_mode"], "shadow");
        assert!(!out.info.contains_key("overturn_hold_labels_stripped"));
        assert!(!out.info.contains_key("overturn_hold_would_block"));
        assert!(!out.info.contains_key("overturn_hold_kind"));
        assert!(
            out.strip_hold_ids.is_empty(),
            "nothing to collapse in shadow"
        );
        let mut marked = out.clone();
        marked.mark_label_collapse();
        assert_eq!(marked, out, "no-op without stripped labels");
        assert_eq!(counter("shadow", "held_label_shadow"), before + 1);
        assert_eq!(counter("shadow", "held_label_stripped"), before_stripped);
    }

                        #[tokio::test]
    async fn holds_are_kind_matched() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![
            hold(1, 10, 30),
            label_hold_row(2, 20, 30, SHR),
            hold(3, 30, 30),
            label_hold_row(3, 31, 60, SHR),
        ]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg_labels(GateMode::Enforce, &[SHR]);

        let before_not_held = counter("enforce", "not_held");
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.skip_status, None);
        assert!(out.strip_labels.is_empty());
        assert!(!out.info.contains_key("overturn_hold_labels_stripped"));
        assert!(!out.info.contains_key("overturn_hold_label_hold_id"));
        assert!(
            !out.info.contains_key("overturn_hold_id"),
            "no suspend was requested"
        );
        assert_eq!(counter("enforce", "not_held"), before_not_held + 1);

        let out = gate
            .evaluate(&c, 1, TOPIC, &[TEMP, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_id"], "10");
        assert!(out.strip_labels.is_empty());
        assert!(!out.info.contains_key("overturn_hold_labels_stripped"));

        let before_not_held = counter("enforce", "not_held");
        let out = gate.evaluate(&c, 2, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, None);
        assert!(!out.info.contains_key("overturn_hold_id"));
        assert!(!out.info.contains_key("overturn_hold_label_hold_id"));
        assert_eq!(counter("enforce", "not_held"), before_not_held + 1);
        assert_eq!(
            store.label_requests().last().unwrap(),
            &Vec::<String>::new()
        );

        let out = gate
            .evaluate(&c, 2, TOPIC, &[PERM, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, None);
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert!(!out.info.contains_key("overturn_hold_id"));

        let before_blocked = counter("enforce", "held_blocked");
        let before_stripped = counter("enforce", "held_label_stripped");
        let out = gate
            .evaluate(&c, 3, TOPIC, &[TEMP, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_id"], "30");
        assert_eq!(out.info["overturn_hold_kind"], "suspend");
        assert!(!out.info.contains_key("overturn_hold_label"));
        assert!(out.strip_labels.is_empty());
        assert!(!out.info.contains_key("overturn_hold_label_hold_id"));
        assert_eq!(counter("enforce", "held_blocked"), before_blocked + 1);
        assert_eq!(counter("enforce", "held_label_stripped"), before_stripped);

        let c_shadow = cfg_labels(GateMode::Shadow, &[SHR]);
        let before_shadow = counter("shadow", "held_shadow");
        let before_label_shadow = counter("shadow", "held_label_shadow");
        let out = gate
            .evaluate(&c_shadow, 3, TOPIC, &[TEMP, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, None);
        assert_eq!(out.info["overturn_hold_would_block"], "true");
        assert_eq!(out.info["overturn_hold_id"], "30");
        assert_eq!(out.info["overturn_hold_label_would_strip"], SHR);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "31");
        assert_eq!(counter("shadow", "held_shadow"), before_shadow + 1);
        assert_eq!(
            counter("shadow", "held_label_shadow"),
            before_label_shadow + 1
        );

        let c_policy = HoldGateConfig {
            perm_suspend_case_groups: vec![54],
            ..cfg_labels(GateMode::Enforce, &[SHR])
        };
        let out = gate
            .evaluate(&c_policy, 3, TOPIC, &[PERM, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, None);
        assert_eq!(out.info["overturn_hold_ignored_hold_id"], "30");
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "31");
    }

            #[tokio::test]
    async fn label_holds_are_name_matched_and_longest_wins() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![
            label_hold_row(1, 10, 30, "SpamMediumRecall"),
            label_hold_row(2, 20, 5, SHR),
            label_hold_row(2, 21, 80, SHR),
        ]);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg_labels(GateMode::Enforce, &[SHR, "SpamMediumRecall"]);
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert!(out.strip_labels.is_empty(), "hold is on another label");
        assert!(!out.info.contains_key("overturn_hold_label_hold_id"));
        let out = gate.evaluate(&c, 2, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(
            out.info["overturn_hold_label_hold_id"], "21",
            "longest wins"
        );
    }

                        #[tokio::test]
    async fn cache_entry_covers_only_the_labels_it_was_probed_for() {
        let _serial = METRICS_LOCK.lock().await;
        let clock = ManualClock::new();
        let store = FakeHoldStore::new(vec![hold(1, 10, 30), label_hold_row(1, 11, 30, SHR)]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        let c = cfg_labels(GateMode::Enforce, &[SHR]);

        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(store.calls(), 1);
        assert_eq!(store.label_requests(), vec![Vec::<String>::new()]);
        assert_eq!(gate.cached_len(), 1);
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(out.info["overturn_hold_label_hold_id"], "11");
        assert_eq!(
            store.calls(),
            2,
            "suspend-only entry does not cover a label"
        );
        assert_eq!(
            store.label_requests(),
            vec![Vec::<String>::new(), vec![SHR.to_owned()]]
        );
        assert_eq!(gate.cached_len(), 1, "replaced, not added");
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(out.info["overturn_hold_probe_ms"], "0");
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(out.info["overturn_hold_probe_ms"], "0");
        let out = gate
            .evaluate(&c, 1, TOPIC, &[TEMP, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(store.calls(), 2, "all three served from the cache");
        let c2 = cfg_labels(GateMode::Enforce, &[SHR, "SpamMediumRecall"]);
        gate.evaluate(&c2, 1, TOPIC, &[label_action("SpamMediumRecall")])
            .await;
        assert_eq!(store.calls(), 3);
        gate.cache.invalidate(&1);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(store.calls(), 4);

        let store = FakeHoldStore::new(vec![hold(5, 50, 30)]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        let out = gate
            .evaluate(&c, 5, TOPIC, &[TEMP, label_action(SHR)])
            .await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        let out = gate.evaluate(&c, 5, TOPIC, &[label_action(SHR)]).await;
        assert!(out.strip_labels.is_empty());
        assert_eq!(store.calls(), 1, "covered by the [SHR] entry");

        let store = FakeHoldStore::new(vec![label_hold_row(6, 60, 30, SHR)]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        gate.evaluate(&c, 6, TOPIC, &[label_action(SHR)]).await;
        let out = gate.evaluate(&c, 6, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, None);
        assert_eq!(store.calls(), 1);
    }

            #[tokio::test]
    async fn label_probe_errors_follow_on_probe_error() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![label_hold_row(1, 10, 30, SHR)]);
        store.set_fail(true);
        let gate = HoldGate::new(Some(store.clone()));
        let c = cfg_labels(GateMode::Enforce, &[SHR]);
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.skip_status, None, "allow: fail-open");
        assert!(out.strip_labels.is_empty());
        assert_eq!(out.info["overturn_hold_probe_error"], "connect_refused");
        assert!(!out.info.contains_key("overturn_hold_label_hold_id"));
        let c_skip = HoldGateConfig {
            on_probe_error: OnProbeError::Skip,
            ..c
        };
        let out = gate.evaluate(&c_skip, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_LOOKUP_FAILED));
        assert!(out.strip_labels.is_empty());
    }

                    #[tokio::test]
    async fn empty_reprobe_after_coverage_miss_drops_the_stale_entry() {
        let _serial = METRICS_LOCK.lock().await;
        let clock = ManualClock::new();
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::with_clock(Some(store.clone()), clock.clone());
        let c = cfg_labels(GateMode::Enforce, &[SHR]);
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
        assert_eq!(gate.cached_len(), 1);
        store.set_holds(vec![]);
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert!(out.strip_labels.is_empty());
        assert_eq!(store.calls(), 2);
        assert_eq!(gate.cached_len(), 0, "stale positive entry dropped");
        let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(out.skip_status, None);
        assert_eq!(store.calls(), 3);
        assert_eq!(gate.cached_len(), 0, "negatives are still never cached");
        store.set_holds(vec![hold(1, 10, 30), label_hold_row(1, 11, 30, SHR)]);
        gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
        assert_eq!(gate.cached_len(), 1);
        let out = gate.evaluate(&c, 1, TOPIC, &[label_action(SHR)]).await;
        assert_eq!(out.strip_labels, vec![SHR.to_owned()]);
        assert_eq!(gate.cached_len(), 1, "replaced with the covered entry");
        assert_eq!(store.calls(), 5);
    }

                                    #[test]
    fn cache_eviction_and_writes_are_compare_and_swap_on_probe_start() {
        let clock = ManualClock::new();
        let gate = HoldGate::with_clock(Some(FakeHoldStore::new(vec![])), clock.clone());
        let shr = vec![SHR.to_owned()];
        let t = |secs: u64| Duration::from_secs(secs);

        let e_started = clock.now();
        clock.advance(t(1));
        let w_started = clock.now();
        clock.advance(t(1));
        gate.remember(1, vec![hold(1, 10, 30)], &shr, 60, w_started);
        assert_eq!(gate.cached_snapshot(1), Some((vec![10], shr.clone())));
        clock.advance(t(1));
        gate.forget_if_older(1, e_started);
        assert_eq!(
            gate.cached_snapshot(1),
            Some((vec![10], shr.clone())),
            "(a) an entry written after the empty probe started survives"
        );

        clock.advance(t(1));
        let e2_started = clock.now();
        clock.advance(t(1));
        gate.forget_if_older(1, e2_started);
        assert_eq!(gate.cached_snapshot(1), None, "(b) older entry evicted");
        gate.forget_if_older(1, e2_started);
        assert_eq!(gate.cached_len(), 0);

        clock.advance(t(5));
        let p1_started = clock.now();
        clock.advance(t(1));
        let p2_started = clock.now();
        clock.advance(t(1));
        gate.remember(
            2,
            vec![hold(2, 20, 30), label_hold_row(2, 21, 30, SHR)],
            &shr,
            60,
            p2_started,
        );
        clock.advance(t(1));
        gate.remember(2, vec![hold(2, 20, 30)], &[], 60, p1_started);
        assert_eq!(
            gate.cached_snapshot(2),
            Some((vec![20, 21], shr.clone())),
            "(c) the later-started probe's write is kept over the earlier one"
        );
        clock.advance(t(1));
        let q1_started = clock.now();
        clock.advance(t(1));
        let q2_started = clock.now();
        clock.advance(t(1));
        gate.remember(3, vec![hold(3, 30, 30)], &[], 60, q1_started);
        clock.advance(t(1));
        gate.remember(
            3,
            vec![hold(3, 30, 30), label_hold_row(3, 31, 30, SHR)],
            &shr,
            60,
            q2_started,
        );
        assert_eq!(
            gate.cached_snapshot(3),
            Some((vec![30], Vec::new())),
            "an entry written after the writer's start is never clobbered"
        );

        let s = clock.now();
        gate.remember(4, vec![hold(4, 40, 30)], &[], 60, s);
        gate.remember(
            4,
            vec![hold(4, 40, 30), label_hold_row(4, 41, 30, SHR)],
            &shr,
            60,
            s,
        );
        assert_eq!(gate.cached_snapshot(4), Some((vec![40, 41], shr.clone())));
        gate.forget_if_older(4, s);
        assert_eq!(
            gate.cached_snapshot(4),
            None,
            "same-instant empty probe evicts"
        );
    }

            #[tokio::test]
    async fn empty_action_list_is_a_no_op() {
        let _serial = METRICS_LOCK.lock().await;
        let store = FakeHoldStore::new(vec![hold(1, 10, 30)]);
        let gate = HoldGate::new(Some(store.clone()));
        assert_eq!(
            gate.evaluate(&cfg(GateMode::Enforce), 1, TOPIC, &[]).await,
            GateOutcome::default()
        );
        assert_eq!(store.calls(), 0);
    }

    #[tokio::test]
    async fn longest_expires_at_wins_for_multiple_holds() {
        let _serial = METRICS_LOCK.lock().await;
        let mut short = hold(20, 201, 5);
        short.head = Some("ShortHead".into());
        let long = hold(20, 202, 80);
        let store = FakeHoldStore::new(vec![short, long.clone()]);
        let gate = HoldGate::new(Some(store));
        match gate.check(&cfg(GateMode::Enforce), 20, TOPIC).await {
            Some(GateVerdict::Held(h)) => assert_eq!(h, long),
            other => panic!("expected Held, got {other:?}"),
        }
    }

                            #[test]
    fn probe_error_codes_are_closed_and_stable() {
        use ProbeErrorCode::*;
        let fwd = |status: u16, code: LedgerCode| HoldStoreError::Upstream {
            status,
            code: Some(code),
        };
        let bare = |status: u16| HoldStoreError::Upstream { status, code: None };
        let cases = [
            (
                HoldStoreError::NotConfigured,
                NotConfigured,
                "not_configured",
            ),
            (HoldStoreError::BadUrl("x".into()), BadUrl, "bad_url"),
            (
                HoldStoreError::Unreachable("x".into()),
                LedgerUnreachable,
                "ledger_unreachable",
            ),
            (
                HoldStoreError::Timeout(Duration::from_millis(1)),
                Timeout,
                "timeout",
            ),
            (
                fwd(503, LedgerCode::ConnectRefused),
                ConnectRefused,
                "connect_refused",
            ),
            (fwd(503, LedgerCode::AuthFailed), AuthFailed, "auth_failed"),
            (fwd(503, LedgerCode::Tls), Tls, "tls"),
            (
                fwd(504, LedgerCode::StatementTimeout),
                StatementTimeout,
                "statement_timeout",
            ),
            (
                fwd(503, LedgerCode::ServerUnhealthy),
                ServerUnhealthy,
                "server_unhealthy",
            ),
            (fwd(503, LedgerCode::PoolWait), PoolWait, "pool_wait"),
            (fwd(500, LedgerCode::Other), Other, "other"),
            (fwd(500, LedgerCode::Decode), Decode, "decode"),
            (fwd(504, LedgerCode::Timeout), Timeout, "timeout"),
            (bare(504), Timeout, "timeout"),
            (bare(502), LedgerStatus, "ledger_status"),
            (bare(404), LedgerStatus, "ledger_status"),
            (HoldStoreError::Decode("x".into()), Decode, "decode"),
        ];
        for (err, code, s) in cases {
            assert_eq!(err.code(), code, "{err}");
            assert_eq!(code.as_str(), s);
            let p = ProbeError::from(&err);
            assert_eq!(p.code, code);
            assert_eq!(p.detail, err.to_string(), "raw text kept for the log only");
        }
        let b = ProbeError::breaker_open(7);
        assert_eq!(b.code, BreakerOpen);
        assert_eq!(BreakerOpen.as_str(), "breaker_open");
        assert!(b.detail.contains("7s"));
        let r = ProbeError::reachable_backoff(3, 7, Some(LedgerStatus));
        assert_eq!(r.code, ReachableBackoff);
        assert_eq!(ReachableBackoff.as_str(), "reachable_backoff");
        assert!(r.detail.contains("7 consecutive"), "{r}");
        assert!(r.detail.contains("last: ledger_status"), "{r}");
        assert!(r.detail.contains("next in 3s"), "{r}");
        assert!(
            ProbeError::reachable_backoff(1, 5, None)
                .detail
                .contains("last: ?")
        );
        for code in [
            ConnectRefused,
            AuthFailed,
            StatementTimeout,
            ServerUnhealthy,
            PoolWait,
            BreakerOpen,
            ReachableBackoff,
            BadUrl,
            NotConfigured,
            Timeout,
            Decode,
            Tls,
            LedgerUnreachable,
            LedgerStatus,
            Other,
        ] {
            assert!(!matches!(code.as_str(), "transport" | "bad_dsn"));
        }
    }

                    #[test]
    fn ledger_codes_parse_into_the_closed_set() {
        for (raw, want) in [
            ("connect_refused", LedgerCode::ConnectRefused),
            ("auth_failed", LedgerCode::AuthFailed),
            ("tls", LedgerCode::Tls),
            ("statement_timeout", LedgerCode::StatementTimeout),
            ("server_unhealthy", LedgerCode::ServerUnhealthy),
            ("pool_wait", LedgerCode::PoolWait),
            ("decode", LedgerCode::Decode),
            ("timeout", LedgerCode::Timeout),
            ("other", LedgerCode::Other),
            (" other ", LedgerCode::Other),
            ("bad_request", LedgerCode::Other),
            ("transport", LedgerCode::Other),
            ("breaker_open", LedgerCode::Other),
            ("ledger_unreachable", LedgerCode::Other),
            ("", LedgerCode::Other),
            ("CONNECT_REFUSED", LedgerCode::Other),
        ] {
            assert_eq!(LedgerCode::parse(raw), want, "{raw:?}");
            assert_eq!(
                LedgerCode::parse(raw).probe_code().as_str(),
                LedgerCode::parse(want.probe_code().as_str())
                    .probe_code()
                    .as_str(),
                "round-trips through its own spelling"
            );
        }
        assert_eq!(GateKind::Suspend.as_str(), "suspend");
        assert_eq!(GateKind::Label.as_str(), "label");
    }

                        #[tokio::test]
    async fn publish_config_drives_the_mode_and_config_gauges() {
        let _serial = METRICS_LOCK.lock().await;
        let gate = HoldGate::new(None);
        let mode_gauge = |m: &str| metrics::HOLD_GATE_MODE.with_label_values(&[m]).get();
        let info_gauge = |m: &str, p: &str| {
            metrics::HOLD_GATE_CONFIG_INFO
                .with_label_values(&[m, p, "suspend"])
                .get()
        };

        assert!(
            gate.publish_config_changed(&cfg(GateMode::Off)),
            "boot: first publish is a change from nothing"
        );
        assert_eq!(
            (
                mode_gauge("off"),
                mode_gauge("shadow"),
                mode_gauge("enforce")
            ),
            (1, 0, 0)
        );
        assert_eq!(info_gauge("off", "allow"), 1);
        assert!(
            !gate.publish_config_changed(&cfg(GateMode::Off)),
            "same config republished: no log"
        );

        assert!(gate.publish_config_changed(&cfg(GateMode::Shadow)));
        assert_eq!(
            (
                mode_gauge("off"),
                mode_gauge("shadow"),
                mode_gauge("enforce")
            ),
            (0, 1, 0)
        );
        assert_eq!(info_gauge("off", "allow"), 0, "previous pair zeroed");
        assert_eq!(info_gauge("shadow", "allow"), 1);

        let c = HoldGateConfig {
            mode: GateMode::Enforce,
            on_probe_error: OnProbeError::Skip,
            ..HoldGateConfig::default()
        };
        assert!(gate.publish_config_changed(&c));
        assert!(!gate.publish_config_changed(&c), "idempotent");
        assert_eq!(
            (
                mode_gauge("off"),
                mode_gauge("shadow"),
                mode_gauge("enforce")
            ),
            (0, 0, 1)
        );
        assert_eq!(info_gauge("shadow", "allow"), 0);
        assert_eq!(info_gauge("enforce", "skip"), 1);
        assert_eq!(info_gauge("enforce", "allow"), 0);

        let mut c2 = c.clone();
        c2.topics = vec!["abuse.v3.score_results".into()];
        assert!(gate.publish_config_changed(&c2), "topics change logs");
        assert!(!gate.publish_config_changed(&c2));
        let mut c3 = c2.clone();
        c3.perm_suspend_case_groups = vec![54, 53];
        assert!(gate.publish_config_changed(&c3), "case-group change logs");
        let mut c4 = c3.clone();
        c4.timeout_ms = 350;
        assert!(gate.publish_config_changed(&c4), "timeout change logs");
        let mut c5 = c4.clone();
        c5.positive_cache_secs = 5;
        assert!(gate.publish_config_changed(&c5), "cache TTL change logs");
        assert!(!gate.publish_config_changed(&c5));
        assert_eq!(info_gauge("enforce", "skip"), 1, "gauge pair untouched");
        assert_eq!(mode_gauge("enforce"), 1);
        let mut c6 = c5.clone();
        c6.on_probe_error = OnProbeError::Allow;
        assert!(gate.publish_config_changed(&c6));
        assert_eq!(info_gauge("enforce", "skip"), 0);
        assert_eq!(info_gauge("enforce", "allow"), 1);
        let mut c7 = c6.clone();
        c7.kinds = vec![GateKind::Suspend, GateKind::Label];
        c7.labels = vec!["SpamHighRecall".into()];
        assert!(gate.publish_config_changed(&c7), "kinds/labels change logs");
        assert!(!gate.publish_config_changed(&c7));
        assert_eq!(info_gauge("enforce", "allow"), 0, "previous triple zeroed");
        assert_eq!(
            metrics::HOLD_GATE_CONFIG_INFO
                .with_label_values(&["enforce", "allow", "suspend,label"])
                .get(),
            1
        );
        let mut c8 = c7.clone();
        c8.labels.push("SpamMediumRecall".into());
        assert!(gate.publish_config_changed(&c8), "labels-only change logs");
        assert_eq!(
            metrics::HOLD_GATE_CONFIG_INFO
                .with_label_values(&["enforce", "allow", "suspend,label"])
                .get(),
            1,
            "gauge triple untouched by a labels change"
        );
    }


                    #[tokio::test]
    async fn bad_url_is_refused_at_construction() {
        let _serial = METRICS_LOCK.lock().await;
        for (url, why) in [
            ("not a url", "does not parse"),
            ("ftp://ledger:8080", "scheme must be http or https"),
            ("http://", "does not parse"),
            ("file:///etc/passwd", "scheme must be http or https"),
        ] {
            let store = HttpHoldStore::new(url, Some("prod"));
            let reason = store.ready.as_ref().expect_err("bad URL");
            assert!(reason.contains(why), "{url}: {reason}");
            assert_eq!(store.endpoint(), "url=<bad URL>");
            let err = store.active_holds(&[1], &[], 200).await.unwrap_err();
            assert!(matches!(err, HoldStoreError::BadUrl(_)), "{url}: {err}");
            assert_eq!(err.code(), ProbeErrorCode::BadUrl);
            assert_eq!(err.breaker_signal(), BreakerSignal::Neutral);
            assert!(matches!(
                store.ping(Duration::from_secs(1)).await,
                Err(HoldStoreError::BadUrl(_))
            ));
        }
        let store = HttpHoldStore::new(" http://ledger.example.invalid:8080/ ", None);
        let ep = store.ready.as_ref().expect("good URL");
        assert_eq!(
            ep.lookup_url.as_str(),
            "http://ledger.example.invalid:8080/v1/holds/lookup"
        );
        assert_eq!(
            ep.ready_url.as_str(),
            "http://ledger.example.invalid:8080/readyz"
        );
        assert_eq!(store.endpoint(), "url=http://ledger.example.invalid:8080");
        let store = Arc::new(HttpHoldStore::new("not a url", None));
        let gate = HoldGate::with_breaker(Some(store), 1, Duration::from_secs(60));
        let c = cfg(GateMode::Enforce);
        for _ in 0..3 {
            let out = gate.evaluate(&c, 1, TOPIC, &[TEMP]).await;
            assert_eq!(out.skip_status, None);
            assert_eq!(out.info["overturn_hold_probe_error"], "bad_url");
        }
        assert!(!gate.breaker_open(), "bad_url is neutral");
        assert_eq!(LOOKUP_PATH, "/v1/holds/lookup");
        assert_eq!(READY_PATH, "/readyz");
        assert_eq!(DEADLINE_HEADER, "x-request-deadline-ms");
        assert_eq!(CONNECT_TIMEOUT, Duration::from_millis(200));
        assert_eq!(MAX_BODY_BYTES, 64 * 1024);
    }

            #[test]
    fn user_agent_carries_version_and_env() {
        let ua = user_agent(Some(" prod "));
        assert_eq!(
            ua,
            format!(
                "xai-abuse-enforcement-service/{} env=prod",
                env!("CARGO_PKG_VERSION")
            )
        );
        assert!(user_agent(None).ends_with(" env=unset"));
        assert!(user_agent(Some("  ")).ends_with(" env=unset"));
    }

    mod http {
        use super::*;
        use wiremock::matchers::{body_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        use crate::ledger_contract_fixtures as contract;

                                        const FIXTURE_REQUEST: &str = contract::LOOKUP_REQUEST;
        const FIXTURE_OK: &str = contract::LOOKUP_RESPONSE_200;

                        const ERROR_FIXTURES: [(&str, u16, LedgerCode); 9] = [
            (
                contract::ERROR_CONNECT_REFUSED,
                503,
                LedgerCode::ConnectRefused,
            ),
            (contract::ERROR_AUTH_FAILED, 503, LedgerCode::AuthFailed),
            (contract::ERROR_TLS, 503, LedgerCode::Tls),
            (
                contract::ERROR_STATEMENT_TIMEOUT,
                504,
                LedgerCode::StatementTimeout,
            ),
            (
                contract::ERROR_SERVER_UNHEALTHY,
                503,
                LedgerCode::ServerUnhealthy,
            ),
            (contract::ERROR_POOL_WAIT, 503, LedgerCode::PoolWait),
            (contract::ERROR_OTHER, 500, LedgerCode::Other),
            (contract::ERROR_DECODE, 500, LedgerCode::Decode),
            (contract::ERROR_BAD_REQUEST, 400, LedgerCode::Other),
        ];

                                                                const TEST_BUDGET_MS: u64 = 5_000;

        fn json(status: u16, body: &str) -> ResponseTemplate {
            ResponseTemplate::new(status).set_body_raw(body.as_bytes().to_vec(), "application/json")
        }

        async fn store(server: &MockServer) -> HttpHoldStore {
            HttpHoldStore::new(&server.uri(), Some("staging"))
        }

                                        #[tokio::test]
        async fn lookup_decodes_the_contract_and_sends_the_headers() {
            let server = MockServer::start().await;
            let expected_body: Value = serde_json::from_str(FIXTURE_REQUEST).unwrap();
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .and(header(DEADLINE_HEADER, "5000"))
                .and(header("content-type", "application/json"))
                .and(header("user-agent", user_agent(Some("staging")).as_str()))
                .and(body_json(&expected_body))
                .respond_with(json(200, FIXTURE_OK))
                .expect(1)
                .mount(&server)
                .await;
            let store = store(&server).await;
            let holds = store
                .active_holds(
                    &[1234567890],
                    &["SpamHighRecall".to_owned()],
                    TEST_BUDGET_MS,
                )
                .await
                .expect("200 decodes");
            assert_eq!(holds.len(), 2);
            let suspend = &holds[0];
            assert_eq!(suspend.user_id, 1234567890);
            assert_eq!(suspend.hold_id, 88123);
            assert_eq!(suspend.action_kind, HOLD_KIND_SUSPEND);
            assert_eq!(suspend.label, None);
            assert_eq!(suspend.head.as_deref(), Some("bb1_reply_abuse_any_v15"));
            assert_eq!(
                suspend.expires_at,
                "2026-12-20T17:03:11Z".parse::<DateTime<Utc>>().unwrap()
            );
            assert_eq!(suspend.case_group_id, Some(64));
            assert_eq!(suspend.reason, "appeal_overturned");
            assert!(suspend.is_suspend());
            let label = &holds[1];
            assert_eq!(label.hold_id, 88124);
            assert_eq!(label.action_kind, HOLD_KIND_LABEL);
            assert_eq!(label.label.as_deref(), Some("SpamHighRecall"));
            assert_eq!(label.head, None);
            assert_eq!(label.case_group_id, None);
            assert_eq!(label.reason, "manual");
            assert!(label.withholds_label("SpamHighRecall"));
            assert!(!label.withholds_label("SpamMediumRecall"));
            let reqs = server.received_requests().await.unwrap();
            assert_eq!(reqs.len(), 1);
            let sent: Value = serde_json::from_slice(&reqs[0].body).unwrap();
            assert_eq!(sent, expected_body);
            let mut keys: Vec<&String> = sent.as_object().unwrap().keys().collect();
            keys.sort();
            assert_eq!(keys, vec!["labels", "user_ids"], "no other fields");
        }

                                                #[tokio::test]
        async fn unknown_fields_are_ignored() {
            let mut body: Value = serde_json::from_str(FIXTURE_OK).unwrap();
            let top = body.as_object_mut().unwrap();
            top.insert(
                "served_by".to_owned(),
                Value::String("xai-abuse-ledger-service-prod-0".to_owned()),
            );
            top.insert("cache".to_owned(), Value::String("none".to_owned()));
            let holds_json = top["holds"].as_array_mut().unwrap();
            assert_eq!(holds_json.len(), 2, "canonical fixture shape");
            for hold in holds_json.iter_mut() {
                hold.as_object_mut().unwrap().insert(
                    "future_field".to_owned(),
                    serde_json::json!({ "nested": true }),
                );
            }
            let body = serde_json::to_string(&body).unwrap();

            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .respond_with(json(200, &body))
                .mount(&server)
                .await;
            let holds = store(&server)
                .await
                .active_holds(&[1234567890], &[], TEST_BUDGET_MS)
                .await
                .expect("extra fields are fine");
            assert_eq!(holds.len(), 2);
            assert_eq!(holds[0].hold_id, 88123);
            assert_eq!(holds[1].hold_id, 88124);
        }

                        #[tokio::test]
        async fn empty_labels_and_empty_holds_round_trip() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .and(body_json(
                    serde_json::json!({"user_ids": [7], "labels": []}),
                ))
                .respond_with(json(200, r#"{"holds": [], "db_ms": 0}"#))
                .expect(1)
                .mount(&server)
                .await;
            let holds = store(&server)
                .await
                .active_holds(&[7], &[], TEST_BUDGET_MS)
                .await;
            assert_eq!(holds.unwrap(), Vec::<Hold>::new());
        }

                        #[tokio::test]
        async fn every_error_code_forwards_with_its_breaker_class() {
            for (body, status, code) in ERROR_FIXTURES {
                let server = MockServer::start().await;
                Mock::given(method("POST"))
                    .and(path(LOOKUP_PATH))
                    .respond_with(json(status, body))
                    .mount(&server)
                    .await;
                let err = store(&server)
                    .await
                    .active_holds(&[1], &[], TEST_BUDGET_MS)
                    .await
                    .expect_err(body);
                match &err {
                    HoldStoreError::Upstream {
                        status: got,
                        code: Some(got_code),
                    } => {
                        assert_eq!(*got, status, "{body}");
                        assert_eq!(*got_code, code, "{body}");
                    }
                    other => panic!("{body}: expected Upstream with a code, got {other:?}"),
                }
                assert_eq!(err.code(), code.probe_code(), "{body}");
                assert_eq!(err.breaker_signal(), code.breaker_signal(), "{body}");
                let raw: Value = serde_json::from_str(body).unwrap();
                let raw_code = raw["code"].as_str().unwrap();
                if code != LedgerCode::Other || raw_code == "other" {
                    assert_eq!(err.code().as_str(), raw_code, "{body}");
                }
                assert_eq!(
                    err.ledger_answered(),
                    Some(true),
                    "{body}: an HTTP answer means the hop works (gauge → 1 in the gate)"
                );
            }
        }

                                #[tokio::test]
        async fn status_without_a_code_is_ledger_status() {
            for (status, body, want, signal) in [
                (
                    502,
                    "",
                    ProbeErrorCode::LedgerStatus,
                    BreakerSignal::Failure,
                ),
                (
                    503,
                    "<html>upstream down</html>",
                    ProbeErrorCode::LedgerStatus,
                    BreakerSignal::Failure,
                ),
                (
                    500,
                    r#"{"error": "boom"}"#,
                    ProbeErrorCode::LedgerStatus,
                    BreakerSignal::Failure,
                ),
                (
                    404,
                    "",
                    ProbeErrorCode::LedgerStatus,
                    BreakerSignal::Reachable,
                ),
                (504, "", ProbeErrorCode::Timeout, BreakerSignal::Failure),
            ] {
                let server = MockServer::start().await;
                Mock::given(method("POST"))
                    .and(path(LOOKUP_PATH))
                    .respond_with(ResponseTemplate::new(status).set_body_string(body))
                    .mount(&server)
                    .await;
                let err = store(&server)
                    .await
                    .active_holds(&[1], &[], TEST_BUDGET_MS)
                    .await
                    .expect_err("non-200");
                assert!(
                    matches!(err, HoldStoreError::Upstream { status: s, code: None } if s == status),
                    "{status} {body:?}: {err:?}"
                );
                assert_eq!(err.code(), want, "{status} {body:?}");
                assert_eq!(err.breaker_signal(), signal, "{status} {body:?}");
            }
        }

                                #[tokio::test]
        async fn slow_response_is_a_timeout_at_timeout_ms() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .respond_with(json(200, FIXTURE_OK).set_delay(Duration::from_secs(5)))
                .mount(&server)
                .await;
            let started = Instant::now();
            let err = store(&server)
                .await
                .active_holds(&[1], &[], 100)
                .await
                .expect_err("slow");
            let elapsed = started.elapsed();
            assert!(
                matches!(err, HoldStoreError::Timeout(d) if d == Duration::from_millis(100)),
                "{err:?}"
            );
            assert_eq!(err.code(), ProbeErrorCode::Timeout);
            assert_eq!(err.breaker_signal(), BreakerSignal::Failure);
            assert!(
                elapsed < Duration::from_secs(3),
                "gave up at the per-request deadline, not after the server's delay: {elapsed:?}"
            );
            assert_eq!(err.ledger_answered(), Some(false), "hop failed → gauge 0");
        }

                                                                                        #[tokio::test]
        async fn unbound_port_is_ledger_unreachable() {
            let store = HttpHoldStore::new("http://127.0.0.1:1", None);
            let err = store
                .active_holds(&[1], &[], TEST_BUDGET_MS)
                .await
                .expect_err("refused");
            assert!(matches!(err, HoldStoreError::Unreachable(_)), "{err:?}");
            assert_eq!(err.code(), ProbeErrorCode::LedgerUnreachable);
            assert_eq!(err.breaker_signal(), BreakerSignal::Failure);
            assert_eq!(err.ledger_answered(), Some(false));
            let ping = store
                .ping(Duration::from_secs(1))
                .await
                .expect_err("refused");
            assert!(matches!(ping, HoldStoreError::Unreachable(_)), "{ping:?}");
        }

                                #[tokio::test]
        async fn oversize_body_is_decode() {
            let server = MockServer::start().await;
            let big = vec![b'x'; MAX_BODY_BYTES + 1];
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(big.clone()))
                .mount(&server)
                .await;
            let err = store(&server)
                .await
                .active_holds(&[1], &[], TEST_BUDGET_MS)
                .await
                .expect_err("oversize");
            assert!(matches!(err, HoldStoreError::Decode(_)), "{err:?}");
            assert!(err.to_string().contains("exceeds"), "{err}");
            assert_eq!(err.code(), ProbeErrorCode::Decode);
            assert_eq!(err.breaker_signal(), BreakerSignal::Neutral);

            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .respond_with(ResponseTemplate::new(503).set_body_bytes(big))
                .mount(&server)
                .await;
            let err = store(&server)
                .await
                .active_holds(&[1], &[], TEST_BUDGET_MS)
                .await
                .expect_err("oversize 503");
            assert!(
                matches!(
                    err,
                    HoldStoreError::Upstream {
                        status: 503,
                        code: None
                    }
                ),
                "{err:?}"
            );
            assert_eq!(err.code(), ProbeErrorCode::LedgerStatus);
        }

                                #[tokio::test]
        async fn malformed_200_body_is_decode() {
            for body in [
                "",
                "not json",
                r#"{"rows": []}"#,
                r#"{"holds": [{"user_id": 1}]}"#,
                r#"{"holds": [{"user_id": "1", "hold_id": 2, "expires_at": "2026-12-20T17:03:11Z", "reason": "manual", "action_kind": "suspend"}]}"#,
                r#"{"holds": [{"user_id": 1, "hold_id": 2, "expires_at": "yesterday", "reason": "manual", "action_kind": "suspend"}]}"#,
            ] {
                let server = MockServer::start().await;
                Mock::given(method("POST"))
                    .and(path(LOOKUP_PATH))
                    .respond_with(json(200, body))
                    .mount(&server)
                    .await;
                let err = store(&server)
                    .await
                    .active_holds(&[1], &[], TEST_BUDGET_MS)
                    .await
                    .expect_err(body);
                assert!(matches!(err, HoldStoreError::Decode(_)), "{body}: {err:?}");
                assert_eq!(err.code(), ProbeErrorCode::Decode);
                assert_eq!(err.breaker_signal(), BreakerSignal::Neutral);
            }
        }

                        #[tokio::test]
        async fn startup_probe_is_a_get_readyz() {
            let _serial = METRICS_LOCK.lock().await;
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path(READY_PATH))
                .and(header("user-agent", user_agent(Some("prod")).as_str()))
                .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
                .expect(1)
                .mount(&server)
                .await;
            let gate = HoldGate::from_url(Some(&server.uri()), true, Some("prod"));
            assert!(gate.startup_probe().await);
            assert_eq!(metrics::HOLD_GATE_DB_CONNECTED.get(), 1);

            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path(READY_PATH))
                .respond_with(json(
                    503,
                    r#"{"code": "server_unhealthy", "retryable": true}"#,
                ))
                .mount(&server)
                .await;
            let gate = HoldGate::from_url(Some(&server.uri()), true, None);
            assert!(!gate.startup_probe().await);
            assert_eq!(metrics::HOLD_GATE_DB_CONNECTED.get(), 0);
            let store = HttpHoldStore::new(&server.uri(), None);
            let err = store.ping(Duration::from_secs(1)).await.expect_err("503");
            assert_eq!(err.code(), ProbeErrorCode::ServerUnhealthy, "{err}");
            assert!(
                server
                    .received_requests()
                    .await
                    .unwrap()
                    .iter()
                    .all(|r| r.method == "GET"),
                "startup probe only GETs"
            );
        }

                                        #[tokio::test]
        async fn gate_over_http_store_blocks_a_held_suspend() {
            let _serial = METRICS_LOCK.lock().await;
            let server = MockServer::start().await;
            let expires = Utc::now() + chrono::Duration::days(30);
            let body = serde_json::json!({
                "holds": [{
                    "user_id": 42, "hold_id": 7, "action_kind": "suspend", "label": null,
                    "head": "FollowBot", "expires_at": expires.to_rfc3339(),
                    "case_group_id": 63, "reason": "appeal_overturned"
                }],
                "db_ms": 1
            });
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .and(header(DEADLINE_HEADER, "5000"))
                .respond_with(ResponseTemplate::new(200).set_body_json(body))
                .expect(1)
                .mount(&server)
                .await;
            let gate = HoldGate::from_url(Some(&server.uri()), false, Some("prod"));
            let c = HoldGateConfig {
                timeout_ms: TEST_BUDGET_MS,
                ..cfg(GateMode::Enforce)
            };
            let out = gate.evaluate(&c, 42, TOPIC, &[TEMP]).await;
            assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
            assert_eq!(out.info["overturn_hold_id"], "7");
            assert_eq!(out.info["overturn_hold_head"], "FollowBot");
            assert_eq!(out.info["overturn_hold_case_group_id"], "63");
            assert_eq!(out.info["overturn_hold_kind"], "suspend");
            assert_eq!(out.info["overturn_hold_env"], "prod");
            assert_eq!(out.info["overturn_hold_mode"], "enforce");
            assert!(!out.info.contains_key("overturn_hold_probe_error"));
            let out = gate.evaluate(&c, 42, TOPIC, &[TEMP]).await;
            assert_eq!(out.skip_status, Some(STATUS_HOLD_OVERTURNED));
            assert_eq!(out.info["overturn_hold_probe_ms"], "0", "cache hit");
            assert!(!gate.breaker_open());
        }

                                                                                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn probe_deadline_is_measured_on_its_own_task_not_the_stalled_caller() {
            let _serial = METRICS_LOCK.lock().await;
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path(LOOKUP_PATH))
                .respond_with(json(200, r#"{"holds":[],"db_ms":0}"#))
                .expect(1)
                .mount(&server)
                .await;
            let gate = HoldGate::from_url(Some(&server.uri()), false, Some("staging"));
            let c = HoldGateConfig {
                timeout_ms: TEST_BUDGET_MS,
                ..cfg(GateMode::Enforce)
            };
            const STALL: Duration = Duration::from_millis(500);
            let delay_count = metrics::HOLD_GATE_PROBE_SCHED_DELAY_SECONDS.get_sample_count();
            let delay_sum = metrics::HOLD_GATE_PROBE_SCHED_DELAY_SECONDS.get_sample_sum();

            let started = Instant::now();
            let mut probe = std::pin::pin!(gate.check_timed(&c, 42, TOPIC, &[]));
            assert!(futures::poll!(probe.as_mut()).is_pending());
            std::thread::sleep(STALL);
            let (lookup, probe_ms, cache_hit) = probe.await;
            let caller_ms = started.elapsed().as_millis() as u64;

            match &lookup {
                Lookup::Holds(h) => assert!(h.is_empty()),
                Lookup::ProbeError(e) => {
                    panic!("the answer, not a probe error charged to the caller's stall: {e}")
                }
                Lookup::Disabled => panic!("gate is enforce"),
            }
            assert!(!cache_hit);
            assert!(
                probe_ms + STALL.as_millis() as u64 / 2 <= caller_ms,
                "probe_ms is the wire time ({probe_ms} ms), not the caller's \
                 {caller_ms} ms (stall {STALL:?})"
            );
            assert_eq!(
                metrics::HOLD_GATE_PROBE_SCHED_DELAY_SECONDS.get_sample_count(),
                delay_count + 1
            );
            assert!(
                metrics::HOLD_GATE_PROBE_SCHED_DELAY_SECONDS.get_sample_sum() - delay_sum
                    >= STALL.as_secs_f64() / 2.0,
                "the stall is reported as scheduling delay"
            );
            assert!(!gate.breaker_open(), "no spurious failure counted");
        }
    }
}
