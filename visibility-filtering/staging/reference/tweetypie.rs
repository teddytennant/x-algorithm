use super::{
    ENV_IMAGE, ENV_TWEETYPIE_CLIENT_ID, ENV_TWEETYPIE_TLS_DOMAIN, ENV_TWEETYPIE_XDS_LISTENER,
};
use crate::config::ENV_REFERENCE;
use crate::filter::{FilterOutcome, FilterRequest, FilterTweets};
use crate::hydration::{HYDRATION_TIMEOUT, request_context};
use crate::models::{Evaluation, RawCandidate, TweetId, Verdict};
use crate::params::{ClientSwitches, LimitedActionsPolicies};
use crate::retweet;
use crate::rules::SafetyLevel;
use crate::rules::metrics::Rpc;
use crate::server_deps::init_client_with_retry;
use crate::staging::reference_compare::resolve_build_sha;
use crate::treatment;
use anyhow::Context;
use futures::future::join;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fmt::Debug;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::spawn;
use tokio::sync::Semaphore;
use tokio::time::{Instant, timeout};
use tonic::metadata::MetadataMap;
use tracing::warn;
use xai_build_version::current_build_information;
use xai_stats_receiver::global_stats_receiver;
use xai_strato::strato_thrift::{StratoResult, strato_decode};
use xai_strato::{StratoGrpc, encode};
use xai_twittercontext_proto::TwitterContextViewer;
use xai_x_rpc::balanced_channel::LbPolicy;
use xai_x_rpc::grpc_client::{ChannelBuilder, TlsMode};
use xai_x_rpc::retry::RetryConfig;
use xai_x_rpc::timed_buffer::DEFAULT_BUFFER_MAX_WAIT;
use xai_x_rpc::total_timeout::DEFAULT_TOTAL_TIMEOUT;
use xai_x_thrift::action::Action;
use xai_x_thrift::get_tweet_fields::{
    GetTweetFieldsOptions, GetTweetFieldsResult, TweetFieldsResultState,
    VISIBILITY_POLICY_USER_VISIBLE,
};
use xai_x_thrift::safety_level::SafetyLevel as ThriftSafetyLevel;
use xai_x_thrift::safety_result::FilteredReason;
use xai_xds_client::StartFrom;

const COLUMN: &str = "tweetypie/getTweetFields.Tweet";
const TWEETYPIE_MAX_BATCH_SIZE: usize = 250;
const MAX_IN_FLIGHT: usize = 32;
const TWEETYPIE_TIMEOUT: Duration = Duration::from_millis(1500);
const COMPARE: &str = "vf_tweetypie_reference_compare";
const BATCHES: &str = "vf_tweetypie_reference_batches";
const RECORD_LINE_MARKER: &str = "vf_tweetypie_reference_record";
const SAFETY_LEVEL: SafetyLevel = SafetyLevel::TimelineHomeHydration;
const RECORD_CAP_PER_MINUTE: u32 = 20;
const NONE: &str = "none";

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum Class {
    Allow,
    BareDrop,
    Drop,
    Suppressed,
    NotFound,
    Failed,
    Missing,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum Failure {
    NoValue,
    StratoError,
    RateLimited,
    DecodeError,
    RpcError,
    OverCapacity,
    TweetypieFailed,
}

#[derive(Clone, Copy, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum BatchOutcome {
    Compared,
    TweetypieTimeout,
    Busy,
}

pub(crate) type Label = (String, String);

fn label(class: Class, reason: impl Into<String>) -> Label {
    (<&str>::from(class).to_string(), reason.into())
}

fn failed(failure: Failure) -> Label {
    label(Class::Failed, <&str>::from(failure))
}

fn ident(s: &str) -> (&str, &str) {
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(s.len());
    s.split_at(end)
}

fn snake(ident: &str) -> String {
    let mut out = String::with_capacity(ident.len() + 4);
    for (i, c) in ident.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

fn debug_label(value: &impl Debug) -> Label {
    let debug = format!("{value:?}");
    let class = snake(ident(&debug).0);
    let Some((_, mut rest)) = debug.split_once("reason: Some(") else {
        return (class, NONE.to_string());
    };
    let reason = loop {
        let (name, after) = ident(rest);
        if let Some(inner) = after.strip_prefix(" {")
            && let Some((_, nested)) = inner.split_once("Some(")
        {
            rest = nested;
            continue;
        }
        let number = after
            .strip_prefix('(')
            .and_then(|r| r.split(')').next())
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit() || c == '-'));
        break match (name, number) {
            ("", _) => "unknown".to_string(),
            (name, Some(n)) => format!("{}_{n}", snake(name)),
            (name, None) => snake(name),
        };
    };
    (class, reason)
}

pub(crate) struct TpResult {
    pub(crate) label: Label,
    result: Option<String>,
    strato_error: Option<String>,
}

fn tp_failed(failure: Failure) -> TpResult {
    TpResult {
        label: failed(failure),
        result: None,
        strato_error: None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Bucket {
    tp: Label,
    vf: Label,
    vf_rule: &'static str,
}

impl Bucket {
    fn counter_labels<'a>(&'a self, build_sha: &'a str) -> [(&'static str, &'a str); 7] {
        [
            ("safety_level", SAFETY_LEVEL.into()),
            ("tp", &self.tp.0),
            ("tp_reason", &self.tp.1),
            ("vf", &self.vf.0),
            ("vf_reason", &self.vf.1),
            ("vf_rule", self.vf_rule),
            ("build_sha", build_sha),
        ]
    }
}

pub(super) async fn build(
    init_deadline: Instant,
    filter_tweets: &Arc<FilterTweets>,
    client_switches: ClientSwitches,
    metadata: Option<&MetadataMap>,
) -> Arc<TweetypieReference> {
    let tweetypie = connect(init_deadline, metadata).await;
    warn!("reference comparator: tweetypie enabled");
    Arc::new(TweetypieReference {
        tweetypie,
        filter_tweets: Arc::clone(filter_tweets),
        client_switches,
        permits: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
        sampler: Mutex::default(),
        build_sha: resolve_build_sha(
            &current_build_information().git_commit_sha,
            env::var(ENV_IMAGE).ok().as_deref(),
        ),
        pod: env::var("HOSTNAME").ok(),
    })
}

#[expect(
    clippy::expect_used,
    reason = "startup fail-fast: init failure is fatal"
)]
pub(crate) async fn connect(init_deadline: Instant, metadata: Option<&MetadataMap>) -> StratoGrpc {
    let [xds_listener, tls_domain, client_id] = &[
        ENV_TWEETYPIE_XDS_LISTENER,
        ENV_TWEETYPIE_TLS_DOMAIN,
        ENV_TWEETYPIE_CLIENT_ID,
    ]
    .map(required_env);
    init_client_with_retry("tweetypie_reference", init_deadline, || async move {
        let channel = ChannelBuilder::new("tweetypie-xds")
            .tls(
                TlsMode::mtls_from_env()
                    .context("S2S cert env vars required for mTLS")?
                    .with_domain_override(tls_domain.clone()),
            )
            .request_timeout(HYDRATION_TIMEOUT)
            .connect_timeout(Duration::from_millis(400))
            .xds(StartFrom::Lds(xds_listener.clone()))
            .await
            .with_context(|| format!("failed to initialize xDS for {xds_listener}"))?
            .eager_resolution(Duration::from_secs(5))
            .aperture(12)
            .buffer_max_wait(DEFAULT_BUFFER_MAX_WAIT)
            .total_timeout(DEFAULT_TOTAL_TIMEOUT)
            .build_load_balanced(LbPolicy::least_request())
            .await
            .context("failed to build xDS LoadBalancedChannel for tweetypie-xds")?;
        anyhow::Ok(StratoGrpc::from_load_balanced_channel(
            channel,
            metadata.cloned(),
            Some(client_id.clone()),
            Some(RetryConfig::for_idempotent()),
            TWEETYPIE_MAX_BATCH_SIZE,
        ))
    })
    .await
    .expect("Failed to initialize Tweetypie client (reference comparator)")
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
fn required_env(name: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| panic!("{name} must be set when {ENV_REFERENCE}=tweetypie"))
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Client {
    #[default]
    Web,
    IosCurrent,
    IosOutdated,
    AndroidCurrent,
    AndroidOutdated,
    AndroidWithoutFosnr,
    MacApp,
}

impl Client {
            const fn app(self) -> (i64, Option<&'static str>) {
        match self {
            Self::Web => (3_033_300, None),
            Self::IosCurrent => (
                129_032,
                Some("Twitter-iPhone/11.11.5 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)"),
            ),
            Self::IosOutdated => (
                129_032,
                Some("Twitter-iPhone/11.11.4 iOS/17.0 (Apple;iPhone15,2;;;;;1;2022)"),
            ),
            Self::AndroidCurrent => (
                258_901,
                Some(
                    "TwitterAndroid/11.11.0-release.00 (311110000-r-0) Pixel 7/14 \
                     (Google;panther;google;panther;0;;1;2022)",
                ),
            ),
            Self::AndroidOutdated => (
                258_901,
                Some(
                    "TwitterAndroid/11.10.9-release.00 (311109000-r-0) Pixel 7/14 \
                     (Google;panther;google;panther;0;;1;2022)",
                ),
            ),
            Self::AndroidWithoutFosnr => (
                258_901,
                Some(
                    "TwitterAndroid/9.82.0-release.00 (29820000-r-0) Pixel 7/14 \
                     (Google;panther;google;panther;0;;1;2022)",
                ),
            ),
            Self::MacApp => (
                557_701,
                Some("Twitter-Mac/11.11.5 macOS/14.0 (Apple;Mac14,2)"),
            ),
        }
    }

    pub(crate) fn context(
        self,
        viewer_id: Option<u64>,
        country_code: Option<&str>,
    ) -> TwitterContextViewer {
        let (client_application_id, user_agent) = self.app();
        TwitterContextViewer {
            user_id: viewer_id.map_or(0, u64::cast_signed),
            client_application_id,
            user_agent: user_agent.unwrap_or_default().to_string(),
            request_country_code: country_code.unwrap_or_default().to_string(),
            ..Default::default()
        }
    }
}

pub(super) struct TweetypieReference {
    tweetypie: StratoGrpc,
    filter_tweets: Arc<FilterTweets>,
    client_switches: ClientSwitches,
    permits: Arc<Semaphore>,
    sampler: Mutex<(u64, HashMap<Bucket, u32>)>,
    build_sha: String,
    pod: Option<String>,
}

impl TweetypieReference {
    pub(super) fn spawn(
        self: &Arc<Self>,
        viewer_id: Option<u64>,
        country_code: Option<String>,
        safety_level: SafetyLevel,
        outcomes: &[FilterOutcome],
    ) {
        let Some(viewer_id) = viewer_id else {
            return;
        };
        match safety_level {
            SafetyLevel::TimelineHome | SafetyLevel::TimelineHomeRecommendations => {}
            SafetyLevel::FilterAll
            | SafetyLevel::TimelineHomeHydration
            | SafetyLevel::ImmersiveExpandedRecommendations => return,
        }
        let tweet_ids: Vec<TweetId> = outcomes
            .iter()
            .filter(|outcome| match &outcome.evaluation {
                Evaluation::Complete { verdict } => !matches!(verdict, Verdict::Withheld(_)),
                Evaluation::NotFound(_) => true,
                Evaluation::Partial { .. } | Evaluation::Failed(_) => false,
            })
            .map(|outcome| outcome.tweet_id)
            .collect();
        if tweet_ids.is_empty() {
            return;
        }
        let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() else {
            self.count(BatchOutcome::Busy);
            return;
        };
        let reference = Arc::clone(self);
        spawn(async move {
            let _permit = permit;
            let outcome = reference.compare(viewer_id, country_code, tweet_ids).await;
            reference.count(outcome);
        });
    }

    async fn compare(
        &self,
        viewer_id: u64,
        country_code: Option<String>,
        tweet_ids: Vec<TweetId>,
    ) -> BatchOutcome {
        let candidates = tweet_ids
            .iter()
            .map(|&tweet_id| RawCandidate {
                tweet_id,
                request_author_id: None,
            })
            .collect();
        let client = Client::Web.context(Some(viewer_id), country_code.as_deref());
        let client_capability =
            self.client_switches
                .resolve(Some(&client), Some(viewer_id), country_code.as_deref());
        let rust = request_context(Instant::now(), None).scope(retweet::evaluate_merging_sources(
            &self.filter_tweets,
            FilterRequest {
                viewer_id: Some(viewer_id),
                country_code: country_code.clone(),
                client_capability,
                safety_level: SAFETY_LEVEL,
                candidates,
                rpc: Rpc::EvaluateTweets,
            },
        ));
        let tp = timeout(
            TWEETYPIE_TIMEOUT,
            get_tweet_fields(&self.tweetypie, &client, &tweet_ids),
        );
        let (rust, tp) = join(rust, tp).await;
        let Ok(tp) = tp else {
            return BatchOutcome::TweetypieTimeout;
        };
        let vf: HashMap<TweetId, &FilterOutcome> = rust
            .iter()
            .map(|outcome| (outcome.tweet_id, outcome))
            .collect();
        let minute = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs() / 60);
        let mut counts: HashMap<Bucket, u64> = HashMap::new();
        let mut sampler = self.sampler.lock().unwrap_or_else(PoisonError::into_inner);
        if sampler.0 != minute {
            *sampler = (minute, HashMap::new());
        }
        for (&tweet_id, tp) in tweet_ids.iter().zip(&tp) {
            let compared = Compared {
                tweet_id,
                viewer_id,
                country_code: country_code.as_deref(),
                tp,
                vf: vf.get(&tweet_id).copied(),
            };
            let bucket = compared.bucket();
            let seen = sampler.1.entry(bucket.clone()).or_default();
            *seen += 1;
            if *seen <= RECORD_CAP_PER_MINUTE {
                #[expect(clippy::print_stdout, reason = "stdout is the record sink")]
                {
                    println!(
                        "{}",
                        compared.to_json(&bucket, &self.build_sha, self.pod.as_deref())
                    );
                }
            }
            *counts.entry(bucket).or_default() += 1;
        }
        drop(sampler);
        if let Some(stats) = global_stats_receiver() {
            for (bucket, count) in &counts {
                stats.incr(COMPARE, &bucket.counter_labels(&self.build_sha), *count);
            }
        }
        BatchOutcome::Compared
    }

    fn count(&self, outcome: BatchOutcome) {
        if let Some(stats) = global_stats_receiver() {
            stats.incr(
                BATCHES,
                &[
                    ("safety_level", SAFETY_LEVEL.into()),
                    ("outcome", outcome.into()),
                    ("build_sha", &self.build_sha),
                ],
                1,
            );
        }
    }
}

pub(crate) async fn get_tweet_fields(
    tweetypie: &StratoGrpc,
    client: &TwitterContextViewer,
    tweet_ids: &[TweetId],
) -> Vec<TpResult> {
    let view = GetTweetFieldsOptions {
        for_user_id: Some(client.user_id).filter(|&user_id| user_id != 0),
        language_tag: Some("en"),
        safety_level: ThriftSafetyLevel::TIMELINE_HOME_HYDRATION,
        visibility_policy: VISIBILITY_POLICY_USER_VISIBLE,
        ..Default::default()
    };
    let calls = tweet_ids
        .iter()
        .map(|id| {
            (
                COLUMN.to_string(),
                "fetch".to_string(),
                vec![encode(&(id.0.cast_signed(), view))],
            )
        })
        .collect();
    tweetypie
        .batch_call(calls, Some(client))
        .await
        .into_iter()
        .map(|result| match result {
            Ok(bytes) => match strato_decode::<GetTweetFieldsResult>(&bytes) {
                Ok(result) => tp_result(result),
                Err(_) => tp_failed(Failure::DecodeError),
            },
            Err(_) => tp_failed(Failure::RpcError),
        })
        .collect()
}

struct Compared<'a> {
    tweet_id: TweetId,
    viewer_id: u64,
    country_code: Option<&'a str>,
    tp: &'a TpResult,
    vf: Option<&'a FilterOutcome>,
}

impl Compared<'_> {
    fn bucket(&self) -> Bucket {
        let vf = self
            .vf
            .map_or_else(|| label(Class::Missing, NONE), vf_label);
        let allow: &str = Class::Allow.into();
        let (tp, vf) = if self.tp.label.0 == allow && vf.0 == allow {
            (label(Class::Allow, NONE), label(Class::Allow, NONE))
        } else {
            (self.tp.label.clone(), vf)
        };
        let vf_rule = self
            .vf
            .and_then(|outcome| treatment::decided_rows(outcome.evaluation.verdict()).next())
            .map_or(NONE, |(rule, _)| rule);
        Bucket { tp, vf, vf_rule }
    }

    fn to_json(&self, bucket: &Bucket, build_sha: &str, pod: Option<&str>) -> Value {
        let vf_rules: Vec<String> = self
            .vf
            .into_iter()
            .flat_map(|outcome| treatment::decided_rows(outcome.evaluation.verdict()))
            .map(|(rule, kind)| format!("{rule}:{kind}"))
            .collect();
        let labels: BTreeMap<i32, Option<i64>> = self
            .vf
            .and_then(|outcome| outcome.safety_labels.as_ref())
            .into_iter()
            .flat_map(|map| &map.labels)
            .map(|(&label_type, label)| (label_type, label.created_at_msec))
            .collect();
        json!({
            "h": RECORD_LINE_MARKER,
            "v": 1,
            "level": <&str>::from(SAFETY_LEVEL),
            "build": build_sha,
            "pod": pod,
            "tweet": self.tweet_id.0,
            "viewer": self.viewer_id,
            "country": self.country_code,
            "tp": [&bucket.tp.0, &bucket.tp.1],
            "tp_result": self.tp.result,
            "tp_strato_error": self.tp.strato_error,
            "vf": [&bucket.vf.0, &bucket.vf.1],
            "vf_status": self.vf.map(|outcome| vf_status(&outcome.evaluation)),
            "vf_rules": vf_rules,
            "labels": labels,
        })
    }
}

fn tp_result(result: StratoResult<GetTweetFieldsResult>) -> TpResult {
    match result {
        StratoResult::Ok { value: None, .. } => tp_failed(Failure::NoValue),
        StratoResult::Ok {
            value: Some(value), ..
        } => TpResult {
            label: tp_label(&value.tweet_result),
            result: Some(format!("{:?}", value.tweet_result)),
            strato_error: None,
        },
        StratoResult::Err { code, message } => TpResult {
            strato_error: Some(format!("{code}: {message}")),
            ..tp_failed(if message.contains("RateLimited") {
                Failure::RateLimited
            } else {
                Failure::StratoError
            })
        },
    }
}

fn vf_status(evaluation: &Evaluation) -> &'static str {
    match evaluation {
        Evaluation::Complete { .. } => "Evaluated",
        Evaluation::Partial { .. } => "Failed",
        Evaluation::NotFound(_) => "NotFound",
        Evaluation::Failed(_) => "LookupFailed",
    }
}

pub(crate) fn vf_label(outcome: &FilterOutcome) -> Label {
    match &outcome.evaluation {
        Evaluation::Complete { verdict } => tp_label(&treatment::thrift_result_state(
            verdict,
            SAFETY_LEVEL,
            &LimitedActionsPolicies::default(),
        )),
        Evaluation::NotFound(_) => label(Class::NotFound, NONE),
        Evaluation::Partial { .. } | Evaluation::Failed(_) => label(Class::Failed, NONE),
    }
}

fn tp_label(state: &TweetFieldsResultState) -> Label {
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "FilteredReason is a generated Thrift union; every variant but SafetyResult labels by its Debug name, including any the IDL adds"
    )]
    let reason = |reason: &FilteredReason| match reason {
        FilteredReason::SafetyResult(result) => debug_label(result).1,
        other => debug_label(other).0,
    };
    match state {
        TweetFieldsResultState::Found(found) => match &found.suppress_reason {
            None => label(Class::Allow, NONE),
            Some(suppress @ FilteredReason::SafetyResult(result)) => {
                (debug_label(&result.action).0, reason(suppress))
            }
            Some(suppress) => label(Class::Suppressed, reason(suppress)),
        },
        TweetFieldsResultState::Filtered(filtered) => {
            #[expect(
                clippy::wildcard_enum_match_arm,
                reason = "FilteredReason is a generated Thrift union; every variant but SafetyResult is a plain drop, including any the IDL adds"
            )]
            let drop_label = match &filtered.reason {
                FilteredReason::SafetyResult(result)
                    if result.reason.is_none()
                        && matches!(&result.action, Action::Drop(drop) if drop.reason.is_none()) =>
                {
                    label(Class::BareDrop, NONE)
                }
                filtered @ FilteredReason::SafetyResult(result) => {
                    (debug_label(&result.action).0, reason(filtered))
                }
                filtered => label(Class::Drop, reason(filtered)),
            };
            drop_label
        }
        TweetFieldsResultState::NotFound(not_found) => label(
            Class::NotFound,
            not_found
                .filtered_reason
                .as_ref()
                .map_or_else(|| NONE.to_string(), reason),
        ),
        TweetFieldsResultState::Failed(result) if result.over_capacity => {
            failed(Failure::OverCapacity)
        }
        TweetFieldsResultState::Failed(_) => failed(Failure::TweetypieFailed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::{Hydrators, Lookup};
    use crate::models::{Decided, DropReason, Withholding};
    use crate::rules::fixtures::CLIENT_CLASSES;
    use std::fs;
    use std::path::Path;
    use strum::VariantArray;
    use xai_visibility_filtering::models::FilteredReason as VfFilteredReason;
    use xai_x_thrift::action::{
        AnyInterstitial, BlockedViewer, Drop as ThriftDrop, Interstitial, InterstitialReason,
        LimitedEngagementReason, LimitedEngagements, LocalizedMessage, TweetInterstitial,
    };
    use xai_x_thrift::get_tweet_fields::TweetFieldsResultFound;
    use xai_x_thrift::safety_result::SafetyResult;
    use xai_x_thrift::tweet_service::{
        TweetFieldsResultFailed, TweetFieldsResultFiltered, TweetFieldsResultNotFound,
    };

    const GOLDEN_RECORD: &str = "scripts/tests/fixtures/tweetypie_reference_record.json";

    fn outcome(verdict: Verdict) -> FilterOutcome {
        FilterOutcome {
            tweet_id: TweetId(1),
            source_tweet_id: None,
            evaluation: Evaluation::Complete { verdict },
            safety_labels: None,
        }
    }

    fn shown() -> Verdict {
        Verdict::Shown {
            notice: None,
            media: None,
            engagement: None,
        }
    }

    fn pair(class: &str, reason: &str) -> Label {
        (class.to_string(), reason.to_string())
    }

    fn compared<'a>(tp: &'a TpResult, vf: Option<&'a FilterOutcome>) -> Compared<'a> {
        Compared {
            tweet_id: TweetId(1),
            viewer_id: 2,
            country_code: Some("us"),
            tp,
            vf,
        }
    }

    #[test]
    fn the_record_names_each_evaluation() {
        assert_eq!(
            [
                Evaluation::Complete { verdict: shown() },
                Evaluation::Partial {
                    verdict: shown(),
                    fail_open_defaults: Hydrators::empty(),
                },
                Evaluation::NotFound(Lookup::Tweet),
                Evaluation::Failed(Lookup::Author),
            ]
            .iter()
            .map(vf_status)
            .collect::<Vec<_>>(),
            ["Evaluated", "Failed", "NotFound", "LookupFailed"]
        );
    }

    #[test]
    fn tweetypie_labels_and_buckets() {
        let safety_result = |action| FilteredReason::SafetyResult(SafetyResult::new(None, action));
        let blocked_viewer = safety_result(Action::LimitedEngagements(LimitedEngagements::new(
            LimitedEngagementReason::BlockedViewer(BlockedViewer::default()),
            None,
            None,
        )));
        let found = |reason| TweetFieldsResultState::Found(TweetFieldsResultFound::new(reason));
        let filtered =
            |reason| TweetFieldsResultState::Filtered(TweetFieldsResultFiltered::new(reason));
        let cases = [
            (found(None), ("allow", "none")),
            (
                found(Some(blocked_viewer)),
                ("limited_engagements", "blocked_viewer"),
            ),
            (
                found(Some(FilteredReason::ReportedTweet(true))),
                ("suppressed", "reported_tweet"),
            ),
            (
                filtered(FilteredReason::AuthorIsSuspended(true)),
                ("drop", "author_is_suspended"),
            ),
            (
                filtered(safety_result(Action::Drop(ThriftDrop::new(None, None)))),
                ("bare_drop", "none"),
            ),
            (
                TweetFieldsResultState::NotFound(TweetFieldsResultNotFound::new(
                    true,
                    None,
                    None::<FilteredReason>,
                )),
                ("not_found", "none"),
            ),
            (
                TweetFieldsResultState::Failed(TweetFieldsResultFailed::new(true, None::<String>)),
                ("failed", "over_capacity"),
            ),
            (
                TweetFieldsResultState::Failed(TweetFieldsResultFailed::new(false, None::<String>)),
                ("failed", "tweetypie_failed"),
            ),
        ];
        for (state, (class, reason)) in cases {
            assert_eq!(tp_label(&state), pair(class, reason), "{state:?}");
        }
        let rate_limited = tp_result(StratoResult::Err {
            code: 305,
            message: "ClientError(RateLimited: exceeded the rate limit)".to_string(),
        });
        assert_eq!(rate_limited.label, pair("failed", "rate_limited"));

        let allow = TpResult {
            label: pair("allow", "safety_result_reason_14"),
            result: None,
            strato_error: None,
        };
        let shown = outcome(shown());
        assert_eq!(
            compared(&allow, Some(&shown)).bucket(),
            Bucket {
                tp: pair("allow", NONE),
                vf: pair("allow", NONE),
                vf_rule: NONE,
            }
        );
        let suspended = outcome(Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(VfFilteredReason::AuthorIsSuspended)),
            by: "suspended_author/drop",
        }));
        let bucket = compared(&allow, Some(&suspended)).bucket();
        assert_eq!(
            (bucket.vf, bucket.vf_rule),
            (pair("drop", "author_is_suspended"), "suspended_author/drop")
        );
        let not_found = TpResult {
            label: pair("not_found", NONE),
            result: None,
            strato_error: None,
        };
        let deleted = FilterOutcome {
            evaluation: Evaluation::NotFound(Lookup::Tweet),
            ..shown
        };
        let bucket = compared(&not_found, Some(&deleted)).bucket();
        assert_eq!(
            (bucket.tp, bucket.vf, bucket.vf_rule),
            (
                pair("not_found", NONE),
                pair("not_found", NONE),
                "not_found"
            )
        );
    }

    #[test]
    fn each_client_resolves_its_fixture_class_capability() {
        let switches = ClientSwitches::for_tests();
        for &client in Client::VARIANTS {
            let name = serde_json::to_value(client).unwrap();
            let class = CLIENT_CLASSES
                .iter()
                .find(|class| name == class.name)
                .unwrap_or_else(|| panic!("no client class named {name}"));
            assert_eq!(
                switches.resolve(
                    Some(&client.context(Some(1), Some("fr"))),
                    Some(1),
                    Some("fr")
                ),
                class.capability,
                "{name}"
            );
        }
    }

    #[test]
    fn a_reason_after_multi_byte_text_keeps_its_label() {
        let message =
            LocalizedMessage::new("Contenu réservé ✓".to_string(), "fr".to_string(), None);
        let action = Action::TweetInterstitial(TweetInterstitial {
            interstitial: Some(AnyInterstitial::Interstitial(Interstitial::new(
                None::<InterstitialReason>,
                message,
            ))),
            limited_engagements: Some(LimitedEngagements::new(
                LimitedEngagementReason::BlockedViewer(BlockedViewer::default()),
                None,
                None,
            )),
            ..TweetInterstitial::default()
        });
        let state = TweetFieldsResultState::Found(TweetFieldsResultFound::new(Some(
            FilteredReason::SafetyResult(SafetyResult::new(None, action)),
        )));
        assert_eq!(
            tp_label(&state),
            pair("tweet_interstitial", "blocked_viewer")
        );
    }

    #[test]
    fn record_and_counter_labels_match_the_golden_fixture() {
        let tp = TpResult {
            label: pair("drop", "author_is_suspended"),
            result: None,
            strato_error: None,
        };
        let vf = outcome(shown());
        let compared = compared(&tp, Some(&vf));
        let bucket = compared.bucket();
        let emitted = json!({
            "compare_metric": COMPARE,
            "compare_labels": bucket.counter_labels("abc123def456").map(|(key, _)| key),
            "record": compared.to_json(&bucket, "abc123def456", Some("xai-vf-service-cand--abcde-0")),
        });
        let cargo = format!("{}/{GOLDEN_RECORD}", env!("CARGO_MANIFEST_DIR"));
        let path = if Path::new(&cargo).exists() {
            cargo
        } else {
            format!("crates/x-product/xai-visibility-filtering-service/{GOLDEN_RECORD}")
        };
        let on_disk: Value = serde_json::from_str(
            &fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}")),
        )
        .unwrap();
        assert_eq!(emitted, on_disk);
    }
}
