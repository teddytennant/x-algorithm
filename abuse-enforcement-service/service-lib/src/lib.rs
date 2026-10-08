pub mod ais_client;
pub mod allowlist;
pub mod api_doc;
pub mod auth;
pub mod config;
pub mod decision;
pub mod dedup_cache;
pub mod entities;
pub mod facts;
pub mod generic_actions;
pub mod gizmoduck;
pub mod gizmoduck_labels;
pub mod growthbook;
pub mod growthbook_writer;
#[cfg(test)]
mod ledger_contract_fixtures;
pub mod limiter;
pub mod manhattan;
pub mod metrics;
pub mod overturn_hold;
pub mod restart_on_config_change;
pub mod rules;
pub mod service;
pub mod sliding_window;
pub mod strato;
pub mod test_user;

use xai_abuse_proto::enforcement as abuse_proto;

use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;

use tokio::sync::Mutex;

use crate::dedup_cache::{DedupCheck, ManhattanDedupCache};
use anyhow::{Context, Result};
use axum::Router;

use clap::Parser;
use futures::future::join_all;
use prost::Message;
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio::task::{JoinHandle, JoinSet};
use tracing::{Instrument, error, info, warn};
use xai_kafka::{
    BatchConsumerConfig, BatchResult, CancellationToken, KafkaBatchProcessor, KafkaConsumerConfig,
    KafkaConsumerConfigBuilder, KafkaMessage, KafkaProducer, KafkaProducerConfigBuilder,
    apply_auth_config, resolve_kafka_brokers, run_batch_consumer, self_delete_pod,
};
use xai_service_runner::{ServerBuilder, ServerInfo};

#[allow(deprecated)]
use xai_strato::{Strato, StratoClientConfig};

use crate::config::Config;
use crate::decision::Decision;
use crate::facts::{EntityFacts, EntityType, Facts, PostFacts, ScoreFacts, UserFacts};
use crate::growthbook::DynamicConfig;
use crate::growthbook_writer::GrowthBookWriter;
use crate::service::{
    AppState, EnforcementAction, PermanentAisFailure, enforce_actions, handle_allowlist_bulk,
    handle_allowlist_delete, handle_allowlist_delete_entity, handle_allowlist_get,
    handle_allowlist_get_entity, handle_allowlist_list, handle_allowlist_put,
    handle_allowlist_put_entity, handle_config_patch_field, handle_config_put, handle_dedup_get,
    handle_dedup_get_entity, handle_rate_limit, handle_rate_limit_increment,
    handle_rate_limit_reset,
};
use crate::strato::{
    fetch_cred, fetch_entity_allowlist, fetch_uas, fetch_user, fetch_user_allowlist,
};

const SERVICE_NAME: &str = env!("CARGO_PKG_NAME");
const SERVICE_VERSION: &str = env!("CARGO_PKG_VERSION");


#[derive(Debug, Deserialize)]
struct TopicEntry {
    #[serde(default)]
    cluster: Option<String>,
    #[serde(default)]
    zone: Option<String>,
    #[serde(flatten)]
    processor: TopicConfig,
}

impl TopicEntry {
            fn cluster<'a>(&'a self, default: &'a str) -> &'a str {
        override_or_default(self.cluster.as_deref(), default)
    }

            fn zone<'a>(&'a self, default: &'a str) -> &'a str {
        override_or_default(self.zone.as_deref(), default)
    }
}

fn override_or_default<'a>(value: Option<&'a str>, default: &'a str) -> &'a str {
    match value.map(str::trim) {
        Some(v) if !v.is_empty() => v,
        _ => default,
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "processor", rename_all = "snake_case")]
enum TopicConfig {
            ScoreResult {
                                #[serde(default)]
        labels: HashSet<String>,
                        #[serde(default)]
        negative_labels: HashSet<String>,
    },
            StatsOnly {
        #[serde(default)]
        should_log_content: bool,
    },
}


const MAX_RETRIES: u32 = 5;

struct RetryEntry {
    score: abuse_proto::ScoreResult,
    attempt: u32,
    next_retry_at: tokio::time::Instant,
        topic: String,
        dedup_cache: ManhattanDedupCache,
                    claim_nonce: Option<u64>,
}

#[derive(Debug)]
struct EnforcementFailed {
    permanent: bool,
    claim_nonce: Option<u64>,
    source: anyhow::Error,
}

impl PartialEq for RetryEntry {
    fn eq(&self, other: &Self) -> bool {
        self.next_retry_at == other.next_retry_at
    }
}
impl Eq for RetryEntry {}
impl PartialOrd for RetryEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RetryEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.next_retry_at.cmp(&self.next_retry_at)
    }
}

#[derive(Clone)]
#[allow(deprecated)]
struct EnforcementCtx {
    ais_client: ais_client::AisClient,
    gd: gizmoduck::GizmoduckCoreClient,
    cred_clients: strato::CredClients,
    uas_strato: Arc<Strato>,
    dynamic_config: DynamicConfig,
            rules_cache: Arc<rules::RulesCache>,
    allowlist: Option<allowlist::ManhattanAllowlist>,
            kafka_producer_decisions: Option<Arc<KafkaProducer>>,
                kafka_producer_decisions_json: Option<Arc<KafkaProducer>>,
                            hold_gate: Arc<overturn_hold::HoldGate>,
}

fn gated_actions(
    specs: &[decision::ActionSpec],
    cfg: &overturn_hold::HoldGateConfig,
) -> Vec<overturn_hold::GatedAction> {
    use overturn_hold::GatedAction;
    let mut out = Vec::new();
    if cfg.gates_suspend() {
        let mut perm: Option<bool> = None;
        for spec in specs {
            if let decision::ActionSpec::SuspendUser { perm: p, .. } = spec {
                let all_perm = perm.get_or_insert(true);
                *all_perm &= *p;
            }
        }
        if let Some(perm) = perm {
            out.push(GatedAction::Suspend { perm });
        }
    }
    for spec in specs {
        if let decision::ActionSpec::AddLabelsV2 { labels, .. } = spec {
            for name in labels {
                let already = out
                    .iter()
                    .any(|a| matches!(a, GatedAction::Label { name: n } if n == name));
                if cfg.gates_label(name) && !already {
                    out.push(GatedAction::Label { name: name.clone() });
                }
            }
        }
    }
    out
}

fn strip_held_labels(specs: &mut Vec<decision::ActionSpec>, held: &[String]) {
    if held.is_empty() {
        return;
    }
    for spec in specs.iter_mut() {
        if let decision::ActionSpec::AddLabelsV2 { labels, .. } = spec {
            labels.retain(|l| !held.contains(l));
        }
    }
    specs.retain(
        |s| !matches!(s, decision::ActionSpec::AddLabelsV2 { labels, .. } if labels.is_empty()),
    );
}

struct ScoreResultProcessor {
    topic: String,
    labels: HashSet<String>,
    negative_labels: HashSet<String>,
    ctx: EnforcementCtx,
    dedup_cache: ManhattanDedupCache,
    retry_queue: Arc<Mutex<BinaryHeap<RetryEntry>>>,
        last_progress: Arc<AtomicI64>,
        max_in_flight: usize,
}


fn decision_outcome(
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
    status: String,
    dry_run: bool,
    info: BTreeMap<String, String>,
) -> abuse_proto::DecisionOutcome {
    let summary = score.summary.as_ref();
    let (entity_type, entity_id) = Facts::resolve_target(score);
    abuse_proto::DecisionOutcome {
        decided_at_ms: now_millis(),
        source_topic: source_topic.to_owned(),
        entity_type: entity_type.as_str().to_owned(),
        entity_id,
        model_version: score.model_version.clone(),
        status,
        dry_run,
        info,
        head: crate::facts::head_label(score).to_owned(),
        fired_heads: summary.map(|s| s.fired_heads.clone()).unwrap_or_default(),
        labels: summary.map(|s| s.labels.clone()).unwrap_or_default(),
        score_id: score.score_id.clone(),
    }
}

const REQUESTED_ACTIONS_DENIED: &str = "requested_actions_denied";

const ALREADY_SUSPENDED: &str = "already_suspended";

const ALREADY_SUSPENDED_STRIPPED: &str = "already_suspended_stripped";

fn strip_redundant_suspends(
    specs: &mut Vec<decision::ActionSpec>,
    user: &crate::facts::GizmoduckFacts,
) -> Vec<decision::ActionSpec> {
    let is_suspend =
        |s: &decision::ActionSpec| matches!(s, decision::ActionSpec::SuspendUser { .. });
    let has_perm = specs
        .iter()
        .any(|s| matches!(s, decision::ActionSpec::SuspendUser { perm: true, .. }));
    if !user.suspended || has_perm || !specs.iter().any(is_suspend) {
        return Vec::new();
    }
    let (stripped, kept) = std::mem::take(specs).into_iter().partition(is_suspend);
    *specs = kept;
    stripped
}

#[derive(Debug, PartialEq)]
enum ExpandedDecision {
    Skip(String),
    Act(Vec<decision::ActionSpec>),
}

fn expand_requested_actions_decision(
    entity_type: EntityType,
    score_facts: &ScoreFacts,
    allowlist: &generic_actions::GenericActionAllowlist,
) -> (ExpandedDecision, Option<String>) {
    let resolved = generic_actions::resolve_requested_actions(
        entity_type,
        &score_facts.requested_actions,
        allowlist,
    );
    let skipped_json = resolved.skipped_info_json();
    if resolved.specs.is_empty() {
        (
            ExpandedDecision::Skip(REQUESTED_ACTIONS_DENIED.into()),
            skipped_json,
        )
    } else {
        (ExpandedDecision::Act(resolved.specs), skipped_json)
    }
}

#[tracing::instrument(skip_all, fields(dry_run, uas))]
async fn run_enforcement_inner(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
) -> Result<abuse_proto::DecisionOutcome> {
    let user_id = score.user_id;
    let (entity_type, entity_id) = Facts::resolve_target(score);
    let dry_run = ctx.dynamic_config.is_dry_run();
    let span = tracing::Span::current();
    span.record("dry_run", dry_run);

    let gated_user_id = match entity_type {
        EntityType::User => entity_id,
        EntityType::Post => user_id,
    };
    if test_user::is_test_user_id(gated_user_id) {
        info!(
            user_id = gated_user_id,
            "test user (id bitmask/range); skipping enforcement"
        );
        let entity = match entity_type {
            EntityType::User => EntityFacts::User(UserFacts::default()),
            EntityType::Post => EntityFacts::Post(PostFacts {
                author_id: user_id,
                ..Default::default()
            }),
        };
        let partial = Facts {
            entity_type,
            entity_id,
            user_id: gated_user_id,
            topic: source_topic.to_owned(),
            score: ScoreFacts::from_score(score),
            entity,
            uas: None,
        };
        return Ok(skip_outcome(
            "test_user_skipped".into(),
            dry_run,
            &partial,
            score,
        ));
    }

    let mut facts = match entity_type {
        EntityType::User => {
            let user_id = entity_id;
            let allowlist = fetch_user_allowlist(ctx.allowlist.as_ref(), user_id).await;
            if allowlist.is_allowlisted {
                let partial = Facts {
                    entity_type,
                    entity_id,
                    user_id,
                    topic: source_topic.to_owned(),
                    score: ScoreFacts::from_score(score),
                    entity: EntityFacts::User(UserFacts {
                        allowlist,
                        ..Default::default()
                    }),
                    uas: None,
                };
                return Ok(skip_outcome(
                    "user_in_allowlist".into(),
                    dry_run,
                    &partial,
                    score,
                ));
            }

            let (user_res, cred_res) = tokio::join!(
                fetch_user(&ctx.gd, user_id),
                fetch_cred(&ctx.cred_clients, user_id),
            );

            Facts {
                entity_type,
                entity_id,
                user_id,
                topic: source_topic.to_owned(),
                score: ScoreFacts::from_score(score),
                entity: EntityFacts::User(UserFacts {
                    allowlist,
                    user: user_res?,
                    cred: cred_res?,
                }),
                uas: None,
            }
        }
        EntityType::Post => {
            let author_id = user_id;
            let (post_allowlist, author_allowlist) = tokio::join!(
                fetch_entity_allowlist(ctx.allowlist.as_ref(), EntityType::Post, entity_id),
                fetch_user_allowlist(ctx.allowlist.as_ref(), author_id),
            );
            if post_allowlist.is_allowlisted || author_allowlist.is_allowlisted {
                let reason = if post_allowlist.is_allowlisted {
                    "post_in_allowlist"
                } else {
                    "user_in_allowlist"
                };
                let partial = Facts {
                    entity_type,
                    entity_id,
                    user_id: author_id,
                    topic: source_topic.to_owned(),
                    score: ScoreFacts::from_score(score),
                    entity: EntityFacts::Post(PostFacts {
                        allowlist: post_allowlist,
                        present: false,
                        author_id,
                        labels: Vec::new(),
                        author: UserFacts {
                            allowlist: author_allowlist,
                            ..Default::default()
                        },
                    }),
                    uas: None,
                };
                return Ok(skip_outcome(reason.into(), dry_run, &partial, score));
            }

            let (user_res, cred_res) = tokio::join!(
                fetch_user(&ctx.gd, author_id),
                fetch_cred(&ctx.cred_clients, author_id),
            );

            Facts {
                entity_type,
                entity_id,
                user_id: author_id,
                topic: source_topic.to_owned(),
                score: ScoreFacts::from_score(score),
                entity: EntityFacts::Post(PostFacts {
                    allowlist: post_allowlist,
                    present: false,
                    author_id,
                    labels: Vec::new(),
                    author: UserFacts {
                        allowlist: author_allowlist,
                        user: user_res?,
                        cred: cred_res?,
                    },
                }),
                uas: None,
            }
        }
    };

    let rules_override = ctx.dynamic_config.enforcement_rules_yaml(facts.entity_type);
    let compiled_rules = ctx
        .rules_cache
        .resolve(facts.entity_type, rules_override.as_ref());
    let decision = match crate::rules::decide_with(&compiled_rules, &facts) {
        Ok(d) => d,
        Err(e) => {
            warn!(
                user_id,
                topic = %facts.topic,
                error = %e,
                "rule pipeline eval errored — skipping; investigate the rule file",
            );
            return Ok(skip_outcome(
                "rule_eval_error".into(),
                dry_run,
                &facts,
                score,
            ));
        }
    };

    let (decision, mut requested_actions_skipped) = match decision {
        Decision::ActRequestedActions => expand_requested_actions_decision(
            facts.entity_type,
            &facts.score,
            &ctx.dynamic_config
                .generic_action_allowlist(facts.entity_type),
        ),
        Decision::Skip(reason) => (ExpandedDecision::Skip(reason), None),
        Decision::Act(specs) => (ExpandedDecision::Act(specs), None),
    };

    match decision {
        ExpandedDecision::Skip(reason) => {
            let mut outcome = skip_outcome(reason, dry_run, &facts, score);
            if let Some(json) = requested_actions_skipped {
                outcome
                    .info
                    .insert("requested_actions_skipped".into(), json);
            }
            Ok(outcome)
        }
        ExpandedDecision::Act(mut specs) => {
            let stripped = strip_redundant_suspends(&mut specs, facts.user());
            let mut already_suspended_stripped = None;
            if !stripped.is_empty() {
                let names: Vec<&'static str> = stripped
                    .iter()
                    .map(|spec| {
                        EnforcementAction::from_spec(
                            spec,
                            facts.entity_id,
                            facts.user_id,
                            Vec::new(),
                        )
                        .name()
                    })
                    .collect();
                let names = serde_json::to_string(&names).unwrap_or_else(|_| "[]".into());
                if dry_run {
                    info!(
                        user_id = facts.user_id,
                        actions = %names,
                        "dry run; would call suspendUser, but the account is already suspended"
                    );
                }
                if specs.is_empty() {
                    info!(
                        user_id = facts.user_id,
                        "suspend target already suspended; skipping decision"
                    );
                    let mut outcome =
                        skip_outcome(ALREADY_SUSPENDED.into(), dry_run, &facts, score);
                    outcome
                        .info
                        .insert(ALREADY_SUSPENDED_STRIPPED.into(), names);
                    if let Some(json) = requested_actions_skipped.take() {
                        outcome
                            .info
                            .insert("requested_actions_skipped".into(), json);
                    }
                    return Ok(outcome);
                }
                info!(
                    user_id = facts.user_id,
                    stripped = %names,
                    "suspend target already suspended; dropping the suspend, keeping the other actions"
                );
                already_suspended_stripped = Some(names);
            }

            // Overturn-hold gate: skip a suspend, or strip a label, that a
            // human reviewer overturned on appeal while that hold is still active.
            let mut hold_gate_info = BTreeMap::new();
            let gate_cfg = ctx.dynamic_config.overturn_hold_gate();
            let gated = gated_actions(&specs, &gate_cfg);
            if !gated.is_empty() {
                let mut gate = ctx
                    .hold_gate
                    .evaluate(&gate_cfg, facts.user_id, source_topic, &gated)
                    .await;
                if let Some(status) = gate.skip_status {
                    return Ok(hold_gate_skip_outcome(
                        status,
                        gate.info,
                        dry_run,
                        &facts,
                        score,
                        requested_actions_skipped.take(),
                    ));
                }
                if !gate.strip_labels.is_empty() {
                    strip_held_labels(&mut specs, &gate.strip_labels);
                    if specs.is_empty() {
                        gate.mark_label_collapse();
                        info!(
                            user_id = facts.user_id,
                            topic = source_topic,
                            labels_stripped = %gate.info
                                .get("overturn_hold_labels_stripped")
                                .map(String::as_str)
                                .unwrap_or(""),
                            "overturn-hold gate (enforce): every action was a held label; skipping decision"
                        );
                        return Ok(hold_gate_skip_outcome(
                            overturn_hold::STATUS_HOLD_OVERTURNED,
                            gate.info,
                            dry_run,
                            &facts,
                            score,
                            requested_actions_skipped.take(),
                        ));
                    }
                }
                hold_gate_info = gate.info;
            }

            if !ctx.dynamic_config.try_enforce(facts.entity_type).await {
                warn!(
                    entity_type = facts.entity_type.as_str(),
                    "max enforcement has been reached; skipping"
                );
                let mut outcome =
                    skip_outcome("max_enforcement_reached".into(), dry_run, &facts, score);
                if let Some(json) = requested_actions_skipped.take() {
                    outcome
                        .info
                        .insert("requested_actions_skipped".into(), json);
                }
                return Ok(outcome);
            }

            let user_id = facts.user_id;
            let entity_id = facts.entity_id;

            facts.uas = fetch_uas(&ctx.uas_strato, user_id).await;
            if let Some(uas) = &facts.uas {
                span.record("uas", tracing::field::display(uas));
            }

            info!(
                "processing score_result: model={}, status={}",
                score.model_version, score.score_status,
            );

            let mut additional_info_map = facts.to_additional_info_map();
            let ais_notes: Vec<String> = additional_info_map
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect();
            let acted_class = specs
                .iter()
                .map(decision::ActionSpec::dedup_action_class)
                .max()
                .unwrap_or(dedup_cache::CLASS_NONE);
            let actions: Vec<EnforcementAction> = specs
                .iter()
                .map(|spec| {
                    EnforcementAction::from_spec(spec, entity_id, user_id, ais_notes.clone())
                })
                .collect();
            let uas_action_dims: Vec<(&'static str, String)> = actions
                .iter()
                .map(|a| (a.name(), a.metric_label()))
                .collect();

            additional_info_map.insert(
                "action_kinds".into(),
                serde_json::to_string(&uas_action_dims.iter().map(|(a, _)| *a).collect::<Vec<_>>())
                    .unwrap_or_else(|_| "[]".into()),
            );
            if let Some(json) = requested_actions_skipped {
                additional_info_map.insert("requested_actions_skipped".into(), json);
            }
            additional_info_map.extend(hold_gate_info);
            if let Some(names) = already_suspended_stripped {
                additional_info_map.insert(ALREADY_SUSPENDED_STRIPPED.into(), names);
            }

            enforce_actions(
                &ctx.ais_client,
                dry_run,
                Some(score),
                source_topic,
                facts.entity_type,
                actions,
            )
            .await?;
            info!("enforcement complete");
            let status = if dry_run { "dry_run" } else { "success" }.to_string();
            if !dry_run {
                additional_info_map.insert("acted_class".into(), acted_class.to_string());
            }
            if let Some(uas) = &facts.uas {
                let dims: Vec<(&str, &str)> = uas_action_dims
                    .iter()
                    .map(|(a, l)| (*a, l.as_str()))
                    .collect();
                metrics::observe_user_active_seconds(uas, facts.entity_type, &status, &dims);
            }
            Ok(decision_outcome(
                score,
                source_topic,
                status,
                dry_run,
                additional_info_map,
            ))
        }
    }
}

fn skip_outcome(
    reason: String,
    dry_run: bool,
    facts: &Facts,
    score: &abuse_proto::ScoreResult,
) -> abuse_proto::DecisionOutcome {
    decision_outcome(
        score,
        &facts.topic,
        reason,
        dry_run,
        facts.to_additional_info_map(),
    )
}

fn hold_gate_skip_outcome(
    status: &str,
    gate_info: BTreeMap<String, String>,
    dry_run: bool,
    facts: &Facts,
    score: &abuse_proto::ScoreResult,
    requested_actions_skipped: Option<String>,
) -> abuse_proto::DecisionOutcome {
    let mut outcome = skip_outcome(status.to_owned(), dry_run, facts, score);
    outcome.info.extend(gate_info);
    if let Some(json) = requested_actions_skipped {
        outcome
            .info
            .insert("requested_actions_skipped".into(), json);
    }
    outcome
}

#[cfg(test)]
fn outcome_holds_full_dedup(status: &str) -> bool {
    dedup_retention_for(status) == DedupRetention::Full
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DedupRetention {
        Full,
            Skip,
                Release,
}

fn dedup_retention_for(status: &str) -> DedupRetention {
    if status == "success" {
        DedupRetention::Full
    } else if status == overturn_hold::STATUS_HOLD_LOOKUP_FAILED {
        DedupRetention::Release
    } else {
        DedupRetention::Skip
    }
}

async fn write_dedup_outcome(
    dedup_cache: &ManhattanDedupCache,
    entity_type: EntityType,
    entity_id: i64,
    score: &abuse_proto::ScoreResult,
    topic: &str,
    outcome: &abuse_proto::DecisionOutcome,
    claim_nonce: Option<u64>,
) {
    let retention = dedup_retention_for(&outcome.status);
    if retention == DedupRetention::Release {
        dedup_cache
            .invalidate(entity_type, entity_id, claim_nonce)
            .await;
        return;
    }
    let mut score_for_dedup = score.clone();
    score_for_dedup.decoded_actions.clear();
    let acted_class = outcome
        .info
        .get("acted_class")
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(dedup_cache::CLASS_NONE);
    let entry = entities::DedupEntry {
        status: &outcome.status,
        acted_class,
        topic,
        dry_run: outcome.dry_run,
        additional_info_map: &outcome.info,
        score_result: score_for_dedup,
    };
    let json_bytes = serde_json::to_vec(&entry).unwrap_or_default();
    let compressed = zstd::encode_all(json_bytes.as_slice(), 3).unwrap_or(json_bytes);
    if retention == DedupRetention::Full {
        dedup_cache
            .update(
                entity_type,
                entity_id,
                &compressed,
                acted_class,
                claim_nonce,
            )
            .await;
    } else {
        dedup_cache
            .update_skip(
                entity_type,
                entity_id,
                &compressed,
                acted_class,
                claim_nonce,
            )
            .await;
    }
}

const DECISIONS_PRODUCER: &str = "decisions";

const DECISIONS_JSON_PRODUCER: &str = "decisions_json";

pub(crate) const ADMIN_ACTIONS_PRODUCER: &str = "admin_actions";

#[derive(Clone, Default)]
pub struct KafkaProducers {
    by_name: HashMap<String, Arc<KafkaProducer>>,
}

impl KafkaProducers {
            pub fn get(&self, name: &str) -> Option<&Arc<KafkaProducer>> {
        self.by_name.get(name)
    }

            pub fn decisions(&self) -> Option<&Arc<KafkaProducer>> {
        self.get(DECISIONS_PRODUCER)
    }

            pub fn decisions_json(&self) -> Option<&Arc<KafkaProducer>> {
        self.get(DECISIONS_JSON_PRODUCER)
    }

            pub fn admin_actions(&self) -> Option<&Arc<KafkaProducer>> {
        self.get(ADMIN_ACTIONS_PRODUCER)
    }

                            pub async fn flush_all(&self, timeout: Duration) {
        for (name, producer) in &self.by_name {
            if let Err(e) = producer.flush(timeout).await {
                warn!("kafka producer '{name}' flush on shutdown failed: {e}");
            }
        }
    }
}

pub(crate) fn spawn_publish<M: prost::Message>(
    producer: Option<&Arc<KafkaProducer>>,
    sink: &'static str,
    key: impl FnOnce() -> Vec<u8>,
    record: &M,
) {
    let Some(producer) = producer else {
        return;
    };
    spawn_send(producer.clone(), sink, key(), record.encode_to_vec());
}

pub(crate) fn spawn_publish_json<M: serde::Serialize>(
    producer: Option<&Arc<KafkaProducer>>,
    sink: &'static str,
    key: impl FnOnce() -> Vec<u8>,
    record: &M,
) {
    let Some(producer) = producer else {
        return;
    };
    let bytes = match serde_json::to_vec(record) {
        Ok(bytes) => bytes,
        Err(e) => {
            warn!("{sink} JSON encode failed, record dropped: {e}");
            metrics::KAFKA_PUBLISH_TOTAL
                .with_label_values(&[sink, "encode_error"])
                .inc();
            return;
        }
    };
    spawn_send(producer.clone(), sink, key(), bytes);
}

fn spawn_send(producer: Arc<KafkaProducer>, sink: &'static str, key: Vec<u8>, bytes: Vec<u8>) {
    tokio::spawn(async move {
        let result = match producer.send_with_key(Some(key.as_slice()), &bytes).await {
            Ok(_) => "ok",
            Err(e) => {
                tracing::debug!("{sink} produce failed (ignored): {e}");
                "error"
            }
        };
        metrics::KAFKA_PUBLISH_TOTAL
            .with_label_values(&[sink, result])
            .inc();
    });
}

fn publish_decision_outcome(ctx: &EnforcementCtx, outcome: &abuse_proto::DecisionOutcome) {
    spawn_publish(
        ctx.kafka_producer_decisions.as_ref(),
        DECISIONS_PRODUCER,
        || outcome.entity_id.to_string().into_bytes(),
        outcome,
    );
    spawn_publish_json(
        ctx.kafka_producer_decisions_json.as_ref(),
        DECISIONS_JSON_PRODUCER,
        || outcome.entity_id.to_string().into_bytes(),
        outcome,
    );
}

async fn run_enforcement(
    ctx: &EnforcementCtx,
    score: &abuse_proto::ScoreResult,
    source_topic: &str,
) -> Result<abuse_proto::DecisionOutcome> {
    let outcome = run_enforcement_inner(ctx, score, source_topic).await?;
    publish_decision_outcome(ctx, &outcome);
    Ok(outcome)
}

async fn build_kafka_producers(cfg: &Config, boot_config: &serde_json::Value) -> KafkaProducers {
    let mut producers = KafkaProducers::default();

    for (name, spec) in growthbook::kafka_producers_from_config(Some(boot_config)) {
        if !spec.enabled {
            warn!("kafka producer '{name}' configured but disabled (enabled=false)");
            continue;
        }
        let Some(topic) = spec.topic else {
            warn!("kafka producer '{name}' enabled but no topic set; skipping");
            continue;
        };
        let cluster =
            override_or_default(spec.cluster.as_deref(), &cfg.kafka_producer_mtls_cluster);
        let zone = override_or_default(spec.zone.as_deref(), &cfg.kafka_producer_mtls_zone);
        let producer_config = match KafkaProducerConfigBuilder::for_cluster_mtls_auto(
            cluster,
            topic.clone(),
            Some(zone),
        ) {
            Ok(builder) => builder.build(),
            Err(e) => {
                error!("kafka producer '{name}' mTLS config failed; skipping: {e}");
                continue;
            }
        };
        let mut producer = KafkaProducer::new(producer_config);
        match producer.start().await {
            Ok(()) => {
                info!("kafka producer '{name}' started -> topic {topic} on {cluster} ({zone})");
                producers.by_name.insert(name, Arc::new(producer));
            }
            Err(e) => {
                error!("kafka producer '{name}' failed to start; continuing without it: {e}")
            }
        }
    }
    producers
}

impl ScoreResultProcessor {
                                                #[tracing::instrument(skip_all)]
    async fn process(&self, score: &abuse_proto::ScoreResult) -> Result<String, EnforcementFailed> {
        let source = "kafka";
        let topic = self.topic.as_str();
        let head = crate::facts::head_label(score);
        let model = score.model_version.as_str();
        let (entity_type, entity_id) = Facts::resolve_target(score);

        if entity_id == 0 {
            warn!(
                entity_type = entity_type.as_str(),
                "score has no entity id; skipping"
            );
            metrics::ENFORCEMENT_TOTAL
                .with_label_values(&[
                    source,
                    topic,
                    entity_type.as_str(),
                    "final",
                    "invalid_entity_id",
                    "",
                    "",
                    head,
                    model,
                ])
                .inc();
            publish_decision_outcome(
                &self.ctx,
                &decision_outcome(
                    score,
                    topic,
                    "invalid_entity_id".into(),
                    false,
                    BTreeMap::new(),
                ),
            );
            return Ok("invalid_entity_id".into());
        }

        let incoming_class = self.ctx.dynamic_config.dedup_action_class_for_head(head);
        let claim_nonce = match self
            .dedup_cache
            .try_insert(entity_type, entity_id, incoming_class)
            .await
        {
            DedupCheck::New { claim_nonce } => claim_nonce,
            DedupCheck::ClassBypass {
                held_class,
                claim_nonce,
            } => {
                info!(
                    incoming_class,
                    held_class, "dedup: class bypass (incoming outranks held entry)"
                );
                metrics::DEDUP_CLASS_BYPASS_TOTAL
                    .with_label_values(&[
                        topic,
                        head,
                        &held_class.to_string(),
                        &incoming_class.to_string(),
                    ])
                    .inc();
                claim_nonce
            }
            DedupCheck::Duplicate => {
                info!("dedup: skipping (already enforced recently)");
                metrics::ENFORCEMENT_TOTAL
                    .with_label_values(&[
                        source,
                        topic,
                        entity_type.as_str(),
                        "final",
                        "dedup_skipped",
                        "",
                        "",
                        head,
                        model,
                    ])
                    .inc();
                publish_decision_outcome(
                    &self.ctx,
                    &decision_outcome(score, topic, "dedup_skipped".into(), false, BTreeMap::new()),
                );
                return Ok("dedup_skipped".into());
            }
        };

        let outcome = match run_enforcement(&self.ctx, score, topic).await {
            Ok(outcome) => outcome,
            Err(source) => {
                let permanent = source.downcast_ref::<PermanentAisFailure>().is_some();
                if permanent {
                    self.dedup_cache
                        .invalidate(entity_type, entity_id, claim_nonce)
                        .await;
                }
                return Err(EnforcementFailed {
                    permanent,
                    claim_nonce,
                    source,
                });
            }
        };

        write_dedup_outcome(
            &self.dedup_cache,
            entity_type,
            entity_id,
            score,
            topic,
            &outcome,
            claim_nonce,
        )
        .await;

        let status = outcome.status;

        if let Some(summary) = score.summary.as_ref() {
            for fh in &summary.fired_heads {
                metrics::FIRED_HEADS_TOTAL
                    .with_label_values(&[
                        topic,
                        entity_type.as_str(),
                        model,
                        fh.name.as_str(),
                        status.as_str(),
                    ])
                    .inc();
            }
        }

        if status != "success" {
            info!("enforcement outcome: {status}");
            metrics::ENFORCEMENT_TOTAL
                .with_label_values(&[
                    source,
                    topic,
                    entity_type.as_str(),
                    "final",
                    &status,
                    "",
                    "",
                    head,
                    model,
                ])
                .inc();
        }

        Ok(status)
    }
}

#[async_trait::async_trait]
impl KafkaBatchProcessor for ScoreResultProcessor {
    async fn process_batch(&self, messages: &[KafkaMessage]) -> Result<BatchResult> {
        self.last_progress.store(now_millis(), Ordering::Relaxed);
        let max_in_flight = self.max_in_flight.max(1);
        let gate = Arc::new(Semaphore::new(max_in_flight));
        let failed_scores: Vec<Option<_>> = join_all(messages.iter().map(|message| {
            let start = std::time::Instant::now();
            let topic = self.topic.as_str();
            let gate = gate.clone();
            let span = tracing::info_span!(
                "message",
                topic = message.topic,
                entity_type = tracing::field::Empty,
                entity_id = tracing::field::Empty
            );
            async move {
                let _permit = gate
                    .acquire_owned()
                    .await
                    .expect("semaphore closed unexpectedly");
                let Some(payload) = &message.payload else {
                    metrics::KAFKA_MESSAGE_LATENCY
                        .with_label_values(&[topic, "", "no_payload", "false"])
                        .observe(start.elapsed().as_secs_f64());
                    return None;
                };

                let score = match abuse_proto::ScoreResult::decode(payload.as_slice()) {
                    Ok(s) => s,
                    Err(e) => {
                        warn!("failed to decode ScoreResult: {e}");
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, "", "decode_failed", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        return None;
                    }
                };

                let (entity_type, entity_id) = Facts::resolve_target(&score);
                let entity_type = entity_type.as_str();
                let span = tracing::Span::current();
                span.record("entity_type", entity_type);
                span.record("entity_id", entity_id);

                let msg_labels: Vec<&String> = score
                    .summary
                    .as_ref()
                    .map(|s| s.labels.iter().collect())
                    .unwrap_or_default();

                if !self.negative_labels.is_empty()
                    && msg_labels.iter().any(|l| self.negative_labels.contains(*l))
                {
                    metrics::KAFKA_MESSAGE_LATENCY
                        .with_label_values(&[topic, entity_type, "negative_label", "false"])
                        .observe(start.elapsed().as_secs_f64());
                    return None;
                }

                if !self.labels.is_empty() {
                    let matches: Vec<&&String> = msg_labels
                        .iter()
                        .filter(|l| self.labels.contains(**l))
                        .collect();
                    if matches.is_empty() {
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, "no_match", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        return None;
                    }

                    info!(?matches, "matching labels");
                }

                match self.process(&score).await {
                    Ok(status) => {
                        let enforced = if status == "success" { "true" } else { "false" };
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, &status, enforced])
                            .observe(start.elapsed().as_secs_f64());
                        None
                    }
                    Err(f) if f.permanent => {
                        error!(
                            "enforcement failed permanently (not retried): {}",
                            f.source
                        );
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, "permanent_failure", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        None
                    }
                    Err(f) => {
                        warn!(error = %format!("{:#}", f.source), "enforcement failed (queued for retry)");
                        metrics::KAFKA_MESSAGE_LATENCY
                            .with_label_values(&[topic, entity_type, "error", "false"])
                            .observe(start.elapsed().as_secs_f64());
                        Some((score, f.claim_nonce))
                    }
                }
            }
            .instrument(span)
        }))
        .await;

        {
            let mut queue = self.retry_queue.lock().await;
            for (score, claim_nonce) in failed_scores.into_iter().flatten() {
                queue.push(RetryEntry {
                    score,
                    attempt: 1,
                    next_retry_at: tokio::time::Instant::now() + Duration::from_secs(1),
                    topic: self.topic.clone(),
                    dedup_cache: self.dedup_cache.clone(),
                    claim_nonce,
                });
            }
            metrics::RETRY_QUEUE_SIZE.set(queue.len() as i64);
        }

        Ok(BatchResult::CommitAll)
    }
}


struct StatsOnlyProcessor {
    should_log_content: bool,
        last_progress: Arc<AtomicI64>,
}

#[async_trait::async_trait]
impl KafkaBatchProcessor for StatsOnlyProcessor {
    async fn process_batch(&self, messages: &[KafkaMessage]) -> Result<BatchResult> {
        self.last_progress.store(now_millis(), Ordering::Relaxed);
        if self.should_log_content {
            for message in messages {
                if let Some(payload) = &message.payload {
                    match abuse_proto::ScoreResult::decode(payload.as_slice()) {
                        Ok(score) => {
                            let json = serde_json::to_value(&score).unwrap_or_default();
                            info!(
                                topic = message.topic,
                                partition = message.partition,
                                offset = message.offset,
                                "message content: {}",
                                json
                            );
                        }
                        Err(e) => {
                            warn!(
                                topic = message.topic,
                                partition = message.partition,
                                offset = message.offset,
                                "failed to decode ScoreResult: {e}"
                            );
                        }
                    }
                }
            }
        }
        Ok(BatchResult::CommitAll)
    }
}


pub fn init_runtime_globals() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init();
    xai_init_utils::init().rustls();
    metrics::init();
}

pub async fn build_state(cfg: Config) -> Result<(Arc<service::AppState>, Config)> {
    info!("Service starting (port={})", cfg.port);

    let ais_cert = cfg
        .strato_client_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.crt".into());
    let ais_key = cfg
        .strato_client_key_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.key".into());
    let ais_ca = cfg
        .strato_ca_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/ca/ca-bundle.crt".into());
    let ais_client = ais_client::AisClient::connect(&ais_client::AisClientConfig {
        wily_path: cfg.ais_wily_path.clone(),
        tls_server_name: cfg.ais_tls_server_name.clone(),
        zone: cfg.ais_zone.clone(),
        s2s_cert_path: ais_cert,
        s2s_key_path: ais_key,
        s2s_ca_path: ais_ca,
    })
    .context("failed to initialise AIS ThriftMux client")?;

    let limiter = match cfg.limiter_datacenter.as_deref() {
        Some(datacenter) if !datacenter.is_empty() => {
            let limiter_cfg = limiter::LimiterConfig {
                datacenter: datacenter.to_owned(),
                feature: cfg.limiter_feature.clone(),
                fail_open: cfg.limiter_fail_open,
                client_id: cfg.limiter_client_id.clone(),
                ca_cert_path: cfg.limiter_ca_cert_path.clone(),
                client_cert_path: cfg.limiter_client_cert_path.clone(),
                client_key_path: cfg.limiter_client_key_path.clone(),
            };
            match limiter::LimiterClient::connect(&limiter_cfg).await {
                Ok(client) => Some(client),
                Err(e) => {
                    warn!("Limiter connect failed: {e}; global enforcement cap disabled");
                    None
                }
            }
        }
        _ => {
            info!("LIMITER_DATACENTER not set; global enforcement cap disabled (all allowed)");
            None
        }
    };

    let mh_client = manhattan::build_client(&cfg).await?;
    let mh_tenant = manhattan::build_tenant(&cfg);
    info!(
        "Manhattan storage: cluster={} app={} dataset={} dc={} (dedup ttl={}s skip_ttl={}s)",
        cfg.mh_cluster,
        cfg.mh_app_id,
        cfg.mh_dataset,
        cfg.mh_datacenter,
        cfg.dedup_ttl_secs,
        cfg.dedup_skip_ttl_secs,
    );
    let dedup_cache = Some(dedup_cache::ManhattanDedupCache::new(
        mh_client.clone(),
        mh_tenant.clone(),
        cfg.dedup_ttl_secs,
        cfg.dedup_skip_ttl_secs,
    ));
    let allowlist = Some(allowlist::ManhattanAllowlist::new(
        mh_client.clone(),
        mh_tenant.clone(),
    ));

    let (sliding_user, sliding_post) = match limiter.as_ref() {
        Some(lim) => {
            let user_sw = sliding_window::SlidingWindowLimiter::new(
                lim.clone(),
                mh_client.clone(),
                mh_tenant.clone(),
                cfg.limiter_fail_open,
                sliding_window::USER_SNAP_PKEY,
                "user",
            );
            let post_sw = sliding_window::SlidingWindowLimiter::new(
                lim.with_bucket(limiter::GLOBAL_BUCKET_POST_ID),
                mh_client.clone(),
                mh_tenant.clone(),
                cfg.limiter_fail_open,
                sliding_window::POST_SNAP_PKEY,
                "post",
            );
            user_sw.spawn_background();
            post_sw.spawn_background();
            info!(
                "Sliding-window enforcement caps enabled for user + post (Limiter counters + Manhattan snapshots)"
            );
            (Some(user_sw), Some(post_sw))
        }
        None => (None, None),
    };

    if cfg
        .growthbook_url
        .as_deref()
        .filter(|s| !s.is_empty())
        .is_none()
        || cfg
            .growthbook_key
            .as_deref()
            .filter(|s| !s.is_empty())
            .is_none()
    {
        anyhow::bail!(
            "GrowthBook is required (GROWTHBOOK_URL + GROWTHBOOK_KEY): \
             needed for live config + rule features \
             xai_abuse_enforcement_service_rule_{{user,post}}"
        );
    }
    let dynamic_config = DynamicConfig::new(
        cfg.growthbook_url.as_deref(),
        cfg.growthbook_key.as_deref(),
        cfg.dry_run,
        cfg.max_enforcements_per_day,
        cfg.max_post_enforcements_per_day,
        sliding_user,
        sliding_post,
    )
    .await
    .map_err(|e| anyhow::anyhow!(e))?;

    let rules_cache = rules::RulesCache::new();
    let user_rules_yaml = dynamic_config
        .enforcement_rules_yaml(EntityType::User)
        .filter(|s| !s.trim().is_empty());
    let post_rules_yaml = dynamic_config
        .enforcement_rules_yaml(EntityType::Post)
        .filter(|s| !s.trim().is_empty());
    let (Some(user_yaml), Some(post_yaml)) = (user_rules_yaml, post_rules_yaml) else {
        anyhow::bail!(
            "GrowthBook rule features must both be set and non-empty at startup \
             (xai_abuse_enforcement_service_rule_user + \
             xai_abuse_enforcement_service_rule_post); refusing to boot on \
             baked-in defaults alone"
        );
    };
    let start = std::time::Instant::now();
    let user_status = rules_cache.status(EntityType::User, Some(&user_yaml));
    let post_status = rules_cache.status(EntityType::Post, Some(&post_yaml));
    if user_status.source != rules::RuleSource::Dynamic {
        anyhow::bail!(
            "failed to compile GrowthBook user rules at startup: {}",
            user_status
                .error
                .as_deref()
                .unwrap_or("override present but not dynamic")
        );
    }
    if post_status.source != rules::RuleSource::Dynamic {
        anyhow::bail!(
            "failed to compile GrowthBook post rules at startup: {}",
            post_status
                .error
                .as_deref()
                .unwrap_or("override present but not dynamic")
        );
    }
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        user_rules = user_status.rule_ids.len(),
        post_rules = post_status.rule_ids.len(),
        user_source = ?user_status.source,
        post_source = ?post_status.source,
        "GrowthBook rule pipelines compiled and live (baked-in defaults retained as fallback)"
    );

    let growthbook_writer = cfg.growthbook_admin_api_key.clone().map(|k| {
        info!(
            "GrowthBook writer enabled (host={}, feature={}, env={})",
            cfg.growthbook_admin_api_host, cfg.growthbook_feature_key, cfg.growthbook_environment,
        );
        GrowthBookWriter::new(&cfg.growthbook_admin_api_host, k)
    });
    if growthbook_writer.is_none() {
        info!("GROWTHBOOK_ADMIN_API_KEY not set — admin config-mutation endpoints will return 503");
    }

    let boot_config = dynamic_config.config().unwrap_or(serde_json::Value::Null);

    let kafka_producers = build_kafka_producers(&cfg, &boot_config).await;

    let startup_probe =
        overturn_hold::startup_probe_requested(cfg.overturn_hold_startup_probe.as_deref());
    let hold_gate = Arc::new(overturn_hold::HoldGate::from_url(
        cfg.overturn_hold_ledger_url.as_deref(),
        startup_probe,
        cfg.overturn_hold_env.as_deref(),
    ));
    if startup_probe {
        let gate = hold_gate.clone();
        tokio::spawn(async move {
            gate.startup_probe().await;
        });
    }
    hold_gate.publish_config(&dynamic_config.overturn_hold_gate());

    let state = Arc::new(AppState {
        ais_client,
        dynamic_config,
        rules_cache: Arc::new(rules_cache),
        dedup_cache,
        limiter,
        allowlist,
        growthbook_writer,
        growthbook_feature_key: cfg.growthbook_feature_key.clone(),
        growthbook_environment: cfg.growthbook_environment.clone(),
        kafka_ready: Arc::new(AtomicBool::new(false)),
        kafka_producers,
        hold_gate,
        boot_config,
    });

    Ok((state, cfg))
}

pub fn build_api_router(state: Arc<service::AppState>, api_keys: auth::ApiKeys) -> Router {
    use axum::routing::{get, patch, post};

    let protected = Router::new()
        .route("/action", post(service::handle_action))
        .route(
            "/config",
            get(service::handle_config).put(handle_config_put),
        )
        .route("/config/{field}", patch(handle_config_patch_field))
        .route("/rules", get(service::handle_rules_get))
        .route("/allowlist", get(handle_allowlist_list))
        .route("/allowlist/bulk", post(handle_allowlist_bulk))
        .route(
            "/allowlist/{user_id}",
            get(handle_allowlist_get)
                .put(handle_allowlist_put)
                .delete(handle_allowlist_delete),
        )
        .route(
            "/allowlist/{entity_type}/{entity_id}",
            get(handle_allowlist_get_entity)
                .put(handle_allowlist_put_entity)
                .delete(handle_allowlist_delete_entity),
        )
        .route(
            "/rate_limit/{entity_type}",
            post(handle_rate_limit_increment),
        )
        .route(
            "/rate_limit/{entity_type}/reset",
            post(handle_rate_limit_reset),
        )
        .with_state(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            api_keys,
            auth::require_api_key,
        ));

    let public = Router::new()
        .route("/dry_run", get(service::handle_dry_run))
        .route("/rate_limit/{entity_type}", get(handle_rate_limit))
        .route("/dedup/{user_id}", get(handle_dedup_get))
        .route(
            "/dedup/{entity_type}/{entity_id}",
            get(handle_dedup_get_entity),
        )
        .with_state(state);

    protected.merge(public)
}

pub fn build_docs_router() -> Router {
    use utoipa::OpenApi;
    use utoipa_swagger_ui::SwaggerUi;

    let openapi = api_doc::ApiDoc::openapi();

    Router::new().merge(SwaggerUi::new("/api/docs").url("/api/openapi.json", openapi))
}

pub(crate) fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn consumer_error_total(watched: &HashSet<String>) -> f64 {
    use prometheus::core::Collector;
    if watched.is_empty() {
        return 0.0;
    }
    xai_kafka::metrics::CONSUMER_ERROR_COUNT
        .collect()
        .iter()
        .flat_map(|mf| &mf.metric)
        .filter(|m| {
            m.label
                .iter()
                .any(|l| l.name() == "topic" && watched.contains(l.value()))
        })
        .filter_map(|m| m.counter.as_ref())
        .filter_map(|c| c.value)
        .sum()
}

fn health_decision(
    consumed_once: bool,
    stale: Duration,
    error_bad_for: Duration,
    stale_after: Duration,
    stale_self_delete_after: Duration,
    error_self_delete_after: Duration,
    error_bad_now: bool,
) -> (bool, bool) {
    let ready = consumed_once && stale < stale_after && !error_bad_now;
    let should_self_delete =
        stale >= stale_self_delete_after || error_bad_for >= error_self_delete_after;
    (ready, should_self_delete)
}

struct WatchdogConfig {
    interval: Duration,
    stale_after: Duration,
    stale_self_delete_after: Duration,
    error_rate_per_sec: f64,
    error_self_delete_after: Duration,
    self_delete_enabled: bool,
            error_topics: HashSet<String>,
}

async fn kafka_health_watchdog(
    last_progress_ms: Arc<AtomicI64>,
    kafka_ready: Arc<AtomicBool>,
    cfg: WatchdogConfig,
) {
    let started_ms = now_millis();
    let interval_secs = cfg.interval.as_secs_f64().max(1.0);
    let mut ticker = tokio::time::interval(cfg.interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut errors_prev = consumer_error_total(&cfg.error_topics);
    let mut error_bad_since: Option<tokio::time::Instant> = None;
    let mut deleted = false;

    loop {
        ticker.tick().await;

        let last = last_progress_ms.load(Ordering::Relaxed);
        let consumed_once = last > 0;
        let reference = if consumed_once { last } else { started_ms };
        let stale = Duration::from_millis((now_millis() - reference).max(0) as u64);

        let errors_now = consumer_error_total(&cfg.error_topics);
        let err_rate = (errors_now - errors_prev).max(0.0) / interval_secs;
        errors_prev = errors_now;
        let error_bad_now = cfg.error_rate_per_sec > 0.0 && err_rate >= cfg.error_rate_per_sec;
        let now = tokio::time::Instant::now();
        let error_bad_for = if error_bad_now {
            now.duration_since(*error_bad_since.get_or_insert(now))
        } else {
            error_bad_since = None;
            Duration::ZERO
        };

        let (healthy, should_self_delete) = health_decision(
            consumed_once,
            stale,
            error_bad_for,
            cfg.stale_after,
            cfg.stale_self_delete_after,
            cfg.error_self_delete_after,
            error_bad_now,
        );

        if kafka_ready.swap(healthy, Ordering::Relaxed) != healthy {
            info!(
                healthy,
                stale_secs = stale.as_secs(),
                err_rate,
                "kafka watchdog: readiness change"
            );
        }

        if should_self_delete && cfg.self_delete_enabled && !deleted {
            deleted = true;
            let reason = if error_bad_now && error_bad_for >= cfg.error_self_delete_after {
                "error_flood"
            } else {
                "stale"
            };
            error!(
                stale_secs = stale.as_secs(),
                err_rate,
                reason,
                "kafka watchdog: bad node (no progress or broker errors); self-deleting pod"
            );
            metrics::KAFKA_SELF_DELETE_TOTAL
                .with_label_values(&[reason])
                .inc();
            self_delete_pod().await;
        }
        if healthy {
            deleted = false;
        }
    }
}

const KAFKA_PREFLIGHT_ATTEMPTS: u32 = 3;
const KAFKA_PREFLIGHT_TIMEOUT: Duration = Duration::from_secs(10);
const KAFKA_PREFLIGHT_BACKOFF: Duration = Duration::from_secs(2);

async fn probe_broker_reachability(config: KafkaConsumerConfig, timeout: Duration) -> Result<()> {
    use rdkafka::ClientConfig;
    use rdkafka::consumer::{BaseConsumer, Consumer};

    let topic = config.base_config.topic.clone();
    let brokers = resolve_kafka_brokers(&config.base_config).await?;

    let mut client_config = ClientConfig::new();
    client_config
        .set("group.id", &config.group_id)
        .set("bootstrap.servers", brokers.join(","))
        .set("enable.ssl.certificate.verification", "false")
        .set("ssl.endpoint.identification.algorithm", "none");

    apply_auth_config(
        &mut client_config,
        config.base_config.ssl.as_ref(),
        config.s2s_certs.as_ref(),
    );

    let consumer: BaseConsumer = client_config
        .create()
        .context("Failed to create BaseConsumer for reachability probe")?;

    tokio::task::spawn_blocking(move || -> Result<()> {
        consumer
            .fetch_metadata(Some(&topic), timeout)
            .context("Kafka broker reachability probe failed")?;
        Ok(())
    })
    .await
    .context("Reachability probe task panicked")?
}

const KAFKA_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(20);

pub struct KafkaConsumers {
    cancel: CancellationToken,
    supervisor: JoinHandle<()>,
}

impl KafkaConsumers {
                                                async fn shutdown(self, timeout: Duration) {
        self.cancel.cancel();
        let mut supervisor = self.supervisor;
        if tokio::time::timeout(timeout, &mut supervisor)
            .await
            .is_err()
        {
            warn!(
                "Kafka consumers did not stop within {timeout:?}; aborting the in-flight batch \
                 (redelivered to the next owner; its dedup claims may be stranded)"
            );
            supervisor.abort();
            let _ = supervisor.await;
        }
    }
}

pub async fn shutdown_kafka(state: &service::AppState, consumers: KafkaConsumers) {
    consumers.shutdown(KAFKA_SHUTDOWN_TIMEOUT).await;
    state
        .kafka_producers
        .flush_all(Duration::from_secs(5))
        .await;
}

pub async fn start_kafka_consumers(
    state: Arc<service::AppState>,
    cfg: &Config,
) -> Result<KafkaConsumers> {
    let cancel = CancellationToken::new();
    if !cfg.kafka_consumer_enabled {
        info!("KAFKA_CONSUMER_ENABLED=false — Kafka consumer disabled");
        state.kafka_ready.store(true, Ordering::Relaxed);
        return Ok(KafkaConsumers {
            cancel,
            supervisor: tokio::spawn(async {}),
        });
    }

    let growthbook_enabled = cfg.growthbook_url.is_some() && cfg.growthbook_key.is_some();
    let topic_labels_json: serde_json::Value =
        match std::fs::read_to_string(&cfg.topic_labels_config) {
            Ok(data) => serde_json::from_str(&data).with_context(|| {
                format!(
                    "failed to parse topic-labels config: {}",
                    cfg.topic_labels_config
                )
            })?,
            Err(e) if growthbook_enabled => {
                warn!(
                    "topic-labels config not found ({e}), falling back to GrowthBook / empty config"
                );
                serde_json::Value::Object(Default::default())
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "failed to read topic-labels config {}: {e}",
                    cfg.topic_labels_config
                ));
            }
        };

    let effective_topic_config =
        growthbook::kafka_consumer_topic_labels_from_config(Some(&state.boot_config))
            .unwrap_or(topic_labels_json);
    let topics = parse_topic_config(effective_topic_config)?;
    let default_cluster = cfg.kafka_consumer_mtls_cluster.as_str();
    let default_zone = cfg.kafka_consumer_mtls_zone.as_str();
    info!("topic config: {} topic(s)", topics.len());
    for (topic, entry) in &topics {
        let cluster = entry.cluster(default_cluster);
        let zone = entry.zone(default_zone);
        match &entry.processor {
            TopicConfig::ScoreResult {
                labels,
                negative_labels,
            } => {
                info!(
                    "  topic={topic}, cluster={cluster}, zone={zone}, processor=score_result, labels={}, negative_labels={}",
                    labels.len(),
                    negative_labels.len()
                );
            }
            TopicConfig::StatsOnly { should_log_content } => {
                info!(
                    "  topic={topic}, cluster={cluster}, zone={zone}, processor=stats_only, should_log_content={should_log_content}"
                );
            }
        }
    }

    let s2s_certs = xai_kafka::load_s2s_certs()
        .context("failed to load S2S mTLS client certificate for Kafka consumers")?;

    let default_target = (default_cluster, default_zone);
    let probe_topic = if cfg.kafka_self_delete_enabled {
        preflight_probe_topic(
            topics
                .iter()
                .map(|(t, e)| (t.as_str(), e.cluster(default_cluster), e.zone(default_zone))),
            default_target,
        )
        .map(str::to_owned)
    } else {
        None
    };
    if cfg.kafka_self_delete_enabled && probe_topic.is_none() {
        info!(
            "Kafka broker preflight skipped: no topic on the default target \
             ({default_cluster}/{default_zone}); the runtime watchdog remains the only bad-node gate"
        );
    }
    if let Some(probe_topic) = probe_topic {
        let probe_config = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
            default_cluster,
            probe_topic.clone(),
            format!("{}-preflight", cfg.kafka_group_id()),
            s2s_certs.clone(),
            Some(default_zone),
        )
        .context("failed to configure mTLS consumer preflight")?
        .with_enable_auto_offset_store(false)
        .with_enable_auto_commit(false)
        .with_fetch_timeout_ms(10000)
        .build();

        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let start = std::time::Instant::now();
            match probe_broker_reachability(probe_config.clone(), KAFKA_PREFLIGHT_TIMEOUT).await {
                Ok(()) => {
                    info!(
                        attempt,
                        elapsed_ms = start.elapsed().as_millis() as u64,
                        topic = %probe_topic,
                        "Kafka broker preflight ok"
                    );
                    break;
                }
                Err(e) if attempt < KAFKA_PREFLIGHT_ATTEMPTS => {
                    warn!(
                        topic = %probe_topic,
                        "Kafka broker preflight attempt {attempt}/{KAFKA_PREFLIGHT_ATTEMPTS} failed \
                         (retrying in {KAFKA_PREFLIGHT_BACKOFF:?}): {e:#}"
                    );
                    tokio::time::sleep(KAFKA_PREFLIGHT_BACKOFF).await;
                }
                Err(e) => {
                    error!(
                        topic = %probe_topic,
                        "Kafka broker preflight failed after {KAFKA_PREFLIGHT_ATTEMPTS} attempts — \
                         likely a bad node; self-deleting to reschedule: {e:#}"
                    );
                    metrics::KAFKA_SELF_DELETE_TOTAL
                        .with_label_values(&["preflight"])
                        .inc();
                    self_delete_pod().await;
                    return Err(anyhow::anyhow!(
                        "Kafka broker preflight failed after {KAFKA_PREFLIGHT_ATTEMPTS} attempts: {e:#}"
                    ));
                }
            }
        }
    }

    let ca = cfg
        .strato_ca_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/ca/ca-bundle.crt".into());
    let crt = cfg
        .strato_client_cert_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.crt".into());
    let key = cfg
        .strato_client_key_path
        .clone()
        .unwrap_or_else(|| "/etc/strato-tls/client/tls.key".into());
    let gd = gizmoduck::GizmoduckCoreClient::connect(&gizmoduck::GizmoduckCoreClientConfig {
        zone: cfg.gizmoduck_zone.clone(),
        ca_cert_path: ca,
        client_cert_path: crt,
        client_key_path: key,
        client_id: cfg.gizmoduck_client_id.clone(),
    })
    .await
    .context("failed to create gizmoduck get-V2 fed-grpc client")?;

    #[allow(deprecated)]
    let cred_hpr = Arc::new(
        Strato::new(&cfg.high_page_rank_column, StratoClientConfig::default())
            .context("Failed to create high-page-rank Strato client")?,
    );
    #[allow(deprecated)]
    let cred_grey = Arc::new(
        Strato::new(&cfg.grey_badge_column, StratoClientConfig::default())
            .context("Failed to create grey-badge Strato client")?,
    );
    let cred_clients = strato::CredClients {
        high_page_rank: cred_hpr,
        grey_badge: cred_grey,
    };
    info!(
        high_page_rank_column = %cfg.high_page_rank_column,
        grey_badge_column = %cfg.grey_badge_column,
        "Cred Strato clients initialised"
    );

    #[allow(deprecated)]
    let uas_strato = Arc::new(
        Strato::new(&cfg.uas_column, StratoClientConfig::default())
            .context("Failed to create UAS Strato client")?,
    );
    info!(
        "UAS Strato client initialised for column {}",
        cfg.uas_column
    );

    let retry_queue = Arc::new(Mutex::new(BinaryHeap::<RetryEntry>::new()));

    let enforcement_ctx = EnforcementCtx {
        ais_client: state.ais_client.clone(),
        gd,
        cred_clients,
        uas_strato,
        dynamic_config: state.dynamic_config.clone(),
        rules_cache: state.rules_cache.clone(),
        allowlist: state.allowlist.clone(),
        kafka_producer_decisions: state.kafka_producers.decisions().cloned(),
        kafka_producer_decisions_json: state.kafka_producers.decisions_json().cloned(),
        hold_gate: state.hold_gate.clone(),
    };

    {
        let retry_ctx = enforcement_ctx.clone();
        let retry_queue = retry_queue.clone();
        tokio::spawn(async move {
            loop {
                let sleep_until = {
                    let queue = retry_queue.lock().await;
                    queue.peek().map(|e| e.next_retry_at)
                };
                match sleep_until {
                    Some(t) => tokio::time::sleep_until(t).await,
                    None => tokio::time::sleep(Duration::from_secs(5)).await,
                }

                let now = tokio::time::Instant::now();
                let mut ready: Vec<RetryEntry> = Vec::new();
                {
                    let mut queue = retry_queue.lock().await;
                    while queue.peek().is_some_and(|e| e.next_retry_at <= now) {
                        ready.push(queue.pop().unwrap());
                    }
                }

                for mut entry in ready {
                    let (entity_type, entity_id) = Facts::resolve_target(&entry.score);
                    let span = tracing::info_span!(
                        "retry",
                        entity_type = entity_type.as_str(),
                        entity_id,
                        attempt = entry.attempt
                    );

                    match run_enforcement(&retry_ctx, &entry.score, &entry.topic)
                        .instrument(span.clone())
                        .await
                    {
                        Ok(outcome) => {
                            info!(parent: &span, "retry resolved: {}", outcome.status);
                            metrics::RETRY_ATTEMPTS
                                .with_label_values(&["success"])
                                .observe(entry.attempt as f64);
                            write_dedup_outcome(
                                &entry.dedup_cache,
                                entity_type,
                                entity_id,
                                &entry.score,
                                &entry.topic,
                                &outcome,
                                entry.claim_nonce,
                            )
                            .await;
                        }
                        Err(e) if e.downcast_ref::<PermanentAisFailure>().is_some() => {
                            error!(parent: &span, "retry abandoned (permanent failure): {e}");
                            metrics::RETRY_ATTEMPTS
                                .with_label_values(&["permanent"])
                                .observe(entry.attempt as f64);
                            entry
                                .dedup_cache
                                .invalidate(entity_type, entity_id, entry.claim_nonce)
                                .await;
                        }
                        Err(e) => {
                            entry.attempt += 1;
                            if entry.attempt > MAX_RETRIES {
                                error!(parent: &span, "retry exhausted after {MAX_RETRIES} attempts: {e}");
                                metrics::RETRY_ATTEMPTS
                                    .with_label_values(&["exhausted"])
                                    .observe((entry.attempt - 1) as f64);
                                entry
                                    .dedup_cache
                                    .invalidate(entity_type, entity_id, entry.claim_nonce)
                                    .await;
                            } else {
                                let backoff = Duration::from_secs(1 << (entry.attempt - 1));
                                warn!(parent: &span, "retry failed: {e}, next in {backoff:?}");
                                entry.next_retry_at = tokio::time::Instant::now() + backoff;
                                retry_queue.lock().await.push(entry);
                            }
                        }
                    }
                }

                metrics::RETRY_QUEUE_SIZE.set(retry_queue.lock().await.len() as i64);
            }
        });
    }

    let last_progress = Arc::new(AtomicI64::new(0));

    let mut consumers: JoinSet<()> = JoinSet::new();
    let mut error_topics: HashSet<String> = HashSet::new();
    let topics_len = topics.len();

    let dedup_cache = state
        .dedup_cache
        .clone()
        .expect("Manhattan dedup cache required when Kafka is enabled");

    let kafka_group_id = cfg.kafka_group_id();

    let max_messages_per_poll = cfg.kafka_max_messages_per_poll.max(1);
    let max_in_flight = cfg.kafka_max_in_flight.max(1);
    info!(
        max_messages_per_poll,
        max_in_flight, "Kafka consumer batch limits"
    );

    let make_batch_config =
        |topic: &str, cluster: &str, zone: &str| -> Result<BatchConsumerConfig> {
            let kafka = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
                cluster,
                topic.to_owned(),
                topic_consumer_group_id(&kafka_group_id, topic),
                s2s_certs.clone(),
                Some(zone),
            )
            .with_context(|| {
                format!("failed to configure {cluster} mTLS consumer for {topic} (zone {zone})")
            })?
            .with_enable_auto_offset_store(false)
            .with_enable_auto_commit(false)
            .with_fetch_timeout_ms(10000)
            .build();

            Ok(BatchConsumerConfig::new(kafka, SERVICE_NAME)
                .with_max_messages_per_poll(max_messages_per_poll))
        };

    for (topic, entry) in topics {
        let cluster = entry.cluster(default_cluster).to_owned();
        let zone = entry.zone(default_zone).to_owned();

        let batch_config = match make_batch_config(&topic, &cluster, &zone) {
            Ok(config) => config,
            Err(e) => {
                error!("skipping Kafka consumer for topic {topic}: {e:#}");
                metrics::KAFKA_CONSUMER_START_TOTAL
                    .with_label_values(&[topic.as_str(), cluster.as_str(), "config_error"])
                    .inc();
                continue;
            }
        };
        metrics::KAFKA_CONSUMER_START_TOTAL
            .with_label_values(&[topic.as_str(), cluster.as_str(), "started"])
            .inc();
        if is_default_target((&cluster, &zone), default_target) {
            error_topics.insert(topic.clone());
        }
        let cancel = cancel.clone();

        match entry.processor {
            TopicConfig::ScoreResult {
                labels,
                negative_labels,
            } => {
                let processor = ScoreResultProcessor {
                    topic: topic.clone(),
                    labels,
                    negative_labels,
                    ctx: enforcement_ctx.clone(),
                    dedup_cache: dedup_cache.clone(),
                    retry_queue: retry_queue.clone(),
                    last_progress: last_progress.clone(),
                    max_in_flight,
                };

                consumers.spawn(async move {
                    info!("starting score_result Kafka consumer for topic {topic} on {cluster}");
                    if let Err(e) = run_batch_consumer(batch_config, processor, cancel).await {
                        error!("Kafka consumer for topic {topic} exited with error: {e}");
                    }
                });
            }
            TopicConfig::StatsOnly { should_log_content } => {
                let processor = StatsOnlyProcessor {
                    should_log_content,
                    last_progress: last_progress.clone(),
                };

                consumers.spawn(async move {
                    info!("starting stats_only Kafka consumer for topic {topic} on {cluster}");
                    if let Err(e) = run_batch_consumer(batch_config, processor, cancel).await {
                        error!(
                            "stats_only Kafka consumer for topic {topic} exited with error: {e}"
                        );
                    }
                });
            }
        }
    }

    if consumers.is_empty() {
        if topics_len > 0 {
            error!(
                "no Kafka consumers started ({} topic(s) configured, all skipped); \
                 serving without consumers — check \
                 abuse_enforcement_kafka_consumer_start_total{{result=\"config_error\"}}",
                topics_len
            );
        }
        state.kafka_ready.store(true, Ordering::Relaxed);
    } else {
        tokio::spawn(kafka_health_watchdog(
            last_progress.clone(),
            state.kafka_ready.clone(),
            WatchdogConfig {
                interval: Duration::from_secs(cfg.kafka_watchdog_interval_secs),
                stale_after: Duration::from_secs(cfg.kafka_watchdog_stale_secs),
                stale_self_delete_after: Duration::from_secs(cfg.kafka_watchdog_self_delete_secs),
                error_rate_per_sec: cfg.kafka_watchdog_error_rate_per_sec,
                error_self_delete_after: Duration::from_secs(
                    cfg.kafka_watchdog_error_self_delete_secs,
                ),
                self_delete_enabled: cfg.kafka_self_delete_enabled,
                error_topics,
            },
        ));
    }

    let rl_config = state.dynamic_config.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            for entity_type in [
                crate::facts::EntityType::User,
                crate::facts::EntityType::Post,
            ] {
                let remaining = rl_config
                    .rate_limit_stats(entity_type)
                    .map_or(-1, |(_used, _cap, remaining)| remaining as i64);
                metrics::RATE_LIMIT_REMAINING
                    .with_label_values(&[entity_type.as_str()])
                    .set(remaining);
            }
        }
    });

    Ok(KafkaConsumers {
        cancel,
        supervisor: supervise(consumers),
    })
}

fn supervise(mut consumers: JoinSet<()>) -> JoinHandle<()> {
    tokio::spawn(async move { while consumers.join_next().await.is_some() {} })
}

fn topic_consumer_group_id(base_group_id: &str, topic: &str) -> String {
    format!("{base_group_id}-{topic}")
}

fn is_default_target(target: (&str, &str), default: (&str, &str)) -> bool {
    target == default
}

fn preflight_probe_topic<'a, I>(topics: I, default: (&str, &str)) -> Option<&'a str>
where
    I: IntoIterator<Item = (&'a str, &'a str, &'a str)>,
{
    topics
        .into_iter()
        .filter(|(_, cluster, zone)| is_default_target((cluster, zone), default))
        .map(|(topic, _, _)| topic)
        .min()
}

fn parse_topic_config(config: serde_json::Value) -> Result<HashMap<String, TopicEntry>> {
    serde_json::from_value(config)
        .context("topic config has wrong shape (expected {topic: {processor, ...}})")
}

pub(crate) fn validate_restart_config(config: &serde_json::Value) -> Result<()> {
    match growthbook::kafka_consumer_topic_labels_from_config(Some(config)) {
        Some(topics) => parse_topic_config(topics).map(drop),
        None => Ok(()),
    }
}

const CONFIG_REFRESH_INTERVAL: Duration = Duration::from_secs(15);

fn config_refresh_tick(
    dynamic_config: &DynamicConfig,
    hold_gate: &overturn_hold::HoldGate,
    boot_config: &serde_json::Value,
) -> restart_on_config_change::Check {
    dynamic_config.refresh();
    hold_gate.publish_config(&dynamic_config.overturn_hold_gate());
    let live = dynamic_config.config().unwrap_or(serde_json::Value::Null);
    let check = restart_on_config_change::check(boot_config, &live, validate_restart_config);
    restart_on_config_change::record(&check);
    check
}

pub fn spawn_config_refresh(
    state: &service::AppState,
    shutdown: xai_service_runner::ShutdownSignal,
) {
    spawn_config_refresh_with(
        state.dynamic_config.clone(),
        state.hold_gate.clone(),
        state.boot_config.clone(),
        shutdown,
        CONFIG_REFRESH_INTERVAL,
    );
}

fn spawn_config_refresh_with(
    dynamic_config: DynamicConfig,
    hold_gate: Arc<overturn_hold::HoldGate>,
    boot_config: serde_json::Value,
    shutdown: xai_service_runner::ShutdownSignal,
    interval: Duration,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await;
        loop {
            tick.tick().await;
            if let restart_on_config_change::Check::Restart(paths) =
                config_refresh_tick(&dynamic_config, &hold_gate, &boot_config)
            {
                info!("restarting to apply config change: paths={paths:?}");
                shutdown.trigger();
                return;
            }
        }
    })
}

pub async fn serve(router: Router, state: &service::AppState, cfg: &Config) -> Result<()> {
    let server = ServerBuilder::new(cfg.port)
        .merge(router)
        .drain_period(Duration::from_secs(cfg.drain_period_secs))
        .info(ServerInfo::new(SERVICE_NAME, SERVICE_VERSION))
        .on_ready(|| async {
            info!("Server ready and accepting requests");
        });
    spawn_config_refresh(state, server.shutdown_signal());
    server.run().await
}

pub async fn run() -> Result<()> {
    init_runtime_globals();

    let cfg = Config::parse();
    let (state, cfg) = build_state(cfg).await?;
    let api_keys = auth::ApiKeys::from_config(&cfg)?;
    if api_keys.is_empty() {
        warn!(
            "API_KEYS_JSON is empty — every request to the protected admin \
             endpoints (/action, /config*, /allowlist*) will return 401"
        );
    } else {
        info!(
            key_ids = ?api_keys.configured_key_ids(),
            "api-key auth configured for protected admin endpoints"
        );
    }
    let router = Router::new()
        .nest("/api", build_api_router(state.clone(), api_keys))
        .merge(build_docs_router());
    let kafka_consumers = start_kafka_consumers(state.clone(), &cfg).await?;

    let served = serve(router, &state, &cfg).await;
    shutdown_kafka(&state, kafka_consumers).await;
    served?;
    info!("Server shutdown complete");
    Ok(())
}

#[cfg(test)]
mod kafka_shutdown_tests {
    use super::*;

    #[tokio::test]
    async fn shutdown_waits_for_the_in_flight_batch() {
        let cancel = CancellationToken::new();
        let finished = Arc::new(AtomicBool::new(false));
        let supervisor = tokio::spawn({
            let (cancel, finished) = (cancel.clone(), finished.clone());
            async move {
                cancel.cancelled().await;
                tokio::time::sleep(Duration::from_millis(50)).await;
                finished.store(true, Ordering::SeqCst);
            }
        });

        KafkaConsumers { cancel, supervisor }
            .shutdown(Duration::from_secs(10))
            .await;

        assert!(finished.load(Ordering::SeqCst));
    }

            struct DropFlag(Arc<AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn shutdown_timeout_aborts_the_consumers() {
        let stopped = Arc::new(AtomicBool::new(false));
        let mut consumers = JoinSet::new();
        {
            let guard = DropFlag(stopped.clone());
            consumers.spawn(async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            });
        }
        let started = std::time::Instant::now();

        KafkaConsumers {
            cancel: CancellationToken::new(),
            supervisor: supervise(consumers),
        }
        .shutdown(Duration::from_millis(50))
        .await;
        assert!(started.elapsed() < Duration::from_secs(5));

        tokio::time::timeout(Duration::from_secs(5), async {
            while !stopped.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the stuck consumer must be aborted, not detached");
    }
}

#[cfg(test)]
mod router_split_tests {
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request;
    use axum::middleware::Next;
    use axum::response::Response;
    use axum::routing::{get, post};

    async fn ok() -> &'static str {
        "ok"
    }

    async fn pass(req: Request<Body>, next: Next) -> Response {
        next.run(req).await
    }

                                            #[test]
    fn same_path_split_across_sub_routers_merges() {
        let protected = Router::new()
            .route("/rate_limit/{entity_type}", post(ok))
            .layer(axum::middleware::from_fn(pass));
        let public = Router::new().route("/rate_limit/{entity_type}", get(ok));
        let _app: Router = protected.merge(public);
    }
}

#[cfg(test)]
mod config_restart_tests {
    use super::validate_restart_config;
    use serde_json::json;

    fn valid_topics() -> serde_json::Value {
        json!({
            "abuse.v3.score_results": {
                "processor": "score_result",
                "labels": ["enforcement_threshold_reached"],
                "cluster": "mltraining",
            },
            "abuse.stats.v1": { "processor": "stats_only" },
        })
    }

    #[test]
    fn valid_topic_map_passes() {
        let config = json!({ "kafka": { "consumer": { "topic_labels": valid_topics() } } });
        validate_restart_config(&config).unwrap();
    }

    #[test]
    fn wrong_shape_topic_map_fails() {
        let array = json!({ "kafka": { "consumer": { "topic_labels": ["abuse.stats.v1"] } } });
        assert!(validate_restart_config(&array).is_err());

        let no_processor = json!({ "kafka": { "consumer": { "topic_labels": {
            "abuse.stats.v1": { "cluster": "phoenix" },
        } } } });
        assert!(validate_restart_config(&no_processor).is_err());

        let unknown_processor = json!({ "kafka": { "consumer": { "topic_labels": {
            "abuse.stats.v1": { "processor": "nope" },
        } } } });
        assert!(validate_restart_config(&unknown_processor).is_err());
    }

    #[test]
    fn absent_topic_map_passes() {
        validate_restart_config(&serde_json::Value::Null).unwrap();
        validate_restart_config(&json!({})).unwrap();
        validate_restart_config(&json!({ "kafka": { "producer": { "decisions": [1] } } })).unwrap();
    }

            #[tokio::test]
    async fn restart_trigger_stops_the_server() {
        use std::time::Duration;

        let server =
            xai_service_runner::ServerBuilder::new(0).drain_period(Duration::from_millis(10));
        let dc = test_dynamic_config(json!({ "topic_labels": {} })).await;
        let refresh = super::spawn_config_refresh_with(
            dc,
            test_hold_gate(),
            json!({}),
            server.shutdown_signal(),
            Duration::from_millis(5),
        );

        tokio::time::timeout(Duration::from_secs(10), server.run())
            .await
            .expect("server did not shut down after the restart trigger")
            .unwrap();
        tokio::time::timeout(Duration::from_secs(10), refresh)
            .await
            .expect("refresh task did not end")
            .unwrap();
    }

            async fn test_dynamic_config(config: serde_json::Value) -> crate::growthbook::DynamicConfig {
        let dc = crate::growthbook::DynamicConfig::new(None, None, false, 1, 1, None, None)
            .await
            .unwrap();
        dc.set_snapshot_for_test(crate::growthbook::ConfigSnapshot {
            config: Some(config),
            ..Default::default()
        });
        dc
    }

    fn test_hold_gate() -> std::sync::Arc<crate::overturn_hold::HoldGate> {
        std::sync::Arc::new(crate::overturn_hold::HoldGate::from_url(None, false, None))
    }

    #[tokio::test]
    async fn refresh_tick_restarts_only_on_a_valid_boot_only_change() {
        use crate::restart_on_config_change::Check;

        let boot =
            json!({ "dry_run": false, "topic_labels": { "a": { "processor": "stats_only" } } });
        let gate = test_hold_gate();

        let mut live = boot.clone();
        live["dry_run"] = json!(true);
        let dc = test_dynamic_config(live).await;
        assert_eq!(
            super::config_refresh_tick(&dc, &gate, &boot),
            Check::Unchanged
        );
        assert!(dc.is_dry_run(), "snapshot kept (no client to refresh from)");

        let mut live = boot.clone();
        live["topic_labels"]["b"] = json!({ "processor": "stats_only" });
        let dc = test_dynamic_config(live).await;
        assert_eq!(
            super::config_refresh_tick(&dc, &gate, &boot),
            Check::Restart(vec!["/topic_labels".to_string()])
        );

        let mut live = boot.clone();
        live["topic_labels"] = json!(["not a map"]);
        let dc = test_dynamic_config(live).await;
        assert_eq!(
            super::config_refresh_tick(&dc, &gate, &boot),
            Check::Invalid
        );
    }

    #[test]
    fn legacy_top_level_topic_map_is_validated() {
        validate_restart_config(&json!({ "topic_labels": valid_topics() })).unwrap();
        let bad = json!({ "topic_labels": { "abuse.stats.v1": "stats_only" } });
        assert!(validate_restart_config(&bad).is_err());
    }
}

#[cfg(test)]
mod kafka_topic_config_tests {
    use super::{
        KAFKA_PREFLIGHT_ATTEMPTS, KAFKA_PREFLIGHT_BACKOFF, KAFKA_PREFLIGHT_TIMEOUT, TopicConfig,
        TopicEntry, consumer_error_total, is_default_target, preflight_probe_topic,
        topic_consumer_group_id,
    };
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use std::time::Duration;
    use xai_kafka::{KafkaConsumerConfigBuilder, S2sCerts};

    #[test]
    fn dynamic_topics_keep_distinct_existing_group_ids() {
        let base_group_id = "xai-abuse-enforcement-service";

        assert_eq!(
            topic_consumer_group_id(base_group_id, "scores.primary"),
            "xai-abuse-enforcement-service-scores.primary"
        );
        assert_eq!(
            topic_consumer_group_id(base_group_id, "scores.secondary"),
            "xai-abuse-enforcement-service-scores.secondary"
        );
    }

    #[test]
    fn preflight_retry_budget_stays_bounded() {
        assert_eq!(KAFKA_PREFLIGHT_ATTEMPTS, 3);
        assert_eq!(KAFKA_PREFLIGHT_TIMEOUT, Duration::from_secs(10));
        assert_eq!(KAFKA_PREFLIGHT_BACKOFF, Duration::from_secs(2));
    }


    fn parse(json: serde_json::Value) -> HashMap<String, TopicEntry> {
        serde_json::from_value(json).expect("topic config should deserialize")
    }

                #[test]
    fn topic_entry_without_override_uses_deployment_default() {
        let topics = parse(json!({
            "abuse.v3.score_results": {
                "processor": "score_result",
                "labels": ["enforcement_threshold_reached"],
            },
            "abuse.stats.v1": { "processor": "stats_only" },
        }));

        for topic in ["abuse.v3.score_results", "abuse.stats.v1"] {
            let entry = &topics[topic];
            assert_eq!(entry.cluster("phoenix"), "phoenix");
            assert_eq!(entry.zone("atla"), "atla");
            assert!(entry.cluster.is_none());
        }
        assert!(matches!(
            topics["abuse.v3.score_results"].processor,
            TopicConfig::ScoreResult { .. }
        ));
        assert!(matches!(
            topics["abuse.stats.v1"].processor,
            TopicConfig::StatsOnly { .. }
        ));
    }

                    #[test]
    fn absent_null_and_blank_all_resolve_to_the_default() {
        let topics = parse(json!({
            "omitted": { "processor": "stats_only" },
            "explicit_null": { "processor": "stats_only", "cluster": null, "zone": null },
            "empty_string": { "processor": "stats_only", "cluster": "", "zone": "" },
            "whitespace": { "processor": "stats_only", "cluster": "   ", "zone": "\t" },
        }));

        for topic in ["omitted", "explicit_null", "empty_string", "whitespace"] {
            let entry = &topics[topic];
            assert_eq!(entry.cluster("phoenix"), "phoenix", "cluster for {topic}");
            assert_eq!(entry.zone("atla"), "atla", "zone for {topic}");
        }

        let padded = parse(json!({
            "t": { "processor": "stats_only", "cluster": " mltraining ", "zone": " atla " },
        }));
        assert_eq!(padded["t"].cluster("phoenix"), "mltraining");
        assert_eq!(padded["t"].zone("atla"), "atla");
    }

            #[test]
    fn topic_entry_override_wins_per_field() {
        let topics = parse(json!({
            "abuse.candidates.impersonation_scam": {
                "processor": "score_result",
                "labels": ["impersonation_scam_threshold_reached"],
                "cluster": "mltraining",
                "zone": "atla",
            },
            "abuse.zone_only.v1": { "processor": "stats_only", "zone": "pdxa" },
        }));

        let scam = &topics["abuse.candidates.impersonation_scam"];
        assert_eq!(scam.cluster("phoenix"), "mltraining");
        assert_eq!(scam.zone("atla"), "atla");
        let TopicConfig::ScoreResult { ref labels, .. } = scam.processor else {
            panic!("expected score_result processor alongside the cluster override");
        };
        assert!(labels.contains("impersonation_scam_threshold_reached"));

        let zone_only = &topics["abuse.zone_only.v1"];
        assert_eq!(zone_only.cluster("phoenix"), "phoenix");
        assert_eq!(zone_only.zone("atla"), "pdxa");
    }

                            #[test]
    fn preflight_probes_first_default_target_topic_only() {
        let topics = [
            ("abuse.candidates.impersonation_scam", "mltraining", "atla"),
            ("abuse.v3.score_results", "phoenix", "atla"),
            ("abuse.embeddings.user_decisions", "phoenix", "atla"),
        ];

        assert_eq!(
            preflight_probe_topic(topics, ("phoenix", "atla")),
            Some("abuse.embeddings.user_decisions")
        );
    }

                    #[test]
    fn zone_only_override_is_not_the_default_target() {
        assert!(!is_default_target(("phoenix", "pdxa"), ("phoenix", "atla")));
        assert!(!is_default_target(
            ("mltraining", "atla"),
            ("phoenix", "atla")
        ));
        assert!(is_default_target(("phoenix", "atla"), ("phoenix", "atla")));

        let topics = [
            ("abuse.a.zone_override", "phoenix", "pdxa"),
            ("abuse.b.default", "phoenix", "atla"),
        ];
        assert_eq!(
            preflight_probe_topic(topics, ("phoenix", "atla")),
            Some("abuse.b.default"),
            "a zone-only override must not be chosen as the boot probe"
        );
    }

            #[test]
    fn preflight_skipped_when_no_topic_on_default_target() {
        let topics = [("abuse.candidates.impersonation_scam", "mltraining", "atla")];

        assert_eq!(preflight_probe_topic(topics, ("phoenix", "atla")), None);
        assert_eq!(preflight_probe_topic([], ("phoenix", "atla")), None);
    }

                        #[test]
    fn error_flood_signal_counts_default_target_topics_only() {
        let default_topic = "test.error.scope.default";
        let override_topic = "test.error.scope.override";
        let bump = |topic: &str, times: u64| {
            for _ in 0..times {
                xai_kafka::metrics::CONSUMER_ERROR_COUNT
                    .with_label_values(&[topic, "group", "BrokerTransportFailure"])
                    .inc();
            }
        };

        let watched: HashSet<String> = [default_topic.to_owned()].into_iter().collect();
        let before = consumer_error_total(&watched);

        bump(override_topic, 50);
        assert_eq!(
            consumer_error_total(&watched),
            before,
            "override-cluster topic errors must not move the eviction signal"
        );

        bump(default_topic, 3);
        assert_eq!(
            consumer_error_total(&watched),
            before + 3.0,
            "default-cluster topic errors must still be counted"
        );

        assert_eq!(consumer_error_total(&HashSet::new()), 0.0);
    }

                            #[test]
    fn override_cluster_and_zone_reach_the_kafka_bootstrap() {
        let certs = S2sCerts::new("/ca.crt", "/tls.crt", "/tls.key");

        let config = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
            "mltraining",
            "abuse.candidates.impersonation_scam",
            "xai-abuse-enforcement-service-fou-staging-abuse.candidates.impersonation_scam",
            certs.clone(),
            Some("atla"),
        )
        .expect("mltraining/atla is a registered mTLS cluster+zone")
        .build();

        assert_eq!(
            config.base_config.dest,
            "mltraining-mtls-bootstrap.kafka.prod.atla-prod-messaging.kafka.kube.atla.twitter.com:9095"
        );
        assert_eq!(
            config.base_config.topic,
            "abuse.candidates.impersonation_scam"
        );

        let default_cluster = KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
            "phoenix",
            "abuse.v3.score_results",
            "group",
            certs.clone(),
            Some("atla"),
        )
        .expect("phoenix/atla is a registered mTLS cluster+zone")
        .build();
        assert!(
            default_cluster
                .base_config
                .dest
                .starts_with("phoenix-mtls-bootstrap."),
            "unexpected default dest: {}",
            default_cluster.base_config.dest
        );
    }

                    #[test]
    fn unresolvable_override_is_an_error_not_a_panic() {
        let certs = S2sCerts::new("/ca.crt", "/tls.crt", "/tls.key");

        for (cluster, zone) in [
            ("not-a-cluster", "atla"),
            ("bluebird-1", "atla"),
            ("mltraining", "iad"),
        ] {
            assert!(
                KafkaConsumerConfigBuilder::for_cluster_mtls_zone(
                    cluster,
                    "some.topic",
                    "group",
                    certs.clone(),
                    Some(zone),
                )
                .is_err(),
                "expected {cluster}/{zone} to be rejected"
            );
        }
    }
}

#[cfg(test)]
mod dedup_retention_tests {
    use super::{DedupRetention, dedup_retention_for, outcome_holds_full_dedup};

                            #[test]
    fn hold_lookup_failed_releases_the_claim_hold_overturned_is_a_skip() {
        assert_eq!(
            dedup_retention_for(crate::overturn_hold::STATUS_HOLD_LOOKUP_FAILED),
            DedupRetention::Release
        );
        assert_eq!(
            dedup_retention_for(crate::overturn_hold::STATUS_HOLD_OVERTURNED),
            DedupRetention::Skip
        );
        assert_eq!(dedup_retention_for("success"), DedupRetention::Full);
        assert_eq!(dedup_retention_for("dry_run"), DedupRetention::Skip);
        assert_eq!(dedup_retention_for("dedup_skipped"), DedupRetention::Skip);
    }

                        #[test]
    fn only_success_holds_full_dedup_window() {
        assert!(outcome_holds_full_dedup("success"));
        for skip in [
            "dry_run",
            "dedup_skipped",
            "very_high_follower_count",
            "high_follower_count",
            "pagerank_skipped",
            "gizmoduck_skipped",
            "user_not_found",
            "test_user_skipped",
            "user_in_allowlist",
            "post_in_allowlist",
            "rule_eval_error",
            "max_enforcement_reached",
            "invalid_entity_id",
            "requested_actions_denied",
            "platform_row_without_requested_actions",
            crate::overturn_hold::STATUS_HOLD_OVERTURNED,
            crate::overturn_hold::STATUS_HOLD_LOOKUP_FAILED,
        ] {
            assert!(!outcome_holds_full_dedup(skip), "{skip} must not hold 24h");
        }
    }
}

#[cfg(test)]
mod hold_gate_trigger_tests {
    use super::*;
    use crate::decision::ActionSpec;
    use crate::facts::RequestedActionFacts;
    use crate::generic_actions::GenericActionAllowlist;
    use crate::overturn_hold::{GateKind, GateMode, GatedAction, HoldGateConfig};

    fn suspend() -> ActionSpec {
        ActionSpec::SuspendUser {
            perm: false,
            policy: "PlatformManipulation".into(),
        }
    }

    fn label() -> ActionSpec {
        ActionSpec::AddLabelsV2 {
            labels: vec!["SpamHighRecall".into()],
            ttl_msec: None,
        }
    }

    fn perm_suspend() -> ActionSpec {
        ActionSpec::SuspendUser {
            perm: true,
            policy: "Cse".into(),
        }
    }

            fn suspend_only_cfg() -> HoldGateConfig {
        HoldGateConfig {
            mode: GateMode::Enforce,
            ..HoldGateConfig::default()
        }
    }

        fn label_cfg(labels: &[&str]) -> HoldGateConfig {
        HoldGateConfig {
            mode: GateMode::Enforce,
            kinds: vec![GateKind::Suspend, GateKind::Label],
            labels: labels.iter().map(|s| (*s).to_owned()).collect(),
            ..HoldGateConfig::default()
        }
    }

    fn labels_spec(labels: &[&str]) -> ActionSpec {
        ActionSpec::AddLabelsV2 {
            labels: labels.iter().map(|s| (*s).to_owned()).collect(),
            ttl_msec: Some(86_400_000),
        }
    }

    fn gated_label(name: &str) -> GatedAction {
        GatedAction::Label { name: name.into() }
    }

                        #[test]
    fn suspend_detection_covers_plain_composite_and_never_labels() {
        let c = suspend_only_cfg();
        assert_eq!(
            gated_actions(&[suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[perm_suspend()], &c),
            vec![GatedAction::PERMANENT_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[label(), suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(
                &[
                    ActionSpec::AddPostLabelsV2 {
                        labels: vec!["Cse".into()],
                        ttl_msec: None,
                    },
                    perm_suspend(),
                ],
                &c
            ),
            vec![GatedAction::PERMANENT_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[suspend(), perm_suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[perm_suspend(), suspend()], &c),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        assert_eq!(
            gated_actions(&[perm_suspend(), label(), perm_suspend()], &c),
            vec![GatedAction::PERMANENT_SUSPEND],
            "all-perm set stays permanent"
        );
        assert!(gated_actions(&[label()], &c).is_empty());
        assert!(gated_actions(&[label(), ActionSpec::Captcha], &c).is_empty());
        assert!(gated_actions(&[ActionSpec::Arkose, ActionSpec::SpamLivenessCheck], &c).is_empty());
        assert!(gated_actions(&[], &c).is_empty());
        assert_eq!(HoldGateConfig::default().kinds, vec![GateKind::Suspend]);
        assert!(HoldGateConfig::default().labels.is_empty());
    }

                        #[test]
    fn label_detection_follows_kinds_and_labels() {
        let c = label_cfg(&["SpamHighRecall"]);
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamHighRecall"])], &c),
            vec![gated_label("SpamHighRecall")]
        );
        assert_eq!(
            gated_actions(
                &[
                    labels_spec(&["SpamMediumRecall", "SpamHighRecall"]),
                    ActionSpec::Captcha,
                ],
                &c
            ),
            vec![gated_label("SpamHighRecall")],
            "labels not in the config are ignored"
        );
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamHighRecall"]), perm_suspend()], &c),
            vec![
                GatedAction::PERMANENT_SUSPEND,
                gated_label("SpamHighRecall")
            ]
        );
        assert_eq!(
            gated_actions(
                &[
                    labels_spec(&["SpamHighRecall"]),
                    labels_spec(&["SpamHighRecall", "Other"]),
                ],
                &c
            ),
            vec![gated_label("SpamHighRecall")]
        );
        let c2 = label_cfg(&["SpamHighRecall", "SpamMediumRecall"]);
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamMediumRecall", "SpamHighRecall"])], &c2),
            vec![
                gated_label("SpamMediumRecall"),
                gated_label("SpamHighRecall")
            ]
        );
        assert!(
            gated_actions(
                &[ActionSpec::AddPostLabelsV2 {
                    labels: vec!["SpamHighRecall".into()],
                    ttl_msec: None,
                }],
                &c
            )
            .is_empty()
        );
        let inert = label_cfg(&[]);
        assert!(gated_actions(&[labels_spec(&["SpamHighRecall"])], &inert).is_empty());
        assert_eq!(
            gated_actions(&[labels_spec(&["SpamHighRecall"]), suspend()], &inert),
            vec![GatedAction::TEMPORARY_SUSPEND]
        );
        let no_label_kind = HoldGateConfig {
            labels: vec!["SpamHighRecall".into()],
            ..suspend_only_cfg()
        };
        assert!(gated_actions(&[labels_spec(&["SpamHighRecall"])], &no_label_kind).is_empty());
        let label_only_kind = HoldGateConfig {
            kinds: vec![GateKind::Label],
            ..label_cfg(&["SpamHighRecall"])
        };
        assert_eq!(
            gated_actions(
                &[labels_spec(&["SpamHighRecall"]), suspend()],
                &label_only_kind
            ),
            vec![gated_label("SpamHighRecall")]
        );
    }

                #[test]
    fn strip_held_labels_removes_only_the_held_names() {
        let held = vec!["SpamHighRecall".to_owned()];
        let mut specs = vec![labels_spec(&["SpamHighRecall"])];
        strip_held_labels(&mut specs, &held);
        assert!(specs.is_empty());
        let mut specs = vec![labels_spec(&["SpamHighRecall", "SpamMediumRecall"])];
        strip_held_labels(&mut specs, &held);
        assert_eq!(specs, vec![labels_spec(&["SpamMediumRecall"])]);
        let mut specs = vec![labels_spec(&["SpamHighRecall"]), suspend()];
        strip_held_labels(&mut specs, &held);
        assert_eq!(specs, vec![suspend()]);
        let post = ActionSpec::AddPostLabelsV2 {
            labels: vec!["SpamHighRecall".into()],
            ttl_msec: None,
        };
        let mut specs = vec![
            ActionSpec::Captcha,
            labels_spec(&["SpamHighRecall"]),
            post.clone(),
        ];
        strip_held_labels(&mut specs, &held);
        assert_eq!(
            specs,
            vec![ActionSpec::Captcha, post],
            "post labels are never touched"
        );
        let before = vec![labels_spec(&["SpamHighRecall"]), suspend()];
        let mut specs = before.clone();
        strip_held_labels(&mut specs, &[]);
        assert_eq!(specs, before);
        let mut specs = vec![labels_spec(&["SpamMediumRecall"])];
        strip_held_labels(&mut specs, &held);
        assert_eq!(specs, vec![labels_spec(&["SpamMediumRecall"])]);
    }

                    #[test]
    fn suspend_detection_covers_generic_suspend_and_suspend_author() {
        let user_allow = GenericActionAllowlist {
            kinds: ["suspend"].iter().map(|s| (*s).to_owned()).collect(),
            suspend_policies: ["PlatformManipulation"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            labels: Default::default(),
        };
        let mut facts = ScoreFacts::from_score(&abuse_proto::ScoreResult::default());
        facts.requested_actions = vec![RequestedActionFacts {
            kind: "suspend".into(),
            policy: "PlatformManipulation".into(),
            head: "IsSpammer".into(),
            ..Default::default()
        }];
        let (decision, _) =
            expand_requested_actions_decision(EntityType::User, &facts, &user_allow);
        let c = suspend_only_cfg();
        match decision {
            ExpandedDecision::Act(specs) => assert_eq!(
                gated_actions(&specs, &c),
                vec![GatedAction::TEMPORARY_SUSPEND]
            ),
            other => panic!("expected Act, got {other:?}"),
        }

        let post_allow = GenericActionAllowlist {
            kinds: ["post_label", "suspend_author"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            suspend_policies: ["Cse"].iter().map(|s| (*s).to_owned()).collect(),
            labels: ["Cse"].iter().map(|s| (*s).to_owned()).collect(),
        };
        let mut facts = ScoreFacts::from_score(&abuse_proto::ScoreResult::default());
        facts.requested_actions = vec![
            RequestedActionFacts {
                kind: "post_label".into(),
                labels: vec!["Cse".into()],
                head: "NearDupEmbeddingCseMatch".into(),
                ..Default::default()
            },
            RequestedActionFacts {
                kind: "suspend_author".into(),
                perm: true,
                policy: "Cse".into(),
                head: "NearDupEmbeddingCseMatch".into(),
                ..Default::default()
            },
        ];
        let (decision, _) =
            expand_requested_actions_decision(EntityType::Post, &facts, &post_allow);
        match decision {
            ExpandedDecision::Act(specs) => {
                assert_eq!(specs.len(), 2, "label + author suspend");
                assert_eq!(
                    gated_actions(&specs, &c),
                    vec![GatedAction::PERMANENT_SUSPEND],
                    "suspend_author perm:true → permanent shape"
                );
            }
            other => panic!("expected Act, got {other:?}"),
        }

        facts.requested_actions.truncate(1);
        let (decision, _) =
            expand_requested_actions_decision(EntityType::Post, &facts, &post_allow);
        match decision {
            ExpandedDecision::Act(specs) => {
                assert!(gated_actions(&specs, &c).is_empty());
                assert!(gated_actions(&specs, &label_cfg(&["Cse"])).is_empty());
            }
            other => panic!("expected Act, got {other:?}"),
        }
    }

                #[test]
    fn hold_skip_propagates_global_dry_run_and_audit_keys() {
        let score = abuse_proto::ScoreResult {
            user_id: 4242,
            model_version: "m@1".into(),
            ..Default::default()
        };
        let facts = Facts {
            entity_type: EntityType::User,
            entity_id: 4242,
            user_id: 4242,
            topic: "abuse.embeddings.user_decisions".into(),
            score: ScoreFacts::from_score(&score),
            entity: EntityFacts::User(UserFacts::default()),
            uas: None,
        };
        let mut gate_info = BTreeMap::new();
        gate_info.insert("overturn_hold_mode".to_owned(), "enforce".to_owned());
        gate_info.insert("overturn_hold_id".to_owned(), "77".to_owned());
        gate_info.insert("overturn_hold_head".to_owned(), "FollowBot".to_owned());
        gate_info.insert(
            "overturn_hold_expires_at".to_owned(),
            "2026-12-01T00:00:00+00:00".to_owned(),
        );
        gate_info.insert("overturn_hold_probe_ms".to_owned(), "2".to_owned());

        for global_dry_run in [true, false] {
            let out = hold_gate_skip_outcome(
                crate::overturn_hold::STATUS_HOLD_OVERTURNED,
                gate_info.clone(),
                global_dry_run,
                &facts,
                &score,
                Some(r#"[{"kind":"label"}]"#.to_owned()),
            );
            assert_eq!(out.status, "hold_overturned");
            assert_eq!(
                out.dry_run, global_dry_run,
                "hold skip must carry the service's global dry-run (G17)"
            );
            assert_eq!(out.source_topic, "abuse.embeddings.user_decisions");
            assert_eq!(out.entity_id, 4242);
            assert_eq!(out.info["overturn_hold_id"], "77");
            assert_eq!(out.info["overturn_hold_head"], "FollowBot");
            assert_eq!(out.info["overturn_hold_mode"], "enforce");
            assert_eq!(out.info["overturn_hold_probe_ms"], "2");
            assert!(out.info.contains_key("overturn_hold_expires_at"));
            assert_eq!(
                out.info["requested_actions_skipped"],
                r#"[{"kind":"label"}]"#
            );
            assert!(
                out.info.contains_key("cred_is_high"),
                "{:?}",
                out.info.keys()
            );
            assert!(!outcome_holds_full_dedup(&out.status));
            assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
        }
        let mut label_info = BTreeMap::new();
        label_info.insert("overturn_hold_mode".to_owned(), "enforce".to_owned());
        label_info.insert(
            "overturn_hold_labels_stripped".to_owned(),
            "SpamHighRecall,SpamMediumRecall".to_owned(),
        );
        label_info.insert("overturn_hold_label_hold_id".to_owned(), "91,92".to_owned());
        let mut gate = crate::overturn_hold::GateOutcome {
            skip_status: None,
            info: label_info,
            strip_labels: vec!["SpamHighRecall".into(), "SpamMediumRecall".into()],
            strip_hold_ids: vec![91, 92],
        };
        gate.mark_label_collapse();
        let out = hold_gate_skip_outcome(
            crate::overturn_hold::STATUS_HOLD_OVERTURNED,
            gate.info,
            false,
            &facts,
            &score,
            None,
        );
        assert_eq!(out.status, "hold_overturned");
        assert!(!out.dry_run);
        assert_eq!(out.info["overturn_hold_kind"], "label");
        assert_eq!(
            out.info["overturn_hold_id"], "91",
            "first stripped label's hold"
        );
        assert_eq!(out.info["overturn_hold_label"], "SpamHighRecall");
        assert_eq!(
            out.info["overturn_hold_labels_stripped"],
            "SpamHighRecall,SpamMediumRecall"
        );
        assert_eq!(out.info["overturn_hold_label_hold_id"], "91,92");
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
        let mut suspend_info = gate_info.clone();
        suspend_info.insert("overturn_hold_kind".to_owned(), "suspend".to_owned());
        let out = hold_gate_skip_outcome(
            crate::overturn_hold::STATUS_HOLD_OVERTURNED,
            suspend_info,
            false,
            &facts,
            &score,
            None,
        );
        assert_eq!(out.info["overturn_hold_kind"], "suspend");
        assert_eq!(out.info["overturn_hold_id"], "77");
        assert!(!out.info.contains_key("overturn_hold_label"));

        let out = hold_gate_skip_outcome(
            crate::overturn_hold::STATUS_HOLD_LOOKUP_FAILED,
            BTreeMap::new(),
            true,
            &facts,
            &score,
            None,
        );
        assert_eq!(out.status, "hold_lookup_failed");
        assert!(out.dry_run);
        assert!(!out.info.contains_key("requested_actions_skipped"));
        assert!(!outcome_holds_full_dedup(&out.status));
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Release);
    }
}

#[cfg(test)]
mod generic_dispatch_tests {
    use super::*;
    use crate::decision::ActionSpec;
    use crate::facts::RequestedActionFacts;
    use crate::generic_actions::GenericActionAllowlist;

    fn score_facts_with(requested: Vec<RequestedActionFacts>) -> ScoreFacts {
        let mut f = ScoreFacts::from_score(&abuse_proto::ScoreResult::default());
        f.requested_actions = requested;
        f
    }

    fn user_allowlist() -> GenericActionAllowlist {
        GenericActionAllowlist {
            kinds: ["suspend", "label"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            suspend_policies: ["PlatformManipulation"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            labels: ["SpamHighRecall"].iter().map(|s| (*s).to_owned()).collect(),
        }
    }

            #[test]
    fn expand_allowed_requested_actions_yields_plain_act_decision() {
        let facts = score_facts_with(vec![RequestedActionFacts {
            kind: "suspend".into(),
            perm: false,
            policy: "PlatformManipulation".into(),
            head: "IsSpammer".into(),
            ..Default::default()
        }]);
        let (decision, skipped) =
            expand_requested_actions_decision(EntityType::User, &facts, &user_allowlist());
        assert_eq!(
            decision,
            ExpandedDecision::Act(vec![ActionSpec::SuspendUser {
                perm: false,
                policy: "PlatformManipulation".into(),
            }])
        );
        assert!(skipped.is_none());
    }

    #[test]
    fn expand_denied_requested_actions_yields_skip_with_reason() {
        let facts = score_facts_with(vec![RequestedActionFacts {
            kind: "suspend".into(),
            policy: "PlatformManipulation".into(),
            head: "IsSpammer".into(),
            ..Default::default()
        }]);
        let (decision, skipped) = expand_requested_actions_decision(
            EntityType::User,
            &facts,
            &GenericActionAllowlist::default(),
        );
        assert_eq!(
            decision,
            ExpandedDecision::Skip(REQUESTED_ACTIONS_DENIED.into())
        );
        let skipped = skipped.expect("refused entries must be reported");
        assert!(skipped.contains("kind_not_allowlisted"), "{skipped}");
    }

    #[test]
    fn expand_empty_requested_actions_yields_skip() {
        let (decision, skipped) = expand_requested_actions_decision(
            EntityType::User,
            &score_facts_with(vec![]),
            &user_allowlist(),
        );
        assert_eq!(
            decision,
            ExpandedDecision::Skip(REQUESTED_ACTIONS_DENIED.into())
        );
        assert!(skipped.is_none());
    }

    #[test]
    fn expand_partial_allowlist_dispatches_allowed_and_reports_skipped() {
        let facts = score_facts_with(vec![
            RequestedActionFacts {
                kind: "label".into(),
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: 1000,
                head: "IsLabelHead".into(),
                ..Default::default()
            },
            RequestedActionFacts {
                kind: "bounce_captcha".into(),
                head: "IsCuspHead".into(),
                ..Default::default()
            },
        ]);
        let (decision, skipped) =
            expand_requested_actions_decision(EntityType::User, &facts, &user_allowlist());
        assert_eq!(
            decision,
            ExpandedDecision::Act(vec![ActionSpec::AddLabelsV2 {
                labels: vec!["SpamHighRecall".into()],
                ttl_msec: Some(1000),
            }])
        );
        let skipped = skipped.expect("refused entry must be reported");
        assert!(skipped.contains("bounce_captcha"), "{skipped}");
        assert!(skipped.contains("kind_not_allowlisted"), "{skipped}");
    }
}

#[cfg(test)]
mod decision_outcome_json_tests {
    use super::*;
    use xai_abuse_proto::enforcement::{
        EntityType as ProtoEntityType, FiredHead, ScoreResult, SummaryCounters,
    };

                            #[test]
    fn decision_outcome_json_mirror_carries_funnel_fields() {
        let score = ScoreResult {
            user_id: 100,
            model_version: "my_model@1".into(),
            entity_type: ProtoEntityType::Post as i32,
            entity_id: 555,
            summary: Some(SummaryCounters {
                labels: vec!["my_model_threshold_reached".into()],
                fired_heads: vec![FiredHead {
                    name: "IsSpamPost".into(),
                    score: 0.99,
                    threshold: 0.9,
                }],
                ..Default::default()
            }),
            score_id: "run-abc-7".into(),
            ..Default::default()
        };
        let mut info = BTreeMap::new();
        info.insert(
            "action_kinds".to_owned(),
            r#"["addPostLabelsV2"]"#.to_owned(),
        );
        let outcome = decision_outcome(&score, "some.topic", "success".into(), true, info);
        assert_eq!(outcome.score_id, "run-abc-7");
        let v = serde_json::to_value(&outcome).expect("DecisionOutcome serializes to JSON");

        assert!(v["decided_at_ms"].as_i64().unwrap() > 0);
        assert_eq!(v["source_topic"], "some.topic");
        assert_eq!(v["entity_type"], "post");
        assert_eq!(v["entity_id"], 555);
        assert_eq!(v["model_version"], "my_model@1");
        assert_eq!(v["status"], "success");
        assert_eq!(v["dry_run"], true);
        assert_eq!(v["head"], "IsSpamPost");
        assert_eq!(v["fired_heads"][0]["name"], "IsSpamPost");
        assert_eq!(v["labels"][0], "my_model_threshold_reached");
        assert_eq!(v["info"]["action_kinds"], r#"["addPostLabelsV2"]"#);
        assert_eq!(v["score_id"], "run-abc-7");

        let legacy = ScoreResult {
            score_id: String::new(),
            ..score
        };
        let outcome = decision_outcome(
            &legacy,
            "some.topic",
            "success".into(),
            true,
            BTreeMap::new(),
        );
        assert_eq!(outcome.score_id, "");
        let v = serde_json::to_value(&outcome).expect("DecisionOutcome serializes to JSON");
        assert_eq!(v["score_id"], "");
    }
}

#[cfg(test)]
mod health_tests {
    use super::health_decision;
    use std::time::Duration;

    const STALE: Duration = Duration::from_secs(120);
    const STALE_DELETE: Duration = Duration::from_secs(240);
    const ERR_DELETE: Duration = Duration::from_secs(60);
    const Z: Duration = Duration::ZERO;
    const S5: Duration = Duration::from_secs(5);

    #[test]
    fn fresh_consumer_is_ready() {
        let (ready, del) = health_decision(true, S5, Z, STALE, STALE_DELETE, ERR_DELETE, false);
        assert!(ready);
        assert!(!del);
    }

    #[test]
    fn stale_under_threshold_is_notready_not_deleted() {
        let s = Duration::from_secs(130);
        let (ready, del) = health_decision(true, s, Z, STALE, STALE_DELETE, ERR_DELETE, false);
        assert!(!ready);
        assert!(!del);
    }

    #[test]
    fn fully_dead_no_progress_self_deletes_at_stale_window() {
        let s = Duration::from_secs(241);
        let (ready, del) = health_decision(false, s, Z, STALE, STALE_DELETE, ERR_DELETE, false);
        assert!(!ready);
        assert!(del);
    }

    #[test]
    fn flapping_node_self_deletes_fast_on_error_window() {
        let (ready, del) = health_decision(
            true,
            S5,
            Duration::from_secs(61),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
        );
        assert!(!ready);
        assert!(del);
        let (ready, del) = health_decision(
            true,
            S5,
            Duration::from_secs(45),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
        );
        assert!(!ready);
        assert!(!del);
    }

    #[test]
    fn brief_error_spike_does_not_evict() {
        let (ready, del) = health_decision(
            true,
            S5,
            Duration::from_secs(10),
            STALE,
            STALE_DELETE,
            ERR_DELETE,
            true,
        );
        assert!(!ready);
        assert!(!del);
    }
}

#[cfg(test)]
mod already_suspended_gate_tests {
    use std::collections::HashMap;

    use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult, PCFLabel, Safety};
    use xai_core_entities::gizmoduck_client::{
        GizmoduckClient, MockGizmoduckClient, QueryFields, UserFields, ViewerData,
    };

    use super::*;
    use crate::decision::ActionSpec;
    use crate::facts::GizmoduckFacts;
    use crate::overturn_hold::{FakeHoldStore, HOLD_KIND_SUSPEND, Hold, HoldGate};

    const UID: i64 = 1_700_000_042;

    fn suspend(perm: bool) -> ActionSpec {
        ActionSpec::SuspendUser {
            perm,
            policy: "PlatformManipulation".into(),
        }
    }

    fn label() -> ActionSpec {
        ActionSpec::AddLabelsV2 {
            labels: vec!["SpamHighRecall".into()],
            ttl_msec: Some(86_400_000),
        }
    }

    fn gizmoduck(suspended: bool) -> GizmoduckFacts {
        GizmoduckFacts {
            present: true,
            suspended,
            ..Default::default()
        }
    }

    fn strip(specs: &[ActionSpec], user: &GizmoduckFacts) -> (Vec<ActionSpec>, Vec<ActionSpec>) {
        let mut kept = specs.to_vec();
        let stripped = strip_redundant_suspends(&mut kept, user);
        (kept, stripped)
    }

    #[test]
    fn temporary_suspend_of_a_suspended_account_is_removed() {
        let suspended = gizmoduck(true);
        assert_eq!(
            strip(&[suspend(false)], &suspended),
            (vec![], vec![suspend(false)])
        );
        assert_eq!(
            strip(&[suspend(false), suspend(false)], &suspended),
            (vec![], vec![suspend(false), suspend(false)])
        );
    }

    #[test]
    fn other_actions_of_the_decision_are_kept_in_order() {
        let suspended = gizmoduck(true);
        assert_eq!(
            strip(
                &[label(), suspend(false), ActionSpec::SpamLivenessCheck],
                &suspended
            ),
            (
                vec![label(), ActionSpec::SpamLivenessCheck],
                vec![suspend(false)]
            )
        );
    }

    #[test]
    fn active_account_keeps_every_action() {
        let active = gizmoduck(false);
        for specs in [
            vec![suspend(false)],
            vec![suspend(true)],
            vec![label(), suspend(false)],
        ] {
            assert_eq!(strip(&specs, &active), (specs.clone(), vec![]));
        }
    }

    #[test]
    fn a_permanent_suspend_keeps_the_decision_whole() {
        let suspended = gizmoduck(true);
        for specs in [
            vec![suspend(true)],
            vec![suspend(false), suspend(true)],
            vec![suspend(true), label(), suspend(false)],
        ] {
            assert_eq!(strip(&specs, &suspended), (specs.clone(), vec![]));
        }
    }

    #[test]
    fn decisions_without_a_suspend_are_untouched() {
        let suspended = gizmoduck(true);
        for specs in [
            vec![label()],
            vec![ActionSpec::Arkose, ActionSpec::Captcha],
            vec![],
        ] {
            assert_eq!(strip(&specs, &suspended), (specs.clone(), vec![]));
        }
    }


            const RULES_YAML: &str = r#"
for_entity: user
rules:
  - id: act_head_suspend
    when: 'score.model_version == "head_v1"'
    then: { kind: act_suspend_user, perm: false, policy: "PlatformManipulation" }
  - id: act_head_suspend_and_label
    when: 'score.model_version == "head_v2"'
    then:
      kind: act_all
      actions:
        - { kind: act_suspend_user, perm: false, policy: "PlatformManipulation" }
        - { kind: act_add_labels_v2, labels: ["SpamHighRecall"], ttl_msec: 86400000 }
  - id: already_suspended
    when: user.suspended
    then: { kind: skip, reason: already_suspended }
  - id: terminal
    when: "true"
    then: { kind: skip, reason: no_match }
"#;

        fn suspend_hold() -> Hold {
        Hold {
            user_id: UID,
            hold_id: 7,
            head: Some("FollowBot".into()),
            expires_at: chrono::Utc::now() + chrono::Duration::days(30),
            case_group_id: Some(63),
            reason: "appeal_overturned".into(),
            action_kind: HOLD_KIND_SUSPEND.into(),
            label: None,
        }
    }

    fn gizmoduck_user(suspended: bool) -> Arc<MockGizmoduckClient> {
        let user = GizmoduckUser {
            user_id: UID as u64,
            safety: Safety {
                suspended,
                ..Default::default()
            },
            ..Default::default()
        };
        Arc::new(MockGizmoduckClient {
            users: HashMap::from([(
                UID,
                Some(GizmoduckUserResult {
                    user: Some(user),
                    response_state: None,
                }),
            )]),
            ..Default::default()
        })
    }

        struct FailingGizmoduck;

    #[async_trait::async_trait]
    impl GizmoduckClient for FailingGizmoduck {
        async fn get_users(
            &self,
            user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>> {
            user_ids
                .into_iter()
                .map(|id| (id, Err(anyhow::anyhow!("gizmoduck unavailable"))))
                .collect()
        }
        async fn get_users_with_perspective(
            &self,
            _viewer_id: i64,
            user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>> {
            self.get_users(user_ids).await
        }
        async fn get_viewer_roles(&self, _user_id: u64) -> anyhow::Result<Vec<String>> {
            unimplemented!()
        }
        async fn get_viewer_data(&self, _user_id: u64) -> anyhow::Result<ViewerData> {
            unimplemented!()
        }
        async fn get_viewer_data_with_fields(
            &self,
            _user_id: u64,
            _query_fields: &[QueryFields],
        ) -> anyhow::Result<ViewerData> {
            unimplemented!()
        }
        async fn get_pcf_labels(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<PCFLabel>> {
            unimplemented!()
        }
        async fn get_profile_description_languages(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<String>>> {
            unimplemented!()
        }
        async fn get_user_fields(
            &self,
            _user_ids: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<UserFields>> {
            unimplemented!()
        }
        async fn get_by_screen_name(
            &self,
            _screen_name: &str,
        ) -> anyhow::Result<Option<GizmoduckUserResult>> {
            unimplemented!()
        }
    }

    struct Harness {
        ctx: EnforcementCtx,
        holds: Arc<FakeHoldStore>,
        gizmoduck_calls: Option<Arc<MockGizmoduckClient>>,
        _strato: wiremock::MockServer,
    }

                    async fn harness(gd: Arc<dyn GizmoduckClient + Send + Sync>, dry_run: bool) -> Harness {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({"v": null})),
            )
            .mount(&server)
            .await;
        #[allow(deprecated)]
        let column = |name: &str| {
            Arc::new(Strato::with_client(
                name,
                server.uri(),
                Strato::build_default_client(&Default::default()).unwrap(),
            ))
        };
        let dynamic_config = DynamicConfig::new(None, None, dry_run, 1_000, 1_000, None, None)
            .await
            .unwrap();
        dynamic_config.set_snapshot_for_test(crate::growthbook::ConfigSnapshot {
            config: Some(serde_json::json!({
                "dry_run": dry_run,
                "overturn_hold_gate": {"mode": "enforce"},
            })),
            rules_user: Some(Arc::from(RULES_YAML)),
            rules_post: None,
        });
        let holds = FakeHoldStore::new(vec![suspend_hold()]);
        let ctx = EnforcementCtx {
            ais_client: ais_client::AisClient::unconnected(),
            gd: gizmoduck::GizmoduckCoreClient::from_client(gd),
            cred_clients: strato::CredClients {
                high_page_rank: column("hpr"),
                grey_badge: column("grey"),
            },
            uas_strato: column("uas"),
            dynamic_config,
            rules_cache: Arc::new(rules::RulesCache::new()),
            allowlist: None,
            kafka_producer_decisions: None,
            kafka_producer_decisions_json: None,
            hold_gate: Arc::new(HoldGate::new(Some(holds.clone()))),
        };
        Harness {
            ctx,
            holds,
            gizmoduck_calls: None,
            _strato: server,
        }
    }

    async fn harness_for(suspended: bool, dry_run: bool) -> Harness {
        let gd = gizmoduck_user(suspended);
        let mut h = harness(gd.clone(), dry_run).await;
        h.gizmoduck_calls = Some(gd);
        h
    }

    fn score(model_version: &str) -> abuse_proto::ScoreResult {
        assert!(!test_user::is_test_user_id(UID));
        abuse_proto::ScoreResult {
            user_id: UID,
            model_version: model_version.into(),
            ..Default::default()
        }
    }

            #[tokio::test]
    async fn control_active_account_reaches_the_hold_gate() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(false, false).await;
        let out = run_enforcement_inner(&h.ctx, &score("head_v1"), "t")
            .await
            .unwrap();
        assert_eq!(out.status, overturn_hold::STATUS_HOLD_OVERTURNED);
        assert_eq!(h.holds.calls(), 1);
    }

                #[tokio::test]
    async fn suspended_account_is_skipped_before_the_hold_gate() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(true, false).await;
        let out = run_enforcement_inner(&h.ctx, &score("head_v1"), "t")
            .await
            .unwrap();
        assert_eq!(out.status, ALREADY_SUSPENDED);
        assert!(!out.dry_run);
        assert_eq!(out.info[ALREADY_SUSPENDED_STRIPPED], r#"["suspendUser"]"#);
        assert_eq!(out.info["gizmoduck_suspended"], "true");
        assert!(!out.info.contains_key("acted_class"));
        assert_eq!(h.holds.calls(), 0);
        assert_eq!(h.gizmoduck_calls.as_ref().unwrap().call_count(), 1);
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
    }

    #[tokio::test]
    async fn logging_only_mode_records_the_same_skip() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(true, true).await;
        let out = run_enforcement_inner(&h.ctx, &score("head_v1"), "t")
            .await
            .unwrap();
        assert_eq!(out.status, ALREADY_SUSPENDED);
        assert!(out.dry_run);
        assert_eq!(h.holds.calls(), 0);
        assert_eq!(dedup_retention_for(&out.status), DedupRetention::Skip);
    }

                    #[tokio::test]
    async fn suspend_plus_label_keeps_the_label() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness_for(true, true).await;
        let out = run_enforcement_inner(&h.ctx, &score("head_v2"), "t")
            .await
            .unwrap();
        assert_eq!(out.status, "dry_run");
        assert_eq!(out.info["action_kinds"], r#"["addLabelsV2"]"#);
        assert_eq!(out.info[ALREADY_SUSPENDED_STRIPPED], r#"["suspendUser"]"#);
        assert_eq!(h.holds.calls(), 0);
    }

            #[tokio::test]
    async fn gizmoduck_read_error_is_retried() {
        let _serial = overturn_hold::tests::METRICS_LOCK.lock().await;
        let h = harness(Arc::new(FailingGizmoduck), false).await;
        let err = run_enforcement_inner(&h.ctx, &score("head_v1"), "t")
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("gizmoduck"), "{err:#}");
        assert_eq!(h.holds.calls(), 0);
    }
}
