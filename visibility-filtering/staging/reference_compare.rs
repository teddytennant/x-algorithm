use crate::models::{Decided, Verdict, Withholding};
use crate::rules::SafetyLevel;
use crate::staging::reference::ENV_IMAGE;
use crate::treatment;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;
use xai_stats_receiver::global_stats_receiver;
use xai_twittercontext_proto::TwitterContextViewer;
use xai_visibility_filtering::models::{Action, FilteredReason};
use xai_visibility_filtering::vf_client::{SafetyLevel as ReferenceSafetyLevel, VfClient};

const COMPARED: &str = "vf_reference_compared";
const EXACT_MATCH: &str = "vf_reference_exact_match";
const DIFFERED: &str = "vf_reference_differed";
const ERROR: &str = "vf_reference_error";
const SKIPPED: &str = "vf_reference_skipped";
const ENABLED: &str = "vf_reference_enabled";

const HARNESS_LINE_MARKER: &str = "vf_reference_compare";
const SCHEMA_VERSION: u32 = 1;
const LINE_BUDGET_BYTES: usize = 12 * 1024;

const REFERENCE_TIMEOUT: Duration = Duration::from_millis(1500);

fn service_triple(
    verdict: &Verdict,
) -> (&'static str, Option<&FilteredReason>, Option<&'static str>) {
    let reason = match verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(reason),
            ..
        }) => Some(reason.legacy()),
        Verdict::Shown {
            notice: _,
            media: Some(Decided { value, .. }),
            engagement: None | Some(_),
        } => Some(value.legacy()),
        Verdict::Withheld(Decided {
            value: Withholding::Tombstone(_),
            ..
        })
        | Verdict::Shown {
            notice: _,
            media: None,
            engagement: None | Some(_),
        } => None,
    };
    let rule = treatment::decided_rows(verdict)
        .next()
        .map(|(rule, _)| rule);
    (treatment::metric_label(verdict), reason, rule)
}

fn reference_action_label(reason: Option<&FilteredReason>) -> &'static str {
    match reason {
        None => "allow",
        Some(FilteredReason::SafetyResult(safety_result)) => match safety_result.action {
            Action::NotEvaluated => "not_evaluated",
            Action::Allow => "allow",
            Action::Drop(_) => "drop",
            Action::Interstitial => "interstitial",
            Action::Downrank => "downrank",
            Action::Tombstone => "tombstone",
            Action::Avoid => "avoid",
        },
        Some(_) => "drop",
    }
}

fn reason_token(reason: &FilteredReason) -> String {
    match reason {
        FilteredReason::SafetyResult(safety_result) => match &safety_result.reason {
            Some(inner) => format!("{inner:?}"),
            None => "SafetyResult".to_string(),
        },
        FilteredReason::TweetMatchesViewerMutedKeyword(_) => {
            "TweetMatchesViewerMutedKeyword".to_string()
        }
        other @ (FilteredReason::ContainNsfwMedia
        | FilteredReason::AuthorBlockViewer
        | FilteredReason::PossiblyUndesirable
        | FilteredReason::UnspecifiedReason
        | FilteredReason::AuthorAccountIsInactive
        | FilteredReason::AuthorIsProtected
        | FilteredReason::AuthorIsUnsafe
        | FilteredReason::ReportedTweet
        | FilteredReason::TweetIsBounced
        | FilteredReason::AuthorIsDeactivated
        | FilteredReason::AuthorIsSuspended
        | FilteredReason::ViewerMutesAuthor
        | FilteredReason::TweetIsNullcast
        | FilteredReason::ExclusiveTweet
        | FilteredReason::ViewerBlocksAuthor) => format!("{other:?}"),
    }
}

fn service_verdict_str(verdict: &Verdict) -> String {
    let (action, reason, rule) = service_triple(verdict);
    let mut out = action.to_string();
    if let Some(reason) = reason {
        out.push(':');
        out.push_str(&reason_token(reason));
    }
    if let Some(rule) = rule {
        out.push('@');
        out.push_str(rule);
    }
    out
}

fn reference_verdict_str(reference: Option<&FilteredReason>) -> String {
    match reference {
        None => "allow".to_string(),
        Some(reason) => format!(
            "{}:{}",
            reference_action_label(reference),
            reason_token(reason)
        ),
    }
}

pub(crate) fn is_exact_match(service: &Verdict, reference: Option<&FilteredReason>) -> bool {
    let (service_action, service_reason, _) = service_triple(service);
    service_action == reference_action_label(reference) && service_reason == reference
}

pub(crate) struct TweetVerdict {
    pub tweet_id: u64,
    pub verdict: Verdict,
}

pub(crate) struct VerdictSender {
    sender: tokio::sync::oneshot::Sender<Vec<TweetVerdict>>,
    #[cfg_attr(not(test), expect(dead_code, reason = "awaited only by tests"))]
    task: tokio::task::JoinHandle<CompareResult>,
}

#[derive(Debug, PartialEq, Eq)]
enum CompareResult {
    Compared,
    ReferenceTimeout,
    VerdictsDropped,
}

impl VerdictSender {
    #[expect(
        clippy::let_underscore_must_use,
        reason = "the compare task holds its receiver until it reads the verdicts, so a failed send means the task already ended and only the comparison is lost"
    )]
    pub(crate) fn send(self, verdicts: Vec<TweetVerdict>) {
        let _ = self.sender.send(verdicts);
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct CompareCounts {
    pub compared: u64,
    pub exact_match: u64,
    pub differed: u64,
    pub errors: HashMap<&'static str, u64>,
}

pub(crate) struct CompareContext<'a> {
    pub viewer_id: u64,
    pub safety_level: SafetyLevel,
    pub dc: &'a str,
    pub build_sha: &'a str,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Diff {
    pub tweet_id: u64,
    pub service: String,
    pub reference: String,
}

pub(crate) fn compare_batch(
    verdicts: &[TweetVerdict],
    reference_results: &HashMap<u64, anyhow::Result<Option<FilteredReason>>>,
) -> (CompareCounts, Vec<Diff>) {
    let mut counts = CompareCounts::default();
    let mut diffs = Vec::new();
    for TweetVerdict { tweet_id, verdict } in verdicts {
        let reference = match reference_results.get(tweet_id) {
            None => {
                *counts.errors.entry("missing_result").or_default() += 1;
                continue;
            }
            Some(Err(_)) => {
                *counts.errors.entry("reference_item").or_default() += 1;
                continue;
            }
            Some(Ok(reason)) => reason,
        };
        counts.compared += 1;
        if is_exact_match(verdict, reference.as_ref()) {
            counts.exact_match += 1;
        } else {
            counts.differed += 1;
            diffs.push(Diff {
                tweet_id: *tweet_id,
                service: service_verdict_str(verdict),
                reference: reference_verdict_str(reference.as_ref()),
            });
        }
    }
    (counts, diffs)
}

fn batch_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}-{:x}", SEQ.fetch_add(1, Ordering::Relaxed))
}

struct Group<'a> {
    service: &'a str,
    reference: &'a str,
    tweet_ids: Vec<u64>,
}

fn group_diffs(diffs: &[Diff]) -> Vec<Group<'_>> {
    let mut index: HashMap<(&str, &str), usize> = HashMap::new();
    let mut groups: Vec<Group<'_>> = Vec::new();
    for diff in diffs {
        let at = *index
            .entry((diff.service.as_str(), diff.reference.as_str()))
            .or_insert_with(|| {
                groups.push(Group {
                    service: &diff.service,
                    reference: &diff.reference,
                    tweet_ids: Vec::new(),
                });
                groups.len() - 1
            });
        #[expect(clippy::indexing_slicing, reason = "at indexes a group already pushed")]
        groups[at].tweet_ids.push(diff.tweet_id);
    }
    groups
}

fn group_slices(group: &Group<'_>, budget: usize) -> Vec<serde_json::Value> {
    let whole = serde_json::json!([group.service, group.reference, group.tweet_ids]);
    if whole.to_string().len() < budget {
        return vec![whole];
    }
    let fixed = serde_json::json!([group.service, group.reference, []])
        .to_string()
        .len()
        + 1;
    let ids_per_slice = budget.saturating_sub(fixed).div_euclid(21).max(1);
    group
        .tweet_ids
        .chunks(ids_per_slice)
        .map(|ids| serde_json::json!([group.service, group.reference, ids]))
        .collect()
}

fn line_json(
    context: &CompareContext<'_>,
    batch: &str,
    chunk: [usize; 2],
    diffs: &[serde_json::Value],
) -> serde_json::Value {
    serde_json::json!({
        "h": HARNESS_LINE_MARKER,
        "v": SCHEMA_VERSION,
        "batch": batch,
        "chunk": chunk,
        "build": context.build_sha,
        "dc": context.dc,
        "level": <&str>::from(context.safety_level),
        "viewer": context.viewer_id,
        "diffs": diffs,
    })
}

pub(crate) fn chunk_lines(
    context: &CompareContext<'_>,
    batch: &str,
    diffs: &[Diff],
) -> Vec<serde_json::Value> {
    if diffs.is_empty() {
        return Vec::new();
    }
    let header_len = line_json(context, batch, [1, 1], &[]).to_string().len();
    let budget = LINE_BUDGET_BYTES.saturating_sub(header_len);
    let mut pages: Vec<Vec<serde_json::Value>> = Vec::new();
    let mut current: Vec<serde_json::Value> = Vec::new();
    let mut used = 0;
    for group in group_diffs(diffs) {
        for slice in group_slices(&group, budget) {
            let cost = slice.to_string().len() + 1;
            if used + cost > budget && !current.is_empty() {
                pages.push(std::mem::take(&mut current));
                used = 0;
            }
            used += cost;
            current.push(slice);
        }
    }
    pages.push(current);
    let total = pages.len();
    pages
        .into_iter()
        .enumerate()
        .map(|(index, page)| line_json(context, batch, [index + 1, total], &page))
        .collect()
}

pub(crate) fn comparable_request(
    safety_level: SafetyLevel,
    viewer_id: Option<u64>,
) -> Result<(ReferenceSafetyLevel, u64), &'static str> {
    let level = match safety_level {
        SafetyLevel::TimelineHome => ReferenceSafetyLevel::TimelineHome,
        SafetyLevel::TimelineHomeRecommendations => {
            ReferenceSafetyLevel::TimelineHomeRecommendations
        }
        SafetyLevel::ImmersiveExpandedRecommendations => {
            ReferenceSafetyLevel::ImmersiveExpandedRecommendations
        }
        SafetyLevel::FilterAll | SafetyLevel::TimelineHomeHydration => {
            return Err("level_unmapped");
        }
    };
    match viewer_id {
        Some(viewer_id) => Ok((level, viewer_id)),
        None => Err("logged_out_viewer"),
    }
}

const BUILD_SHA_LEN: usize = 12;

pub(crate) fn resolve_build_sha(compiled: &str, image: Option<&str>) -> String {
    if let Some(sha) = sha_prefix(compiled) {
        return sha.to_owned();
    }
    if let Some(image) = image
        && let Some(tag) = image.rsplit(':').next()
        && let Some(sha) = sha_prefix(tag)
    {
        return sha.to_owned();
    }
    let mut fallback = compiled.to_owned();
    fallback.truncate(BUILD_SHA_LEN);
    fallback
}

fn sha_prefix(s: &str) -> Option<&str> {
    s.get(..BUILD_SHA_LEN)
        .filter(|prefix| prefix.bytes().all(|b| b.is_ascii_hexdigit()))
}

pub struct ReferenceCompareHarness {
    reference: Arc<dyn VfClient + Send + Sync>,
    dc: String,
    build_sha: String,
}

impl ReferenceCompareHarness {
    pub(crate) fn new(reference: Arc<dyn VfClient + Send + Sync>, datacenter: &str) -> Self {
        let compiled = xai_build_version::current_build_information().git_commit_sha;
        let image = std::env::var(ENV_IMAGE).ok();
        let build_sha = resolve_build_sha(&compiled, image.as_deref());
        let harness = Self {
            reference,
            dc: datacenter.to_string(),
            build_sha,
        };
        info!(
            build_sha = %harness.build_sha,
            "reference_compare: harness enabled"
        );
        harness.incr(ENABLED, &[]);
        harness
    }

    pub(crate) fn begin_compare(
        self: &Arc<Self>,
        viewer_id: Option<u64>,
        country_code: Option<String>,
        safety_level: SafetyLevel,
        tweet_ids: Vec<u64>,
    ) -> Option<VerdictSender> {
        let (reference_level, viewer_id) = match comparable_request(safety_level, viewer_id) {
            Ok(comparable) => comparable,
            Err(reason) => {
                self.incr(SKIPPED, &[("reason", reason)]);
                return None;
            }
        };
        if tweet_ids.is_empty() {
            return None;
        }
        let (tx, rx) = tokio::sync::oneshot::channel::<Vec<TweetVerdict>>();
        let harness = Arc::clone(self);
        let task = tokio::spawn(async move {
            let viewer = TwitterContextViewer {
                user_id: viewer_id.cast_signed(),
                request_country_code: country_code.unwrap_or_default(),
                ..Default::default()
            };
            let reference_fut =
                harness
                    .reference
                    .get_result(tweet_ids, reference_level, viewer_id, Some(viewer));
            let (reference_outcome, verdicts) =
                futures::future::join(tokio::time::timeout(REFERENCE_TIMEOUT, reference_fut), rx)
                    .await;
            let reference_results: HashMap<u64, anyhow::Result<Option<FilteredReason>>> =
                match reference_outcome {
                    Ok(results) => results
                        .into_iter()
                        .map(|(id, r)| (id, r.map(|t| t.reason)))
                        .collect(),
                    Err(_) => {
                        harness.incr(ERROR, &[("kind", "timeout")]);
                        return CompareResult::ReferenceTimeout;
                    }
                };
            let Ok(verdicts) = verdicts else {
                return CompareResult::VerdictsDropped;
            };
            let context = CompareContext {
                viewer_id,
                safety_level,
                dc: &harness.dc,
                build_sha: &harness.build_sha,
            };
            let (counts, diffs) = compare_batch(&verdicts, &reference_results);
            harness.emit(safety_level, &counts);
            if !diffs.is_empty() {
                for line in chunk_lines(&context, &batch_id(), &diffs) {
                    #[expect(clippy::print_stdout, reason = "stdout is the diff sink")]
                    {
                        println!("{line}");
                    }
                }
            }
            CompareResult::Compared
        });
        Some(VerdictSender { sender: tx, task })
    }

    fn emit(&self, safety_level: SafetyLevel, counts: &CompareCounts) {
        let level = <&str>::from(safety_level);
        self.incr_nonzero(COMPARED, &[("safety_level", level)], counts.compared);
        self.incr_nonzero(EXACT_MATCH, &[("safety_level", level)], counts.exact_match);
        self.incr_nonzero(DIFFERED, &[("safety_level", level)], counts.differed);
        for (kind, count) in &counts.errors {
            self.incr_nonzero(ERROR, &[("kind", kind)], *count);
        }
    }

    fn incr(&self, metric: &str, labels: &[(&str, &str)]) {
        self.incr_nonzero(metric, labels, 1);
    }

    fn incr_nonzero(&self, metric: &str, labels: &[(&str, &str)], count: u64) {
        if count == 0 {
            return;
        }
        if let Some(sr) = global_stats_receiver() {
            let mut stamped: Vec<(&str, &str)> = labels.to_vec();
            stamped.push(("build_sha", &self.build_sha));
            sr.incr(metric, &stamped, count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MediaInterstitial, MediaRestriction};
    use xai_visibility_filtering::models::{
        Action, DropReason, KeywordMatch, SafetyResult as ReferenceSafetyResult,
    };
    use xai_visibility_filtering::tweet_safety_label::SafetyLabelFailure;
    use xai_visibility_filtering::vf_client::TweetVisibility;
    use xai_x_thrift::action::InterstitialReason;

    fn reference_allow() -> Option<FilteredReason> {
        None
    }

    fn reference_bare_drop() -> Option<FilteredReason> {
        Some(FilteredReason::AuthorIsSuspended)
    }

    fn reference_safety_result(action: Action) -> Option<FilteredReason> {
        Some(FilteredReason::SafetyResult(ReferenceSafetyResult {
            reason: None,
            action,
        }))
    }

    fn reference_muted_keyword() -> Option<FilteredReason> {
        Some(FilteredReason::TweetMatchesViewerMutedKeyword(
            KeywordMatch {
                keyword: "spoilers".to_string(),
            },
        ))
    }

    fn service_allow() -> Verdict {
        Verdict::Shown {
            notice: None,
            media: None,
            engagement: None,
        }
    }

    fn service_drop_of(reason: FilteredReason) -> Verdict {
        Verdict::Withheld(Decided {
            value: Withholding::Drop(crate::models::DropReason::Legacy(reason)),
            by: "suspended_author/drop",
        })
    }

    fn service_drop() -> Verdict {
        service_drop_of(FilteredReason::AuthorIsSuspended)
    }

    fn service_interstitial() -> Verdict {
        Verdict::Shown {
            notice: None,
            media: Some(Decided {
                value: MediaRestriction::MediaInterstitial(MediaInterstitial {
                    legacy: FilteredReason::ContainNsfwMedia,
                    reason: InterstitialReason::Sensitive(true),
                    prompt: None,
                }),
                by: "nsfw_media",
            }),
            engagement: None,
        }
    }

    #[test]
    fn strict_equality_no_normalization() {
        assert!(is_exact_match(&service_allow(), reference_allow().as_ref()));
        assert!(is_exact_match(
            &service_drop(),
            reference_bare_drop().as_ref()
        ));
        assert!(!is_exact_match(
            &service_drop_of(FilteredReason::AuthorIsUnsafe),
            reference_bare_drop().as_ref()
        ));
        assert!(!is_exact_match(
            &service_interstitial(),
            reference_allow().as_ref()
        ));
        assert!(!is_exact_match(
            &service_allow(),
            reference_safety_result(Action::Avoid).as_ref()
        ));
        assert!(!is_exact_match(
            &service_drop(),
            reference_safety_result(Action::Drop(DropReason {})).as_ref()
        ));
        assert!(!is_exact_match(
            &service_allow(),
            reference_muted_keyword().as_ref()
        ));
        assert!(!is_exact_match(
            &service_drop(),
            reference_muted_keyword().as_ref()
        ));
    }

    fn verdict(tweet_id: u64, verdict: Verdict) -> TweetVerdict {
        TweetVerdict { tweet_id, verdict }
    }

    fn context() -> CompareContext<'static> {
        CompareContext {
            viewer_id: 99,
            safety_level: SafetyLevel::TimelineHomeRecommendations,
            dc: "atla",
            build_sha: "abc123def456",
        }
    }

    #[test]
    fn compare_batch_counts_policy_free_and_collects_differing_pairs_only() {
        let verdicts = vec![
            verdict(1, service_allow()),
            verdict(2, service_drop()),
            verdict(3, service_allow()),
            verdict(4, service_allow()),
            verdict(5, service_allow()),
        ];
        let reference_results: HashMap<u64, anyhow::Result<Option<FilteredReason>>> =
            HashMap::from([
                (1, Ok(reference_allow())),
                (2, Ok(reference_allow())),
                (3, Ok(reference_bare_drop())),
                (4, Err(anyhow::anyhow!("reference error"))),
            ]);

        let (counts, diffs) = compare_batch(&verdicts, &reference_results);

        assert_eq!(counts.compared, 3);
        assert_eq!(counts.exact_match, 1);
        assert_eq!(counts.differed, 2);
        assert_eq!(
            counts.errors,
            HashMap::from([("reference_item", 1), ("missing_result", 1)])
        );
        assert_eq!(diffs.len(), 2, "differing pairs only: {diffs:?}");
    }

    const RECORDER_LINE_FIXTURE: &str = "scripts/tests/fixtures/recorder_lines.json";

    #[test]
    fn recorder_line_fixture_matches_chunk_lines() {
        use xai_visibility_filtering::models::SafetyResultReason;
        let avoid_nsfw = Some(FilteredReason::SafetyResult(ReferenceSafetyResult {
            reason: Some(SafetyResultReason::NsfwHighPrecision),
            action: Action::Avoid,
        }));
        let pairs = [
            (service_allow(), avoid_nsfw.clone()),
            (service_drop(), reference_allow()),
            (service_interstitial(), reference_allow()),
            (service_allow(), reference_bare_drop()),
            (service_allow(), reference_muted_keyword()),
            (service_allow(), avoid_nsfw),
            (
                service_allow(),
                reference_safety_result(Action::NotEvaluated),
            ),
            (service_allow(), reference_safety_result(Action::Allow)),
            (
                service_allow(),
                reference_safety_result(Action::Drop(DropReason {})),
            ),
            (
                service_allow(),
                reference_safety_result(Action::Interstitial),
            ),
            (service_allow(), reference_safety_result(Action::Downrank)),
            (service_allow(), reference_safety_result(Action::Tombstone)),
            (service_allow(), reference_safety_result(Action::Avoid)),
        ];
        let diffs: Vec<Diff> = pairs
            .iter()
            .enumerate()
            .map(|(i, (service, reference))| Diff {
                tweet_id: ID + i as u64,
                service: service_verdict_str(service),
                reference: reference_verdict_str(reference.as_ref()),
            })
            .collect();
        let emitted = serde_json::Value::Array(chunk_lines(&context(), "b1", &diffs));

        let cargo = format!("{}/{RECORDER_LINE_FIXTURE}", env!("CARGO_MANIFEST_DIR"));
        let ws =
            format!("crates/x-product/xai-visibility-filtering-service/{RECORDER_LINE_FIXTURE}");
        if std::env::var_os("VF_WRITE_FIXTURES").is_some() {
            std::fs::write(
                &cargo,
                serde_json::to_string_pretty(&emitted).unwrap() + "\n",
            )
            .unwrap();
        }
        let path = if std::path::Path::new(&cargo).exists() {
            cargo
        } else {
            ws
        };
        let on_disk: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}")),
        )
        .unwrap();
        assert_eq!(
            on_disk, emitted,
            "{RECORDER_LINE_FIXTURE} differs from chunk_lines"
        );
    }

    #[test]
    fn muted_keyword_payload_never_reaches_the_line() {
        let encoded = reference_verdict_str(reference_muted_keyword().as_ref());
        assert_eq!(encoded, "drop:TweetMatchesViewerMutedKeyword");
        assert!(!encoded.contains("spoilers"), "viewer content leaked");
    }

    fn diff(tweet_id: u64, service: &str, reference: &str) -> Diff {
        Diff {
            tweet_id,
            service: service.to_string(),
            reference: reference.to_string(),
        }
    }

    const ID: u64 = 1_000_000_000_000_000_000;

    #[test]
    fn identical_pairs_group_into_one_diffs_entry() {
        let diffs = vec![
            diff(1, "allow", "avoid:SafetyResult"),
            diff(2, "drop:ContainNsfwMedia@nsfw_media", "allow"),
            diff(3, "allow", "avoid:SafetyResult"),
            diff(4, "allow", "avoid:SafetyResult"),
        ];

        let lines = chunk_lines(&context(), "b1", &diffs);

        assert_eq!(lines.len(), 1, "a full request fits one line");
        assert_eq!(lines[0]["chunk"], serde_json::json!([1, 1]));
        assert_eq!(
            lines[0]["diffs"],
            serde_json::json!([
                ["allow", "avoid:SafetyResult", [1, 3, 4]],
                ["drop:ContainNsfwMedia@nsfw_media", "allow", [2]],
            ])
        );
    }

    #[test]
    fn lines_split_at_the_byte_budget_and_stay_self_contained() {
        let diffs: Vec<Diff> = (0..300)
            .map(|i| {
                diff(
                    ID + i,
                    &format!("interstitial:ContainNsfwMedia@nsfw_user/blur/sensitive_user{i}"),
                    "avoid:SafetyResult",
                )
            })
            .collect();

        let lines = chunk_lines(&context(), "b3", &diffs);

        assert!(lines.len() > 1, "300 distinct pairs exceed one budget");
        let total = lines.len();
        let mut seen = 0;
        for (i, line) in lines.iter().enumerate() {
            assert!(line.to_string().len() <= LINE_BUDGET_BYTES);
            assert_eq!(line["h"], "vf_reference_compare");
            assert_eq!(line["v"], 1);
            assert_eq!(line["batch"], "b3");
            assert_eq!(line["chunk"], serde_json::json!([i + 1, total]));
            assert_eq!(line["build"], "abc123def456");
            assert_eq!(line["dc"], "atla");
            assert_eq!(line["level"], "timeline_home_recommendations");
            assert_eq!(line["viewer"], 99);
            seen += line["diffs"].as_array().unwrap().len();
        }
        assert_eq!(seen, 300, "splitting is lossless");
    }

    #[test]
    fn oversized_single_group_splits_its_id_list() {
        let diffs: Vec<Diff> = (0..1000)
            .map(|i| diff(ID + i, "allow", "avoid:SafetyResult"))
            .collect();

        let lines = chunk_lines(&context(), "b4", &diffs);

        assert!(lines.len() > 1, "1000 ids exceed one budget");
        let ids: Vec<u64> = lines
            .iter()
            .flat_map(|line| line["diffs"].as_array().unwrap().iter())
            .flat_map(|group| group[2].as_array().unwrap().iter())
            .map(|id| id.as_u64().unwrap())
            .collect();
        assert_eq!(ids.len(), 1000, "splitting is lossless");
        assert_eq!(ids[0], ID);
        assert_eq!(ids[999], ID + 999);
        for line in &lines {
            assert!(line.to_string().len() <= LINE_BUDGET_BYTES);
            assert_eq!(line["diffs"][0][0], "allow");
            assert_eq!(line["diffs"][0][1], "avoid:SafetyResult");
        }
    }

    #[test]
    fn group_slices_splits_at_the_budget_and_keeps_id_order() {
        for (id_count, budget, sizes) in [
            (0, 100, &[0][..]),
            (5, 112, &[5][..]),
            (5, 111, &[4, 1][..]),
            (5, 110, &[4, 1][..]),
            (5, 96, &[3, 2][..]),
        ] {
            let group = Group {
                service: "a",
                reference: "b",
                tweet_ids: (0..id_count).map(|i| ID + i).collect(),
            };

            let slices = group_slices(&group, budget);

            let slice_sizes: Vec<usize> = slices
                .iter()
                .map(|slice| slice[2].as_array().unwrap().len())
                .collect();
            assert_eq!(slice_sizes, sizes, "{id_count} ids, budget {budget}");
            let sliced_ids: Vec<u64> = slices
                .iter()
                .flat_map(|slice| slice[2].as_array().unwrap().iter())
                .map(|id| id.as_u64().unwrap())
                .collect();
            assert_eq!(
                sliced_ids, group.tweet_ids,
                "{id_count} ids, budget {budget}"
            );
        }
    }

    #[test]
    fn chunk_lines_fills_a_page_to_the_exact_byte_budget() {
        let header_len = line_json(&context(), "b5", [1, 1], &[]).to_string().len();
        let budget = LINE_BUDGET_BYTES - header_len;
        let slice_cost = |service: &str, id: u64| {
            serde_json::json!([service, "avoid:SafetyResult", [id]])
                .to_string()
                .len()
                + 1
        };
        let pad = budget - slice_cost("allow", ID) - slice_cost("", ID + 1);

        for (extra, lines) in [(0, 1), (1, 2)] {
            let diffs = vec![
                diff(ID, "allow", "avoid:SafetyResult"),
                diff(ID + 1, &"x".repeat(pad + extra), "avoid:SafetyResult"),
            ];

            assert_eq!(chunk_lines(&context(), "b5", &diffs).len(), lines);
        }
    }

    type RecordedCall = (Vec<u64>, ReferenceSafetyLevel, u64, Option<String>);

    enum FakeReply {
        Immediate,
        Hangs,
    }

    struct FakeReference {
        calls: std::sync::Mutex<Vec<RecordedCall>>,
        called: tokio::sync::Notify,
        reply: FakeReply,
    }

    #[tonic::async_trait]
    impl VfClient for FakeReference {
        async fn get_result(
            &self,
            tweet_ids: Vec<u64>,
            safety_level: ReferenceSafetyLevel,
            for_user_id: u64,
            context: Option<TwitterContextViewer>,
        ) -> HashMap<u64, anyhow::Result<TweetVisibility>> {
            self.calls.lock().unwrap().push((
                tweet_ids.clone(),
                safety_level,
                for_user_id,
                context.map(|c| c.request_country_code),
            ));
            self.called.notify_one();
            if matches!(self.reply, FakeReply::Hangs) {
                std::future::pending::<()>().await;
            }
            tweet_ids
                .into_iter()
                .map(|id| {
                    (
                        id,
                        Ok(TweetVisibility {
                            action: Action::Allow,
                            reason: None,
                            safety_labels: Err(SafetyLabelFailure::LookupFailed),
                        }),
                    )
                })
                .collect()
        }
    }

    fn fake_harness(reply: FakeReply) -> (Arc<ReferenceCompareHarness>, Arc<FakeReference>) {
        let fake = Arc::new(FakeReference {
            calls: std::sync::Mutex::new(Vec::new()),
            called: tokio::sync::Notify::new(),
            reply,
        });
        (
            Arc::new(ReferenceCompareHarness::new(
                Arc::<FakeReference>::clone(&fake),
                "atla",
            )),
            fake,
        )
    }

    #[tokio::test]
    async fn begin_compare_skips_unmappable_requests_without_calling_reference() {
        let (harness, fake) = fake_harness(FakeReply::Immediate);

        for (level, viewer, expected) in [
            (
                SafetyLevel::TimelineHome,
                Some(7),
                Ok((ReferenceSafetyLevel::TimelineHome, 7)),
            ),
            (
                SafetyLevel::TimelineHomeRecommendations,
                Some(7),
                Ok((ReferenceSafetyLevel::TimelineHomeRecommendations, 7)),
            ),
            (
                SafetyLevel::ImmersiveExpandedRecommendations,
                Some(7),
                Ok((ReferenceSafetyLevel::ImmersiveExpandedRecommendations, 7)),
            ),
            (SafetyLevel::FilterAll, Some(7), Err("level_unmapped")),
            (
                SafetyLevel::TimelineHomeHydration,
                Some(7),
                Err("level_unmapped"),
            ),
            (SafetyLevel::TimelineHome, None, Err("logged_out_viewer")),
        ] {
            assert_eq!(comparable_request(level, viewer), expected);
        }
        assert!(
            harness
                .begin_compare(Some(7), None, SafetyLevel::FilterAll, vec![1])
                .is_none()
        );
        assert!(
            harness
                .begin_compare(Some(7), None, SafetyLevel::TimelineHome, vec![])
                .is_none()
        );
        assert!(fake.calls.lock().unwrap().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn begin_compare_fetches_reference_before_the_verdicts_arrive() {
        let (harness, fake) = fake_harness(FakeReply::Immediate);
        let VerdictSender { sender, task } = harness
            .begin_compare(
                Some(99),
                Some("de".to_string()),
                SafetyLevel::TimelineHomeRecommendations,
                vec![1, 2],
            )
            .expect("comparable request");

        tokio::time::timeout(Duration::from_secs(1), fake.called.notified())
            .await
            .expect("reference called before any verdict was sent");
        assert_eq!(
            *fake.calls.lock().unwrap(),
            vec![(
                vec![1, 2],
                ReferenceSafetyLevel::TimelineHomeRecommendations,
                99,
                Some("de".to_string()),
            )]
        );

        assert!(
            sender
                .send(vec![
                    verdict(1, service_allow()),
                    verdict(2, service_allow())
                ])
                .is_ok()
        );
        assert_eq!(task.await.unwrap(), CompareResult::Compared);
    }

    #[tokio::test(start_paused = true)]
    async fn reference_timeout_ends_the_task_without_comparing() {
        let (harness, _fake) = fake_harness(FakeReply::Hangs);
        let VerdictSender { sender, task } = harness
            .begin_compare(Some(99), None, SafetyLevel::TimelineHome, vec![1])
            .expect("comparable request");

        assert!(sender.send(vec![verdict(1, service_drop())]).is_ok());
        assert_eq!(task.await.unwrap(), CompareResult::ReferenceTimeout);
    }

    #[tokio::test]
    async fn dropped_verdict_sender_ends_the_task() {
        let (harness, _fake) = fake_harness(FakeReply::Immediate);
        let VerdictSender { sender, task } = harness
            .begin_compare(Some(99), None, SafetyLevel::TimelineHome, vec![1])
            .expect("comparable request");

        drop(sender);
        assert_eq!(task.await.unwrap(), CompareResult::VerdictsDropped);
    }
}
