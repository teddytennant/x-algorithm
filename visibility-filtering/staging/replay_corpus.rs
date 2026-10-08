use crate::filter::{FilterRequest, FilterTweets};
use crate::hydration::request_context;
use crate::hydration::sources::Id;
use crate::models::{ClientCapability, RawCandidate, TweetId};
use crate::retweet;
use crate::rules::SafetyLevel;
use crate::rules::metrics::Rpc;
use crate::staging::recording::Recording;
use crate::staging::reference::tweetypie::{Client, Label, vf_label};
use crate::treatment;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use tokio::time::Instant;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Case {
    pub(crate) id: String,
    pub(crate) tags: Vec<String>,
    pub(crate) captured_at_unix: u64,
    pub(crate) build: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) viewer_id: Option<u64>,
    pub(crate) country_code: Option<String>,
    #[serde(default)]
    pub(crate) client: Client,
    pub(crate) client_capability: ClientCapability,
    pub(crate) tweets: Vec<Expected>,
    pub(crate) recording: Recording,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Expected {
    pub(crate) tweet_id: u64,
        pub(crate) tags: Vec<String>,
    pub(crate) tweetypie: Label,
        pub(crate) vf: Label,
    pub(crate) vf_rule: Option<String>,
}

pub(crate) struct Evaluated {
    pub(crate) tweet_id: u64,
    pub(crate) label: Label,
    pub(crate) rule: Option<&'static str>,
}

pub(crate) struct Fixtures {
    pub(crate) users: BTreeSet<u64>,
    pub(crate) tweets: BTreeSet<u64>,
    pub(crate) communities: BTreeSet<u64>,
    pub(crate) articles: BTreeSet<u64>,
    pub(crate) places: BTreeSet<u64>,
}

impl Fixtures {
    pub(crate) fn load(path: &Path) -> anyhow::Result<Self> {
        #[derive(Deserialize)]
        struct File {
            users: Vec<Fixture>,
            tweets: Vec<Fixture>,
            #[serde(default)]
            communities: Vec<Fixture>,
            #[serde(default)]
            articles: Vec<Fixture>,
            #[serde(default)]
            places: Vec<Fixture>,
        }
        #[derive(Deserialize)]
        struct Fixture {
            id: u64,
        }
        let text = fs::read_to_string(path).with_context(|| path.display().to_string())?;
        let file: File = serde_json::from_str(&text).with_context(|| path.display().to_string())?;
        let ids = |fixtures: Vec<Fixture>| fixtures.into_iter().map(|f| f.id).collect();
        Ok(Self {
            users: ids(file.users),
            tweets: ids(file.tweets),
            communities: ids(file.communities),
            articles: ids(file.articles),
            places: ids(file.places),
        })
    }

    pub(crate) fn strangers(&self, recording: &Recording) -> anyhow::Result<Vec<String>> {
        let mut strangers: Vec<String> = recording
            .ids()?
            .into_iter()
            .filter_map(|id| {
                let (kind, id, fixtures) = match id {
                    Id::User(id) => ("user", id, &self.users),
                    Id::Tweet(id) => ("tweet", id, &self.tweets),
                    Id::Community(id) => ("community", id, &self.communities),
                    Id::Article(id) => ("article", id, &self.articles),
                    Id::Place(id) => ("place", id, &self.places),
                };
                (!fixtures.contains(&id)).then(|| format!("{kind} {id}"))
            })
            .collect();
        strangers.sort();
        strangers.dedup();
        Ok(strangers)
    }
}

pub(crate) async fn evaluate(
    filter_tweets: &FilterTweets,
    viewer_id: Option<u64>,
    country_code: Option<String>,
    client_capability: ClientCapability,
    tweet_ids: &[u64],
) -> Vec<Evaluated> {
    let candidates = tweet_ids
        .iter()
        .map(|&tweet_id| RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id: None,
        })
        .collect();
    let request = FilterRequest {
        viewer_id,
        country_code,
        client_capability,
        safety_level: SafetyLevel::TimelineHomeHydration,
        candidates,
        rpc: Rpc::EvaluateTweets,
    };
    request_context(Instant::now(), None)
        .scope(retweet::evaluate_merging_sources(filter_tweets, request))
        .await
        .iter()
        .map(|outcome| Evaluated {
            tweet_id: outcome.tweet_id.0,
            label: vf_label(outcome),
            rule: treatment::decided_rows(outcome.evaluation.verdict())
                .next()
                .map(|(rule, _)| rule),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{CountryList, CountryLists};
    use crate::rules::RuleEngine;
    use crate::staging::recording::Replay;
    use crate::staging::reference::tweetypie::Class;
    use serde::de::DeserializeOwned;
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::Arc;

    const CORPUS: &str = "tests/replay_corpus";
                const SCREENED_DROPS: [&str; 9] = [
        "tweet_is_bounced",
        "author_block_viewer",
        "author_is_protected",
        "author_is_suspended",
        "exclusive_tweet",
        "nsfw_viewer_is_underage",
        "nsfw_viewer_has_no_stated_age",
        "nsfw_logged_out",
        "premium_tweet",
    ];

    fn renders_alike(vf: &Label, tweetypie: &Label) -> bool {
        vf == tweetypie
            || (vf.0 == <&str>::from(Class::BareDrop)
                && tweetypie.0 == <&str>::from(Class::Drop)
                && !SCREENED_DROPS.contains(&tweetypie.1.as_str()))
    }

    #[derive(Deserialize)]
    struct KnownDivergence {
        case: String,
        tweet_id: u64,
                vf: Label,
        reason: String,
    }

            fn corpus_dir() -> PathBuf {
        if let Some(dir) = std::env::var_os("VF_REPLAY_CORPUS") {
            return dir.into();
        }
        crate_path(CORPUS)
    }

    fn crate_path(path: &str) -> PathBuf {
        let cargo = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        if cargo.exists() {
            cargo
        } else {
            Path::new("crates/x-product/xai-visibility-filtering-service").join(path)
        }
    }

    fn read<T: DeserializeOwned>(path: &Path) -> T {
        serde_json::from_str(&fs::read_to_string(path).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

        #[derive(Deserialize)]
    struct Stored {
        #[serde(flatten)]
        case: Case,
                country_lists: Option<BTreeMap<CountryList, Vec<String>>>,
    }

    fn stored(dir: &Path) -> Vec<Stored> {
        let cases = dir.join("cases");
        let mut paths: Vec<PathBuf> = fs::read_dir(&cases)
            .unwrap_or_else(|e| panic!("{}: {e}", cases.display()))
            .map(|entry| entry.unwrap().path())
            .collect();
        paths.sort();
        assert!(!paths.is_empty(), "{} holds no case", cases.display());
        paths.iter().map(|path| read(path)).collect()
    }

    fn cases(dir: &Path) -> Vec<Case> {
        stored(dir).into_iter().map(|stored| stored.case).collect()
    }

            const DECIDED_AT_LEAST: [&str; 64] = [
        "author_blocks_viewer_exclusive_content/drop/unspecified",
        "blocked_viewer/limited_engagement",
        "blocked_viewer/limited_engagement/root_author_blocked_viewer",
        "bounce/drop/bounced",
        "creator_tweet_nsfw/drop/nsfw_media",
        "deactivated_author/drop",
        "exclusive_tweet/drop",
        "fosnr_abuse/drop/undesirable",
        "fosnr_abuse_insults_non_follower/drop/undesirable",
        "fosnr_author/appealable",
        "fosnr_civic_integrity/drop/undesirable",
        "fosnr_fallback/drop/undesirable",
        "fosnr_hateful_conduct/drop/undesirable",
        "fosnr_violent_speech/drop/undesirable",
        "gore_and_violence_high_precision/blur",
        "gore_and_violence_high_precision/legacy_interstitial",
        "gore_and_violence_high_precision/tombstone/update_app_android",
        "gore_and_violence_high_precision/tombstone/update_app_ios",
        "gore_and_violence_ignoring_settings/blur",
        "gore_and_violence_reported_heuristics/blur/sensitive",
        "gore_and_violence_reported_heuristics/legacy_interstitial",
        "limit_replies_by_invitation/limited_engagement/conversation_control",
        "limit_replies_co/limited_engagement/conversation_control",
        "limit_replies_community/limited_engagement/conversation_control",
        "limit_replies_my_network/limited_engagement/conversation_control",
        "limit_replies_subscribers/limited_engagement/conversation_control",
        "limit_replies_verified/limited_engagement/conversation_control",
        "nsfw_account/legacy_interstitial",
        "nsfw_account/tombstone/age_verification",
        "nsfw_account/tombstone/update_app_android",
        "nsfw_account/tombstone/update_app_ios",
        "nsfw_admin/blur/sensitive",
        "nsfw_admin/blur/sensitive/age_prompt",
        "nsfw_card_image/blur/sensitive",
        "nsfw_card_image/blur/sensitive/age_prompt",
        "nsfw_card_image/legacy_interstitial",
        "nsfw_card_image/tombstone/age_verification",
        "nsfw_card_image/tombstone/update_app_android",
        "nsfw_card_image/tombstone/update_app_ios",
        "nsfw_high_precision/blur/nudity",
        "nsfw_high_precision/blur/nudity/age_prompt",
        "nsfw_high_precision/legacy_interstitial",
        "nsfw_high_precision/tombstone/age_verification",
        "nsfw_high_precision/tombstone/local_regulations",
        "nsfw_high_precision/tombstone/update_app_android",
        "nsfw_high_precision/tombstone/update_app_ios",
        "nsfw_reported_heuristics/blur/sensitive",
        "nsfw_reported_heuristics/blur/sensitive/age_prompt",
        "nsfw_reported_heuristics/legacy_interstitial",
        "nsfw_reported_heuristics/tombstone/age_verification",
        "nsfw_reported_heuristics/tombstone/update_app_android",
        "nsfw_reported_heuristics/tombstone/update_app_ios",
        "nsfw_user/blur/sensitive_user",
        "nsfw_user/blur/sensitive_user/age_prompt",
        "offboarded_author/drop/inactive",
        "pdna/drop/safety_result",
        "protected_author/drop",
        "read_only_viewer/limited_engagement",
        "sensitive_viewer_logged_out/drop",
        "sensitive_viewer_no_stated_age/drop",
        "sensitive_viewer_underage/drop",
        "spam/drop/undesirable",
        "stale_tweet/limited_engagement",
        "suspended_author/drop",
    ];

    struct Replayed {
        failures: Vec<String>,
                decided: BTreeSet<&'static str>,
                diverging: BTreeSet<&'static str>,
    }

    async fn replay_cases(dir: &Path) -> Replayed {
        let divergences: Vec<KnownDivergence> = read(&dir.join("known_divergences.json"));
        let shared: Recording = read(&dir.join("shared.json"));
        let shared_lists: BTreeMap<CountryList, Vec<String>> =
            read(&dir.join("country_lists.json"));
        let mut failures = Vec::new();
        let mut decided = BTreeSet::new();
        let mut diverging = BTreeSet::new();
        let mut replayed = Vec::new();
        for Stored {
            case,
            country_lists,
        } in stored(dir)
        {
            replayed.extend(case.tweets.iter().map(|t| (case.id.clone(), t.tweet_id)));
            let recording = case.recording.with_shared(&shared);
            let sources = match recording.into_replay(case.viewer_id) {
                Ok(sources) => Arc::new(sources),
                Err(e) => {
                    failures.push(format!("{}: {e:#}", case.id));
                    continue;
                }
            };
            let country_lists =
                CountryLists::from_codes(country_lists.as_ref().unwrap_or(&shared_lists))
                    .unwrap_or_else(|e| panic!("{}: {e}", case.id));
            let filter_tweets = FilterTweets::new(
                Arc::<Replay>::clone(&sources),
                RuleEngine::with_country_lists(Arc::new(country_lists)),
            );
            let tweet_ids: Vec<u64> = case.tweets.iter().map(|t| t.tweet_id).collect();
            let evaluated = evaluate(
                &filter_tweets,
                case.viewer_id,
                case.country_code,
                case.client_capability,
                &tweet_ids,
            )
            .await;
            let misses = sources.misses();
            if !misses.is_empty() {
                failures.push(format!(
                    "{}: the recording lacks {misses:?}; re-capture it with --test-users from fixtures.json",
                    case.id
                ));
                continue;
            }
            for expected in &case.tweets {
                let Some(vf) = evaluated.iter().find(|vf| vf.tweet_id == expected.tweet_id) else {
                    failures.push(format!("{}/{}: no outcome", case.id, expected.tweet_id));
                    continue;
                };
                let known = divergences
                    .iter()
                    .find(|d| d.case == case.id && d.tweet_id == expected.tweet_id);
                let key = format!("{}/{}", case.id, expected.tweet_id);
                match (renders_alike(&vf.label, &expected.tweetypie), known) {
                    (true, None) => decided.extend(vf.rule),
                    (false, Some(known)) if vf.label == known.vf => diverging.extend(vf.rule),
                    (false, Some(known)) => failures.push(format!(
                        "{key}: vf {:?} by {:?}, its known divergence pins {:?} ({})",
                        vf.label, vf.rule, known.vf, known.reason
                    )),
                    (false, None) => failures.push(format!(
                        "{key} [{}; {}]: vf {:?} by {:?}, tweetypie {:?} (vf at capture {:?} by {:?})",
                        case.tags.join(","),
                        expected.tags.join(","),
                        vf.label,
                        vf.rule,
                        expected.tweetypie,
                        expected.vf,
                        expected.vf_rule,
                    )),
                    (true, Some(known)) => failures.push(format!(
                        "{key}: now matches tweetypie; remove its known divergence ({})",
                        known.reason
                    )),
                }
            }
        }
        failures.extend(
            divergences
                .iter()
                .filter(|d| !replayed.contains(&(d.case.clone(), d.tweet_id)))
                .map(|d| {
                    format!(
                        "{}/{}: known divergence names no replayed tweet",
                        d.case, d.tweet_id
                    )
                }),
        );
        Replayed {
            failures,
            decided,
            diverging,
        }
    }

    fn wired_rules() -> Vec<&'static str> {
        RuleEngine::for_tests().wired_rule_names(SafetyLevel::TimelineHomeHydration)
    }

    impl Replayed {
                fn undecided(&self) -> Vec<&'static str> {
            wired_rules()
                .into_iter()
                .filter(|rule| !self.decided.contains(rule))
                .collect()
        }
    }

    #[tokio::test]
    async fn every_case_replays_to_tweetypies_answer() {
        let replayed = replay_cases(&corpus_dir()).await;
        let wired = wired_rules();
        let undecided = replayed.undecided();
        let report = format!(
            "replay corpus decides {} of {} level-82 rules\nreplay corpus leaves undecided: {}\n",
            wired.len() - undecided.len(),
            wired.len(),
            undecided.join(", ")
        );
        std::io::stderr().write_all(report.as_bytes()).unwrap();
        assert!(
            replayed.failures.is_empty(),
            "\n{}",
            replayed.failures.join("\n")
        );
    }

    #[tokio::test]
    async fn replay_corpus_decides_at_least() {
        let decided = replay_cases(&corpus_dir()).await.decided;
        let lost: Vec<&str> = DECIDED_AT_LEAST
            .into_iter()
            .filter(|rule| !decided.contains(rule))
            .collect();
        assert!(
            lost.is_empty(),
            "the replay corpus no longer decides {lost:?}: a case deciding it was removed or no longer replays to Tweetypie's answer"
        );
    }

            #[tokio::test]
    async fn every_level_82_rule_is_decided_diverging_or_unreachable() {
        #[derive(Deserialize)]
        struct Unreachable {
            rule: String,
        }
        let replayed = replay_cases(&corpus_dir()).await;
        let unreachable: Vec<Unreachable> = read(&crate_path("scripts/corpus/unreachable.json"));
        let listed = |rule: &str| unreachable.iter().any(|u| u.rule == rule);
        let unlisted: Vec<&str> = replayed
            .undecided()
            .into_iter()
            .filter(|rule| !replayed.diverging.contains(rule) && !listed(rule))
            .collect();
        let wired = wired_rules();
        let stale: Vec<&str> = unreachable
            .iter()
            .map(|u| u.rule.as_str())
            .filter(|rule| {
                replayed.decided.contains(rule)
                    || replayed.diverging.contains(rule)
                    || !wired.contains(rule)
            })
            .collect();
        assert!(
            unlisted.is_empty() && stale.is_empty(),
            "level-82 rules no case decides or pins by a known divergence, missing from scripts/corpus/unreachable.json: {unlisted:?}\nunreachable.json rules a case decides, a known divergence pins or level 82 does not wire: {stale:?}"
        );
    }

    #[test]
    fn shared_json_holds_every_viewer_free_answer_and_only_those() {
        let dir = corpus_dir();
        let shared: Recording = read(&dir.join("shared.json"));
        let (unshared, _) = shared.split_shared();
        let mut misplaced: Vec<String> = unshared
            .names()
            .iter()
            .map(|name| format!("shared.json: {name}"))
            .collect();
        for case in cases(&dir) {
            let (_, shared) = case.recording.split_shared();
            misplaced.extend(
                shared
                    .names()
                    .iter()
                    .map(|name| format!("{}: {name}", case.id)),
            );
        }
        assert!(misplaced.is_empty(), "misplaced: {misplaced:?}");
    }

    #[test]
    fn every_id_the_corpus_names_is_a_fixture() {
        let dir = corpus_dir();
        let fixtures = Fixtures::load(&dir.join("fixtures.json")).unwrap();
        let shared: Recording = read(&dir.join("shared.json"));
        let named = |file: &str, recording: &Recording| -> Vec<String> {
            fixtures
                .strangers(recording)
                .unwrap_or_else(|e| panic!("{file}: {e:#}"))
                .into_iter()
                .map(|stranger| format!("{file}: {stranger}"))
                .collect()
        };
        let mut strangers = named("shared.json", &shared);
        for case in cases(&dir) {
            strangers.extend(named(&case.id, &case.recording));
            if let Some(viewer_id) = case.viewer_id
                && !fixtures.users.contains(&viewer_id)
            {
                strangers.push(format!("{}: user {viewer_id}", case.id));
            }
            strangers.extend(
                case.tweets
                    .iter()
                    .filter(|tweet| !fixtures.tweets.contains(&tweet.tweet_id))
                    .map(|tweet| format!("{}: tweet {}", case.id, tweet.tweet_id)),
            );
        }
        assert!(strangers.is_empty(), "not in fixtures.json: {strangers:?}");
    }

            const NEVER_ALLOWED: [(&str, &str); 11] = [
        (
            "author:deactivated_author",
            "Tweetypie drops a deactivated author's post for every viewer, the author included",
        ),
        (
            "creator_exclusive_image",
            "an NSFW-labeled subscribers-only image is dropped for every viewer but its author, who sees it blurred; vfcreatr's exclusive text post is the allowed control",
        ),
        (
            "erased_author_text",
            "Tweetypie answers an erased author's post not found",
        ),
        (
            "label_fosnr_abuse",
            "FOSNR drops the post for non-followers and puts a notice on it for the author and followers",
        ),
        ("label_fosnr_abuse_insults", "as label_fosnr_abuse"),
        ("label_fosnr_civic_integrity", "as label_fosnr_abuse"),
        ("label_fosnr_hateful_conduct", "as label_fosnr_abuse"),
        ("label_fosnr_violent_speech", "as label_fosnr_abuse"),
        (
            "label_gore_and_violence_high_precision",
            "every client gets a media treatment, a tombstone or an interstitial",
        ),
        (
            "offboarded_author_text",
            "Tweetypie drops an offboarded author's post for every viewer",
        ),
        (
            "stale_original",
            "a superseded edit is limited for every viewer; stale_edit is its allowed control",
        ),
    ];

    #[test]
    fn every_tag_has_an_allowed_pair_or_a_reason_it_cannot() {
        let mut allowed: BTreeMap<String, bool> = BTreeMap::new();
        for case in cases(&corpus_dir()) {
            for tweet in &case.tweets {
                let allow = tweet.tweetypie.0 == <&str>::from(Class::Allow);
                for tag in &tweet.tags {
                    *allowed.entry(tag.clone()).or_default() |= allow;
                }
            }
        }
        let exempt = |tag: &str| NEVER_ALLOWED.iter().any(|(never, _)| *never == tag);
        let missing: Vec<&str> = allowed
            .iter()
            .filter(|&(tag, &allow)| !allow && !exempt(tag))
            .map(|(tag, _)| tag.as_str())
            .collect();
        let stale: Vec<&str> = NEVER_ALLOWED
            .iter()
            .map(|(tag, _)| *tag)
            .filter(|tag| allowed.get(*tag) != Some(&false))
            .collect();
        assert!(
            missing.is_empty() && stale.is_empty(),
            "tags with no pair Tweetypie allows, missing from NEVER_ALLOWED: {missing:?}\nNEVER_ALLOWED tags the corpus allows or no longer has: {stale:?}"
        );
    }

    #[test]
    fn every_fixture_user_states_what_gizmoduck_answers_for_it() {
        #[derive(Deserialize)]
        struct File {
            users: Vec<User>,
        }
        #[derive(Deserialize)]
        struct User {
            id: u64,
            gizmoduck: Option<BTreeMap<String, serde::de::IgnoredAny>>,
        }
        let file: File = read(&corpus_dir().join("fixtures.json"));
        let missing: Vec<String> = file
            .users
            .iter()
            .filter(|user| user.gizmoduck.is_none())
            .map(|user| user.id.to_string())
            .collect();
        assert!(
            missing.is_empty(),
            "fixtures.json users without the gizmoduck object vf-fixture-mint --check compares: {}",
            missing.join(", ")
        );
    }
}
