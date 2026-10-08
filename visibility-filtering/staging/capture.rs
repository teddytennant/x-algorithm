use crate::config::ENV_APP_ENV;
use crate::filter::FilterTweets;
use crate::models::TweetId;
use crate::params::{ClientSwitches, CountryList, CountryLists, SwitchFiles};
use crate::rules::RuleEngine;
use crate::server_deps::{CLIENT_INIT_RETRY_BUDGET, prod_sources};
use crate::staging::recording::{Recorder, Recording};
use crate::staging::reference::ENV_IMAGE;
use crate::staging::reference::tweetypie::{self, Class, Client, get_tweet_fields};
use crate::staging::reference_compare::resolve_build_sha;
use crate::staging::replay_corpus::{Case, Expected, Fixtures, evaluate};
use anyhow::{Context, bail};
use arc_swap::ArcSwap;
use futures::future::join;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, stdin};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::Instant;
use tonic::metadata::{MetadataMap, MetadataValue};
use xai_build_version::current_build_information;

const TEST_USERS_DTAB: &str = "/s/gizmoduck/test-users-permanent=>/s/gizmoduck/gizmoduck";

pub const FIXTURES_IN_IMAGE: &str = "/config/replay_corpus/fixtures.json";

#[derive(Deserialize)]
struct CaptureRequest {
    id: String,
    #[serde(default)]
    tags: Vec<String>,
        viewer_id: Option<u64>,
    country_code: Option<String>,
    #[serde(default)]
    client: Client,
    tweets: Vec<CaptureTweet>,
}

#[derive(Serialize)]
struct Captured {
    #[serde(flatten)]
    case: Case,
            country_lists: BTreeMap<CountryList, Vec<String>>,
    shared: Recording,
}

impl Captured {
    fn new(case: Case, country_lists: BTreeMap<CountryList, Vec<String>>) -> Self {
        let (recording, shared) = case.recording.split_shared();
        Self {
            case: Case { recording, ..case },
            country_lists,
            shared,
        }
    }
}

#[derive(Deserialize)]
struct CaptureTweet {
    id: u64,
    #[serde(default)]
    tags: Vec<String>,
}

#[expect(clippy::print_stdout, reason = "stdout is the case sink")]
#[expect(clippy::print_stderr, reason = "stderr is the progress log")]
pub async fn run(datacenter: &str, test_users: bool, fixtures: &Path) -> anyhow::Result<()> {
    let app_env = env::var(ENV_APP_ENV).unwrap_or_default();
    if !matches!(app_env.as_str(), "staging" | "devel") {
        bail!("capture runs only in staging or devel: {ENV_APP_ENV}={app_env:?}");
    }
    let init_deadline = Instant::now() + CLIENT_INIT_RETRY_BUDGET;
    let recorder = Arc::new(Recorder::default());
    let metadata = test_users.then(test_users_metadata);
    let (sources, _) = prod_sources(
        datacenter,
        init_deadline,
        false,
        None,
        None,
        metadata.as_ref(),
    )
    .await;
    let feature_switches = SwitchFiles::beside(&crate::config::fs_path())
        .load(None)
        .context("feature switch files")?;
    let country_lists = Arc::new(CountryLists::starting_at_default());
    country_lists.refresh(&feature_switches);
    let client_switches = ClientSwitches::new(Arc::new(ArcSwap::from_pointee(feature_switches)));
    let filter_tweets = FilterTweets::new(
        Arc::new(sources.observed(Arc::clone(&recorder))),
        RuleEngine::with_country_lists(Arc::clone(&country_lists)),
    );
    let fixtures = Fixtures::load(fixtures)?;
    let tweetypie = tweetypie::connect(init_deadline, metadata.as_ref()).await;
    let build = resolve_build_sha(
        &current_build_information().git_commit_sha,
        env::var(ENV_IMAGE).ok().as_deref(),
    );

    for line in stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: CaptureRequest = serde_json::from_str(&line).context("request line")?;
        if let Some(viewer_id) = request.viewer_id
            && !fixtures.users.contains(&viewer_id)
        {
            eprintln!(
                "capture: skipped {}: viewer {viewer_id} is not in fixtures.json",
                request.id
            );
            continue;
        }
        let strange_tweets: Vec<u64> = request
            .tweets
            .iter()
            .map(|tweet| tweet.id)
            .filter(|id| !fixtures.tweets.contains(id))
            .collect();
        if !strange_tweets.is_empty() {
            eprintln!(
                "capture: skipped {}: tweets {strange_tweets:?} are not in fixtures.json",
                request.id
            );
            continue;
        }
        let client = request
            .client
            .context(request.viewer_id, request.country_code.as_deref());
        let client_capability = client_switches.resolve(
            Some(&client),
            request.viewer_id,
            request.country_code.as_deref(),
        );
        let tweet_ids: Vec<u64> = request.tweets.iter().map(|tweet| tweet.id).collect();
        let ids: Vec<TweetId> = tweet_ids.iter().copied().map(TweetId).collect();
        recorder.take();
        let (vf, tp) = join(
            evaluate(
                &filter_tweets,
                request.viewer_id,
                request.country_code.clone(),
                client_capability,
                &tweet_ids,
            ),
            get_tweet_fields(&tweetypie, &client, &ids),
        )
        .await;
        let recording = recorder.take();
        let unsuccessful = recording.unsuccessful();
        if !unsuccessful.is_empty() {
            eprintln!(
                "capture: skipped {}: failed or partial answers {unsuccessful:?}",
                request.id
            );
            continue;
        }
        match fixtures.strangers(&recording) {
            Ok(strangers) if strangers.is_empty() => {}
            Ok(strangers) => {
                eprintln!(
                    "capture: skipped {}: {strangers:?} are not in fixtures.json",
                    request.id
                );
                continue;
            }
            Err(e) => {
                eprintln!("capture: skipped {}: {e:#}", request.id);
                continue;
            }
        }
        let tweets: Vec<Expected> = request
            .tweets
            .into_iter()
            .zip(tp)
            .filter_map(|(tweet, tp)| {
                let vf = vf.iter().find(|vf| vf.tweet_id == tweet.id)?;
                let failed = <&str>::from(Class::Failed);
                (vf.label.0 != failed && tp.label.0 != failed).then(|| Expected {
                    tweet_id: tweet.id,
                    tags: tweet.tags,
                    tweetypie: tp.label,
                    vf: vf.label.clone(),
                    vf_rule: vf.rule.map(str::to_owned),
                })
            })
            .collect();
        if tweets.is_empty() {
            eprintln!(
                "capture: skipped {}: no tweet answered on both sides",
                request.id
            );
            continue;
        }
        let case = Case {
            id: request.id,
            tags: request.tags,
            captured_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_secs()),
            build: build.clone(),
            viewer_id: request.viewer_id,
            country_code: request.country_code,
            client: request.client,
            client_capability,
            tweets,
            recording,
        };
        println!(
            "{}",
            serde_json::to_string(&Captured::new(case, country_lists.codes()))?
        );
    }
    Ok(())
}

pub fn test_users_metadata() -> MetadataMap {
    let mut metadata = MetadataMap::new();
    metadata.insert("dtab-local", MetadataValue::from_static(TEST_USERS_DTAB));
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::Hydrated;
    use crate::hydration::sources::{Observer, exchange};
    use crate::models::ClientCapability;
    use std::iter;

    #[test]
    fn a_captured_line_is_its_case_beside_the_shared_answers() {
        let recorder = Arc::new(Recorder::default());
        recorder.answered::<exchange::TesPureCore>(iter::once((1, Hydrated::NotFound)));
        recorder.answered::<exchange::GizmoduckViewer>(iter::once((50, Hydrated::NotFound)));
        let case = Case {
            id: "a".to_owned(),
            tags: vec![],
            captured_at_unix: 0,
            build: String::new(),
            viewer_id: Some(50),
            country_code: None,
            client: Client::Web,
            client_capability: ClientCapability::default(),
            tweets: vec![],
            recording: recorder.take(),
        };

        let mut line = serde_json::to_value(Captured::new(case, BTreeMap::new())).unwrap();
        let shared = line.as_object_mut().unwrap().remove("shared");
        assert_eq!(
            shared,
            Some(serde_json::json!({"pure_cores": {"1": "not_found"}}))
        );
        let case: Case = serde_json::from_value(line).unwrap();
        assert_eq!(case.id, "a");
        assert_eq!(case.recording.names(), ["viewers"]);
    }

    #[test]
    fn a_request_line_names_its_client_or_is_web() {
        let client = |line: &str| serde_json::from_str::<CaptureRequest>(line).unwrap().client;
        assert_eq!(
            client(r#"{"id":"a","viewer_id":1,"tweets":[]}"#),
            Client::Web
        );
        assert_eq!(
            client(r#"{"id":"a","viewer_id":1,"client":"ios_outdated","tweets":[]}"#),
            Client::IosOutdated
        );
    }
}
