mod age_gating;
mod age_verification;
mod article;
mod author_state;
mod baseline;
mod builders;
mod community;
mod conversation_control;
mod exclusive_content;
mod interstitial;
mod legacy_interstitial;
mod local_regulations;
mod local_tweet;
mod oon_media;
mod oon_tweet_label;
mod oon_user_label;
mod relationship;
mod takedown;
mod trusted_friends;
mod tweet_label;
mod tweet_state;

use crate::hydration::{Hydrator, Hydrators};
use crate::models::{HydratedTweetCandidate, Verdict, ViewerFeatures};
use crate::rules::fixtures::{logged_out_viewer, viewer, VIEWER_ID};
use crate::rules::{RuleEngine, SafetyLevel};
use crate::treatment::proto_action;
use prost::Message;
use std::collections::BTreeSet;
use SafetyLevel::{
    FilterAll, ImmersiveExpandedRecommendations, TimelineHome, TimelineHomeHydration,
    TimelineHomeRecommendations,
};

enum Role {
    NonFollower,
    Author,
    Follower,
    LoggedOut,
    As(&'static str, ViewerFeatures),
}

struct Row {
    name: &'static str,
    post: HydratedTweetCandidate,
    expect: Vec<(SafetyLevel, Role, Verdict)>,
}

struct CorpusCase {
    name: String,
    level: SafetyLevel,
    viewer: ViewerFeatures,
    candidate: HydratedTweetCandidate,
    expected: Verdict,
}

impl Row {
    fn expand(self) -> impl Iterator<Item = CorpusCase> {
        let Row { name, post, expect } = self;
        expect.into_iter().map(move |(level, role, expected)| {
            let mut candidate = post.clone();
            let (role_name, viewer) = match role {
                Role::NonFollower => ("non_follower", viewer(VIEWER_ID)),
                Role::Author => ("author", viewer(candidate.author_id)),
                Role::Follower => {
                    candidate.edges = candidate.edges.with(Hydrator::Follows);
                    ("follower", viewer(VIEWER_ID))
                }
                Role::LoggedOut => {
                    candidate.edges = Hydrators::empty();
                    ("logged_out", logged_out_viewer())
                }
                Role::As(role_name, viewer) => (role_name, viewer),
            };
            CorpusCase {
                name: format!("{name}/{level:?}/{role_name}"),
                level,
                viewer,
                candidate,
                expected,
            }
        })
    }
}

fn deciders(verdict: &Verdict) -> Vec<&'static str> {
    match verdict {
        Verdict::Withheld(decided) => vec![decided.by],
        Verdict::Shown {
            notice,
            media,
            engagement,
        } => notice
            .iter()
            .map(|notice| notice.by)
            .chain(media.iter().map(|blur| blur.by))
            .chain(engagement.iter().map(|limit| limit.by))
            .collect(),
    }
}

#[test]
fn golden_corpus_pins_policy_verdicts() {
    let rule_engine = RuleEngine::for_tests();
    let cases = corpus();
    let names: BTreeSet<&str> = cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names.len(), cases.len(), "duplicate corpus case name");
    let mut failures = Vec::new();
    for case in cases {
        let verdict = rule_engine
            .evaluate(case.level, &case.viewer, &case.candidate)
            .into_verdict();
        if matches!(&case.expected, Verdict::Shown { media: Some(_), .. }) {
            let (action, reason) = proto_action(verdict.clone());
            assert_eq!(action.encode_to_vec(), [0x20, 0x01], "{}", case.name);
            assert_eq!(
                reason.unwrap().encode_to_vec(),
                [0x08, 0x01],
                "{}",
                case.name
            );
        }
        if verdict != case.expected {
            failures.push(format!(
                "{} [{:?}]:\n  expected {:?}\n  got      {:?}",
                case.name, case.level, case.expected, verdict,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} corpus case(s) diverged:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore]
fn bench_corpus_evaluation() {
    use std::hint::black_box;
    use std::time::Instant;

    const ROUNDS: u32 = 2_000;
    let rule_engine = RuleEngine::for_tests();
    let cases = corpus();
    let home_hydration: Vec<&CorpusCase> = cases
        .iter()
        .filter(|case| case.level == TimelineHomeHydration)
        .collect();
    for (label, cases) in [
        ("all", cases.iter().collect::<Vec<_>>()),
        ("home_hydration", home_hydration),
    ] {
        let start = Instant::now();
        for _ in 0..ROUNDS {
            for case in &cases {
                black_box(rule_engine.evaluate(
                    case.level,
                    black_box(&case.viewer),
                    black_box(&case.candidate),
                ));
            }
        }
        let evaluations = u128::from(ROUNDS) * cases.len() as u128;
        println!(
            "BENCH {label}: {} cases, {} ns/evaluation",
            cases.len(),
            start.elapsed().as_nanos() / evaluations.max(1)
        );
    }
}

#[test]
fn every_node_failing_changes_no_corpus_verdict() {
    let rule_engine = RuleEngine::for_tests();
    for mut case in corpus() {
        case.candidate.failed = Hydrators::all();
        let verdict = rule_engine
            .evaluate(case.level, &case.viewer, &case.candidate)
            .into_verdict();
        assert_eq!(verdict, case.expected, "{}", case.name);
    }
}

#[test]
fn every_wired_rule_decides_a_corpus_case() {
    let rule_engine = RuleEngine::for_tests();
    let wired: BTreeSet<&'static str> = [
        FilterAll,
        TimelineHome,
        TimelineHomeRecommendations,
        TimelineHomeHydration,
        ImmersiveExpandedRecommendations,
    ]
    .into_iter()
    .flat_map(|level| rule_engine.wired_rule_names(level))
    .collect();
    let deciders: BTreeSet<&'static str> = corpus()
        .iter()
        .flat_map(|c| deciders(&c.expected))
        .collect();
    let missing: Vec<&&'static str> = wired.difference(&deciders).collect();
    assert!(
        missing.is_empty(),
        "rules wired in RuleEngine but never the decider of any corpus case: {missing:?}"
    );
}

fn corpus() -> Vec<CorpusCase> {
    rows().into_iter().flat_map(Row::expand).collect()
}

fn rows() -> Vec<Row> {
    [
        baseline::rows(),
        relationship::rows(),
        author_state::rows(),
        tweet_label::rows(),
        tweet_state::rows(),
        community::rows(),
        takedown::rows(),
        article::rows(),
        age_gating::rows(),
        age_verification::rows(),
        exclusive_content::rows(),
        trusted_friends::rows(),
        conversation_control::rows(),
        interstitial::rows(),
        legacy_interstitial::rows(),
        local_regulations::rows(),
        local_tweet::rows(),
        oon_media::rows(),
        oon_tweet_label::rows(),
        oon_user_label::rows(),
    ]
    .into_iter()
    .flatten()
    .collect()
}
