use crate::filter::{FilterOutcome, FilterRequest, FilterTweets, HydratedRequest};
use crate::hydration::Hydrators;
use crate::models::{
    Decided, Evaluation, HydratedTweetCandidate, TweetFeatures, TweetId, Verdict, Withholding,
};
use crate::rules::metrics as ft_metrics;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

const RETWEET_SOURCES: &str = "evaluate_tweets_retweet_sources";

pub(crate) async fn evaluate_merging_sources(
    filter_tweets: &FilterTweets,
    request: FilterRequest,
) -> Vec<FilterOutcome> {
    let hydrated = filter_tweets.hydrate_with_retweet_sources(request).await;
    let (in_batch, fetched) = source_ids(&hydrated);
    if in_batch.is_empty() && fetched.is_empty() {
        return filter_tweets.evaluate_in_request_order(&hydrated);
    }
    ft_metrics::incr_nonzero(
        RETWEET_SOURCES,
        &[("outcome", "in_batch")],
        in_batch.len() as u64,
    );
    ft_metrics::incr_nonzero(
        RETWEET_SOURCES,
        &[("outcome", "fetched")],
        fetched.len() as u64,
    );
    let copied_retweets = copy_from_sources(&hydrated);
    let mut outcomes = filter_tweets.evaluate(
        &hydrated,
        hydrated.requested_ids().chain(fetched.iter().copied()),
        &copied_retweets,
    );
    let mut sources: HashMap<TweetId, Evaluation> = in_batch
        .iter()
        .filter_map(|&source_id| Some((source_id, outcomes.get(&source_id)?.evaluation.clone())))
        .collect();
    sources.extend(fetched.iter().filter_map(|source_id| {
        let source = outcomes.remove(source_id)?;
        Some((source.tweet_id, source.evaluation))
    }));
    let mut merged: Vec<FilterOutcome> = Vec::with_capacity(hydrated.requested_ids().len());
    for tweet_id in hydrated.requested_ids() {
        let outcome = match outcomes.remove(&tweet_id) {
            Some(outcome) => merge_source(outcome, &sources),
            None => match merged.iter().rfind(|outcome| outcome.tweet_id == tweet_id) {
                Some(outcome) => outcome.clone(),
                None => continue,
            },
        };
        merged.push(outcome);
    }
    merged
}

fn merge_source(outcome: FilterOutcome, sources: &HashMap<TweetId, Evaluation>) -> FilterOutcome {
    let evaluation = match (outcome.evaluation, outcome.source_tweet_id) {
        (Evaluation::Complete { verdict }, Some(source_id)) => match sources.get(&source_id) {
            Some(Evaluation::Complete { verdict: source }) => Evaluation::Complete {
                verdict: merge_verdict(verdict, source),
            },
            Some(Evaluation::Partial { .. } | Evaluation::NotFound(_) | Evaluation::Failed(_))
            | None => Evaluation::Partial {
                verdict,
                fail_open_defaults: Hydrators::empty(),
            },
        },
        (evaluation, _) => evaluation,
    };
    FilterOutcome {
        evaluation,
        ..outcome
    }
}

fn source_ids(hydrated: &HydratedRequest) -> (HashSet<TweetId>, HashSet<TweetId>) {
    let hydration = hydrated.hydration();
    let sources: HashSet<TweetId> = hydrated
        .requested_ids()
        .filter_map(|tweet_id| hydration.tweet(tweet_id)?.source_to_merge())
        .collect();
    if sources.is_empty() {
        return Default::default();
    }
    let requested: HashSet<TweetId> = hydrated.requested_ids().collect();
    sources
        .into_iter()
        .partition(|source_id| requested.contains(source_id))
}

#[derive(Clone, Copy)]
struct CopiedFromSource {
    has_media: bool,
}

impl CopiedFromSource {
    fn of(source: &TweetFeatures) -> Self {
        Self {
            has_media: source.media.has_uploaded_media,
        }
    }

    fn is_on(self, retweet: &TweetFeatures) -> bool {
        retweet.media.has_media == self.has_media
    }

    fn copy_onto(self, retweet: &mut TweetFeatures) {
        retweet.media.has_media = self.has_media;
    }
}

fn copy_from_sources(hydrated: &HydratedRequest) -> HashMap<TweetId, HydratedTweetCandidate> {
    let hydration = hydrated.hydration();
    let source = |source_id| hydration.tweet(source_id)?.candidate();
    let mut copied_retweets = HashMap::new();
    for tweet_id in hydrated.requested_ids() {
        if let Entry::Vacant(entry) = copied_retweets.entry(tweet_id)
            && let Some(retweet) = hydration.tweet(tweet_id)
            && let Some(source) = retweet.source_tweet_id().and_then(source)
            && let Some(candidate) = retweet.candidate()
        {
            let copied = CopiedFromSource::of(&source.tweet_features);
            if !copied.is_on(&candidate.tweet_features) {
                let mut candidate = candidate.clone();
                copied.copy_onto(&mut candidate.tweet_features);
                entry.insert(candidate);
            }
        }
    }
    copied_retweets
}

fn merge_verdict(retweet: Verdict, source: &Verdict) -> Verdict {
    match (&retweet, source) {
        (
            Verdict::Withheld(Decided {
                value: Withholding::Drop(_),
                ..
            }),
            Verdict::Withheld(Decided {
                value: Withholding::Tombstone(_),
                ..
            }),
        ) => retweet,
        (_, Verdict::Withheld(_)) => source.clone(),
        (Verdict::Withheld(_), _) => retweet,
        (
            Verdict::Shown {
                notice: None,
                media: None,
                engagement: None,
            },
            _,
        ) => source.clone(),
        _ => retweet,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeDirection, Graph};
    use crate::hydration::Lookup;
    use crate::hydration::plan::Source;
    use crate::hydration::sources::{InMemorySources, control, suspended};
    use crate::models::{
        ClientCapability, DropReason, LimitedEngagement, LimitedEngagementReason, MediaFeature,
        MediaInterstitial, MediaRestriction, NsfwFeature, RawCandidate, SafetyLabelType,
        TombstoneReason, TweetFeatures,
    };
    use crate::rules::fixtures::{allow, legacy_interstitial, limited, noticed};
    use crate::rules::metrics::Rpc;
    use crate::rules::{RuleEngine, SafetyLevel};
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::time::Duration;
    use xai_core_entities::entities::{ConversationControlArm, PureCoreData};
    use xai_visibility_filtering::models::FilteredReason;
    use xai_visibility_filtering_proto as vf_pb;
    use xai_x_thrift::action::InterstitialReason;

    #[test]
    fn merge_verdict_prefers_the_more_severe_withholding_then_restricted_retweet() {
        let dropped = Verdict::Withheld(Decided {
            value: Withholding::Drop(DropReason::Legacy(FilteredReason::AuthorIsSuspended)),
            by: "suspended_author/drop",
        });
        let tombstoned = Verdict::Withheld(Decided {
            value: Withholding::Tombstone(TombstoneReason::LocalRegulations),
            by: "nsfw_high_precision/tombstone/local_regulations",
        });
        let blurred = Verdict::Shown {
            notice: None,
            media: Some(Decided {
                value: MediaRestriction::MediaInterstitial(MediaInterstitial {
                    legacy: FilteredReason::ContainNsfwMedia,
                    reason: InterstitialReason::Sensitive(true),
                    prompt: None,
                }),
                by: "nsfw_user/blur/sensitive_user",
            }),
            engagement: None,
        };
        let limited = Verdict::Shown {
            notice: None,
            media: None,
            engagement: Some(Decided {
                value: LimitedEngagement::new(LimitedEngagementReason::ConversationControl),
                by: "limit_replies_by_invitation/limited_engagement/conversation_control",
            }),
        };
        let unrestricted = Verdict::Shown {
            notice: None,
            media: None,
            engagement: None,
        };
        let noticed = noticed(
            true,
            false,
            "fosnr_abuse_insults_follower/soft_intervention/abuse",
        );
        for (retweet, source, merged) in [
            (&Verdict::not_found(), &dropped, &dropped),
            (&dropped, &tombstoned, &dropped),
            (&tombstoned, &dropped, &dropped),
            (&unrestricted, &tombstoned, &tombstoned),
            (&dropped, &limited, &dropped),
            (&unrestricted, &limited, &limited),
            (&blurred, &limited, &blurred),
            (&unrestricted, &noticed, &noticed),
            (&limited, &noticed, &limited),
        ] {
            assert_eq!(merge_verdict(retweet.clone(), source), *merged);
        }
    }

    #[tokio::test]
    async fn a_retweet_has_media_exactly_when_its_source_uploaded_some_in_the_batch_or_fetched() {
        let features = |nsfw_admin, media| TweetFeatures {
            nsfw: NsfwFeature {
                user: false,
                admin: nsfw_admin,
            },
            media,
            ..Default::default()
        };
        let media = |has_media, has_uploaded_media| MediaFeature {
            has_media,
            has_uploaded_media,
            ..Default::default()
        };
        let retweet = |author_id, source_tweet_id| PureCoreData {
            author_id,
            source_tweet_id: Some(source_tweet_id),
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .pure_core(4, retweet(40, 6))
                .pure_core(5, retweet(50, 7))
                .tweet(6, 60)
                .tweet(7, 70)
                .authors(&[40, 50, 60, 70])
                .tweet_features(4, features(true, media(false, false)))
                .tweet_features(5, features(true, media(true, true)))
                .tweet_features(6, features(false, media(true, true)))
                .tweet_features(7, features(false, media(true, false))),
        );
        let filter_tweets = FilterTweets::new(sources, RuleEngine::for_tests());
        let blurred = Evaluation::Complete {
            verdict: legacy_interstitial("nsfw_account/legacy_interstitial"),
        };
        let allowed = Evaluation::Complete { verdict: allow() };
        for tweet_ids in [vec![4, 5, 6, 7], vec![4, 5]] {
            let outcomes = evaluate_merging_sources(
                &filter_tweets,
                FilterRequest {
                    viewer_id: Some(1),
                    country_code: None,
                    client_capability: ClientCapability::default(),
                    safety_level: SafetyLevel::TimelineHomeHydration,
                    candidates: tweet_ids
                        .iter()
                        .map(|&tweet_id| RawCandidate {
                            tweet_id: TweetId(tweet_id),
                            request_author_id: None,
                        })
                        .collect(),
                    rpc: Rpc::EvaluateTweets,
                },
            )
            .await;
            assert_eq!(outcomes[0].evaluation, blurred, "{tweet_ids:?}");
            assert_eq!(outcomes[1].evaluation, allowed, "{tweet_ids:?}");
        }
    }

    fn request(tweet_ids: &[u64]) -> FilterRequest {
        FilterRequest {
            viewer_id: Some(1),
            country_code: None,
            client_capability: ClientCapability::default(),
            safety_level: SafetyLevel::TimelineHomeHydration,
            candidates: tweet_ids
                .iter()
                .map(|&tweet_id| RawCandidate {
                    tweet_id: TweetId(tweet_id),
                    request_author_id: None,
                })
                .collect(),
            rpc: Rpc::EvaluateTweets,
        }
    }

    fn retweet_world(share_names_author: bool) -> InMemorySources {
        let shared = |author_id, source_tweet_id, source_user_id| PureCoreData {
            source_user_id: Some(source_user_id).filter(|_| share_names_author),
            ..retweet(author_id, source_tweet_id, source_user_id)
        };
        let community = |root| control(ConversationControlArm::Community, root, &[]);
        InMemorySources::default()
            .pure_core(1, shared(10, 5, 20))
            .tweet(2, 20)
            .pure_core(3, shared(30, 5, 20))
            .pure_core(4, shared(40, 6, 60))
            .tweet(5, 20)
            .pure_core(
                6,
                PureCoreData {
                    author_id: 60,
                    conversation_id: Some(100),
                    in_reply_to_tweet_id: Some(100),
                    in_reply_to_user_id: Some(70),
                    ..Default::default()
                },
            )
            .control(2, community(90))
            .control(5, community(80))
            .authors(&[10, 20, 30, 40, 60])
    }

    fn retweet(author_id: u64, source_tweet_id: u64, source_user_id: u64) -> PureCoreData {
        PureCoreData {
            author_id,
            source_tweet_id: Some(source_tweet_id),
            source_user_id: Some(source_user_id),
            ..Default::default()
        }
    }

    const TWEET_KEYED: [Source; 4] = [
        Source::TesPureCore,
        Source::TesTweet,
        Source::TesConversationControl,
        Source::SafetyLabels,
    ];

    fn batches(sources: &InMemorySources) -> String {
        let mut lines: Vec<String> = TWEET_KEYED
            .into_iter()
            .chain([Source::GizmoduckViewer, Source::GizmoduckAuthor])
            .map(|source| format!("{source:?} {:?}", sources.keys(source)))
            .collect();
        for queries in sources.selects() {
            let queries: Vec<String> = queries
                .iter()
                .map(|q| {
                    let direction = match q.direction {
                        EdgeDirection::Forward => "fwd",
                        EdgeDirection::Reverse => "rev",
                    };
                    format!(
                        "{}-{direction}{:?}",
                        <&str>::from(q.graph),
                        q.destination_ids
                    )
                })
                .collect();
            lines.push(format!("Flock {}", queries.join(" ")));
        }
        lines.join("\n")
    }

    #[tokio::test]
    async fn sources_join_unsent_calls_and_sent_calls_follow_with_only_new_keys() {
        let not_a_reply = || {
            retweet_world(true).pure_core(
                6,
                PureCoreData {
                    author_id: 60,
                    ..Default::default()
                },
            )
        };
        let tweet_keyed = |keys: &str| {
            TWEET_KEYED
                .map(|source| format!("{source:?} {keys}"))
                .join("\n")
        };
        let rows = [
            (
                "a",
                retweet_world(true),
                &[2][..],
                format!(
                    "{}\nGizmoduckViewer [[1]]\nGizmoduckAuthor [[20]]
Flock follows-rev[90] super_follows-fwd[]
Flock follows-fwd[20] blocks-rev[20]",
                    tweet_keyed("[[2]]")
                ),
            ),
            (
                "b",
                retweet_world(true),
                &[1, 2, 3, 5][..],
                format!(
                    "{}\nGizmoduckViewer [[1]]\nGizmoduckAuthor [[10, 20, 30]]
Flock follows-rev[80, 90] super_follows-fwd[]
Flock follows-fwd[10, 20, 30] blocks-rev[10, 20, 30]",
                    tweet_keyed("[[1, 2, 3, 5]]")
                ),
            ),
            (
                "c",
                retweet_world(true),
                &[1, 2, 3, 4][..],
                format!(
                    "{}\nGizmoduckViewer [[1]]\nGizmoduckAuthor [[10, 20, 30, 40, 60]]
Flock follows-rev[90] super_follows-fwd[]
Flock follows-fwd[10, 20, 30, 40, 60] blocks-rev[10, 20, 30, 40, 60]
Flock follows-rev[80] super_follows-fwd[]
Flock follows-fwd[] blocks-rev[70]",
                    tweet_keyed("[[1, 2, 3, 4], [5, 6]]")
                ),
            ),
            (
                "c, share without the source author",
                retweet_world(false),
                &[1, 2, 3, 4][..],
                format!(
                    "{}\nGizmoduckViewer [[1]]\nGizmoduckAuthor [[10, 20, 30, 40], [60]]
Flock follows-rev[90] super_follows-fwd[]
Flock follows-fwd[10, 20, 30, 40] blocks-rev[10, 20, 30, 40]
Flock follows-rev[80] super_follows-fwd[]
Flock follows-fwd[60] blocks-rev[60, 70]",
                    tweet_keyed("[[1, 2, 3, 4], [5, 6]]")
                ),
            ),
            (
                "c, source 6 no reply",
                not_a_reply(),
                &[1, 2, 3, 4][..],
                format!(
                    "{}\nGizmoduckViewer [[1]]\nGizmoduckAuthor [[10, 20, 30, 40, 60]]
Flock follows-rev[90] super_follows-fwd[]
Flock follows-fwd[10, 20, 30, 40, 60] blocks-rev[10, 20, 30, 40, 60]
Flock follows-rev[80] super_follows-fwd[]",
                    tweet_keyed("[[1, 2, 3, 4], [5, 6]]")
                ),
            ),
        ];
        for (name, world, tweet_ids, expected) in rows {
            let sources = Arc::new(world);
            let filter_tweets = FilterTweets::new(
                Arc::<InMemorySources>::clone(&sources),
                RuleEngine::for_tests(),
            );
            let outcomes = evaluate_merging_sources(&filter_tweets, request(tweet_ids)).await;
            assert_eq!(batches(&sources), expected, "{name}");
            assert_no_key_asked_twice(&sources, name);
            assert!(
                outcomes
                    .iter()
                    .all(|outcome| matches!(outcome.evaluation, Evaluation::Complete { .. })),
                "{name}"
            );
        }
        for tweet_ids in [&[2][..], &[1, 2, 3, 5]] {
            let plain = Arc::new(retweet_world(true));
            FilterTweets::new(
                Arc::<InMemorySources>::clone(&plain),
                RuleEngine::for_tests(),
            )
            .hydrate(request(tweet_ids))
            .await;
            let merged = Arc::new(retweet_world(true));
            let filter_tweets = FilterTweets::new(
                Arc::<InMemorySources>::clone(&merged),
                RuleEngine::for_tests(),
            );
            evaluate_merging_sources(&filter_tweets, request(tweet_ids)).await;
            assert_eq!(merged.calls(), plain.calls(), "{tweet_ids:?}");
            assert_eq!(batches(&merged), batches(&plain), "{tweet_ids:?}");
        }
    }

    fn assert_no_key_asked_twice(sources: &InMemorySources, name: &str) {
        let mut asked: HashSet<String> = HashSet::new();
        for source in TWEET_KEYED
            .into_iter()
            .chain([Source::GizmoduckViewer, Source::GizmoduckAuthor])
        {
            for key in sources.keys(source).into_iter().flatten() {
                assert!(
                    asked.insert(format!("{source:?} {key}")),
                    "{name}: {source:?} {key}"
                );
            }
        }
        for query in sources.selects().into_iter().flatten() {
            for key in &query.destination_ids {
                let edge = format!("{:?} {:?} {key}", query.graph, query.direction);
                assert!(asked.insert(edge.clone()), "{name}: {edge}");
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Arm {
        Complete,
        Partial,
        NotFound(Lookup),
        Failed(Lookup),
    }

    fn arms(outcomes: &[FilterOutcome]) -> Vec<Arm> {
        outcomes
            .iter()
            .map(|outcome| match outcome.evaluation {
                Evaluation::Complete { .. } => Arm::Complete,
                Evaluation::Partial { .. } => Arm::Partial,
                Evaluation::NotFound(lookup) => Arm::NotFound(lookup),
                Evaluation::Failed(lookup) => Arm::Failed(lookup),
            })
            .collect()
    }

    fn starts(sources: &InMemorySources, t0: tokio::time::Instant) -> Vec<Vec<Duration>> {
        TWEET_KEYED
            .into_iter()
            .chain([
                Source::GizmoduckViewer,
                Source::GizmoduckAuthor,
                Source::Flock,
            ])
            .map(|source| {
                sources
                    .starts(source)
                    .into_iter()
                    .map(|started| started - t0)
                    .collect()
            })
            .collect()
    }

    fn slow_world(share_names_author: bool) -> InMemorySources {
        let ms = Duration::from_millis;
        retweet_world(share_names_author)
            .latency(Source::TesPureCore, ms(10))
            .latency(Source::TesTweet, ms(30))
            .latency(Source::TesConversationControl, ms(20))
            .latency(Source::SafetyLabels, ms(15))
            .latency(Source::GizmoduckViewer, ms(5))
            .latency(Source::GizmoduckAuthor, ms(10))
            .latency(Source::Flock, ms(10))
    }

    #[tokio::test(start_paused = true)]
    async fn requested_calls_start_as_they_do_without_sources() {
        let ms = Duration::from_millis;
        for share_names_author in [true, false] {
            let alone = Arc::new(slow_world(share_names_author));
            let t0 = tokio::time::Instant::now();
            FilterTweets::new(
                Arc::<InMemorySources>::clone(&alone),
                RuleEngine::for_tests(),
            )
            .hydrate(request(&[1, 2, 3, 4]))
            .await;
            let alone = starts(&alone, t0);

            let expanded = Arc::new(slow_world(share_names_author));
            let t0 = tokio::time::Instant::now();
            let filter_tweets = FilterTweets::new(
                Arc::<InMemorySources>::clone(&expanded),
                RuleEngine::for_tests(),
            );
            evaluate_merging_sources(&filter_tweets, request(&[1, 2, 3, 4])).await;
            let elapsed = t0.elapsed();
            let expanded = starts(&expanded, t0);

            for (alone, expanded) in alone.iter().zip(&expanded) {
                assert_eq!(alone[..], expanded[..alone.len()], "{share_names_author}");
            }
            for tweet_keyed in &expanded[..4] {
                assert_eq!(tweet_keyed, &[ms(0), ms(10)], "{share_names_author}");
            }
            let authors = if share_names_author {
                vec![ms(10)]
            } else {
                vec![ms(10), ms(20)]
            };
            assert_eq!(expanded[5], authors, "{share_names_author}");
            assert_eq!(
                expanded[6],
                [ms(10), ms(20), ms(20), ms(30)],
                "{share_names_author}"
            );
            assert_eq!(elapsed, ms(40), "{share_names_author}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_source_failure_fails_only_its_retweets() {
        use crate::hydration::HYDRATION_TIMEOUT;
        use Arm::{Complete, Partial};
        let rows = [
            (
                "source tweet row",
                retweet_world(true).fail_key(Source::TesTweet, 6),
                [
                    Complete,
                    Complete,
                    Complete,
                    Arm::Failed(Lookup::SharedTweet),
                ],
                Duration::ZERO,
            ),
            (
                "source pure core",
                retweet_world(true).fail_key(Source::TesPureCore, 6),
                [
                    Complete,
                    Complete,
                    Complete,
                    Arm::Failed(Lookup::SharedTweet),
                ],
                Duration::ZERO,
            ),
            (
                "source-only author in the shared author call",
                retweet_world(true).fail_key(Source::GizmoduckAuthor, 60),
                [
                    Complete,
                    Complete,
                    Complete,
                    Arm::Failed(Lookup::SharedAuthor),
                ],
                Duration::ZERO,
            ),
            (
                "deleted source",
                retweet_world(true).pure_core(4, retweet(40, 7, 70)),
                [
                    Complete,
                    Complete,
                    Complete,
                    Arm::NotFound(Lookup::SharedTweet),
                ],
                Duration::ZERO,
            ),
            (
                "source reply-root select",
                retweet_world(true).fail_edge(Graph::Blocks, 70),
                [Complete, Complete, Complete, Partial],
                Duration::ZERO,
            ),
            (
                "source root-edge select, the root requested as a Subscribers root",
                retweet_world(true)
                    .control(2, control(ConversationControlArm::Subscribers, 80, &[]))
                    .fail_edge(Graph::Follows, 80),
                [Partial, Complete, Partial, Complete],
                Duration::ZERO,
            ),
            (
                "timed-out source root select",
                retweet_world(true).key_latency(Source::Flock, 80, HYDRATION_TIMEOUT * 2),
                [Partial, Complete, Partial, Complete],
                HYDRATION_TIMEOUT,
            ),
            (
                "timed-out source controls",
                retweet_world(true).key_latency(
                    Source::TesConversationControl,
                    5,
                    HYDRATION_TIMEOUT * 2,
                ),
                [Partial, Complete, Partial, Partial],
                HYDRATION_TIMEOUT,
            ),
        ];
        for (name, world, expected, elapsed) in rows {
            let sources = Arc::new(world);
            let filter_tweets = FilterTweets::new(
                Arc::<InMemorySources>::clone(&sources),
                RuleEngine::for_tests(),
            );
            let t0 = tokio::time::Instant::now();
            let outcomes = evaluate_merging_sources(&filter_tweets, request(&[1, 2, 3, 4])).await;
            assert_eq!(t0.elapsed(), elapsed, "{name}");
            assert_eq!(arms(&outcomes), expected, "{name}");
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Answer {
        Found,
        NotFound,
        Failed,
    }

    fn shared_lookups(
        author: Answer,
        shared_tweet: Answer,
        shared_author: Answer,
        share_names_author: bool,
    ) -> InMemorySources {
        let share = PureCoreData {
            source_user_id: Some(20).filter(|_| share_names_author),
            ..retweet(10, 5, 20)
        };
        let sources = InMemorySources::default().pure_core(1, share);
        let sources = match shared_tweet {
            Answer::Found => sources.tweet(5, 20),
            Answer::NotFound => sources,
            Answer::Failed => sources.tweet(5, 20).fail_key(Source::TesPureCore, 5),
        };
        [(10, author), (20, shared_author)].into_iter().fold(
            sources,
            |sources, (user_id, answer)| match answer {
                Answer::Found => sources.authors(&[user_id]),
                Answer::NotFound => sources,
                Answer::Failed => sources.fail_key(Source::GizmoduckAuthor, user_id),
            },
        )
    }

    async fn retweet_status(sources: InMemorySources, tweet_ids: &[u64]) -> Evaluation {
        let filter_tweets = FilterTweets::new(Arc::new(sources), RuleEngine::for_tests());
        evaluate_merging_sources(&filter_tweets, request(tweet_ids))
            .await
            .remove(0)
            .evaluation
    }

    #[tokio::test]
    async fn a_retweet_resolves_only_when_its_shared_tweet_and_shared_author_answer() {
        use Answer::{Failed, Found, NotFound};
        let judged = Evaluation::Complete { verdict: allow() };
        let both = |expected: Evaluation| [expected.clone(), expected];
        let rows = [
            (Found, Found, Found, both(judged.clone())),
            (
                Found,
                Found,
                NotFound,
                both(Evaluation::NotFound(Lookup::SharedAuthor)),
            ),
            (
                Found,
                Found,
                Failed,
                both(Evaluation::Failed(Lookup::SharedAuthor)),
            ),
            (
                Found,
                NotFound,
                Found,
                both(Evaluation::NotFound(Lookup::SharedTweet)),
            ),
            (
                Found,
                NotFound,
                NotFound,
                both(Evaluation::NotFound(Lookup::SharedTweet)),
            ),
            (
                Found,
                NotFound,
                Failed,
                both(Evaluation::NotFound(Lookup::SharedTweet)),
            ),
            (
                Found,
                Failed,
                Found,
                both(Evaluation::Failed(Lookup::SharedTweet)),
            ),
            (
                Found,
                Failed,
                NotFound,
                [
                    Evaluation::NotFound(Lookup::SharedAuthor),
                    Evaluation::Failed(Lookup::SharedTweet),
                ],
            ),
            (
                Found,
                Failed,
                Failed,
                both(Evaluation::Failed(Lookup::SharedTweet)),
            ),
            (
                Failed,
                Found,
                NotFound,
                both(Evaluation::NotFound(Lookup::SharedAuthor)),
            ),
            (
                Failed,
                Found,
                Failed,
                both(Evaluation::Failed(Lookup::Author)),
            ),
            (
                Failed,
                Failed,
                Found,
                both(Evaluation::Failed(Lookup::SharedTweet)),
            ),
            (
                NotFound,
                Failed,
                Found,
                both(Evaluation::NotFound(Lookup::Author)),
            ),
        ];
        for (author, shared_tweet, shared_author, expected) in rows {
            for (tweet_ids, expected) in [&[1][..], &[1, 5]].into_iter().zip(expected) {
                assert_eq!(
                    retweet_status(
                        shared_lookups(author, shared_tweet, shared_author, true),
                        tweet_ids
                    )
                    .await,
                    expected,
                    "{author:?} {shared_tweet:?} {shared_author:?} {tweet_ids:?}"
                );
            }
        }

        let unnamed = [
            (Found, NotFound, Evaluation::NotFound(Lookup::SharedAuthor)),
            (Found, Failed, Evaluation::Failed(Lookup::SharedAuthor)),
            (Failed, NotFound, Evaluation::Failed(Lookup::SharedTweet)),
        ];
        for (shared_tweet, shared_author, expected) in unnamed {
            assert_eq!(
                retweet_status(
                    shared_lookups(Found, shared_tweet, shared_author, false),
                    &[1]
                )
                .await,
                expected,
                "unnamed {shared_tweet:?} {shared_author:?}"
            );
        }

        let filter_tweets = FilterTweets::new(
            Arc::new(shared_lookups(Found, NotFound, NotFound, true)),
            RuleEngine::for_tests(),
        );
        let outcome = filter_tweets
            .run(FilterRequest {
                rpc: Rpc::FilterTweets,
                ..request(&[1])
            })
            .await
            .outcomes
            .remove(0);
        assert_eq!(outcome.evaluation, judged);
    }

    #[tokio::test(start_paused = true)]
    async fn a_requested_key_asked_by_a_source_batch_shares_its_fate() {
        use Arm::{Complete, Partial};
        let ms = Duration::from_millis;
        let world = || {
            retweet_world(true)
                .control(5, control(ConversationControlArm::Community, 90, &[]))
                .key_latency(Source::TesConversationControl, 2, ms(30))
        };
        for (world, expected) in [
            (world(), [Complete; 4]),
            (
                world().fail_edge(Graph::Follows, 90),
                [Partial, Partial, Partial, Complete],
            ),
        ] {
            let sources = Arc::new(world);
            let filter_tweets = FilterTweets::new(
                Arc::<InMemorySources>::clone(&sources),
                RuleEngine::for_tests(),
            );
            let t0 = tokio::time::Instant::now();
            let outcomes = evaluate_merging_sources(&filter_tweets, request(&[1, 2, 3, 4])).await;
            assert_eq!(arms(&outcomes), expected);
            let root_selects: Vec<(Duration, Vec<u64>)> = sources
                .selects()
                .into_iter()
                .zip(sources.starts(Source::Flock))
                .filter_map(|(queries, started)| {
                    let query = queries.iter().find(|q| q.graph == Graph::Follows)?;
                    (query.direction == EdgeDirection::Reverse)
                        .then(|| (started - t0, query.destination_ids.clone()))
                })
                .collect();
            assert_eq!(root_selects, [(ms(0), vec![90])]);
            assert_eq!(
                sources.keys(Source::TesConversationControl),
                [vec![1, 2, 3, 4], vec![5, 6]]
            );
        }
    }

    #[tokio::test]
    async fn the_source_of_a_failed_retweet_changes_no_verdict() {
        let world = || retweet_world(true).fail_key(Source::TesTweet, 4);
        let judged = |world: InMemorySources| async move {
            let sources = Arc::new(world);
            let filter_tweets = FilterTweets::new(
                Arc::<InMemorySources>::clone(&sources),
                RuleEngine::for_tests(),
            );
            let outcomes = evaluate_merging_sources(&filter_tweets, request(&[1, 2, 3, 4])).await;
            assert_eq!(
                sources.keys(Source::TesPureCore),
                [vec![1, 2, 3, 4], vec![5, 6]]
            );
            outcomes
                .into_iter()
                .map(|outcome| outcome.evaluation)
                .collect::<Vec<_>>()
        };
        let healthy = judged(world()).await;
        assert_eq!(healthy[3], Evaluation::Failed(Lookup::Tweet));
        for source_side in [
            world().user(60, suspended()),
            world().fail_key(Source::TesPureCore, 6),
            world().fail_key(Source::GizmoduckAuthor, 60),
            world().fail_edge(Graph::Blocks, 70),
        ] {
            assert_eq!(judged(source_side).await, healthy);
        }
    }

    #[tokio::test]
    async fn a_follower_of_the_source_author_sees_its_fosnr_notice_on_any_retweet() {
        let fosnr = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                SafetyLabelType::FOSNR_ABUSE_INSULTS.0,
                vf_pb::SafetyLabel::default(),
            )]),
        };
        let sources = InMemorySources::default()
            .pure_core(1, retweet(10, 5, 20))
            .tweet(5, 20)
            .labels(5, fosnr)
            .authors(&[10, 20])
            .edge(Graph::Follows, 1, 20);
        let filter_tweets = FilterTweets::new(Arc::new(sources), RuleEngine::for_tests());
        let outcomes = evaluate_merging_sources(&filter_tweets, request(&[1])).await;
        assert_eq!(
            outcomes[0].evaluation,
            Evaluation::Complete {
                verdict: noticed(
                    true,
                    false,
                    "fosnr_abuse_insults_follower/soft_intervention/abuse"
                ),
            }
        );
    }

    #[tokio::test]
    async fn a_viewer_removed_from_the_community_is_limited_on_any_retweet_of_its_post() {
        let sources = InMemorySources::default()
            .pure_core(1, retweet(10, 5, 20))
            .tweet(5, 20)
            .tweet_features(
                5,
                TweetFeatures {
                    community_id: NonZeroU64::new(500),
                    ..TweetFeatures::default()
                },
            )
            .authors(&[10, 20])
            .removed_from(500);
        let filter_tweets = FilterTweets::new(Arc::new(sources), RuleEngine::for_tests());
        let request = FilterRequest {
            client_capability: ClientCapability {
                community_viewer_removed_limits: true,
                ..ClientCapability::default()
            },
            ..request(&[1, 5])
        };
        let outcomes = evaluate_merging_sources(&filter_tweets, request).await;
        let removed_limit = Evaluation::Complete {
            verdict: limited(
                LimitedEngagementReason::CommunityTweetViewerRemoved,
                "community_tweet_viewer_removed/limited_engagement",
            ),
        };
        assert_eq!(outcomes[0].evaluation, removed_limit);
        assert_eq!(outcomes[1].evaluation, removed_limit);
    }

    #[tokio::test(start_paused = true)]
    async fn a_root_follows_the_viewer_by_the_select_that_asked_it() {
        use ConversationControlArm::{MyNetwork, Subscribers};
        let sources = Arc::new(
            retweet_world(true)
                .control(2, control(MyNetwork, 90, &[]))
                .control(5, control(Subscribers, 90, &[]))
                .edge(Graph::Follows, 90, 1)
                .key_latency(Source::TesConversationControl, 2, Duration::from_millis(30)),
        );
        let filter_tweets = FilterTweets::new(
            Arc::<InMemorySources>::clone(&sources),
            RuleEngine::for_tests(),
        );
        let outcomes = evaluate_merging_sources(&filter_tweets, request(&[1, 2, 3, 4])).await;
        assert!(
            outcomes
                .iter()
                .all(|outcome| matches!(outcome.evaluation, Evaluation::Complete { .. }))
        );
        let root_selects: Vec<String> = batches(&sources)
            .lines()
            .filter(|line| line.contains("super_follows-fwd"))
            .map(str::to_owned)
            .collect();
        assert_eq!(
            root_selects,
            [
                "Flock follows-rev[] super_follows-fwd[90]",
                "Flock follows-rev[90] super_follows-fwd[]",
            ]
        );
        assert!(sources.keys(Source::Wingman).is_empty());
    }

    #[tokio::test]
    async fn each_request_kind_labels_its_phase_samples() {
        use crate::rules::metrics::RetweetSources::{Fetched, InBatch, NoSource};
        let filter_tweets =
            FilterTweets::new(Arc::new(retweet_world(true)), RuleEngine::for_tests());
        for (tweet_ids, expected) in [
            (&[2][..], NoSource),
            (&[1, 2, 3, 5], InBatch),
            (&[1, 2, 3, 4], Fetched),
        ] {
            let hydrated = filter_tweets
                .hydrate_with_retweet_sources(request(tweet_ids))
                .await;
            assert_eq!(hydrated.retweet_sources(), expected, "{tweet_ids:?}");
        }
        let plain = filter_tweets.hydrate(request(&[1, 2, 3, 4])).await;
        assert_eq!(plain.retweet_sources(), NoSource);
        assert_eq!(<&str>::from(NoSource), "none");
    }
}
