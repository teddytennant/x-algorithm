use crate::hydration::sources::Sources;
use crate::hydration::{
    Cause, HydratedTweet, Hydration, HydrationRequest, Hydrators, Lookup, Unresolved,
};
use crate::models::{ClientCapability, Evaluation, HydratedTweetCandidate, RawCandidate, TweetId};
use crate::rules::metrics::{self as ft_metrics, RetweetSources, Rpc};
use crate::rules::{RuleEngine, SafetyLevel};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use xai_visibility_filtering_proto as vf_pb;

pub struct FilterRequest {
    pub viewer_id: Option<u64>,
    pub country_code: Option<String>,
    pub client_capability: ClientCapability,
    pub safety_level: SafetyLevel,
    pub candidates: Vec<RawCandidate>,
    pub rpc: Rpc,
}

#[derive(Clone)]
pub struct FilterOutcome {
    pub tweet_id: TweetId,
    pub source_tweet_id: Option<TweetId>,
    pub evaluation: Evaluation,
    pub safety_labels: Option<vf_pb::SafetyLabelMap>,
}

pub struct FilterResponse {
    pub outcomes: Vec<FilterOutcome>,
}

pub struct FilterTweets {
    sources: Arc<dyn Sources>,
    rule_engine: RuleEngine,
    #[cfg(test)]
    pub(crate) client_capabilities: std::sync::Mutex<Vec<ClientCapability>>,
}

impl FilterTweets {
    pub(crate) fn new(sources: Arc<dyn Sources>, rule_engine: RuleEngine) -> Self {
        Self {
            sources,
            rule_engine,
            #[cfg(test)]
            client_capabilities: std::sync::Mutex::default(),
        }
    }

    pub async fn run(&self, request: FilterRequest) -> FilterResponse {
        let hydrated = self.hydrate(request).await;
        FilterResponse {
            outcomes: self.evaluate_in_request_order(&hydrated),
        }
    }

    pub(crate) async fn hydrate(&self, request: FilterRequest) -> HydratedRequest {
        self.hydrate_request(request, false).await
    }

    pub(crate) async fn hydrate_with_retweet_sources(
        &self,
        request: FilterRequest,
    ) -> HydratedRequest {
        self.hydrate_request(request, true).await
    }

    async fn hydrate_request(
        &self,
        request: FilterRequest,
        is_expanding_retweet_sources: bool,
    ) -> HydratedRequest {
        #[cfg(test)]
        self.client_capabilities
            .lock()
            .unwrap()
            .push(request.client_capability);
        let started = Instant::now();
        let hydration = self
            .rule_engine
            .plan(request.safety_level)
            .hydrate(
                &*self.sources,
                HydrationRequest::new(
                    request.viewer_id,
                    request.country_code,
                    request.client_capability,
                    &request.candidates,
                )
                .with_retweet_sources(is_expanding_retweet_sources),
            )
            .await;
        let hydrated_at = Instant::now();
        let retweet_sources = if !is_expanding_retweet_sources {
            RetweetSources::NoSource
        } else if hydration.has_fetched_sources() {
            RetweetSources::Fetched
        } else if request.candidates.iter().any(|candidate| {
            hydration
                .tweet(candidate.tweet_id)
                .is_some_and(|tweet| tweet.source_to_merge().is_some())
        }) {
            RetweetSources::InBatch
        } else {
            RetweetSources::NoSource
        };
        ft_metrics::record_phase(
            request.rpc,
            "hydration",
            retweet_sources,
            hydrated_at - started,
        );
        HydratedRequest {
            safety_level: request.safety_level,
            rpc: request.rpc,
            retweet_sources,
            candidates: request.candidates,
            hydration,
        }
    }

    pub(crate) fn evaluate_in_request_order(
        &self,
        hydrated: &HydratedRequest,
    ) -> Vec<FilterOutcome> {
        let evaluating = Instant::now();
        let outcomes = hydrated
            .requested_ids()
            .map(|tweet_id| self.outcome(hydrated, tweet_id, None))
            .collect();
        ft_metrics::record_phase(
            hydrated.rpc,
            "post_hydration",
            hydrated.retweet_sources,
            evaluating.elapsed(),
        );
        outcomes
    }

    pub(crate) fn evaluate(
        &self,
        hydrated: &HydratedRequest,
        ids: impl IntoIterator<Item = TweetId>,
        copied_retweets: &HashMap<TweetId, HydratedTweetCandidate>,
    ) -> HashMap<TweetId, FilterOutcome> {
        let evaluating = Instant::now();
        let mut outcomes = HashMap::new();
        for tweet_id in ids {
            outcomes.entry(tweet_id).or_insert_with(|| {
                self.outcome(hydrated, tweet_id, copied_retweets.get(&tweet_id))
            });
        }
        ft_metrics::record_phase(
            hydrated.rpc,
            "post_hydration",
            hydrated.retweet_sources,
            evaluating.elapsed(),
        );
        outcomes
    }

    fn outcome(
        &self,
        hydrated: &HydratedRequest,
        tweet_id: TweetId,
        copied_retweet: Option<&HydratedTweetCandidate>,
    ) -> FilterOutcome {
        let tweet = hydrated.hydration.tweet(tweet_id);
        let evaluation = match tweet {
            Some(HydratedTweet::Resolved {
                candidate,
                has_failed_node,
                ..
            }) => {
                let evaluation = self.rule_engine.evaluate(
                    hydrated.safety_level,
                    hydrated.hydration.viewer(),
                    copied_retweet.unwrap_or(candidate),
                );
                match evaluation {
                    Evaluation::Complete { verdict } if *has_failed_node => Evaluation::Partial {
                        verdict,
                        fail_open_defaults: Hydrators::empty(),
                    },
                    Evaluation::Complete { .. }
                    | Evaluation::Partial { .. }
                    | Evaluation::NotFound(_)
                    | Evaluation::Failed(_) => evaluation,
                }
            }
            Some(HydratedTweet::Unresolved {
                reason: Unresolved { lookup, cause },
                ..
            }) => match cause {
                Cause::NotFound => Evaluation::NotFound(*lookup),
                Cause::Failed => Evaluation::Failed(*lookup),
            },
            None => Evaluation::Failed(Lookup::Tweet),
        };
        FilterOutcome {
            tweet_id,
            source_tweet_id: tweet.and_then(HydratedTweet::source_tweet_id),
            evaluation,
            safety_labels: tweet
                .and_then(HydratedTweet::safety_labels)
                .map(|labels| vf_pb::SafetyLabelMap::clone(labels)),
        }
    }
}

pub(crate) struct HydratedRequest {
    safety_level: SafetyLevel,
    rpc: Rpc,
    retweet_sources: RetweetSources,
    candidates: Vec<RawCandidate>,
    hydration: Hydration,
}

impl HydratedRequest {
    pub(crate) fn hydration(&self) -> &Hydration {
        &self.hydration
    }

    #[cfg(test)]
    pub(crate) fn retweet_sources(&self) -> RetweetSources {
        self.retweet_sources
    }

    pub(crate) fn requested_ids(&self) -> impl ExactSizeIterator<Item = TweetId> + '_ {
        self.candidates.iter().map(|candidate| candidate.tweet_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeQuery, Graph};
    use crate::hydration::plan::Source;
    use crate::hydration::sources::{control, Fault, InMemorySources};
    use crate::hydration::{author_fallback_cache, tweet_fallback_cache, Hydrator};
    use crate::models::{LimitedEngagementReason, TweetFeatures, Verdict};
    use crate::rules::fixtures::{allow, dropped, limited};
    use xai_core_entities::entities::{
        ConversationControlArm, GizmoduckUser, GizmoduckUserResult, PureCoreData, UserResponseState,
    };
    use xai_visibility_filtering::models::FilteredReason;

    fn candidate(tweet_id: u64, author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id: author_id,
        }
    }

    fn user(user_id: u64, response_state: UserResponseState) -> GizmoduckUserResult {
        GizmoduckUserResult {
            user: Some(GizmoduckUser {
                user_id,
                ..Default::default()
            }),
            response_state: Some(response_state),
        }
    }

    fn service(sources: &Arc<InMemorySources>) -> FilterTweets {
        FilterTweets::new(
            Arc::<InMemorySources>::clone(sources),
            RuleEngine::for_tests(),
        )
    }

    async fn home_hydration(
        sources: &Arc<InMemorySources>,
        viewer_id: Option<u64>,
        ids: &[u64],
    ) -> Vec<Evaluation> {
        service(sources)
            .run(FilterRequest {
                viewer_id,
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHomeHydration,
                candidates: ids.iter().map(|&id| candidate(id, None)).collect(),
                rpc: Rpc::FilterTweets,
            })
            .await
            .outcomes
            .into_iter()
            .map(|outcome| outcome.evaluation)
            .collect()
    }

    #[tokio::test(start_paused = true)]
    async fn pure_core_timeout_fails_every_candidate_at_hydration_timeout() {
        let sources = Arc::new(
            InMemorySources::default()
                .tweet_features(1, TweetFeatures::default())
                .tweet_features(2, TweetFeatures::default())
                .authors(&[20])
                .fault(Source::TesPureCore, Fault::Hangs),
        );
        let started = tokio::time::Instant::now();
        let response = tokio::time::timeout(
            crate::hydration::HYDRATION_TIMEOUT * 2,
            service(&sources).run(FilterRequest {
                viewer_id: Some(50),
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![candidate(1, None), candidate(2, Some(20))],
                rpc: Rpc::FilterTweets,
            }),
        )
        .await
        .unwrap();
        assert_eq!(started.elapsed(), crate::hydration::HYDRATION_TIMEOUT);
        assert_eq!(
            response
                .outcomes
                .into_iter()
                .map(|outcome| outcome.evaluation)
                .collect::<Vec<_>>(),
            vec![Evaluation::Failed(Lookup::Tweet); 2]
        );
    }

    #[tokio::test]
    async fn home_hydration_limits_posts_whose_author_or_direct_reply_root_blocks_the_viewer() {
        let reply = |author_id, in_reply_to_tweet_id, in_reply_to_user_id| PureCoreData {
            author_id,
            conversation_id: Some(100),
            in_reply_to_tweet_id: Some(in_reply_to_tweet_id),
            in_reply_to_user_id: Some(in_reply_to_user_id),
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .pure_core(1, reply(10, 100, 30))
                .pure_core(2, reply(20, 101, 40))
                .tweet(3, 10)
                .authors(&[10, 20])
                .edge(Graph::Blocks, 20, 50)
                .edge(Graph::Blocks, 30, 50),
        );
        let service = &service(&sources);
        let verdicts = |viewer_id| async move {
            service
                .run(FilterRequest {
                    viewer_id,
                    country_code: None,
                    client_capability: ClientCapability::default(),
                    safety_level: SafetyLevel::TimelineHomeHydration,
                    candidates: vec![candidate(1, None), candidate(2, None), candidate(3, None)],
                    rpc: Rpc::FilterTweets,
                })
                .await
                .outcomes
                .into_iter()
                .map(|outcome| outcome.evaluation)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            verdicts(Some(50)).await,
            vec![
                Evaluation::Complete {
                    verdict: limited(
                        LimitedEngagementReason::RootAuthorBlockedViewer,
                        "blocked_viewer/limited_engagement/root_author_blocked_viewer",
                    ),
                },
                Evaluation::Complete {
                    verdict: limited(
                        LimitedEngagementReason::BlockedViewer,
                        "blocked_viewer/limited_engagement",
                    ),
                },
                Evaluation::Complete { verdict: allow() },
            ]
        );
        assert_eq!(
            verdicts(None).await,
            vec![Evaluation::Complete { verdict: allow() }; 3]
        );
        assert_eq!(
            sources.selects(),
            [vec![
                EdgeQuery::forward(Graph::Follows, vec![10, 20]),
                EdgeQuery::reverse(Graph::Blocks, vec![10, 20, 30]),
            ]]
        );
    }

    #[tokio::test]
    async fn trusted_friends_posts_show_to_list_members_and_owners_and_a_failed_lookup_drops_them()
    {
        let trusted_friends = |list_id| TweetFeatures {
            trusted_friends_list_id: Some(list_id),
            ..Default::default()
        };
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, trusted_friends(7))
                .tweet(2, 11)
                .tweet_features(2, trusted_friends(7))
                .tweet(3, 12)
                .tweet_features(3, trusted_friends(9))
                .tweet(4, 13)
                .authors(&[10, 11, 12, 13])
                .trusted_friend(7, 50)
        };
        let shown = || Evaluation::Complete { verdict: allow() };
        let trusted_friends_drop = || Evaluation::Complete {
            verdict: dropped(
                FilteredReason::UnspecifiedReason,
                "trusted_friends_tweet/drop/unspecified",
            ),
        };

        let healthy = Arc::new(world());
        assert_eq!(
            home_hydration(&healthy, Some(50), &[1, 2, 3, 4]).await,
            [shown(), shown(), trusted_friends_drop(), shown()]
        );
        assert_eq!(healthy.keys(Source::TrustedFriends), [vec![7, 9]]);

        let failed = world().fault(Source::TrustedFriends, Fault::Fails);
        assert_eq!(
            home_hydration(&Arc::new(failed), Some(50), &[1, 4]).await,
            [trusted_friends_drop(), shown()]
        );

        for (viewer_id, ids) in [(None, &[1, 2][..]), (Some(50), &[4][..])] {
            let sources = Arc::new(world());
            home_hydration(&sources, viewer_id, ids).await;
            assert!(!sources.calls().contains(&Source::TrustedFriends));
        }
    }

    #[tokio::test]
    async fn local_posts_limit_viewers_outside_their_place_and_a_failed_lookup_limits_none() {
        const A: u64 = 0xa000_0000_0000_0001;
        const B: u64 = 0xb000_0000_0000_0002;
        let local = |place| TweetFeatures {
            narrowcast_place_id: Some(place),
            ..Default::default()
        };
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, local(A))
                .control(1, control(ConversationControlArm::Local, 10, &[]))
                .tweet(2, 11)
                .tweet_features(2, local(B))
                .control(2, control(ConversationControlArm::Local, 11, &[]))
                .tweet(3, 12)
                .authors(&[10, 11, 12])
        };
        let shown = || Evaluation::Complete { verdict: allow() };
        let local_limit = || Evaluation::Complete {
            verdict: limited(
                LimitedEngagementReason::LocalTweet,
                "local_tweet/limited_engagement",
            ),
        };

        let located = Arc::new(world().located_in(B));
        assert_eq!(
            home_hydration(&located, Some(50), &[1, 2, 3]).await,
            [local_limit(), shown(), shown()]
        );
        assert_eq!(located.keys(Source::UserLocation), [vec![A, B]]);

        assert_eq!(
            home_hydration(&Arc::new(world()), Some(50), &[1, 2]).await,
            [local_limit(), local_limit()]
        );

        let failed = world()
            .located_in(B)
            .fault(Source::UserLocation, Fault::Fails);
        assert_eq!(
            home_hydration(&Arc::new(failed), Some(50), &[1, 3]).await,
            [
                Evaluation::Partial {
                    verdict: allow(),
                    fail_open_defaults: Hydrators::of(Hydrator::OutsideNarrowcastPlace),
                },
                shown(),
            ]
        );
    }

    #[tokio::test]
    async fn a_circle_row_served_from_the_fallback_cache_still_asks_the_trusted_friends_list() {
        let trusted_friends = |list_id| TweetFeatures {
            trusted_friends_list_id: Some(list_id),
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .with_tweet_cache(tweet_fallback_cache(8))
                .tweet(1, 10)
                .tweet_features(1, trusted_friends(7))
                .tweet(2, 11)
                .tweet_features(2, trusted_friends(9))
                .authors(&[10, 11])
                .trusted_friend(7, 50),
        );
        let judged = || async {
            service(&sources)
                .run(FilterRequest {
                    viewer_id: Some(50),
                    country_code: None,
                    client_capability: ClientCapability::default(),
                    safety_level: SafetyLevel::TimelineHomeHydration,
                    candidates: vec![candidate(1, None), candidate(2, None)],
                    rpc: Rpc::FilterTweets,
                })
                .await
                .outcomes
                .into_iter()
                .map(|outcome| outcome.evaluation)
                .collect::<Vec<_>>()
        };
        let expected = [
            Evaluation::Complete { verdict: allow() },
            Evaluation::Complete {
                verdict: dropped(
                    FilteredReason::UnspecifiedReason,
                    "trusted_friends_tweet/drop/unspecified",
                ),
            },
        ];

        assert_eq!(judged().await, expected);
        sources.break_source(Source::TesTweet, Fault::Fails);
        assert_eq!(judged().await, expected);
        assert_eq!(
            sources.keys(Source::TrustedFriends),
            [vec![7, 9], vec![7, 9]]
        );
    }

    #[tokio::test]
    async fn pure_core_alone_says_a_post_is_a_retweet() {
        let retweet = PureCoreData {
            author_id: 10,
            source_tweet_id: Some(5),
            source_user_id: Some(20),
            ..Default::default()
        };
        let sources = Arc::new(
            InMemorySources::default()
                .pure_core(1, retweet)
                .tweet(2, 10)
                .authors(&[10])
                .edge(Graph::MuteRetweets, 50, 10),
        );
        let outcomes = service(&sources)
            .run(FilterRequest {
                viewer_id: Some(50),
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![candidate(1, None), candidate(2, None)],
                rpc: Rpc::FilterTweets,
            })
            .await
            .outcomes;
        assert_eq!(
            outcomes
                .into_iter()
                .map(|outcome| outcome.evaluation)
                .collect::<Vec<_>>(),
            vec![
                Evaluation::Complete {
                    verdict: dropped(
                        FilteredReason::UnspecifiedReason,
                        "viewer_mutes_retweets/drop/unspecified",
                    ),
                },
                Evaluation::Complete { verdict: allow() },
            ]
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum PureCore {
        Found,
        NotFound,
        Failed,
        FailedCacheHit,
    }

    #[derive(Clone, Copy, Debug)]
    enum TweetRow {
        Found,
        NotFound,
        Failed,
        FailedCacheHit,
    }

    #[derive(Clone, Copy, Debug)]
    enum Author {
        Found,
        Partial,
        NotFound,
        Failed,
        FailedCacheHit,
        PartialCacheHit,
    }

    fn lookups(
        pure_core: PureCore,
        tweet_row: TweetRow,
        author: Author,
    ) -> (InMemorySources, Vec<(Source, Fault)>) {
        let mut broken = Vec::new();
        let sources = InMemorySources::default()
            .with_tweet_cache(tweet_fallback_cache(8))
            .with_author_cache(author_fallback_cache(8));
        let sources = match pure_core {
            PureCore::Found => sources.tweet(1, 10),
            PureCore::NotFound => sources,
            PureCore::Failed => sources.tweet(1, 10).fail_key(Source::TesPureCore, 1),
            PureCore::FailedCacheHit => {
                broken.push((Source::TesPureCore, Fault::Fails));
                sources.tweet(1, 10)
            }
        };
        let nullcast = TweetFeatures {
            is_nullcast: true,
            ..Default::default()
        };
        let sources = match tweet_row {
            TweetRow::Found => sources.tweet_features(1, nullcast),
            TweetRow::NotFound => sources.without_tweet_row(1),
            TweetRow::Failed => sources.fail_key(Source::TesTweet, 1),
            TweetRow::FailedCacheHit => {
                broken.push((Source::TesTweet, Fault::Fails));
                sources.tweet_features(1, nullcast)
            }
        };
        let sources = match author {
            Author::Found => sources.authors(&[10]),
            Author::Partial => sources.user(10, user(10, UserResponseState::Partial)),
            Author::NotFound => sources,
            Author::Failed => sources.fail_key(Source::GizmoduckAuthor, 10),
            Author::FailedCacheHit => {
                broken.push((Source::GizmoduckAuthor, Fault::Fails));
                sources.authors(&[10])
            }
            Author::PartialCacheHit => {
                broken.push((Source::GizmoduckAuthor, Fault::AnswersPartial));
                sources.authors(&[10])
            }
        };
        (sources, broken)
    }

    #[tokio::test]
    async fn a_post_is_judged_only_when_its_tweet_and_author_lookups_answer() {
        let nullcast_drop = dropped(FilteredReason::TweetIsNullcast, "nullcasted_tweet/drop");
        let judged = Evaluation::Complete {
            verdict: nullcast_drop.clone(),
        };
        let not_found = Evaluation::NotFound;
        let failed = Evaluation::Failed;
        let rows = [
            (
                PureCore::Found,
                TweetRow::Found,
                Author::Found,
                judged.clone(),
            ),
            (
                PureCore::Found,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::Found,
                TweetRow::Failed,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::NotFound,
                TweetRow::Found,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::NotFound,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::NotFound,
                TweetRow::Failed,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::Failed,
                TweetRow::Found,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::Failed,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::Failed,
                TweetRow::Failed,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::Found,
                Author::Found,
                judged.clone(),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::NotFound,
                Author::Found,
                not_found(Lookup::Tweet),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::Failed,
                Author::Found,
                failed(Lookup::Tweet),
            ),
            (
                PureCore::Found,
                TweetRow::FailedCacheHit,
                Author::Found,
                judged.clone(),
            ),
            (
                PureCore::FailedCacheHit,
                TweetRow::FailedCacheHit,
                Author::FailedCacheHit,
                judged.clone(),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::Partial,
                Evaluation::Partial {
                    verdict: nullcast_drop,
                    fail_open_defaults: Hydrators::of(Hydrator::AuthorSafety),
                },
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::NotFound,
                not_found(Lookup::Author),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::Failed,
                failed(Lookup::Author),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::FailedCacheHit,
                judged.clone(),
            ),
            (
                PureCore::Found,
                TweetRow::Found,
                Author::PartialCacheHit,
                judged,
            ),
        ];
        for (pure_core, tweet_row, author, expected) in rows {
            for request_author_id in [None, Some(10)] {
                let (sources, broken) = lookups(pure_core, tweet_row, author);
                let sources = Arc::new(sources);
                let run = || async {
                    service(&sources)
                        .run(FilterRequest {
                            viewer_id: Some(50),
                            country_code: None,
                            client_capability: ClientCapability::default(),
                            safety_level: SafetyLevel::TimelineHome,
                            candidates: vec![candidate(1, request_author_id)],
                            rpc: Rpc::FilterTweets,
                        })
                        .await
                        .outcomes
                        .remove(0)
                        .evaluation
                };
                run().await;
                for (source, fault) in broken {
                    sources.break_source(source, fault);
                }
                assert_eq!(
                    run().await,
                    expected,
                    "{pure_core:?} {tweet_row:?} {author:?} {request_author_id:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_node_failed_for_one_request_author_of_a_repeated_tweet_fails_every_occurrence() {
        let sources = Arc::new(
            InMemorySources::default()
                .tweet(1, 10)
                .authors(&[11])
                .fail_key(Source::GizmoduckAuthor, 10),
        );
        let outcomes = service(&sources)
            .run(FilterRequest {
                viewer_id: Some(50),
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![candidate(1, Some(10)), candidate(1, Some(11))],
                rpc: Rpc::FilterTweets,
            })
            .await
            .outcomes;
        assert_eq!(
            outcomes
                .into_iter()
                .map(|outcome| outcome.evaluation)
                .collect::<Vec<_>>(),
            vec![Evaluation::Failed(Lookup::Author); 2]
        );
    }

    #[tokio::test]
    async fn run_preserves_order_duplicates_unresolved_tweets_and_labels() {
        let labels = vf_pb::SafetyLabelMap {
            labels: HashMap::from([(999_999, vf_pb::SafetyLabel::default())]),
        };
        let sources = Arc::new(
            InMemorySources::default()
                .tweet(2, 20)
                .tweet_features(1, TweetFeatures::default())
                .fail_key(Source::TesPureCore, 1)
                .authors(&[20])
                .labels(1, Default::default())
                .labels(2, labels),
        );
        let response = service(&sources)
            .run(FilterRequest {
                viewer_id: None,
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHome,
                candidates: vec![
                    candidate(2, Some(20)),
                    candidate(1, None),
                    candidate(2, Some(20)),
                ],
                rpc: Rpc::FilterTweets,
            })
            .await;

        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.tweet_id)
                .collect::<Vec<_>>(),
            vec![TweetId(2), TweetId(1), TweetId(2)]
        );
        assert_eq!(
            response
                .outcomes
                .iter()
                .map(|outcome| outcome.evaluation.clone())
                .collect::<Vec<_>>(),
            vec![
                Evaluation::Complete { verdict: allow() },
                Evaluation::Failed(Lookup::Tweet),
                Evaluation::Complete { verdict: allow() },
            ]
        );
        assert_eq!(
            response.outcomes[1].evaluation.verdict(),
            &Verdict::lookup_failed()
        );
        assert!(response
            .outcomes
            .iter()
            .all(|outcome| outcome.safety_labels.is_some()));
        assert!(!response.outcomes[0]
            .safety_labels
            .as_ref()
            .unwrap()
            .labels
            .is_empty());
        assert_eq!(
            response.outcomes[0].safety_labels,
            response.outcomes[2].safety_labels
        );
    }
}
