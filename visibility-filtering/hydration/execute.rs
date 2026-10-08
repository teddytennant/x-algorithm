use crate::hydration::batch::{Hydrated, HydrationBatch, RawHydrationBatch};
use crate::hydration::decode::author::{AuthorColumn, DecodedAuthor};
use crate::hydration::decode::tweet::{PureCoreColumn, TweetRowColumn};
use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::fallback_cache::{Column, FallbackCache};
use crate::hydration::metrics::{
    self, record_batch_size, record_expanded_batch, record_flock_missing_keys,
    record_keyed_hydrator_request, record_trusted_friends_answers, record_viewer_country,
    record_wingman_second_degree,
};
use crate::hydration::plan::{Edge, Group, Source};
use crate::hydration::sources::Sources;
use crate::hydration::store::{CallRequest, Landing, Store};
use crate::hydration::{Hydration, HydrationPlan, HydrationRequest, HYDRATION_TIMEOUT};
use crate::models::{ArticleLifecycle, CommunityModeration, PureCore, TweetFeatures};
use crate::rules::SafetyLevel;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use rustc_hash::{FxHashMap, FxHashSet};
use std::convert::identity;
use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use xai_core_entities::entities::ConversationControl;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_rpc::WithBudget;

#[derive(Debug, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum ViewerCountry {
    NoAllowedList,
    Found,
    NoRow,
    Failed,
}

pub(super) enum Reply {
    PureCores(RawHydrationBatch<PureCore>),
    Tweets(RawHydrationBatch<TweetFeatures>),
    Controls(RawHydrationBatch<ConversationControl>),
    Labels(RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>),
    Viewer(RawHydrationBatch<DecodedViewer>),
    Authors(RawHydrationBatch<DecodedAuthor>),
    Select(Vec<RawHydrationBatch<bool>>),
    Edge(Edge, RawHydrationBatch<bool>),
    ViewerCountry(RawHydrationBatch<Arc<str>>),
    CommunityModerations(RawHydrationBatch<CommunityModeration>),
    CommunityModerators(RawHydrationBatch<bool>),
    CommunityViewerRemovals(RawHydrationBatch<bool>),
    ArticleLifecycles(RawHydrationBatch<ArticleLifecycle>),
}

type InFlight<'p> = Pin<Box<dyn Future<Output = (CallRequest<'p>, Reply)> + Send + 'p>>;

struct Timed {
    client: String,
    method: String,
    level: SafetyLevel,
    candidate_count_by_claimed_key: FxHashMap<u64, usize>,
}

impl Timed {
    async fn run<V, A: AsRef<[RawHydrationBatch<V>]>>(
        &self,
        call: impl Future<Output = A>,
        timed_out: impl FnOnce(RawHydrationBatch<V>) -> A,
    ) -> A {
        let start = Instant::now();
        let answer = call
            .with_budget(HYDRATION_TIMEOUT)
            .await
            .unwrap_or_else(|_| {
                let claimed = self.candidate_count_by_claimed_key.keys().copied();
                timed_out(HydrationBatch::timed_out(claimed))
            });
        record_keyed_hydrator_request(
            &self.client,
            &self.method,
            self.level,
            &self.candidate_count_by_claimed_key,
            &answer,
            start.elapsed().as_secs_f64() * 1000.0,
        );
        answer
    }

    fn keyed<'p, V, G>(
        self,
        call: CallRequest<'p>,
        get: G,
        after: impl FnOnce(RawHydrationBatch<V>) -> RawHydrationBatch<V> + Send + 'p,
        reply: fn(RawHydrationBatch<V>) -> Reply,
    ) -> InFlight<'p>
    where
        V: Send + 'p,
        G: for<'c> FnOnce(&'c CallRequest<'p>) -> BoxFuture<'c, RawHydrationBatch<V>> + Send + 'p,
    {
        Box::pin(async move {
            let answer = self.run(get(&call), identity).await;
            (call, reply(after(answer)))
        })
    }
}

impl HydrationPlan {
    pub(crate) async fn hydrate(
        &self,
        sources: &dyn Sources,
        request: HydrationRequest<'_>,
    ) -> Hydration {
        let started = Instant::now();
        let mut store = Store::new(
            self,
            request.viewer_id,
            request.client_capability,
            request.raw_candidates,
            request.is_expanding_retweet_sources,
        );
        let mut running: FuturesUnordered<InFlight<'_>> = FuturesUnordered::new();
        let inputless = || self.groups().filter(|g| g.input.is_none());
        let mut ready: Vec<&Group> = inputless().collect();
        loop {
            while let Some(group) = ready.pop() {
                if let Some(call) = store.offer(group) {
                    running.extend(self.send(call, sources));
                }
            }
            let Some((call, reply)) = running.next().await else {
                break;
            };
            let group = call.group;
            if store.land(&call, reply, started.elapsed()) == Landing::SourcesJoined {
                ready.extend(inputless());
            }
            ready.extend(self.readers(group));
        }
        if store.is_country_lookup_skipped() {
            record_viewer_country(ViewerCountry::NoAllowedList.into(), self.level());
        }
        metrics::record_tes_join_latency(
            self.level(),
            store
                .tweets_elapsed
                .map_or(store.core_elapsed, |tweets| tweets.max(store.core_elapsed)),
        );
        store.assemble(request)
    }

    fn send<'p>(
        &'p self,
        mut call: CallRequest<'p>,
        sources: &'p dyn Sources,
    ) -> Option<InFlight<'p>> {
        let level = self.level();
        let group = call.group;
        let (client, method) = group.label();
        if let Some(size) = call.batch_size {
            record_batch_size(&client, size);
        }
        if !call.is_first {
            record_expanded_batch(&client, &method, call.key_count);
        }
        let timed = Timed {
            client,
            method,
            level,
            candidate_count_by_claimed_key: mem::take(&mut call.candidate_count_by_claimed_key),
        };
        let in_flight: InFlight<'p> = match group.source {
            Source::TesPureCore => timed.keyed(
                call,
                |call| sources.pure_cores(&call.keys),
                move |pure_cores| fall_back::<PureCoreColumn>(sources.tweet_cache(), pure_cores),
                Reply::PureCores,
            ),
            Source::TesTweet => timed.keyed(
                call,
                |call| sources.tweets(&call.keys),
                move |tweets| fall_back::<TweetRowColumn>(sources.tweet_cache(), tweets),
                Reply::Tweets,
            ),
            Source::TesConversationControl => timed.keyed(
                call,
                |call| sources.conversation_controls(&call.keys),
                identity,
                Reply::Controls,
            ),
            Source::SafetyLabels => timed.keyed(
                call,
                |call| sources.safety_labels(&call.keys),
                identity,
                Reply::Labels,
            ),
            Source::GizmoduckViewer => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |_| sources.viewer(viewer_id, group.fields()),
                    identity,
                    Reply::Viewer,
                )
            }
            Source::GizmoduckAuthor => timed.keyed(
                call,
                |call| sources.users(&call.keys, group.fields()),
                move |authors| fall_back::<AuthorColumn>(sources.author_cache(), authors),
                Reply::Authors,
            ),
            Source::Flock => {
                let viewer_id = call.viewer_id?;
                Box::pin(async move {
                    let select = async {
                        let edges = sources.select_edges(viewer_id, &call.queries).await;
                        let (edges, missing) = missing_sets_read_no_edge(edges);
                        record_flock_missing_keys(&timed.client, &timed.method, level, missing);
                        edges
                    };
                    let per_query = |timed_out| vec![timed_out; call.queries.len()];
                    let edges = timed.run(select, per_query).await;
                    (call, Reply::Select(edges))
                })
            }
            Source::ViewerCountry => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |_| sources.viewer_country(viewer_id),
                    move |country| {
                        let result = match country.hydrated(&viewer_id) {
                            Some(Hydrated::Found(_)) => ViewerCountry::Found,
                            Some(Hydrated::NotFound) => ViewerCountry::NoRow,
                            _ => ViewerCountry::Failed,
                        };
                        record_viewer_country(result.into(), level);
                        country
                    },
                    Reply::ViewerCountry,
                )
            }
            Source::Wingman => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.second_degree(viewer_id, &call.keys),
                    move |answers| {
                        record_holds(answers, |in_network, not_in_network| {
                            record_wingman_second_degree(in_network, not_in_network, level);
                        })
                    },
                    |answers| Reply::Edge(Edge::SecondDegree, answers),
                )
            }
            Source::TrustedFriends => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.trusted_friends(viewer_id, &call.keys),
                    move |answers| {
                        record_holds(answers, |member_or_owner, neither| {
                            record_trusted_friends_answers(member_or_owner, neither, level);
                        })
                    },
                    |answers| Reply::Edge(Edge::TrustedFriends, answers),
                )
            }
            Source::UserLocation => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.outside_places(viewer_id, &call.keys),
                    identity,
                    |answers| Reply::Edge(Edge::OutsidePlace, answers),
                )
            }
            Source::CommunityModeration => timed.keyed(
                call,
                |call| sources.community_moderations(&call.community_posts),
                identity,
                Reply::CommunityModerations,
            ),
            Source::CommunityModerator => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.community_moderators(viewer_id, &call.keys),
                    identity,
                    Reply::CommunityModerators,
                )
            }
            Source::CommunityViewerRemoved => {
                let viewer_id = call.viewer_id?;
                timed.keyed(
                    call,
                    move |call| sources.community_viewer_removals(viewer_id, &call.keys),
                    identity,
                    Reply::CommunityViewerRemovals,
                )
            }
            Source::ArticleLifecycle => timed.keyed(
                call,
                |call| sources.article_lifecycles(&call.keys),
                identity,
                Reply::ArticleLifecycles,
            ),
        };
        Some(in_flight)
    }
}

fn fall_back<C: Column>(
    cache: Option<&FallbackCache<C::Entry>>,
    batch: RawHydrationBatch<C::Value>,
) -> RawHydrationBatch<C::Value> {
    match cache {
        Some(cache) => cache.resolve_hydration_batch::<C>(batch),
        None => batch,
    }
}

fn missing_sets_read_no_edge(
    edges: Vec<RawHydrationBatch<bool>>,
) -> (Vec<RawHydrationBatch<bool>>, usize) {
    let mut missing = FxHashSet::default();
    let edges = edges
        .into_iter()
        .map(|edge| {
            let mut answers = edge.into_hydrated();
            for (destination, answer) in &mut answers {
                if let Hydrated::Partial(holds) = *answer {
                    *answer = Hydrated::Found(holds);
                    missing.insert(*destination);
                }
            }
            HydrationBatch::from_hydrated(answers)
        })
        .collect();
    (edges, missing.len())
}

fn record_holds(
    answers: RawHydrationBatch<bool>,
    record: impl FnOnce(usize, usize),
) -> RawHydrationBatch<bool> {
    let answers = answers.into_hydrated();
    let answered = |holds: bool| {
        answers
            .values()
            .filter(|answer| answer.value() == Some(&holds))
            .count()
    };
    record(answered(true), answered(false));
    HydrationBatch::from_hydrated(answers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeDirection, EdgeQuery, Graph};
    use crate::hydration::decode::author::fallback_cache;
    use crate::hydration::decode::tweet::tweet_fallback_cache;
    use crate::hydration::sources::{control, suspended, Fault, InMemorySources};
    use crate::hydration::{Cause, HydratedTweet, Hydrator, Hydrators, Lookup, Unresolved};
    use crate::models::{
        ArticleLifecycle, ClientCapability, Evaluation, HydratedTweetCandidate,
        LimitedEngagementReason, RawCandidate, TweetId, Viewer, ViewerFeatures, ViewerProfile,
    };
    use crate::rules::fixtures::{allow, limited};
    use crate::rules::{RuleEngine, SafetyLevel};
    use std::collections::{HashMap, HashSet};
    use std::num::NonZeroU64;
    use xai_core_entities::entities::{
        ConversationControlArm, ExtendedProfile, GizmoduckUser, GizmoduckUserResult, PureCoreData,
        UserResponseState,
    };
    use xai_core_entities::gizmoduck_client::QueryFields;

    const VIEWER: u64 = 50;

    fn raw(tweet_id: u64, request_author_id: Option<u64>) -> RawCandidate {
        RawCandidate {
            tweet_id: TweetId(tweet_id),
            request_author_id,
        }
    }

    struct InRequestOrder {
        viewer_features: ViewerFeatures,
        candidates: Vec<HydratedTweetCandidate>,
        safety_labels: HashMap<TweetId, Arc<vf_pb::SafetyLabelMap>>,
        failed_ids: HashSet<TweetId>,
        unresolved: HashMap<TweetId, Unresolved>,
        failed_nodes: Vec<Result<Hydrators, Unresolved>>,
    }

    fn in_request_order(hydration: &Hydration, raw: &[RawCandidate]) -> InRequestOrder {
        let tweets = || {
            raw.iter()
                .filter_map(|c| Some((c.tweet_id, hydration.tweet(c.tweet_id)?)))
        };
        InRequestOrder {
            viewer_features: hydration.viewer().clone(),
            candidates: tweets()
                .filter_map(|(_, tweet)| tweet.candidate().cloned())
                .collect(),
            safety_labels: tweets()
                .filter_map(|(id, tweet)| Some((id, Arc::clone(tweet.safety_labels()?))))
                .collect(),
            failed_ids: tweets()
                .filter(|(_, tweet)| {
                    matches!(
                        tweet,
                        HydratedTweet::Resolved {
                            has_failed_node: true,
                            ..
                        }
                    )
                })
                .map(|(id, _)| id)
                .collect(),
            unresolved: tweets()
                .filter_map(|(id, tweet)| match tweet {
                    HydratedTweet::Resolved { .. } => None,
                    HydratedTweet::Unresolved { reason, .. } => Some((id, *reason)),
                })
                .collect(),
            failed_nodes: tweets()
                .map(|(_, tweet)| match tweet {
                    HydratedTweet::Resolved { candidate, .. } => Ok(candidate.failed),
                    HydratedTweet::Unresolved { reason, .. } => Err(*reason),
                })
                .collect(),
        }
    }

    async fn hydrate(
        sources: &InMemorySources,
        level: SafetyLevel,
        viewer_id: Option<u64>,
        raw: &[RawCandidate],
    ) -> InRequestOrder {
        let hydration = RuleEngine::for_tests()
            .plan(level)
            .hydrate(
                sources,
                HydrationRequest::new(
                    viewer_id,
                    Some("US".into()),
                    ClientCapability {
                        community_viewer_removed_limits: true,
                        ..ClientCapability::default()
                    },
                    raw,
                ),
            )
            .await;
        in_request_order(&hydration, raw)
    }

    fn tweet_failed() -> Unresolved {
        Unresolved {
            lookup: Lookup::Tweet,
            cause: Cause::Failed,
        }
    }

    fn ids(ids: &[u64]) -> HashSet<TweetId> {
        ids.iter().copied().map(TweetId).collect()
    }

    fn exclusive_tweet() -> TweetFeatures {
        TweetFeatures {
            exclusive_conversation_author_id: Some(30),
            ..Default::default()
        }
    }

    fn article_tweet(article_id: u64) -> TweetFeatures {
        TweetFeatures {
            article_id: NonZeroU64::new(article_id),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn failed_ids_reports_exactly_the_candidates_each_node_flags() {
        use ConversationControlArm::{Co, Subscribers};
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let rows = [
            ("healthy home", TimelineHome, world(), vec![], vec![]),
            (
                "healthy home hydration",
                TimelineHomeHydration,
                world(),
                vec![],
                vec![],
            ),
            (
                "failed pure core",
                TimelineHome,
                world().fault(Source::TesPureCore, Fault::Fails),
                vec![],
                vec![(TweetId(1), tweet_failed()), (TweetId(2), tweet_failed())],
            ),
            (
                "incomplete author",
                TimelineHome,
                world().user(
                    10,
                    GizmoduckUserResult {
                        response_state: Some(UserResponseState::Failed),
                        ..suspended()
                    },
                ),
                vec![],
                vec![(
                    TweetId(1),
                    Unresolved {
                        lookup: Lookup::Author,
                        cause: Cause::Failed,
                    },
                )],
            ),
            (
                "failed author-keyed select",
                TimelineHome,
                world().fail_graph(Graph::Mutes),
                vec![1, 2],
                vec![],
            ),
            (
                "failed blocked-by select",
                TimelineHomeHydration,
                world().fail_graph(Graph::Blocks),
                vec![1, 2],
                vec![],
            ),
            (
                "failed tweet row",
                TimelineHome,
                world().fail_key(Source::TesTweet, 1),
                vec![],
                vec![(TweetId(1), tweet_failed())],
            ),
            (
                "failed label lookup",
                TimelineHome,
                world().fail_key(Source::SafetyLabels, 1),
                vec![1],
                vec![],
            ),
            (
                "failed exclusive select",
                TimelineHome,
                world()
                    .tweet_features(1, exclusive_tweet())
                    .fail_graph(Graph::SuperFollows),
                vec![1],
                vec![],
            ),
            (
                "failed root-edge select",
                TimelineHomeHydration,
                world()
                    .control(1, control(Subscribers, 30, &[]))
                    .fail_graph(Graph::SuperFollows),
                vec![1],
                vec![],
            ),
            (
                "failed conversation-control row",
                TimelineHomeHydration,
                world().fail_key(Source::TesConversationControl, 1),
                vec![1],
                vec![],
            ),
            (
                "failed country lookup",
                TimelineHomeHydration,
                world()
                    .control(1, control(Co, 30, &["us"]))
                    .control(2, control(Co, 30, &[]))
                    .fault(Source::ViewerCountry, Fault::Fails),
                vec![1],
                vec![],
            ),
            (
                "failed viewer",
                TimelineHome,
                world().fault(Source::GizmoduckViewer, Fault::Fails),
                vec![1, 2],
                vec![],
            ),
            (
                "failed article lifecycle",
                TimelineHomeHydration,
                world()
                    .tweet_features(1, article_tweet(70))
                    .fault(Source::ArticleLifecycle, Fault::Fails),
                vec![1],
                vec![],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, sources, failed, unresolved) in rows {
            let hydrated = hydrate(&sources, level, Some(VIEWER), &raw).await;
            assert_eq!(
                (hydrated.failed_ids, hydrated.unresolved),
                (ids(&failed), unresolved.into_iter().collect()),
                "{name}"
            );
        }
    }

    #[tokio::test]
    async fn each_candidate_carries_the_nodes_that_failed_for_it() {
        use ConversationControlArm::MyNetwork;
        use Hydrator::{
            BlockedByAuthor, Blocks, Follows, MuteRetweets, Mutes, RootFollowsViewer,
            RootFollowsViewerSecondDegree,
        };
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .authors(&[10, 20])
        };
        let retweet = PureCoreData {
            author_id: 10,
            source_tweet_id: Some(5),
            source_user_id: Some(30),
            ..Default::default()
        };
        let relationships = Hydrators::of(Follows).with(Blocks).with(Mutes);
        let rows = [
            (
                "failed tweet row",
                TimelineHome,
                Some(VIEWER),
                world()
                    .tweet_features(2, exclusive_tweet())
                    .fail_key(Source::TesTweet, 1),
                [Err(tweet_failed()), Ok(Hydrators::empty())],
            ),
            (
                "failed root edge on a MyNetwork reply",
                TimelineHomeHydration,
                Some(VIEWER),
                world()
                    .control(1, control(MyNetwork, 30, &[]))
                    .fail_graph(Graph::Follows),
                [
                    Ok(Hydrators::of(RootFollowsViewer)
                        .with(RootFollowsViewerSecondDegree)
                        .with(Follows)
                        .with(BlockedByAuthor)),
                    Ok(Hydrators::of(Follows).with(BlockedByAuthor)),
                ],
            ),
            (
                "failed pure core, authors from the request",
                TimelineHome,
                Some(VIEWER),
                world().fault(Source::TesPureCore, Fault::Fails),
                [Err(tweet_failed()); 2],
            ),
            (
                "failed relationships select on a retweet and an original",
                TimelineHome,
                Some(VIEWER),
                world().pure_core(1, retweet).fail_graph(Graph::Mutes),
                [Ok(relationships.with(MuteRetweets)), Ok(relationships)],
            ),
        ];
        let raw = [raw(1, Some(10)), raw(2, Some(20))];
        for (name, level, viewer_id, sources, expected) in rows {
            let hydrated = hydrate(&sources, level, viewer_id, &raw).await;
            assert_eq!(hydrated.failed_nodes, expected, "{name}");
        }
    }

    #[tokio::test]
    async fn safety_labels_carry_only_the_tweets_whose_labels_were_found() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet(2, 20)
            .tweet(3, 30)
            .labels(2, Default::default())
            .fail_key(Source::SafetyLabels, 1);
        let raw = [raw(1, None), raw(2, None), raw(3, None)];
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(
            hydrated
                .safety_labels
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            ids(&[2])
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_timed_out_label_lookup_fails_every_tweet() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .fault(Source::SafetyLabels, Fault::Hangs);
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &[raw(1, None)]).await;
        assert_eq!(hydrated.failed_ids, ids(&[1]));
        assert!(hydrated.safety_labels.is_empty());
    }

    fn author_keyed(sources: &InMemorySources) -> bool {
        sources.calls().contains(&Source::GizmoduckAuthor) || !sources.selects().is_empty()
    }

    #[tokio::test(start_paused = true)]
    async fn author_keyed_calls_wait_for_pure_core() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .fault(Source::TesPureCore, Fault::Hangs);
        let raw = [raw(1, None)];
        let hydration = hydrate(&sources, SafetyLevel::TimelineHome, Some(VIEWER), &raw);
        tokio::pin!(hydration);
        let early = tokio::time::timeout(HYDRATION_TIMEOUT / 2, &mut hydration).await;
        assert!(early.is_err());
        assert!(!author_keyed(&sources));
        let hydrated = hydration.await;
        assert!(!author_keyed(&sources));
        assert!(hydrated.candidates.is_empty());
        assert_eq!(
            hydrated.unresolved,
            HashMap::from([(TweetId(1), tweet_failed())])
        );
    }

    #[tokio::test(start_paused = true)]
    async fn author_calls_do_not_wait_for_tweets() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended())
            .edge(Graph::Follows, VIEWER, 10)
            .fault(Source::TesTweet, Fault::Hangs);
        let raw = [raw(1, None)];
        let started = tokio::time::Instant::now();
        let hydration = hydrate(&sources, SafetyLevel::TimelineHome, Some(VIEWER), &raw);
        tokio::pin!(hydration);
        let early = tokio::time::timeout(HYDRATION_TIMEOUT / 2, &mut hydration).await;
        assert!(early.is_err());
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10]]);
        assert_eq!(sources.selects().len(), 1);

        let hydrated = hydration.await;
        assert_eq!(started.elapsed(), HYDRATION_TIMEOUT);
        assert_eq!(
            hydrated.unresolved,
            HashMap::from([(TweetId(1), tweet_failed())])
        );
    }

    fn sorted(mut calls: Vec<Source>) -> Vec<String> {
        let mut names: Vec<String> = calls.drain(..).map(|s| format!("{s:?}")).collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn empty_key_sets_and_logged_out_viewers_send_no_call() {
        let raw = [raw(1, None)];
        let logged_out = InMemorySources::default().tweet(1, 10);
        hydrate(&logged_out, SafetyLevel::TimelineHomeHydration, None, &raw).await;
        assert_eq!(
            sorted(logged_out.calls()),
            [
                "GizmoduckAuthor",
                "SafetyLabels",
                "TesConversationControl",
                "TesPureCore",
                "TesTweet"
            ]
        );

        let logged_in = InMemorySources::default().tweet(1, 10);
        hydrate(
            &logged_in,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert!(!logged_in.calls().contains(&Source::ViewerCountry));
        assert!(!logged_in.calls().contains(&Source::UserLocation));
        assert_eq!(
            logged_in.selects(),
            [vec![
                EdgeQuery {
                    graph: Graph::Follows,
                    direction: EdgeDirection::Forward,
                    destination_ids: vec![10],
                },
                EdgeQuery {
                    graph: Graph::Blocks,
                    direction: EdgeDirection::Reverse,
                    destination_ids: vec![10],
                },
            ]]
        );
    }

    #[tokio::test]
    async fn only_retweeters_are_asked_for_the_mute_retweets_edge() {
        let relationships = |authors: &[u64], retweeters: &[u64]| {
            vec![
                EdgeQuery::forward(Graph::Follows, authors.to_vec()),
                EdgeQuery::forward(Graph::Blocks, authors.to_vec()),
                EdgeQuery::forward(Graph::Mutes, authors.to_vec()),
                EdgeQuery::forward(Graph::MuteRetweets, retweeters.to_vec()),
            ]
        };
        let retweet = PureCoreData {
            author_id: 10,
            source_tweet_id: Some(5),
            source_user_id: Some(20),
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .pure_core(1, retweet)
            .tweet(2, 30)
            .tweet(5, 20);
        let raw_candidates = [raw(1, None), raw(2, None)];
        let request = HydrationRequest::new(
            Some(VIEWER),
            None,
            ClientCapability::default(),
            &raw_candidates,
        )
        .with_retweet_sources(true);
        RuleEngine::for_tests()
            .plan(SafetyLevel::TimelineHome)
            .hydrate(&sources, request)
            .await;
        assert_eq!(sources.selects(), [relationships(&[10, 20, 30], &[10])]);

        let originals = InMemorySources::default().tweet(2, 30);
        hydrate(
            &originals,
            SafetyLevel::TimelineHome,
            Some(VIEWER),
            &[raw(2, None)],
        )
        .await;
        assert_eq!(originals.selects(), [relationships(&[30], &[])]);
    }

    #[tokio::test]
    async fn filter_all_calls_pure_core_only_and_keeps_the_request_side_viewer() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet_features(1, exclusive_tweet());
        let hydrated = hydrate(
            &sources,
            SafetyLevel::FilterAll,
            Some(VIEWER),
            &[raw(1, None)],
        )
        .await;
        assert_eq!(sources.calls(), [Source::TesPureCore]);
        assert_eq!(
            hydrated.viewer_features.viewer,
            Viewer::LoggedIn {
                id: VIEWER,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: false,
            }
        );
        assert_eq!(hydrated.viewer_features.country_code.as_deref(), Some("us"));
        assert_eq!(hydrated.candidates[0].author_id, 10);
        assert!(hydrated.safety_labels.is_empty());
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_failed_viewer_call_fails_both_viewer_nodes_unverified() {
        use crate::hydration::plan::HydrationPlan;
        use xai_core_entities::entities::{Label, Labels};
        use xai_x_thrift::user_labels::LabelValue;
        let viewer_nodes = Hydrators::of(Hydrator::ViewerProfile).with(Hydrator::ViewerLabels);
        let labeled = GizmoduckUser {
            labels: Labels {
                labels: vec![Label {
                    label_value: LabelValue::AGE_VERIFIED_18.0,
                    created_at_msec: 0,
                }],
            },
            ..Default::default()
        };
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .viewer(VIEWER, labeled)
            .fault(Source::GizmoduckViewer, Fault::Fails);
        let raw = [raw(1, None)];
        let hydration = HydrationPlan::new(SafetyLevel::TimelineHomeHydration, viewer_nodes)
            .hydrate(
                &sources,
                HydrationRequest::new(Some(VIEWER), None, ClientCapability::default(), &raw),
            )
            .await;
        let hydrated = in_request_order(&hydration, &raw);
        assert_eq!(hydrated.candidates[0].failed, viewer_nodes);
        assert_eq!(
            hydrated.viewer_features.viewer,
            Viewer::LoggedIn {
                id: VIEWER,
                profile: ViewerProfile::default(),
                has_age_verified_18_label: false,
            }
        );
    }

    #[tokio::test]
    async fn a_level_that_plans_the_viewer_profile_decodes_it() {
        let sources = InMemorySources::default().viewer(
            VIEWER,
            GizmoduckUser {
                extended_profile: Some(ExtendedProfile {
                    age_in_years: Some(30),
                }),
                ..Default::default()
            },
        );
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, Some(VIEWER), &[]).await;
        let Viewer::LoggedIn { profile, .. } = hydrated.viewer_features.viewer else {
            panic!("a logged-in request stays logged in")
        };
        assert_ne!(profile, ViewerProfile::default());
    }

    #[tokio::test]
    async fn exclusive_edges_dedup_conversation_authors() {
        let sources = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 20)
                .tweet(3, 40)
                .authors(&[10, 20, 40])
                .tweet_features(1, exclusive_tweet())
                .tweet_features(2, exclusive_tweet())
                .edge(Graph::SuperFollows, VIEWER, 30)
        };
        let raw = [raw(1, None), raw(2, None), raw(1, None), raw(3, Some(40))];
        for viewer_id in [Some(VIEWER), None] {
            let sources = sources();
            let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, viewer_id, &raw).await;
            assert_eq!(sources.keys(Source::TesPureCore), [vec![1, 2, 3]]);
            assert_eq!(sources.keys(Source::TesTweet), [vec![1, 2, 3]]);
            let exclusive = (Some(30), viewer_id.is_some());
            assert_eq!(
                hydrated
                    .candidates
                    .iter()
                    .map(|c| (
                        c.tweet_features.exclusive_conversation_author_id,
                        c.edges.contains(Hydrator::SuperFollowsExclusive)
                    ))
                    .collect::<Vec<_>>(),
                [exclusive, exclusive, exclusive, (None, false)]
            );
            let super_follows: Vec<Vec<u64>> = sources
                .selects()
                .into_iter()
                .flatten()
                .filter(|query| query.graph == Graph::SuperFollows)
                .map(|query| query.destination_ids)
                .collect();
            let expected: &[Vec<u64>] = if viewer_id.is_some() {
                &[vec![30]]
            } else {
                &[]
            };
            assert_eq!(super_follows, expected);
        }
    }

    #[tokio::test]
    async fn a_super_follows_key_two_groups_read_is_asked_once() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .tweet(2, 20)
            .authors(&[10, 20])
            .tweet_features(1, exclusive_tweet())
            .control(2, control(ConversationControlArm::Subscribers, 30, &[]))
            .edge(Graph::SuperFollows, VIEWER, 30);
        let raw = [raw(1, None), raw(2, None)];
        let hydrated = hydrate(
            &sources,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        let super_follows: Vec<Vec<u64>> = sources
            .selects()
            .into_iter()
            .flatten()
            .filter(|query| query.graph == Graph::SuperFollows)
            .map(|query| query.destination_ids)
            .collect();
        assert_eq!(super_follows, [vec![30]]);
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (
                    c.edges.contains(Hydrator::SuperFollowsExclusive),
                    c.edges.contains(Hydrator::SuperFollowsRoot)
                ))
                .collect::<Vec<_>>(),
            [(true, false), (false, true)]
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    fn root_edges(sources: &InMemorySources) -> Vec<Vec<EdgeQuery>> {
        sources
            .selects()
            .into_iter()
            .filter(|queries| {
                queries.iter().any(|query| {
                    query.graph == Graph::Follows && query.direction == EdgeDirection::Reverse
                })
            })
            .collect()
    }

    #[tokio::test]
    async fn one_select_carries_both_root_edges_and_a_failure_fails_every_tweet_it_keyed() {
        use ConversationControlArm::{ByInvitation, Community, MyNetwork, Subscribers};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .authors(&[10])
                .control(1, control(Community, 30, &[]))
                .control(2, control(MyNetwork, 30, &[]))
                .control(3, control(Subscribers, 40, &[]))
                .control(4, control(ByInvitation, 40, &[]))
                .edge(Graph::Follows, 30, VIEWER)
                .edge(Graph::SuperFollows, VIEWER, 40)
        };
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let facts = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| {
                    (
                        c.edges.contains(Hydrator::RootFollowsViewer),
                        c.edges.contains(Hydrator::SuperFollowsRoot),
                    )
                })
                .collect::<Vec<_>>()
        };

        let healthy = world();
        let hydrated = hydrate(
            &healthy,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert_eq!(
            root_edges(&healthy),
            [vec![
                EdgeQuery {
                    graph: Graph::Follows,
                    direction: EdgeDirection::Reverse,
                    destination_ids: vec![30],
                },
                EdgeQuery {
                    graph: Graph::SuperFollows,
                    direction: EdgeDirection::Forward,
                    destination_ids: vec![40],
                },
            ]]
        );
        assert_eq!(
            facts(&hydrated),
            [(true, false), (true, false), (false, true), (false, false)]
        );
        assert!(hydrated.failed_ids.is_empty());

        let failed = world().fail_graph(Graph::SuperFollows);
        let hydrated = hydrate(
            &failed,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert_eq!(facts(&hydrated), [(false, false); 4]);
        assert_eq!(hydrated.failed_ids, ids(&[1, 2, 3]));

        let logged_out = world();
        let hydrated = hydrate(&logged_out, SafetyLevel::TimelineHomeHydration, None, &raw).await;
        assert!(root_edges(&logged_out).is_empty());
        assert_eq!(facts(&hydrated), [(false, false); 4]);
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_query_missing_from_the_select_answer_reads_no_edge_and_fails_no_candidate() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .edge(Graph::Follows, VIEWER, 10)
            .edge(Graph::Mutes, VIEWER, 10)
            .miss_graph(Graph::Mutes);
        let hydrated = hydrate(
            &sources,
            SafetyLevel::TimelineHome,
            Some(VIEWER),
            &[raw(1, None)],
        )
        .await;
        assert_eq!(
            hydrated.candidates[0].edges,
            Hydrators::of(Hydrator::Follows)
        );
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn wingman_asks_once_for_my_network_roots_the_followed_by_edge_answered_no() {
        use ConversationControlArm::{Community, MyNetwork};
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .tweet(5, 10)
                .tweet(6, 10)
                .authors(&[10])
                .control(1, control(Community, 30, &[]))
                .control(2, control(MyNetwork, 30, &[]))
                .control(3, control(MyNetwork, 60, &[]))
                .control(4, control(MyNetwork, 60, &[]))
                .control(5, control(MyNetwork, 70, &[]))
                .control(6, control(Community, 80, &[]))
                .edge(Graph::Follows, 30, VIEWER)
                .second_degree_path(60, VIEWER)
        };
        let raw = [1, 2, 3, 4, 5, 6].map(|id| raw(id, None));
        let second_degree = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| c.edges.contains(Hydrator::RootFollowsViewerSecondDegree))
                .collect::<Vec<_>>()
        };
        let level = SafetyLevel::TimelineHomeHydration;

        let healthy = world();
        let hydrated = hydrate(&healthy, level, Some(VIEWER), &raw).await;
        assert_eq!(healthy.keys(Source::Wingman), [vec![60, 70]]);
        assert_eq!(
            second_degree(&hydrated),
            [false, false, true, true, false, false]
        );
        assert!(hydrated.failed_ids.is_empty());

        let failed = world().fault(Source::Wingman, Fault::Fails);
        let hydrated = hydrate(&failed, level, Some(VIEWER), &raw).await;
        assert_eq!(second_degree(&hydrated), [false; 6]);
        assert!(hydrated.failed_ids.is_empty());

        for (sources, viewer_id, raw) in [
            (world().fail_graph(Graph::Follows), Some(VIEWER), &raw[..]),
            (world(), None, &raw[..]),
            (world(), Some(VIEWER), &raw[..2]),
        ] {
            let hydrated = hydrate(&sources, level, viewer_id, raw).await;
            assert!(sources.keys(Source::Wingman).is_empty());
            assert_eq!(second_degree(&hydrated), vec![false; raw.len()]);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_root_edge_call_started_before_pure_core_lands_fails_its_candidate_when_it_times_out()
    {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .control(1, control(ConversationControlArm::Subscribers, 40, &[]))
            .edge(Graph::SuperFollows, VIEWER, 40)
            .fault(Source::TesPureCore, Fault::Delays(HYDRATION_TIMEOUT / 2))
            .hang_graph(Graph::SuperFollows);
        let raw = [raw(1, None)];
        let hydration = hydrate(
            &sources,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        );
        tokio::pin!(hydration);
        let early = tokio::time::timeout(HYDRATION_TIMEOUT / 4, &mut hydration).await;
        assert!(early.is_err());
        assert!(sources
            .selects()
            .concat()
            .contains(&EdgeQuery::forward(Graph::SuperFollows, vec![40])));
        assert!(!sources.calls().contains(&Source::GizmoduckAuthor));

        let hydrated = hydration.await;
        assert!(!hydrated.candidates[0]
            .edges
            .contains(Hydrator::SuperFollowsRoot));
        assert_eq!(
            hydrated.candidates[0].failed,
            Hydrators::of(Hydrator::SuperFollowsRoot)
        );
        assert_eq!(hydrated.failed_ids, ids(&[1]));
    }

    #[tokio::test(start_paused = true)]
    async fn a_timed_out_root_edge_call_fails_its_candidate_whichever_input_lands_first() {
        let level = SafetyLevel::TimelineHomeHydration;
        for delayed in [Source::TesConversationControl, Source::TesPureCore] {
            let sources = InMemorySources::default()
                .tweet(1, 10)
                .authors(&[10])
                .control(1, control(ConversationControlArm::Subscribers, 40, &[]))
                .edge(Graph::SuperFollows, VIEWER, 40)
                .fault(delayed, Fault::Delays(HYDRATION_TIMEOUT / 2))
                .hang_graph(Graph::SuperFollows);
            let hydrated = hydrate(&sources, level, Some(VIEWER), &[raw(1, None)]).await;
            let candidate = &hydrated.candidates[0];
            assert_eq!(
                (
                    candidate.failed,
                    RuleEngine::for_tests().evaluate(level, &hydrated.viewer_features, candidate),
                ),
                (
                    Hydrators::of(Hydrator::SuperFollowsRoot),
                    Evaluation::Partial {
                        verdict: limited(
                            LimitedEngagementReason::ConversationControl,
                            "limit_replies_subscribers/limited_engagement/conversation_control",
                        ),
                        fail_open_defaults: Hydrators::of(Hydrator::SuperFollowsRoot),
                    },
                ),
                "{delayed:?} lands last"
            );
        }
    }

    #[tokio::test]
    async fn one_lifecycle_call_asks_each_distinct_article_once() {
        use ArticleLifecycle::{Draft, Published};
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        for viewer_id in [Some(VIEWER), None] {
            let sources = InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .tweet(3, 10)
                .tweet(4, 10)
                .authors(&[10])
                .tweet_features(1, article_tweet(70))
                .tweet_features(2, article_tweet(70))
                .tweet_features(3, article_tweet(80))
                .lifecycle(70, Draft)
                .lifecycle(80, Published);
            let hydrated = hydrate(
                &sources,
                SafetyLevel::TimelineHomeHydration,
                viewer_id,
                &raw,
            )
            .await;
            assert_eq!(
                sources.keys(Source::ArticleLifecycle),
                [vec![70, 80]],
                "{viewer_id:?}"
            );
            assert_eq!(
                hydrated
                    .candidates
                    .iter()
                    .map(|c| c.article_lifecycle)
                    .collect::<Vec<_>>(),
                [Some(Draft), Some(Draft), Some(Published), None],
                "{viewer_id:?}"
            );
        }
    }

    #[tokio::test]
    async fn one_country_lookup_reaches_every_co_tweet_that_needs_it() {
        use ConversationControlArm::Co;
        let raw = [raw(1, None), raw(2, None)];
        let world = |countries: &[&str]| {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet(2, 10)
                .authors(&[10])
                .control(1, control(Co, 30, countries))
                .control(2, control(Co, 30, countries))
                .country(VIEWER, "us")
        };
        let country = |hydrated: &InRequestOrder| {
            hydrated
                .candidates
                .iter()
                .map(|c| {
                    c.conversation_control
                        .as_ref()
                        .unwrap()
                        .viewer_country
                        .as_deref()
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        };

        let listed = world(&["us"]);
        let hydrated = hydrate(
            &listed,
            SafetyLevel::TimelineHomeHydration,
            Some(VIEWER),
            &raw,
        )
        .await;
        assert_eq!(listed.keys(Source::ViewerCountry), [vec![VIEWER]]);
        assert_eq!(
            country(&hydrated),
            [Some("us".to_owned()), Some("us".to_owned())]
        );

        for (sources, viewer_id) in [(world(&[]), Some(VIEWER)), (world(&["us"]), None)] {
            let hydrated = hydrate(
                &sources,
                SafetyLevel::TimelineHomeHydration,
                viewer_id,
                &raw,
            )
            .await;
            assert!(sources.keys(Source::ViewerCountry).is_empty());
            assert_eq!(country(&hydrated), [None, None]);
        }
    }

    #[tokio::test]
    async fn authors_share_one_key_and_a_missing_user_is_complete() {
        let sources = InMemorySources::default().tweet(1, 10).tweet(2, 10);
        let raw = [raw(1, None), raw(2, None)];
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(sources.keys(Source::GizmoduckAuthor), [vec![10]]);
        assert!(hydrated
            .candidates
            .iter()
            .all(|c| !c.author_features.is_suspended));
        assert!(hydrated.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn repeated_ids_share_the_first_occurrences_resolution() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended());
        let raw = [raw(1, Some(10)), raw(1, Some(20)), raw(1, None)];
        let hydrated = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (c.author_id, c.author_features.is_suspended))
                .collect::<Vec<_>>(),
            [(10, true); 3]
        );
        assert!(hydrated.failed_ids.is_empty());
        assert!(hydrated.unresolved.is_empty());
    }

    #[tokio::test]
    async fn the_author_cache_serves_the_last_known_author_when_the_call_fails() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .user(10, suspended())
            .with_author_cache(fallback_cache(8));
        let raw = [raw(1, None)];
        let first = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert!(first.candidates[0].author_features.is_suspended);

        sources.break_source(Source::GizmoduckAuthor, Fault::Fails);
        let second = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert!(second.candidates[0].author_features.is_suspended);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn the_tweet_cache_serves_the_last_known_core_when_the_call_fails() {
        let sources = InMemorySources::default()
            .tweet(1, 10)
            .authors(&[10])
            .with_tweet_cache(tweet_fallback_cache(8));
        let raw = [raw(1, None)];
        let first = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(first.candidates[0].author_id, 10);

        sources.break_source(Source::TesPureCore, Fault::Fails);
        let second = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(second.candidates[0].author_id, 10);
        assert!(second.failed_ids.is_empty());
    }

    #[tokio::test]
    async fn a_pure_core_not_found_is_not_served_when_the_call_fails() {
        let sources = InMemorySources::default()
            .tweet_features(1, TweetFeatures::default())
            .with_tweet_cache(tweet_fallback_cache(8));
        let raw = [raw(1, Some(10))];
        let first = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(
            first.unresolved,
            HashMap::from([(
                TweetId(1),
                Unresolved {
                    lookup: Lookup::Tweet,
                    cause: Cause::NotFound,
                }
            )])
        );

        sources.break_source(Source::TesPureCore, Fault::Fails);
        let second = hydrate(&sources, SafetyLevel::TimelineHome, None, &raw).await;
        assert_eq!(
            second.unresolved,
            HashMap::from([(TweetId(1), tweet_failed())])
        );
    }

    #[tokio::test]
    async fn gizmoduck_calls_ask_for_every_field_any_level_reads() {
        use QueryFields::{ACCOUNT, EXTENDED_PROFILE, LABELS, SAFETY};
        for (level, viewer_fields) in [
            (
                SafetyLevel::TimelineHome,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY],
            ),
            (
                SafetyLevel::TimelineHomeHydration,
                vec![ACCOUNT, EXTENDED_PROFILE, SAFETY, LABELS],
            ),
        ] {
            let sources = InMemorySources::default().tweet(1, 10);
            hydrate(&sources, level, Some(VIEWER), &[raw(1, None)]).await;
            assert_eq!(
                sources.fields(Source::GizmoduckViewer),
                [viewer_fields],
                "{level:?}"
            );
            assert_eq!(
                sources.fields(Source::GizmoduckAuthor),
                [vec![SAFETY, LABELS]],
                "{level:?}"
            );
        }
    }

    #[test]
    fn a_first_call_sent_after_sources_join_records_the_requests_own_batch_size() {
        use crate::models::AuthorId;
        let engine = RuleEngine::for_tests();
        let plan = engine.plan(SafetyLevel::TimelineHomeHydration);
        let raw = [raw(1, None)];
        let mut store = Store::new(plan, Some(VIEWER), ClientCapability::default(), &raw, true);
        let call = store.offer(plan.groups().next().unwrap()).unwrap();
        let retweet = PureCore {
            author_id: AuthorId(10),
            source_tweet_id: Some(TweetId(5)),
            source_author_id: Some(AuthorId(20)),
            direct_reply_root_author_id: None,
        };
        let cores =
            HydrationBatch::from_results([1], HashMap::from([(1, Ok::<_, ()>(Some(retweet)))]));
        assert_eq!(
            store.land(&call, Reply::PureCores(cores), std::time::Duration::ZERO),
            Landing::SourcesJoined
        );
        for source in [Source::SafetyLabels, Source::Flock] {
            let group = plan.groups().find(|group| group.source == source).unwrap();
            let call = store.offer(group).unwrap();
            assert_eq!(
                (call.is_first, call.batch_size),
                (true, Some(1)),
                "{source:?}"
            );
        }
    }

    fn call_labels(sources: &InMemorySources) -> Vec<String> {
        let mut labels: Vec<String> = sources
            .calls()
            .into_iter()
            .filter(|source| *source != Source::Flock)
            .map(|source| format!("{source:?}"))
            .chain(sources.selects().into_iter().map(|queries| {
                let graphs: Vec<&str> = queries.iter().map(|q| <&str>::from(q.graph)).collect();
                format!("Flock {}", graphs.join(","))
            }))
            .collect();
        labels.sort();
        labels
    }

    #[tokio::test(start_paused = true)]
    async fn a_hung_source_delays_only_the_calls_waiting_on_it() {
        use ConversationControlArm::{Co, Community, MyNetwork};
        use SafetyLevel::{TimelineHome, TimelineHomeHydration};
        let world = || {
            InMemorySources::default()
                .pure_core(
                    1,
                    PureCoreData {
                        author_id: 10,
                        conversation_id: Some(100),
                        in_reply_to_tweet_id: Some(100),
                        in_reply_to_user_id: Some(30),
                        ..Default::default()
                    },
                )
                .tweet(2, 20)
                .tweet_features(1, exclusive_tweet())
                .control(1, control(Community, 30, &[]))
                .control(2, control(Co, 30, &["us"]))
                .tweet(3, 20)
                .tweet_features(
                    3,
                    TweetFeatures {
                        article_id: article_tweet(70).article_id,
                        trusted_friends_list_id: Some(7),
                        narrowcast_place_id: Some(0xa000_0000_0000_0001),
                        ..community_tweet(500)
                    },
                )
                .community_moderation(3, HIDDEN)
                .control(3, control(MyNetwork, 40, &[]))
                .lifecycle(70, ArticleLifecycle::Published)
        };
        let rows: [(SafetyLevel, Source, &[&str]); 21] = [
            (
                TimelineHome,
                Source::TesPureCore,
                &[
                    "Flock follows,blocks,mutes,mute_retweets",
                    "GizmoduckAuthor",
                ],
            ),
            (
                TimelineHome,
                Source::TesTweet,
                &["Flock super_follows", "TrustedFriends"],
            ),
            (TimelineHome, Source::SafetyLabels, &[]),
            (TimelineHome, Source::GizmoduckViewer, &[]),
            (TimelineHome, Source::GizmoduckAuthor, &[]),
            (TimelineHome, Source::Flock, &[]),
            (
                TimelineHomeHydration,
                Source::TesPureCore,
                &[
                    "CommunityModeration",
                    "CommunityModerator",
                    "Flock follows,blocks",
                    "GizmoduckAuthor",
                ],
            ),
            (
                TimelineHomeHydration,
                Source::TesTweet,
                &[
                    "ArticleLifecycle",
                    "CommunityModeration",
                    "CommunityModerator",
                    "CommunityViewerRemoved",
                    "Flock super_follows",
                    "TrustedFriends",
                    "UserLocation",
                ],
            ),
            (
                TimelineHomeHydration,
                Source::TesConversationControl,
                &["Flock follows,super_follows", "ViewerCountry", "Wingman"],
            ),
            (TimelineHomeHydration, Source::SafetyLabels, &[]),
            (TimelineHomeHydration, Source::GizmoduckViewer, &[]),
            (TimelineHomeHydration, Source::GizmoduckAuthor, &[]),
            (TimelineHomeHydration, Source::Flock, &["Wingman"]),
            (TimelineHomeHydration, Source::ViewerCountry, &[]),
            (TimelineHomeHydration, Source::Wingman, &[]),
            (
                TimelineHomeHydration,
                Source::CommunityModeration,
                &["CommunityModerator"],
            ),
            (TimelineHomeHydration, Source::CommunityModerator, &[]),
            (TimelineHomeHydration, Source::CommunityViewerRemoved, &[]),
            (TimelineHomeHydration, Source::ArticleLifecycle, &[]),
            (TimelineHomeHydration, Source::TrustedFriends, &[]),
            (TimelineHomeHydration, Source::UserLocation, &[]),
        ];
        let raw = [raw(1, None), raw(2, None), raw(3, None)];
        for (level, hung, waiting) in rows {
            let healthy = world();
            hydrate(&healthy, level, Some(VIEWER), &raw).await;
            let every_call = call_labels(&healthy);
            let groups = RuleEngine::for_tests().plan(level).groups().count();
            assert_eq!(every_call.len(), groups, "{level:?}: {every_call:?}");
            assert!(
                waiting
                    .iter()
                    .all(|call| every_call.iter().any(|c| c == call)),
                "{level:?} {hung:?}: {every_call:?}"
            );

            let sources = world().fault(hung, Fault::Hangs);
            let started = tokio::time::Instant::now();
            let hydration = hydrate(&sources, level, Some(VIEWER), &raw);
            tokio::pin!(hydration);
            let early = tokio::time::timeout(HYDRATION_TIMEOUT / 2, &mut hydration).await;
            assert!(early.is_err(), "{level:?} {hung:?}");
            let expected: Vec<String> = every_call
                .iter()
                .filter(|call| !waiting.contains(&call.as_str()))
                .cloned()
                .collect();
            assert_eq!(call_labels(&sources), expected, "{level:?} {hung:?}");
            hydration.await;
            assert_eq!(started.elapsed(), HYDRATION_TIMEOUT, "{level:?} {hung:?}");
        }
    }

    const HIDDEN: CommunityModeration = CommunityModeration {
        is_hidden: true,
        is_author_removed: false,
    };

    fn community_tweet(community_id: u64) -> TweetFeatures {
        use std::num::NonZeroU64;
        TweetFeatures {
            community_id: NonZeroU64::new(community_id),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn only_others_community_posts_are_looked_up_and_only_moderated_ones_ask_the_viewer() {
        use crate::hydration::community_source::CommunityPost;
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, community_tweet(500))
                .community_moderation(1, HIDDEN)
                .tweet(2, 10)
                .tweet_features(2, community_tweet(501))
                .tweet(3, 10)
                .tweet(4, VIEWER)
                .tweet_features(4, community_tweet(500))
                .community_moderation(4, HIDDEN)
                .authors(&[10, VIEWER])
                .moderator_of(500)
        };
        let post = |tweet_id, author_id, community_id| CommunityPost {
            tweet_id,
            author_id,
            community_id,
        };
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let level = SafetyLevel::TimelineHomeHydration;

        let logged_in = world();
        let hydrated = hydrate(&logged_in, level, Some(VIEWER), &raw).await;
        assert_eq!(
            logged_in.community_posts(),
            [post(1, 10, 500), post(2, 10, 501)]
        );
        assert_eq!(logged_in.keys(Source::CommunityModerator), [vec![500]]);
        assert_eq!(
            hydrated
                .candidates
                .iter()
                .map(|c| (c.community_moderation, c.viewer_is_community_moderator))
                .collect::<Vec<_>>(),
            [
                (HIDDEN, Some(true)),
                (CommunityModeration::default(), None),
                (CommunityModeration::default(), None),
                (CommunityModeration::default(), None),
            ]
        );

        let logged_out = world();
        hydrate(&logged_out, level, None, &raw).await;
        assert_eq!(
            logged_out.keys(Source::CommunityModeration),
            [vec![1, 2, 4]]
        );
        assert!(logged_out.keys(Source::CommunityModerator).is_empty());

        let failing = world().fault(Source::CommunityModerator, Fault::Fails);
        let hydrated = hydrate(&failing, level, Some(VIEWER), &raw).await;
        let first = hydrated.candidates.first().unwrap();
        assert_eq!(first.viewer_is_community_moderator, None);
        assert_eq!(
            RuleEngine::for_tests()
                .evaluate(level, &hydrated.viewer_features, first)
                .into_verdict(),
            allow()
        );
    }

    #[tokio::test]
    async fn every_community_post_asks_once_per_community_whether_the_viewer_was_removed() {
        let world = || {
            InMemorySources::default()
                .tweet(1, 10)
                .tweet_features(1, community_tweet(500))
                .tweet(2, 10)
                .tweet_features(2, community_tweet(501))
                .tweet(3, 10)
                .tweet(4, VIEWER)
                .tweet_features(4, community_tweet(500))
                .authors(&[10, VIEWER])
                .removed_from(500)
        };
        let level = SafetyLevel::TimelineHomeHydration;
        let raw = [raw(1, None), raw(2, None), raw(3, None), raw(4, None)];
        let removed = |hydrated: &InRequestOrder| -> Vec<bool> {
            hydrated
                .candidates
                .iter()
                .map(|c| c.viewer_is_removed_from_community)
                .collect()
        };

        let logged_in = world();
        let hydrated = hydrate(&logged_in, level, Some(VIEWER), &raw).await;
        assert_eq!(
            logged_in.keys(Source::CommunityViewerRemoved),
            [vec![500, 501]]
        );
        assert_eq!(removed(&hydrated), [true, false, false, true]);
        assert!(hydrated.failed_ids.is_empty());

        let logged_out = world();
        let hydrated = hydrate(&logged_out, level, None, &raw).await;
        assert!(logged_out.keys(Source::CommunityViewerRemoved).is_empty());
        assert_eq!(removed(&hydrated), [false; 4]);

        let failing = world().fault(Source::CommunityViewerRemoved, Fault::Fails);
        let hydrated = hydrate(&failing, level, Some(VIEWER), &raw).await;
        assert_eq!(removed(&hydrated), [false; 4]);
        assert_eq!(
            hydrated.failed_nodes,
            [
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
                Ok(Hydrators::empty()),
                Ok(Hydrators::of(Hydrator::CommunityViewerRemoved)),
            ]
        );
    }
}
