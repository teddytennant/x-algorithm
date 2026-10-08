use crate::clients::about_this_account_client::AboutThisAccountClient;
use crate::clients::article_client::ArticleClient;
use crate::clients::gizmoduck_client::GizmoduckLookup;
use crate::clients::socialgraph_client::{EdgeQuery, SocialgraphClient};
use crate::clients::trusted_friends_client::TrustedFriendsClient;
use crate::clients::user_location_client::UserLocationClient;
use crate::clients::wingman_client::WingmanClient;
use crate::hydration::batch::{Hydrated, HydrationBatch, HydrationError, RawHydrationBatch};
use crate::hydration::community_source::{CommunityPost, CommunitySource};
use crate::hydration::decode::article::decode_lifecycle;
use crate::hydration::decode::author::AuthorFallbackCache;
pub(crate) use crate::hydration::decode::author::{author_batch, DecodedAuthor};
use crate::hydration::decode::tweet::{pure_core, TweetFallbackCache};
use crate::hydration::decode::viewer::decode_viewer;
pub(crate) use crate::hydration::decode::viewer::DecodedViewer;
use crate::hydration::tweet_source::{decode_tweet, TweetSource};
use crate::models::{ArticleLifecycle, CommunityModeration, PureCore, TweetFeatures};
use crate::safety_label_source::SafetyLabelSource;
pub(crate) use exchange::{Bytes, EdgeKey, Exchange, Id, Scope};
use prost::Message;
use rustc_hash::FxHashMap;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter;
use std::sync::Arc;
use tracing::warn;
use wingman_client::Exists;
use xai_core_entities::entities::{ConversationControl, GizmoduckUser, PureCoreData};
use xai_core_entities::gizmoduck_client::{GizmoduckClient, QueryFields};
use xai_core_entities::tweet_entity_service_client::TESClient;
use xai_visibility_filtering_proto as vf_pb;

pub(crate) mod exchange;

#[tonic::async_trait]
pub(crate) trait Sources: Send + Sync {
    async fn pure_cores(&self, tweet_ids: &[u64]) -> RawHydrationBatch<PureCore>;

    async fn tweets(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetFeatures>;

    async fn conversation_controls(
        &self,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<ConversationControl>;

    async fn safety_labels(
        &self,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>>;

    async fn viewer(
        &self,
        viewer_id: u64,
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedViewer>;

    async fn users(
        &self,
        user_ids: &[u64],
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedAuthor>;

    async fn select_edges(
        &self,
        viewer_id: u64,
        queries: &[EdgeQuery],
    ) -> Vec<RawHydrationBatch<bool>>;

    async fn viewer_country(&self, viewer_id: u64) -> RawHydrationBatch<Arc<str>>;

    async fn second_degree(
        &self,
        viewer_id: u64,
        root_author_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn community_moderations(
        &self,
        posts: &[CommunityPost],
    ) -> RawHydrationBatch<CommunityModeration>;

    async fn community_moderators(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn community_viewer_removals(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool>;

    async fn article_lifecycles(&self, article_ids: &[u64]) -> RawHydrationBatch<ArticleLifecycle>;

    async fn trusted_friends(&self, viewer_id: u64, list_ids: &[u64]) -> RawHydrationBatch<bool>;

    async fn outside_places(&self, viewer_id: u64, place_ids: &[u64]) -> RawHydrationBatch<bool>;

    fn tweet_cache(&self) -> Option<&TweetFallbackCache> {
        None
    }

    fn author_cache(&self) -> Option<&AuthorFallbackCache> {
        None
    }
}

fn edge_batches(
    queries: &[EdgeQuery],
    sets: Option<&[Option<HashSet<u64>>]>,
) -> Vec<RawHydrationBatch<bool>> {
    queries
        .iter()
        .enumerate()
        .map(|(position, query)| {
            let set = sets.map(|sets| sets.get(position).and_then(Option::as_ref));
            let answers = query.destination_ids.iter().map(|&destination| {
                let answer = match set {
                    None => Hydrated::Failed(HydrationError::Error),
                    Some(None) => Hydrated::Partial(false),
                    Some(Some(set)) => Hydrated::Found(set.contains(&destination)),
                };
                (destination, answer)
            });
            HydrationBatch::from_hydrated(answers.collect())
        })
        .collect()
}

pub(crate) fn core_batch(cores: RawHydrationBatch<PureCoreData>) -> RawHydrationBatch<PureCore> {
    cores.map(|core| pure_core(&core))
}

pub(crate) fn tweet_batch<B: AsRef<[u8]>>(
    tweet_ids: &[u64],
    values: HashMap<u64, anyhow::Result<B>>,
) -> RawHydrationBatch<TweetFeatures> {
    let tweets: FxHashMap<_, _> = values
        .into_iter()
        .map(|(id, value)| (id, value.and_then(|bytes| decode_tweet(bytes.as_ref()))))
        .collect();
    HydrationBatch::from_results(tweet_ids.iter().copied(), tweets)
}

pub(crate) fn label_batch<E>(
    tweet_ids: &[u64],
    labels: FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, E>>,
) -> RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>> {
    let labels: FxHashMap<_, _> = labels
        .into_iter()
        .map(|(id, labels)| (id, labels.map(Some)))
        .collect();
    HydrationBatch::from_results(tweet_ids.iter().copied(), labels)
}

pub(crate) fn viewer_batch<E>(
    viewer_id: u64,
    viewer: Result<Option<GizmoduckUser>, E>,
    fields: &[QueryFields],
) -> RawHydrationBatch<DecodedViewer> {
    let viewer = viewer.map(|user| Some(decode_viewer(user.as_ref(), fields)));
    one_viewer_batch(viewer_id, viewer)
}

pub(crate) fn country_batch<E>(
    viewer_id: u64,
    country: Result<Option<String>, E>,
) -> RawHydrationBatch<Arc<str>> {
    one_viewer_batch(viewer_id, country.map(|country| country.map(Arc::from)))
}

fn one_viewer_batch<V, E>(viewer_id: u64, answer: Result<Option<V>, E>) -> RawHydrationBatch<V> {
    HydrationBatch::from_results([viewer_id], HashMap::from([(viewer_id, answer)]))
}

pub(crate) fn lifecycle_batch(rows: RawHydrationBatch<i32>) -> RawHydrationBatch<ArticleLifecycle> {
    let decoded = rows.into_hydrated().into_iter().map(|(id, row)| {
        let lifecycle = match row {
            Hydrated::Found(value) => decode_lifecycle(value)
                .map_or(Hydrated::Failed(HydrationError::Error), Hydrated::Found),
            Hydrated::Partial(value) => decode_lifecycle(value)
                .map_or(Hydrated::Failed(HydrationError::Error), Hydrated::Partial),
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed(error) => Hydrated::Failed(error),
        };
        (id, lifecycle)
    });
    HydrationBatch::from_hydrated(decoded.collect())
}

pub(crate) trait Observer: Send + Sync {
    fn answered<X: Exchange>(&self, answers: impl Iterator<Item = (X::Key, Hydrated<X::Wire>)>);

    fn asked<X: Exchange>(&self, fields: &[QueryFields]);
}

impl Observer for () {
    fn answered<X: Exchange>(&self, _: impl Iterator<Item = (X::Key, Hydrated<X::Wire>)>) {}

    fn asked<X: Exchange>(&self, _: &[QueryFields]) {}
}

fn seen<V, W, E>(
    result: Option<&Result<V, E>>,
    wire: impl FnOnce(&V) -> Hydrated<W>,
) -> Hydrated<W> {
    match result {
        Some(Ok(value)) => wire(value),
        Some(Err(_)) | None => Hydrated::Failed(HydrationError::Error),
    }
}

fn found<W: Clone>(value: Option<&W>) -> Hydrated<W> {
    value.cloned().map_or(Hydrated::NotFound, Hydrated::Found)
}

pub(crate) struct ProdSources<O = ()> {
    tes: Arc<dyn TESClient + Send + Sync>,
    tweets: TweetSource,
    gizmoduck: GizmoduckLookup,
    socialgraph: Arc<dyn SocialgraphClient + Send + Sync>,
    about_this_account: Arc<dyn AboutThisAccountClient>,
    wingman: Arc<dyn WingmanClient>,
    articles: Arc<dyn ArticleClient>,
    trusted_friends: Arc<dyn TrustedFriendsClient>,
    user_location: Arc<dyn UserLocationClient>,
    safety_labels: Arc<SafetyLabelSource>,
    communities: CommunitySource,
    author_cache: Option<AuthorFallbackCache>,
    tweet_cache: Option<TweetFallbackCache>,
    observer: O,
}

impl ProdSources {
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per backend and cache"
    )]
    pub(crate) fn new(
        tes: Arc<dyn TESClient + Send + Sync>,
        tweets: TweetSource,
        gizmoduck: Arc<dyn GizmoduckClient + Send + Sync>,
        socialgraph: Arc<dyn SocialgraphClient + Send + Sync>,
        about_this_account: Arc<dyn AboutThisAccountClient>,
        wingman: Arc<dyn WingmanClient>,
        articles: Arc<dyn ArticleClient>,
        trusted_friends: Arc<dyn TrustedFriendsClient>,
        user_location: Arc<dyn UserLocationClient>,
        safety_labels: Arc<SafetyLabelSource>,
        communities: CommunitySource,
        author_cache: Option<AuthorFallbackCache>,
        tweet_cache: Option<TweetFallbackCache>,
    ) -> Self {
        Self {
            tes,
            tweets,
            gizmoduck: GizmoduckLookup::new(gizmoduck),
            socialgraph,
            about_this_account,
            wingman,
            articles,
            trusted_friends,
            user_location,
            safety_labels,
            communities,
            author_cache,
            tweet_cache,
            observer: (),
        }
    }

    pub(crate) fn observed<O: Observer>(self, observer: O) -> ProdSources<O> {
        ProdSources {
            tes: self.tes,
            tweets: self.tweets,
            gizmoduck: self.gizmoduck,
            socialgraph: self.socialgraph,
            about_this_account: self.about_this_account,
            wingman: self.wingman,
            articles: self.articles,
            trusted_friends: self.trusted_friends,
            user_location: self.user_location,
            safety_labels: self.safety_labels,
            communities: self.communities,
            author_cache: self.author_cache,
            tweet_cache: self.tweet_cache,
            observer,
        }
    }
}

impl<O: Observer> ProdSources<O> {
    fn relay<X: Exchange, E>(
        &self,
        keys: impl Iterator<Item = X::Key> + Clone,
        answers: HashMap<X::Key, Result<Option<X::Wire>, E>>,
    ) -> HydrationBatch<X::Key, X::Wire>
    where
        X::Key: Eq + Hash,
        X::Wire: Clone,
    {
        self.observer.answered::<X>(keys.clone().map(|key| {
            let answer = seen(answers.get(&key), |wire| found(wire.as_ref()));
            (key, answer)
        }));
        HydrationBatch::from_results(keys, answers)
    }
}

#[tonic::async_trait]
impl<O: Observer> Sources for ProdSources<O> {
    async fn pure_cores(&self, tweet_ids: &[u64]) -> RawHydrationBatch<PureCore> {
        let cores = self.tes.get_tweet_core_datas(tweet_ids.to_vec()).await;
        core_batch(self.relay::<exchange::TesPureCore, _>(tweet_ids.iter().copied(), cores))
    }

    async fn tweets(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetFeatures> {
        let values = self.tweets.get_tweet_values(tweet_ids).await;
        self.observer
            .answered::<exchange::TesTweet>(tweet_ids.iter().map(|&id| {
                let answer = seen(values.get(&id), |bytes| {
                    Hydrated::Found(Bytes(bytes.to_vec()))
                });
                (id, answer)
            }));
        tweet_batch(tweet_ids, values)
    }

    async fn conversation_controls(
        &self,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<ConversationControl> {
        let controls = self.tes.get_conversation_controls(tweet_ids.to_vec()).await;
        self.relay::<exchange::TesConversationControl, _>(tweet_ids.iter().copied(), controls)
    }

    async fn safety_labels(
        &self,
        tweet_ids: &[u64],
    ) -> RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>> {
        let labels = self.safety_labels.get(tweet_ids).await;
        self.observer
            .answered::<exchange::SafetyLabels>(tweet_ids.iter().map(|&id| {
                let answer = seen(labels.get(&id), |labels| {
                    Hydrated::Found(Bytes(labels.encode_to_vec()))
                });
                (id, answer)
            }));
        label_batch(tweet_ids, labels)
    }

    async fn viewer(
        &self,
        viewer_id: u64,
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedViewer> {
        let viewer = self
            .gizmoduck
            .get_viewer(viewer_id, fields)
            .await
            .inspect_err(|error| warn!(%error, "Gizmoduck viewer lookup failed; failing open"));
        self.observer.asked::<exchange::GizmoduckViewer>(fields);
        self.observer
            .answered::<exchange::GizmoduckViewer>(iter::once_with(|| {
                (viewer_id, seen(Some(&viewer), |user| found(user.as_ref())))
            }));
        viewer_batch(viewer_id, viewer, fields)
    }

    async fn users(
        &self,
        user_ids: &[u64],
        fields: &[QueryFields],
    ) -> RawHydrationBatch<DecodedAuthor> {
        let users = self.gizmoduck.get_users(user_ids.to_vec(), fields).await;
        self.observer.asked::<exchange::GizmoduckAuthor>(fields);
        author_batch(self.relay::<exchange::GizmoduckAuthor, _>(user_ids.iter().copied(), users))
    }

    async fn select_edges(
        &self,
        viewer_id: u64,
        queries: &[EdgeQuery],
    ) -> Vec<RawHydrationBatch<bool>> {
        let sets = self.socialgraph.select_edges(viewer_id, queries).await;
        let edges = edge_batches(queries, sets.as_deref());
        self.observer
            .answered::<exchange::Flock>(queries.iter().zip(&edges).flat_map(|(query, batch)| {
                query.destination_ids.iter().map(move |&destination| {
                    let answer = batch
                        .hydrated(&destination)
                        .cloned()
                        .unwrap_or(Hydrated::Failed(HydrationError::Error));
                    (EdgeKey::of(query, destination), answer)
                })
            }));
        edges
    }

    async fn viewer_country(&self, viewer_id: u64) -> RawHydrationBatch<Arc<str>> {
        let country = self
            .about_this_account
            .tfe_top_country(viewer_id)
            .await
            .inspect_err(|error| warn!(%error, "tfe_top_country lookup failed"));
        self.observer
            .answered::<exchange::ViewerCountry>(iter::once_with(|| {
                (
                    viewer_id,
                    seen(Some(&country), |country| found(country.as_ref())),
                )
            }));
        country_batch(viewer_id, country)
    }

    async fn second_degree(
        &self,
        viewer_id: u64,
        root_author_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let answers = self
            .wingman
            .batch_exists_intersect(viewer_id, root_author_ids)
            .await;
        let answers = root_author_ids
            .iter()
            .copied()
            .zip(answers.into_iter().flatten())
            .map(|(root, answer)| {
                let answer = match answer {
                    Exists::Found => Ok(Some(true)),
                    Exists::NotFound => Ok(Some(false)),
                    Exists::Incomplete | Exists::ItemError => Err(answer),
                };
                (root, answer)
            })
            .collect();
        self.relay::<exchange::Wingman, _>(root_author_ids.iter().copied(), answers)
    }

    async fn community_moderations(
        &self,
        posts: &[CommunityPost],
    ) -> RawHydrationBatch<CommunityModeration> {
        let moderations = self.communities.moderations(posts).await;
        self.relay::<exchange::CommunityModeration, _>(
            posts.iter().map(|post| post.tweet_id),
            moderations,
        )
    }

    async fn community_moderators(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let moderators = self.communities.moderators(viewer_id, community_ids).await;
        self.relay::<exchange::CommunityModerator, _>(community_ids.iter().copied(), moderators)
    }

    async fn community_viewer_removals(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> RawHydrationBatch<bool> {
        let removals = self
            .communities
            .viewer_removals(viewer_id, community_ids)
            .await;
        self.relay::<exchange::CommunityViewerRemoved, _>(community_ids.iter().copied(), removals)
    }

    async fn article_lifecycles(&self, article_ids: &[u64]) -> RawHydrationBatch<ArticleLifecycle> {
        let rows = self.articles.lifecycles(article_ids).await;
        lifecycle_batch(
            self.relay::<exchange::ArticleLifecycle, _>(article_ids.iter().copied(), rows),
        )
    }

    async fn trusted_friends(&self, viewer_id: u64, list_ids: &[u64]) -> RawHydrationBatch<bool> {
        let answers = self
            .trusted_friends
            .batch_is_member_or_owner(viewer_id, list_ids)
            .await;
        let answers = list_ids
            .iter()
            .copied()
            .zip(answers)
            .map(|(list_id, answer)| (list_id, answer.map(Some)))
            .collect();
        self.relay::<exchange::TrustedFriends, _>(list_ids.iter().copied(), answers)
    }

    async fn outside_places(&self, viewer_id: u64, place_ids: &[u64]) -> RawHydrationBatch<bool> {
        let places = self
            .user_location
            .places(viewer_id)
            .await
            .inspect_err(|error| warn!(%error, "Geoduck userLocation lookup failed"));
        let answers = place_ids
            .iter()
            .map(|&place| {
                let outside = places.as_ref().map(|places| Some(!places.contains(&place)));
                (place, outside)
            })
            .collect();
        self.relay::<exchange::UserLocation, _>(place_ids.iter().copied(), answers)
    }

    fn tweet_cache(&self) -> Option<&TweetFallbackCache> {
        self.tweet_cache.as_ref()
    }

    fn author_cache(&self) -> Option<&AuthorFallbackCache> {
        self.author_cache.as_ref()
    }
}

#[cfg(test)]
pub(crate) use in_memory::{control, suspended, Fault, InMemorySources};

#[cfg(test)]
mod in_memory {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeDirection, Graph};
    use crate::hydration::plan::Source;
    use std::iter;
    use std::sync::Mutex;
    use std::time::Duration;
    use tokio::time::{sleep, Instant};
    use xai_core_entities::entities::{
        ConversationControlArm, GizmoduckUser, GizmoduckUserResult, PureCoreData, Safety,
        UserResponseState,
    };

    pub(crate) fn found_user(user_id: u64) -> GizmoduckUserResult {
        GizmoduckUserResult {
            user: Some(GizmoduckUser {
                user_id,
                ..Default::default()
            }),
            response_state: Some(UserResponseState::Found),
        }
    }

    pub(crate) fn suspended() -> GizmoduckUserResult {
        GizmoduckUserResult {
            user: Some(GizmoduckUser {
                safety: Safety {
                    suspended: true,
                    ..Default::default()
                },
                ..Default::default()
            }),
            response_state: Some(UserResponseState::Found),
        }
    }

    pub(crate) fn control(
        arm: ConversationControlArm,
        root: u64,
        countries: &[&str],
    ) -> ConversationControl {
        ConversationControl {
            arm,
            conversation_tweet_author_id: root,
            invited_user_ids: vec![],
            invite_via_mention: None,
            allowed_country_codes: countries.iter().map(|c| (*c).to_owned()).collect(),
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Fault {
        Fails,
        Hangs,
        Delays(std::time::Duration),
        AnswersPartial,
    }

    #[derive(Default)]
    pub(crate) struct InMemorySources {
        pure_cores: HashMap<u64, PureCoreData>,
        tweets: HashMap<u64, TweetFeatures>,
        controls: HashMap<u64, ConversationControl>,
        labels: HashMap<u64, Arc<vf_pb::SafetyLabelMap>>,
        viewers: HashMap<u64, GizmoduckUser>,
        users: HashMap<u64, GizmoduckUserResult>,
        edges: HashSet<(Graph, u64, u64)>,
        countries: HashMap<u64, Arc<str>>,
        second_degree: HashSet<(u64, u64)>,
        community_moderations: HashMap<u64, CommunityModeration>,
        moderated_communities: HashSet<u64>,
        removed_from_communities: HashSet<u64>,
        community_posts: Mutex<Vec<CommunityPost>>,
        lifecycles: HashMap<u64, ArticleLifecycle>,
        trusted_friends: HashSet<(u64, u64)>,
        viewer_places: HashSet<u64>,
        faults: Mutex<Vec<(Source, Fault)>>,
        failed_keys: HashSet<(Source, u64)>,
        latencies: HashMap<Source, Duration>,
        key_latencies: HashMap<(Source, u64), Duration>,
        failed_graphs: HashSet<Graph>,
        failed_edges: HashSet<(Graph, u64)>,
        hung_graphs: HashSet<Graph>,
        missing_graphs: HashSet<Graph>,
        author_cache: Option<AuthorFallbackCache>,
        tweet_cache: Option<TweetFallbackCache>,
        calls: Mutex<Vec<(Source, Vec<u64>)>>,
        starts: Mutex<Vec<(Source, Instant)>>,
        selects: Mutex<Vec<Vec<EdgeQuery>>>,
        fields: Mutex<Vec<(Source, Vec<QueryFields>)>>,
    }

    impl InMemorySources {
        pub(crate) fn tweet(self, tweet_id: u64, author_id: u64) -> Self {
            self.pure_core(
                tweet_id,
                PureCoreData {
                    author_id,
                    ..Default::default()
                },
            )
        }

        pub(crate) fn pure_core(mut self, tweet_id: u64, core: PureCoreData) -> Self {
            self.pure_cores.insert(tweet_id, core);
            self.tweets.entry(tweet_id).or_default();
            self
        }

        pub(crate) fn without_tweet_row(mut self, tweet_id: u64) -> Self {
            self.tweets.remove(&tweet_id);
            self
        }

        pub(crate) fn tweet_features(mut self, tweet_id: u64, tweet: TweetFeatures) -> Self {
            self.tweets.insert(tweet_id, tweet);
            self
        }

        pub(crate) fn control(mut self, tweet_id: u64, control: ConversationControl) -> Self {
            self.controls.insert(tweet_id, control);
            self
        }

        pub(crate) fn labels(mut self, tweet_id: u64, labels: vf_pb::SafetyLabelMap) -> Self {
            self.labels.insert(tweet_id, Arc::new(labels));
            self
        }

        pub(crate) fn viewer(mut self, viewer_id: u64, user: GizmoduckUser) -> Self {
            self.viewers.insert(viewer_id, user);
            self
        }

        pub(crate) fn user(mut self, user_id: u64, user: GizmoduckUserResult) -> Self {
            self.users.insert(user_id, user);
            self
        }

        pub(crate) fn authors(self, user_ids: &[u64]) -> Self {
            user_ids.iter().fold(self, |sources, &user_id| {
                sources.user(user_id, found_user(user_id))
            })
        }

        pub(crate) fn edge(mut self, graph: Graph, source: u64, destination: u64) -> Self {
            self.edges.insert((graph, source, destination));
            self
        }

        pub(crate) fn country(mut self, viewer_id: u64, country: &str) -> Self {
            self.countries.insert(viewer_id, Arc::from(country));
            self
        }

        pub(crate) fn second_degree_path(mut self, root_author: u64, viewer_id: u64) -> Self {
            self.second_degree.insert((root_author, viewer_id));
            self
        }

        pub(crate) fn community_moderation(
            mut self,
            tweet_id: u64,
            moderation: CommunityModeration,
        ) -> Self {
            self.community_moderations.insert(tweet_id, moderation);
            self
        }

        pub(crate) fn moderator_of(mut self, community_id: u64) -> Self {
            self.moderated_communities.insert(community_id);
            self
        }

        pub(crate) fn removed_from(mut self, community_id: u64) -> Self {
            self.removed_from_communities.insert(community_id);
            self
        }

        pub(crate) fn community_posts(&self) -> Vec<CommunityPost> {
            self.community_posts.lock().unwrap().clone()
        }

        pub(crate) fn lifecycle(mut self, article_id: u64, lifecycle: ArticleLifecycle) -> Self {
            self.lifecycles.insert(article_id, lifecycle);
            self
        }

        pub(crate) fn trusted_friend(mut self, list_id: u64, viewer_id: u64) -> Self {
            self.trusted_friends.insert((list_id, viewer_id));
            self
        }

        pub(crate) fn located_in(mut self, place_id: u64) -> Self {
            self.viewer_places.insert(place_id);
            self
        }

        pub(crate) fn fault(self, source: Source, fault: Fault) -> Self {
            self.break_source(source, fault);
            self
        }

        pub(crate) fn break_source(&self, source: Source, fault: Fault) {
            self.faults.lock().unwrap().push((source, fault));
        }

        pub(crate) fn fail_graph(mut self, graph: Graph) -> Self {
            self.failed_graphs.insert(graph);
            self
        }

        pub(crate) fn fail_edge(mut self, graph: Graph, destination: u64) -> Self {
            self.failed_edges.insert((graph, destination));
            self
        }

        pub(crate) fn hang_graph(mut self, graph: Graph) -> Self {
            self.hung_graphs.insert(graph);
            self
        }

        pub(crate) fn miss_graph(mut self, graph: Graph) -> Self {
            self.missing_graphs.insert(graph);
            self
        }

        pub(crate) fn fail_key(mut self, source: Source, key: u64) -> Self {
            self.failed_keys.insert((source, key));
            self
        }

        pub(crate) fn latency(mut self, source: Source, latency: Duration) -> Self {
            self.latencies.insert(source, latency);
            self
        }

        pub(crate) fn key_latency(mut self, source: Source, key: u64, latency: Duration) -> Self {
            self.key_latencies.insert((source, key), latency);
            self
        }

        pub(crate) fn with_author_cache(mut self, cache: AuthorFallbackCache) -> Self {
            self.author_cache = Some(cache);
            self
        }

        pub(crate) fn with_tweet_cache(mut self, cache: TweetFallbackCache) -> Self {
            self.tweet_cache = Some(cache);
            self
        }

        pub(crate) fn calls(&self) -> Vec<Source> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .map(|(source, _)| *source)
                .collect()
        }

        pub(crate) fn starts(&self, source: Source) -> Vec<Instant> {
            self.starts
                .lock()
                .unwrap()
                .iter()
                .filter(|(called, _)| *called == source)
                .map(|(_, started)| *started)
                .collect()
        }

        pub(crate) fn keys(&self, source: Source) -> Vec<Vec<u64>> {
            self.calls
                .lock()
                .unwrap()
                .iter()
                .filter(|(called, _)| *called == source)
                .map(|(_, keys)| keys.clone())
                .collect()
        }

        pub(crate) fn selects(&self) -> Vec<Vec<EdgeQuery>> {
            self.selects.lock().unwrap().clone()
        }

        pub(crate) fn fields(&self, source: Source) -> Vec<Vec<QueryFields>> {
            self.fields
                .lock()
                .unwrap()
                .iter()
                .filter(|(called, _)| *called == source)
                .map(|(_, fields)| fields.clone())
                .collect()
        }

        fn record_fields(&self, source: Source, fields: &[QueryFields]) {
            self.fields.lock().unwrap().push((source, fields.to_vec()));
        }

        async fn enter(&self, source: Source, keys: &[u64]) -> bool {
            let mut sorted = keys.to_vec();
            sorted.sort_unstable();
            self.calls.lock().unwrap().push((source, sorted));
            self.starts.lock().unwrap().push((source, Instant::now()));
            let latency = keys
                .iter()
                .filter_map(|&key| self.key_latencies.get(&(source, key)))
                .chain(self.latencies.get(&source))
                .max();
            if let Some(latency) = latency {
                sleep(*latency).await;
            }
            match self.fault_for(source) {
                Some(Fault::Hangs) => std::future::pending().await,
                Some(Fault::Delays(delay)) => {
                    tokio::time::sleep(delay).await;
                    false
                }
                Some(Fault::Fails) => true,
                Some(Fault::AnswersPartial) | None => false,
            }
        }

        fn fault_for(&self, source: Source) -> Option<Fault> {
            self.faults
                .lock()
                .unwrap()
                .iter()
                .find(|(faulty, _)| *faulty == source)
                .map(|(_, fault)| *fault)
        }

        async fn keyed<V: Clone>(
            &self,
            source: Source,
            ids: &[u64],
            values: &HashMap<u64, V>,
        ) -> RawHydrationBatch<V> {
            let fails = self.enter(source, ids).await;
            let results: FxHashMap<_, _> = ids
                .iter()
                .map(|&id| {
                    let result = if fails || self.failed_keys.contains(&(source, id)) {
                        Err(())
                    } else {
                        Ok(values.get(&id).cloned())
                    };
                    (id, result)
                })
                .collect();
            HydrationBatch::from_results(ids.iter().copied(), results)
        }
    }

    #[tonic::async_trait]
    impl Sources for InMemorySources {
        async fn pure_cores(&self, tweet_ids: &[u64]) -> RawHydrationBatch<PureCore> {
            self.keyed(Source::TesPureCore, tweet_ids, &self.pure_cores)
                .await
                .map(|core| pure_core(&core))
        }

        async fn tweets(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetFeatures> {
            self.keyed(Source::TesTweet, tweet_ids, &self.tweets).await
        }

        async fn conversation_controls(
            &self,
            tweet_ids: &[u64],
        ) -> RawHydrationBatch<ConversationControl> {
            self.keyed(Source::TesConversationControl, tweet_ids, &self.controls)
                .await
        }

        async fn safety_labels(
            &self,
            tweet_ids: &[u64],
        ) -> RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>> {
            self.keyed(Source::SafetyLabels, tweet_ids, &self.labels)
                .await
        }

        async fn viewer(
            &self,
            viewer_id: u64,
            fields: &[QueryFields],
        ) -> RawHydrationBatch<DecodedViewer> {
            self.record_fields(Source::GizmoduckViewer, fields);
            let viewer = HashMap::from([(viewer_id, self.viewers.get(&viewer_id).cloned())]);
            self.keyed(Source::GizmoduckViewer, &[viewer_id], &viewer)
                .await
                .map(|user| decode_viewer(user.as_ref(), fields))
        }

        async fn users(
            &self,
            user_ids: &[u64],
            fields: &[QueryFields],
        ) -> RawHydrationBatch<DecodedAuthor> {
            self.record_fields(Source::GizmoduckAuthor, fields);
            let users = self
                .keyed(Source::GizmoduckAuthor, user_ids, &self.users)
                .await;
            let users = match self.fault_for(Source::GizmoduckAuthor) {
                Some(Fault::AnswersPartial) => users.map(|user| GizmoduckUserResult {
                    response_state: Some(UserResponseState::Partial),
                    ..user
                }),
                _ => users,
            };
            author_batch(users)
        }

        async fn select_edges(
            &self,
            viewer_id: u64,
            queries: &[EdgeQuery],
        ) -> Vec<RawHydrationBatch<bool>> {
            let mut recorded = queries.to_vec();
            for query in &mut recorded {
                query.destination_ids.sort_unstable();
            }
            self.selects.lock().unwrap().push(recorded);
            let failed_graph = queries.iter().any(|query| {
                self.failed_graphs.contains(&query.graph)
                    || query
                        .destination_ids
                        .iter()
                        .any(|&id| self.failed_edges.contains(&(query.graph, id)))
            });
            let keys: Vec<u64> = iter::once(viewer_id)
                .chain(
                    queries
                        .iter()
                        .flat_map(|query| query.destination_ids.iter().copied()),
                )
                .collect();
            let fails = self.enter(Source::Flock, &keys).await || failed_graph;
            if queries
                .iter()
                .any(|query| self.hung_graphs.contains(&query.graph))
            {
                std::future::pending::<()>().await;
            }
            let answer = |query: &EdgeQuery| {
                let holds = |&id: &u64| {
                    let edge = match query.direction {
                        EdgeDirection::Forward => (query.graph, viewer_id, id),
                        EdgeDirection::Reverse => (query.graph, id, viewer_id),
                    };
                    self.edges.contains(&edge)
                };
                (!self.missing_graphs.contains(&query.graph)).then(|| {
                    query
                        .destination_ids
                        .iter()
                        .copied()
                        .filter(holds)
                        .collect()
                })
            };
            let sets: Option<Vec<_>> = (!fails).then(|| queries.iter().map(answer).collect());
            edge_batches(queries, sets.as_deref())
        }

        async fn viewer_country(&self, viewer_id: u64) -> RawHydrationBatch<Arc<str>> {
            self.keyed(Source::ViewerCountry, &[viewer_id], &self.countries)
                .await
        }

        async fn second_degree(
            &self,
            viewer_id: u64,
            root_author_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            let paths = root_author_ids
                .iter()
                .map(|&root| (root, self.second_degree.contains(&(root, viewer_id))))
                .collect();
            self.keyed(Source::Wingman, root_author_ids, &paths).await
        }

        async fn community_moderations(
            &self,
            posts: &[CommunityPost],
        ) -> RawHydrationBatch<CommunityModeration> {
            self.community_posts
                .lock()
                .unwrap()
                .extend_from_slice(posts);
            let tweet_ids: Vec<u64> = posts.iter().map(|post| post.tweet_id).collect();
            let moderations = tweet_ids
                .iter()
                .map(|&id| {
                    let moderation = self.community_moderations.get(&id).copied();
                    (id, moderation.unwrap_or_default())
                })
                .collect();
            self.keyed(Source::CommunityModeration, &tweet_ids, &moderations)
                .await
        }

        async fn community_moderators(
            &self,
            _viewer_id: u64,
            community_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            let moderators = community_ids
                .iter()
                .map(|&id| (id, self.moderated_communities.contains(&id)))
                .collect();
            self.keyed(Source::CommunityModerator, community_ids, &moderators)
                .await
        }

        async fn community_viewer_removals(
            &self,
            _viewer_id: u64,
            community_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            let removals = community_ids
                .iter()
                .map(|&id| (id, self.removed_from_communities.contains(&id)))
                .collect();
            self.keyed(Source::CommunityViewerRemoved, community_ids, &removals)
                .await
        }

        async fn article_lifecycles(
            &self,
            article_ids: &[u64],
        ) -> RawHydrationBatch<ArticleLifecycle> {
            self.keyed(Source::ArticleLifecycle, article_ids, &self.lifecycles)
                .await
        }

        async fn trusted_friends(
            &self,
            viewer_id: u64,
            list_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            let lists = list_ids
                .iter()
                .map(|&list| (list, self.trusted_friends.contains(&(list, viewer_id))))
                .collect();
            self.keyed(Source::TrustedFriends, list_ids, &lists).await
        }

        async fn outside_places(
            &self,
            _viewer_id: u64,
            place_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            let outside = place_ids
                .iter()
                .map(|&place| (place, !self.viewer_places.contains(&place)))
                .collect();
            self.keyed(Source::UserLocation, place_ids, &outside).await
        }

        fn tweet_cache(&self) -> Option<&TweetFallbackCache> {
            self.tweet_cache.as_ref()
        }

        fn author_cache(&self) -> Option<&AuthorFallbackCache> {
            self.author_cache.as_ref()
        }
    }
}
