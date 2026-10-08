use crate::hydration::batch::Hydrated;
use crate::hydration::plan::Source;
use crate::hydration::sources::{Exchange, Id, Observer, Scope, exchange};
use crate::models::CommunityModeration;
use anyhow::{Context, anyhow};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::iter;
use std::sync::{Arc, Mutex, PoisonError};
use strum::VariantArray;
use xai_core_entities::gizmoduck_client::QueryFields;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Answer<V> {
    Found(V),
    Partial(V),
    NotFound,
    Failed,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "CommunityModeration")]
struct CommunityModerationJson {
    is_hidden: bool,
    is_author_removed: bool,
}

impl Serialize for CommunityModeration {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        CommunityModerationJson::serialize(self, serializer)
    }
}

impl<'de> Deserialize<'de> for CommunityModeration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        CommunityModerationJson::deserialize(deserializer)
    }
}

type Answers = BTreeMap<String, BTreeMap<String, Answer<serde_json::Value>>>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(from = "BTreeMap<String, Kept>", into = "BTreeMap<String, Kept>")]
pub(crate) struct Recording {
    answers: Answers,
    fields: BTreeMap<String, Vec<QueryFields>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum Kept {
    Answers(BTreeMap<String, Answer<serde_json::Value>>),
    Fields(Vec<QueryFields>),
}

impl From<BTreeMap<String, Kept>> for Recording {
    fn from(kept: BTreeMap<String, Kept>) -> Self {
        let mut recording = Self::default();
        for (name, kept) in kept {
            match kept {
                Kept::Answers(answers) => {
                    recording.answers.insert(name, answers);
                }
                Kept::Fields(fields) => {
                    recording.fields.insert(name, fields);
                }
            }
        }
        recording
    }
}

impl From<Recording> for BTreeMap<String, Kept> {
    fn from(recording: Recording) -> Self {
        let answers = recording
            .answers
            .into_iter()
            .map(|(name, answers)| (name, Kept::Answers(answers)));
        let fields = recording
            .fields
            .into_iter()
            .map(|(name, fields)| (name, Kept::Fields(fields)));
        answers.chain(fields).collect()
    }
}

trait Visit {
    fn exchange<X: Exchange>(&mut self);
}

fn visit(source: Source, visitor: &mut impl Visit) {
    match source {
        Source::TesPureCore => visitor.exchange::<exchange::TesPureCore>(),
        Source::TesTweet => visitor.exchange::<exchange::TesTweet>(),
        Source::TesConversationControl => visitor.exchange::<exchange::TesConversationControl>(),
        Source::SafetyLabels => visitor.exchange::<exchange::SafetyLabels>(),
        Source::GizmoduckViewer => visitor.exchange::<exchange::GizmoduckViewer>(),
        Source::GizmoduckAuthor => visitor.exchange::<exchange::GizmoduckAuthor>(),
        Source::Flock => visitor.exchange::<exchange::Flock>(),
        Source::ViewerCountry => visitor.exchange::<exchange::ViewerCountry>(),
        Source::Wingman => visitor.exchange::<exchange::Wingman>(),
        Source::CommunityModeration => visitor.exchange::<exchange::CommunityModeration>(),
        Source::CommunityModerator => visitor.exchange::<exchange::CommunityModerator>(),
        Source::CommunityViewerRemoved => visitor.exchange::<exchange::CommunityViewerRemoved>(),
        Source::ArticleLifecycle => visitor.exchange::<exchange::ArticleLifecycle>(),
        Source::TrustedFriends => visitor.exchange::<exchange::TrustedFriends>(),
        Source::UserLocation => visitor.exchange::<exchange::UserLocation>(),
    }
}

fn visit_all<V: Visit>(mut visitor: V) -> V {
    for &source in Source::VARIANTS {
        visit(source, &mut visitor);
    }
    visitor
}

fn owned_names<X: Exchange>() -> impl Iterator<Item = &'static str> {
    iter::once(X::NAME).chain(X::FIELDS)
}

fn parse<X: Exchange>(
    key: &str,
    answer: &Answer<serde_json::Value>,
) -> anyhow::Result<(X::Key, Option<X::Wire>)> {
    let parsed = key
        .parse()
        .map_err(|_| anyhow!("{}/{key}: not a key", X::NAME))?;
    let wire = match answer {
        Answer::Found(json) | Answer::Partial(json) => {
            Some(X::Wire::deserialize(json).with_context(|| format!("{}/{key}", X::NAME))?)
        }
        Answer::NotFound | Answer::Failed => None,
    };
    Ok((parsed, wire))
}

struct Named<'a> {
    recording: &'a Recording,
    owned: BTreeSet<&'static str>,
    ids: Vec<anyhow::Result<Vec<Id>>>,
}

impl Visit for Named<'_> {
    fn exchange<X: Exchange>(&mut self) {
        self.owned.extend(owned_names::<X>());
        let answers = self.recording.answers.get(X::NAME).into_iter().flatten();
        self.ids.extend(answers.map(|(key, answer)| {
            let (parsed, wire) = parse::<X>(key, answer)?;
            X::ids(&parsed, wire.as_ref()).with_context(|| format!("{}/{key}", X::NAME))
        }));
    }
}

struct Unsuccessful<'a> {
    answers: &'a Answers,
    keys: Vec<String>,
}

impl Visit for Unsuccessful<'_> {
    fn exchange<X: Exchange>(&mut self) {
        let answers = self.answers.get(X::NAME).into_iter().flatten();
        self.keys.extend(
            answers
                .filter(|(_, answer)| match answer {
                    Answer::Failed => true,
                    Answer::Partial(_) => !X::KEEPS_PARTIAL,
                    Answer::Found(_) | Answer::NotFound => false,
                })
                .map(|(key, _)| format!("{}/{key}", X::NAME)),
        );
    }
}

struct Split {
    case: Recording,
    shared: Recording,
}

impl Visit for Split {
    fn exchange<X: Exchange>(&mut self) {
        if X::SCOPE != Scope::Shared {
            return;
        }
        if let Some(answers) = self.case.answers.remove(X::NAME) {
            self.shared.answers.insert(X::NAME.to_owned(), answers);
        }
        if let Some(name) = X::FIELDS
            && let Some(fields) = self.case.fields.remove(name)
        {
            self.shared.fields.insert(name.to_owned(), fields);
        }
    }
}

impl Recording {
            pub(crate) fn ids(&self) -> anyhow::Result<Vec<Id>> {
        let named = visit_all(Named {
            recording: self,
            owned: BTreeSet::new(),
            ids: vec![],
        });
        let recorded = self.answers.keys().chain(self.fields.keys());
        if let Some(stray) = recorded
            .clone()
            .find(|name| !named.owned.contains(name.as_str()))
        {
            anyhow::bail!("{stray}: no Exchange records by this name");
        }
        Ok(named
            .ids
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?
            .concat())
    }

            pub(crate) fn unsuccessful(&self) -> Vec<String> {
        let mut unsuccessful = visit_all(Unsuccessful {
            answers: &self.answers,
            keys: vec![],
        })
        .keys;
        unsuccessful.sort();
        unsuccessful
    }

            pub(crate) fn split_shared(self) -> (Self, Self) {
        let split = visit_all(Split {
            case: self,
            shared: Self::default(),
        });
        (split.case, split.shared)
    }
}

fn recorded<W: Serialize>(answer: Hydrated<W>) -> Answer<serde_json::Value> {
    match answer {
        Hydrated::Found(wire) => serde_json::to_value(wire).map_or(Answer::Failed, Answer::Found),
        Hydrated::Partial(wire) => {
            serde_json::to_value(wire).map_or(Answer::Failed, Answer::Partial)
        }
        Hydrated::NotFound => Answer::NotFound,
        Hydrated::Failed(_) => Answer::Failed,
    }
}

#[derive(Default)]
pub(crate) struct Recorder(Mutex<Recording>);

impl Recorder {
    pub(crate) fn take(&self) -> Recording {
        std::mem::take(&mut self.lock())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Recording> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Observer for Arc<Recorder> {
    fn answered<X: Exchange>(&self, answers: impl Iterator<Item = (X::Key, Hydrated<X::Wire>)>) {
        let answers: Vec<_> = answers
            .map(|(key, answer)| (key.to_string(), recorded(answer)))
            .collect();
        if !answers.is_empty() {
            let mut recording = self.lock();
            let recorded = recording.answers.entry(X::NAME.to_owned()).or_default();
            recorded.extend(answers);
        }
    }

    fn asked<X: Exchange>(&self, fields: &[QueryFields]) {
        let Some(name) = X::FIELDS else {
            return;
        };
        if fields.is_empty() {
            return;
        }
        let mut recording = self.lock();
        let recorded = recording.fields.entry(name.to_owned()).or_default();
        for field in fields {
            if !recorded.contains(field) {
                recorded.push(*field);
            }
        }
    }
}

#[cfg(test)]
pub(crate) use replay::Replay;

#[cfg(test)]
mod replay {
    use super::*;
    use crate::clients::socialgraph_client::EdgeQuery;
    use crate::hydration::batch::{HydrationBatch, HydrationError, RawHydrationBatch};
    use crate::hydration::community_source::CommunityPost;
    use crate::hydration::sources::{
        Bytes, DecodedAuthor, DecodedViewer, EdgeKey, Sources, author_batch, core_batch,
        country_batch, label_batch, lifecycle_batch, tweet_batch, viewer_batch,
    };
    use crate::models::{ArticleLifecycle, PureCore, TweetFeatures};
    use prost::Message;
    use std::collections::HashMap;
    use std::hash::Hash;
    use xai_core_entities::entities::ConversationControl;
    use xai_visibility_filtering_proto as vf_pb;

            pub(crate) struct Replay {
        recording: Recording,
        misses: Mutex<Vec<String>>,
    }

    fn result<V>(answer: Hydrated<V>) -> Result<Option<V>, HydrationError> {
        match answer {
            Hydrated::Found(value) | Hydrated::Partial(value) => Ok(Some(value)),
            Hydrated::NotFound => Ok(None),
            Hydrated::Failed(error) => Err(error),
        }
    }

    impl Replay {
        fn answer<X: Exchange>(&self, key: &X::Key) -> Hydrated<X::Wire> {
            let name = X::NAME;
            let recorded = self.recording.answers.get(name);
            let wire = |json: &serde_json::Value| {
                X::Wire::deserialize(json).unwrap_or_else(|e| panic!("{name}/{key}: {e}"))
            };
            match recorded.and_then(|answers| answers.get(&key.to_string())) {
                Some(Answer::Found(json)) => Hydrated::Found(wire(json)),
                Some(Answer::Partial(json)) if X::KEEPS_PARTIAL => Hydrated::Partial(wire(json)),
                Some(Answer::Partial(_)) => {
                    panic!("{name}/{key}: a partial answer, which capture never keeps")
                }
                Some(Answer::NotFound) => Hydrated::NotFound,
                Some(Answer::Failed) => Hydrated::Failed(HydrationError::Error),
                None => {
                    self.misses.lock().unwrap().push(format!("{name}/{key}"));
                    Hydrated::NotFound
                }
            }
        }

        fn batch<X: Exchange>(
            &self,
            keys: impl Iterator<Item = X::Key>,
        ) -> HydrationBatch<X::Key, X::Wire>
        where
            X::Key: Eq + Hash,
        {
            HydrationBatch::from_hydrated(
                keys.map(|key| {
                    let answer = self.answer::<X>(&key);
                    (key, answer)
                })
                .collect(),
            )
        }

                                fn found<X: Exchange<Key = u64>>(
            &self,
            keys: &[u64],
        ) -> HashMap<u64, anyhow::Result<X::Wire>> {
            keys.iter()
                .map(|&key| {
                    let wire = self.answer::<X>(&key).into_value();
                    (
                        key,
                        wire.ok_or_else(|| anyhow!("{}/{key}: no answer", X::NAME)),
                    )
                })
                .collect()
        }

        fn asked<X: Exchange>(&self, fields: &[QueryFields]) {
            let Some(name) = X::FIELDS else {
                return;
            };
            let recorded = self.recording.fields.get(name);
            let mut missing: Vec<&QueryFields> = fields
                .iter()
                .filter(|field| !recorded.is_some_and(|recorded| recorded.contains(field)))
                .collect();
            missing.dedup();
            if !missing.is_empty() {
                self.misses
                    .lock()
                    .unwrap()
                    .push(format!("{name} {missing:?}"));
            }
        }

                pub(crate) fn misses(&self) -> Vec<String> {
            let mut misses = self.misses.lock().unwrap().clone();
            misses.sort();
            misses.dedup();
            misses
        }
    }

    #[tonic::async_trait]
    impl Sources for Replay {
        async fn pure_cores(&self, tweet_ids: &[u64]) -> RawHydrationBatch<PureCore> {
            core_batch(self.batch::<exchange::TesPureCore>(tweet_ids.iter().copied()))
        }

        async fn tweets(&self, tweet_ids: &[u64]) -> RawHydrationBatch<TweetFeatures> {
            tweet_batch(tweet_ids, self.found::<exchange::TesTweet>(tweet_ids))
        }

        async fn conversation_controls(
            &self,
            tweet_ids: &[u64],
        ) -> RawHydrationBatch<ConversationControl> {
            self.batch::<exchange::TesConversationControl>(tweet_ids.iter().copied())
        }

        async fn safety_labels(
            &self,
            tweet_ids: &[u64],
        ) -> RawHydrationBatch<Arc<vf_pb::SafetyLabelMap>> {
            let labels = self
                .found::<exchange::SafetyLabels>(tweet_ids)
                .into_iter()
                .map(|(id, bytes)| {
                    let labels = bytes.and_then(|Bytes(bytes)| {
                        Ok(Arc::new(vf_pb::SafetyLabelMap::decode(bytes.as_slice())?))
                    });
                    (id, labels)
                })
                .collect();
            label_batch(tweet_ids, labels)
        }

        async fn viewer(
            &self,
            viewer_id: u64,
            fields: &[QueryFields],
        ) -> RawHydrationBatch<DecodedViewer> {
            self.asked::<exchange::GizmoduckViewer>(fields);
            let viewer = result(self.answer::<exchange::GizmoduckViewer>(&viewer_id));
            viewer_batch(viewer_id, viewer, fields)
        }

        async fn users(
            &self,
            user_ids: &[u64],
            fields: &[QueryFields],
        ) -> RawHydrationBatch<DecodedAuthor> {
            self.asked::<exchange::GizmoduckAuthor>(fields);
            author_batch(self.batch::<exchange::GizmoduckAuthor>(user_ids.iter().copied()))
        }

        async fn select_edges(
            &self,
            _viewer_id: u64,
            queries: &[EdgeQuery],
        ) -> Vec<RawHydrationBatch<bool>> {
            queries
                .iter()
                .map(|query| {
                    let answers = query.destination_ids.iter().map(|&destination| {
                        let key = EdgeKey::of(query, destination);
                        (destination, self.answer::<exchange::Flock>(&key))
                    });
                    HydrationBatch::from_hydrated(answers.collect())
                })
                .collect()
        }

        async fn viewer_country(&self, viewer_id: u64) -> RawHydrationBatch<Arc<str>> {
            let country = result(self.answer::<exchange::ViewerCountry>(&viewer_id));
            country_batch(viewer_id, country)
        }

        async fn second_degree(
            &self,
            _viewer_id: u64,
            root_author_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            self.batch::<exchange::Wingman>(root_author_ids.iter().copied())
        }

        async fn community_moderations(
            &self,
            posts: &[CommunityPost],
        ) -> RawHydrationBatch<CommunityModeration> {
            self.batch::<exchange::CommunityModeration>(posts.iter().map(|post| post.tweet_id))
        }

        async fn community_moderators(
            &self,
            _viewer_id: u64,
            community_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            self.batch::<exchange::CommunityModerator>(community_ids.iter().copied())
        }

        async fn community_viewer_removals(
            &self,
            _viewer_id: u64,
            community_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            self.batch::<exchange::CommunityViewerRemoved>(community_ids.iter().copied())
        }

        async fn article_lifecycles(
            &self,
            article_ids: &[u64],
        ) -> RawHydrationBatch<ArticleLifecycle> {
            lifecycle_batch(self.batch::<exchange::ArticleLifecycle>(article_ids.iter().copied()))
        }

        async fn trusted_friends(
            &self,
            _viewer_id: u64,
            list_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            self.batch::<exchange::TrustedFriends>(list_ids.iter().copied())
        }

        async fn outside_places(
            &self,
            _viewer_id: u64,
            place_ids: &[u64],
        ) -> RawHydrationBatch<bool> {
            self.batch::<exchange::UserLocation>(place_ids.iter().copied())
        }
    }

    struct ViewerKeyed<'a> {
        recording: &'a Recording,
        names: Vec<&'static str>,
    }

    impl Visit for ViewerKeyed<'_> {
        fn exchange<X: Exchange>(&mut self) {
            if X::SCOPE == Scope::Viewer {
                let recording = self.recording;
                self.names.extend(owned_names::<X>().filter(|name| {
                    recording.answers.contains_key(*name) || recording.fields.contains_key(*name)
                }));
            }
        }
    }

    impl Recording {
                fn viewer_keyed(&self) -> Vec<&'static str> {
            visit_all(ViewerKeyed {
                recording: self,
                names: vec![],
            })
            .names
        }

                        pub(crate) fn into_replay(self, viewer_id: Option<u64>) -> anyhow::Result<Replay> {
            if viewer_id.is_none() {
                let keyed = self.viewer_keyed();
                anyhow::ensure!(
                    keyed.is_empty(),
                    "a logged-out recording holds viewer-keyed {keyed:?}"
                );
            }
            Ok(Replay {
                recording: self,
                misses: Mutex::default(),
            })
        }

                pub(crate) fn names(&self) -> Vec<&str> {
            let mut names: Vec<&str> = self
                .answers
                .keys()
                .chain(self.fields.keys())
                .map(String::as_str)
                .collect();
            names.sort_unstable();
            names
        }

                pub(crate) fn with_shared(mut self, shared: &Self) -> Self {
            for (name, answers) in &shared.answers {
                let own = self.answers.entry(name.clone()).or_default();
                for (key, answer) in answers {
                    own.entry(key.clone()).or_insert_with(|| answer.clone());
                }
            }
            for (name, fields) in &shared.fields {
                let own = self.fields.entry(name.clone()).or_default();
                for field in fields {
                    if !own.contains(field) {
                        own.push(*field);
                    }
                }
            }
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clients::socialgraph_client::{EdgeDirection, EdgeQuery, Graph};
    use crate::filter::{FilterRequest, FilterTweets};
    use crate::hydration::batch::{HydrationError, RawHydrationBatch};
    use crate::hydration::sources::{Bytes, EdgeKey, InMemorySources, Sources};
    use crate::models::{ClientCapability, RawCandidate, TweetId, ViewerProfile};
    use crate::rules::metrics::Rpc;
    use crate::rules::{RuleEngine, SafetyLevel};
    use std::collections::HashMap;
    use std::fmt;
    use xai_core_entities::entities::{
        ConversationControl, GizmoduckUser, GizmoduckUserResult, PureCoreData,
    };
    use xai_visibility_filtering_proto as vf_pb;

    fn failed<V>() -> Hydrated<V> {
        Hydrated::Failed(HydrationError::Error)
    }

    #[tokio::test]
    async fn a_call_the_recording_lacks_is_named() {
        let recorder = Arc::new(Recorder::default());
        recorder.answered::<exchange::TesPureCore>(iter::once((
            1,
            Hydrated::Found(PureCoreData {
                author_id: 10,
                ..Default::default()
            }),
        )));
        let sources = Arc::new(recorder.take().into_replay(Some(50)).unwrap());
        FilterTweets::new(Arc::<Replay>::clone(&sources), RuleEngine::for_tests())
            .run(FilterRequest {
                viewer_id: Some(50),
                country_code: None,
                client_capability: ClientCapability::default(),
                safety_level: SafetyLevel::TimelineHomeHydration,
                candidates: vec![RawCandidate {
                    tweet_id: TweetId(1),
                    request_author_id: None,
                }],
                rpc: Rpc::FilterTweets,
            })
            .await;

        assert!(sources.misses().contains(&"tweets/1".to_string()));
    }

    #[tokio::test]
    async fn what_the_recorder_observes_replays_as_observed() {
        const VIEWER: u64 = 50;
        let paths = [(20, true), (21, false)];
        let lists = [(7, true), (8, false)];
        let edges = [
            (EdgeDirection::Forward, 30, Hydrated::Found(true)),
            (EdgeDirection::Forward, 31, Hydrated::Found(false)),
            (EdgeDirection::Reverse, 32, Hydrated::Found(true)),
            (EdgeDirection::Reverse, 33, Hydrated::Found(false)),
            (EdgeDirection::Forward, 34, Hydrated::Partial(false)),
        ];
        let query = |direction, destination| EdgeQuery {
            graph: Graph::Follows,
            direction,
            destination_ids: vec![destination],
        };
        let recorder = Arc::new(Recorder::default());
        recorder.answered::<exchange::TesPureCore>(iter::once((1, Hydrated::NotFound)));
        recorder.answered::<exchange::TesConversationControl>(iter::once((1, Hydrated::NotFound)));
        recorder.answered::<exchange::GizmoduckAuthor>(iter::once((10, Hydrated::NotFound)));
        recorder.answered::<exchange::GizmoduckViewer>(iter::once((VIEWER, Hydrated::NotFound)));
        recorder.answered::<exchange::ViewerCountry>(iter::once((
            VIEWER,
            Hydrated::Found("us".to_owned()),
        )));
        recorder.answered::<exchange::Wingman>(
            paths
                .into_iter()
                .map(|(root, path)| (root, Hydrated::Found(path))),
        );
        recorder.answered::<exchange::TrustedFriends>(
            lists
                .into_iter()
                .map(|(list, holds)| (list, Hydrated::Found(holds))),
        );
        recorder.answered::<exchange::Flock>(edges.iter().map(
            |(direction, destination, answer)| {
                (
                    EdgeKey::of(&query(*direction, *destination), *destination),
                    answer.clone(),
                )
            },
        ));
        let recording: Recording =
            serde_json::from_value(serde_json::to_value(recorder.take()).unwrap()).unwrap();
        assert_eq!(recording.unsuccessful(), Vec::<String>::new());
        let sources = recording.into_replay(Some(VIEWER)).unwrap();

        assert!(matches!(
            sources.pure_cores(&[1]).await.hydrated(&1),
            Some(Hydrated::NotFound)
        ));
        assert!(matches!(
            sources.conversation_controls(&[1]).await.hydrated(&1),
            Some(Hydrated::NotFound)
        ));
        assert!(matches!(
            sources.users(&[10], &[]).await.hydrated(&10),
            Some(Hydrated::NotFound)
        ));
        assert!(matches!(
            Sources::viewer(&sources, VIEWER, &[]).await.hydrated(&VIEWER),
            Some(Hydrated::Found(viewer)) if viewer.profile == ViewerProfile::default()
        ));
        assert_eq!(
            sources.viewer_country(VIEWER).await.hydrated(&VIEWER),
            Some(&Hydrated::Found(Arc::from("us")))
        );
        for (root, path) in paths {
            assert_eq!(
                sources.second_degree(VIEWER, &[root]).await.hydrated(&root),
                Some(&Hydrated::Found(path)),
                "second_degree/{root}"
            );
        }
        for (list, holds) in lists {
            assert_eq!(
                sources
                    .trusted_friends(VIEWER, &[list])
                    .await
                    .hydrated(&list),
                Some(&Hydrated::Found(holds)),
                "trusted_friends/{list}"
            );
        }
        for (direction, destination, answer) in edges {
            let answers = sources
                .select_edges(VIEWER, &[query(direction, destination)])
                .await;
            assert_eq!(
                answers
                    .first()
                    .and_then(|batch| batch.hydrated(&destination)),
                Some(&answer),
                "{direction:?}/{destination}"
            );
        }
        assert_eq!(sources.misses(), Vec::<String>::new());
    }

    #[test]
    fn every_answer_a_fallback_cache_could_replace_is_unsuccessful() {
        let recorder = Arc::new(Recorder::default());
        recorder.answered::<exchange::TesPureCore>(
            [(1, failed()), (2, Hydrated::NotFound)].into_iter(),
        );
        recorder.answered::<exchange::TesTweet>(iter::once((1, failed())));
        recorder.answered::<exchange::GizmoduckAuthor>(iter::once((10, failed())));
        recorder.answered::<exchange::GizmoduckViewer>(iter::once((
            50,
            Hydrated::Partial(GizmoduckUser::default()),
        )));
        let flock_left_out = EdgeKey {
            graph: Graph::Follows,
            direction: EdgeDirection::Forward,
            destination: 30,
        };
        recorder
            .answered::<exchange::Flock>(iter::once((flock_left_out, Hydrated::Partial(false))));

        assert_eq!(
            recorder.take().unsuccessful(),
            ["pure_cores/1", "tweets/1", "users/10", "viewers/50"]
        );
    }

    #[test]
    fn a_logged_out_recording_holds_no_viewer_keyed_answer() {
        let recorder = Arc::new(Recorder::default());
        recorder.answered::<exchange::TesPureCore>(iter::once((1, Hydrated::NotFound)));
        let logged_out = recorder.take();
        recorder.answered::<exchange::GizmoduckViewer>(iter::once((50, Hydrated::NotFound)));
        let planted = logged_out.clone().with_shared(&recorder.take());

        assert!(logged_out.into_replay(None).is_ok());
        assert!(planted.clone().into_replay(Some(50)).is_ok());
        assert_eq!(
            planted.into_replay(None).err().map(|e| e.to_string()),
            Some(r#"a logged-out recording holds viewer-keyed ["viewers"]"#.to_owned())
        );
    }

    mod completeness {
        use super::*;
        use crate::clients::about_this_account_client::AboutThisAccountClient;
        use crate::clients::article_client::ArticleClient;
        use crate::clients::socialgraph_client::SocialgraphClient;
        use crate::clients::trusted_friends_client::TrustedFriendsClient;
        use crate::clients::user_location_client::UserLocationClient;
        use crate::clients::wingman_client::WingmanClient;
        use crate::hydration::community_source::{CommunityPost, CommunitySource};
        use crate::hydration::sources::{ProdSources, control};
        use crate::hydration::tweet_source::TweetSource;
        use crate::safety_label_source::SafetyLabelSource;
        use crate::safety_label_source::lookup::{
            LookupError, ManhattanLookup, RemoteSource, TwemcacheLookup,
        };
        use crate::safety_label_source::types::{FailureKind, ManhattanOutcome, TwemcacheOutcome};
        use rustc_hash::FxHashMap;
        use std::any::TypeId;
        use std::collections::HashSet;
        use std::hash::Hash;
        use thrift::protocol::{
            TBinaryOutputProtocol, TFieldIdentifier, TListIdentifier, TOutputProtocol,
            TStructIdentifier, TType,
        };
        use tonic::metadata::MetadataMap;
        use tonic::{Request, Response, Status};
        use tower::ServiceExt;
        use tower::util::BoxCloneSyncService;
        use xai_core_entities::entities::{
            ApiCounts, CashtagAttachments, ConversationControlArm, EditControl,
            EscherbirdEntityAnnotation, ExclusiveTweetControl, GizmoduckAccount, MediaEntities,
            PCFLabel, QuotedTweet, ReactionContext, Safety, TakedownReason, TrustedFriendsControl,
            UrlEntities, UserResponseState,
        };
        use xai_core_entities::gizmoduck_client::{
            GizmoduckClient, LookupContext, MockGizmoduckClient, UserFields, ViewerData,
        };
        use xai_core_entities::tweet_entity_service_client::{MockTESClient, TESClient};
        use xai_strato::StratoGrpc;
        use xai_strato::strato_proto::rpc_server::{Rpc, RpcServer};
        use xai_strato::strato_proto::{Call3, Issue3, ResultList, result};
        use xai_x_thrift::entities::ApiMediaEntity;
        use xai_x_thrift::tweets::ApiPerspective;

        const VIEWER: u64 = 50;
        const AUTHOR: u64 = 10;
        const SUSPENDED_AUTHOR: u64 = 11;
        const SOURCE_AUTHOR: u64 = 12;
        const REPLIED_AUTHOR: u64 = 13;
        const INVITED: u64 = 14;
        const LABELED: u64 = 15;
        const TWEET: u64 = 1;
        const SOURCE_TWEET: u64 = 2;
        const REPLIED_TWEET: u64 = 3;
        const CONVERSATION: u64 = 4;
        const COMMUNITY: u64 = 40;
        const ARTICLE: u64 = 60;
        const CIRCLE: u64 = 7;
        const PLACE: u64 = 70;
        const EXCLUSIVE_AUTHOR: u64 = 16;
        const INITIAL_TWEET: u64 = 5;
        const EDIT_TWEET: u64 = 6;
        const ROOT: u64 = 20;
        const FOLLOWER: u64 = 30;
        const VIEWER_AS_ANSWERED: u64 = 51;
        const AUTHOR_AS_ANSWERED: u64 = 18;
        const SUSPENDED_AUTHOR_AS_ANSWERED: u64 = 19;
        const CORE_AUTHOR: u64 = 21;
        const CORE_REPLIED_AUTHOR: u64 = 22;
        const DIRECTED_AT: u64 = 23;
        const CORE_SOURCE_AUTHOR: u64 = 24;
        const TWEET_AS_ANSWERED: u64 = 25;
        const CORE_REPLIED_TWEET: u64 = 26;
        const CORE_SOURCE_TWEET: u64 = 27;
        const PARENT_TWEET: u64 = 28;
        const CORE_CONVERSATION: u64 = 29;
        const SECOND_TWEET: u64 = 33;
        const SECOND_EDIT_TWEET: u64 = 34;
        const SECOND_COMMUNITY: u64 = 41;
                const MISSING: u64 = 8;
                const FAILED: u64 = 9;
                const ABSENT: u64 = 17;
                const EDGES: [(Graph, EdgeDirection, bool); 3] = [
            (Graph::Follows, EdgeDirection::Forward, false),
            (Graph::Follows, EdgeDirection::Reverse, true),
            (Graph::Blocks, EdgeDirection::Forward, true),
        ];
        const FIELDS: [&[QueryFields]; 2] = [&[QueryFields::SAFETY], &[QueryFields::ACCOUNT]];
        const POST: CommunityPost = CommunityPost {
            tweet_id: TWEET,
            author_id: AUTHOR,
            community_id: COMMUNITY,
        };
        const MISSING_POST: CommunityPost = CommunityPost {
            tweet_id: MISSING,
            ..POST
        };
        const FAILED_POST: CommunityPost = CommunityPost {
            tweet_id: FAILED,
            ..POST
        };

                struct Failing<C>(C);

        fn failing<K: Eq + Hash, V>(
            mut answers: HashMap<K, anyhow::Result<V>>,
            key: impl Fn(u64) -> K,
        ) -> HashMap<K, anyhow::Result<V>> {
            answers.remove(&key(ABSENT));
            if let Some(answer) = answers.get_mut(&key(FAILED)) {
                *answer = Err(anyhow!("{FAILED} fails"));
            }
            answers
        }

                        macro_rules! failing_tes {
            ($($method:ident => $value:ty),* $(,)?) => {
                #[tonic::async_trait]
                impl TESClient for Failing<MockTESClient> {
                    $(
                        async fn $method(
                            &self,
                            ids: Vec<u64>,
                        ) -> HashMap<u64, anyhow::Result<Option<$value>>> {
                            failing(self.0.$method(ids).await, |id| id)
                        }
                    )*

                    async fn get_core_data_and_api_counts(
                        &self,
                        tweet_id: u64,
                    ) -> (
                        anyhow::Result<Option<PureCoreData>>,
                        anyhow::Result<Option<ApiCounts>>,
                    ) {
                        self.0.get_core_data_and_api_counts(tweet_id).await
                    }

                    async fn get_status_perspectives(
                        &self,
                        ids: Vec<u64>,
                        metadata: Option<&MetadataMap>,
                    ) -> HashMap<u64, anyhow::Result<Option<ApiPerspective>>> {
                        failing(self.0.get_status_perspectives(ids, metadata).await, |id| id)
                    }

                    async fn get_api_media_entities(
                        &self,
                        ids: Vec<u64>,
                        metadata: Option<&MetadataMap>,
                    ) -> HashMap<u64, anyhow::Result<Option<Vec<ApiMediaEntity>>>> {
                        failing(self.0.get_api_media_entities(ids, metadata).await, |id| id)
                    }
                }
            };
        }

        failing_tes! {
            get_tweet_core_datas => PureCoreData,
            get_tweet_media_entities => MediaEntities,
            get_subscription_author_ids => u64,
            get_conversation_controls => ConversationControl,
            get_quoted_tweets => QuotedTweet,
            get_reaction_context => ReactionContext,
            get_min_video_durations => i64,
            get_media_count => i64,
            get_nullcast => bool,
            get_community => i64,
            get_nsfw_user => bool,
            get_nsfw_admin => bool,
            get_has_takedown => bool,
            get_takedown_country_codes => Vec<String>,
            get_takedown_reasons => Vec<TakedownReason>,
            get_language_code => String,
            get_api_counts => ApiCounts,
            get_is_article => bool,
            get_is_premium => bool,
            get_urls => UrlEntities,
            get_cashtag_attachments => CashtagAttachments,
            get_exclusive_controls => ExclusiveTweetControl,
            get_trusted_friends_controls => TrustedFriendsControl,
            get_grok_post_ids => String,
            get_edit_control => EditControl,
            get_escherbird_entity_annotations => Vec<EscherbirdEntityAnnotation>,
        }

        type Users = HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>>;

        #[tonic::async_trait]
        impl GizmoduckClient for Failing<MockGizmoduckClient> {
            async fn get_users_with_context(
                &self,
                ids: Vec<i64>,
                context: Option<LookupContext>,
                fields: &[QueryFields],
            ) -> Users {
                let users = self.0.get_users_with_context(ids, context, fields).await;
                failing(users, u64::cast_signed)
            }

            async fn get_users(&self, _: Vec<i64>) -> Users {
                unreachable!()
            }

            async fn get_users_with_perspective(&self, _: i64, _: Vec<i64>) -> Users {
                unreachable!()
            }

            async fn get_viewer_roles(&self, _: u64) -> anyhow::Result<Vec<String>> {
                unreachable!()
            }

            async fn get_viewer_data(&self, _: u64) -> anyhow::Result<ViewerData> {
                unreachable!()
            }

            async fn get_viewer_data_with_fields(
                &self,
                _: u64,
                _: &[QueryFields],
            ) -> anyhow::Result<ViewerData> {
                unreachable!()
            }

            async fn get_pcf_labels(&self, _: Vec<i64>) -> HashMap<i64, anyhow::Result<PCFLabel>> {
                unreachable!()
            }

            async fn get_profile_description_languages(
                &self,
                _: Vec<i64>,
            ) -> HashMap<i64, anyhow::Result<Option<String>>> {
                unreachable!()
            }

            async fn get_user_fields(
                &self,
                _: Vec<i64>,
            ) -> HashMap<i64, anyhow::Result<UserFields>> {
                unreachable!()
            }

            async fn get_by_screen_name(
                &self,
                _: &str,
            ) -> anyhow::Result<Option<GizmoduckUserResult>> {
                unreachable!()
            }
        }

        struct Fake;

        #[tonic::async_trait]
        impl AboutThisAccountClient for Fake {
            async fn tfe_top_country(&self, viewer_id: u64) -> anyhow::Result<Option<String>> {
                match viewer_id {
                    MISSING => Ok(None),
                    FAILED => Err(anyhow!("country of {viewer_id} fails")),
                    _ => Ok(Some("us".to_owned())),
                }
            }
        }

        #[tonic::async_trait]
        impl ArticleClient for Fake {
            async fn lifecycles(&self, ids: &[u64]) -> HashMap<u64, anyhow::Result<Option<i32>>> {
                const SOFT_DELETED: i32 = 2;
                ids.iter()
                    .map(|&id| {
                        let lifecycle = match id {
                            MISSING => Ok(None),
                            FAILED => Err(anyhow!("article {id} fails")),
                            _ => Ok(Some(SOFT_DELETED)),
                        };
                        (id, lifecycle)
                    })
                    .collect()
            }
        }

        #[tonic::async_trait]
        impl TrustedFriendsClient for Fake {
            async fn batch_is_member_or_owner(
                &self,
                _: u64,
                ids: &[u64],
            ) -> Vec<anyhow::Result<bool>> {
                ids.iter()
                    .map(|&id| match id {
                        MISSING => Ok(false),
                        FAILED => Err(anyhow!("list {id} fails")),
                        _ => Ok(true),
                    })
                    .collect()
            }
        }

        #[tonic::async_trait]
        impl UserLocationClient for Fake {
            async fn places(&self, user_id: u64) -> anyhow::Result<HashSet<u64>> {
                match user_id {
                    FAILED => Err(anyhow!("location of {user_id} fails")),
                    _ => Ok(HashSet::from([PLACE])),
                }
            }
        }

        #[tonic::async_trait]
        impl WingmanClient for Fake {
            async fn batch_exists_intersect(
                &self,
                _: u64,
                ids: &[u64],
            ) -> Option<Vec<wingman_client::Exists>> {
                let exists = ids.iter().map(|&id| match id {
                    MISSING => wingman_client::Exists::NotFound,
                    FAILED => wingman_client::Exists::ItemError,
                    _ => wingman_client::Exists::Found,
                });
                Some(exists.collect())
            }
        }

                        #[tonic::async_trait]
        impl SocialgraphClient for Fake {
            async fn select_edges(
                &self,
                _: u64,
                queries: &[EdgeQuery],
            ) -> Option<Vec<Option<HashSet<u64>>>> {
                let asks = |query: &EdgeQuery, id| query.destination_ids.contains(&id);
                if queries.iter().any(|query| asks(query, FAILED)) {
                    return None;
                }
                let sets = queries.iter().map(|query| {
                    let holds = EDGES.contains(&(query.graph, query.direction, true));
                    let set = query.destination_ids.iter().copied().filter(|_| holds);
                    (!asks(query, MISSING)).then(|| set.collect())
                });
                Some(sets.collect())
            }
        }

        #[tonic::async_trait]
        impl TwemcacheLookup for Fake {
            async fn get(&self, ids: &[u64]) -> FxHashMap<u64, TwemcacheOutcome> {
                ids.iter()
                    .map(|&id| {
                        let outcome = match id {
                            MISSING => TwemcacheOutcome::NotFound,
                            _ => TwemcacheOutcome::Miss,
                        };
                        (id, outcome)
                    })
                    .collect()
            }
        }

        #[tonic::async_trait]
        impl ManhattanLookup for Fake {
            async fn get(&self, ids: &[u64]) -> FxHashMap<u64, ManhattanOutcome> {
                let label = vf_pb::SafetyLabel {
                    applicable_users: vec![LABELED.cast_signed()],
                    ..Default::default()
                };
                let labels = vf_pb::SafetyLabelMap {
                    labels: [(1, label)].into(),
                };
                ids.iter()
                    .map(|&id| {
                        let outcome = match id {
                            FAILED => ManhattanOutcome::Failure(LookupError::new(
                                FailureKind::ManhattanFetch,
                                "fails",
                            )),
                            _ => ManhattanOutcome::Resolved(labels.clone()),
                        };
                        (id, outcome)
                    })
                    .collect()
            }
        }

        enum Thrift {
            Struct(Vec<(i16, Thrift)>),
            Bool(bool),
            I64(u64),
            I64s(Vec<u64>),
            Str(String),
        }

        impl Thrift {
            fn kind(&self) -> TType {
                match self {
                    Self::Struct(_) => TType::Struct,
                    Self::Bool(_) => TType::Bool,
                    Self::I64(_) => TType::I64,
                    Self::I64s(_) => TType::List,
                    Self::Str(_) => TType::String,
                }
            }

            fn write(&self, proto: &mut dyn TOutputProtocol) {
                match self {
                    Self::Struct(fields) => {
                        proto
                            .write_struct_begin(&TStructIdentifier::new(""))
                            .unwrap();
                        for (id, value) in fields {
                            proto
                                .write_field_begin(&TFieldIdentifier::new("", value.kind(), *id))
                                .unwrap();
                            value.write(proto);
                            proto.write_field_end().unwrap();
                        }
                        proto.write_field_stop().unwrap();
                        proto.write_struct_end().unwrap();
                    }
                    Self::Bool(value) => proto.write_bool(*value).unwrap(),
                    Self::I64(value) => proto.write_i64(value.cast_signed()).unwrap(),
                    Self::I64s(values) => {
                        let len = i32::try_from(values.len()).unwrap();
                        proto
                            .write_list_begin(&TListIdentifier::new(TType::I64, len))
                            .unwrap();
                        for value in values {
                            proto.write_i64(value.cast_signed()).unwrap();
                        }
                        proto.write_list_end().unwrap();
                    }
                    Self::Str(value) => proto.write_string(value).unwrap(),
                }
            }
        }

                fn mval(value: Option<Thrift>) -> Vec<u8> {
            let value = value.map(|value| (118, Thrift::Struct(vec![(26900, value)])));
            let envelope = [2556, 4]
                .into_iter()
                .fold(Thrift::Struct(value.into_iter().collect()), |inner, id| {
                    Thrift::Struct(vec![(id, inner)])
                });
            let mut bytes = vec![];
            envelope.write(&mut TBinaryOutputProtocol::new(&mut bytes, false));
            bytes
        }

        fn tweet() -> Thrift {
            use Thrift::{Bool, I64, I64s, Str, Struct};
            let core_data = Struct(vec![
                (1, I64(CORE_AUTHOR)),
                (
                    5,
                    Struct(vec![
                        (1, I64(CORE_REPLIED_TWEET)),
                        (2, I64(CORE_REPLIED_AUTHOR)),
                    ]),
                ),
                (6, Struct(vec![(1, I64(DIRECTED_AT))])),
                (
                    7,
                    Struct(vec![
                        (1, I64(CORE_SOURCE_TWEET)),
                        (2, I64(CORE_SOURCE_AUTHOR)),
                        (3, I64(PARENT_TWEET)),
                    ]),
                ),
                (9, Bool(true)),
                (14, I64(CORE_CONVERSATION)),
            ]);
            Struct(vec![
                (1, I64(TWEET_AS_ANSWERED)),
                (2, core_data),
                (
                    125,
                    Struct(vec![(1, I64s(vec![COMMUNITY, SECOND_COMMUNITY]))]),
                ),
                (155, Struct(vec![(1, I64(EXCLUSIVE_AUTHOR))])),
                (156, Struct(vec![(1, I64(CIRCLE))])),
                (
                    157,
                    Struct(vec![(
                        2,
                        Struct(vec![
                            (1, I64(INITIAL_TWEET)),
                            (2, Struct(vec![(1, I64s(vec![EDIT_TWEET]))])),
                        ]),
                    )]),
                ),
                (170, Struct(vec![(1, I64(ARTICLE))])),
                (172, Struct(vec![(1, Str(format!("{PLACE:x}")))])),
            ])
        }

        fn second_tweet() -> Thrift {
            use Thrift::{I64s, Struct};
            Struct(vec![
                (2, Struct(vec![])),
                (
                    157,
                    Struct(vec![(1, Struct(vec![(1, I64s(vec![SECOND_EDIT_TWEET]))]))]),
                ),
            ])
        }

        struct Strato;

        #[tonic::async_trait]
        impl Rpc for Strato {
            async fn issue3(
                &self,
                request: Request<Issue3>,
            ) -> Result<Response<ResultList>, Status> {
                use Thrift::{Bool, Struct};
                fn asks(call: &Call3, id: u64) -> bool {
                    let key = xai_strato::encode(&(id, ()));
                    call.args.first().is_some_and(|asked| *asked == key)
                }
                let results = request
                    .get_ref()
                    .calls
                    .iter()
                    .filter(|call| !asks(call, ABSENT))
                    .map(|call| {
                        if asks(call, FAILED) {
                            return Ok(xai_strato::strato_proto::Result {
                                result_type: Some(result::ResultType::Err(
                                    xai_strato::strato_proto::Error {
                                        code: 5,
                                        message: "fails".to_owned(),
                                    },
                                )),
                            });
                        }
                        let value = match call.path.as_str() {
                            _ if asks(call, MISSING) => None,
                            "tweetypie/federated/tweetForVisibility.Tweet"
                                if asks(call, SECOND_TWEET) =>
                            {
                                Some(second_tweet())
                            }
                            "tweetypie/federated/tweetForVisibility.Tweet" => Some(tweet()),
                            "communities/moderationState.TweetCommunityRelationship"
                            | "communities/moderationState.UserCommunityRelationship" => {
                                Some(Struct(vec![(2, Struct(vec![]))]))
                            }
                            "communities/visibility/visibilityFeatures.Community" => {
                                Some(Struct(vec![(1, Struct(vec![(4, Bool(true))]))]))
                            }
                            "communities/isRemoved.Community" => Some(Bool(true)),
                            path => return Err(Status::unimplemented(path)),
                        };
                        Ok(xai_strato::strato_proto::Result {
                            result_type: Some(result::ResultType::Ok(mval(value).into())),
                        })
                    })
                    .collect::<Result<_, _>>()?;
                Ok(Response::new(ResultList { results }))
            }
        }

        fn strato() -> StratoGrpc {
            StratoGrpc::from_boxed_service(BoxCloneSyncService::new(ServiceExt::<
                tonic::codegen::http::Request<tonic::body::Body>,
            >::map_err(
                RpcServer::new(Strato),
                |never| match never {},
            )))
        }

        fn prod_sources() -> ProdSources {
            let found = |user| {
                Some(GizmoduckUserResult {
                    user: Some(user),
                    response_state: Some(UserResponseState::Found),
                })
            };
            let tes = MockTESClient {
                core_data: HashMap::from([(
                    TWEET,
                    Some(PureCoreData {
                        author_id: AUTHOR,
                        source_tweet_id: Some(SOURCE_TWEET),
                        source_user_id: Some(SOURCE_AUTHOR),
                        in_reply_to_tweet_id: Some(REPLIED_TWEET),
                        in_reply_to_user_id: Some(REPLIED_AUTHOR),
                        conversation_id: Some(CONVERSATION),
                        ..Default::default()
                    }),
                )]),
                conversation_controls: HashMap::from([(
                    TWEET,
                    Some(ConversationControl {
                        invited_user_ids: vec![INVITED],
                        ..control(ConversationControlArm::ByInvitation, AUTHOR, &[])
                    }),
                )]),
                ..Default::default()
            };
            let gizmoduck = MockGizmoduckClient {
                users: HashMap::from([
                    (
                        VIEWER.cast_signed(),
                        found(GizmoduckUser {
                            user_id: VIEWER_AS_ANSWERED,
                            account: GizmoduckAccount {
                                nsfw_view: true,
                                country_code: Some("US".to_owned()),
                                ..Default::default()
                            },
                            ..Default::default()
                        }),
                    ),
                    (
                        AUTHOR.cast_signed(),
                        found(GizmoduckUser {
                            user_id: AUTHOR_AS_ANSWERED,
                            ..Default::default()
                        }),
                    ),
                    (
                        SUSPENDED_AUTHOR.cast_signed(),
                        found(GizmoduckUser {
                            user_id: SUSPENDED_AUTHOR_AS_ANSWERED,
                            safety: Safety {
                                suspended: true,
                                ..Default::default()
                            },
                            ..Default::default()
                        }),
                    ),
                ]),
                ..Default::default()
            };
            let fake = Arc::new(Fake);
            ProdSources::new(
                Arc::new(Failing(tes)),
                TweetSource {
                    grpc_client: Arc::new(strato()),
                },
                Arc::new(Failing(gizmoduck)),
                Arc::clone(&fake) as _,
                Arc::clone(&fake) as _,
                Arc::clone(&fake) as _,
                Arc::clone(&fake) as _,
                Arc::clone(&fake) as _,
                Arc::clone(&fake) as _,
                Arc::new(SafetyLabelSource::new(
                    Arc::new(RemoteSource::new(Arc::clone(&fake), fake)),
                    None,
                )),
                CommunitySource {
                    grpc_client: strato(),
                },
                None,
                None,
            )
        }

        fn shown<V: fmt::Debug>(batch: &RawHydrationBatch<V>, keys: &[u64]) -> String {
            let shown: Vec<String> = keys
                .iter()
                .map(|key| format!("{key}: {:?}", batch.hydrated(key)))
                .collect();
            shown.join(", ")
        }

        async fn ask(sources: &dyn Sources, source: Source) -> String {
            match source {
                Source::TesPureCore => {
                    let keys = [TWEET, MISSING, FAILED, ABSENT];
                    shown(&sources.pure_cores(&keys).await, &keys)
                }
                Source::TesTweet => {
                    let keys = [TWEET, SECOND_TWEET, MISSING, FAILED, ABSENT];
                    shown(&sources.tweets(&keys).await, &keys)
                }
                Source::TesConversationControl => {
                    let keys = [TWEET, MISSING, FAILED, ABSENT];
                    shown(&sources.conversation_controls(&keys).await, &keys)
                }
                Source::SafetyLabels => {
                    let keys = [TWEET, MISSING, FAILED];
                    shown(&sources.safety_labels(&keys).await, &keys)
                }
                Source::GizmoduckViewer => {
                    let asked = FIELDS
                        .map(|fields| (VIEWER, fields))
                        .into_iter()
                        .chain([MISSING, FAILED].map(|viewer| (viewer, FIELDS[0])));
                    let mut shown_viewers = vec![];
                    for (viewer_id, fields) in asked {
                        let viewer = sources
                            .viewer(viewer_id, fields)
                            .await
                            .map(|viewer| (viewer.profile, viewer.has_age_verified_18_label));
                        shown_viewers.push(shown(&viewer, &[viewer_id]));
                    }
                    shown_viewers.join(", ")
                }
                Source::GizmoduckAuthor => {
                    let asked: [&[u64]; 2] =
                        [&[AUTHOR, MISSING, FAILED, ABSENT], &[SUSPENDED_AUTHOR]];
                    let mut shown_authors = vec![];
                    for (authors, fields) in asked.into_iter().zip(FIELDS) {
                        shown_authors.push(shown(&sources.users(authors, fields).await, authors));
                    }
                    shown_authors.join(", ")
                }
                Source::Flock => {
                    let query = |graph, direction, destination| EdgeQuery {
                        graph,
                        direction,
                        destination_ids: vec![destination],
                    };
                    let held: Vec<EdgeQuery> = EDGES
                        .map(|(graph, direction, _)| query(graph, direction, FOLLOWER))
                        .into_iter()
                        .chain([query(Graph::Follows, EdgeDirection::Forward, MISSING)])
                        .collect();
                    let failed = vec![query(Graph::Follows, EdgeDirection::Forward, FAILED)];
                    let mut shown_edges = vec![];
                    for queries in [held, failed] {
                        let batches = sources.select_edges(VIEWER, &queries).await;
                        for (query, batch) in queries.iter().zip(&batches) {
                            shown_edges.push(shown(batch, &query.destination_ids));
                        }
                    }
                    shown_edges.join(", ")
                }
                Source::ViewerCountry => {
                    let mut shown_countries = vec![];
                    for viewer in [VIEWER, MISSING, FAILED] {
                        let country = sources.viewer_country(viewer).await;
                        shown_countries.push(shown(&country, &[viewer]));
                    }
                    shown_countries.join(", ")
                }
                Source::Wingman => {
                    let keys = [ROOT, MISSING, FAILED];
                    shown(&sources.second_degree(VIEWER, &keys).await, &keys)
                }
                Source::CommunityModeration => {
                    let posts = [POST, MISSING_POST, FAILED_POST];
                    let batch = sources.community_moderations(&posts).await;
                    shown(&batch, &posts.map(|post| post.tweet_id))
                }
                Source::CommunityModerator => {
                    let keys = [COMMUNITY, MISSING, FAILED];
                    shown(&sources.community_moderators(VIEWER, &keys).await, &keys)
                }
                Source::CommunityViewerRemoved => {
                    let keys = [COMMUNITY, MISSING, FAILED];
                    shown(
                        &sources.community_viewer_removals(VIEWER, &keys).await,
                        &keys,
                    )
                }
                Source::ArticleLifecycle => {
                    let keys = [ARTICLE, MISSING, FAILED];
                    shown(&sources.article_lifecycles(&keys).await, &keys)
                }
                Source::TrustedFriends => {
                    let keys = [CIRCLE, MISSING, FAILED];
                    shown(&sources.trusted_friends(VIEWER, &keys).await, &keys)
                }
                Source::UserLocation => {
                    let keys = [PLACE, MISSING];
                    let located = shown(&sources.outside_places(VIEWER, &keys).await, &keys);
                    let failed = shown(&sources.outside_places(FAILED, &[FAILED]).await, &[FAILED]);
                    format!("{located}, {failed}")
                }
            }
        }

        async fn answered() -> (Vec<(Source, String)>, Recording) {
            let recorder = Arc::new(Recorder::default());
            let prod = prod_sources().observed(Arc::clone(&recorder));
            let mut answers = vec![];
            for &source in Source::VARIANTS {
                answers.push((source, ask(&prod, source).await));
            }
            (answers, recorder.take())
        }

        struct Forget<'a>(&'a mut Recording);

        impl Visit for Forget<'_> {
            fn exchange<X: Exchange>(&mut self) {
                self.0.answers.remove(X::NAME);
            }
        }

        fn without(recording: &Recording, source: Source) -> Recording {
            let mut recording = recording.clone();
            visit(source, &mut Forget(&mut recording));
            recording
        }

        fn defaults() -> InMemorySources {
            let user = || GizmoduckUserResult {
                user: Some(GizmoduckUser::default()),
                response_state: Some(UserResponseState::Found),
            };
            InMemorySources::default()
                .pure_core(TWEET, PureCoreData::default())
                .labels(TWEET, vf_pb::SafetyLabelMap::default())
                .viewer(VIEWER, GizmoduckUser::default())
                .user(AUTHOR, user())
                .user(SUSPENDED_AUTHOR, user())
                .country(VIEWER, "")
        }

        #[tokio::test]
        async fn no_fake_answers_as_defaults_or_a_replay_without_its_source() {
            let (answers, recording) = answered().await;

            let mut indistinct = vec![];
            for (source, answer) in answers {
                let replay = without(&recording, source)
                    .into_replay(Some(VIEWER))
                    .unwrap();
                let replayed = ask(&replay, source).await;
                if replayed == answer || ask(&defaults(), source).await == answer {
                    indistinct.push(source);
                }
            }
            assert_eq!(indistinct, Vec::<Source>::new());
        }

        #[tokio::test]
        async fn a_recording_keeps_the_corpus_shape_and_names_failed_keys() {
            let (_, recording) = answered().await;
            let json = serde_json::to_value(&recording).unwrap();

            assert_eq!(
                json["trusted_friends"],
                serde_json::json!({"7": {"found": true}, "8": {"found": false}, "9": "failed"})
            );
            assert_eq!(
                json["edges"],
                serde_json::json!({
                    "blocks/Forward/30": {"found": true},
                    "follows/Forward/30": {"found": false},
                    "follows/Forward/8": {"partial": false},
                    "follows/Forward/9": "failed",
                    "follows/Reverse/30": {"found": true},
                })
            );
            for fields in ["viewer_fields", "user_fields"] {
                assert_eq!(json[fields], serde_json::json!(["sAFETY", "aCCOUNT"]));
            }
            for bytes in ["tweets", "safety_labels"] {
                let hex = json[bytes][TWEET.to_string()]["found"].as_str();
                assert!(
                    hex.is_some_and(|hex| hex.bytes().all(|byte| byte.is_ascii_hexdigit())),
                    "{bytes}: {hex:?}"
                );
            }
            assert_eq!(
                recording.unsuccessful(),
                [
                    "article_lifecycles/9",
                    "community_moderations/9",
                    "community_moderators/9",
                    "community_viewer_removals/9",
                    "conversation_controls/17",
                    "conversation_controls/9",
                    "edges/follows/Forward/9",
                    "outside_places/9",
                    "pure_cores/17",
                    "pure_cores/9",
                    "safety_labels/9",
                    "second_degree/9",
                    "trusted_friends/9",
                    "tweets/17",
                    "tweets/9",
                    "users/17",
                    "users/9",
                    "viewer_countries/9",
                    "viewers/9"
                ]
            );
        }

        #[tokio::test]
        async fn every_source_records_each_answer_its_backend_gives() {
            let (_, recording) = answered().await;
            let kinds: BTreeMap<&str, BTreeSet<&str>> = recording
                .answers
                .iter()
                .map(|(name, answers)| {
                    let kinds = answers.values().map(|answer| match answer {
                        Answer::Found(_) => "found",
                        Answer::Partial(_) => "partial",
                        Answer::NotFound => "not_found",
                        Answer::Failed => "failed",
                    });
                    (name.as_str(), kinds.collect())
                })
                .collect();

            let found_or_failed = || BTreeSet::from(["found", "failed"]);
            let every = || BTreeSet::from(["found", "not_found", "failed"]);
            assert_eq!(
                kinds,
                BTreeMap::from([
                    ("article_lifecycles", every()),
                    ("community_moderations", found_or_failed()),
                    ("community_moderators", found_or_failed()),
                    ("community_viewer_removals", found_or_failed()),
                    ("conversation_controls", every()),
                    ("edges", BTreeSet::from(["found", "partial", "failed"])),
                    ("outside_places", found_or_failed()),
                    ("pure_cores", every()),
                    ("safety_labels", found_or_failed()),
                    ("second_degree", found_or_failed()),
                    ("trusted_friends", found_or_failed()),
                    ("tweets", found_or_failed()),
                    ("users", every()),
                    ("viewer_countries", every()),
                    ("viewers", every()),
                ])
            );
        }

        #[tokio::test]
        async fn a_logged_out_replay_refuses_every_viewer_keyed_answer() {
            let (_, recording) = answered().await;

            assert_eq!(
                recording.into_replay(None).err().map(|e| e.to_string()),
                Some(
                    "a logged-out recording holds viewer-keyed [\"viewers\", \"viewer_fields\", \
                     \"edges\", \"viewer_countries\", \"second_degree\", \
                     \"community_moderators\", \"community_viewer_removals\", \"trusted_friends\", \
                     \"outside_places\"]"
                        .to_owned()
                )
            );
        }

        #[tokio::test]
        async fn shared_json_takes_the_viewer_free_answers() {
            let (_, recording) = answered().await;
            let (case, shared) = recording.clone().split_shared();

            assert_eq!(
                shared.names(),
                [
                    "conversation_controls",
                    "pure_cores",
                    "safety_labels",
                    "tweets",
                    "user_fields",
                    "users"
                ]
            );
            assert_eq!(
                case.names(),
                [
                    "article_lifecycles",
                    "community_moderations",
                    "community_moderators",
                    "community_viewer_removals",
                    "edges",
                    "outside_places",
                    "second_degree",
                    "trusted_friends",
                    "viewer_countries",
                    "viewer_fields",
                    "viewers"
                ]
            );
            assert_eq!(case.with_shared(&shared), recording);
        }

        #[tokio::test]
        async fn every_source_replays_as_prod_answered() {
            let (answers, recorded) = answered().await;
            let recording: Recording =
                serde_json::from_value(serde_json::to_value(recorded).unwrap()).unwrap();

            let mut unrecorded = vec![];
            let mut diffs = vec![];
            for (source, answer) in answers {
                let replay = recording.clone().into_replay(Some(VIEWER)).unwrap();
                let replayed = ask(&replay, source).await;
                let misses = replay.misses();
                if replayed != answer || !misses.is_empty() {
                    unrecorded.push(source);
                    diffs.push(format!(
                        "{source:?}: prod {answer}, replay {replayed}, misses {misses:?}"
                    ));
                }
            }
            assert_eq!(unrecorded, Vec::<Source>::new(), "{diffs:#?}");
        }

                fn in_bytes(name: &str) -> Option<Vec<Id>> {
            let users = [
                CORE_AUTHOR,
                CORE_REPLIED_AUTHOR,
                DIRECTED_AT,
                CORE_SOURCE_AUTHOR,
                EXCLUSIVE_AUTHOR,
                CIRCLE,
            ];
            let tweets = [
                TWEET,
                SECOND_TWEET,
                TWEET_AS_ANSWERED,
                CORE_REPLIED_TWEET,
                CORE_SOURCE_TWEET,
                PARENT_TWEET,
                CORE_CONVERSATION,
                INITIAL_TWEET,
                EDIT_TWEET,
                SECOND_EDIT_TWEET,
            ];
            match name {
                "tweets" => Some(
                    users
                        .map(Id::User)
                        .into_iter()
                        .chain(tweets.map(Id::Tweet))
                        .chain([COMMUNITY, SECOND_COMMUNITY].map(Id::Community))
                        .chain([Id::Article(ARTICLE), Id::Place(PLACE)])
                        .collect(),
                ),
                "safety_labels" => Some(vec![Id::User(LABELED), Id::Tweet(TWEET)]),
                _ => None,
            }
        }

                const NO_IDS: [&str; 1] = [exchange::ArticleLifecycle::NAME];

        fn numbers(json: &serde_json::Value, into: &mut Vec<u64>) {
            match json {
                serde_json::Value::Number(number) => into.extend(
                    number
                        .as_u64()
                        .or_else(|| number.as_i64().map(i64::cast_unsigned))
                        .filter(|&number| number != 0),
                ),
                serde_json::Value::Array(values) => {
                    values.iter().for_each(|value| numbers(value, into));
                }
                serde_json::Value::Object(fields) => {
                    fields.values().for_each(|value| numbers(value, into));
                }
                serde_json::Value::Null
                | serde_json::Value::Bool(_)
                | serde_json::Value::String(_) => {}
            }
        }

                        struct Scan<'a> {
            recording: &'a Recording,
            walked: &'a [Id],
            unwalked: Vec<String>,
        }

        impl Visit for Scan<'_> {
            fn exchange<X: Exchange>(&mut self) {
                let name = X::NAME;
                let Some(answers) = self.recording.answers.get(name) else {
                    self.unwalked.push(format!("{name}: answered nothing"));
                    return;
                };
                let bytes = TypeId::of::<X::Wire>() == TypeId::of::<Bytes>();
                if bytes {
                    let planted = in_bytes(name)
                        .unwrap_or_else(|| panic!("{name}: in_bytes names no id its bytes hold"));
                    self.unwalked.extend(
                        planted
                            .into_iter()
                            .filter(|id| !self.walked.contains(id))
                            .map(|id| format!("{name}: {id:?}")),
                    );
                }
                for (key, answer) in answers {
                    let mut named: Vec<u64> = key
                        .split(|c: char| !c.is_ascii_digit())
                        .filter_map(|digits| digits.parse().ok())
                        .collect();
                    if let Answer::Found(json) | Answer::Partial(json) = answer
                        && !bytes
                        && !NO_IDS.contains(&name)
                    {
                        numbers(json, &mut named);
                    }
                    let walked = |number: &u64| {
                        self.walked.iter().any(|&id| {
                            let (Id::User(id)
                            | Id::Tweet(id)
                            | Id::Community(id)
                            | Id::Article(id)
                            | Id::Place(id)) = id;
                            id == *number
                        })
                    };
                    self.unwalked.extend(
                        named
                            .iter()
                            .filter(|number| !walked(number))
                            .map(|number| format!("{name}/{key}: {number}")),
                    );
                }
            }
        }

        #[tokio::test]
        async fn every_id_a_source_records_is_walked() {
            let mut unwalked = vec![];
            for &source in Source::VARIANTS {
                let recorder = Arc::new(Recorder::default());
                ask(&prod_sources().observed(Arc::clone(&recorder)), source).await;
                let recording = recorder.take();
                let walked = recording.ids().unwrap();
                let mut scan = Scan {
                    recording: &recording,
                    walked: &walked,
                    unwalked: vec![],
                };
                visit(source, &mut scan);
                unwalked.extend(
                    scan.unwalked
                        .into_iter()
                        .map(|unwalked| format!("{source:?}: {unwalked}")),
                );
            }
            assert_eq!(unwalked, Vec::<String>::new());
        }
    }
}
