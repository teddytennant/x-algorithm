use crate::clients::socialgraph_client::{EdgeDirection, EdgeQuery, Graph};
use crate::hydration::tweet_source::decode_tweet;
use crate::models;
use anyhow::anyhow;
use prost::Message;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{self, Display};
use std::iter;
use std::str::FromStr;
use strum::VariantArray;
use thrift::protocol::{TInputProtocol, TOutputProtocol, TSerializable, TType};
use xai_core_entities::entities::{
    ConversationControl, EditControl, GizmoduckUser, GizmoduckUserResult, PureCoreData,
};
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Id {
    User(u64),
    Tweet(u64),
    Community(u64),
    Article(u64),
    Place(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    Shared,
    Case,
    Viewer,
}

pub(crate) trait Exchange {
    const NAME: &'static str;
    const SCOPE: Scope;
    const FIELDS: Option<&'static str> = None;
    const KEEPS_PARTIAL: bool = false;
    type Key: Display + FromStr;
    type Wire: Serialize + DeserializeOwned + 'static;

    fn ids(key: &Self::Key, wire: Option<&Self::Wire>) -> anyhow::Result<Vec<Id>>;
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Bytes(pub(crate) Vec<u8>);

impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl Serialize for Bytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Bytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let hex = String::deserialize(deserializer)?;
        (0..hex.len())
            .step_by(2)
            .map(|at| {
                hex.get(at..at + 2)
                    .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            })
            .collect::<Option<Vec<u8>>>()
            .map(Self)
            .ok_or_else(|| serde::de::Error::custom("odd-length or non-hex bytes"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EdgeKey {
    pub(crate) graph: Graph,
    pub(crate) direction: EdgeDirection,
    pub(crate) destination: u64,
}

impl EdgeKey {
    pub(crate) fn of(query: &EdgeQuery, destination: u64) -> Self {
        Self {
            graph: query.graph,
            direction: query.direction,
            destination,
        }
    }
}

impl Display for EdgeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let graph: &str = self.graph.into();
        write!(f, "{graph}/{:?}/{}", self.direction, self.destination)
    }
}

impl FromStr for EdgeKey {
    type Err = anyhow::Error;

    fn from_str(key: &str) -> anyhow::Result<Self> {
        let mut parts = key.splitn(3, '/');
        let (Some(graph), Some(direction), Some(destination)) =
            (parts.next(), parts.next(), parts.next())
        else {
            anyhow::bail!("{key}: not a graph/direction/destination key");
        };
        Ok(Self {
            graph: *Graph::VARIANTS
                .iter()
                .find(|known| <&str>::from(**known) == graph)
                .ok_or_else(|| anyhow!("{key}: no graph {graph}"))?,
            direction: *EdgeDirection::VARIANTS
                .iter()
                .find(|known| format!("{known:?}") == direction)
                .ok_or_else(|| anyhow!("{key}: no direction {direction}"))?,
            destination: destination.parse()?,
        })
    }
}

pub(crate) struct TesPureCore;

impl Exchange for TesPureCore {
    const NAME: &'static str = "pure_cores";
    const SCOPE: Scope = Scope::Shared;
    type Key = u64;
    type Wire = PureCoreData;

    fn ids(&tweet_id: &u64, core: Option<&PureCoreData>) -> anyhow::Result<Vec<Id>> {
        let linked = core.into_iter().flat_map(|core| {
            [
                Some(Id::User(core.author_id)),
                core.source_user_id.map(Id::User),
                core.in_reply_to_user_id.map(Id::User),
                core.source_tweet_id.map(Id::Tweet),
                core.in_reply_to_tweet_id.map(Id::Tweet),
                core.conversation_id.map(Id::Tweet),
            ]
        });
        Ok(iter::once(Id::Tweet(tweet_id))
            .chain(linked.flatten())
            .collect())
    }
}

pub(crate) struct TesTweet;

impl Exchange for TesTweet {
    const NAME: &'static str = "tweets";
    const SCOPE: Scope = Scope::Shared;
    type Key = u64;
    type Wire = Bytes;

    fn ids(&tweet_id: &u64, bytes: Option<&Bytes>) -> anyhow::Result<Vec<Id>> {
        let mut ids = vec![Id::Tweet(tweet_id)];
        let Some(Bytes(bytes)) = bytes else {
            return Ok(ids);
        };
        match strato_decode::<RawTweetIds>(bytes)? {
            StratoResult::Ok { value, .. } => ids.extend(value.unwrap_or_default().ids),
            StratoResult::Err { code, message } => {
                anyhow::bail!("Strato error code {code}: {message}")
            }
        }
        if let Some(tweet) = decode_tweet(bytes)? {
            ids.extend(
                [
                    tweet.exclusive_conversation_author_id,
                    tweet.trusted_friends_list_id,
                ]
                .into_iter()
                .flatten()
                .map(Id::User),
            );
            ids.extend(edit_ids(tweet.edit_control).into_iter().map(Id::Tweet));
            ids.extend(tweet.narrowcast_place_id.map(Id::Place));
        }
        Ok(ids)
    }
}

fn edit_ids(edit_control: Option<EditControl>) -> Vec<u64> {
    match edit_control {
        Some(EditControl::Initial(initial)) => initial.edit_tweet_ids,
        Some(EditControl::Edit(edit)) => iter::once(edit.initial_tweet_id)
            .chain(
                edit.edit_control_initial
                    .into_iter()
                    .flat_map(|initial| initial.edit_tweet_ids),
            )
            .collect(),
        None => vec![],
    }
}

type IdAt = (&'static [i16], fn(u64) -> Id);

const RAW_ID_PATHS: [IdAt; 11] = [
    (&[1], Id::Tweet),
    (&[2, 1], Id::User),
    (&[2, 5, 1], Id::Tweet),
    (&[2, 5, 2], Id::User),
    (&[2, 6, 1], Id::User),
    (&[2, 7, 1], Id::Tweet),
    (&[2, 7, 2], Id::User),
    (&[2, 7, 3], Id::Tweet),
    (&[2, 14], Id::Tweet),
    (&[125, 1], Id::Community),
    (&[170, 1], Id::Article),
];

#[derive(Default)]
struct RawTweetIds {
    ids: Vec<Id>,
}

impl RawTweetIds {
    fn read(&mut self, proto: &mut dyn TInputProtocol, at: &mut Vec<i16>) -> thrift::Result<()> {
        proto.read_struct_begin()?;
        loop {
            let field = proto.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            at.push(field.id.unwrap_or_default());
            let id = RAW_ID_PATHS
                .iter()
                .find(|(path, _)| *path == at.as_slice())
                .map(|&(_, id)| id);
            match (id, field.field_type) {
                (Some(id), TType::I64) => self.push(id, proto.read_i64()?),
                (Some(id), TType::List) => {
                    let list = proto.read_list_begin()?;
                    if list.element_type != TType::I64 {
                        return Err(not_an_id(at));
                    }
                    for _ in 0..list.size {
                        self.push(id, proto.read_i64()?);
                    }
                    proto.read_list_end()?;
                }
                (Some(_), _) => return Err(not_an_id(at)),
                (None, TType::Struct)
                    if RAW_ID_PATHS.iter().any(|(path, _)| path.starts_with(at)) =>
                {
                    self.read(proto, at)?;
                }
                (None, field_type) => proto.skip(field_type)?,
            }
            at.pop();
            proto.read_field_end()?;
        }
        proto.read_struct_end()
    }

    fn push(&mut self, id: fn(u64) -> Id, value: i64) {
        if value != 0 {
            self.ids.push(id(value.cast_unsigned()));
        }
    }
}

fn not_an_id(at: &[i16]) -> thrift::Error {
    thrift::new_protocol_error(
        thrift::ProtocolErrorKind::InvalidData,
        format!("tweet field {at:?} is not an i64 id"),
    )
}

impl TSerializable for RawTweetIds {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut ids = Self::default();
        ids.read(proto, &mut vec![])?;
        Ok(ids)
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(thrift::new_protocol_error(
            thrift::ProtocolErrorKind::NotImplemented,
            "RawTweetIds is decode-only",
        ))
    }
}

pub(crate) struct TesConversationControl;

impl Exchange for TesConversationControl {
    const NAME: &'static str = "conversation_controls";
    const SCOPE: Scope = Scope::Shared;
    type Key = u64;
    type Wire = ConversationControl;

    fn ids(&tweet_id: &u64, control: Option<&ConversationControl>) -> anyhow::Result<Vec<Id>> {
        let users = control.into_iter().flat_map(|control| {
            iter::once(control.conversation_tweet_author_id)
                .chain(control.invited_user_ids.iter().copied())
        });
        Ok(iter::once(Id::Tweet(tweet_id))
            .chain(users.map(Id::User))
            .collect())
    }
}

pub(crate) struct SafetyLabels;

impl Exchange for SafetyLabels {
    const NAME: &'static str = "safety_labels";
    const SCOPE: Scope = Scope::Shared;
    type Key = u64;
    type Wire = Bytes;

    fn ids(&tweet_id: &u64, bytes: Option<&Bytes>) -> anyhow::Result<Vec<Id>> {
        let labels = match bytes {
            Some(Bytes(bytes)) => vf_pb::SafetyLabelMap::decode(bytes.as_slice())?.labels,
            None => Default::default(),
        };
        anyhow::ensure!(
            !labels.values().any(|label| matches!(
                label.safety_label_source,
                Some(vf_pb::safety_label::SafetyLabelSource::ToolAction(_))
            )),
            "a tool-action label names an employee"
        );
        iter::once(Ok(Id::Tweet(tweet_id)))
            .chain(
                labels
                    .into_values()
                    .flat_map(|label| label.applicable_users)
                    .map(|user| Ok(Id::User(u64::try_from(user)?))),
            )
            .collect()
    }
}

pub(crate) struct GizmoduckViewer;

impl Exchange for GizmoduckViewer {
    const NAME: &'static str = "viewers";
    const SCOPE: Scope = Scope::Viewer;
    const FIELDS: Option<&'static str> = Some("viewer_fields");
    type Key = u64;
    type Wire = GizmoduckUser;

    fn ids(&viewer_id: &u64, user: Option<&GizmoduckUser>) -> anyhow::Result<Vec<Id>> {
        Ok(iter::once(viewer_id)
            .chain(user.map(|user| user.user_id))
            .map(Id::User)
            .collect())
    }
}

pub(crate) struct GizmoduckAuthor;

impl Exchange for GizmoduckAuthor {
    const NAME: &'static str = "users";
    const SCOPE: Scope = Scope::Shared;
    const FIELDS: Option<&'static str> = Some("user_fields");
    type Key = u64;
    type Wire = GizmoduckUserResult;

    fn ids(&user_id: &u64, result: Option<&GizmoduckUserResult>) -> anyhow::Result<Vec<Id>> {
        let answered = result.and_then(|result| result.user.as_ref());
        Ok(iter::once(user_id)
            .chain(answered.map(|user| user.user_id))
            .map(Id::User)
            .collect())
    }
}

pub(crate) struct Flock;

impl Exchange for Flock {
    const NAME: &'static str = "edges";
    const SCOPE: Scope = Scope::Viewer;
    const KEEPS_PARTIAL: bool = true;
    type Key = EdgeKey;
    type Wire = bool;

    fn ids(key: &EdgeKey, _: Option<&bool>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::User(key.destination)])
    }
}

pub(crate) struct ViewerCountry;

impl Exchange for ViewerCountry {
    const NAME: &'static str = "viewer_countries";
    const SCOPE: Scope = Scope::Viewer;
    type Key = u64;
    type Wire = String;

    fn ids(&viewer_id: &u64, _: Option<&String>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::User(viewer_id)])
    }
}

pub(crate) struct Wingman;

impl Exchange for Wingman {
    const NAME: &'static str = "second_degree";
    const SCOPE: Scope = Scope::Viewer;
    type Key = u64;
    type Wire = bool;

    fn ids(&root: &u64, _: Option<&bool>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::User(root)])
    }
}

pub(crate) struct TrustedFriends;

impl Exchange for TrustedFriends {
    const NAME: &'static str = "trusted_friends";
    const SCOPE: Scope = Scope::Viewer;
    type Key = u64;
    type Wire = bool;

    fn ids(&list_id: &u64, _: Option<&bool>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::User(list_id)])
    }
}

pub(crate) struct UserLocation;

impl Exchange for UserLocation {
    const NAME: &'static str = "outside_places";
    const SCOPE: Scope = Scope::Viewer;
    type Key = u64;
    type Wire = bool;

    fn ids(&place_id: &u64, _: Option<&bool>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::Place(place_id)])
    }
}

pub(crate) struct ArticleLifecycle;

impl Exchange for ArticleLifecycle {
    const NAME: &'static str = "article_lifecycles";
    const SCOPE: Scope = Scope::Case;
    type Key = u64;
    type Wire = i32;

    fn ids(&article_id: &u64, _: Option<&i32>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::Article(article_id)])
    }
}

pub(crate) struct CommunityModeration;

impl Exchange for CommunityModeration {
    const NAME: &'static str = "community_moderations";
    const SCOPE: Scope = Scope::Case;
    type Key = u64;
    type Wire = models::CommunityModeration;

    fn ids(&tweet_id: &u64, _: Option<&models::CommunityModeration>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::Tweet(tweet_id)])
    }
}

pub(crate) struct CommunityModerator;

impl Exchange for CommunityModerator {
    const NAME: &'static str = "community_moderators";
    const SCOPE: Scope = Scope::Viewer;
    type Key = u64;
    type Wire = bool;

    fn ids(&community_id: &u64, _: Option<&bool>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::Community(community_id)])
    }
}

pub(crate) struct CommunityViewerRemoved;

impl Exchange for CommunityViewerRemoved {
    const NAME: &'static str = "community_viewer_removals";
    const SCOPE: Scope = Scope::Viewer;
    type Key = u64;
    type Wire = bool;

    fn ids(&community_id: &u64, _: Option<&bool>) -> anyhow::Result<Vec<Id>> {
        Ok(vec![Id::Community(community_id)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vf_pb::safety_label::SafetyLabelSource;

    fn label_ids(source: Option<SafetyLabelSource>) -> anyhow::Result<Vec<Id>> {
        let label = vf_pb::SafetyLabel {
            safety_label_source: source,
            applicable_users: vec![7],
            ..Default::default()
        };
        let labels = vf_pb::SafetyLabelMap {
            labels: [(1, label)].into(),
        };
        SafetyLabels::ids(&1, Some(&Bytes(labels.encode_to_vec())))
    }

    #[test]
    fn a_recorded_tool_action_label_is_refused() {
        assert_eq!(label_ids(None).unwrap(), [Id::Tweet(1), Id::User(7)]);
        let botmaker = SafetyLabelSource::BotmakerAction(vf_pb::BotmakerAction::default());
        assert_eq!(
            label_ids(Some(botmaker)).unwrap(),
            [Id::Tweet(1), Id::User(7)]
        );
        let tool = SafetyLabelSource::ToolAction(vf_pb::ToolAction::default());
        assert_eq!(
            label_ids(Some(tool)).unwrap_err().to_string(),
            "a tool-action label names an employee"
        );
    }
}
