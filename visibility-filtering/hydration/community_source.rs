use crate::models::CommunityModeration;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TOutputProtocol, TSerializable, TStructIdentifier, TType,
};
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::{encode, Bytes, MValCodec, StratoGrpc};
use xai_twittercontext_proto::TwitterContextViewer;

const TWEET_MODERATION: &str = "communities/moderationState.TweetCommunityRelationship";
const AUTHOR_MODERATION: &str = "communities/moderationState.UserCommunityRelationship";
const VISIBILITY_FEATURES: &str = "communities/visibility/visibilityFeatures.Community";
const IS_REMOVED: &str = "communities/isRemoved.Community";

const HIDDEN: i16 = 2;
const REMOVED: i16 = 2;
const V1: i16 = 1;
const VIEWER_IS_COMMUNITY_MODERATOR: i16 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CommunityPost {
    pub(crate) tweet_id: u64,
    pub(crate) author_id: u64,
    pub(crate) community_id: u64,
}

pub(crate) struct CommunitySource {
    pub(crate) grpc_client: StratoGrpc,
}

impl CommunitySource {
    pub(crate) async fn moderations(
        &self,
        posts: &[CommunityPost],
    ) -> HashMap<u64, Result<Option<CommunityModeration>>> {
        let calls = posts
            .iter()
            .flat_map(|post| {
                let relationship = UserCommunityRelationshipId {
                    user_id: post.author_id,
                    community_id: post.community_id,
                };
                [
                    fetch(TWEET_MODERATION, encode(&(post.tweet_id, ()))),
                    fetch(AUTHOR_MODERATION, encode(&(relationship, ()))),
                ]
            })
            .collect();
        let mut replies = self.grpc_client.batch_call(calls, None).await.into_iter();
        posts
            .iter()
            .map(|post| {
                let is_hidden = read(replies.next(), |arm: UnionArm| arm.0 == Some(HIDDEN));
                let is_author_removed =
                    read(replies.next(), |arm: UnionArm| arm.0 == Some(REMOVED));
                let moderation = is_hidden.and_then(|is_hidden| {
                    Ok(Some(CommunityModeration {
                        is_hidden,
                        is_author_removed: is_author_removed?,
                    }))
                });
                (post.tweet_id, moderation)
            })
            .collect()
    }

    pub(crate) async fn moderators(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> HashMap<u64, Result<Option<bool>>> {
        self.fetch_as_viewer(
            VISIBILITY_FEATURES,
            viewer_id,
            community_ids,
            |features: ViewerIsModerator| features.0,
        )
        .await
    }

    pub(crate) async fn viewer_removals(
        &self,
        viewer_id: u64,
        community_ids: &[u64],
    ) -> HashMap<u64, Result<Option<bool>>> {
        self.fetch_as_viewer(
            IS_REMOVED,
            viewer_id,
            community_ids,
            |removed: IsRemoved| removed.0,
        )
        .await
    }

    async fn fetch_as_viewer<T: TSerializable>(
        &self,
        column: &str,
        viewer_id: u64,
        community_ids: &[u64],
        holds: impl Fn(T) -> bool,
    ) -> HashMap<u64, Result<Option<bool>>> {
        let calls = community_ids
            .iter()
            .map(|&community_id| fetch(column, encode(&(community_id, ()))))
            .collect();
        let viewer = TwitterContextViewer {
            user_id: viewer_id.cast_signed(),
            ..Default::default()
        };
        let mut replies = self
            .grpc_client
            .batch_call(calls, Some(&viewer))
            .await
            .into_iter();
        community_ids
            .iter()
            .map(|&community_id| (community_id, read(replies.next(), &holds).map(Some)))
            .collect()
    }
}

fn fetch(column: &str, key: Vec<u8>) -> (String, String, Vec<Vec<u8>>) {
    (column.to_string(), "fetch".to_string(), vec![key])
}

fn read<T: TSerializable>(
    reply: Option<Result<Bytes>>,
    holds: impl FnOnce(T) -> bool,
) -> Result<bool> {
    let bytes = reply.ok_or_else(|| anyhow!("Strato answered fewer calls than it was sent"))??;
    match strato_decode::<T>(&bytes)? {
        StratoResult::Ok { value, .. } => Ok(value.is_some_and(holds)),
        StratoResult::Err { code, message } => Err(anyhow!("Strato error code {code}: {message}")),
    }
}

#[derive(Default)]
struct UserCommunityRelationshipId {
    user_id: u64,
    community_id: u64,
}

impl UserCommunityRelationshipId {
    fn write(&self, proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        proto.write_struct_begin(&TStructIdentifier::new("UserCommunityRelationshipId"))?;
        for (id, value) in [(1, self.user_id), (2, self.community_id)] {
            proto.write_field_begin(&TFieldIdentifier::new("", TType::I64, id))?;
            proto.write_i64(value.cast_signed())?;
            proto.write_field_end()?;
        }
        proto.write_field_stop()?;
        proto.write_struct_end()
    }

    fn read(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut key = Self::default();
        read_fields(proto, |proto, field| {
            if is(field, 1, TType::I64) {
                key.user_id = proto.read_i64()?.cast_unsigned();
            } else if is(field, 2, TType::I64) {
                key.community_id = proto.read_i64()?.cast_unsigned();
            } else {
                return Ok(false);
            }
            Ok(true)
        })?;
        Ok(key)
    }
}

impl MValCodec for UserCommunityRelationshipId {
    fn thrift_type() -> TType {
        TType::Struct
    }

    fn from_thrift(proto: &mut dyn TInputProtocol) -> Self {
        Self::read(proto).unwrap_or_default()
    }

    #[expect(
        clippy::expect_used,
        reason = "the binary protocol writes into the encode buffer, which cannot fail"
    )]
    fn to_thrift(&self, proto: &mut dyn TOutputProtocol) {
        self.write(proto)
            .expect("UserCommunityRelationshipId encodes");
    }
}

struct UnionArm(Option<i16>);

impl TSerializable for UnionArm {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut arm = None;
        read_fields(proto, |_, field| {
            arm = arm.or(field.id);
            Ok(false)
        })?;
        Ok(Self(arm))
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(decode_only("UnionArm"))
    }
}

struct ViewerIsModerator(bool);

impl TSerializable for ViewerIsModerator {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut is_moderator = false;
        read_fields(proto, |proto, field| {
            if !is(field, V1, TType::Struct) {
                return Ok(false);
            }
            read_fields(proto, |proto, field| {
                if !is(field, VIEWER_IS_COMMUNITY_MODERATOR, TType::Bool) {
                    return Ok(false);
                }
                is_moderator = proto.read_bool()?;
                Ok(true)
            })?;
            Ok(true)
        })?;
        Ok(Self(is_moderator))
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(decode_only("ViewerIsModerator"))
    }
}

struct IsRemoved(bool);

impl TSerializable for IsRemoved {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        Ok(Self(proto.read_bool()?))
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(decode_only("IsRemoved"))
    }
}

fn read_fields(
    proto: &mut dyn TInputProtocol,
    mut read_field: impl FnMut(&mut dyn TInputProtocol, &TFieldIdentifier) -> thrift::Result<bool>,
) -> thrift::Result<()> {
    proto.read_struct_begin()?;
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        if !read_field(proto, &field)? {
            proto.skip(field.field_type)?;
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()
}

fn is(field: &TFieldIdentifier, id: i16, field_type: TType) -> bool {
    field.id == Some(id) && field.field_type == field_type
}

fn decode_only(name: &str) -> thrift::Error {
    thrift::new_protocol_error(
        thrift::ProtocolErrorKind::NotImplemented,
        format!("{name} is decode-only"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrift::protocol::{TBinaryInputProtocol, TBinaryOutputProtocol};

    fn fetched(value: Option<&[u8]>) -> Option<Result<Bytes>> {
        match value {
            Some(value) => answered(&[&[0x0C, 0x69, 0x14][..], value, &[0x00]].concat()),
            None => answered(&[0x01, 0x23, 0x58, 0x00]),
        }
    }

    fn answered(option: &[u8]) -> Option<Result<Bytes>> {
        let ok_value = [0x0C, 0x00, 0x04, 0x0C, 0x09, 0xFC, 0x0C, 0x00, 0x76];
        Some(Ok(Bytes::from(
            [&ok_value[..], option, &[0x00, 0x00, 0x00]].concat(),
        )))
    }

    fn strato_error() -> Option<Result<Bytes>> {
        let code = [0x08, 0xAD, 0xED, 0x00, 0x00, 0x00, 0x05, 0x00];
        let err = [0x0C, 0x00, 0x04, 0x0C, 0x11, 0x45];
        Some(Ok(Bytes::from([&err[..], &code, &[0x00, 0x00]].concat())))
    }

    #[test]
    fn a_moderation_state_reads_its_union_arm_and_an_error_answer_fails() {
        let hidden = |arm: UnionArm| arm.0 == Some(HIDDEN);
        let rows = [
            (fetched(Some(&[0x0C, 0x00, 0x02, 0x00, 0x00])), Some(true)),
            (fetched(Some(&[0x0C, 0x00, 0x01, 0x00, 0x00])), Some(false)),
            (fetched(None), Some(false)),
            (strato_error(), None),
            (Some(Err(anyhow!("Strato error (code 5): denied"))), None),
            (Some(Ok(Bytes::from_static(&[0xFF]))), None),
            (None, None),
        ];
        for (reply, expected) in rows {
            assert_eq!(read(reply, hidden).ok(), expected);
        }
    }

    #[test]
    fn the_moderator_flag_is_read_from_the_v1_arm() {
        for (flag, expected) in [(0x01, true), (0x00, false)] {
            let features = [
                0x0C, 0x00, 0x01, 0x08, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x02, 0x00, 0x04, flag,
                0x00, 0x00,
            ];
            let is_moderator = read(fetched(Some(&features)), |features: ViewerIsModerator| {
                features.0
            });
            assert_eq!(is_moderator.ok(), Some(expected));
        }
    }

    #[test]
    fn the_removal_flag_is_the_column_bool_and_no_row_is_not_removed() {
        for (reply, expected) in [
            (answered(&[0x02, 0x69, 0x14, 0x01, 0x00]), Some(true)),
            (answered(&[0x02, 0x69, 0x14, 0x00, 0x00]), Some(false)),
            (fetched(None), Some(false)),
            (strato_error(), None),
        ] {
            let is_removed = read(reply, |removed: IsRemoved| removed.0);
            assert_eq!(is_removed.ok(), expected);
        }
    }

    #[test]
    fn the_author_moderation_key_writes_its_thrift_field_ids() {
        let key = UserCommunityRelationshipId {
            user_id: 1_500_000_000_000_000_000,
            community_id: 7,
        };
        let mut bytes = Vec::new();
        key.write(&mut TBinaryOutputProtocol::new(&mut bytes, true))
            .unwrap();
        assert_eq!(
            bytes,
            [
                &[0x0A, 0x00, 0x01][..],
                &1_500_000_000_000_000_000_i64.to_be_bytes(),
                &[0x0A, 0x00, 0x02],
                &7_i64.to_be_bytes(),
                &[0x00],
            ]
            .concat()
        );
        let read = UserCommunityRelationshipId::read(&mut TBinaryInputProtocol::new(
            bytes.as_slice(),
            true,
        ))
        .unwrap();
        assert_eq!(
            (read.user_id, read.community_id),
            (key.user_id, key.community_id)
        );
    }
}
