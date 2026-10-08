use anyhow::{bail, Result};
use std::iter;
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TOutputProtocol, TSerializable, TStructIdentifier, TType,
};
use thrift::{new_protocol_error, ProtocolErrorKind};
use tonic::async_trait;
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::{encode, MValCodec, StratoGrpc};

const IS_MEMBER_COLUMN: &str = "trusted-friends/visibility-filtering/isMember.TrustedFriendsList";
const IS_OWNER_COLUMN: &str = "trusted-friends/visibility-filtering/isOwner.TrustedFriendsList";

#[async_trait]
pub trait TrustedFriendsClient: Send + Sync {
    async fn batch_is_member_or_owner(&self, viewer_id: u64, list_ids: &[u64])
        -> Vec<Result<bool>>;
}

pub struct ProdTrustedFriendsClient {
    grpc: StratoGrpc,
}

impl ProdTrustedFriendsClient {
    pub fn new(grpc: StratoGrpc) -> Self {
        Self { grpc }
    }
}

#[async_trait]
impl TrustedFriendsClient for ProdTrustedFriendsClient {
    async fn batch_is_member_or_owner(
        &self,
        viewer_id: u64,
        list_ids: &[u64],
    ) -> Vec<Result<bool>> {
        let view = View {
            user_id: viewer_id.cast_signed(),
        };
        let calls = list_ids
            .iter()
            .flat_map(|list_id| {
                let key = encode(&(list_id.cast_signed(), &view));
                [IS_MEMBER_COLUMN, IS_OWNER_COLUMN]
                    .map(|column| (column.to_string(), "fetch".to_string(), vec![key.clone()]))
            })
            .collect();
        let mut answers = self
            .grpc
            .batch_call(calls, None)
            .await
            .into_iter()
            .map(|bytes| decode_answer(&bytes?));
        iter::from_fn(|| Some(member_or_owner(answers.next()?, answers.next()?))).collect()
    }
}

fn member_or_owner(member: Result<bool>, owner: Result<bool>) -> Result<bool> {
    match (member, owner) {
        (Ok(true), _) | (_, Ok(true)) => Ok(true),
        (Err(error), _) | (_, Err(error)) => Err(error),
        (Ok(false), Ok(false)) => Ok(false),
    }
}

fn decode_answer(bytes: &[u8]) -> Result<bool> {
    match strato_decode::<Answer>(bytes)? {
        StratoResult::Ok { value, .. } => Ok(value.is_some_and(|Answer(answer)| answer)),
        StratoResult::Err { code, message } => bail!("Strato error (code {code}): {message}"),
    }
}

struct View {
    user_id: i64,
}

const USER_ID_FIELD: i16 = 11846;

impl MValCodec for View {
    fn thrift_type() -> TType {
        TType::Struct
    }

    #[expect(clippy::unimplemented, reason = "a view is only encoded")]
    fn from_thrift(_proto: &mut dyn TInputProtocol) -> Self {
        unimplemented!("a view is only encoded")
    }

    #[expect(
        clippy::unwrap_used,
        reason = "`xai_strato::encode` writes into a Vec, which cannot fail"
    )]
    fn to_thrift(&self, proto: &mut dyn TOutputProtocol) {
        proto
            .write_struct_begin(&TStructIdentifier::new("View"))
            .unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("userId", TType::I64, USER_ID_FIELD))
            .unwrap();
        proto.write_i64(self.user_id).unwrap();
        proto.write_field_end().unwrap();
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }
}

struct Answer(bool);

impl TSerializable for Answer {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        Ok(Self(i_prot.read_bool()?))
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(new_protocol_error(
            ProtocolErrorKind::NotImplemented,
            "Answer is decode-only",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;
    use thrift::protocol::TBinaryOutputProtocol;

    fn answer(value: Option<bool>) -> bool {
        let mut bytes = Vec::new();
        let mut proto = TBinaryOutputProtocol::new(&mut bytes, false);
        proto
            .write_struct_begin(&TStructIdentifier::new(""))
            .unwrap();
        for id in [4, 2556, 118] {
            proto
                .write_field_begin(&TFieldIdentifier::new("", TType::Struct, id))
                .unwrap();
            proto
                .write_struct_begin(&TStructIdentifier::new(""))
                .unwrap();
        }
        let (option_type, option_field) = match value {
            Some(_) => (TType::Bool, 26900),
            None => (TType::Void, 9048),
        };
        proto
            .write_field_begin(&TFieldIdentifier::new("", option_type, option_field))
            .unwrap();
        if let Some(value) = value {
            proto.write_bool(value).unwrap();
        }
        for _ in 0..4 {
            proto.write_field_stop().unwrap();
        }
        decode_answer(&bytes).unwrap()
    }

    #[test]
    fn the_column_answer_decodes_and_a_missing_value_answers_false() {
        assert!(answer(Some(true)));
        assert!(!answer(Some(false)));
        assert!(!answer(None));
    }

    #[test]
    fn either_column_answering_true_shows_and_otherwise_a_failure_fails() {
        let failed = || Err(anyhow!("unavailable"));
        assert!(member_or_owner(Ok(true), failed()).unwrap());
        assert!(member_or_owner(failed(), Ok(true)).unwrap());
        assert!(!member_or_owner(Ok(false), Ok(false)).unwrap());
        assert!(member_or_owner(Ok(false), failed()).is_err());
        assert!(member_or_owner(failed(), Ok(false)).is_err());
    }
}
