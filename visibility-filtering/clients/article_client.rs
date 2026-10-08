use anyhow::bail;
use std::collections::HashMap;
use thrift::protocol::{TInputProtocol, TOutputProtocol, TSerializable, TType};
use tonic::async_trait;
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::StratoGrpc;

const LIFECYCLE_COLUMN: &str = "article/fields/lifecycleState.ArticleEntity";

#[async_trait]
pub trait ArticleClient: Send + Sync {
    async fn lifecycles(&self, article_ids: &[u64]) -> HashMap<u64, anyhow::Result<Option<i32>>>;
}

pub struct ProdArticleClient {
    grpc: StratoGrpc,
}

impl ProdArticleClient {
    pub fn new(grpc: StratoGrpc) -> Self {
        Self { grpc }
    }
}

#[async_trait]
impl ArticleClient for ProdArticleClient {
    async fn lifecycles(&self, article_ids: &[u64]) -> HashMap<u64, anyhow::Result<Option<i32>>> {
        let calls = article_ids
            .iter()
            .map(|id| {
                (
                    LIFECYCLE_COLUMN.to_owned(),
                    "fetch".to_owned(),
                    vec![xai_strato::encode(&(id.cast_signed(), ()))],
                )
            })
            .collect();
        let replies = self.grpc.batch_call(calls, None).await;
        article_ids
            .iter()
            .copied()
            .zip(replies.into_iter().map(|reply| decode_lifecycle(&reply?)))
            .collect()
    }
}

fn decode_lifecycle(bytes: &[u8]) -> anyhow::Result<Option<i32>> {
    match strato_decode::<ArticleLifecycleState>(bytes)? {
        StratoResult::Ok { value, .. } => Ok(value.map(|state| state.lifecycle)),
        StratoResult::Err { code, message } => bail!("Strato error (code {code}): {message}"),
    }
}

struct ArticleLifecycleState {
    lifecycle: i32,
}

const LIFECYCLE: i16 = 1;

impl TSerializable for ArticleLifecycleState {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        i_prot.read_struct_begin()?;
        let mut lifecycle = None;
        loop {
            let field = i_prot.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            if field.id == Some(LIFECYCLE) && field.field_type == TType::I32 {
                lifecycle = Some(i_prot.read_i32()?);
            } else {
                i_prot.skip(field.field_type)?;
            }
            i_prot.read_field_end()?;
        }
        i_prot.read_struct_end()?;
        let Some(lifecycle) = lifecycle else {
            return Err(thrift::new_protocol_error(
                thrift::ProtocolErrorKind::InvalidData,
                "ArticleLifecycleState has no lifecycle",
            ));
        };
        Ok(Self { lifecycle })
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(thrift::new_protocol_error(
            thrift::ProtocolErrorKind::NotImplemented,
            "ArticleLifecycleState is decode-only",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrift::protocol::{
        TBinaryInputProtocol, TBinaryOutputProtocol, TFieldIdentifier, TStructIdentifier,
    };

    fn decode_row(lifecycle: Option<i32>) -> thrift::Result<i32> {
        let mut bytes = Vec::new();
        let mut o_prot = TBinaryOutputProtocol::new(&mut bytes, false);
        o_prot
            .write_struct_begin(&TStructIdentifier::new(""))
            .unwrap();
        if let Some(lifecycle) = lifecycle {
            o_prot
                .write_field_begin(&TFieldIdentifier::new("", TType::I32, LIFECYCLE))
                .unwrap();
            o_prot.write_i32(lifecycle).unwrap();
        }
        o_prot
            .write_field_begin(&TFieldIdentifier::new("", TType::I64, 2))
            .unwrap();
        o_prot.write_i64(1_790_813_967).unwrap();
        o_prot.write_field_stop().unwrap();
        let mut i_prot = TBinaryInputProtocol::new(&bytes[..], false);
        ArticleLifecycleState::read_from_in_protocol(&mut i_prot).map(|state| state.lifecycle)
    }

    #[test]
    fn reads_the_lifecycle_field_and_rejects_a_row_without_one() {
        assert_eq!(decode_row(Some(4)).unwrap(), 4);
        assert!(decode_row(None).is_err());
    }
}
