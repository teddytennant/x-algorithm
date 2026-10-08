use crate::clients::read_fields;
use anyhow::bail;
use thrift::protocol::{TInputProtocol, TOutputProtocol, TSerializable, TType};
use tonic::async_trait;
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::StratoGrpc;

const TFE_TOP_COUNTRY_COLUMN: &str = "about_this_account/tfe_top_country.User";

#[async_trait]
pub trait AboutThisAccountClient: Send + Sync {
    async fn tfe_top_country(&self, user_id: u64) -> anyhow::Result<Option<String>>;
}

pub struct ProdAboutThisAccountClient {
    grpc: StratoGrpc,
}

impl ProdAboutThisAccountClient {
    pub fn new(grpc: StratoGrpc) -> Self {
        Self { grpc }
    }
}

#[async_trait]
impl AboutThisAccountClient for ProdAboutThisAccountClient {
    async fn tfe_top_country(&self, user_id: u64) -> anyhow::Result<Option<String>> {
        let bytes = self
            .grpc
            .call(
                TFE_TOP_COUNTRY_COLUMN,
                "fetch",
                vec![xai_strato::encode(&(user_id.cast_signed(), ()))],
                None,
            )
            .await?;
        match strato_decode::<UserTfeTopCountry>(&bytes)? {
            StratoResult::Ok { value, .. } => Ok(value.and_then(|row| row.weighted_top_country)),
            StratoResult::Err { code, message } => bail!("Strato error (code {code}): {message}"),
        }
    }
}

#[derive(Debug, Default, PartialEq)]
struct UserTfeTopCountry {
    weighted_top_country: Option<String>,
}

const WEIGHTED_TOP_COUNTRY: i16 = 2;
const COUNTRY_ID: i16 = 1;

impl TSerializable for UserTfeTopCountry {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut weighted_top_country = None;
        read_fields(i_prot, |i_prot, field| {
            if field.id != Some(WEIGHTED_TOP_COUNTRY) || field.field_type != TType::Struct {
                return Ok(false);
            }
            read_fields(i_prot, |i_prot, field| {
                if field.id != Some(COUNTRY_ID) || field.field_type != TType::String {
                    return Ok(false);
                }
                weighted_top_country = Some(i_prot.read_string()?);
                Ok(true)
            })?;
            Ok(true)
        })?;
        Ok(Self {
            weighted_top_country,
        })
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(thrift::new_protocol_error(
            thrift::ProtocolErrorKind::NotImplemented,
            "UserTfeTopCountry is decode-only",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrift::protocol::{
        TBinaryInputProtocol, TBinaryOutputProtocol, TFieldIdentifier, TStructIdentifier,
    };

    fn country_details(
        o_prot: &mut TBinaryOutputProtocol<&mut Vec<u8>>,
        id: i16,
        country_id: &str,
    ) {
        o_prot
            .write_field_begin(&TFieldIdentifier::new("", TType::Struct, id))
            .unwrap();
        o_prot
            .write_struct_begin(&TStructIdentifier::new(""))
            .unwrap();
        o_prot
            .write_field_begin(&TFieldIdentifier::new("", TType::String, 1))
            .unwrap();
        o_prot.write_string(country_id).unwrap();
        o_prot
            .write_field_begin(&TFieldIdentifier::new("", TType::Bool, 2))
            .unwrap();
        o_prot.write_bool(true).unwrap();
        o_prot.write_field_stop().unwrap();
        o_prot.write_struct_end().unwrap();
        o_prot.write_field_end().unwrap();
    }

    fn decode_row(details: &[(i16, &str)]) -> UserTfeTopCountry {
        let mut bytes = Vec::new();
        let mut o_prot = TBinaryOutputProtocol::new(&mut bytes, false);
        o_prot
            .write_struct_begin(&TStructIdentifier::new(""))
            .unwrap();
        o_prot
            .write_field_begin(&TFieldIdentifier::new("", TType::I64, 1))
            .unwrap();
        o_prot.write_i64(7).unwrap();
        for (id, country_id) in details {
            country_details(&mut o_prot, *id, country_id);
        }
        o_prot
            .write_field_begin(&TFieldIdentifier::new("", TType::I64, 5))
            .unwrap();
        o_prot.write_i64(1).unwrap();
        o_prot.write_field_stop().unwrap();
        let mut i_prot = TBinaryInputProtocol::new(&bytes[..], false);
        UserTfeTopCountry::read_from_in_protocol(&mut i_prot).unwrap()
    }

    #[test]
    fn decodes_only_the_weighted_top_country() {
        let full = decode_row(&[(3, "FR"), (2, "nz"), (4, "DE")]);
        assert_eq!(full.weighted_top_country.as_deref(), Some("nz"));
        assert_eq!(decode_row(&[(3, "FR")]), UserTfeTopCountry::default());
    }
}
