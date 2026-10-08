use crate::clients::read_fields;
use anyhow::bail;
use std::collections::HashSet;
use thrift::protocol::{
    TFieldIdentifier, TInputProtocol, TOutputProtocol, TSerializable, TSetIdentifier,
    TStructIdentifier, TType,
};
use tonic::async_trait;
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::{encode, MValCodec, StratoGrpc};

const USER_LOCATION_COLUMN: &str = "geo/service/userLocation";

const PLACE_TYPES: [i32; 5] = [4, 3, 7, 9, 1];

const GEODUCK_IDL_DEFAULT_FLAGS: [(&str, i16, bool); 3] = [
    ("simple_reverse_geocode", 3, false),
    ("include_debug_information", 4, false),
    ("filter_by_confidence", 5, true),
];

const FOUND: i16 = 1;
const PLACE_MAP: i16 = 5;

#[async_trait]
pub trait UserLocationClient: Send + Sync {
    async fn places(&self, user_id: u64) -> anyhow::Result<HashSet<u64>>;
}

pub struct ProdUserLocationClient {
    grpc: StratoGrpc,
}

impl ProdUserLocationClient {
    pub fn new(grpc: StratoGrpc) -> Self {
        Self { grpc }
    }
}

#[async_trait]
impl UserLocationClient for ProdUserLocationClient {
    async fn places(&self, user_id: u64) -> anyhow::Result<HashSet<u64>> {
        let view = UserLocationRequest {
            user_id: user_id.cast_signed(),
        };
        let bytes = self
            .grpc
            .call(
                USER_LOCATION_COLUMN,
                "fetch",
                vec![encode(&((), &view))],
                None,
            )
            .await?;
        decode_places(&bytes, user_id)
    }
}

fn decode_places(bytes: &[u8], user_id: u64) -> anyhow::Result<HashSet<u64>> {
    match strato_decode::<UserLocationResponse>(bytes)? {
        StratoResult::Ok { value, .. } => Ok(value
            .and_then(|response| {
                response
                    .found
                    .into_iter()
                    .find(|(user, _)| *user == user_id)
            })
            .map(|(_, places)| places)
            .unwrap_or_default()),
        StratoResult::Err { code, message } => bail!("Strato error (code {code}): {message}"),
    }
}

struct UserLocationRequest {
    user_id: i64,
}

impl MValCodec for UserLocationRequest {
    fn thrift_type() -> TType {
        TType::Struct
    }

    #[expect(clippy::unimplemented, reason = "a view is only encoded")]
    fn from_thrift(_proto: &mut dyn TInputProtocol) -> Self {
        unimplemented!("a view is only encoded")
    }

    #[expect(
        clippy::unwrap_used,
        reason = "`xai_strato::encode` writes into a Vec, which cannot fail, and five place types fit an i32"
    )]
    fn to_thrift(&self, proto: &mut dyn TOutputProtocol) {
        proto
            .write_struct_begin(&TStructIdentifier::new("UserLocationRequest"))
            .unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("userIds", TType::List, 1))
            .unwrap();
        vec![self.user_id].to_thrift(proto);
        proto.write_field_end().unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("place_query", TType::Struct, 2))
            .unwrap();
        proto
            .write_struct_begin(&TStructIdentifier::new("PlaceQuery"))
            .unwrap();
        proto
            .write_field_begin(&TFieldIdentifier::new("place_types", TType::Set, 1))
            .unwrap();
        let place_types = i32::try_from(PLACE_TYPES.len()).unwrap();
        proto
            .write_set_begin(&TSetIdentifier::new(TType::I32, place_types))
            .unwrap();
        for place_type in PLACE_TYPES {
            proto.write_i32(place_type).unwrap();
        }
        proto.write_set_end().unwrap();
        proto.write_field_end().unwrap();
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
        proto.write_field_end().unwrap();
        for (name, id, value) in GEODUCK_IDL_DEFAULT_FLAGS {
            proto
                .write_field_begin(&TFieldIdentifier::new(name, TType::Bool, id))
                .unwrap();
            proto.write_bool(value).unwrap();
            proto.write_field_end().unwrap();
        }
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }
}

struct UserLocationResponse {
    found: Vec<(u64, HashSet<u64>)>,
}

impl TSerializable for UserLocationResponse {
    fn read_from_in_protocol(i_prot: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        let mut found = Vec::new();
        read_fields(i_prot, |i_prot, field| {
            if field.id != Some(FOUND) || field.field_type != TType::Map {
                return Ok(false);
            }
            let users = i_prot.read_map_begin()?;
            for _ in 0..users.size {
                let user = i_prot.read_i64()?.cast_unsigned();
                found.push((user, read_place_ids(i_prot)?));
            }
            i_prot.read_map_end()?;
            Ok(true)
        })?;
        Ok(Self { found })
    }

    fn write_to_out_protocol(&self, _o_prot: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(thrift::new_protocol_error(
            thrift::ProtocolErrorKind::NotImplemented,
            "UserLocationResponse is decode-only",
        ))
    }
}

fn read_place_ids(i_prot: &mut dyn TInputProtocol) -> thrift::Result<HashSet<u64>> {
    let mut places = HashSet::new();
    read_fields(i_prot, |i_prot, field| {
        if field.id != Some(PLACE_MAP) || field.field_type != TType::Map {
            return Ok(false);
        }
        let place_map = i_prot.read_map_begin()?;
        for _ in 0..place_map.size {
            let is_asked = PLACE_TYPES.contains(&i_prot.read_i32()?);
            let ids = i_prot.read_set_begin()?;
            for _ in 0..ids.size {
                let id = i_prot.read_i64()?.cast_unsigned();
                if is_asked {
                    places.insert(id);
                }
            }
            i_prot.read_set_end()?;
        }
        i_prot.read_map_end()?;
        Ok(true)
    })?;
    Ok(places)
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrift::protocol::{TBinaryOutputProtocol, TMapIdentifier};

    type Proto<'a> = TBinaryOutputProtocol<&'a mut Vec<u8>>;

    const USER: u64 = 7;
    const OTHER_USER: u64 = 8;
    const CITY: i32 = 3;
    const COUNTRY: i32 = 1;
    const METRO: i32 = 7;
    const POI: i32 = 5;
    const ABOVE_I64_MAX: u64 = 0xa000_0000_0000_0001;

    fn field(proto: &mut Proto<'_>, ty: TType, id: i16) {
        proto
            .write_field_begin(&TFieldIdentifier::new("", ty, id))
            .unwrap();
    }

    fn begin_struct(proto: &mut Proto<'_>) {
        proto
            .write_struct_begin(&TStructIdentifier::new(""))
            .unwrap();
    }

    fn mval(value: impl FnOnce(&mut Proto<'_>)) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut proto = TBinaryOutputProtocol::new(&mut bytes, false);
        begin_struct(&mut proto);
        for id in [4, 2556, 118] {
            field(&mut proto, TType::Struct, id);
            begin_struct(&mut proto);
        }
        value(&mut proto);
        for _ in 0..4 {
            proto.write_field_stop().unwrap();
        }
        bytes
    }

    type Found<'a> = (u64, Option<&'a [(i32, u64)]>);

    fn located(users: &[Found<'_>]) -> Vec<u8> {
        mval(|p| {
            field(p, TType::Struct, 26900);
            begin_struct(p);
            field(p, TType::Map, 1);
            p.write_map_begin(&TMapIdentifier::new(
                TType::I64,
                TType::Struct,
                users.len() as i32,
            ))
            .unwrap();
            for &(user, place_map) in users {
                p.write_i64(user.cast_signed()).unwrap();
                begin_struct(p);
                field(p, TType::I64, 1);
                p.write_i64(user.cast_signed()).unwrap();
                if let Some(place_map) = place_map {
                    field(p, TType::Map, 5);
                    p.write_map_begin(&TMapIdentifier::new(
                        TType::I32,
                        TType::Set,
                        place_map.len() as i32,
                    ))
                    .unwrap();
                    for &(place_type, id) in place_map {
                        p.write_i32(place_type).unwrap();
                        p.write_set_begin(&TSetIdentifier::new(TType::I64, 1))
                            .unwrap();
                        p.write_i64(id.cast_signed()).unwrap();
                    }
                }
                field(p, TType::I64, 4);
                p.write_i64(1).unwrap();
                p.write_field_stop().unwrap();
            }
            p.write_field_stop().unwrap();
        })
    }

    fn strato_error(code: i32, reason: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut proto = TBinaryOutputProtocol::new(&mut bytes, false);
        begin_struct(&mut proto);
        field(&mut proto, TType::Struct, 4);
        begin_struct(&mut proto);
        field(&mut proto, TType::Struct, 4421);
        begin_struct(&mut proto);
        field(&mut proto, TType::I32, -21011);
        proto.write_i32(code).unwrap();
        field(&mut proto, TType::String, -28092);
        proto.write_string(reason).unwrap();
        for _ in 0..3 {
            proto.write_field_stop().unwrap();
        }
        bytes
    }

    #[test]
    fn the_view_asks_the_five_place_types_of_one_user() {
        assert_eq!(
            encode(&((), &UserLocationRequest { user_id: 7 })),
            [
                0x0c, 0x00, 0x04, 0x01, 0x00, 0x00, 0x0c, 0x00, 0x01, 0x0f, 0x00, 0x01, 0x0a, 0x00,
                0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x0c, 0x00, 0x02,
                0x0e, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
                0x00, 0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x01,
                0x00, 0x02, 0x00, 0x03, 0x00, 0x02, 0x00, 0x04, 0x00, 0x02, 0x00, 0x05, 0x01, 0x00,
                0x00, 0x00,
            ]
        );
    }

    #[test]
    fn decodes_the_users_ids_of_the_five_place_types() {
        let response = located(&[
            (OTHER_USER, Some(&[(CITY, 21)])),
            (
                USER,
                Some(&[(CITY, 11), (COUNTRY, 12), (POI, 13), (METRO, ABOVE_I64_MAX)]),
            ),
        ]);

        assert_eq!(
            decode_places(&response, USER).unwrap(),
            HashSet::from([11, 12, ABOVE_I64_MAX])
        );
    }

    #[test]
    fn a_user_geoduck_does_not_locate_has_no_places() {
        for (name, response) in [
            (
                "only another user found",
                located(&[(OTHER_USER, Some(&[(CITY, 11)]))]),
            ),
            ("found without a place map", located(&[(USER, None)])),
            (
                "no value",
                mval(|p| {
                    field(p, TType::Void, 9048);
                }),
            ),
        ] {
            assert_eq!(
                decode_places(&response, USER).unwrap(),
                HashSet::new(),
                "{name}"
            );
        }

        let error = decode_places(&strato_error(2, "boom"), USER).unwrap_err();
        assert_eq!(error.to_string(), "Strato error (code 2): boom");
    }
}
