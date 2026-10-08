use crate::models::{MediaFeature, NsfwFeature, TweetFeatures};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;
use thrift::protocol::{TInputProtocol, TOutputProtocol, TSerializable, TType};
use xai_core_entities::entities::{
    EditControl, ExclusiveTweetControl, TakedownReason, TrustedFriendsControl,
};
use xai_strato::strato_thrift::{strato_decode, StratoResult};
use xai_strato::{encode, Bytes, MValCodec, StratoGrpc};

const COLUMN: &str = "tweetypie/federated/tweetForVisibility.Tweet";
const OPERATION: &str = "fetch";

pub(crate) struct TweetSource {
    pub(crate) grpc_client: Arc<StratoGrpc>,
}

impl TweetSource {
    pub(crate) async fn get_tweet_values(&self, tweet_ids: &[u64]) -> HashMap<u64, Result<Bytes>> {
        let calls = tweet_ids
            .iter()
            .map(|tweet_id| {
                (
                    COLUMN.to_string(),
                    OPERATION.to_string(),
                    vec![encode(&(*tweet_id, ()))],
                )
            })
            .collect();

        let result_batch = self.grpc_client.batch_call(calls, None).await;
        tweet_ids.iter().copied().zip(result_batch).collect()
    }
}

pub(crate) fn decode_tweet(bytes: &[u8]) -> Result<Option<TweetFeatures>> {
    match std::panic::catch_unwind(|| strato_decode::<Tweet>(bytes))
        .map_err(|_| anyhow!("MVal decoder panicked"))?
    {
        Ok(StratoResult::Ok { value: None, .. }) => Ok(None),
        Ok(StratoResult::Ok {
            value: Some(tweet), ..
        }) => tweet
            .project()
            .map(Some)
            .ok_or_else(|| anyhow!("tweet value without coreData")),
        Ok(StratoResult::Err { code, message }) => {
            Err(anyhow!("Strato error code {code}: {message}"))
        }
        Err(e) => Err(anyhow!("MVal decode error: {e}")),
    }
}

#[derive(Default)]
struct Tweet {
    core_data: Option<CoreData>,
    media: Vec<Media>,
    takedown_reasons: Vec<TakedownReason>,
    community_id: Option<NonZeroU64>,
    exclusive_tweet_control: Option<ExclusiveTweetControl>,
    trusted_friends_list_id: Option<u64>,
    edit_control: Option<EditControl>,
    has_media_refs: bool,
    has_media_keys: bool,
    has_card_reference: bool,
    article_id: Option<NonZeroU64>,
    narrowcast_place_id: Option<u64>,
}

#[derive(Default)]
struct CoreData {
    nsfw_user: bool,
    nsfw_admin: bool,
    nullcast: bool,
}

#[derive(Default)]
struct Media {
    has_media_key: bool,
    restrictions: Option<MediaRestrictions>,
}

#[derive(Default)]
struct MediaRestrictions {
    is_dmca: bool,
    geo_allow_list: Vec<String>,
    geo_deny_list: Vec<String>,
}

impl Tweet {
    fn project(self) -> Option<TweetFeatures> {
        let core_data = self.core_data?;
        Some(TweetFeatures {
            is_nullcast: core_data.nullcast,
            nsfw: NsfwFeature {
                user: core_data.nsfw_user,
                admin: core_data.nsfw_admin,
            },
            takedown_reasons: self.takedown_reasons,
            media: MediaFeature {
                has_media: self.has_media_refs || self.has_card_reference,
                has_uploaded_media: self.has_media_keys,
                ..media_feature(self.media)
            },
            community_id: self.community_id,
            trusted_friends_list_id: self.trusted_friends_list_id,
            edit_control: self.edit_control,
            exclusive_conversation_author_id: self
                .exclusive_tweet_control
                .map(|control| control.conversation_author_id),
            article_id: self.article_id,
            narrowcast_place_id: self.narrowcast_place_id,
        })
    }
}

fn media_feature(entities: Vec<Media>) -> MediaFeature {
    let mut feature = MediaFeature::default();
    for restrictions in entities
        .into_iter()
        .filter(|e| e.has_media_key)
        .filter_map(|e| e.restrictions)
    {
        feature.has_dmca_media |= restrictions.is_dmca;
        feature.geo_allow_list.extend(restrictions.geo_allow_list);
        feature.geo_deny_list.extend(restrictions.geo_deny_list);
    }
    feature
}

impl TSerializable for Tweet {
    fn read_from_in_protocol(proto: &mut dyn TInputProtocol) -> thrift::Result<Self> {
        proto.read_struct_begin()?;
        let mut tweet = Tweet::default();
        loop {
            let field = proto.read_field_begin()?;
            if field.field_type == TType::Stop {
                break;
            }
            match field.id {
                Some(2) => tweet.core_data = Some(read_core_data(proto)?),
                Some(7) => {
                    let list = proto.read_list_begin()?;
                    let mut media = Vec::with_capacity(usize::try_from(list.size).unwrap_or(0));
                    for _ in 0..list.size {
                        media.push(read_media(proto)?);
                    }
                    proto.read_list_end()?;
                    tweet.media = media;
                }
                Some(30) => tweet.takedown_reasons = Vec::<TakedownReason>::from_thrift(proto),
                Some(118) => {
                    proto.skip(field.field_type)?;
                    tweet.has_card_reference = true;
                }
                Some(125) => tweet.community_id = read_first_community_id(proto)?,
                Some(155) => {
                    tweet.exclusive_tweet_control = Some(ExclusiveTweetControl::from_thrift(proto));
                }
                Some(156) => {
                    tweet.trusted_friends_list_id =
                        Some(TrustedFriendsControl::from_thrift(proto).trusted_friends_list_id);
                }
                Some(157) => tweet.edit_control = Some(EditControl::from_thrift(proto)),
                Some(162) => tweet.has_media_refs = skip_list_non_empty(proto)?,
                Some(170) => {
                    tweet.article_id = read_id_field(proto, |proto| {
                        Ok(NonZeroU64::new(proto.read_i64()?.cast_unsigned()))
                    })?;
                }
                Some(172) => {
                    tweet.narrowcast_place_id = read_id_field(proto, |proto| {
                        Ok(u64::from_str_radix(&proto.read_string()?, 16).ok())
                    })?;
                }
                Some(32766) => tweet.has_media_keys = skip_list_non_empty(proto)?,
                _ => proto.skip(field.field_type)?,
            }
            proto.read_field_end()?;
        }
        proto.read_struct_end()?;
        Ok(tweet)
    }

    fn write_to_out_protocol(&self, _proto: &mut dyn TOutputProtocol) -> thrift::Result<()> {
        Err(thrift::new_protocol_error(
            thrift::ProtocolErrorKind::NotImplemented,
            "Tweet is decode-only",
        ))
    }
}

fn read_core_data(proto: &mut dyn TInputProtocol) -> thrift::Result<CoreData> {
    proto.read_struct_begin()?;
    let mut core_data = CoreData::default();
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(9) => core_data.nsfw_user = proto.read_bool()?,
            Some(10) => core_data.nsfw_admin = proto.read_bool()?,
            Some(11) => core_data.nullcast = proto.read_bool()?,
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(core_data)
}

fn read_media(proto: &mut dyn TInputProtocol) -> thrift::Result<Media> {
    proto.read_struct_begin()?;
    let mut media = Media::default();
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(21) => {
                proto.skip(field.field_type)?;
                media.has_media_key = true;
            }
            Some(22) => media.restrictions = read_additional_metadata_restrictions(proto)?,
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(media)
}

fn read_additional_metadata_restrictions(
    proto: &mut dyn TInputProtocol,
) -> thrift::Result<Option<MediaRestrictions>> {
    proto.read_struct_begin()?;
    let mut restrictions = None;
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(12) => restrictions = Some(read_restrictions(proto)?),
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(restrictions)
}

fn read_restrictions(proto: &mut dyn TInputProtocol) -> thrift::Result<MediaRestrictions> {
    proto.read_struct_begin()?;
    let mut restrictions = MediaRestrictions::default();
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(1) => restrictions.is_dmca = proto.read_bool()?,
            Some(3) => {
                (restrictions.geo_allow_list, restrictions.geo_deny_list) =
                    read_geo_restrictions(proto)?;
            }
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(restrictions)
}

fn read_geo_restrictions(
    proto: &mut dyn TInputProtocol,
) -> thrift::Result<(Vec<String>, Vec<String>)> {
    proto.read_struct_begin()?;
    let (mut allow, mut deny) = (Vec::new(), Vec::new());
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(1) => allow = read_string_list(proto)?,
            Some(2) => deny = read_string_list(proto)?,
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok((allow, deny))
}

fn read_string_list(proto: &mut dyn TInputProtocol) -> thrift::Result<Vec<String>> {
    let list = proto.read_list_begin()?;
    let mut strings = Vec::with_capacity(usize::try_from(list.size).unwrap_or(0));
    for _ in 0..list.size {
        strings.push(proto.read_string()?);
    }
    proto.read_list_end()?;
    Ok(strings)
}

fn skip_list_non_empty(proto: &mut dyn TInputProtocol) -> thrift::Result<bool> {
    let list = proto.read_list_begin()?;
    for _ in 0..list.size {
        proto.skip(list.element_type)?;
    }
    proto.read_list_end()?;
    Ok(list.size > 0)
}

fn read_first_community_id(proto: &mut dyn TInputProtocol) -> thrift::Result<Option<NonZeroU64>> {
    proto.read_struct_begin()?;
    let mut first = None;
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(1) => {
                let list = proto.read_list_begin()?;
                if list.size > 0 {
                    first = NonZeroU64::new(proto.read_i64()?.cast_unsigned());
                }
                for _ in 1..list.size {
                    proto.skip(list.element_type)?;
                }
                proto.read_list_end()?;
            }
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(first)
}

fn read_id_field<T>(
    proto: &mut dyn TInputProtocol,
    read: impl Fn(&mut dyn TInputProtocol) -> thrift::Result<Option<T>>,
) -> thrift::Result<Option<T>> {
    proto.read_struct_begin()?;
    let mut id = None;
    loop {
        let field = proto.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        match field.id {
            Some(1) => id = read(proto)?,
            _ => proto.skip(field.field_type)?,
        }
        proto.read_field_end()?;
    }
    proto.read_struct_end()?;
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrift::protocol::{
        TBinaryOutputProtocol, TFieldIdentifier, TListIdentifier, TStructIdentifier,
    };
    use xai_core_entities::entities::{EditControlInitial, MediaEntity};
    use xai_x_thrift::media_common::MediaKey;
    use xai_x_thrift::media_entity::ColorValue;
    use xai_x_thrift::media_information::{
        AdditionalMetadata, AltText, CalibrationBucket, ClassInfo, ClassificationInfo,
        ContentIdInfo, DomainRestrictions, GeoRestrictions, Restrictions,
    };

    type Proto<'a> = TBinaryOutputProtocol<&'a mut Vec<u8>>;
    type Fields<'a> = &'a mut dyn FnMut(&mut Proto<'_>);

    const TWEET_ID: i64 = 10;
    const AUTHOR_ID: i64 = 7001;
    const CONVERSATION_AUTHOR_ID: i64 = 7003;
    const TRUSTED_FRIENDS_LIST_ID: i64 = 9001;
    const ARTICLE_ID: i64 = 9002;

    fn field(proto: &mut Proto<'_>, id: i16, ty: TType, value: impl FnOnce(&mut Proto<'_>)) {
        proto
            .write_field_begin(&TFieldIdentifier::new("", ty, id))
            .unwrap();
        value(proto);
        proto.write_field_end().unwrap();
    }

    fn bare_struct(proto: &mut Proto<'_>, fields: impl FnOnce(&mut Proto<'_>)) {
        proto
            .write_struct_begin(&TStructIdentifier::new(""))
            .unwrap();
        fields(proto);
        proto.write_field_stop().unwrap();
        proto.write_struct_end().unwrap();
    }

    fn structure(proto: &mut Proto<'_>, id: i16, fields: impl FnOnce(&mut Proto<'_>)) {
        field(proto, id, TType::Struct, |p| bare_struct(p, fields));
    }

    fn list(
        proto: &mut Proto<'_>,
        id: i16,
        ty: TType,
        len: usize,
        elements: impl FnOnce(&mut Proto<'_>),
    ) {
        field(proto, id, TType::List, |p| {
            p.write_list_begin(&TListIdentifier::new(ty, len as i32))
                .unwrap();
            elements(p);
            p.write_list_end().unwrap();
        });
    }

    fn i64_field(proto: &mut Proto<'_>, id: i16, value: i64) {
        field(proto, id, TType::I64, |p| p.write_i64(value).unwrap());
    }

    fn bool_field(proto: &mut Proto<'_>, id: i16, value: bool) {
        field(proto, id, TType::Bool, |p| p.write_bool(value).unwrap());
    }

    fn narrowcast_place(proto: &mut Proto<'_>, id: &str) {
        structure(proto, 172, |p| {
            field(p, 1, TType::String, |p| p.write_string(id).unwrap());
        });
    }

    fn encode_option(option: Fields<'_>) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut proto = TBinaryOutputProtocol::new(&mut bytes, false);
        bare_struct(&mut proto, |p| {
            structure(p, 4, |p| {
                structure(p, 2556, |p| structure(p, 118, |p| option(p)));
            });
        });
        bytes
    }

    fn encode_tweet(core_data: Fields<'_>, rest: Fields<'_>) -> Vec<u8> {
        encode_option(&mut |p| {
            structure(p, 26900, |p| {
                i64_field(p, 1, TWEET_ID);
                structure(p, 2, |p| {
                    i64_field(p, 1, AUTHOR_ID);
                    field(p, 2, TType::String, |p| {
                        p.write_string("tweet text").unwrap()
                    });
                    core_data(p);
                });
                rest(p);
            });
        })
    }

    fn non_empty_list(proto: &mut Proto<'_>, id: i16, non_empty: bool) {
        list(proto, id, TType::Struct, usize::from(non_empty), |p| {
            if non_empty {
                bare_struct(p, |_| {});
            }
        });
    }

    fn media_fixture(
        has_refs: bool,
        has_card: bool,
        keys: Option<bool>,
        entities: &[MediaEntity],
    ) -> Vec<u8> {
        encode_tweet(&mut |_| {}, &mut |p| {
            list(p, 7, TType::Struct, entities.len(), |p| {
                for entity in entities {
                    entity.write_to_out_protocol(p).unwrap();
                }
            });
            if has_card {
                structure(p, 118, |_| {});
            }
            non_empty_list(p, 162, has_refs);
            if let Some(non_empty) = keys {
                non_empty_list(p, 32766, non_empty);
            }
        })
    }

    fn restricted_media(has_key: bool, dmca: bool, allow: &[&str], deny: &[&str]) -> MediaEntity {
        MediaEntity {
            media_key: has_key.then(MediaKey::default),
            additional_metadata: Some(AdditionalMetadata {
                restrictions: Some(Restrictions {
                    is_dmca: Some(dmca),
                    geo_restrictions: Some(GeoRestrictions {
                        whitelisted_country_codes: Some(
                            allow.iter().map(|s| s.to_string()).collect(),
                        ),
                        blacklisted_country_codes: Some(
                            deny.iter().map(|s| s.to_string()).collect(),
                        ),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn decode_fixture(bytes: &[u8]) -> TweetFeatures {
        decode_tweet(bytes).unwrap().unwrap()
    }

    #[test]
    fn projects_every_read_field_of_a_found_tweet() {
        let bytes = encode_tweet(
            &mut |p| {
                bool_field(p, 9, true);
                bool_field(p, 10, false);
                bool_field(p, 11, true);
            },
            &mut |p| {
                list(p, 7, TType::Struct, 1, |p| {
                    restricted_media(true, true, &[], &[])
                        .write_to_out_protocol(p)
                        .unwrap()
                });
                list(p, 30, TType::Struct, 1, |p| {
                    bare_struct(p, |p| structure(p, 4, |_| {}))
                });
                structure(p, 118, |_| {});
                structure(p, 125, |p| {
                    list(p, 1, TType::I64, 1, |p| p.write_i64(500).unwrap())
                });
                structure(p, 155, |p| i64_field(p, 1, CONVERSATION_AUTHOR_ID));
                structure(p, 156, |p| i64_field(p, 1, TRUSTED_FRIENDS_LIST_ID));
                structure(p, 157, |p| {
                    structure(p, 1, |p| {
                        list(p, 1, TType::I64, 1, |p| p.write_i64(TWEET_ID).unwrap());
                        i64_field(p, 3, 4);
                    })
                });
                list(p, 162, TType::Struct, 1, |p| bare_struct(p, |_| {}));
                structure(p, 170, |p| i64_field(p, 1, ARTICLE_ID));
                narrowcast_place(p, "7f00000000000001");
                list(p, 32766, TType::Struct, 1, |p| bare_struct(p, |_| {}));
            },
        );
        let media = MediaFeature {
            has_media: true,
            has_uploaded_media: true,
            has_dmca_media: true,
            ..Default::default()
        };
        let edit_control = Some(EditControl::Initial(EditControlInitial {
            edit_tweet_ids: vec![TWEET_ID as u64],
            edits_remaining: Some(4),
            ..Default::default()
        }));

        assert_eq!(
            decode_fixture(&bytes),
            TweetFeatures {
                media,
                takedown_reasons: vec![TakedownReason::Dmca],
                nsfw: NsfwFeature {
                    user: true,
                    admin: false
                },
                is_nullcast: true,
                community_id: NonZeroU64::new(500),
                trusted_friends_list_id: Some(TRUSTED_FRIENDS_LIST_ID as u64),
                edit_control,
                exclusive_conversation_author_id: Some(CONVERSATION_AUTHOR_ID as u64),
                article_id: NonZeroU64::new(ARTICLE_ID as u64),
                narrowcast_place_id: Some(0x7f00_0000_0000_0001),
            }
        );
    }

    #[test]
    fn a_narrowcast_place_keeps_all_64_bits_of_its_hex_and_a_non_hex_one_is_no_place() {
        for (id, expected) in [
            ("a000000000000001", Some(0xa000_0000_0000_0001)),
            ("not hex", None),
        ] {
            let bytes = encode_tweet(&mut |_| {}, &mut |p| {
                narrowcast_place(p, id);
                list(p, 32766, TType::Struct, 1, |p| bare_struct(p, |_| {}));
            });

            let features = decode_fixture(&bytes);
            assert_eq!(features.narrowcast_place_id, expected, "{id}");
            assert!(
                features.media.has_uploaded_media,
                "{id}: the field after the place did not decode"
            );
        }
    }

    #[test]
    fn an_article_with_id_0_is_no_article() {
        let bytes = encode_tweet(&mut |_| {}, &mut |p| {
            structure(p, 170, |p| i64_field(p, 1, 0));
        });

        assert_eq!(decode_fixture(&bytes).article_id, None);
    }

    #[test]
    fn plain_tweet_decodes_default_features() {
        let bytes = encode_tweet(&mut |_| {}, &mut |_| {});

        assert_eq!(decode_fixture(&bytes), TweetFeatures::default());
    }

    #[test]
    fn media_restrictions_fold_only_across_keyed_entities() {
        let bytes = media_fixture(
            true,
            false,
            Some(true),
            &[
                restricted_media(true, true, &["us"], &["de"]),
                restricted_media(true, false, &["gb"], &["fr"]),
                restricted_media(false, true, &["ignored"], &["ignored"]),
                MediaEntity::default(),
            ],
        );

        assert_eq!(
            decode_fixture(&bytes).media,
            MediaFeature {
                has_media: true,
                has_uploaded_media: true,
                has_dmca_media: true,
                geo_allow_list: vec!["us".to_string(), "gb".to_string()],
                geo_deny_list: vec!["de".to_string(), "fr".to_string()],
            }
        );
    }

    #[test]
    fn media_fields_vf_does_not_read_are_skipped() {
        let classified = MediaEntity {
            url: Some("https://t.co/x".to_string()),
            dominant_color_grid: Some(vec![ColorValue::default(); 3]),
            media_key: Some(MediaKey::default()),
            additional_metadata: Some(AdditionalMetadata {
                alt_text: Some(AltText {
                    text: Some("a cat".to_string()),
                }),
                classifications: Some(ClassificationInfo {
                    probabilities: Some(vec![ClassInfo {
                        label: Some("nsfw".to_string()),
                        calibration_buckets: Some(vec![CalibrationBucket::default(); 2]),
                        model_version: Some("v7".to_string()),
                        ..Default::default()
                    }]),
                    predictions: Some(vec![ClassInfo::default()]),
                }),
                restrictions: Some(Restrictions {
                    is_dmca: Some(true),
                    domain_restrictions: Some(DomainRestrictions {
                        whitelist: Some(vec!["x.com".to_string()]),
                    }),
                    geo_restrictions: Some(GeoRestrictions {
                        whitelisted_country_codes: Some(vec!["us".to_string()]),
                        blacklisted_country_codes: Some(vec!["de".to_string()]),
                    }),
                    content_id_info: Some(ContentIdInfo {
                        copyright_holder_name: Some("holder".to_string()),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let unknown = |p: &mut Proto<'_>| {
            structure(p, 900, |p| {
                list(p, 1, TType::Struct, 1, |p| {
                    bare_struct(p, |p| {
                        field(p, 2, TType::String, |p| p.write_string("x").unwrap())
                    })
                });
            });
        };
        let bytes = encode_tweet(&mut |_| {}, &mut |p| {
            list(p, 7, TType::Struct, 2, |p| {
                classified.write_to_out_protocol(p).unwrap();
                bare_struct(p, |p| {
                    unknown(p);
                    structure(p, 22, |p| {
                        unknown(p);
                        structure(p, 12, |p| {
                            unknown(p);
                            structure(p, 3, |p| {
                                unknown(p);
                                list(p, 2, TType::String, 1, |p| p.write_string("fr").unwrap());
                            });
                        });
                    });
                    structure(p, 21, |p| i64_field(p, 2, 1));
                });
            });
            structure(p, 155, |p| i64_field(p, 1, CONVERSATION_AUTHOR_ID));
        });

        let features = decode_fixture(&bytes);
        assert_eq!(
            features.media,
            MediaFeature {
                has_dmca_media: true,
                geo_allow_list: vec!["us".to_string()],
                geo_deny_list: vec!["de".to_string(), "fr".to_string()],
                ..Default::default()
            }
        );
        assert_eq!(
            features.exclusive_conversation_author_id,
            Some(CONVERSATION_AUTHOR_ID as u64)
        );
    }

    #[test]
    fn keyed_media_without_restrictions_adds_nothing() {
        let keyed = |additional_metadata| MediaEntity {
            media_key: Some(MediaKey::default()),
            additional_metadata,
            ..Default::default()
        };
        let bytes = media_fixture(
            false,
            false,
            None,
            &[
                keyed(None),
                keyed(Some(AdditionalMetadata::default())),
                keyed(Some(AdditionalMetadata {
                    restrictions: Some(Restrictions::default()),
                    ..Default::default()
                })),
            ],
        );

        assert_eq!(decode_fixture(&bytes).media, MediaFeature::default());
    }

    #[test]
    fn media_presence_comes_from_refs_or_card_reference_and_uploads_from_own_media_keys() {
        for (name, bytes, has_media, has_uploaded_media) in [
            (
                "own media keys",
                media_fixture(true, false, Some(true), &[]),
                true,
                true,
            ),
            (
                "empty media keys",
                media_fixture(true, false, Some(false), &[]),
                true,
                false,
            ),
            (
                "no media keys field",
                media_fixture(true, false, None, &[]),
                true,
                false,
            ),
            ("card", media_fixture(false, true, None, &[]), true, false),
            (
                "entities only",
                media_fixture(false, false, None, &[MediaEntity::default()]),
                false,
                false,
            ),
        ] {
            let media = decode_fixture(&bytes).media;
            assert_eq!(
                (media.has_media, media.has_uploaded_media),
                (has_media, has_uploaded_media),
                "{name}"
            );
        }
    }

    #[test]
    fn community_id_is_the_first_of_the_community_id_list() {
        for (community_ids, expected) in [
            (vec![500, 600], NonZeroU64::new(500)),
            (
                vec![1_500_000_000_000_000_000],
                NonZeroU64::new(1_500_000_000_000_000_000),
            ),
            (vec![], None),
        ] {
            let bytes = encode_tweet(&mut |_| {}, &mut |p| {
                structure(p, 125, |p| {
                    list(p, 1, TType::I64, community_ids.len(), |p| {
                        for id in &community_ids {
                            p.write_i64(*id).unwrap();
                        }
                    });
                    field(p, 2, TType::String, |p| p.write_string("channel").unwrap());
                });
                structure(p, 155, |p| i64_field(p, 1, CONVERSATION_AUTHOR_ID));
            });

            let features = decode_fixture(&bytes);
            assert_eq!(features.community_id, expected, "{community_ids:?}");
            assert_eq!(
                features.exclusive_conversation_author_id,
                Some(CONVERSATION_AUTHOR_ID as u64),
                "{community_ids:?}: the field after communities did not decode"
            );
        }
    }

    #[test]
    fn a_panicking_sub_reader_is_an_error_not_a_panic() {
        let bytes = encode_tweet(&mut |_| {}, &mut |p| {
            structure(p, 155, |p| i64_field(p, 1, CONVERSATION_AUTHOR_ID))
        });
        let end = bytes
            .windows(8)
            .position(|window| window == CONVERSATION_AUTHOR_ID.to_be_bytes())
            .unwrap()
            + 4;

        let error = decode_tweet(&bytes[..end]).unwrap_err();
        assert_eq!(error.to_string(), "MVal decoder panicked");
    }

    #[test]
    fn an_absent_value_is_missing_and_a_value_without_core_data_is_an_error() {
        let absent = encode_option(&mut |p| field(p, 9048, TType::Void, |_| {}));
        assert!(decode_tweet(&absent).unwrap().is_none());

        let without_core_data =
            encode_option(&mut |p| structure(p, 26900, |p| i64_field(p, 1, TWEET_ID)));
        let error = decode_tweet(&without_core_data).unwrap_err();
        assert_eq!(error.to_string(), "tweet value without coreData");
    }
}
