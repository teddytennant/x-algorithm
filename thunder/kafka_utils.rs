use anyhow::{Context, Result};
use std::sync::Arc;
use xai_kafka::config::SslConfig;
use xai_kafka::{
    KafkaConsumerBuilder, KafkaConsumerConfigBuilder, KafkaProducerBuilder, KafkaProducerConfig,
    KafkaProducerConfigBuilder,
};
use xai_wily::WilyConfig;

use crate::{
    args,
    kafka::{
        tweet_events_listener::start_tweet_event_processing,
        tweet_events_listener_v2::start_tweet_event_processing_v2,
    },
};

const TWEET_EVENT_TOPIC: &str = "tweet_events";
const TWEET_EVENT_DEST: &str = "kafka.tweet-events.example.invalid";

const IN_NETWORK_EVENTS_CLUSTER: &str = "phoenix";
const IN_NETWORK_EVENTS_DEST: &str = "kafka.phoenix-bootstrap.example.invalid";
const IN_NETWORK_EVENTS_TOPIC: &str = "innetwork_post";

#[derive(Debug, PartialEq, Eq)]
enum InNetworkEventsAuth<'a> {
    Scram,
    Mtls {
        cluster: &'static str,
        zone: &'a str,
    },
}

impl<'a> InNetworkEventsAuth<'a> {
    fn from_mtls_zone(zone: Option<&'a str>) -> Self {
        match zone {
            Some(zone) => Self::Mtls {
                cluster: IN_NETWORK_EVENTS_CLUSTER,
                zone,
            },
            None => Self::Scram,
        }
    }
}

fn in_network_events_consumer_auth(args: &args::Args) -> InNetworkEventsAuth<'_> {
    InNetworkEventsAuth::from_mtls_zone(args.in_network_events_consumer_mtls_zone.as_deref())
}

fn in_network_events_producer_auth(args: &args::Args) -> InNetworkEventsAuth<'_> {
    InNetworkEventsAuth::from_mtls_zone(args.in_network_events_producer_mtls_zone.as_deref())
}

pub async fn start_kafka(
    args: &args::Args,
    post_store: Arc<crate::posts::post_store::PostStore>,
    xai_user: &str,
    tx: tokio::sync::mpsc::Sender<i64>,
) -> Result<()> {
    let sasl_password = std::env::var("SASL_PASSWORD")
        .ok()
        .or(args.sasl_password.clone())
        .context(
            "SASL password must be provided via SASL_PASSWORD env var or --sasl-password arg",
        )?;

    let producer_sasl_password = std::env::var("PRODUCER_SASL_PASSWORD")
        .ok()
        .or(args.producer_sasl_password.clone());

    if args.is_serving {
        let unique_id = uuid::Uuid::new_v4().to_string();
        let group_id = format!("{}-{}", args.kafka_group_id, unique_id);

        let consumer_builder = match in_network_events_consumer_auth(args) {
            InNetworkEventsAuth::Mtls { cluster, zone } => {
                KafkaConsumerConfigBuilder::for_cluster_mtls_auto(
                    cluster,
                    IN_NETWORK_EVENTS_TOPIC,
                    group_id.clone(),
                    Some(zone),
                )
                .context("Failed to build Phoenix mTLS Kafka consumer config")?
            }
            InNetworkEventsAuth::Scram => KafkaConsumerConfigBuilder::new(
                args.in_network_events_consumer_dest.clone(),
                IN_NETWORK_EVENTS_TOPIC,
                group_id,
            )
            .with_wily_config(WilyConfig::default())
            .with_ssl(SslConfig {
                security_protocol: args.security_protocol.clone(),
                sasl_mechanism: Some(args.producer_sasl_mechanism.clone()),
                sasl_username: Some(args.producer_sasl_username.clone()),
                sasl_password: producer_sasl_password.clone(),
            }),
        };

        let v2_tweet_events_consumer_config = consumer_builder
            .with_auto_offset_reset(args.auto_offset_reset.clone())
            .with_enable_auto_offset_store(true)
            .with_enable_auto_commit(true)
            .with_fetch_timeout_ms(args.fetch_timeout_ms)
            .with_max_partition_fetch_bytes(1024 * 1024 * 100)
            .with_skip_to_latest(args.skip_to_latest);

        start_tweet_event_processing_v2(
            v2_tweet_events_consumer_config,
            Arc::clone(&post_store),
            args,
            tx,
        )
        .await;
    }

    if !args.is_serving {
        let tweet_events_consumer_config = KafkaConsumerBuilder::new(
            TWEET_EVENT_DEST.to_string(),
            TWEET_EVENT_TOPIC.to_string(),
            format!("{}-{}", args.kafka_group_id, xai_user),
        )
        .with_wily_config(WilyConfig::default())
        .with_ssl(SslConfig {
            security_protocol: args.security_protocol.clone(),
            sasl_mechanism: Some(args.sasl_mechanism.clone()),
            sasl_username: Some(args.sasl_username.clone()),
            sasl_password: Some(sasl_password.clone()),
        })
        .with_auto_offset_reset(args.auto_offset_reset.clone())
        .with_enable_auto_commit(false)
        .with_fetch_timeout_ms(args.fetch_timeout_ms)
        .with_max_partition_fetch_bytes(1024 * 1024 * 10)
        .with_skip_to_latest(args.skip_to_latest);

        let producer_config: KafkaProducerConfig = match in_network_events_producer_auth(args) {
            InNetworkEventsAuth::Mtls { cluster, zone } => {
                KafkaProducerConfigBuilder::for_cluster_mtls_auto(
                    cluster,
                    IN_NETWORK_EVENTS_TOPIC,
                    Some(zone),
                )
                .context("Failed to build Phoenix mTLS Kafka producer config")?
                .with_compression_type_opt(None)
                .build()
            }
            InNetworkEventsAuth::Scram => KafkaProducerBuilder::new(
                IN_NETWORK_EVENTS_DEST.to_string(),
                IN_NETWORK_EVENTS_TOPIC.to_string(),
            )
            .with_wily_config(WilyConfig::default())
            .with_ssl(SslConfig {
                security_protocol: args.security_protocol.clone(),
                sasl_mechanism: Some(args.producer_sasl_mechanism.clone()),
                sasl_username: Some(args.producer_sasl_username.clone()),
                sasl_password: producer_sasl_password.clone(),
            })
            .into(),
        };

        start_tweet_event_processing(tweet_events_consumer_config, producer_config, args).await;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn capi_serving_consumer_uses_phoenix_mtls_in_explicit_zone() {
        let args = args::Args::parse_from([
            "thunder",
            "--kafka-group-id",
            "thunder",
            "--in-network-events-consumer-mtls-zone",
            "atla",
        ]);

        assert_eq!(
            in_network_events_consumer_auth(&args),
            InNetworkEventsAuth::Mtls {
                cluster: "phoenix",
                zone: "atla",
            }
        );
    }

    #[test]
    fn legacy_serving_consumer_retains_scram_without_mtls_zone() {
        let args = args::Args::parse_from(["thunder", "--kafka-group-id", "thunder"]);

        assert_eq!(
            in_network_events_consumer_auth(&args),
            InNetworkEventsAuth::Scram
        );
    }

    #[test]
    fn feeder_producer_uses_phoenix_mtls_in_explicit_zone() {
        let args = args::Args::parse_from([
            "thunder",
            "--kafka-group-id",
            "thunder",
            "--in-network-events-producer-mtls-zone",
            "atla",
        ]);

        assert_eq!(
            in_network_events_producer_auth(&args),
            InNetworkEventsAuth::Mtls {
                cluster: "phoenix",
                zone: "atla",
            }
        );
    }

    #[test]
    fn feeder_producer_retains_scram_without_mtls_zone() {
        let args = args::Args::parse_from(["thunder", "--kafka-group-id", "thunder"]);

        assert_eq!(
            in_network_events_producer_auth(&args),
            InNetworkEventsAuth::Scram
        );
    }

    #[test]
    fn producer_and_consumer_mtls_zones_are_independent() {
        let args = args::Args::parse_from([
            "thunder",
            "--kafka-group-id",
            "thunder",
            "--in-network-events-producer-mtls-zone",
            "atla",
        ]);

        assert_eq!(
            in_network_events_consumer_auth(&args),
            InNetworkEventsAuth::Scram
        );
        assert!(matches!(
            in_network_events_producer_auth(&args),
            InNetworkEventsAuth::Mtls { .. }
        ));
    }
}
