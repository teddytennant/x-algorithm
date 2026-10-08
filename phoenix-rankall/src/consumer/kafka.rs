use std::env;

use anyhow::{Context, Result};
use async_trait::async_trait;
use log::info;
use xai_kafka::{config::SslConfig, KafkaConsumer, KafkaConsumerConfigBuilder, KafkaMessage};
use xai_wily::WilyConfig;

use super::{ConsumerError, EventConsumer, RawBatch};
use crate::config::{Cli, KafkaAuthMode};

pub struct KafkaEventConsumer {
    inner: KafkaConsumer,
    max_poll_records: usize,
    last_batch_metadata: Vec<xai_kafka::KafkaMessageMetadata>,
}

impl KafkaEventConsumer {
    pub async fn from_cli(cli: &Cli) -> Result<Self> {
        let topic = cli.effective_topic().to_string();

        let config = match cli.kafka_auth {
            KafkaAuthMode::Scram => Self::scram_config(cli, &topic)?,
            KafkaAuthMode::Mtls => Self::mtls_config(cli, &topic)?,
        };

        let start_from_minutes_ago = if cli.seek_hours > 0 {
            Some(cli.seek_hours * 60)
        } else {
            None
        };

        info!(
            "kafka consumer config: auth={:?}, topic={}, group={}, seek_hours={}, auto_offset_reset={}",
            cli.kafka_auth, topic, cli.group, cli.seek_hours, cli.auto_offset_reset,
        );

        let config = config
            .with_auto_offset_reset(cli.auto_offset_reset.clone())
            .with_enable_auto_offset_store(false)
            .with_enable_auto_commit(false)
            .with_fetch_timeout_ms(5000)
            .with_start_from_minutes_ago(start_from_minutes_ago);
        let mut inner = KafkaConsumer::new(config);
        inner
            .start()
            .await
            .context("failed to start Kafka consumer")?;

        Ok(Self {
            inner,
            max_poll_records: cli.max_poll_records,
            last_batch_metadata: Vec::new(),
        })
    }

    fn scram_config(cli: &Cli, topic: &str) -> Result<KafkaConsumerConfigBuilder> {
        let bootstrap = cli.bootstrap.clone();

        let is_wily = bootstrap.starts_with('/') || bootstrap.starts_with("s/");
        let (wily_config, brokers) = if is_wily {
            (Some(WilyConfig::default()), None)
        } else {
            let broker_list: Vec<String> = bootstrap.split(',').map(|s| s.to_string()).collect();
            (None, Some(broker_list))
        };

        let sasl_username = env::var("KAFKA_SASL_USERNAME")
            .context("KAFKA_SASL_USERNAME env var is required for Kafka pipelines")?;
        let sasl_password = env::var("KAFKA_SASL_PASSWORD")
            .context("KAFKA_SASL_PASSWORD env var is required for Kafka pipelines")?;
        let sasl_mechanism =
            env::var("KAFKA_SASL_MECHANISM").unwrap_or_else(|_| "PLAIN".to_string());

        let ssl = SslConfig {
            security_protocol: "SASL_SSL".to_string(),
            sasl_mechanism: Some(sasl_mechanism),
            sasl_username: Some(sasl_username),
            sasl_password: Some(sasl_password),
        };

        Ok(
            KafkaConsumerConfigBuilder::new(bootstrap, topic, &cli.group)
                .with_brokers(brokers)
                .with_wily_config(wily_config)
                .with_ssl(ssl),
        )
    }

    fn mtls_config(cli: &Cli, topic: &str) -> Result<KafkaConsumerConfigBuilder> {
        let cluster = cli.mtls_cluster()?;
        KafkaConsumerConfigBuilder::for_cluster_mtls_auto(
            cluster,
            topic,
            &cli.group,
            cli.kafka_zone.as_deref(),
        )
        .with_context(|| format!("failed to configure mTLS consumer for Kafka cluster '{cluster}'"))
    }

    fn extract_metadata(msg: &KafkaMessage) -> xai_kafka::KafkaMessageMetadata {
        xai_kafka::KafkaMessageMetadata {
            topic: msg.topic.clone(),
            partition: msg.partition,
            offset: msg.offset,
            timestamp: msg.timestamp,
        }
    }
}

#[async_trait]
impl EventConsumer for KafkaEventConsumer {
    async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
        let messages = self
            .inner
            .poll(self.max_poll_records)
            .await
            .map_err(ConsumerError::Kafka)?;

        self.last_batch_metadata = messages.iter().map(Self::extract_metadata).collect();

        let payloads: RawBatch = messages.into_iter().filter_map(|msg| msg.payload).collect();

        Ok(payloads)
    }

    async fn commit(&mut self) -> Result<(), ConsumerError> {
        if self.last_batch_metadata.is_empty() {
            return Ok(());
        }

        self.inner
            .commit(&self.last_batch_metadata)
            .map_err(ConsumerError::Kafka)?;

        self.last_batch_metadata.clear();
        Ok(())
    }

    async fn close(&mut self) -> Result<(), ConsumerError> {
        if !self.last_batch_metadata.is_empty() {
            let _ = self.commit().await;
        }
        self.inner.stop().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_metadata_from_message() {
        let msg = KafkaMessage {
            topic: "test_topic".to_string(),
            partition: 3,
            offset: 42,
            key: None,
            payload: Some(vec![1, 2, 3]),
            timestamp: Some(1700000000000),
            ..Default::default()
        };

        let meta = KafkaEventConsumer::extract_metadata(&msg);
        assert_eq!(meta.topic, "test_topic");
        assert_eq!(meta.partition, 3);
        assert_eq!(meta.offset, 42);
        assert_eq!(meta.timestamp, Some(1700000000000));
    }
}
