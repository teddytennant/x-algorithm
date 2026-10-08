use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "In network X posts", long_about = None)]
pub struct Args {
    #[arg(long)]
    pub kafka_group_id: String,

    #[arg(long, default_value = "earliest")]
    pub auto_offset_reset: String,

    #[arg(long, default_value = "1000")]
    pub kafka_batch_size: usize,

    #[arg(long, default_value = "5000")]
    pub fetch_timeout_ms: u64,

    #[arg(long, default_value = "scram-client")]
    pub sasl_username: String,

    #[arg(long)]
    pub sasl_password: Option<String>,

    #[arg(long, default_value = "SCRAM-SHA-256")]
    pub sasl_mechanism: String,

    #[arg(long, default_value = "SASL_SSL")]
    pub security_protocol: String,

    #[arg(long, default_value = "120")]
    pub tweet_events_num_partitions: i32,

    #[arg(long, default_value = "180")]
    pub timeline_events_num_partitions: i32,

    #[arg(long, default_value = "false")]
    pub skip_to_latest: bool,

    #[arg(long, default_value = "30")]
    pub lag_monitor_interval_secs: u64,

    #[arg(long, default_value_t = 9090)]
    pub grpc_port: u16,

    #[arg(long, default_value_t = 8080)]
    pub http_port: u16,

    #[arg(long, default_value_t = 172800)]
    pub post_retention_seconds: u64,

    #[arg(long, default_value = "32")]
    pub kafka_num_threads: usize,

    #[arg(long, default_value = "32")]
    pub kafka_tweet_events_v2_num_partitions: usize,

    #[arg(long, default_value = "false")]
    pub is_serving: bool,

    #[arg(long, default_value = "SCRAM-SHA-512")]
    pub producer_sasl_mechanism: String,

    #[arg(long, default_value = "scram-client-phoenix")]
    pub producer_sasl_username: String,

    #[arg(long)]
    pub producer_sasl_password: Option<String>,

    #[arg(long, default_value = "/s/kafka/phoenix-kafka-scram-bootstrap")]
    pub in_network_events_consumer_dest: String,

    #[arg(long)]
    pub in_network_events_consumer_mtls_zone: Option<String>,

    #[arg(long)]
    pub in_network_events_producer_mtls_zone: Option<String>,

    #[arg(long, default_value = "innetwork-posts")]
    pub o2_bucket: String,

    #[arg(long)]
    pub o2_prefix: Option<String>,

    #[arg(long, default_value_t = false)]
    pub enable_profiling: bool,

    #[arg(long, default_value_t = 0)]
    pub request_timeout_ms: u64,

    #[arg(long, default_value_t = 8000)]
    pub qps_limit: u32,
}
