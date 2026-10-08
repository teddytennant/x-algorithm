// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use moka::sync::Cache;
use tonic::transport::{Channel, Endpoint};
use tracing::warn;
use xai_recsys_sid_proto::LookupSidsRequest;
use xai_recsys_sid_proto::sid_lookup_service_client::SidLookupServiceClient;

use crate::metrics::SID_SERVICE_LOOKUPS;

pub struct SeedSidLookup {
    client: Option<SidLookupServiceClient<Channel>>,
    cache: Cache<i64, Vec<i32>>,
    timeout: Duration,
}

impl SeedSidLookup {
    pub fn new(endpoint: &str, timeout: Duration, cache_size: u64) -> Result<Self> {
        let client = if endpoint.is_empty() {
            None
        } else {
            let url = if endpoint.starts_with("http://") {
                endpoint.to_string()
            } else {
                format!("http://{endpoint}")
            };
            let channel = Endpoint::from_shared(url.clone())
                .with_context(|| format!("invalid SID endpoint {url}"))?
                .timeout(timeout)
                .connect_timeout(Duration::from_secs(5))
                .tcp_keepalive(Some(Duration::from_secs(30)))
                .http2_keep_alive_interval(Duration::from_secs(30))
                .keep_alive_while_idle(true)
                .connect_lazy();
            Some(SidLookupServiceClient::new(channel))
        };
        Ok(Self {
            client,
            cache: Cache::builder()
                .max_capacity(cache_size)
                .time_to_live(Duration::from_secs(6 * 3600))
                .build(),
            timeout,
        })
    }

    pub async fn warm_up(&self) {
        let Some(client) = self.client.as_ref() else {
            return;
        };
        let request = LookupSidsRequest { post_ids: vec![] };
        match client.clone().lookup_sids(request).await {
            Ok(_) => SID_SERVICE_LOOKUPS.with_label_values(&["warm_up"]).inc(),
            Err(status) => warn!(error = %status, "SID service warm-up failed"),
        }
    }

    pub async fn lookup(&self, post_ids: &[i64]) -> HashMap<i64, Vec<i32>> {
        let mut found = HashMap::with_capacity(post_ids.len());
        let mut missing = Vec::new();
        for &post_id in post_ids {
            match self.cache.get(&post_id) {
                Some(codes) => {
                    found.insert(post_id, codes);
                }
                None => missing.push(post_id),
            }
        }
        SID_SERVICE_LOOKUPS
            .with_label_values(&["cache_hit"])
            .inc_by(found.len() as u64);
        let Some(client) = self.client.as_ref() else {
            return found;
        };
        if missing.is_empty() {
            return found;
        }
        let request = LookupSidsRequest {
            post_ids: missing.clone(),
        };
        let response =
            tokio::time::timeout(self.timeout, client.clone().lookup_sids(request)).await;
        match response {
            Ok(Ok(response)) => {
                let results = response.into_inner().results;
                for (post_id, post_sids) in missing.into_iter().zip(results) {
                    if post_sids.codes.is_empty() {
                        SID_SERVICE_LOOKUPS.with_label_values(&["empty"]).inc();
                        continue;
                    }
                    SID_SERVICE_LOOKUPS.with_label_values(&["hit"]).inc();
                    self.cache.insert(post_id, post_sids.codes.clone());
                    found.insert(post_id, post_sids.codes);
                }
            }
            Ok(Err(status)) => {
                SID_SERVICE_LOOKUPS
                    .with_label_values(&["error"])
                    .inc_by(missing.len() as u64);
                warn!(error = %status, "SID service lookup failed");
            }
            Err(_) => {
                SID_SERVICE_LOOKUPS
                    .with_label_values(&["timeout"])
                    .inc_by(missing.len() as u64);
            }
        }
        found
    }
}
