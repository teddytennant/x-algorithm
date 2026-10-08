// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tracing::{info, warn};

use crate::metrics;
use crate::repo::HoldRepo;

pub const CONTRACT_CHECK_BUDGET: Duration = Duration::from_secs(5);
pub const CONTRACT_BACKOFF_MIN: Duration = Duration::from_secs(1);
pub const CONTRACT_BACKOFF_MAX: Duration = Duration::from_secs(30);
pub const DB_UP_INTERVAL: Duration = Duration::from_secs(10);
pub const DB_UP_BUDGET: Duration = Duration::from_secs(5);

pub fn next_backoff(current: Duration) -> Duration {
    (current * 2).min(CONTRACT_BACKOFF_MAX)
}

pub async fn contract_check_until_ok(repo: Arc<dyn HoldRepo>, ready: Arc<AtomicBool>) {
    let mut backoff = CONTRACT_BACKOFF_MIN;
    let mut attempt: u64 = 0;
    loop {
        attempt += 1;
        match repo.contract_check(CONTRACT_CHECK_BUDGET).await {
            Ok(()) => {
                metrics::CONTRACT_OK.set(1);
                metrics::DB_UP.set(1);
                ready.store(true, Ordering::Release);
                info!(
                    attempt,
                    endpoint = %repo.endpoint(),
                    "boot contract check passed: ledger.enforcement_holds has the probe's \
                     columns; /readyz is now 200"
                );
                return;
            }
            Err(e) => {
                metrics::CONTRACT_OK.set(0);
                warn!(
                    attempt,
                    code = e.code().as_str(),
                    error = %e,
                    retry_in_ms = backoff.as_millis() as u64,
                    endpoint = %repo.endpoint(),
                    "boot contract check failed; /readyz stays 503 (pod takes no traffic)"
                );
                tokio::time::sleep(backoff).await;
                backoff = next_backoff(backoff);
            }
        }
    }
}

pub async fn db_up_loop(repo: Arc<dyn HoldRepo>) {
    let mut tick = tokio::time::interval(DB_UP_INTERVAL);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        match repo.ping(DB_UP_BUDGET).await {
            Ok(()) => metrics::DB_UP.set(1),
            Err(e) => {
                metrics::DB_UP.set(0);
                warn!(code = e.code().as_str(), error = %e, "db_up probe failed");
            }
        }
        if let Some(pool) = repo.pool_status() {
            metrics::set_pool(pool);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pg::{ConnectFailure, PgError};
    use crate::repo::FakeHoldRepo;

    #[test]
    fn backoff_doubles_and_caps() {
        let mut b = CONTRACT_BACKOFF_MIN;
        let mut seen = vec![b];
        for _ in 0..8 {
            b = next_backoff(b);
            seen.push(b);
        }
        assert_eq!(seen[0], Duration::from_secs(1));
        assert_eq!(seen[1], Duration::from_secs(2));
        assert_eq!(seen[4], Duration::from_secs(16));
        assert_eq!(seen[5], CONTRACT_BACKOFF_MAX);
        assert_eq!(seen[8], CONTRACT_BACKOFF_MAX);
    }

    #[tokio::test(start_paused = true)]
    async fn ready_flips_only_after_the_contract_check_passes() {
        let repo = Arc::new(FakeHoldRepo::default());
        repo.push_contract(Err(PgError::Query {
            sqlstate: "42703".into(),
            message: "column \"label\" does not exist".into(),
        }));
        repo.push_contract(Err(PgError::Connect {
            kind: ConnectFailure::Refused,
            message: "refused".into(),
        }));
        let ready = Arc::new(AtomicBool::new(false));
        let task = tokio::spawn(contract_check_until_ok(repo.clone(), ready.clone()));
        tokio::task::yield_now().await;
        assert!(!ready.load(Ordering::Acquire));
        tokio::time::advance(Duration::from_secs(1)).await;
        tokio::task::yield_now().await;
        assert!(!ready.load(Ordering::Acquire));
        tokio::time::advance(Duration::from_secs(2)).await;
        task.await.unwrap();
        assert!(ready.load(Ordering::Acquire));
        assert_eq!(metrics::CONTRACT_OK.get(), 1);
    }
}
