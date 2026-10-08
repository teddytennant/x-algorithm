// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::pg::PgError;

pub const HOLD_KIND_SUSPEND: &str = "suspend";
pub const HOLD_KIND_LABEL: &str = "label";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hold {
    pub user_id: i64,
    pub hold_id: i64,
    pub action_kind: String,
    pub label: Option<String>,
    pub head: Option<String>,
    pub expires_at: DateTime<Utc>,
    pub case_group_id: Option<i32>,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStatus {
    pub size: usize,
    pub available: usize,
    pub waiting: usize,
    pub max_size: usize,
}

#[async_trait]
pub trait HoldRepo: Send + Sync {
    async fn lookup(
        &self,
        user_ids: &[i64],
        labels: &[String],
        budget: Duration,
    ) -> Result<Vec<Hold>, PgError>;

    async fn contract_check(&self, budget: Duration) -> Result<(), PgError>;

    async fn ping(&self, budget: Duration) -> Result<(), PgError>;

    fn pool_status(&self) -> Option<PoolStatus>;

    fn endpoint(&self) -> String;
}

pub type LookupCall = (Vec<i64>, Vec<String>, Duration);

pub struct FakeHoldRepo {
    default: Mutex<Result<Vec<Hold>, PgError>>,
    queue: Mutex<Vec<Result<Vec<Hold>, PgError>>>,
    contract: Mutex<Vec<Result<(), PgError>>>,
    pings: Mutex<Vec<Result<(), PgError>>>,
    calls: Mutex<Vec<LookupCall>>,
    pool: Option<PoolStatus>,
}

impl Default for FakeHoldRepo {
    fn default() -> Self {
        Self {
            default: Mutex::new(Ok(Vec::new())),
            queue: Mutex::new(Vec::new()),
            contract: Mutex::new(Vec::new()),
            pings: Mutex::new(Vec::new()),
            calls: Mutex::new(Vec::new()),
            pool: None,
        }
    }
}

impl FakeHoldRepo {
    pub fn returning(holds: Vec<Hold>) -> Self {
        Self {
            default: Mutex::new(Ok(holds)),
            ..Self::default()
        }
    }

    pub fn failing(err: PgError) -> Self {
        Self {
            default: Mutex::new(Err(err)),
            ..Self::default()
        }
    }

    pub fn push_lookup(&self, res: Result<Vec<Hold>, PgError>) {
        self.queue.lock().unwrap().insert(0, res);
    }

    pub fn push_contract(&self, res: Result<(), PgError>) {
        self.contract.lock().unwrap().insert(0, res);
    }

    pub fn push_ping(&self, res: Result<(), PgError>) {
        self.pings.lock().unwrap().insert(0, res);
    }

    pub fn with_pool(mut self, pool: PoolStatus) -> Self {
        self.pool = Some(pool);
        self
    }

    pub fn calls(&self) -> Vec<LookupCall> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl HoldRepo for FakeHoldRepo {
    async fn lookup(
        &self,
        user_ids: &[i64],
        labels: &[String],
        budget: Duration,
    ) -> Result<Vec<Hold>, PgError> {
        self.calls
            .lock()
            .unwrap()
            .push((user_ids.to_vec(), labels.to_vec(), budget));
        if let Some(next) = self.queue.lock().unwrap().pop() {
            return next;
        }
        self.default.lock().unwrap().clone()
    }

    async fn contract_check(&self, _budget: Duration) -> Result<(), PgError> {
        self.contract.lock().unwrap().pop().unwrap_or(Ok(()))
    }

    async fn ping(&self, _budget: Duration) -> Result<(), PgError> {
        self.pings.lock().unwrap().pop().unwrap_or(Ok(()))
    }

    fn pool_status(&self) -> Option<PoolStatus> {
        self.pool
    }

    fn endpoint(&self) -> String {
        "fake".to_owned()
    }
}
