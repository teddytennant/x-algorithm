use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use log::{error, info, warn};
use tokio_util::sync::CancellationToken;

use crate::consumer::{ConsumerError, EventConsumer};
use crate::metrics::PipelineMetrics;
use crate::processor::RecordProcessor;
use crate::store::SnapshotStore;

const STATS_LOG_INTERVAL: u64 = 100_000;

pub struct Pipeline {
    consumer: Box<dyn EventConsumer>,
    processor: Arc<Mutex<Box<dyn RecordProcessor>>>,
    store: Box<dyn SnapshotStore>,
    metrics: PipelineMetrics,
    commit_after_dump: bool,
    commit_interval_secs: u64,
}

impl Pipeline {
    pub fn new(
        consumer: Box<dyn EventConsumer>,
        processor: Box<dyn RecordProcessor>,
        store: Box<dyn SnapshotStore>,
        metrics: PipelineMetrics,
    ) -> Self {
        Self {
            consumer,
            processor: Arc::new(Mutex::new(processor)),
            store,
            metrics,
            commit_after_dump: true,
            commit_interval_secs: 0,
        }
    }

        pub fn with_commit_strategy(
        mut self,
        commit_after_dump: bool,
        commit_interval_secs: u64,
    ) -> Self {
        self.commit_after_dump = commit_after_dump;
        self.commit_interval_secs = commit_interval_secs;
        self
    }

                                                                                                    pub async fn run(mut self, cancel: CancellationToken) -> anyhow::Result<()> {
        info!("pipeline starting");

        self.store.start().await?;

        let mut total_consumed: u64 = 0;
        let mut total_processed: u64 = 0;

        let result = self
            .run_loop(&cancel, &mut total_consumed, &mut total_processed)
            .await;

        info!("pipeline shutting down (consumed={total_consumed}, processed={total_processed})");

        if let Err(e) = self.consumer.commit().await {
            warn!("failed to commit offsets during shutdown: {e}");
        }
        if let Err(e) = self.store.stop().await {
            warn!("failed to stop store during shutdown: {e}");
        }
        if let Err(e) = self.consumer.close().await {
            warn!("failed to close consumer during shutdown: {e}");
        }

        info!("pipeline shutdown complete");
        result
    }

    async fn run_loop(
        &mut self,
        cancel: &CancellationToken,
        total_consumed: &mut u64,
        total_processed: &mut u64,
    ) -> anyhow::Result<()> {
        let dump_completed = self.store.dump_completed_flag();
        let commit_interval = if self.commit_interval_secs > 0 {
            Some(Duration::from_secs(self.commit_interval_secs))
        } else {
            None
        };
        let mut last_interval_commit = Instant::now();

        loop {
            if cancel.is_cancelled() {
                info!("shutdown signal received");
                return Ok(());
            }

            if self.commit_after_dump
                && dump_completed.swap(false, std::sync::atomic::Ordering::AcqRel)
            {
                if let Err(e) = self.consumer.commit().await {
                    warn!("offset commit after dump failed: {e}");
                } else {
                    info!("offsets committed after successful dump");
                }
            }

            let poll_start = Instant::now();
            let raw_batch = tokio::select! {
                batch = self.consumer.poll_batch() => batch,
                _ = cancel.cancelled() => {
                    info!("shutdown signal received during poll");
                    return Ok(());
                }
            };

            let raw_batch = match raw_batch {
                Ok(batch) => batch,
                Err(ConsumerError::Closed) => {
                    info!("consumer closed");
                    return Ok(());
                }
                Err(e) => {
                    error!("consumer poll error: {e}");
                    continue;
                }
            };

            let poll_duration = poll_start.elapsed();
            self.metrics
                .batch_poll_duration_seconds
                .with_label_values(&[] as &[&str])
                .observe(poll_duration.as_secs_f64());

            if raw_batch.is_empty() {
                continue;
            }

            let batch_size = raw_batch.len();
            *total_consumed += batch_size as u64;
            self.metrics
                .records_processed_total
                .inc_by(batch_size as u64);
            self.metrics
                .batch_size
                .with_label_values(&[] as &[&str])
                .observe(batch_size as f64);

            let process_start = Instant::now();
            let processor = Arc::clone(&self.processor);
            let records = tokio::task::spawn_blocking(move || {
                let mut proc = processor.lock().expect("processor lock poisoned");
                proc.process_batch(&raw_batch)
            })
            .await?;

            let process_duration = process_start.elapsed();
            self.metrics
                .batch_process_duration_seconds
                .with_label_values(&[] as &[&str])
                .observe(process_duration.as_secs_f64());

            let record_count = records.len();

            if !records.is_empty() {
                let store_start = Instant::now();
                self.store.add_batch(records).await?;
                let store_duration = store_start.elapsed();
                self.metrics
                    .batch_store_duration_seconds
                    .with_label_values(&[] as &[&str])
                    .observe(store_duration.as_secs_f64());
            }

            if let Some(interval) = commit_interval
                && last_interval_commit.elapsed() >= interval
            {
                if let Err(e) = self.consumer.commit().await {
                    warn!("interval offset commit failed (will retry): {e}");
                } else {
                    last_interval_commit = Instant::now();
                }
            }

            *total_processed += record_count as u64;

            {
                let proc = self.processor.lock().expect("processor lock poisoned");
                let stats = proc.stats();
                self.metrics.update_processor_stats("", stats);
            }

            if *total_consumed % STATS_LOG_INTERVAL < batch_size as u64 {
                info!(
                    "progress: consumed={total_consumed}, processed={total_processed}, \
                     last_batch={batch_size}, last_records={record_count}, \
                     poll={poll_duration:?}, process={process_duration:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consumer::RawBatch;
    use crate::processor::ProcessorStats;
    use crate::processor::record::IndexRecord;
    use crate::store::{SnapshotStore, StoreError};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};


    struct MockConsumer {
        batches: Vec<RawBatch>,
        index: usize,
        commit_count: Arc<AtomicUsize>,
        close_called: Arc<AtomicUsize>,
    }

    impl MockConsumer {
        fn new(
            batches: Vec<RawBatch>,
            commit_count: Arc<AtomicUsize>,
            close_called: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                batches,
                index: 0,
                commit_count,
                close_called,
            }
        }
    }

    #[async_trait]
    impl EventConsumer for MockConsumer {
        async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
            if self.index < self.batches.len() {
                let batch = self.batches[self.index].clone();
                self.index += 1;
                Ok(batch)
            } else {
                Err(ConsumerError::Closed)
            }
        }

        async fn commit(&mut self) -> Result<(), ConsumerError> {
            self.commit_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn close(&mut self) -> Result<(), ConsumerError> {
            self.close_called.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }


    struct MockProcessor {
        stats: ProcessorStats,
        call_count: Arc<AtomicUsize>,
    }

    impl MockProcessor {
        fn new(call_count: Arc<AtomicUsize>) -> Self {
            Self {
                stats: ProcessorStats::default(),
                call_count,
            }
        }
    }

    impl RecordProcessor for MockProcessor {
        fn process_batch(&mut self, raw: &[Vec<u8>]) -> Vec<IndexRecord> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            self.stats.total_processed += raw.len() as u64;
            raw.iter()
                .enumerate()
                .map(|(i, _)| IndexRecord::Core {
                    post_id: (i + 1) as i64,
                    author_id: 100,
                    index_name: "1fav".to_string(),
                })
                .collect()
        }

        fn stats(&self) -> &ProcessorStats {
            &self.stats
        }
    }


    struct MockStore {
        batches_received: Arc<AtomicUsize>,
        records_received: Arc<AtomicUsize>,
        started: Arc<AtomicUsize>,
        stopped: Arc<AtomicUsize>,
        dump_flag: Arc<AtomicBool>,
    }

    impl MockStore {
        fn new(
            batches_received: Arc<AtomicUsize>,
            records_received: Arc<AtomicUsize>,
            started: Arc<AtomicUsize>,
            stopped: Arc<AtomicUsize>,
        ) -> Self {
            Self {
                batches_received,
                records_received,
                started,
                stopped,
                dump_flag: Arc::new(AtomicBool::new(false)),
            }
        }

        fn with_dump_flag(mut self, flag: Arc<AtomicBool>) -> Self {
            self.dump_flag = flag;
            self
        }
    }

    #[async_trait]
    impl SnapshotStore for MockStore {
        async fn add_batch(&self, records: Vec<IndexRecord>) -> Result<(), StoreError> {
            self.batches_received.fetch_add(1, Ordering::SeqCst);
            self.records_received
                .fetch_add(records.len(), Ordering::SeqCst);
            Ok(())
        }

        async fn start(&self) -> Result<(), StoreError> {
            self.started.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn stop(&self) -> Result<(), StoreError> {
            self.stopped.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn dump_completed_flag(&self) -> Arc<AtomicBool> {
            Arc::clone(&self.dump_flag)
        }
    }

    fn make_test_pipeline(
        consumer: Box<dyn EventConsumer>,
        store: MockStore,
        commit_after_dump: bool,
        commit_interval_secs: u64,
    ) -> (Pipeline, Arc<AtomicUsize>) {
        let process_count = Arc::new(AtomicUsize::new(0));
        let processor = MockProcessor::new(process_count.clone());
        let metrics = PipelineMetrics::new(prometheus::Registry::new()).expect("metrics init");
        let pipeline = Pipeline::new(consumer, Box::new(processor), Box::new(store), metrics)
            .with_commit_strategy(commit_after_dump, commit_interval_secs);
        (pipeline, process_count)
    }


    #[tokio::test]
    async fn pipeline_processes_all_batches_then_shuts_down() {
        let commit_count = Arc::new(AtomicUsize::new(0));
        let close_called = Arc::new(AtomicUsize::new(0));
        let consumer = MockConsumer::new(
            vec![
                vec![vec![1, 2], vec![3, 4]],
                vec![vec![5, 6]],
                vec![vec![7, 8], vec![9, 10], vec![11, 12]],
            ],
            commit_count.clone(),
            close_called.clone(),
        );

        let store = MockStore::new(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );

        let (pipeline, _) = make_test_pipeline(Box::new(consumer), store, true, 0);
        pipeline
            .run(CancellationToken::new())
            .await
            .expect("pipeline should succeed");

        assert!(
            commit_count.load(Ordering::SeqCst) >= 1,
            "at least 1 commit (shutdown)"
        );
        assert_eq!(close_called.load(Ordering::SeqCst), 1, "consumer closed");
    }

    #[tokio::test]
    async fn commit_after_dump_only_commits_when_flag_set() {
        let commit_count = Arc::new(AtomicUsize::new(0));
        let dump_flag = Arc::new(AtomicBool::new(false));

        struct SlowConsumer {
            batch_count: AtomicUsize,
            commit_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl EventConsumer for SlowConsumer {
            async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
                let n = self.batch_count.fetch_add(1, Ordering::SeqCst);
                if n >= 10 {
                    return Err(ConsumerError::Closed);
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                Ok(vec![vec![1, 2]])
            }
            async fn commit(&mut self) -> Result<(), ConsumerError> {
                self.commit_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            async fn close(&mut self) -> Result<(), ConsumerError> {
                Ok(())
            }
        }

        let consumer = SlowConsumer {
            batch_count: AtomicUsize::new(0),
            commit_count: commit_count.clone(),
        };

        let store = MockStore::new(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        )
        .with_dump_flag(dump_flag.clone());

        let flag_clone = dump_flag.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            flag_clone.store(true, Ordering::Release);
        });

        let (pipeline, _) = make_test_pipeline(Box::new(consumer), store, true, 0);
        pipeline
            .run(CancellationToken::new())
            .await
            .expect("pipeline should succeed");

        let commits = commit_count.load(Ordering::SeqCst);
        assert!(
            commits >= 2,
            "expected at least 2 commits (dump + shutdown), got {commits}"
        );
    }

    #[tokio::test]
    async fn interval_commit_fires_without_dump() {
        let commit_count = Arc::new(AtomicUsize::new(0));

        struct TimedConsumer {
            commit_count: Arc<AtomicUsize>,
            start: std::time::Instant,
        }

        #[async_trait]
        impl EventConsumer for TimedConsumer {
            async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
                if self.start.elapsed() > std::time::Duration::from_millis(1500) {
                    return Err(ConsumerError::Closed);
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(vec![vec![1, 2]])
            }
            async fn commit(&mut self) -> Result<(), ConsumerError> {
                self.commit_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            async fn close(&mut self) -> Result<(), ConsumerError> {
                Ok(())
            }
        }

        let consumer = TimedConsumer {
            commit_count: commit_count.clone(),
            start: std::time::Instant::now(),
        };

        let store = MockStore::new(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );

        let (pipeline, _) = make_test_pipeline(Box::new(consumer), store, false, 1);
        pipeline
            .run(CancellationToken::new())
            .await
            .expect("pipeline should succeed");

        let commits = commit_count.load(Ordering::SeqCst);
        assert!(
            commits >= 2,
            "expected at least 2 commits (interval + shutdown), got {commits}"
        );
    }

    #[tokio::test]
    async fn no_commits_in_loop_when_dump_only_and_no_notification() {
        let commit_count = Arc::new(AtomicUsize::new(0));

        struct QuickConsumer {
            remaining: AtomicUsize,
            commit_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl EventConsumer for QuickConsumer {
            async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
                let n = self.remaining.fetch_sub(1, Ordering::SeqCst);
                if n == 0 {
                    return Err(ConsumerError::Closed);
                }
                Ok(vec![vec![1, 2]])
            }
            async fn commit(&mut self) -> Result<(), ConsumerError> {
                self.commit_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            async fn close(&mut self) -> Result<(), ConsumerError> {
                Ok(())
            }
        }

        let consumer = QuickConsumer {
            remaining: AtomicUsize::new(5),
            commit_count: commit_count.clone(),
        };

        let store = MockStore::new(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );

        let (pipeline, _) = make_test_pipeline(Box::new(consumer), store, true, 0);
        pipeline
            .run(CancellationToken::new())
            .await
            .expect("pipeline should succeed");

        assert_eq!(
            commit_count.load(Ordering::SeqCst),
            1,
            "exactly 1 commit (shutdown only)"
        );
    }

    #[tokio::test]
    async fn pipeline_respects_cancellation() {
        struct InfiniteConsumer {
            commit_count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl EventConsumer for InfiniteConsumer {
            async fn poll_batch(&mut self) -> Result<RawBatch, ConsumerError> {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(vec![vec![1, 2]])
            }
            async fn commit(&mut self) -> Result<(), ConsumerError> {
                self.commit_count.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            async fn close(&mut self) -> Result<(), ConsumerError> {
                Ok(())
            }
        }

        let commit_count = Arc::new(AtomicUsize::new(0));
        let consumer = InfiniteConsumer {
            commit_count: commit_count.clone(),
        };
        let store = MockStore::new(
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            cancel_clone.cancel();
        });

        let (pipeline, _) = make_test_pipeline(Box::new(consumer), store, true, 0);
        pipeline.run(cancel).await.expect("pipeline should succeed");

        assert!(
            commit_count.load(Ordering::SeqCst) >= 1,
            "should have committed at least once (shutdown)"
        );
    }
}
