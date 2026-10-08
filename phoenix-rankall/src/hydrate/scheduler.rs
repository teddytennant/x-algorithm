use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use log::info;
use tokio::sync::{Notify, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::features::{PostFeatures, FEATURES_VERSION};
use super::fetcher::PostFeatureFetcher;
use super::metrics::{
    HydrateMetrics, DECISION_ALREADY_QUEUED, DECISION_ENQUEUED, DECISION_QUEUE_FULL,
    DECISION_RECENTLY_FETCHED, OUTCOME_EMPTY, OUTCOME_ERROR, OUTCOME_OK, QUEUE_IN_FLIGHT,
};
use crate::processor::record::PostId;

const FLUSH_INTERVAL: Duration = Duration::from_secs(2);
const BACKFILL_SCAN_INTERVAL: Duration = Duration::from_secs(20);
const STATS_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    Live,
    Refresh,
    Backfill,
}

impl Priority {
    const ALL: [Priority; 3] = [Priority::Live, Priority::Refresh, Priority::Backfill];

    fn label(self) -> &'static str {
        match self {
            Priority::Live => "live",
            Priority::Refresh => "refresh",
            Priority::Backfill => "backfill",
        }
    }
}

pub struct Hydrated<P> {
    pub post_id: PostId,
    pub payload: Option<P>,
    pub features: PostFeatures,
}

pub struct Coverage {
    pub window: String,
    pub rows: usize,
    pub fetched: usize,
}

#[async_trait]
pub trait FeatureSink<P>: Send + Sync {
    async fn apply(&self, hydrated: Vec<Hydrated<P>>);
    async fn unfetched(&self, limit: usize) -> Vec<PostId>;
    async fn coverage(&self) -> Vec<Coverage>;
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_in_flight: usize,
    pub live_capacity: usize,
    pub refresh_capacity: usize,
    pub backfill_chunk: usize,
    pub min_refetch_interval: Duration,
}

impl SchedulerConfig {
    fn capacity(&self, priority: Priority) -> usize {
        match priority {
            Priority::Live => self.live_capacity,
            Priority::Refresh => self.refresh_capacity,
            Priority::Backfill => usize::MAX,
        }
    }
}

struct Job<P> {
    post_id: PostId,
    priority: Priority,
    payload: Option<P>,
}

enum Offer<P> {
    Queued,
    Skipped(&'static str),
    Full(Job<P>),
}

struct Queues<P> {
    queues: [VecDeque<Job<P>>; 3],
    pending: HashSet<PostId>,
    recent: HashMap<PostId, Instant>,
}

impl<P> Queues<P> {
    fn new() -> Self {
        Self {
            queues: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
            pending: HashSet::new(),
            recent: HashMap::new(),
        }
    }

    fn offer(&mut self, job: Job<P>, config: &SchedulerConfig, now: Instant) -> Offer<P> {
        if self.pending.contains(&job.post_id) {
            return Offer::Skipped(DECISION_ALREADY_QUEUED);
        }
        if self
            .recent
            .get(&job.post_id)
            .is_some_and(|at| now.duration_since(*at) < config.min_refetch_interval)
        {
            return Offer::Skipped(DECISION_RECENTLY_FETCHED);
        }
        let queue = &mut self.queues[job.priority as usize];
        if queue.len() >= config.capacity(job.priority) {
            return Offer::Full(job);
        }
        self.pending.insert(job.post_id);
        queue.push_back(job);
        Offer::Queued
    }

    fn pop(&mut self) -> Option<Job<P>> {
        self.queues.iter_mut().find_map(VecDeque::pop_front)
    }

    fn finish(&mut self, post_id: PostId, fetched_at: Option<Instant>) {
        self.pending.remove(&post_id);
        if let Some(at) = fetched_at {
            self.recent.insert(post_id, at);
        }
    }

    fn prune(&mut self, min_refetch_interval: Duration, now: Instant) {
        self.recent
            .retain(|_, at| now.duration_since(*at) < min_refetch_interval);
    }

    fn len(&self, priority: Priority) -> usize {
        self.queues[priority as usize].len()
    }
}

pub struct HydrationScheduler<P> {
    fetcher: Arc<PostFeatureFetcher>,
    config: SchedulerConfig,
    metrics: HydrateMetrics,
    queues: Mutex<Queues<P>>,
    results: Mutex<Vec<Hydrated<P>>>,
    work: Notify,
    space: Notify,
}

impl<P: Send + 'static> HydrationScheduler<P> {
    pub fn new(
        fetcher: Arc<PostFeatureFetcher>,
        config: SchedulerConfig,
        metrics: HydrateMetrics,
    ) -> Arc<Self> {
        Arc::new(Self {
            fetcher,
            config,
            metrics,
            queues: Mutex::new(Queues::new()),
            results: Mutex::new(Vec::new()),
            work: Notify::new(),
            space: Notify::new(),
        })
    }

    pub fn offer(&self, priority: Priority, jobs: impl IntoIterator<Item = (PostId, Option<P>)>) {
        let now = Instant::now();
        let mut queued = false;
        let mut queues = self.lock_queues();
        for (post_id, payload) in jobs {
            let job = Job {
                post_id,
                priority,
                payload,
            };
            let decision = match queues.offer(job, &self.config, now) {
                Offer::Queued => {
                    queued = true;
                    DECISION_ENQUEUED
                }
                Offer::Skipped(decision) => decision,
                Offer::Full(_) => DECISION_QUEUE_FULL,
            };
            self.count_decision(priority, decision);
        }
        drop(queues);
        if queued {
            self.work.notify_one();
        }
    }

    pub async fn submit(&self, priority: Priority, post_id: PostId, payload: Option<P>) {
        let mut job = Job {
            post_id,
            priority,
            payload,
        };
        loop {
            let space = self.space.notified();
            tokio::pin!(space);
            space.as_mut().enable();
            let offer = self.lock_queues().offer(job, &self.config, Instant::now());
            match offer {
                Offer::Queued => {
                    self.count_decision(priority, DECISION_ENQUEUED);
                    self.work.notify_one();
                    return;
                }
                Offer::Skipped(decision) => {
                    self.count_decision(priority, decision);
                    return;
                }
                Offer::Full(returned) => {
                    job = returned;
                    space.await;
                }
            }
        }
    }

    pub fn spawn(
        self: &Arc<Self>,
        sink: Arc<dyn FeatureSink<P>>,
        cancel: CancellationToken,
    ) -> JoinHandle<()> {
        let tasks = [
            tokio::spawn(Arc::clone(self).dispatch(cancel.clone())),
            tokio::spawn(Arc::clone(self).flush_loop(Arc::clone(&sink), cancel.clone())),
            tokio::spawn(Arc::clone(self).backfill_loop(Arc::clone(&sink), cancel.clone())),
            tokio::spawn(Arc::clone(self).stats_loop(sink, cancel)),
        ];
        tokio::spawn(async move {
            for task in tasks {
                let _ = task.await;
            }
        })
    }

    async fn dispatch(self: Arc<Self>, cancel: CancellationToken) {
        let slots = Arc::new(Semaphore::new(self.config.max_in_flight));
        loop {
            let slot = tokio::select! {
                slot = Arc::clone(&slots).acquire_owned() => slot.expect("slots are never closed"),
                _ = cancel.cancelled() => return,
            };
            let job = tokio::select! {
                job = self.next_job() => job,
                _ = cancel.cancelled() => return,
            };
            let this = Arc::clone(&self);
            tokio::spawn(async move {
                this.hydrate(job).await;
                drop(slot);
            });
        }
    }

    async fn next_job(&self) -> Job<P> {
        loop {
            let work = self.work.notified();
            if let Some(job) = self.lock_queues().pop() {
                self.space.notify_waiters();
                return job;
            }
            work.await;
        }
    }

    async fn hydrate(&self, job: Job<P>) {
        let result = self.fetcher.fetch(job.post_id).await;
        let fetched_at =
            (result.is_ok() && !self.config.min_refetch_interval.is_zero()).then(Instant::now);
        self.lock_queues().finish(job.post_id, fetched_at);
        let outcome = match result {
            Ok(features) => {
                let outcome = if features.author_id != 0 {
                    OUTCOME_OK
                } else {
                    OUTCOME_EMPTY
                };
                self.lock_results().push(Hydrated {
                    post_id: job.post_id,
                    payload: job.payload,
                    features,
                });
                outcome
            }
            Err(_) => OUTCOME_ERROR,
        };
        self.metrics
            .fetch_total
            .with_label_values(&[job.priority.label(), outcome])
            .inc();
    }

    async fn flush_loop(self: Arc<Self>, sink: Arc<dyn FeatureSink<P>>, cancel: CancellationToken) {
        loop {
            let stopping = tokio::select! {
                _ = tokio::time::sleep(FLUSH_INTERVAL) => false,
                _ = cancel.cancelled() => true,
            };
            let hydrated = std::mem::take(&mut *self.lock_results());
            if !hydrated.is_empty() {
                sink.apply(hydrated).await;
            }
            if stopping {
                return;
            }
        }
    }

    async fn backfill_loop(
        self: Arc<Self>,
        sink: Arc<dyn FeatureSink<P>>,
        cancel: CancellationToken,
    ) {
        if self.config.backfill_chunk == 0 {
            return;
        }
        loop {
            tokio::select! {
                _ = tokio::time::sleep(BACKFILL_SCAN_INTERVAL) => {}
                _ = cancel.cancelled() => return,
            }
            if self.lock_queues().len(Priority::Backfill) >= self.config.backfill_chunk / 2 {
                continue;
            }
            let post_ids = sink.unfetched(self.config.backfill_chunk).await;
            self.offer(
                Priority::Backfill,
                post_ids.into_iter().map(|id| (id, None)),
            );
        }
    }

    async fn stats_loop(self: Arc<Self>, sink: Arc<dyn FeatureSink<P>>, cancel: CancellationToken) {
        loop {
            tokio::select! {
                _ = tokio::time::sleep(STATS_INTERVAL) => {}
                _ = cancel.cancelled() => return,
            }
            let queue_sizes = self.record_queue_sizes();
            let coverage = self.record_coverage(sink.coverage().await);
            info!("post features v{FEATURES_VERSION}: coverage {coverage} | queues {queue_sizes}");
        }
    }

    fn record_queue_sizes(&self) -> String {
        let mut queues = self.lock_queues();
        queues.prune(self.config.min_refetch_interval, Instant::now());
        let mut queued = 0;
        let mut parts = Vec::new();
        for priority in Priority::ALL {
            let len = queues.len(priority);
            queued += len;
            self.set_queue_size(priority.label(), len);
            parts.push(format!("{}={len}", priority.label()));
        }
        let in_flight = queues.pending.len() - queued;
        self.set_queue_size(QUEUE_IN_FLIGHT, in_flight);
        parts.push(format!("{QUEUE_IN_FLIGHT}={in_flight}"));
        parts.join(" ")
    }

    fn record_coverage(&self, coverage: Vec<Coverage>) -> String {
        let mut parts = Vec::with_capacity(coverage.len());
        for Coverage {
            window,
            rows,
            fetched,
        } in coverage
        {
            let fraction = if rows == 0 {
                0.0
            } else {
                fetched as f64 / rows as f64
            };
            self.metrics
                .coverage
                .with_label_values(&[&window])
                .set(fraction);
            self.metrics
                .unfetched_rows
                .with_label_values(&[&window])
                .set((rows - fetched) as f64);
            parts.push(format!(
                "{window}: {fetched}/{rows} ({:.1}%)",
                100.0 * fraction
            ));
        }
        parts.join(", ")
    }

    fn set_queue_size(&self, queue: &str, len: usize) {
        self.metrics
            .queue_size
            .with_label_values(&[queue])
            .set(len as f64);
    }

    fn count_decision(&self, priority: Priority, decision: &str) {
        self.metrics
            .sightings_total
            .with_label_values(&[priority.label(), decision])
            .inc();
    }

    fn lock_queues(&self) -> MutexGuard<'_, Queues<P>> {
        self.queues.lock().expect("hydration queues lock poisoned")
    }

    fn lock_results(&self) -> MutexGuard<'_, Vec<Hydrated<P>>> {
        self.results
            .lock()
            .expect("hydration results lock poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(live_capacity: usize, min_refetch_interval: Duration) -> SchedulerConfig {
        SchedulerConfig {
            max_in_flight: 1,
            live_capacity,
            refresh_capacity: 1,
            backfill_chunk: 0,
            min_refetch_interval,
        }
    }

    fn job(post_id: PostId, priority: Priority) -> Job<()> {
        Job {
            post_id,
            priority,
            payload: None,
        }
    }

    #[test]
    fn pops_by_priority_and_drops_duplicates_and_overflow() {
        let config = config(2, Duration::ZERO);
        let now = Instant::now();
        let mut queues = Queues::new();
        assert!(matches!(
            queues.offer(job(1, Priority::Backfill), &config, now),
            Offer::Queued
        ));
        assert!(matches!(
            queues.offer(job(2, Priority::Refresh), &config, now),
            Offer::Queued
        ));
        assert!(matches!(
            queues.offer(job(3, Priority::Refresh), &config, now),
            Offer::Full(_)
        ));
        assert!(matches!(
            queues.offer(job(4, Priority::Live), &config, now),
            Offer::Queued
        ));
        assert!(matches!(
            queues.offer(job(1, Priority::Live), &config, now),
            Offer::Skipped(DECISION_ALREADY_QUEUED)
        ));
        let order: Vec<PostId> = std::iter::from_fn(|| queues.pop().map(|j| j.post_id)).collect();
        assert_eq!(order, vec![4, 2, 1]);
    }

    #[test]
    fn recent_fetch_gates_until_interval_passes() {
        let interval = Duration::from_secs(1800);
        let config = config(10, interval);
        let fetched_at = Instant::now();
        let mut queues = Queues::new();
        queues.finish(7, Some(fetched_at));
        assert!(matches!(
            queues.offer(job(7, Priority::Live), &config, fetched_at + interval / 2),
            Offer::Skipped(DECISION_RECENTLY_FETCHED)
        ));
        queues.prune(interval, fetched_at + interval);
        assert!(matches!(
            queues.offer(job(7, Priority::Live), &config, fetched_at + interval),
            Offer::Queued
        ));
    }

    #[tokio::test]
    async fn submit_waits_for_space_then_queues() {
        let scheduler = HydrationScheduler::<()>::new(
            Arc::new(PostFeatureFetcher::new(Vec::new())),
            config(1, Duration::ZERO),
            HydrateMetrics::for_tests(),
        );
        scheduler.submit(Priority::Live, 1, None).await;
        let waiting = tokio::spawn({
            let scheduler = Arc::clone(&scheduler);
            async move { scheduler.submit(Priority::Live, 2, None).await }
        });
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        assert_eq!(scheduler.next_job().await.post_id, 1);
        waiting.await.unwrap();
        assert_eq!(scheduler.next_job().await.post_id, 2);
    }
}
