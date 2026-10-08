use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use quick_cache::sync::{Cache, EntryAction, EntryResult};
use quick_cache::OptionsBuilder;

use crate::hydration::batch::{Hydrated, RawHydrationBatch};
use crate::hydration::metrics::{record_fallback_cache_entries, FallbackCacheCounts};

const CACHE_SHARDS: usize = 64;
const OCCUPANCY_SAMPLE_INTERVAL: u64 = 1024;
const NO_WAIT: Option<Duration> = Some(Duration::ZERO);

pub(crate) trait Column {
    type Entry: Clone;
    type Value: Clone;
    type Stored;
    const NAME: &'static str;

    fn store(value: &Self::Value) -> Self::Stored;
    fn new_entry(stored: Self::Stored) -> Self::Entry;
    fn replace(entry: &mut Self::Entry, stored: Self::Stored) -> Option<Self::Stored>;
    fn get(entry: &Self::Entry) -> Option<Self::Value>;
    fn holds(entry: &Self::Entry) -> bool;
    fn others_hold(entry: &Self::Entry) -> bool;
    fn clear(entry: &mut Self::Entry) -> Option<Self::Stored>;
}

pub(crate) struct FallbackCache<E> {
    cache: &'static str,
    entries: Cache<u64, E>,
    resolved_batches: AtomicU64,
}

impl<E: Clone> FallbackCache<E> {
    #[expect(
        clippy::expect_used,
        reason = "both options `build` requires are set, so it cannot fail"
    )]
    pub(crate) fn new(cache: &'static str, capacity: usize) -> Self {
        let options = OptionsBuilder::new()
            .shards(CACHE_SHARDS)
            .estimated_items_capacity(capacity)
            .weight_capacity(u64::try_from(capacity).unwrap_or(u64::MAX))
            .build()
            .expect("capacity options are set");
        let entries = Cache::with_options(
            options,
            Default::default(),
            Default::default(),
            Default::default(),
        );
        entries.reserve(capacity);
        Self {
            cache,
            entries,
            resolved_batches: AtomicU64::new(0),
        }
    }

    pub(crate) fn resolve_hydration_batch<C: Column<Entry = E>>(
        &self,
        batch: RawHydrationBatch<C::Value>,
    ) -> RawHydrationBatch<C::Value> {
        let mut counts = FallbackCacheCounts::default();
        let resolved = batch
            .into_hydrated()
            .into_iter()
            .map(|(key, hydrated)| {
                let hydrated = match hydrated {
                    Hydrated::Found(value) => {
                        counts.resident += usize::from(self.set::<C>(key, &value));
                        counts.fresh += 1;
                        Hydrated::Found(value)
                    }
                    Hydrated::NotFound => {
                        counts.resident += usize::from(self.clear::<C>(key));
                        counts.not_found += 1;
                        Hydrated::NotFound
                    }
                    Hydrated::Partial(value) => {
                        match self.entries.get(&key).as_ref().and_then(C::get) {
                            Some(cached) => {
                                counts.resident += 1;
                                counts.partial_stale += 1;
                                Hydrated::Found(cached)
                            }
                            None => {
                                counts.partial += 1;
                                Hydrated::Partial(value)
                            }
                        }
                    }
                    Hydrated::Failed(error) => {
                        match self.entries.get(&key).as_ref().and_then(C::get) {
                            Some(value) => {
                                counts.stale += 1;
                                Hydrated::Found(value)
                            }
                            None => {
                                counts.unavailable += 1;
                                Hydrated::Failed(error)
                            }
                        }
                    }
                };
                (key, hydrated)
            })
            .collect();

        counts.record(self.cache, C::NAME);
        if self
            .resolved_batches
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(OCCUPANCY_SAMPLE_INTERVAL)
        {
            record_fallback_cache_entries(self.cache, self.entries.len());
        }
        RawHydrationBatch::from_hydrated(resolved)
    }

    #[expect(
        clippy::let_underscore_must_use,
        reason = "a placeholder insert fails only when a concurrent write already replaced it, and that write stands"
    )]
    fn set<C: Column<Entry = E>>(&self, key: u64, value: &C::Value) -> bool {
        let mut stored = Some(C::store(value));
        let written = self.entries.entry(&key, NO_WAIT, |_, entry| {
            EntryAction::Retain(stored.take().and_then(|stored| C::replace(entry, stored)))
        });
        match written {
            EntryResult::Retained(previous) => previous.is_some(),
            EntryResult::Vacant(guard) => {
                if let Some(stored) = stored {
                    let _ = guard.insert(C::new_entry(stored));
                }
                false
            }
            EntryResult::Removed(..) | EntryResult::Replaced(..) | EntryResult::Timeout => false,
        }
    }

    fn clear<C: Column<Entry = E>>(&self, key: u64) -> bool {
        let mut seen = None;
        self.entries.remove_if(&key, |entry| {
            let others_hold = C::others_hold(entry);
            seen = Some((C::holds(entry), others_hold));
            !others_hold
        });
        let Some((held, others_hold)) = seen else {
            return false;
        };
        if held && others_hold {
            let mut cleared = None;
            self.entries.entry(&key, NO_WAIT, |_, entry| {
                cleared = C::clear(entry);
                if C::others_hold(entry) {
                    EntryAction::Retain(())
                } else {
                    EntryAction::Remove
                }
            });
        }
        held
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::{HydrationBatch, HydrationError};

    type Pair = (Option<String>, Option<String>);

    struct Left;
    struct Right;

    impl Column for Left {
        type Entry = Pair;
        type Value = String;
        type Stored = String;
        const NAME: &'static str = "left";

        fn store(value: &String) -> String {
            value.clone()
        }
        fn new_entry(stored: String) -> Pair {
            (Some(stored), None)
        }
        fn replace(entry: &mut Pair, stored: String) -> Option<String> {
            entry.0.replace(stored)
        }
        fn get(entry: &Pair) -> Option<String> {
            entry.0.clone()
        }
        fn holds(entry: &Pair) -> bool {
            entry.0.is_some()
        }
        fn others_hold(entry: &Pair) -> bool {
            entry.1.is_some()
        }
        fn clear(entry: &mut Pair) -> Option<String> {
            entry.0.take()
        }
    }

    impl Column for Right {
        type Entry = Pair;
        type Value = String;
        type Stored = String;
        const NAME: &'static str = "right";

        fn store(value: &String) -> String {
            value.clone()
        }
        fn new_entry(stored: String) -> Pair {
            (None, Some(stored))
        }
        fn replace(entry: &mut Pair, stored: String) -> Option<String> {
            entry.1.replace(stored)
        }
        fn get(entry: &Pair) -> Option<String> {
            entry.1.clone()
        }
        fn holds(entry: &Pair) -> bool {
            entry.1.is_some()
        }
        fn others_hold(entry: &Pair) -> bool {
            entry.0.is_some()
        }
        fn clear(entry: &mut Pair) -> Option<String> {
            entry.1.take()
        }
    }

    fn cache() -> FallbackCache<Pair> {
        FallbackCache::new("test", 8)
    }

    fn batch(
        entries: impl IntoIterator<Item = (u64, Hydrated<String>)>,
    ) -> HydrationBatch<u64, String> {
        HydrationBatch::from_hydrated(entries.into_iter().collect())
    }

    fn found(value: &str) -> Hydrated<String> {
        Hydrated::Found(value.to_string())
    }

    fn failed() -> Hydrated<String> {
        Hydrated::Failed(HydrationError::Timeout)
    }

    #[test]
    fn recovers_only_resident_failed_keys() {
        let cache = cache();
        cache.resolve_hydration_batch::<Left>(batch([(1, found("cached"))]));

        let resolved = cache.resolve_hydration_batch::<Left>(batch([
            (1, failed()),
            (2, failed()),
            (3, found("fresh")),
        ]));

        assert_eq!(resolved.get(&1), Some(&"cached".to_string()));
        assert!(matches!(resolved.hydrated(&2), Some(Hydrated::Failed(_))));
        assert_eq!(resolved.get(&3), Some(&"fresh".to_string()));
    }

    #[test]
    fn a_later_value_replaces_the_cached_one() {
        let cache = cache();
        cache.resolve_hydration_batch::<Left>(batch([(1, found("old"))]));
        cache.resolve_hydration_batch::<Left>(batch([(1, found("new"))]));

        let failed = cache.resolve_hydration_batch::<Left>(batch([(1, failed())]));

        assert_eq!(failed.get(&1), Some(&"new".to_string()));
    }

    #[test]
    fn a_partial_answer_takes_the_complete_entry_and_leaves_it() {
        let cache = cache();
        cache.resolve_hydration_batch::<Left>(batch([(1, found("complete"))]));

        let partial = cache.resolve_hydration_batch::<Left>(batch([(
            1,
            Hydrated::Partial("partial".to_string()),
        )]));
        let later = cache.resolve_hydration_batch::<Left>(batch([(1, failed())]));

        assert_eq!(partial.hydrated(&1), Some(&found("complete")));
        assert_eq!(later.hydrated(&1), Some(&found("complete")));
    }

    #[test]
    fn a_partial_answer_creates_no_entry() {
        let cache = cache();
        cache.resolve_hydration_batch::<Left>(batch([(
            1,
            Hydrated::Partial("partial".to_string()),
        )]));

        let later = cache.resolve_hydration_batch::<Left>(batch([(1, failed())]));

        assert_eq!(later.hydrated(&1), Some(&failed()));
    }

    #[test]
    fn authoritative_not_found_invalidates_stale_value() {
        let cache = cache();
        cache.resolve_hydration_batch::<Left>(batch([(1, found("cached"))]));

        cache.resolve_hydration_batch::<Left>(batch([(1, Hydrated::NotFound)]));
        let failed = cache.resolve_hydration_batch::<Left>(batch([(1, failed())]));

        assert!(matches!(failed.hydrated(&1), Some(Hydrated::Failed(_))));
    }

    #[test]
    fn each_column_serves_only_what_its_own_source_wrote() {
        let cache = cache();
        cache.resolve_hydration_batch::<Left>(batch([(1, found("left")), (2, found("left"))]));
        cache.resolve_hydration_batch::<Right>(batch([(1, found("right"))]));

        let left = cache.resolve_hydration_batch::<Left>(batch([(1, failed()), (2, failed())]));
        let right = cache.resolve_hydration_batch::<Right>(batch([(1, failed()), (2, failed())]));

        assert_eq!(left.hydrated(&1), Some(&found("left")));
        assert_eq!(left.hydrated(&2), Some(&found("left")));
        assert_eq!(right.hydrated(&1), Some(&found("right")));
        assert_eq!(right.hydrated(&2), Some(&failed()));
    }

    #[test]
    fn a_not_found_clears_only_its_own_column() {
        let cache = cache();
        cache.resolve_hydration_batch::<Left>(batch([(1, found("left"))]));
        cache.resolve_hydration_batch::<Right>(batch([(1, found("right"))]));

        cache.resolve_hydration_batch::<Right>(batch([(1, Hydrated::NotFound)]));
        let left = cache.resolve_hydration_batch::<Left>(batch([(1, failed())]));
        let right = cache.resolve_hydration_batch::<Right>(batch([(1, failed())]));

        assert_eq!(left.hydrated(&1), Some(&found("left")));
        assert_eq!(right.hydrated(&1), Some(&failed()));

        cache.resolve_hydration_batch::<Left>(batch([(1, Hydrated::NotFound)]));
        assert!(cache.entries.is_empty());
    }
}
