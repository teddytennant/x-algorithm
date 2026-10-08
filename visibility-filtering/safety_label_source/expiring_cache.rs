use std::hash::Hash;
use std::time::Duration;

use quanta::{Clock, Instant};
use quick_cache::sync::Cache;

pub(crate) enum Lookup<V> {
    Found(V),
    Expired,
    NotFound,
}

#[derive(Clone)]
struct Entry<V> {
    value: V,
    expires_at: Instant,
}

pub(crate) struct ExpiringCache<K, V> {
    cache: Cache<K, Entry<V>>,
    clock: Clock,
}

impl<K, V> ExpiringCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub(crate) fn with_clock(capacity: usize, clock: Clock) -> Self {
        let cache = Cache::new(capacity);
        cache.reserve(capacity);
        Self { cache, clock }
    }

    pub(crate) fn get(&self, key: &K) -> Lookup<V> {
        let now = self.clock.now();
        match self.cache.get(key) {
            Some(entry) if now < entry.expires_at => Lookup::Found(entry.value),
            Some(_) => Lookup::Expired,
            None => Lookup::NotFound,
        }
    }

    pub(crate) fn insert(&self, key: K, value: V, ttl: Duration) {
        self.cache.insert(
            key,
            Entry {
                value,
                expires_at: self.clock.now() + ttl,
            },
        );
    }

    #[cfg(test)]
    pub(crate) fn expiry_of(&self, key: &K) -> Option<Instant> {
        self.cache.get(key).map(|entry| entry.expires_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTL: Duration = Duration::from_secs(30);

    #[test]
    fn entry_past_ttl_is_expired() {
        let (clock, mock) = Clock::mock();
        let cache = ExpiringCache::<u64, u32>::with_clock(8, clock);
        cache.insert(1, 42, TTL);
        mock.increment(TTL + Duration::from_secs(1));
        assert!(matches!(cache.get(&1), Lookup::Expired));
    }

    #[test]
    fn entry_exactly_at_ttl_is_expired() {
        let (clock, mock) = Clock::mock();
        let cache = ExpiringCache::<u64, u32>::with_clock(8, clock);
        cache.insert(1, 42, TTL);
        mock.increment(TTL);
        assert!(matches!(cache.get(&1), Lookup::Expired));
    }

    #[test]
    fn reinsert_restamps_expiry() {
        let (clock, mock) = Clock::mock();
        let cache = ExpiringCache::<u64, u32>::with_clock(8, clock);
        cache.insert(1, 42, TTL);
        assert!(matches!(cache.get(&1), Lookup::Found(42)));
        let first = cache.expiry_of(&1).unwrap();

        mock.increment(TTL + Duration::from_secs(1));
        cache.insert(1, 7, TTL);

        let restamped = cache.expiry_of(&1).unwrap();
        assert!(restamped > first);
        assert!(matches!(cache.get(&1), Lookup::Found(7)));
    }
}
