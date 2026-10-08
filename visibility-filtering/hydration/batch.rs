use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::slice;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HydrationError {
    Timeout,
    Error,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Hydrated<V> {
    Found(V),
    Partial(V),
    NotFound,
    Failed(HydrationError),
}

impl<V> Hydrated<V> {
    pub(crate) fn value(&self) -> Option<&V> {
        match self {
            Hydrated::Found(value) | Hydrated::Partial(value) => Some(value),
            Hydrated::NotFound | Hydrated::Failed(_) => None,
        }
    }

    pub(crate) fn into_value(self) -> Option<V> {
        match self {
            Hydrated::Found(value) | Hydrated::Partial(value) => Some(value),
            Hydrated::NotFound | Hydrated::Failed(_) => None,
        }
    }

    pub(crate) fn is_complete(&self) -> bool {
        matches!(self, Hydrated::Found(_) | Hydrated::NotFound)
    }
}

impl<V, E> From<Result<Option<V>, E>> for Hydrated<V> {
    fn from(result: Result<Option<V>, E>) -> Self {
        match result {
            Ok(Some(value)) => Hydrated::Found(value),
            Ok(None) => Hydrated::NotFound,
            Err(_) => Hydrated::Failed(HydrationError::Error),
        }
    }
}

#[derive(Clone)]
pub(crate) struct HydrationBatch<K, V> {
    results: FxHashMap<K, Hydrated<V>>,
}

pub(crate) type RawHydrationBatch<V> = HydrationBatch<u64, V>;

impl<K: Eq + Hash, V> HydrationBatch<K, V> {
    pub(crate) fn empty() -> Self {
        Self {
            results: FxHashMap::default(),
        }
    }

    pub(crate) fn from_results<E, S: BuildHasher>(
        expected: impl IntoIterator<Item = K>,
        mut results: HashMap<K, Result<Option<V>, E>, S>,
    ) -> Self {
        Self::from_expected(expected, |key| {
            results
                .remove(key)
                .map(Hydrated::from)
                .unwrap_or(Hydrated::Failed(HydrationError::Error))
        })
    }

    pub(crate) fn from_hydrated(results: FxHashMap<K, Hydrated<V>>) -> Self {
        Self { results }
    }

    fn from_expected(
        expected: impl IntoIterator<Item = K>,
        mut resolve: impl FnMut(&K) -> Hydrated<V>,
    ) -> Self {
        let mut results = FxHashMap::default();
        for key in expected {
            if let Entry::Vacant(entry) = results.entry(key) {
                let hydrated = resolve(entry.key());
                entry.insert(hydrated);
            }
        }
        Self { results }
    }

    pub(crate) fn timed_out(expected: impl IntoIterator<Item = K>) -> Self {
        Self {
            results: expected
                .into_iter()
                .map(|key| (key, Hydrated::Failed(HydrationError::Timeout)))
                .collect(),
        }
    }

    pub(crate) fn hydrated(&self, key: &K) -> Option<&Hydrated<V>> {
        self.results.get(key)
    }

    pub(crate) fn into_hydrated(self) -> FxHashMap<K, Hydrated<V>> {
        self.results
    }

    #[cfg(test)]
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        self.results.get(key).and_then(Hydrated::value)
    }

    pub(crate) fn map<V2>(self, mut f: impl FnMut(V) -> V2) -> HydrationBatch<K, V2> {
        HydrationBatch {
            results: self
                .results
                .into_iter()
                .map(|(key, hydrated)| {
                    let hydrated = match hydrated {
                        Hydrated::Found(value) => Hydrated::Found(f(value)),
                        Hydrated::Partial(value) => Hydrated::Partial(f(value)),
                        Hydrated::NotFound => Hydrated::NotFound,
                        Hydrated::Failed(e) => Hydrated::Failed(e),
                    };
                    (key, hydrated)
                })
                .collect(),
        }
    }
}

impl<K, V> AsRef<[HydrationBatch<K, V>]> for HydrationBatch<K, V> {
    fn as_ref(&self) -> &[Self] {
        slice::from_ref(self)
    }
}

impl<K: Eq + Hash, V> Default for HydrationBatch<K, V> {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(results: HashMap<u64, Result<Option<u32>, &str>>) -> HydrationBatch<u64, u32> {
        HydrationBatch::from_results(results.keys().copied().collect::<Vec<_>>(), results)
    }

    #[test]
    fn found_not_found_and_failed_stay_distinguishable() {
        let batch = batch(HashMap::from([
            (1, Ok(Some(7))),
            (2, Ok(None)),
            (3, Err("backend unavailable")),
        ]));

        assert_eq!(batch.get(&1), Some(&7));
        assert_eq!(batch.hydrated(&2), Some(&Hydrated::NotFound));
        assert_eq!(
            batch.hydrated(&3),
            Some(&Hydrated::Failed(HydrationError::Error))
        );
    }

    #[test]
    fn absent_expected_keys_become_missing() {
        let batch: HydrationBatch<u64, u32> =
            HydrationBatch::from_results([1, 2], HashMap::from([(1, Ok::<_, &str>(Some(7)))]));

        assert_eq!(
            batch.hydrated(&2),
            Some(&Hydrated::Failed(HydrationError::Error))
        );
    }

    #[test]
    fn map_preserves_tri_state() {
        let mapped = batch(HashMap::from([
            (1, Ok(Some(7))),
            (2, Ok(None)),
            (3, Err("boom")),
        ]))
        .map(|v| v * 10);

        assert_eq!(mapped.get(&1), Some(&70));
        assert_eq!(mapped.hydrated(&2), Some(&Hydrated::NotFound));
        assert!(matches!(mapped.hydrated(&3), Some(&Hydrated::Failed(_))));
    }
}
