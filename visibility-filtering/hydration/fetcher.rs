use crate::hydration::batch::{Hydrated, HydrationBatch, HydrationError};
use crate::hydration::Cause;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

pub(super) struct Fetcher<V> {
    states: FxHashMap<u64, State<V>>,
}

enum State<V> {
    Pending,
    Landed(Hydrated<V>),
}

impl<V> State<V> {
    fn is_complete(&self) -> bool {
        match self {
            State::Landed(answer) => answer.is_complete(),
            State::Pending => false,
        }
    }
}

impl<V> Default for Fetcher<V> {
    fn default() -> Self {
        Self {
            states: FxHashMap::default(),
        }
    }
}

impl<V> Fetcher<V> {
    pub(super) fn land(&mut self, claimed: &[u64], batch: HydrationBatch<u64, V>) {
        let mut answers = batch.into_hydrated();
        for &key in claimed {
            let answer = answers.remove(&key);
            debug_assert!(answer.is_some(), "a call answers every key it claimed");
            let answer = answer.unwrap_or(Hydrated::Failed(HydrationError::Error));
            self.states.insert(key, State::Landed(answer));
        }
    }

    pub(super) fn get(&self, key: u64) -> Option<&V> {
        match self.states.get(&key)? {
            State::Landed(answer) => answer.value(),
            State::Pending => None,
        }
    }

    pub(super) fn take(&mut self, key: u64) -> Option<V> {
        match self.states.remove(&key)? {
            State::Landed(answer) => answer.into_value(),
            State::Pending => None,
        }
    }
}

pub(super) trait AnyFetcher {
    fn claim(&mut self, keys: Vec<u64>) -> Vec<u64>;
    fn has_claimed(&self) -> bool;
    fn is_claimed(&self, key: u64) -> bool;
    fn has_incomplete(&self) -> bool;
    fn is_incomplete(&self, key: u64) -> bool;
    fn miss(&self, key: u64) -> Option<Cause>;
}

impl<V> AnyFetcher for Fetcher<V> {
    fn claim(&mut self, mut keys: Vec<u64>) -> Vec<u64> {
        self.states.reserve(keys.len());
        keys.retain(|&key| match self.states.entry(key) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(State::Pending);
                true
            }
        });
        keys
    }

    fn has_claimed(&self) -> bool {
        !self.states.is_empty()
    }

    fn is_claimed(&self, key: u64) -> bool {
        self.states.contains_key(&key)
    }

    fn has_incomplete(&self) -> bool {
        self.states.values().any(|state| !state.is_complete())
    }

    fn is_incomplete(&self, key: u64) -> bool {
        self.states
            .get(&key)
            .is_some_and(|state| !state.is_complete())
    }

    fn miss(&self, key: u64) -> Option<Cause> {
        match self.states.get(&key) {
            Some(State::Landed(Hydrated::Found(_) | Hydrated::Partial(_))) => None,
            Some(State::Landed(Hydrated::NotFound)) => Some(Cause::NotFound),
            Some(State::Landed(Hydrated::Failed(_)) | State::Pending) | None => Some(Cause::Failed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_found_and_not_found_keys_are_complete() {
        let mut fetcher = Fetcher::default();
        let keys = fetcher.claim(vec![1, 2, 3, 4, 5]);
        fetcher.land(
            &keys[..4],
            HydrationBatch::from_hydrated(FxHashMap::from_iter([
                (1, Hydrated::Found(7)),
                (2, Hydrated::NotFound),
                (3, Hydrated::Partial(7)),
                (4, Hydrated::Failed(HydrationError::Timeout)),
            ])),
        );

        let incomplete: Vec<u64> = keys
            .into_iter()
            .filter(|&key| fetcher.is_incomplete(key))
            .collect();
        assert_eq!(incomplete, [3, 4, 5]);
        assert_eq!(fetcher.get(3), Some(&7));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a call answers every key it claimed")]
    fn a_call_that_leaves_out_a_claimed_key_breaks_the_fetcher_contract() {
        let mut fetcher = Fetcher::default();
        let keys = fetcher.claim(vec![1, 2]);
        fetcher.land(
            &keys,
            HydrationBatch::from_hydrated(FxHashMap::from_iter([(1, Hydrated::Found(7))])),
        );
    }
}
