use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Default)]
pub(crate) struct RecordingReceiver {
    counters: Mutex<HashMap<String, u64>>,
}

impl RecordingReceiver {
    pub(crate) fn counter(&self, key: &str) -> u64 {
        *self.counters.lock().unwrap().get(key).unwrap_or(&0)
    }
}

impl xai_stats_receiver::StatsReceiverExt for RecordingReceiver {
    fn incr(&self, name: &str, scopes: &[(&str, &str)], value: u64) {
        let mut key = name.to_string();
        for (k, v) in scopes {
            key.push('|');
            key.push_str(k);
            key.push('=');
            key.push_str(v);
        }
        *self.counters.lock().unwrap().entry(key).or_default() += value;
    }
    fn observe(
        &self,
        _: &str,
        _: &[(&str, &str)],
        _: f64,
        _: xai_stats_receiver::HistogramBuckets,
    ) {
    }
    fn observe_expo(&self, _: &str, _: &[(&str, &str)], _: f64) {}
    fn observe_vm(&self, _: &str, _: &[(&str, &str)], _: f64) {}
    fn gauge(&self, _: &str, _: &[(&str, &str)], _: f64) {}
}
