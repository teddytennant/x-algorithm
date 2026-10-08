use serde_json::Value;
use tracing::error;

use crate::metrics::CONFIG_RESTART_PENDING;

pub const OVERRIDE_KEY: &str = "restart_on_config_change";

pub const DEFAULT_WATCHED: &[&str] = &[
    "/kafka/consumer/topic_labels",
    "/topic_labels",
    "/kafka/producer",
];

pub fn watched_paths(config: &Value) -> Vec<String> {
    match config.get(OVERRIDE_KEY).and_then(Value::as_array) {
        Some(entries) => entries
            .iter()
            .filter_map(Value::as_str)
            .filter(|p| p.starts_with('/'))
            .map(str::to_string)
            .collect(),
        None => DEFAULT_WATCHED.iter().map(|p| p.to_string()).collect(),
    }
}

pub fn changed_paths(boot: &Value, live: &Value, paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|p| boot.pointer(p) != live.pointer(p))
        .cloned()
        .collect()
}

#[derive(Debug, PartialEq)]
pub enum Check {
    Unchanged,
    Invalid,
    Restart(Vec<String>),
}

pub fn check(boot: &Value, live: &Value, validate: fn(&Value) -> anyhow::Result<()>) -> Check {
    let changed = changed_paths(boot, live, &watched_paths(live));
    if changed.is_empty() {
        return Check::Unchanged;
    }
    match validate(live) {
        Ok(()) => Check::Restart(changed),
        Err(e) => {
            error!(
                "config change needs a restart but does not validate; not restarting: paths={changed:?} error={e:#}"
            );
            Check::Invalid
        }
    }
}

pub fn record(check: &Check) {
    let (valid, invalid) = match check {
        Check::Unchanged => (0, 0),
        Check::Invalid => (0, 1),
        Check::Restart(_) => (1, 0),
    };
    CONFIG_RESTART_PENDING
        .with_label_values(&["true"])
        .set(valid);
    CONFIG_RESTART_PENDING
        .with_label_values(&["false"])
        .set(invalid);
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn ok(_: &Value) -> anyhow::Result<()> {
        Ok(())
    }

    fn reject(_: &Value) -> anyhow::Result<()> {
        anyhow::bail!("invalid")
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn defaults() -> Vec<String> {
        strings(DEFAULT_WATCHED)
    }

    fn boot() -> Value {
        json!({
            "kafka": {
                "consumer": {"topic_labels": {"a": ["x"]}},
                "producer": {"sink": "one"}
            },
            "other": 1
        })
    }

    fn with_topics(topics: Value) -> Value {
        let mut v = boot();
        v["kafka"]["consumer"]["topic_labels"] = topics;
        v
    }

    #[test]
    fn watched_paths_default_unless_override_is_an_array() {
        for cfg in [
            json!({}),
            json!({OVERRIDE_KEY: null}),
            json!({OVERRIDE_KEY: "/a"}),
            json!({OVERRIDE_KEY: {"x": 1}}),
        ] {
            assert_eq!(watched_paths(&cfg), defaults(), "{cfg}");
        }
    }

    #[test]
    fn watched_paths_override_replaces_default_and_drops_bad_entries() {
        let cfg = json!({OVERRIDE_KEY: ["/a", "", "b", 5, "/c/d"]});
        assert_eq!(watched_paths(&cfg), strings(&["/a", "/c/d"]));
        assert!(watched_paths(&json!({OVERRIDE_KEY: []})).is_empty());
    }

    #[test]
    fn changed_paths_detects_added_removed_and_nested() {
        let paths = strings(&["/a", "/b"]);
        assert_eq!(
            changed_paths(&json!({"a": 1}), &json!({"b": 2}), &paths),
            paths
        );

        let mut live = boot();
        live["kafka"]["producer"]["sink"] = json!("two");
        assert_eq!(
            changed_paths(&boot(), &live, &defaults()),
            strings(&["/kafka/producer"])
        );
    }

    #[test]
    fn changed_paths_ignores_key_order_and_unwatched() {
        let paths = strings(&["/m"]);
        let boot: Value = serde_json::from_str(r#"{"m": {"a": 1, "b": 2}, "x": 1}"#).unwrap();
        let live: Value = serde_json::from_str(r#"{"m": {"b": 2, "a": 1}, "x": 2}"#).unwrap();
        assert!(changed_paths(&boot, &live, &paths).is_empty());
    }

    #[test]
    fn check_outcomes() {
        let mut unwatched = boot();
        unwatched["other"] = json!(2);
        assert_eq!(check(&boot(), &unwatched, ok), Check::Unchanged);

        let changed = with_topics(json!({"b": ["y"]}));
        assert_eq!(
            check(&boot(), &changed, ok),
            Check::Restart(strings(&["/kafka/consumer/topic_labels"]))
        );
        assert_eq!(check(&boot(), &changed, reject), Check::Invalid);

        let mut off = changed.clone();
        off[OVERRIDE_KEY] = json!([]);
        assert_eq!(check(&boot(), &off, ok), Check::Unchanged);

        let mut producers_only = changed;
        producers_only[OVERRIDE_KEY] = json!(["/kafka/producer"]);
        assert_eq!(check(&boot(), &producers_only, ok), Check::Unchanged);
    }
}
