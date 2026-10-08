pub const ENV_AUTHOR_CACHE_CAPACITY: &str = "VF_AUTHOR_CACHE_CAPACITY";
pub const ENV_TWEET_CACHE_CAPACITY: &str = "VF_TWEET_CACHE_CAPACITY";
pub const ENV_SAFETY_LABEL_CACHE_CAPACITY: &str = "VF_SAFETY_LABEL_CACHE_CAPACITY";
pub const ENV_CACHE_WARM_ENABLED: &str = "VF_CACHE_WARM_ENABLED";
pub const ENV_REFERENCE: &str = "VF_REFERENCE";
pub const ENV_DARK_TRAFFIC_ENABLED: &str = "DARK_TRAFFIC_ENABLED";
pub const ENV_APP_ENV: &str = "APP_ENV";
pub const ENV_FS_PATH: &str = "VF_FS_PATH";
pub const ENV_GIZMODUCK_CLIENT_ID: &str = "VF_GIZMODUCK_CLIENT_ID";
pub const ENV_TWEMCACHE_CLIENT_NAME: &str = "VF_TWEMCACHE_CLIENT_NAME";

const DEFAULT_FS_PATH: &str = "/usr/local/config/features/visibility/main/rust_vf.yml";

pub fn fs_path() -> String {
    std::env::var(ENV_FS_PATH).unwrap_or_else(|_| DEFAULT_FS_PATH.to_string())
}

pub(crate) fn gizmoduck_client_id() -> String {
    resolve_gizmoduck_client_id(
        std::env::var(ENV_GIZMODUCK_CLIENT_ID).ok().as_deref(),
        std::env::var(ENV_APP_ENV).ok().as_deref(),
    )
}

pub(crate) fn twemcache_client_name() -> String {
    resolve_twemcache_client_name(std::env::var(ENV_TWEMCACHE_CLIENT_NAME).ok().as_deref())
}

pub fn resolve_gizmoduck_client_id(configured: Option<&str>, app_env: Option<&str>) -> String {
    match configured {
        Some(id) => id.to_string(),
        None => format!("visibility-filtering-service.{}", app_env.unwrap_or("prod")),
    }
}

pub fn resolve_twemcache_client_name(configured: Option<&str>) -> String {
    configured
        .unwrap_or("visibility-filtering-service")
        .to_string()
}

#[expect(clippy::panic, reason = "startup fail-fast on misconfiguration")]
pub(crate) fn refuse_reference() {
    if let Ok(reference) = std::env::var(ENV_REFERENCE)
        && reference != "none"
    {
        panic!("{ENV_REFERENCE}={reference} needs the staging binary's serve");
    }
}

const DEFAULT_CACHE_CAPACITY: usize = 1_000_000;

pub(crate) fn author_cache_capacity() -> Option<usize> {
    cache_capacity(ENV_AUTHOR_CACHE_CAPACITY)
}

pub(crate) fn tweet_cache_capacity() -> Option<usize> {
    cache_capacity(ENV_TWEET_CACHE_CAPACITY)
}

pub(crate) fn safety_label_cache_capacity() -> Option<usize> {
    cache_capacity(ENV_SAFETY_LABEL_CACHE_CAPACITY)
}

fn cache_capacity(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok();
    let capacity = parse_capacity(value.as_deref());
    if let Some(value) = value
        && value.trim().parse::<usize>().is_err()
    {
        tracing::warn!(
            name,
            value,
            ?capacity,
            "unparsable cache capacity; keeping the default"
        );
    }
    capacity
}

fn parse_capacity(value: Option<&str>) -> Option<usize> {
    let capacity = value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_CACHE_CAPACITY);
    (capacity > 0).then_some(capacity)
}

pub(crate) fn cache_warm_enabled() -> bool {
    parse_env_flag(std::env::var(ENV_CACHE_WARM_ENABLED).ok().as_deref())
}

pub(crate) fn parse_env_flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_CACHE_CAPACITY, parse_capacity, parse_env_flag, resolve_gizmoduck_client_id,
        resolve_twemcache_client_name,
    };

    #[test]
    fn zero_turns_a_cache_off_and_an_unparsable_capacity_keeps_the_default() {
        for (value, expected) in [
            (None, Some(DEFAULT_CACHE_CAPACITY)),
            (Some(""), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("-5"), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("-1"), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("2M"), Some(DEFAULT_CACHE_CAPACITY)),
            (Some("0"), None),
            (Some(" 0 "), None),
            (Some(" 2000000 "), Some(2_000_000)),
        ] {
            assert_eq!(parse_capacity(value), expected, "{value:?}");
        }
    }

    #[test]
    fn client_ids_default_to_historical_values_and_overrides_win() {
        assert_eq!(
            resolve_gizmoduck_client_id(None, Some("prod")),
            "visibility-filtering-service.prod"
        );
        assert_eq!(
            resolve_gizmoduck_client_id(None, Some("staging")),
            "visibility-filtering-service.staging"
        );
        assert_eq!(
            resolve_gizmoduck_client_id(None, None),
            "visibility-filtering-service.prod"
        );
        assert_eq!(
            resolve_twemcache_client_name(None),
            "visibility-filtering-service"
        );
    }

    #[test]
    fn environment_flags_accept_only_enabled_values() {
        for (value, expected) in [
            (None, false),
            (Some(""), false),
            (Some("0"), false),
            (Some("false"), false),
            (Some("FALSE"), false),
            (Some("no"), false),
            (Some("off"), false),
            (Some("other"), false),
            (Some("1"), true),
            (Some("true"), true),
            (Some("TRUE"), true),
            (Some("yes"), true),
            (Some("on"), true),
        ] {
            assert_eq!(parse_env_flag(value), expected, "{value:?}");
        }
    }
}
