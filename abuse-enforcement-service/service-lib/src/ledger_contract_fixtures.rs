pub(crate) const LOOKUP_REQUEST: &str =
    r#"{"user_ids": [1234567890], "labels": ["SpamHighRecall"]}"#;

pub(crate) const LOOKUP_RESPONSE_200: &str = r#"{
  "holds": [
    {
      "user_id": 1234567890,
      "hold_id": 88123,
      "action_kind": "suspend",
      "label": null,
      "head": "bb1_reply_abuse_any_v15",
      "expires_at": "2026-12-20T17:03:11Z",
      "case_group_id": 64,
      "reason": "appeal_overturned"
    },
    {
      "user_id": 1234567890,
      "hold_id": 88124,
      "action_kind": "label",
      "label": "SpamHighRecall",
      "head": null,
      "expires_at": "2026-10-01T00:00:00Z",
      "case_group_id": null,
      "reason": "manual"
    }
  ],
  "db_ms": 1
}"#;

pub(crate) const ERROR_CONNECT_REFUSED: &str = r#"{"code": "connect_refused", "retryable": true}"#;
pub(crate) const ERROR_AUTH_FAILED: &str = r#"{"code": "auth_failed", "retryable": true}"#;
pub(crate) const ERROR_TLS: &str = r#"{"code": "tls", "retryable": true}"#;
pub(crate) const ERROR_STATEMENT_TIMEOUT: &str =
    r#"{"code": "statement_timeout", "retryable": true}"#;
pub(crate) const ERROR_SERVER_UNHEALTHY: &str =
    r#"{"code": "server_unhealthy", "retryable": true}"#;
pub(crate) const ERROR_POOL_WAIT: &str = r#"{"code": "pool_wait", "retryable": true}"#;
pub(crate) const ERROR_OTHER: &str = r#"{"code": "other", "retryable": false}"#;
pub(crate) const ERROR_DECODE: &str = r#"{"code": "decode", "retryable": false}"#;
pub(crate) const ERROR_BAD_REQUEST: &str = r#"{"code": "bad_request", "retryable": false}"#;
