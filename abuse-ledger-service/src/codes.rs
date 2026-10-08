// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    BadRequest,
    ConnectRefused,
    AuthFailed,
    Tls,
    StatementTimeout,
    ServerUnhealthy,
    PoolWait,
    Timeout,
    Decode,
    Other,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::BadRequest => "bad_request",
            ErrorCode::ConnectRefused => "connect_refused",
            ErrorCode::AuthFailed => "auth_failed",
            ErrorCode::Tls => "tls",
            ErrorCode::StatementTimeout => "statement_timeout",
            ErrorCode::ServerUnhealthy => "server_unhealthy",
            ErrorCode::PoolWait => "pool_wait",
            ErrorCode::Timeout => "timeout",
            ErrorCode::Decode => "decode",
            ErrorCode::Other => "other",
        }
    }

    pub fn status(self) -> StatusCode {
        match self {
            ErrorCode::BadRequest => StatusCode::BAD_REQUEST,
            ErrorCode::ConnectRefused
            | ErrorCode::AuthFailed
            | ErrorCode::Tls
            | ErrorCode::ServerUnhealthy
            | ErrorCode::PoolWait => StatusCode::SERVICE_UNAVAILABLE,
            ErrorCode::StatementTimeout | ErrorCode::Timeout => StatusCode::GATEWAY_TIMEOUT,
            ErrorCode::Decode | ErrorCode::Other => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn retryable(self) -> bool {
        match self {
            ErrorCode::BadRequest | ErrorCode::Decode | ErrorCode::Other => false,
            ErrorCode::ConnectRefused
            | ErrorCode::AuthFailed
            | ErrorCode::Tls
            | ErrorCode::StatementTimeout
            | ErrorCode::ServerUnhealthy
            | ErrorCode::PoolWait
            | ErrorCode::Timeout => true,
        }
    }

    pub const ALL: [ErrorCode; 10] = [
        ErrorCode::BadRequest,
        ErrorCode::ConnectRefused,
        ErrorCode::AuthFailed,
        ErrorCode::Tls,
        ErrorCode::StatementTimeout,
        ErrorCode::ServerUnhealthy,
        ErrorCode::PoolWait,
        ErrorCode::Timeout,
        ErrorCode::Decode,
        ErrorCode::Other,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: ErrorCode,
    pub retryable: bool,
}

impl ErrorBody {
    pub fn new(code: ErrorCode) -> Self {
        Self {
            code,
            retryable: code.retryable(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_closed_and_stable() {
        let expected = [
            (ErrorCode::BadRequest, "bad_request", 400, false),
            (ErrorCode::ConnectRefused, "connect_refused", 503, true),
            (ErrorCode::AuthFailed, "auth_failed", 503, true),
            (ErrorCode::Tls, "tls", 503, true),
            (ErrorCode::StatementTimeout, "statement_timeout", 504, true),
            (ErrorCode::ServerUnhealthy, "server_unhealthy", 503, true),
            (ErrorCode::PoolWait, "pool_wait", 503, true),
            (ErrorCode::Timeout, "timeout", 504, true),
            (ErrorCode::Decode, "decode", 500, false),
            (ErrorCode::Other, "other", 500, false),
        ];
        assert_eq!(expected.len(), ErrorCode::ALL.len());
        for (code, s, status, retryable) in expected {
            assert!(ErrorCode::ALL.contains(&code));
            assert_eq!(code.as_str(), s);
            assert_eq!(code.status().as_u16(), status, "{s}");
            assert_eq!(code.retryable(), retryable, "{s}");
            let json = serde_json::to_string(&code).unwrap();
            assert_eq!(json, format!("\"{s}\""));
            let back: ErrorCode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, code);
        }
    }

    #[test]
    fn error_body_has_exactly_two_fields() {
        let body = ErrorBody::new(ErrorCode::StatementTimeout);
        let v: serde_json::Value = serde_json::to_value(&body).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 2);
        assert_eq!(obj["code"], "statement_timeout");
        assert_eq!(obj["retryable"], true);
    }
}
