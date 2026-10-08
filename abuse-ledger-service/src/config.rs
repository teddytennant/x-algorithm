// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::time::Duration;

use clap::Parser;

#[derive(Clone)]
pub struct SecretString(String);

impl SecretString {
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SecretString(***)")
    }
}

impl std::str::FromStr for SecretString {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(s.to_owned()))
    }
}

pub const MIN_STATEMENT_TIMEOUT_MS: u64 = 10;
pub const MAX_STATEMENT_TIMEOUT_MS: u64 = 5_000;
pub const MAX_POOL: usize = 32;

#[derive(Debug, Parser, Clone)]
#[command(name = "xai-abuse-ledger-service")]
pub struct Config {
    #[arg(long, env = "DATABASE_URL", hide_env_values = true)]
    pub database_url: SecretString,

    #[arg(long, env = "LEDGER_ENV", default_value = "unknown")]
    pub ledger_env: String,

    #[arg(long, env = "STATEMENT_TIMEOUT_MS", default_value_t = 150)]
    pub statement_timeout_ms: u64,

    #[arg(long, env = "POOL_MAX", default_value_t = 8)]
    pub pool_max: usize,

    #[arg(long, env = "PORT", default_value_t = 8080)]
    pub port: u16,

    #[arg(long, env = "XAI_COMMIT", default_value = "unknown")]
    pub commit: String,
}

impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            (MIN_STATEMENT_TIMEOUT_MS..=MAX_STATEMENT_TIMEOUT_MS)
                .contains(&self.statement_timeout_ms),
            "STATEMENT_TIMEOUT_MS={} must be within {MIN_STATEMENT_TIMEOUT_MS}..={MAX_STATEMENT_TIMEOUT_MS}",
            self.statement_timeout_ms
        );
        anyhow::ensure!(
            (1..=MAX_POOL).contains(&self.pool_max),
            "POOL_MAX={} must be within 1..={MAX_POOL}",
            self.pool_max
        );
        anyhow::ensure!(
            !self.database_url.expose().trim().is_empty(),
            "DATABASE_URL is empty"
        );
        Ok(())
    }

    pub fn statement_timeout(&self) -> Duration {
        Duration::from_millis(self.statement_timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Config {
        Config::try_parse_from(
            std::iter::once("xai-abuse-ledger-service").chain(args.iter().copied()),
        )
        .expect("parses")
    }

    #[test]
    fn defaults_match_the_plan() {
        let c = parse(&["--database-url", "postgresql://u:p@ledger-rw/l"]);
        assert_eq!(c.statement_timeout_ms, 150);
        assert_eq!(c.pool_max, 8);
        assert_eq!(c.port, 8080);
        assert_eq!(c.ledger_env, "unknown");
        assert_eq!(c.commit, "unknown");
        c.validate().unwrap();
        assert_eq!(c.statement_timeout(), Duration::from_millis(150));
    }

    #[test]
    fn dsn_is_required_and_never_debug_printed() {
        assert!(
            Config::try_parse_from(["xai-abuse-ledger-service"]).is_err(),
            "no DATABASE_URL must fail to parse"
        );
        let c = parse(&["--database-url", "postgresql://u:secret@ledger-rw/l"]);
        let dbg = format!("{c:?}");
        assert!(dbg.contains("SecretString(***)"));
        assert!(!dbg.contains("secret@"));
    }

    #[test]
    fn out_of_range_values_are_refused() {
        let c = parse(&[
            "--database-url",
            "postgresql://u:p@ledger-rw/l",
            "--statement-timeout-ms",
            "0",
        ]);
        assert!(c.validate().is_err());
        let c = parse(&[
            "--database-url",
            "postgresql://u:p@ledger-rw/l",
            "--statement-timeout-ms",
            "6000",
        ]);
        assert!(c.validate().is_err());
        let c = parse(&[
            "--database-url",
            "postgresql://u:p@ledger-rw/l",
            "--pool-max",
            "0",
        ]);
        assert!(c.validate().is_err());
        let c = parse(&[
            "--database-url",
            "postgresql://u:p@ledger-rw/l",
            "--pool-max",
            "33",
        ]);
        assert!(c.validate().is_err());
        let c = parse(&["--database-url", "  "]);
        assert!(c.validate().is_err());
    }
}
