// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
pub const PROBE_SQL: &str =
    "SELECT user_id, hold_id, head, expires_at, case_group_id, reason, action_kind, label \
     FROM ledger.enforcement_holds \
     WHERE user_id = ANY($1::bigint[]) AND cleared_at IS NULL AND expires_at > now() \
     AND (action_kind = 'suspend' OR (action_kind = 'label' AND label = ANY($2::text[])))";

pub const CONTRACT_SQL: &str =
    "SELECT user_id, hold_id, head, expires_at, case_group_id, reason, action_kind, label \
     FROM ledger.enforcement_holds LIMIT 0";

pub const PING_SQL: &str = "SELECT 1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_sql_is_the_ledger_read_side() {
        assert_eq!(
            PROBE_SQL,
            "SELECT user_id, hold_id, head, expires_at, case_group_id, reason, action_kind, label \
             FROM ledger.enforcement_holds \
             WHERE user_id = ANY($1::bigint[]) AND cleared_at IS NULL AND expires_at > now() \
             AND (action_kind = 'suspend' OR (action_kind = 'label' AND label = ANY($2::text[])))"
        );
        assert!(PROBE_SQL.contains("FROM ledger.enforcement_holds"));
        assert!(PROBE_SQL.contains("user_id = ANY($1::bigint[])"));
        assert!(PROBE_SQL.contains("action_kind = 'suspend'"));
        assert!(PROBE_SQL.contains("action_kind = 'label' AND label = ANY($2::text[])"));
        assert!(PROBE_SQL.contains("cleared_at IS NULL"));
        assert!(PROBE_SQL.contains("expires_at > now()"));
        assert_eq!(PING_SQL, "SELECT 1");
    }

    #[test]
    fn contract_sql_projects_the_probe_columns() {
        let probe_cols = PROBE_SQL
            .strip_prefix("SELECT ")
            .unwrap()
            .split(" FROM ")
            .next()
            .unwrap();
        let contract_cols = CONTRACT_SQL
            .strip_prefix("SELECT ")
            .unwrap()
            .split(" FROM ")
            .next()
            .unwrap();
        assert_eq!(probe_cols, contract_cols);
        assert!(CONTRACT_SQL.ends_with("FROM ledger.enforcement_holds LIMIT 0"));
        for sql in [PROBE_SQL, CONTRACT_SQL, PING_SQL] {
            assert!(!sql.contains("information_schema"));
            assert!(!sql.contains("NULL::text"));
            let upper = sql.to_ascii_uppercase();
            for verb in ["INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "CREATE"] {
                assert!(!upper.contains(verb), "{sql}");
            }
        }
    }
}
