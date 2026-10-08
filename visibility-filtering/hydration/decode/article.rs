use crate::models::ArticleLifecycle;
use anyhow::bail;

pub(crate) fn decode_lifecycle(value: i32) -> anyhow::Result<ArticleLifecycle> {
    match value {
        1 => Ok(ArticleLifecycle::Published),
        2 => Ok(ArticleLifecycle::SoftDeleted),
        3 => Ok(ArticleLifecycle::Draft),
        other => bail!("ArticleLifecycleState.lifecycle {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_each_lifecycle_and_rejects_any_other() {
        assert_eq!(decode_lifecycle(1).unwrap(), ArticleLifecycle::Published);
        assert_eq!(decode_lifecycle(2).unwrap(), ArticleLifecycle::SoftDeleted);
        assert_eq!(decode_lifecycle(3).unwrap(), ArticleLifecycle::Draft);
        assert!(decode_lifecycle(4).is_err());
    }
}
