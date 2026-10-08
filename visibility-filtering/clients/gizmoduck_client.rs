use anyhow::anyhow;
use std::collections::HashMap;
use std::sync::Arc;
use xai_core_entities::entities::{GizmoduckUser, GizmoduckUserResult};
use xai_core_entities::gizmoduck_client::{GizmoduckClient, LookupContext, QueryFields};

pub struct GizmoduckLookup {
    inner: Arc<dyn GizmoduckClient + Send + Sync>,
}

fn author_hydration_lookup_context() -> LookupContext {
    LookupContext {
        for_user_id: None,
        include_deactivated: true,
        include_failed: true,
        include_erased: true,
        include_no_screen_name_users: true,
        include_offboarded: true,
        ..Default::default()
    }
}

impl GizmoduckLookup {
    pub fn new(inner: Arc<dyn GizmoduckClient + Send + Sync>) -> Self {
        Self { inner }
    }

    pub async fn get_users(
        &self,
        user_ids: Vec<u64>,
        fields: &[QueryFields],
    ) -> HashMap<u64, anyhow::Result<Option<GizmoduckUserResult>>> {
        let ids: Vec<i64> = user_ids.into_iter().map(u64::cast_signed).collect();
        self.inner
            .get_users_with_context(ids, Some(author_hydration_lookup_context()), fields)
            .await
            .into_iter()
            .map(|(id, result)| (id.cast_unsigned(), result))
            .collect()
    }

    pub async fn get_viewer(
        &self,
        viewer_id: u64,
        fields: &[QueryFields],
    ) -> anyhow::Result<Option<GizmoduckUser>> {
        let result = self
            .inner
            .get_users_with_context(vec![viewer_id.cast_signed()], None, fields)
            .await
            .into_values()
            .next()
            .ok_or_else(|| anyhow!("Empty batch response from gizmoduck"))??;
        Ok(result.and_then(|result| result.user))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tonic::async_trait;
    use xai_core_entities::entities::PCFLabel;
    use xai_core_entities::gizmoduck_client::{UserFields, ViewerData};

    type Users = HashMap<i64, anyhow::Result<Option<GizmoduckUserResult>>>;

    #[derive(Default)]
    struct Recording(Mutex<Vec<Option<LookupContext>>>);

    #[async_trait]
    impl GizmoduckClient for Recording {
        async fn get_users_with_context(
            &self,
            _: Vec<i64>,
            context: Option<LookupContext>,
            _: &[QueryFields],
        ) -> Users {
            self.0.lock().unwrap().push(context);
            HashMap::new()
        }

        async fn get_users(&self, _: Vec<i64>) -> Users {
            unreachable!()
        }

        async fn get_users_with_perspective(&self, _: i64, _: Vec<i64>) -> Users {
            unreachable!()
        }

        async fn get_viewer_roles(&self, _: u64) -> anyhow::Result<Vec<String>> {
            unreachable!()
        }

        async fn get_viewer_data(&self, _: u64) -> anyhow::Result<ViewerData> {
            unreachable!()
        }

        async fn get_viewer_data_with_fields(
            &self,
            _: u64,
            _: &[QueryFields],
        ) -> anyhow::Result<ViewerData> {
            unreachable!()
        }

        async fn get_pcf_labels(&self, _: Vec<i64>) -> HashMap<i64, anyhow::Result<PCFLabel>> {
            unreachable!()
        }

        async fn get_profile_description_languages(
            &self,
            _: Vec<i64>,
        ) -> HashMap<i64, anyhow::Result<Option<String>>> {
            unreachable!()
        }

        async fn get_user_fields(&self, _: Vec<i64>) -> HashMap<i64, anyhow::Result<UserFields>> {
            unreachable!()
        }

        async fn get_by_screen_name(&self, _: &str) -> anyhow::Result<Option<GizmoduckUserResult>> {
            unreachable!()
        }
    }

    #[tokio::test]
    async fn author_lookups_include_deactivated_erased_and_offboarded_users() {
        let client = Arc::new(Recording::default());
        GizmoduckLookup::new(Arc::<Recording>::clone(&client))
            .get_users(vec![10], &[QueryFields::SAFETY])
            .await;
        let contexts = client.0.lock().unwrap();
        let [Some(context)] = contexts.as_slice() else {
            panic!("one lookup with a context: {contexts:?}")
        };
        assert!(context.include_deactivated);
        assert!(context.include_erased);
        assert!(context.include_offboarded);
    }

    #[tokio::test]
    async fn viewer_lookups_pass_no_context() {
        let client = Arc::new(Recording::default());
        let viewer = GizmoduckLookup::new(Arc::<Recording>::clone(&client))
            .get_viewer(10, &[QueryFields::SAFETY])
            .await;
        assert!(viewer.is_err(), "an empty batch is an error");
        assert!(matches!(client.0.lock().unwrap().as_slice(), [None]));
    }
}
