use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use xai_feature_switches::{
    AuthorRulesEvaluator, FeatureSwitches, Params, RecipientBuilder, SimpleRecipient,
};
use xai_vm_ranker_proto::ViewerContext;

const TWEPOCH_MS: i64 = 1_288_834_974_657;
const FIRST_SNOWFLAKE_ID: u64 = 362_387_865_600_000;
const SNOWFLAKE_TIMESTAMP_SHIFT: u32 = 22;
const MS_PER_DAY: i64 = 86_400_000;
const MS_PER_MINUTE: i64 = 60_000;

pub struct RankingConfig {
    feature_switches: Arc<FeatureSwitches>,
    author_rules: AuthorRulesEvaluator,
}

impl RankingConfig {
    pub async fn load(
        features_path: &Path,
        abdecider_path: &Path,
        process_overrides: Vec<(String, String)>,
        impressions: Option<&str>,
    ) -> Result<Self> {
        let mut builder = FeatureSwitches::builder()
            .load_file(features_path)
            .with_context(|| format!("loading feature switches from {}", features_path.display()))?
            .abdecider_path(abdecider_path)
            .fs_overrides(process_overrides);
        if let Some(datacenter) = impressions {
            builder = builder.kafka_impressor_enabled(true).datacenter(datacenter);
        }
        let feature_switches = builder.build().await.context("building feature switches")?;
        Ok(Self::new(feature_switches))
    }

    pub fn from_yaml(yaml: &str) -> Result<Self> {
        Ok(Self::new(Arc::new(FeatureSwitches::load_string(yaml)?)))
    }

    pub fn new(feature_switches: Arc<FeatureSwitches>) -> Self {
        Self {
            author_rules: AuthorRulesEvaluator::new(Arc::clone(&feature_switches)),
            feature_switches,
        }
    }

    pub fn author_rules(&self) -> &AuthorRulesEvaluator {
        &self.author_rules
    }

    pub fn resolve(&self, viewer: &ViewerContext) -> Params {
        let mut results = self.feature_switches.match_recipient(&recipient(viewer));
        for (key, value) in &viewer.fs_overrides {
            results.override_fs(key.clone(), value);
        }
        results.into()
    }

    pub fn resolve_anonymous(&self) -> Params {
        self.feature_switches
            .match_recipient(&SimpleRecipient::default())
            .into()
    }
}

pub fn recipient(viewer: &ViewerContext) -> SimpleRecipient {
    let account_age_ms =
        snowflake_creation_ms(viewer.user_id).map(|created| viewer.now_ms - created);
    let account_age_days = account_age_ms
        .filter(|&age| age >= 0)
        .map_or(i64::MAX, |age| age / MS_PER_DAY);
    let account_creation_date =
        snowflake_creation_ms(viewer.user_id).map_or(0, |created| created / MS_PER_DAY);
    let account_age_minutes = account_age_ms
        .filter(|&age| age >= 0)
        .map(|age| age / MS_PER_MINUTE);
    let resurrection = viewer.resurrection_time_ms.filter(|&t| t >= 0);
    let user_resurrected_date = resurrection.map(|t| t / MS_PER_DAY);
    let days_since_resurrection = resurrection.map(|t| (viewer.now_ms - t) / MS_PER_DAY);
    let minutes_since_resurrection = resurrection.map(|t| (viewer.now_ms - t) / MS_PER_MINUTE);

    RecipientBuilder::new()
        .user_id(viewer.user_id)
        .country(&viewer.country_code)
        .language(&viewer.language_code)
        .client_app_id(viewer.client_app_id)
        .client_version_opt(viewer.client_version.as_deref())
        .user_roles(viewer.user_roles.iter().cloned())
        .custom_string("datacenter", &viewer.datacenter)
        .custom_i64("account_age_days", account_age_days)
        .custom_i64("account_creation_date", account_creation_date)
        .custom_bool("has_phone_number", viewer.has_phone_number)
        .custom_string("product", &viewer.product)
        .custom_opt_i64("user_resurrected_date", user_resurrected_date)
        .custom_opt_i64("days_since_resurrection", days_since_resurrection)
        .custom_opt_i64("account_age_minutes", account_age_minutes)
        .custom_opt_i64("minutes_since_resurrection", minutes_since_resurrection)
        .build()
}

pub fn snowflake_creation_ms(id: u64) -> Option<i64> {
    (id >= FIRST_SNOWFLAKE_ID).then(|| (id >> SNOWFLAKE_TIMESTAMP_SHIFT) as i64 + TWEPOCH_MS)
}
