mod client_switches;
mod limited_actions_policy;

use crate::limited_actions_copy::DriftCheck;
use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use strum::VariantArray;
use xai_feature_switches::{Feature, FeatureSwitches, RecipientBuilder, Value};
use xai_stats_receiver::StatsReceiverExt;

pub(crate) use client_switches::ClientSwitches;
pub(crate) use limited_actions_policy::{LimitedActionType, LimitedActionsPolicies};

const SCALA_FILES: [&str; 6] = [
    "age_verification.yml",
    "community_tweet.yml",
    "country_specific_nsfw_content_gating.yml",
    "freedom_of_speech_not_reach.yml",
    "media_visibility_treatments.yml",
    "stale_tweet.yml",
];

const LIMITED_ACTIONS_POLICY_FILE: &str =
    "../../visibility-limited-actions/main/limited_actions_policy.yml";

const LOAD_FAILURE_COUNTER: &str = "feature_switch_load_failures";

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, strum::VariantArray,
)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CountryList {
    NsfwGating,
    AgeVerification,
    Tombstone,
    LocalRegulations,
}

impl CountryList {
    const fn key(self) -> &'static str {
        match self {
            Self::NsfwGating => "country_specific_nsfw_content_gating_countries",
            Self::AgeVerification => {
                "country_specific_nsfw_content_gating_age_verification_countries"
            }
            Self::Tombstone => "country_specific_nsfw_content_gating_tombstone_countries",
            Self::LocalRegulations => {
                "country_specific_nsfw_content_gating_local_regulations_countries"
            }
        }
    }

    const fn default_codes(self) -> &'static [&'static str] {
        const AGE_VERIFICATION: &[&str] = &[
            "at", "au", "be", "bg", "br", "cy", "cz", "de", "dk", "ee", "es", "fi", "fr", "gb",
            "gr", "hr", "hu", "ie", "it", "lt", "lu", "lv", "mt", "nl", "pl", "pt", "ro", "se",
            "si", "sk",
        ];
        match self {
            Self::NsfwGating => &[
                "ar", "au", "br", "ca", "de", "es", "fr", "gb", "id", "it", "kr", "mx", "nl", "ph",
                "pt", "th",
            ],
            Self::AgeVerification | Self::Tombstone => AGE_VERIFICATION,
            Self::LocalRegulations => &["in"],
        }
    }
}

pub(crate) struct CountryLists {
    lists: ArcSwap<Vec<Vec<String>>>,
}

impl CountryLists {
    pub fn starting_at_default() -> Self {
        let lists = CountryList::VARIANTS
            .iter()
            .map(|list| owned(list.default_codes()))
            .collect();
        Self {
            lists: ArcSwap::from_pointee(lists),
        }
    }

    pub fn contains(&self, list: CountryList, country_code: &str) -> bool {
        #[expect(
            clippy::indexing_slicing,
            reason = "`lists` maps `CountryList::VARIANTS`, which is in declaration order"
        )]
        let codes = &self.lists.load()[list as usize];
        codes.iter().any(|c| c == country_code)
    }

    pub(crate) fn codes(&self) -> BTreeMap<CountryList, Vec<String>> {
        CountryList::VARIANTS
            .iter()
            .copied()
            .zip(self.lists.load().iter().cloned())
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn from_codes(codes: &BTreeMap<CountryList, Vec<String>>) -> anyhow::Result<Self> {
        let lists = CountryList::VARIANTS
            .iter()
            .map(|list| {
                codes
                    .get(list)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("country_lists lacks {list:?}"))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            lists: ArcSwap::from_pointee(lists),
        })
    }

    pub fn refresh(&self, feature_switches: &FeatureSwitches) {
        let results = feature_switches.match_recipient(&RecipientBuilder::new().build());
        let lists = CountryList::VARIANTS
            .iter()
            .map(|&list| load(list, results.get_array_no_impression(list.key())))
            .collect();
        self.lists.store(Arc::new(lists));
    }
}

pub(crate) struct SwitchFiles {
    files: Vec<(PathBuf, Vec<Feature>)>,
}

impl SwitchFiles {
    pub fn beside(fs_path: &str) -> Self {
        let fs_path = Path::new(fs_path);
        let dir = fs_path.parent().unwrap_or_else(|| Path::new(""));
        let files = std::iter::once(fs_path.to_path_buf())
            .chain(SCALA_FILES.iter().map(|name| dir.join(name)))
            .chain(std::iter::once(dir.join(LIMITED_ACTIONS_POLICY_FILE)))
            .map(|path| (path, Vec::new()))
            .collect();
        Self { files }
    }

    pub fn load(&mut self, stats: Option<&dyn StatsReceiverExt>) -> Option<FeatureSwitches> {
        for (path, last_good) in &mut self.files {
            match parse_file(path) {
                Ok(features) => *last_good = features,
                Err(error) => {
                    let file = path.file_name().and_then(|name| name.to_str());
                    load_failed(file.unwrap_or_default(), &error, stats);
                }
            }
        }
        let features = self
            .files
            .iter()
            .flat_map(|(_, features)| features.iter().cloned())
            .collect();
        FeatureSwitches::new(features)
            .map_err(|error| load_failed("all", &error, stats))
            .ok()
    }
}

fn parse_file(path: &Path) -> xai_feature_switches::Result<Vec<Feature>> {
    let features = xai_feature_switches::load_yaml_file(path)?;
    FeatureSwitches::new(features.clone())?;
    Ok(features)
}

fn load_failed(
    file: &str,
    error: &xai_feature_switches::Error,
    stats: Option<&dyn StatsReceiverExt>,
) {
    tracing::warn!(
        file,
        %error,
        "feature switches: file failed to load; keeping its last good values"
    );
    if let Some(stats) = stats {
        stats.incr(LOAD_FAILURE_COUNTER, &[("file", file)], 1);
    }
}

pub(crate) fn spawn_refresh(
    mut files: SwitchFiles,
    feature_switches: Arc<ArcSwap<FeatureSwitches>>,
    country_lists: Arc<CountryLists>,
    copy_drift: DriftCheck,
    stats: Option<Arc<dyn StatsReceiverExt>>,
) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            if let Some(engine) = files.load(stats.as_deref()) {
                feature_switches.store(Arc::new(engine));
            }
            country_lists.refresh(&feature_switches.load());
            copy_drift.run(stats.as_deref());
        }
    });
}

fn owned(codes: &[&str]) -> Vec<String> {
    codes.iter().map(|code| code.to_string()).collect()
}

fn parse(values: &[Value]) -> Option<Vec<String>> {
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|code| code.len() == 2 && code.bytes().all(|b| b.is_ascii_alphabetic()))
                .map(str::to_ascii_lowercase)
        })
        .collect()
}

fn load(list: CountryList, configured: Option<&Vec<Value>>) -> Vec<String> {
    let Some(values) = configured else {
        return owned(list.default_codes());
    };
    parse(values).unwrap_or_else(|| {
        tracing::warn!(
            key = list.key(),
            configured = ?values,
            "country list: an entry is not a two-letter code; using the default"
        );
        owned(list.default_codes())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use xai_stats_receiver::HistogramBuckets;

    #[derive(Default)]
    pub(super) struct Counted(Mutex<Vec<(String, String, u64)>>);

    impl StatsReceiverExt for Counted {
        fn incr(&self, name: &str, scopes: &[(&str, &str)], value: u64) {
            let labels = scopes
                .iter()
                .map(|(label, tag)| format!("{label}={tag}"))
                .collect::<Vec<_>>()
                .join(",");
            self.0
                .lock()
                .unwrap()
                .push((name.to_string(), labels, value));
        }
        fn observe(&self, _: &str, _: &[(&str, &str)], _: f64, _: HistogramBuckets) {}
        fn observe_expo(&self, _: &str, _: &[(&str, &str)], _: f64) {}
        fn observe_vm(&self, _: &str, _: &[(&str, &str)], _: f64) {}
        fn gauge(&self, _: &str, _: &[(&str, &str)], _: f64) {}
    }

    impl Counted {
        pub(super) fn take(&self) -> Vec<(String, String, u64)> {
            std::mem::take(&mut self.0.lock().unwrap())
        }
    }

    fn engine(yaml: &str) -> FeatureSwitches {
        FeatureSwitches::load_string(yaml).unwrap()
    }

    #[test]
    fn refresh_reads_each_key_and_falls_back_to_its_default() {
        let lists = CountryLists::starting_at_default();
        assert!(lists.contains(CountryList::NsfwGating, "de"));
        assert!(lists.contains(CountryList::LocalRegulations, "in"));
        assert!(!lists.contains(CountryList::NsfwGating, "xx"));

        lists.refresh(&engine(
            r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_countries:
      type: array
      default:
      - "XX"
    country_specific_nsfw_content_gating_local_regulations_countries:
      type: array
      default: []
    country_specific_nsfw_content_gating_tombstone_countries:
      type: string
      default: "xx"
"#,
        ));
        assert!(lists.contains(CountryList::NsfwGating, "xx"));
        assert!(!lists.contains(CountryList::NsfwGating, "de"));
        assert!(lists.contains(CountryList::AgeVerification, "fr"));
        assert!(!lists.contains(CountryList::LocalRegulations, "in"));
        assert!(lists.contains(CountryList::Tombstone, "fr"));
        assert!(!lists.contains(CountryList::Tombstone, "xx"));

        lists.refresh(&engine("other:\n  parameters: {}\n"));
        assert!(lists.contains(CountryList::NsfwGating, "de"));
        assert!(!lists.contains(CountryList::NsfwGating, "xx"));
        assert!(lists.contains(CountryList::LocalRegulations, "in"));
    }

    #[test]
    fn one_invalid_entry_falls_back_to_the_default() {
        let lists = CountryLists::starting_at_default();
        lists.refresh(&engine(
            r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_age_verification_countries:
      type: array
      default: ["xx", " fr"]
    country_specific_nsfw_content_gating_tombstone_countries:
      type: array
      default: ["xx", "fra"]
    country_specific_nsfw_content_gating_local_regulations_countries:
      type: array
      default: ["xx", 7]
"#,
        ));
        for (list, default) in [
            (CountryList::AgeVerification, "fr"),
            (CountryList::Tombstone, "fr"),
            (CountryList::LocalRegulations, "in"),
        ] {
            assert!(lists.contains(list, default), "{list:?}");
            assert!(!lists.contains(list, "xx"), "{list:?}");
        }
    }

    #[test]
    fn a_file_that_fails_to_load_keeps_its_last_good_values() {
        const AGE_VERIFICATION: &str = r#"
age_verification:
  parameters:
    age_verification_ios_tombstone_rule_enabled: {type: boolean, default: true}
"#;
        const COUNTRIES: &str = r#"
country_specific_nsfw_content_gating:
  parameters:
    country_specific_nsfw_content_gating_local_regulations_countries:
      type: array
      default: ["us"]
"#;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("visibility/main");
        let policy_dir = root.path().join("visibility-limited-actions/main");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&policy_dir).unwrap();
        let write = |name: &str, yaml: &str| std::fs::write(dir.join(name), yaml).unwrap();
        write("age_verification.yml", AGE_VERIFICATION);
        write("country_specific_nsfw_content_gating.yml", COUNTRIES);
        write(
            "other.yml",
            "other:\n  parameters:\n    other_flag: {type: boolean, default: true}\n",
        );
        let fs_path = dir.join("rust_vf.yml");
        let mut files = SwitchFiles::beside(fs_path.to_str().unwrap());
        let stats = Counted::default();
        let ios_tombstone = |engine: &FeatureSwitches| {
            engine
                .match_recipient(&RecipientBuilder::new().build())
                .get_bool_no_impression("age_verification_ios_tombstone_rule_enabled")
        };
        let failed = |file: &str| (LOAD_FAILURE_COUNTER.to_string(), format!("file={file}"), 1);

        let engine = files.load(Some(&stats)).unwrap();
        assert_eq!(ios_tombstone(&engine), Some(true));
        assert!(!engine
            .match_recipient(&RecipientBuilder::new().build())
            .is_set("other_flag"));
        assert_eq!(
            stats.take(),
            [
                failed("rust_vf.yml"),
                failed("community_tweet.yml"),
                failed("freedom_of_speech_not_reach.yml"),
                failed("media_visibility_treatments.yml"),
                failed("stale_tweet.yml"),
                failed("limited_actions_policy.yml"),
            ]
        );

        write("rust_vf.yml", "rust_vf:\n  parameters: {}\n");
        write(
            "media_visibility_treatments.yml",
            "media_visibility_treatments:\n  parameters: {}\n",
        );
        write("stale_tweet.yml", "stale_tweet:\n  parameters: {}\n");
        write(
            "community_tweet.yml",
            "community_tweet:\n  parameters: {}\n",
        );
        write(
            "freedom_of_speech_not_reach.yml",
            "freedom_of_speech_not_reach:\n  parameters: {}\n",
        );
        std::fs::write(
            policy_dir.join("limited_actions_policy.yml"),
            "limited_actions_policy:\n  parameters: {}\n",
        )
        .unwrap();
        for (broken, code) in [
            (
                "age_verification:\n  parameters: {}\n  rules:\n  - query: \"([user_id in 1]\"\n    values: {}\n",
                "xx",
            ),
            ("age_verification: [", "yy"),
        ] {
            write("age_verification.yml", broken);
            write(
                "country_specific_nsfw_content_gating.yml",
                &COUNTRIES.replace("us", code),
            );
            let engine = files.load(Some(&stats)).unwrap();
            assert_eq!(ios_tombstone(&engine), Some(true), "{broken}");
            let lists = CountryLists::starting_at_default();
            lists.refresh(&engine);
            assert!(
                lists.contains(CountryList::LocalRegulations, code),
                "{broken}"
            );
            assert_eq!(stats.take(), [failed("age_verification.yml")], "{broken}");
        }
    }
}
