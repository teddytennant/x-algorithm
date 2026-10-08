use crate::clients::author_brand_safety_client::AuthorBrandSafetyClient;
use crate::models::brand_safety::{
    botmaker_rule_category, botmaker_rule_id_from, compute_verdict, compute_verdict_v2,
    has_grok_label, truncate_description, with_author_fallback, worst_verdict,
    AuthorBrandSafetyFallback, BrandSafetyVerdict,
};
use crate::models::candidate::{CandidateHelpers, PostCandidate, SafetyLabelInfo};
use crate::models::query::ScoredPostsQuery;
use crate::params::{EnableAdsAuthorBrandSafetyFallback, EnableAdsBrandSafetyVerdictV2};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tonic::async_trait;
use xai_candidate_pipeline::hydrator::Hydrator;
use xai_safety_label_store::types::SafetyLabelMap;
use xai_stats_receiver::global_stats_receiver;
use xai_visibility_filtering::tweet_safety_label::TweetSafetyLabelClient;
use xai_x_thrift::tweet_safety_label::SafetyLabelType;

const NSFW_AUTHOR_METRIC: &str = "AdsBrandSafetyVf.nsfw_author";
const UNSCORED_AUTHOR_FALLBACK_METRIC: &str = "AdsBrandSafetyVf.unscored_author_fallback";

pub struct AdsBrandSafetyVfHydrator {
    pub client: Arc<dyn TweetSafetyLabelClient>,
    pub author_client: Option<Arc<dyn AuthorBrandSafetyClient>>,
}

impl AdsBrandSafetyVfHydrator {
    async fn author_fallbacks(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
        label_map: &HashMap<u64, SafetyLabelMap>,
        failed_ids: &HashSet<u64>,
    ) -> Option<HashMap<u64, Option<AuthorBrandSafetyFallback>>> {
        let author_client = self.author_client.as_ref()?;
        if !query.params.get(EnableAdsAuthorBrandSafetyFallback) {
            return None;
        }
        let authors: Vec<u64> = candidates
            .iter()
            .filter(|c| {
                let id = c.retweeted_tweet_id.unwrap_or(c.tweet_id);
                !failed_ids.contains(&id) && label_map.get(&id).is_none_or(|l| !has_grok_label(l))
            })
            .map(|c| c.get_original_author_id())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if authors.is_empty() {
            return Some(HashMap::new());
        }
        let fetched = author_client.fetch(&authors).await.unwrap_or_else(|e| {
            tracing::warn!("author fallback fetch failed: {e}");
            HashMap::new()
        });
        Some(fetched)
    }
}

fn author_fallback_for(
    fetched: &HashMap<u64, Option<AuthorBrandSafetyFallback>>,
    author_id: u64,
) -> (Option<AuthorBrandSafetyFallback>, &'static str) {
    match fetched.get(&author_id) {
        Some(Some(fallback)) => (Some(*fallback), "row"),
        Some(None) => (Some(AuthorBrandSafetyFallback::Nsfa), "no_row"),
        None => (Some(AuthorBrandSafetyFallback::Nsfa), "fetch_failed"),
    }
}

fn to_safety_label_infos(labels: &SafetyLabelMap) -> impl Iterator<Item = SafetyLabelInfo> {
    labels.iter().map(|(k, v)| SafetyLabelInfo {
        label_type: *k,
        description: v.source.as_deref().map(truncate_description),
        source: botmaker_rule_id_from(v).map(|id| botmaker_rule_category(id).to_string()),
    })
}

#[async_trait]
impl Hydrator<ScoredPostsQuery, PostCandidate> for AdsBrandSafetyVfHydrator {
    async fn hydrate(
        &self,
        query: &ScoredPostsQuery,
        candidates: &[PostCandidate],
    ) -> Vec<Result<PostCandidate, String>> {
        let mut all_ids: HashSet<u64> = HashSet::new();
        for c in candidates {
            all_ids.insert(c.retweeted_tweet_id.unwrap_or(c.tweet_id));
            if let Some(qt_id) = c.quoted_tweet_id {
                all_ids.insert(qt_id);
            }
            all_ids.extend(c.ancestors.iter().copied());
        }

        let tweet_ids: Vec<u64> = all_ids.into_iter().collect();
        let batch = match self.client.get_safety_labels(tweet_ids).await {
            Ok(batch) => batch,
            Err(e) => {
                let err = format!("VF get_safety_labels failed: {e}");
                return candidates.iter().map(|_| Err(err.clone())).collect();
            }
        };

        let failed_ids: HashSet<u64> = batch.failures.keys().copied().collect();
        let label_map = batch.labels;
        let author_fallbacks = self
            .author_fallbacks(query, candidates, &label_map, &failed_ids)
            .await;

        let v2 = query.params.get(EnableAdsBrandSafetyVerdictV2);
        let require_ptos_review = author_fallbacks.is_none();
        let compute = |labels: &SafetyLabelMap, tweet_id: u64| {
            if v2 {
                compute_verdict_v2(labels, tweet_id, require_ptos_review)
            } else {
                compute_verdict(labels, tweet_id, require_ptos_review)
            }
        };

        let mut nsfw_author_seen: u64 = 0;
        let mut nsfw_author_dropped: u64 = 0;
        let mut unscored_by_fallback: HashMap<(&'static str, &'static str, bool), u64> =
            HashMap::new();

        let results: Vec<Result<PostCandidate, String>> = candidates
            .iter()
            .map(|c| {
                let primary_id = c.retweeted_tweet_id.unwrap_or(c.tweet_id);

                if failed_ids.contains(&primary_id) {
                    return Err(format!("VF lookup failed for tweet {primary_id}"));
                }

                let empty = HashMap::new();
                let primary_labels = label_map.get(&primary_id).unwrap_or(&empty);
                let unscored = !has_grok_label(primary_labels);
                let (author_fallback, source) = match &author_fallbacks {
                    Some(fetched) if unscored => {
                        author_fallback_for(fetched, c.get_original_author_id())
                    }
                    _ => (None, "off"),
                };
                let mut verdict = match author_fallback {
                    Some(fallback) if unscored => {
                        compute(&with_author_fallback(primary_labels, fallback), primary_id)
                    }
                    _ => compute(primary_labels, primary_id),
                };
                if unscored {
                    let fallback = author_fallback.map_or("none", |f| f.as_str());
                    let ptos_reviewed =
                        primary_labels.contains_key(&SafetyLabelType::PTOS_REVIEWED);
                    *unscored_by_fallback
                        .entry((fallback, source, ptos_reviewed))
                        .or_default() += 1;
                }
                let mut safety_labels: Vec<SafetyLabelInfo> =
                    to_safety_label_infos(primary_labels).collect();

                if let Some(qt_id) = c.quoted_tweet_id {
                    if failed_ids.contains(&qt_id) {
                        verdict = worst_verdict(&verdict, &BrandSafetyVerdict::MediumRisk);
                    } else {
                        let qt_labels = label_map.get(&qt_id).unwrap_or(&empty);
                        verdict = worst_verdict(&verdict, &compute(qt_labels, qt_id));
                        safety_labels.extend(to_safety_label_infos(qt_labels));
                    }
                }

                for &ancestor_id in &c.ancestors {
                    if failed_ids.contains(&ancestor_id) {
                        verdict = worst_verdict(&verdict, &BrandSafetyVerdict::MediumRisk);
                    } else {
                        let ancestor_labels = label_map.get(&ancestor_id).unwrap_or(&empty);
                        verdict = worst_verdict(&verdict, &compute(ancestor_labels, ancestor_id));
                        safety_labels.extend(to_safety_label_infos(ancestor_labels));
                    }
                }

                if c.nsfw_author_ads == Some(true) {
                    nsfw_author_seen += 1;
                    let before = verdict;
                    verdict = worst_verdict(&verdict, &BrandSafetyVerdict::HighRisk);
                    if verdict != before {
                        nsfw_author_dropped += 1;
                    }
                }

                safety_labels.sort_unstable_by_key(|l| i32::from(l.label_type));
                safety_labels.dedup_by(|a, b| a.label_type == b.label_type);

                Ok(PostCandidate {
                    brand_safety_verdict: Some(verdict),
                    safety_labels,
                    ..Default::default()
                })
            })
            .collect();

        if let Some(receiver) = global_stats_receiver() {
            if nsfw_author_seen > 0 {
                receiver.incr(NSFW_AUTHOR_METRIC, &[("outcome", "seen")], nsfw_author_seen);
            }
            if nsfw_author_dropped > 0 {
                receiver.incr(
                    NSFW_AUTHOR_METRIC,
                    &[("outcome", "dropped")],
                    nsfw_author_dropped,
                );
            }
            for ((fallback, source, ptos_reviewed), count) in unscored_by_fallback {
                let ptos_reviewed = if ptos_reviewed { "true" } else { "false" };
                receiver.incr(
                    UNSCORED_AUTHOR_FALLBACK_METRIC,
                    &[
                        ("fallback", fallback),
                        ("source", source),
                        ("ptos_reviewed", ptos_reviewed),
                    ],
                    count,
                );
            }
        }

        results
    }

    fn update(&self, candidate: &mut PostCandidate, hydrated: PostCandidate) {
        candidate.brand_safety_verdict = hydrated.brand_safety_verdict;
        candidate.safety_labels = hydrated.safety_labels;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::query::ScoredPostsQuery;
    use std::sync::Mutex;
    use xai_safety_label_store::types::SafetyLabelMap;
    use xai_visibility_filtering::tweet_safety_label::{SafetyLabelFailure, SafetyLabelsBatch};
    use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelType};

    struct FakeVfClient {
        batch: SafetyLabelsBatch,
    }

    #[async_trait]
    impl TweetSafetyLabelClient for FakeVfClient {
        async fn get_safety_labels(
            &self,
            _tweet_ids: Vec<u64>,
        ) -> Result<SafetyLabelsBatch, tonic::Status> {
            Ok(self.batch.clone())
        }
    }

    #[tokio::test]
    async fn safe_verdict_with_grok_sfa() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }

    #[tokio::test]
    async fn medium_risk_when_not_scored() {
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, HashMap::new())]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn vf_failure_returns_error() {
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::new(),
                failures: HashMap::from([(1, SafetyLabelFailure::LookupFailed)]),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert!(results[0].is_err());
    }

    #[tokio::test]
    async fn quoted_tweet_failure_gives_medium_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::from([(2, SafetyLabelFailure::LookupFailed)]),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            quoted_tweet_id: Some(2),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn retweet_uses_retweeted_id() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(100, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            retweeted_tweet_id: Some(100),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }

    #[tokio::test]
    async fn safe_when_candidate_and_ancestors_safe() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels.clone()), (10, labels.clone()), (11, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10, 11],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }

    #[tokio::test]
    async fn ancestor_high_risk_escalates_verdict() {
        let mut safe_labels: SafetyLabelMap = HashMap::new();
        safe_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let mut risky_labels: SafetyLabelMap = HashMap::new();
        risky_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        risky_labels.insert(SafetyLabelType::NSFW_HIGH_PRECISION, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, safe_labels), (10, risky_labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::HighRisk)
        );
        assert!(hydrated
            .safety_labels
            .iter()
            .any(|l| l.label_type == SafetyLabelType::NSFW_HIGH_PRECISION));
    }

    #[tokio::test]
    async fn ancestor_low_risk_escalates_safe_candidate() {
        let mut safe_labels: SafetyLabelMap = HashMap::new();
        safe_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let mut low_risk_labels: SafetyLabelMap = HashMap::new();
        low_risk_labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        low_risk_labels.insert(
            SafetyLabelType::NSFA_LIMITED_INVENTORY,
            SafetyLabel::default(),
        );
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, safe_labels), (10, low_risk_labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::LowRisk)
        );
    }

    #[tokio::test]
    async fn ancestor_failure_gives_medium_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::from([(10, SafetyLabelFailure::LookupFailed)]),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn unscored_ancestor_gives_medium_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels), (10, HashMap::new())]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            ancestors: vec![10],
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::MediumRisk)
        );
    }

    #[tokio::test]
    async fn batch_rpc_failure_returns_errors_for_all() {
        struct FailingClient;
        #[async_trait]
        impl TweetSafetyLabelClient for FailingClient {
            async fn get_safety_labels(
                &self,
                _: Vec<u64>,
            ) -> Result<SafetyLabelsBatch, tonic::Status> {
                Err(tonic::Status::unavailable("vf down"))
            }
        }

        let hydrator = AdsBrandSafetyVfHydrator {
            client: Arc::new(FailingClient),
            author_client: None,
        };
        let candidates = vec![
            PostCandidate {
                tweet_id: 1,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                ..Default::default()
            },
        ];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        assert!(results[0].is_err());
        assert!(results[1].is_err());
    }

    #[tokio::test]
    async fn nsfw_author_escalates_to_high_risk() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            nsfw_author_ads: Some(true),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::HighRisk)
        );
    }

    #[tokio::test]
    async fn non_nsfw_author_keeps_safe_verdict() {
        let mut labels: SafetyLabelMap = HashMap::new();
        labels.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, labels)]),
                failures: HashMap::new(),
            },
        });
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: None,
        };
        let candidates = vec![PostCandidate {
            tweet_id: 1,
            nsfw_author_ads: Some(false),
            ..Default::default()
        }];

        let results = hydrator
            .hydrate(&ScoredPostsQuery::default(), &candidates)
            .await;

        let hydrated = results[0].as_ref().unwrap();
        assert_eq!(
            hydrated.brand_safety_verdict,
            Some(BrandSafetyVerdict::Safe)
        );
    }

    const POST_CUTOFF: u64 = 2_054_275_414_225_846_272;

    #[derive(Default)]
    struct RecordingAuthorClient {
        asked: Mutex<Vec<u64>>,
    }

    #[async_trait]
    impl AuthorBrandSafetyClient for RecordingAuthorClient {
        async fn fetch(
            &self,
            author_ids: &[u64],
        ) -> Result<HashMap<u64, Option<AuthorBrandSafetyFallback>>, String> {
            self.asked.lock().unwrap().extend_from_slice(author_ids);
            Ok(author_ids
                .iter()
                .map(|id| (*id, Some(AuthorBrandSafetyFallback::Sfa)))
                .collect())
        }
    }

    fn query_with_author_fallback(enabled: &str) -> ScoredPostsQuery {
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        results.override_fs(
            "rust_home_mixer_ads_bs_author_fallback_enabled".to_string(),
            enabled,
        );
        ScoredPostsQuery {
            params: results.into(),
            ..Default::default()
        }
    }

    async fn hydrate_with_author_fallback(
        enabled: &str,
    ) -> (Vec<Option<BrandSafetyVerdict>>, Vec<u64>) {
        let mut limited: SafetyLabelMap = HashMap::new();
        limited.insert(SafetyLabelType::GROK_NSFA_LIMITED, SafetyLabel::default());
        let mut sfa: SafetyLabelMap = HashMap::new();
        sfa.insert(SafetyLabelType::GROK_SFA, SafetyLabel::default());
        let client = Arc::new(FakeVfClient {
            batch: SafetyLabelsBatch {
                labels: HashMap::from([(1, HashMap::new()), (2, limited), (POST_CUTOFF, sfa)]),
                failures: HashMap::new(),
            },
        });
        let author_client = Arc::new(RecordingAuthorClient::default());
        let hydrator = AdsBrandSafetyVfHydrator {
            client,
            author_client: Some(author_client.clone()),
        };
        let candidates = vec![
            PostCandidate {
                tweet_id: 1,
                author_id: 7,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: 2,
                author_id: 8,
                ..Default::default()
            },
            PostCandidate {
                tweet_id: POST_CUTOFF,
                author_id: 9,
                ..Default::default()
            },
        ];
        let results = hydrator
            .hydrate(&query_with_author_fallback(enabled), &candidates)
            .await;
        let verdicts = results
            .iter()
            .map(|r| r.as_ref().unwrap().brand_safety_verdict)
            .collect();
        let asked = author_client.asked.lock().unwrap().clone();
        (verdicts, asked)
    }

    #[test]
    fn missing_row_or_failed_lookup_fails_closed_to_nsfa() {
        let fetched = HashMap::from([(7, Some(AuthorBrandSafetyFallback::Sfa)), (8, None)]);
        let nsfa = Some(AuthorBrandSafetyFallback::Nsfa);
        assert_eq!(
            author_fallback_for(&fetched, 7),
            (Some(AuthorBrandSafetyFallback::Sfa), "row")
        );
        assert_eq!(author_fallback_for(&fetched, 8), (nsfa, "no_row"));
        assert_eq!(author_fallback_for(&fetched, 9), (nsfa, "fetch_failed"));
    }

    #[tokio::test]
    async fn author_fallback_fetches_only_authors_of_posts_without_grok_labels() {
        let (_, asked) = hydrate_with_author_fallback("true").await;
        assert_eq!(asked, vec![7]);

        let (_, asked) = hydrate_with_author_fallback("false").await;
        assert!(asked.is_empty(), "param off: no fetch");
    }

    #[tokio::test]
    async fn param_applies_fallback_and_drops_ptos_check_but_never_overrides_grok_label() {
        let (off, _) = hydrate_with_author_fallback("false").await;
        assert_eq!(
            off,
            vec![
                Some(BrandSafetyVerdict::MediumRisk),
                Some(BrandSafetyVerdict::LowRisk),
                Some(BrandSafetyVerdict::MediumRisk),
            ],
            "off: unscored and unreviewed posts fail closed"
        );

        let (on, _) = hydrate_with_author_fallback("true").await;
        assert_eq!(
            on,
            vec![
                Some(BrandSafetyVerdict::Safe),
                Some(BrandSafetyVerdict::LowRisk),
                Some(BrandSafetyVerdict::Safe),
            ],
            "on: author SFA fallback, Grok label kept, PTOS check off"
        );
    }
}
