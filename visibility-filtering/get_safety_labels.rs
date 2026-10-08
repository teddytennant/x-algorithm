use crate::caller_identity::{self, Endpoint};
use crate::safety_label_source::metrics::{self, BatchStage, RequestMetricsGuard};
use crate::safety_label_source::types::FailureKind;
use crate::safety_label_source::{LookupError, SafetyLabelSource};
use enum_map::EnumMap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use xai_visibility_filtering_proto as vf_pb;

pub struct GetSafetyLabelsEndpoint {
    safety_label_source: Arc<SafetyLabelSource>,
}

impl GetSafetyLabelsEndpoint {
    pub fn new(safety_label_source: Arc<SafetyLabelSource>) -> Self {
        Self {
            safety_label_source,
        }
    }

    pub async fn handle(
        &self,
        request: Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<Response<vf_pb::GetSafetyLabelsResponse>, Status> {
        let caller = caller_identity::record(Endpoint::GetSafetyLabels, &request);
        let request_metrics = RequestMetricsGuard::new();
        match self.handle_inner(request).await {
            Ok(response) => {
                request_metrics.mark_success();
                caller.mark_success();
                Ok(response)
            }
            Err(status) => {
                request_metrics.mark_failure();
                caller.mark_failure();
                Err(status)
            }
        }
    }

    async fn handle_inner(
        &self,
        request: Request<vf_pb::GetSafetyLabelsRequest>,
    ) -> Result<Response<vf_pb::GetSafetyLabelsResponse>, Status> {
        let req = request.into_inner();

        let unique_ids: Vec<u64> = req
            .tweet_ids
            .into_iter()
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect();
        metrics::record_batch_size(BatchStage::Request, unique_ids.len());

        let resolved = self.safety_label_source.get(&unique_ids).await;
        let outcome = GetSafetyLabelsOutcome::try_from_resolved(resolved, unique_ids.len())?;
        let failed_count = outcome.failure_count();

        metrics::record_lookup_tweet_ids(outcome.success_count(), failed_count);
        for (kind, &count) in &outcome.failures {
            metrics::record_lookup_failures(kind, count);
        }

        Ok(Response::new(vf_pb::GetSafetyLabelsResponse {
            results: outcome.results,
            failed_ids: outcome.failed_ids,
        }))
    }
}

#[derive(Debug)]
pub(crate) struct GetSafetyLabelsOutcome {
    requested_count: usize,
    pub(crate) results: HashMap<u64, vf_pb::SafetyLabelMap>,
    pub(crate) failed_ids: Vec<u64>,
    pub(crate) failures: EnumMap<FailureKind, usize>,
}

impl GetSafetyLabelsOutcome {
    pub(crate) fn try_from_resolved(
        resolved: FxHashMap<u64, Result<Arc<vf_pb::SafetyLabelMap>, LookupError>>,
        requested_count: usize,
    ) -> Result<Self, Status> {
        let mut results = HashMap::with_capacity(resolved.len());
        let mut failed_ids = Vec::new();
        let mut failures = EnumMap::default();

        for (id, lookup_result) in resolved {
            match lookup_result {
                Ok(label_map) => {
                    results.insert(id, Arc::unwrap_or_clone(label_map));
                }
                Err(e) => {
                    failed_ids.push(id);
                    failures[e.kind] += 1;
                }
            }
        }

        failed_ids.sort_unstable();

        let result = Self {
            requested_count,
            results,
            failed_ids,
            failures,
        };

        if result.accounted_count() != result.requested_count {
            return Err(Status::internal(format!(
                "GetSafetyLabels invariant violation: requested={} accounted={}",
                result.requested_count,
                result.accounted_count()
            )));
        }

        Ok(result)
    }

    pub(crate) fn success_count(&self) -> usize {
        self.results.len()
    }

    pub(crate) fn failure_count(&self) -> usize {
        self.failed_ids.len()
    }

    fn accounted_count(&self) -> usize {
        self.results.len() + self.failed_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    fn labels_with_entry(type_id: i32) -> vf_pb::SafetyLabelMap {
        vf_pb::SafetyLabelMap {
            labels: HashMap::from([(
                type_id,
                vf_pb::SafetyLabel {
                    score: Some(0.5),
                    applicable_users: Vec::new(),
                    holdback_experiment: None,
                    source: None,
                    created_at_msec: None,
                    expires_at_msec: None,
                    applicable_countries: Vec::new(),
                    safety_label_source: None,
                },
            )]),
        }
    }

    #[test]
    fn try_from_resolved_reports_partial_failure() {
        let outcome = GetSafetyLabelsOutcome::try_from_resolved(
            FxHashMap::from_iter([
                (1, Ok(Arc::new(labels_with_entry(11)))),
                (4, Ok(Arc::new(labels_with_entry(44)))),
                (
                    2,
                    Err(LookupError::new(FailureKind::ManhattanDecode, "decode")),
                ),
                (
                    3,
                    Err(LookupError::new(FailureKind::ManhattanFetch, "mh down")),
                ),
            ]),
            4,
        )
        .unwrap();

        assert_eq!(outcome.success_count(), 2);
        assert_eq!(outcome.results.len(), 2);
        assert_eq!(outcome.failed_ids, vec![2, 3]);
        assert_eq!(outcome.failures[FailureKind::ManhattanFetch], 1);
        assert_eq!(outcome.failures[FailureKind::ManhattanDecode], 1);
        assert_eq!(outcome.failure_count(), 2);
    }

    #[test]
    fn try_from_resolved_rejects_missing_ids() {
        let status = GetSafetyLabelsOutcome::try_from_resolved(
            FxHashMap::from_iter([(7, Ok(Arc::new(labels_with_entry(22))))]),
            2,
        )
        .unwrap_err();

        assert_eq!(status.code(), Code::Internal);
        assert!(status.message().contains("requested=2 accounted=1"));
    }
}
