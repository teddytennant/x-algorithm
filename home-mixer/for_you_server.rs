use crate::candidate_pipeline::for_you_candidate_pipeline::ForYouCandidatePipeline;
use crate::models::query::ScoredPostsQuery;
use crate::params;
use crate::server::QueryBuilder;
use crate::util::feed_log;
use crate::util::under_the_hood;
use crate::util::urt;
use tonic::Status;
use xai_candidate_pipeline::candidate_pipeline::CandidatePipeline;
use xai_home_mixer_proto::{FeedItem, PipelineTrace};

pub(crate) struct ForYouFeedOutput {
    pub items: Vec<FeedItem>,
    pub pipeline_trace: Option<PipelineTrace>,
}

pub struct ForYouFeedServer {
    pub(crate) query_builder: QueryBuilder,
    pipeline: ForYouCandidatePipeline,
}

impl ForYouFeedServer {
    pub fn new(query_builder: QueryBuilder, pipeline: ForYouCandidatePipeline) -> Self {
        Self {
            query_builder,
            pipeline,
        }
    }

    pub(crate) async fn get_for_you_feed(
        &self,
        query: ScoredPostsQuery,
    ) -> Result<ForYouFeedOutput, Status> {
        if params::TEST_USER_IDS.contains(&query.user_id) {
            return Ok(ForYouFeedOutput {
                items: vec![],
                pipeline_trace: None,
            });
        }

        let (result, trace) = self.pipeline.execute_traced(query).await;
        feed_log::log_feed_response(&result);

        Ok(ForYouFeedOutput {
            pipeline_trace: trace
                .filter(|_| result.query.is_under_the_hood_request)
                .map(|trace| under_the_hood::pipeline_trace(&trace)),
            items: result.selected_candidates,
        })
    }

    pub(crate) async fn get_for_you_feed_urt(
        &self,
        query: ScoredPostsQuery,
    ) -> Result<Vec<u8>, Status> {
        feed_log::log_feed_request(&query);

        let cursor = query.cursor.clone();
        let request_context = query.request_context.clone();
        let client_app_id = query.client_app_id;
        let viewer_id = query.user_id;
        let language_code = query.language_code.clone();
        let country_code = query.country_code.clone();
        let request_type = query.request_type;
        let request_id = query.request_id;
        let entry_id_nonce = query
            .params
            .get(params::EnableEntryIdRandomHash)
            .then_some(query.request_id);

        let output = self.get_for_you_feed(query).await?;

        let timeline_response = urt::make_urt_timeline(
            &output.items,
            cursor.as_ref(),
            &request_context,
            client_app_id,
            viewer_id,
            &language_code,
            if country_code.is_empty() {
                None
            } else {
                Some(&country_code)
            },
            entry_id_nonce,
            request_type,
            request_id,
        );

        xai_urt_thrift::serialize_binary(&timeline_response)
            .map_err(|e| Status::internal(format!("failed to serialize URT: {e}")))
    }
}
