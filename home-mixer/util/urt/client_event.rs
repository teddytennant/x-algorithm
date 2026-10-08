use super::controller_data;
use crate::models::query::RequestType;
use crate::util::string_case::upper_snake_to_pascal;
use xai_home_mixer_proto::{ScoredPost, ServedType};
use xai_recsys_proto::AdIndexInfo;
use xai_urt_thrift::metadata::{ClientEventDetails, ClientEventInfo, TimelinesDetails};

pub(super) const ELEMENT_TWEET: &str = "tweet";
pub(super) const ELEMENT_USER: &str = "user";
pub(super) const COMPONENT_WTF: &str = "suggest_who_to_follow";
const COMPONENT_ADS_FOR_YOU: &str = "for_you_promoted";
const ADS_INJECTION_TYPE_FOR_YOU: &str = "ForYouPromoted";
const COMPONENT_ADS_FOLLOWING: &str = "following_promoted";
const ADS_INJECTION_TYPE_FOLLOWING: &str = "FollowingPromoted";
const COMPONENT_ADS_RANKED_FOLLOWING: &str = "ranked_following_promoted";
const ADS_INJECTION_TYPE_RANKED_FOLLOWING: &str = "RankedFollowingPromoted";
const WTF_INJECTION_TYPE: &str = "WhoToFollow";
const COMPONENT_VIDEO_CAROUSEL: &str = "video_carousel";

pub(super) fn served_type_component(st: i32) -> String {
    ServedType::try_from(st)
        .unwrap_or(ServedType::Undefined)
        .as_str_name()
        .to_ascii_lowercase()
}

pub(super) fn empty_details() -> ClientEventDetails {
    ClientEventDetails {
        timelines_details: None,
        trends_details: None,
        search_request_details: None,
        search_result_details: None,
        moments_details: None,
        conversation_details: None,
        live_event_details: None,
        periscope_details: None,
        topic_details: None,
        news_details: None,
        guide_details: None,
        notification_details: None,
        article_details: None,
        commerce_details: None,
        ai_trend_details: None,
        spotlight_details: None,
        entry_point_pivot_details: None,
        adindex_details: None,
    }
}

pub(super) fn post_client_event_info(
    component: &str,
    post: &ScoredPost,
    position: i32,
) -> ClientEventInfo {
    let cd = controller_data::post_controller_data(
        &post.tweet_type_metrics,
        post.last_scored_timestamp_ms,
        position,
        post.prediction_request_id,
    );
    let mut details = empty_details();
    details.timelines_details = Some(TimelinesDetails {
        injection_type: Some(upper_snake_to_pascal(component)),
        controller_data: cd,
        source_data: None,
    });

    ClientEventInfo {
        component: Some(component.to_string()),
        element: Some(ELEMENT_TWEET.to_string()),
        details: Some(details),
        action: None,
        entity_token: None,
    }
}

pub(super) fn video_carousel_item_client_event_info(
    video: &ScoredPost,
    position: i32,
) -> ClientEventInfo {
    post_client_event_info(COMPONENT_VIDEO_CAROUSEL, video, position)
}

pub(super) fn video_carousel_module_client_event_info() -> ClientEventInfo {
    let mut details = empty_details();
    details.timelines_details = Some(TimelinesDetails {
        injection_type: Some(upper_snake_to_pascal(COMPONENT_VIDEO_CAROUSEL)),
        controller_data: None,
        source_data: None,
    });

    ClientEventInfo {
        component: Some(COMPONENT_VIDEO_CAROUSEL.to_string()),
        element: None,
        details: Some(details),
        action: None,
        entity_token: None,
    }
}

pub(super) fn ad_client_event_info(ad: &AdIndexInfo, request_type: RequestType) -> ClientEventInfo {
    let (component, injection_type) = match request_type {
        RequestType::Following => (COMPONENT_ADS_FOLLOWING, ADS_INJECTION_TYPE_FOLLOWING),
        RequestType::RankedFollowing => (
            COMPONENT_ADS_RANKED_FOLLOWING,
            ADS_INJECTION_TYPE_RANKED_FOLLOWING,
        ),
        _ => (COMPONENT_ADS_FOR_YOU, ADS_INJECTION_TYPE_FOR_YOU),
    };
    let mut details = empty_details();
    details.timelines_details = Some(TimelinesDetails {
        injection_type: Some(injection_type.to_string()),
        controller_data: controller_data::ad_item_controller_data(),
        source_data: None,
    });
    details.adindex_details = controller_data::ad_index_controller_data(ad);

    ClientEventInfo {
        component: Some(component.to_string()),
        element: Some(ELEMENT_TWEET.to_string()),
        details: Some(details),
        action: None,
        entity_token: None,
    }
}

pub(super) fn wtf_item_client_event_info(tracking_token: Option<&str>) -> ClientEventInfo {
    let mut details = empty_details();
    details.timelines_details = Some(TimelinesDetails {
        injection_type: Some(WTF_INJECTION_TYPE.to_string()),
        controller_data: tracking_token.and_then(controller_data::wtf_controller_data),
        source_data: tracking_token.map(|s| s.to_string()),
    });

    ClientEventInfo {
        component: Some(COMPONENT_WTF.to_string()),
        element: Some(ELEMENT_USER.to_string()),
        details: Some(details),
        action: None,
        entity_token: None,
    }
}

pub(super) fn wtf_module_client_event_info(tracking_token: Option<&str>) -> ClientEventInfo {
    let mut details = empty_details();
    details.timelines_details = Some(TimelinesDetails {
        injection_type: Some(WTF_INJECTION_TYPE.to_string()),
        controller_data: tracking_token.and_then(controller_data::wtf_controller_data),
        source_data: tracking_token.map(|s| s.to_string()),
    });

    ClientEventInfo {
        component: Some(COMPONENT_WTF.to_string()),
        element: None,
        details: Some(details),
        action: None,
        entity_token: None,
    }
}
