use crate::ads::{
    AdsBlender, MultiRiskAdsBlender, PartitionOrganicAdsBlender, SafeGapAdsBlender,
    TimeGapAdsBlender, TimeGapConfig,
};
use crate::frames;
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    AdsBlenderType, AdsTimeGapClampHi, AdsTimeGapClampLo, AdsTimeGapMinOrganicGap, AdsTimeGapTSec,
    VideoCarouselPosition, FEED_SURVEY_POSITION, PROMPTS_POSITION, WHO_TO_FOLLOW_POSITION,
};
use xai_candidate_pipeline::selector::{SelectResult, Selector};
use xai_home_mixer_proto::{
    feed_item, FeedItem, FeedSurvey, Frame, Prompt, PushToHomePost, ScoredPost,
    VideoCarouselModule, WhoToFollowModule,
};
use xai_recsys_proto::AdIndexInfo;

pub struct BlenderSelector {
    safe_gap_blender: SafeGapAdsBlender,
    partition_organic_blender: PartitionOrganicAdsBlender,
    multi_risk_blender: MultiRiskAdsBlender,
}

impl BlenderSelector {
    pub fn new() -> Self {
        Self {
            safe_gap_blender: SafeGapAdsBlender::default(),
            partition_organic_blender: PartitionOrganicAdsBlender,
            multi_risk_blender: MultiRiskAdsBlender,
        }
    }
}

impl Selector<ScoredPostsQuery, FeedItem> for BlenderSelector {
    fn select(
        &self,
        query: &ScoredPostsQuery,
        candidates: Vec<FeedItem>,
    ) -> SelectResult<FeedItem> {
        let PartitionedFeedItems {
            posts,
            ads,
            wtf_modules,
            prompts,
            push_to_home,
            frames,
            feed_survey,
            video_carousel,
        } = partition_feed_items(candidates);

        let input_post_count = posts.len();
        let input_ad_count = ads.len();

        let time_gap_blender;
        let blender: &dyn AdsBlender = match query.params.get(AdsBlenderType).as_str() {
            "partition_organic_low_risk" => &self.partition_organic_blender,
            "safe_gap" => &self.safe_gap_blender,
            "multi_risk" => &self.multi_risk_blender,
            "time_gap" => {
                time_gap_blender = TimeGapAdsBlender {
                    config: TimeGapConfig {
                        t_sec: query.params.get(AdsTimeGapTSec),
                        clamp_lo: query.params.get(AdsTimeGapClampLo),
                        clamp_hi: query.params.get(AdsTimeGapClampHi),
                        min_organic_gap: query.params.get(AdsTimeGapMinOrganicGap) as usize,
                    },
                };
                &time_gap_blender
            }
            _ => &self.partition_organic_blender,
        };
        let mut drops = Vec::new();
        let mut blended = blender.blend(posts, ads, &mut drops);
        query.ad_drops.extend(drops);

        insert_prompts(&mut blended, prompts);
        insert_who_to_follow(&mut blended, wtf_modules);
        insert_video_carousel(
            &mut blended,
            video_carousel,
            query.params.get(VideoCarouselPosition) as usize,
        );
        pin_push_to_home(&mut blended, push_to_home);
        insert_frames(&mut blended, frames);
        insert_feed_survey(&mut blended, feed_survey);

        let output_post_count = blended
            .iter()
            .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
            .count();
        let output_ad_count = blended
            .iter()
            .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
            .count();

        let dropped_posts = input_post_count.saturating_sub(output_post_count);
        let dropped_ads = input_ad_count.saturating_sub(output_ad_count);
        let non_selected = build_non_selected_placeholders(dropped_posts, dropped_ads);

        SelectResult {
            selected: blended,
            non_selected,
        }
    }

    fn score(&self, _candidate: &FeedItem) -> f64 {
        0.0
    }
}

fn insert_prompts(blended: &mut Vec<FeedItem>, prompts: Vec<Prompt>) {
    for (i, prompt) in prompts.into_iter().enumerate() {
        blended.insert(
            i,
            FeedItem {
                position: PROMPTS_POSITION,
                item: Some(feed_item::Item::Prompt(prompt)),
            },
        );
    }
}

fn insert_who_to_follow(blended: &mut Vec<FeedItem>, wtf_modules: Vec<WhoToFollowModule>) {
    let Some(wtf) = wtf_modules.into_iter().next() else {
        return;
    };
    let insert_idx = WHO_TO_FOLLOW_POSITION.saturating_sub(1).min(blended.len());
    blended.insert(
        insert_idx,
        FeedItem {
            position: WHO_TO_FOLLOW_POSITION as i32,
            item: Some(feed_item::Item::WhoToFollow(wtf)),
        },
    );
}

fn insert_video_carousel(
    blended: &mut Vec<FeedItem>,
    video_carousel: Option<VideoCarouselModule>,
    position: usize,
) {
    let Some(carousel) = video_carousel else {
        return;
    };
    let insert_idx = position.saturating_sub(1).min(blended.len());
    blended.insert(
        insert_idx,
        FeedItem {
            position: position as i32,
            item: Some(feed_item::Item::VideoCarousel(carousel)),
        },
    );
}

fn pin_push_to_home(blended: &mut Vec<FeedItem>, push_to_home: Option<PushToHomePost>) {
    let Some(pth) = push_to_home else {
        return;
    };
    blended.insert(
        0,
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::PushToHome(pth)),
        },
    );
}

fn insert_frames(blended: &mut Vec<FeedItem>, frames: Vec<Frame>) {
    if frames.is_empty() {
        return;
    }
    let planned = frames::plan(frames, blended.len());
    insert_planned_frames(blended, planned);
}

fn insert_planned_frames(blended: &mut Vec<FeedItem>, planned: Vec<(usize, Frame)>) {
    let mut next_index = 0;
    for (already_inserted, (slot, frame)) in planned.into_iter().enumerate() {
        let insert_idx = (slot + already_inserted).max(next_index).min(blended.len());
        blended.insert(
            insert_idx,
            FeedItem {
                position: insert_idx as i32,
                item: Some(feed_item::Item::Frame(frame)),
            },
        );
        next_index = insert_idx + 1;
    }
}

fn insert_feed_survey(blended: &mut Vec<FeedItem>, feed_survey: Option<FeedSurvey>) {
    let Some(survey) = feed_survey else {
        return;
    };
    let insert_idx = FEED_SURVEY_POSITION.saturating_sub(1).min(blended.len());
    blended.insert(
        insert_idx,
        FeedItem {
            position: FEED_SURVEY_POSITION as i32,
            item: Some(feed_item::Item::FeedSurvey(survey)),
        },
    );
}

fn build_non_selected_placeholders(dropped_posts: usize, dropped_ads: usize) -> Vec<FeedItem> {
    let mut non_selected = Vec::with_capacity(dropped_posts + dropped_ads);
    for _ in 0..dropped_posts {
        non_selected.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost::default())),
        });
    }
    for _ in 0..dropped_ads {
        non_selected.push(FeedItem {
            position: 0,
            item: Some(feed_item::Item::Ad(AdIndexInfo::default())),
        });
    }
    non_selected
}

struct PartitionedFeedItems {
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    wtf_modules: Vec<WhoToFollowModule>,
    prompts: Vec<Prompt>,
    push_to_home: Option<PushToHomePost>,
    frames: Vec<Frame>,
    feed_survey: Option<FeedSurvey>,
    video_carousel: Option<VideoCarouselModule>,
}

fn partition_feed_items(items: Vec<FeedItem>) -> PartitionedFeedItems {
    let mut posts = Vec::new();
    let mut ads = Vec::new();
    let mut wtf_modules = Vec::new();
    let mut prompts = Vec::new();
    let mut push_to_home = None;
    let mut frames = Vec::new();
    let mut feed_survey = None;
    let mut video_carousel = None;
    for item in items {
        match item.item {
            Some(feed_item::Item::Post(post)) => posts.push(post),
            Some(feed_item::Item::Ad(ad)) => ads.push(ad),
            Some(feed_item::Item::WhoToFollow(wtf)) => wtf_modules.push(wtf),
            Some(feed_item::Item::Prompt(prompt)) => prompts.push(prompt),
            Some(feed_item::Item::PushToHome(pth)) => push_to_home = Some(pth),
            Some(feed_item::Item::Frame(f)) => frames.push(f),
            Some(feed_item::Item::FeedSurvey(s)) => feed_survey = Some(s),
            Some(feed_item::Item::VideoCarousel(c)) => video_carousel = Some(c),
            None => {}
        }
    }
    PartitionedFeedItems {
        posts,
        ads,
        wtf_modules,
        prompts,
        push_to_home,
        frames,
        feed_survey,
        video_carousel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(route: &str) -> Frame {
        Frame {
            payload: "RENDERED".to_string(),
            height: 100,
            is_dismissible: false,
            placement_id: 0,
            route: route.to_string(),
            occurrence: 0,
            fetch_route: route.to_string(),
        }
    }

    fn post_item() -> FeedItem {
        FeedItem {
            position: 0,
            item: Some(feed_item::Item::Post(ScoredPost::default())),
        }
    }

    fn routes(items: &[FeedItem]) -> Vec<String> {
        items
            .iter()
            .filter_map(|i| match &i.item {
                Some(feed_item::Item::Frame(f)) => Some(f.route.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_frame_is_retained_by_the_partition() {
        let items = vec![
            FeedItem {
                position: 0,
                item: Some(feed_item::Item::Frame(frame("/cards/nfl/timeline/pinned"))),
            },
            FeedItem {
                position: 0,
                item: Some(feed_item::Item::Frame(frame("/cards/nfl/timeline/card"))),
            },
        ];
        assert_eq!(partition_feed_items(items).frames.len(), 2);
    }

    #[test]
    fn frames_sharing_a_slot_keep_their_planned_order() {
        let mut blended: Vec<FeedItem> = (0..20).map(|_| post_item()).collect();
        insert_planned_frames(
            &mut blended,
            vec![
                (10, frame("/first")),
                (10, frame("/second")),
                (10, frame("/third")),
            ],
        );
        assert_eq!(routes(&blended), vec!["/first", "/second", "/third"]);
    }

    fn frame_indices(items: &[FeedItem]) -> Vec<usize> {
        items
            .iter()
            .enumerate()
            .filter(|(_, i)| matches!(i.item, Some(feed_item::Item::Frame(_))))
            .map(|(index, _)| index)
            .collect()
    }

    fn gaps_between(indices: &[usize]) -> Vec<usize> {
        indices
            .windows(2)
            .map(|pair| pair[1] - pair[0] - 1)
            .collect()
    }

    fn nfl_cards(count: u32) -> Vec<Frame> {
        (0..count)
            .map(|occurrence| Frame {
                occurrence,
                ..frame("/cards/nfl/timeline/card")
            })
            .collect()
    }

    #[test]
    fn exactly_seven_feed_items_sit_between_consecutive_nfl_cards() {
        let mut blended: Vec<FeedItem> = (0..40).map(|_| post_item()).collect();
        insert_frames(&mut blended, nfl_cards(5));
        assert_eq!(gaps_between(&frame_indices(&blended)), vec![7, 7, 7, 7]);
    }

    #[test]
    fn the_spacing_holds_when_only_some_payloads_come_back() {
        let mut blended: Vec<FeedItem> = (0..40).map(|_| post_item()).collect();
        let arrived = vec![
            Frame {
                occurrence: 0,
                ..frame("/cards/nfl/timeline/card")
            },
            Frame {
                occurrence: 3,
                ..frame("/cards/nfl/timeline/card")
            },
            Frame {
                occurrence: 4,
                ..frame("/cards/nfl/timeline/card")
            },
        ];
        insert_frames(&mut blended, arrived);
        assert_eq!(gaps_between(&frame_indices(&blended)), vec![7, 7]);
    }

    #[test]
    fn pinned_frames_stack_without_an_organic_post_between_them() {
        let mut blended: Vec<FeedItem> = (0..20).map(|_| post_item()).collect();
        insert_frames(
            &mut blended,
            vec![
                frame("/cards/nfl/timeline/pinned"),
                frame("/cards/soccer/timelineCarousel"),
            ],
        );
        assert_eq!(frame_indices(&blended), vec![0, 1]);
    }

    #[test]
    fn the_pinned_nfl_card_sits_above_organic_posts() {
        let mut blended = vec![post_item(), post_item()];
        insert_frames(&mut blended, vec![frame("/cards/nfl/timeline/pinned")]);
        assert_eq!(routes(&blended), vec!["/cards/nfl/timeline/pinned"]);
        assert!(
            matches!(
                blended.first().and_then(|i| i.item.as_ref()),
                Some(feed_item::Item::Frame(_))
            ),
            "the pinned card must not sit below an organic post"
        );
    }
}
