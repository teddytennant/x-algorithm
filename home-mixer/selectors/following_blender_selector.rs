use crate::ads::{AdsBlender, FollowingAdBlender};
use crate::models::query::ScoredPostsQuery;
use crate::params::{
    FOLLOWING_MAX_RESULT_SIZE, FOLLOWING_WHO_TO_FOLLOW_MIN_POSTS, PROMPTS_POSITION,
    WHO_TO_FOLLOW_POSITION,
};
use xai_candidate_pipeline::selector::{SelectResult, Selector};
use xai_home_mixer_proto::{feed_item, FeedItem, Prompt, ScoredPost, WhoToFollowModule};
use xai_recsys_proto::AdIndexInfo;

pub struct FollowingBlenderSelector;

impl FollowingBlenderSelector {
    pub fn new() -> Self {
        Self
    }
}

impl Selector<ScoredPostsQuery, FeedItem> for FollowingBlenderSelector {
    fn select(
        &self,
        _query: &ScoredPostsQuery,
        candidates: Vec<FeedItem>,
    ) -> SelectResult<FeedItem> {
        let PartitionedFeedItems {
            posts,
            ads,
            wtf_modules,
            prompts,
        } = partition_feed_items(candidates);

        let input_post_count = posts.len();
        let input_ad_count = ads.len();

        let blender = FollowingAdBlender {
            result_size: FOLLOWING_MAX_RESULT_SIZE,
        };
        let mut blended = blender.blend(posts, ads, &mut Vec::new());

        insert_prompts(&mut blended, prompts);
        if input_post_count > FOLLOWING_WHO_TO_FOLLOW_MIN_POSTS {
            insert_who_to_follow(&mut blended, wtf_modules);
        }

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
}

fn partition_feed_items(items: Vec<FeedItem>) -> PartitionedFeedItems {
    let mut posts = Vec::new();
    let mut ads = Vec::new();
    let mut wtf_modules = Vec::new();
    let mut prompts = Vec::new();
    for item in items {
        match item.item {
            Some(feed_item::Item::Post(post)) => posts.push(post),
            Some(feed_item::Item::Ad(ad)) => ads.push(ad),
            Some(feed_item::Item::WhoToFollow(wtf)) => wtf_modules.push(wtf),
            Some(feed_item::Item::Prompt(prompt)) => prompts.push(prompt),
            _ => {}
        }
    }
    PartitionedFeedItems {
        posts,
        ads,
        wtf_modules,
        prompts,
    }
}
