use crate::ads::multi_risk_blender::*;
use xai_ads_injection_proto::ads_injected_timeline::{AdDropReason, DroppedAd};
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo, BrandSafetyRiskLevel};

fn make_post(tweet_id: u64) -> ScoredPost {
    ScoredPost {
        tweet_id,
        score: 1.0 - (tweet_id as f32 * 0.01),
        ..Default::default()
    }
}

fn make_avoid_post(tweet_id: u64) -> ScoredPost {
    ScoredPost {
        tweet_id,
        brand_safety_verdict: BrandSafetyVerdict::MediumRisk.into(),
        score: 1.0 - (tweet_id as f32 * 0.01),
        ..Default::default()
    }
}

fn make_high_risk_post(tweet_id: u64) -> ScoredPost {
    ScoredPost {
        tweet_id,
        brand_safety_verdict: BrandSafetyVerdict::HighRisk.into(),
        score: 1.0 - (tweet_id as f32 * 0.01),
        ..Default::default()
    }
}

fn make_normal_ad(post_id: i64) -> AdIndexInfo {
    AdIndexInfo {
        post_id,
        ad_adjacency_control: Some(AdAdjacencyControl {
            brand_safety_risk: BrandSafetyRiskLevel::BsrNormal.into(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_bsr_high_ad(post_id: i64) -> AdIndexInfo {
    AdIndexInfo {
        post_id,
        ad_adjacency_control: Some(AdAdjacencyControl {
            brand_safety_risk: BrandSafetyRiskLevel::BsrHigh.into(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn make_bsr_high_ad_with_handles(post_id: i64, handles: &[i64]) -> AdIndexInfo {
    AdIndexInfo {
        post_id,
        ad_adjacency_control: Some(AdAdjacencyControl {
            brand_safety_risk: BrandSafetyRiskLevel::BsrHigh.into(),
            handles: handles.to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn ad_count(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .count()
}

fn ad_neighbour_verdicts(items: &[FeedItem]) -> Vec<(BrandSafetyVerdict, BrandSafetyVerdict)> {
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if matches!(item.item, Some(feed_item::Item::Ad(_))) {
            let above = if i > 0 {
                match &items[i - 1].item {
                    Some(feed_item::Item::Post(p)) => p.brand_safety_verdict(),
                    _ => BrandSafetyVerdict::VerdictUnspecified,
                }
            } else {
                BrandSafetyVerdict::VerdictUnspecified
            };
            let below = if i + 1 < items.len() {
                match &items[i + 1].item {
                    Some(feed_item::Item::Post(p)) => p.brand_safety_verdict(),
                    _ => BrandSafetyVerdict::VerdictUnspecified,
                }
            } else {
                BrandSafetyVerdict::VerdictUnspecified
            };
            result.push((above, below));
        }
    }
    result
}

fn ad_bsr_levels(items: &[FeedItem]) -> Vec<BrandSafetyRiskLevel> {
    items
        .iter()
        .filter_map(|i| match &i.item {
            Some(feed_item::Item::Ad(ad)) => Some(
                ad.ad_adjacency_control
                    .as_ref()
                    .map(|c| c.brand_safety_risk())
                    .unwrap_or(BrandSafetyRiskLevel::BsrUnknown),
            ),
            _ => None,
        })
        .collect()
}

#[test]
fn bsr_high_ad_adjacent_to_medium_risk() {
    let mut posts: Vec<_> = (1..=6).map(make_avoid_post).collect();
    posts.extend((7..=12).map(make_post));
    let result = blend_impl(posts, vec![make_bsr_high_ad(100)], 5);
    assert_eq!(ad_count(&result), 1);
    let (above, below) = ad_neighbour_verdicts(&result)[0];
    assert_eq!(above, BrandSafetyVerdict::MediumRisk);
    assert_eq!(below, BrandSafetyVerdict::MediumRisk);
}

#[test]
fn normal_ad_not_adjacent_to_medium_risk_when_bsr_high_fills() {
    let mut posts: Vec<_> = (1..=6).map(make_avoid_post).collect();
    posts.extend((7..=12).map(make_post));
    let result = blend_impl(posts, vec![make_normal_ad(100), make_bsr_high_ad(200)], 5);
    let bsr_levels = ad_bsr_levels(&result);
    let neighbours = ad_neighbour_verdicts(&result);
    assert!(bsr_levels.contains(&BrandSafetyRiskLevel::BsrHigh));
    for (i, bsr) in bsr_levels.iter().enumerate() {
        let (above, below) = neighbours[i];
        if *bsr == BrandSafetyRiskLevel::BsrNormal {
            assert_ne!(above, BrandSafetyVerdict::MediumRisk);
            assert_ne!(below, BrandSafetyVerdict::MediumRisk);
        }
        if *bsr == BrandSafetyRiskLevel::BsrHigh {
            assert_eq!(above, BrandSafetyVerdict::MediumRisk);
            assert_eq!(below, BrandSafetyVerdict::MediumRisk);
        }
        assert_ne!(above, BrandSafetyVerdict::HighRisk);
        assert_ne!(below, BrandSafetyVerdict::HighRisk);
    }
}

#[test]
fn high_risk_never_adjacent_including_bsr_high() {
    let mut posts: Vec<_> = (1..=6).map(make_high_risk_post).collect();
    posts.extend((7..=12).map(make_post));
    let result = blend_impl(posts, vec![make_bsr_high_ad(100), make_normal_ad(200)], 5);
    assert!(ad_count(&result) > 0);
    for (above, below) in ad_neighbour_verdicts(&result) {
        assert_ne!(above, BrandSafetyVerdict::HighRisk);
        assert_ne!(below, BrandSafetyVerdict::HighRisk);
        assert_ne!(above, BrandSafetyVerdict::MediumRisk);
        assert_ne!(below, BrandSafetyVerdict::MediumRisk);
    }
}

#[test]
fn all_medium_risk_places_bsr_high_drops_normal() {
    let posts: Vec<_> = (1..=8).map(make_avoid_post).collect();
    let with_high = blend_impl(posts.clone(), vec![make_bsr_high_ad(100)], 5);
    assert_eq!(ad_count(&with_high), 1);
    let (above, below) = ad_neighbour_verdicts(&with_high)[0];
    assert_eq!(above, BrandSafetyVerdict::MediumRisk);
    assert_eq!(below, BrandSafetyVerdict::MediumRisk);

    let with_normal = blend_impl(posts, vec![make_normal_ad(100)], 5);
    assert_eq!(ad_count(&with_normal), 0);
}

#[test]
fn excluded_high_does_not_burn_medium_pair() {
    let mut medium: Vec<_> = (1..=4).map(make_avoid_post).collect();
    medium[0].author_id = 9999;
    medium[1].author_id = 9999;
    let mut posts = medium;
    posts.extend((5..=12).map(make_post));

    let result = blend_impl(
        posts,
        vec![
            make_bsr_high_ad_with_handles(100, &[9999]),
            make_bsr_high_ad(200),
        ],
        5,
    );
    assert_eq!(ad_count(&result), 2);

    let mut neighbours_by_ad = std::collections::HashMap::new();
    for (i, item) in result.iter().enumerate() {
        let Some(feed_item::Item::Ad(ad)) = &item.item else {
            continue;
        };
        let above = match &result[i - 1].item {
            Some(feed_item::Item::Post(p)) => p.brand_safety_verdict(),
            _ => BrandSafetyVerdict::VerdictUnspecified,
        };
        let below = match &result[i + 1].item {
            Some(feed_item::Item::Post(p)) => p.brand_safety_verdict(),
            _ => BrandSafetyVerdict::VerdictUnspecified,
        };
        neighbours_by_ad.insert(ad.post_id, (above, below));
    }

    let (above_100, below_100) = neighbours_by_ad[&100];
    assert_ne!(above_100, BrandSafetyVerdict::MediumRisk);
    assert_ne!(below_100, BrandSafetyVerdict::MediumRisk);

    let (above_200, below_200) = neighbours_by_ad[&200];
    assert_eq!(above_200, BrandSafetyVerdict::MediumRisk);
    assert_eq!(below_200, BrandSafetyVerdict::MediumRisk);
}

#[test]
fn records_drop_reasons() {
    let mut posts: Vec<_> = (1..=12).map(make_post).collect();
    posts[0].tweet_text = "new tires".to_string();
    let ad = |post_id: i64, insert_position: i32| AdIndexInfo {
        insert_position,
        ..make_normal_ad(post_id)
    };
    let mut keyword_ad = ad(100, 3);
    keyword_ad.ad_adjacency_control.as_mut().unwrap().keywords = vec!["tires".to_string()];
    let ads = vec![keyword_ad, ad(200, 6), ad(300, 9), ad(400, 12), ad(500, 15)];

    let mut drops = Vec::new();
    let result = blend_with_drops(posts, ads, 5, &mut drops);

    assert_eq!(ad_count(&result), 3);
    let summary: Vec<_> = drops
        .iter()
        .map(|d| {
            (
                d.tweet_id,
                d.drop_reason(),
                d.ad_info.as_ref().unwrap().insert_position,
                d.blocking_tweet_id,
                d.matched_keyword.as_str(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            (100, AdDropReason::ExcludedKeyword, 3, 1, "tires"),
            (500, AdDropReason::SpacingCap, 15, 0, ""),
        ]
    );
}

fn drop_summary(drops: &[DroppedAd]) -> Vec<(u64, AdDropReason, u64, u64)> {
    drops
        .iter()
        .map(|d| {
            (
                d.tweet_id,
                d.drop_reason(),
                d.blocking_tweet_id,
                d.matched_user_id,
            )
        })
        .collect()
}

#[test]
fn records_blocking_neighbours() {
    let mut posts: Vec<_> = (1..=12).map(make_post).collect();
    posts[0].brand_safety_verdict = BrandSafetyVerdict::LowRisk.into();
    posts[1].author_id = 9999;
    let mut sensitive_ad = make_normal_ad(100);
    sensitive_ad
        .ad_adjacency_control
        .as_mut()
        .unwrap()
        .brand_safety_risk = BrandSafetyRiskLevel::BsrLow.into();
    let mut handle_ad = make_normal_ad(200);
    handle_ad.ad_adjacency_control.as_mut().unwrap().handles = vec![9999];

    let mut drops = Vec::new();
    let result = blend_with_drops(
        posts,
        vec![sensitive_ad, handle_ad, make_normal_ad(300)],
        5,
        &mut drops,
    );

    assert_eq!(ad_count(&result), 1);
    assert_eq!(
        drop_summary(&drops),
        vec![
            (100, AdDropReason::LowRiskNeighbour, 1, 0),
            (200, AdDropReason::ExcludedHandle, 2, 9999),
        ]
    );
}

#[test]
fn records_medium_risk_path_drops() {
    let mut posts: Vec<_> = (1..=8).map(make_avoid_post).collect();
    posts[0].author_id = 9999;

    let mut drops = Vec::new();
    let result = blend_with_drops(
        posts,
        vec![
            make_bsr_high_ad_with_handles(100, &[9999]),
            make_normal_ad(200),
        ],
        5,
        &mut drops,
    );

    assert_eq!(ad_count(&result), 0);
    assert_eq!(
        drop_summary(&drops),
        vec![
            (100, AdDropReason::ExcludedHandle, 1, 9999),
            (200, AdDropReason::NoEligibleNeighbours, 0, 0),
        ]
    );
}
