use crate::ads::safe_gap_blender::*;
use crate::ads::util::*;
use crate::ads::AdsBlender;
use crate::params::RESULT_SIZE;
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_recsys_proto::AdIndexInfo;

fn make_post(tweet_id: u64) -> ScoredPost {
    ScoredPost {
        tweet_id,
        ..Default::default()
    }
}

fn make_avoid_post(tweet_id: u64) -> ScoredPost {
    ScoredPost {
        tweet_id,
        brand_safety_verdict: BrandSafetyVerdict::MediumRisk.into(),
        ..Default::default()
    }
}

fn make_ad(post_id: i64) -> AdIndexInfo {
    AdIndexInfo {
        post_id,
        ..Default::default()
    }
}

fn make_ad_at(post_id: i64, insert_position: i32) -> AdIndexInfo {
    AdIndexInfo {
        post_id,
        insert_position,
        ..Default::default()
    }
}

fn ad_count(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .count()
}

fn ad_positions(items: &[FeedItem]) -> Vec<i32> {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .map(|i| i.position)
        .collect()
}

#[test]
fn test_has_avoid_returns_false_for_safe_verdict() {
    let post = ScoredPost {
        brand_safety_verdict: BrandSafetyVerdict::Safe.into(),
        ..Default::default()
    };
    assert!(!has_avoid(&post));
}

#[test]
fn test_has_avoid_returns_false_for_limited_verdict() {
    let post = ScoredPost {
        brand_safety_verdict: BrandSafetyVerdict::LowRisk.into(),
        ..Default::default()
    };
    assert!(!has_avoid(&post));
}

#[test]
fn test_has_avoid_returns_false_for_default_verdict() {
    assert!(!has_avoid(&make_post(1)));
}

#[test]
fn test_has_avoid_returns_true_for_avoid_verdict() {
    assert!(has_avoid(&make_avoid_post(1)));
}

#[test]
fn test_find_safe_gaps_all_safe() {
    let posts = vec![make_post(1), make_post(2), make_post(3)];
    assert_eq!(find_safe_gaps(&posts), vec![1, 2]);
}

#[test]
fn test_find_safe_gaps_avoid_blocks_adjacent_gaps() {
    let posts = vec![make_post(1), make_avoid_post(2), make_post(3), make_post(4)];
    assert_eq!(find_safe_gaps(&posts), vec![3]);
}

#[test]
fn test_find_safe_gaps_all_avoid() {
    let posts = vec![make_avoid_post(1), make_avoid_post(2), make_avoid_post(3)];
    assert_eq!(find_safe_gaps(&posts), Vec::<usize>::new());
}

#[test]
fn test_find_safe_gaps_empty_posts() {
    assert_eq!(find_safe_gaps(&[]), Vec::<usize>::new());
}

#[test]
fn test_compute_spacing_standard() {
    let ads = vec![
        make_ad_at(1, 1),
        make_ad_at(2, 4),
        make_ad_at(3, 7),
        make_ad_at(4, 10),
        make_ad_at(5, 13),
    ];
    assert_eq!(
        compute_spacing(&ads),
        AdSpacing {
            requested: 3,
            min: 2
        }
    );
}

#[test]
fn test_compute_spacing_wide() {
    let ads = vec![
        make_ad_at(1, 1),
        make_ad_at(2, 8),
        make_ad_at(3, 15),
        make_ad_at(4, 22),
        make_ad_at(5, 29),
    ];
    assert_eq!(
        compute_spacing(&ads),
        AdSpacing {
            requested: 7,
            min: 4
        }
    );
}

#[test]
fn test_compute_spacing_single_ad_uses_default() {
    assert_eq!(compute_spacing(&[make_ad_at(1, 5)]), DEFAULT_SPACING);
}

#[test]
fn test_compute_spacing_empty_uses_default() {
    assert_eq!(compute_spacing(&[]), DEFAULT_SPACING);
}

#[test]
fn test_compute_spacing_same_positions_uses_default() {
    let ads = vec![make_ad(1), make_ad(2), make_ad(3)];
    assert_eq!(compute_spacing(&ads), DEFAULT_SPACING);
}

#[test]
fn test_compute_spacing_small_gap_falls_back_to_default() {
    let ads = vec![make_ad_at(1, 1), make_ad_at(2, 4), make_ad_at(3, 5)];
    assert_eq!(compute_spacing(&ads), DEFAULT_SPACING);
}

#[test]
fn test_compute_spacing_gap_of_two_falls_back_to_default() {
    let ads = vec![make_ad_at(1, 1), make_ad_at(2, 3), make_ad_at(3, 5)];
    assert_eq!(compute_spacing(&ads), DEFAULT_SPACING);
}

#[test]
fn test_compute_spacing_consecutive_positions_fall_back_to_default() {
    let ads = vec![make_ad_at(1, 1), make_ad_at(2, 2), make_ad_at(3, 3)];
    assert_eq!(compute_spacing(&ads), DEFAULT_SPACING);
}

#[test]
fn test_compute_spacing_gap_of_three_is_accepted() {
    let ads = vec![make_ad_at(1, 1), make_ad_at(2, 4), make_ad_at(3, 7)];
    assert_eq!(
        compute_spacing(&ads),
        AdSpacing {
            requested: 3,
            min: 2
        }
    );
}

#[test]
fn test_compute_spacing_unsorted_positions() {
    let ads = vec![make_ad_at(1, 15), make_ad_at(2, 1), make_ad_at(3, 8)];
    assert_eq!(
        compute_spacing(&ads),
        AdSpacing {
            requested: 7,
            min: 4
        }
    );
}

#[test]
fn test_find_best_gap_exact_match() {
    assert_eq!(find_best_gap(&[3, 5, 7, 9], 7, 5), Some((2, 7)));
}

#[test]
fn test_find_best_gap_prefers_closer_below() {
    assert_eq!(find_best_gap(&[5, 7, 10], 8, 5), Some((1, 7)));
}

#[test]
fn test_find_best_gap_prefers_closer_above() {
    assert_eq!(find_best_gap(&[5, 10], 8, 5), Some((1, 10)));
}

#[test]
fn test_find_best_gap_all_below_ideal() {
    assert_eq!(find_best_gap(&[5, 7, 9], 20, 5), Some((2, 9)));
}

#[test]
fn test_find_best_gap_all_above_ideal() {
    assert_eq!(find_best_gap(&[5, 7, 9], 3, 5), Some((0, 5)));
}

#[test]
fn test_find_best_gap_none_above_min() {
    assert_eq!(find_best_gap(&[1, 2, 3], 8, 5), None);
}

#[test]
fn test_find_best_gap_empty() {
    assert_eq!(find_best_gap(&[], 5, 3), None);
}

#[test]
fn test_assign_targets_ideal_spacing() {
    let s = AdSpacing {
        requested: 3,
        min: 2,
    };
    let gaps: Vec<usize> = (1..=9).collect();
    assert_eq!(assign_ads_to_gaps(&gaps, 3, &s, 1), vec![1, 4, 7]);
}

#[test]
fn test_assign_falls_back_to_min_when_ideal_blocked() {
    let s = AdSpacing {
        requested: 3,
        min: 2,
    };
    assert_eq!(
        assign_ads_to_gaps(&[1, 2, 3, 5, 6, 7, 8, 9], 3, &s, 1),
        vec![1, 3, 7]
    );
}

#[test]
fn test_assign_wider_spacing_targets_ideal() {
    let s = AdSpacing {
        requested: 7,
        min: 4,
    };
    let gaps: Vec<usize> = (1..=20).collect();
    assert_eq!(assign_ads_to_gaps(&gaps, 3, &s, 1), vec![1, 8, 15]);
}

#[test]
fn test_assign_limited_by_num_ads() {
    let s = AdSpacing {
        requested: 3,
        min: 2,
    };
    assert_eq!(assign_ads_to_gaps(&[1, 2, 3, 4], 1, &s, 1), vec![1]);
}

#[test]
fn test_assign_no_safe_gaps() {
    let s = AdSpacing {
        requested: 3,
        min: 2,
    };
    assert_eq!(assign_ads_to_gaps(&[], 3, &s, 1), Vec::<usize>::new());
}

#[test]
fn test_assign_sparse_gaps_beyond_ideal() {
    let s = AdSpacing {
        requested: 3,
        min: 2,
    };
    assert_eq!(assign_ads_to_gaps(&[1, 10, 20], 3, &s, 1), vec![1, 10, 20]);
}

#[test]
fn test_no_ads() {
    let blender = SafeGapAdsBlender::default();
    let posts = vec![make_post(1), make_post(2), make_post(3)];
    let result = blender.blend(posts, vec![], &mut Vec::new());

    assert_eq!(result.len(), 3);
    for (i, item) in result.iter().enumerate() {
        assert_eq!(item.position, i as i32);
        assert!(matches!(item.item, Some(feed_item::Item::Post(_))));
    }
}

#[test]
fn test_no_posts() {
    let blender = SafeGapAdsBlender::default();
    let result = blender.blend(vec![], vec![make_ad(100), make_ad(200)], &mut Vec::new());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_too_few_posts_skips_ads() {
    let blender = SafeGapAdsBlender::default();
    let posts: Vec<_> = (1..=4).map(make_post).collect();
    let result = blender.blend(posts, vec![make_ad(100), make_ad(200)], &mut Vec::new());

    assert_eq!(result.len(), 4);
    assert_eq!(ad_count(&result), 0);
}

#[test]
fn test_ads_placed_at_ideal_spacing() {
    let posts: Vec<_> = (1..=7).map(make_post).collect();
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 4)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(result.len(), 9);
    assert_eq!(ad_count(&result), 2);
    for (i, item) in result.iter().enumerate() {
        assert_eq!(item.position, i as i32);
    }
    assert_eq!(ad_positions(&result), vec![1, 5]);
}

#[test]
fn test_ad_never_at_position_zero() {
    let posts: Vec<_> = (1..=3).map(make_post).collect();
    let result = blend_impl(posts, vec![make_ad(100)], 0, RESULT_SIZE);

    assert_eq!(result.len(), 4);
    assert!(matches!(result[0].item, Some(feed_item::Item::Post(_))));
    assert!(matches!(result[1].item, Some(feed_item::Item::Ad(_))));
}

#[test]
fn test_ad_not_placed_next_to_avoid_post() {
    let posts = vec![make_post(1), make_avoid_post(2), make_post(3), make_post(4)];
    let result = blend_impl(posts, vec![make_ad(100)], 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 1);
    assert_eq!(ad_positions(&result), vec![3]);
}

#[test]
fn test_avoid_at_start_pushes_ad_down() {
    let posts = vec![make_avoid_post(1), make_post(2), make_post(3)];
    let result = blend_impl(posts, vec![make_ad(100)], 0, RESULT_SIZE);

    assert_eq!(ad_positions(&result), vec![2]);
}

#[test]
fn test_avoid_at_end_still_allows_earlier_ads() {
    let posts = vec![make_post(1), make_post(2), make_avoid_post(3)];
    let result = blend_impl(posts, vec![make_ad(100)], 0, RESULT_SIZE);

    assert_eq!(ad_positions(&result), vec![1]);
}

#[test]
fn test_all_posts_avoid_drops_all_ads() {
    let posts: Vec<_> = (1..=3).map(make_avoid_post).collect();
    let result = blend_impl(posts, vec![make_ad(100), make_ad(200)], 0, RESULT_SIZE);

    assert_eq!(result.len(), 3);
    assert_eq!(ad_count(&result), 0);
}

#[test]
fn test_ideal_spacing_with_many_ads() {
    let posts: Vec<_> = (1..=10).map(make_post).collect();
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 4), make_ad_at(300, 7)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(result.len(), 13);
    assert_eq!(ad_count(&result), 3);
    let positions = ad_positions(&result);
    assert_eq!(positions, vec![1, 5, 9]);

    for w in positions.windows(2) {
        let between = (w[1] - w[0] - 1) as usize;
        assert_eq!(between, 3, "expected 3 posts between ads, got {between}");
    }
}

#[test]
fn test_excess_ads_dropped_from_bottom() {
    let posts: Vec<_> = (1..=5).map(make_post).collect();
    let ads = vec![
        make_ad_at(100, 1),
        make_ad_at(200, 4),
        make_ad_at(300, 7),
        make_ad_at(400, 10),
    ];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 2);
    let placed_ids: Vec<i64> = result
        .iter()
        .filter_map(|item| match &item.item {
            Some(feed_item::Item::Ad(ad)) => Some(ad.post_id),
            _ => None,
        })
        .collect();
    assert_eq!(placed_ids, vec![100, 200]);
}

#[test]
fn test_two_avoid_posts_create_isolated_safe_zones() {
    let posts = vec![
        make_post(1),
        make_avoid_post(2),
        make_post(3),
        make_post(4),
        make_post(5),
        make_avoid_post(6),
        make_post(7),
        make_post(8),
    ];
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 4)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 2);
    assert_eq!(ad_positions(&result), vec![3, 8]);
}

#[test]
fn test_same_insert_positions_uses_default_spacing() {
    let posts: Vec<_> = (1..=7).map(make_post).collect();
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 1)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 2);
    assert_eq!(ad_positions(&result), vec![1, 5]);
}

#[test]
fn test_wide_spacing_targets_ideal_gap() {
    let posts: Vec<_> = (1..=20).map(make_post).collect();
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 8), make_ad_at(300, 15)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 3);
    let positions = ad_positions(&result);
    assert_eq!(positions, vec![1, 9, 17]);

    for w in positions.windows(2) {
        let between = (w[1] - w[0] - 1) as usize;
        assert_eq!(between, 7, "expected 7 posts between ads, got {between}");
    }
}

#[test]
fn test_avoid_expands_then_recovers_ideal() {
    let mut posts: Vec<_> = (1..=13).map(make_post).collect();
    posts[3] = make_avoid_post(4);

    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 4), make_ad_at(300, 7)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 3);
    let positions = ad_positions(&result);
    assert_eq!(positions, vec![1, 6, 9]);
}

#[test]
fn test_tight_insert_positions_fall_back_to_default_spacing() {
    let posts: Vec<_> = (1..=10).map(make_post).collect();
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 2), make_ad_at(300, 3)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 3);
    let positions = ad_positions(&result);
    assert_eq!(positions, vec![1, 5, 9]);

    for w in positions.windows(2) {
        let between = (w[1] - w[0] - 1) as usize;
        assert!(
            between >= 3,
            "only {between} posts between ads; expected at least 3"
        );
    }
}

#[test]
fn test_all_zero_insert_positions_fall_back_to_default_spacing() {
    let posts: Vec<_> = (1..=10).map(make_post).collect();
    let ads = vec![make_ad_at(100, 0), make_ad_at(200, 0), make_ad_at(300, 0)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 3);
    let positions = ad_positions(&result);
    assert_eq!(positions, vec![1, 4, 8]);
}

#[test]
fn test_gap_of_two_falls_back_to_default_spacing_e2e() {
    let posts: Vec<_> = (1..=10).map(make_post).collect();
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 3), make_ad_at(300, 5)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 3);
    let positions = ad_positions(&result);
    assert_eq!(positions, vec![1, 5, 9]);
}

#[test]
fn test_never_fewer_than_two_posts_between_ads() {
    let posts: Vec<_> = (1..=30).map(make_post).collect();
    let ads: Vec<_> = (0..10).map(|i| make_ad_at(100 + i, i as i32)).collect();

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    let positions = ad_positions(&result);
    for w in positions.windows(2) {
        let between = (w[1] - w[0] - 1) as usize;
        assert!(
            between >= 2,
            "only {between} posts between ads at positions {} and {}; \
             need at least 2",
            w[0],
            w[1]
        );
    }
}

#[test]
fn test_excess_ads_dropped_never_trailing() {
    let posts: Vec<_> = (1..=5).map(make_post).collect();
    let ads: Vec<_> = (0..10).map(|i| make_ad_at(100 + i, i as i32)).collect();

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert!(result.len() > 5);
    assert!(
        matches!(result.last().unwrap().item, Some(feed_item::Item::Post(_))),
        "last item must be a post, not an ad"
    );
}

#[test]
fn test_avoid_posts_dont_inflate_spacing() {
    let posts = vec![
        make_avoid_post(1),
        make_avoid_post(2),
        make_post(3),
        make_avoid_post(4),
        make_avoid_post(5),
        make_post(6),
        make_post(7),
        make_post(8),
        make_post(9),
        make_post(10),
        make_avoid_post(11),
    ];
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 9), make_ad_at(300, 17)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    assert_eq!(ad_count(&result), 2);
    assert_eq!(ad_positions(&result), vec![6, 10]);
}

#[test]
fn test_positions_are_sequential() {
    let posts: Vec<_> = (1..=10).map(make_post).collect();
    let ads = vec![make_ad_at(100, 1), make_ad_at(200, 4)];

    let result = blend_impl(posts, ads, 0, RESULT_SIZE);

    for (i, item) in result.iter().enumerate() {
        assert_eq!(item.position, i as i32);
    }
}
