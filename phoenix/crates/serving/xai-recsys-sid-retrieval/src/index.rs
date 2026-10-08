// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::collections::HashSet;

pub const SID_LEVELS: usize = 6;
const BITS_PER_LEVEL: usize = 8;
const MAX_CODE: i64 = 255;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedPost {
    pub key: u64,
    pub post_id: i64,
    pub author_id: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedKey {
    pub key: u64,
    pub valid_depth: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub post_id: i64,
    pub author_id: i64,
    pub seed_post_id: i64,
    pub shared_prefix_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetrieveParams {
    pub max_results: usize,
    pub max_per_seed: usize,
    pub min_prefix_depth: usize,
    pub max_prefix_depth: usize,
}

pub struct SidIndex {
    by_key: Vec<IndexedPost>,
    by_post: Vec<(i64, u64)>,
    snapshot_timestamp_secs: i64,
    skipped_posts: usize,
}

fn is_valid_code(code: i64) -> bool {
    (0..=MAX_CODE).contains(&code)
}

pub fn pack_full_key(codes: &[i64]) -> Option<u64> {
    if codes.len() != SID_LEVELS || !codes.iter().all(|&c| is_valid_code(c)) {
        return None;
    }
    Some(
        codes
            .iter()
            .fold(0u64, |key, &code| (key << BITS_PER_LEVEL) | code as u64),
    )
}

pub fn seed_key(codes: &[i32]) -> Option<SeedKey> {
    let valid_depth = codes
        .iter()
        .take(SID_LEVELS)
        .take_while(|&&c| is_valid_code(c as i64))
        .count();
    if valid_depth == 0 {
        return None;
    }
    let key = (0..SID_LEVELS).fold(0u64, |key, level| {
        let code = if level < valid_depth {
            codes[level] as u64
        } else {
            0
        };
        (key << BITS_PER_LEVEL) | code
    });
    Some(SeedKey { key, valid_depth })
}

pub fn codes_from_key(key: u64) -> Vec<i32> {
    (0..SID_LEVELS)
        .map(|level| {
            let shift = BITS_PER_LEVEL * (SID_LEVELS - 1 - level);
            ((key >> shift) & MAX_CODE as u64) as i32
        })
        .collect()
}

fn prefix_range(key: u64, depth: usize) -> (u64, u64) {
    let shift = BITS_PER_LEVEL * (SID_LEVELS - depth);
    let lo = (key >> shift) << shift;
    (lo, lo + (1u64 << shift))
}

impl SidIndex {
    pub fn build(
        mut posts: Vec<IndexedPost>,
        snapshot_timestamp_secs: i64,
        skipped_posts: usize,
    ) -> Self {
        posts.sort_unstable_by(|a, b| a.post_id.cmp(&b.post_id));
        posts.dedup_by_key(|p| p.post_id);
        let by_post = posts.iter().map(|p| (p.post_id, p.key)).collect();
        posts.sort_unstable_by(|a, b| a.key.cmp(&b.key).then(b.post_id.cmp(&a.post_id)));
        Self {
            by_key: posts,
            by_post,
            snapshot_timestamp_secs,
            skipped_posts,
        }
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn snapshot_timestamp_secs(&self) -> i64 {
        self.snapshot_timestamp_secs
    }

    pub fn skipped_posts(&self) -> usize {
        self.skipped_posts
    }

    pub fn lookup(&self, post_id: i64) -> Option<u64> {
        self.by_post
            .binary_search_by_key(&post_id, |&(id, _)| id)
            .ok()
            .map(|i| self.by_post[i].1)
    }

    fn key_range(&self, lo: u64, hi: u64) -> (usize, usize) {
        let start = self.by_key.partition_point(|p| p.key < lo);
        let end = self.by_key.partition_point(|p| p.key < hi);
        (start, end)
    }

    pub fn retrieve_for_seed(
        &self,
        seed_post_id: i64,
        seed: SeedKey,
        params: &RetrieveParams,
        excluded: &HashSet<i64>,
    ) -> Vec<Candidate> {
        let max_depth = params.max_prefix_depth.min(seed.valid_depth);
        let mut picked = Vec::new();
        if params.min_prefix_depth == 0 || max_depth < params.min_prefix_depth {
            return picked;
        }
        let mut consumed: Option<(usize, usize)> = None;
        for depth in (params.min_prefix_depth..=max_depth).rev() {
            let remaining = params.max_per_seed.saturating_sub(picked.len());
            if remaining == 0 {
                break;
            }
            let (lo, hi) = prefix_range(seed.key, depth);
            let (start, end) = self.key_range(lo, hi);
            let mut level: Vec<&IndexedPost> = match consumed {
                None => self.by_key[start..end].iter().collect(),
                Some((inner_start, inner_end)) => self.by_key[start..inner_start]
                    .iter()
                    .chain(self.by_key[inner_end..end].iter())
                    .collect(),
            };
            level.retain(|p| !excluded.contains(&p.post_id));
            if level.len() > remaining {
                level.select_nth_unstable_by(remaining, |a, b| b.post_id.cmp(&a.post_id));
                level.truncate(remaining);
            }
            level.sort_unstable_by(|a, b| b.post_id.cmp(&a.post_id));
            picked.extend(level.into_iter().map(|p| Candidate {
                post_id: p.post_id,
                author_id: p.author_id,
                seed_post_id,
                shared_prefix_depth: depth as u32,
            }));
            consumed = Some((start, end));
        }
        picked
    }

    pub fn retrieve(
        &self,
        seeds: &[(i64, SeedKey)],
        params: &RetrieveParams,
    ) -> (Vec<Candidate>, Vec<usize>) {
        let excluded: HashSet<i64> = seeds.iter().map(|&(id, _)| id).collect();
        let per_seed: Vec<Vec<Candidate>> = seeds
            .iter()
            .map(|&(id, key)| self.retrieve_for_seed(id, key, params, &excluded))
            .collect();
        interleave(per_seed, params.max_results)
    }
}

fn interleave(per_seed: Vec<Vec<Candidate>>, max_results: usize) -> (Vec<Candidate>, Vec<usize>) {
    let mut counts = vec![0usize; per_seed.len()];
    let mut cursors = vec![0usize; per_seed.len()];
    let mut emitted = HashSet::new();
    let mut out = Vec::new();
    while out.len() < max_results {
        let mut progressed = false;
        for (seed_idx, candidates) in per_seed.iter().enumerate() {
            if out.len() >= max_results {
                break;
            }
            while cursors[seed_idx] < candidates.len() {
                let candidate = candidates[cursors[seed_idx]];
                cursors[seed_idx] += 1;
                if emitted.insert(candidate.post_id) {
                    out.push(candidate);
                    counts[seed_idx] += 1;
                    progressed = true;
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    (out, counts)
}
