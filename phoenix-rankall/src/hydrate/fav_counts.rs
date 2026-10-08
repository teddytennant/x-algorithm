use std::collections::HashMap;

use futures::{stream, StreamExt};
use serde_json::{json, Value};

use super::strato::{json_i64, StratoColumn};
use crate::processor::record::PostId;

pub const COLUMN: &str = "content_understanding/favoriteCountsBatched";

const IDS_PER_CALL: usize = 100;

pub struct FavCounts {
    column: StratoColumn,
    max_concurrent_calls: usize,
}

impl FavCounts {
    pub fn new(column: StratoColumn, max_concurrent_calls: usize) -> Self {
        Self {
            column,
            max_concurrent_calls,
        }
    }

    pub async fn fetch(&self, post_ids: &[PostId]) -> HashMap<PostId, i64> {
        let calls: Vec<_> = post_ids
            .chunks(IDS_PER_CALL)
            .map(|chunk| self.fetch_chunk(chunk))
            .collect();
        stream::iter(calls)
            .buffer_unordered(self.max_concurrent_calls)
            .fold(
                HashMap::with_capacity(post_ids.len()),
                |mut acc, part| async move {
                    acc.extend(part);
                    acc
                },
            )
            .await
    }

    async fn fetch_chunk(&self, post_ids: &[PostId]) -> HashMap<PostId, i64> {
        let ids: Vec<String> = post_ids.iter().map(ToString::to_string).collect();
        match self
            .column
            .fetch(&Value::Null, &json!({ "postIds": ids }))
            .await
        {
            Ok(response) => parse_fav_counts(&response["v"]),
            Err(_) => HashMap::new(),
        }
    }
}

fn parse_fav_counts(entries: &Value) -> HashMap<PostId, i64> {
    entries
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let post_id = json_i64(&entry["postId"]);
            (post_id > 0).then(|| (post_id, json_i64(&entry["favCount"]).max(0)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_response() {
        let entries = json!([
            {"postId": "101", "favCount": "42"},
            {"postId": "102", "favCount": "7"},
            {"postId": "103"}
        ]);
        let favs = parse_fav_counts(&entries);
        assert_eq!(favs.len(), 3);
        assert_eq!(favs[&101], 42);
        assert_eq!(favs[&102], 7);
        assert_eq!(favs[&103], 0);
    }
}
