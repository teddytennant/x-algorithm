// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use arc_swap::ArcSwapOption;
use arrow::array::{Array, AsArray};
use arrow::datatypes::Int64Type;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::index::{IndexedPost, SidIndex, pack_full_key};
use crate::metrics;

const POST_ID_COLUMN: &str = "post_id";
const AUTHOR_ID_COLUMN: &str = "author_id";
const SID_COLUMN: &str = "post_sid";
const BATCH_SIZE: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub path: PathBuf,
    pub timestamp_secs: i64,
}

pub fn resolve_snapshot(path: &Path) -> Result<Snapshot> {
    let resolved = std::fs::canonicalize(path)
        .with_context(|| format!("failed to resolve snapshot {}", path.display()))?;
    let timestamp_secs = match timestamp_from_file_name(&resolved) {
        Some(ts) => ts,
        None => std::fs::metadata(&resolved)?
            .modified()?
            .duration_since(UNIX_EPOCH)?
            .as_secs() as i64,
    };
    Ok(Snapshot {
        path: resolved,
        timestamp_secs,
    })
}

fn timestamp_from_file_name(path: &Path) -> Option<i64> {
    let stem = path.file_stem()?.to_str()?;
    let (_, suffix) = stem.rsplit_once('_')?;
    suffix.parse().ok()
}

pub fn load_index(snapshot: &Snapshot) -> Result<SidIndex> {
    let file = File::open(&snapshot.path)
        .with_context(|| format!("failed to open {}", snapshot.path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let column_index = |name: &str| {
        builder
            .schema()
            .index_of(name)
            .map_err(|_| anyhow!("snapshot is missing column {name}"))
    };
    let roots = [
        column_index(POST_ID_COLUMN)?,
        column_index(AUTHOR_ID_COLUMN)?,
        column_index(SID_COLUMN)?,
    ];
    let expected_rows = builder.metadata().file_metadata().num_rows().max(0) as usize;
    let projection = ProjectionMask::roots(builder.parquet_schema(), roots);
    let reader = builder
        .with_projection(projection)
        .with_batch_size(BATCH_SIZE)
        .build()?;

    let mut posts = Vec::with_capacity(expected_rows);
    let mut skipped = 0usize;
    for batch in reader {
        let batch = batch?;
        let post_ids = batch
            .column_by_name(POST_ID_COLUMN)
            .and_then(|c| c.as_primitive_opt::<Int64Type>())
            .ok_or_else(|| anyhow!("{POST_ID_COLUMN} is not int64"))?;
        let author_ids = batch
            .column_by_name(AUTHOR_ID_COLUMN)
            .and_then(|c| c.as_primitive_opt::<Int64Type>())
            .ok_or_else(|| anyhow!("{AUTHOR_ID_COLUMN} is not int64"))?;
        let sids = batch
            .column_by_name(SID_COLUMN)
            .and_then(|c| c.as_list_opt::<i32>())
            .ok_or_else(|| anyhow!("{SID_COLUMN} is not a list"))?;
        let codes = sids
            .values()
            .as_primitive_opt::<Int64Type>()
            .ok_or_else(|| anyhow!("{SID_COLUMN} items are not int64"))?;
        let offsets = sids.value_offsets();
        for row in 0..batch.num_rows() {
            if post_ids.is_null(row) || sids.is_null(row) {
                skipped += 1;
                continue;
            }
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            let row_codes = &codes.values()[start..end];
            let Some(key) = pack_full_key(row_codes) else {
                skipped += 1;
                continue;
            };
            posts.push(IndexedPost {
                key,
                post_id: post_ids.value(row),
                author_id: if author_ids.is_null(row) {
                    0
                } else {
                    author_ids.value(row)
                },
            });
        }
    }
    Ok(SidIndex::build(posts, snapshot.timestamp_secs, skipped))
}

pub async fn watch_snapshots(
    path: PathBuf,
    poll_interval: Duration,
    index: Arc<ArcSwapOption<SidIndex>>,
    ready_tx: watch::Sender<bool>,
) {
    let mut loaded: Option<PathBuf> = None;
    loop {
        match resolve_snapshot(&path) {
            Ok(snapshot) if loaded.as_ref() != Some(&snapshot.path) => {
                let started = Instant::now();
                let to_load = snapshot.clone();
                match tokio::task::spawn_blocking(move || load_index(&to_load)).await {
                    Ok(Ok(new_index)) => {
                        let elapsed = started.elapsed().as_secs_f64();
                        metrics::record_index_load(&new_index, elapsed);
                        info!(
                            snapshot = %snapshot.path.display(),
                            snapshot_timestamp_secs = snapshot.timestamp_secs,
                            posts = new_index.len(),
                            skipped = new_index.skipped_posts(),
                            load_secs = elapsed,
                            "loaded SID index"
                        );
                        index.store(Some(Arc::new(new_index)));
                        loaded = Some(snapshot.path);
                        let _ = ready_tx.send(true);
                    }
                    Ok(Err(e)) => {
                        metrics::INDEX_LOADS.with_label_values(&["error"]).inc();
                        warn!(snapshot = %snapshot.path.display(), error = %e, "SID index load failed");
                    }
                    Err(e) => {
                        metrics::INDEX_LOADS.with_label_values(&["error"]).inc();
                        warn!(error = %e, "SID index load task failed");
                    }
                }
            }
            Ok(_) => {}
            Err(e) => {
                metrics::INDEX_LOADS
                    .with_label_values(&["resolve_error"])
                    .inc();
                warn!(error = %e, "failed to resolve SID snapshot");
            }
        }
        tokio::time::sleep(poll_interval).await;
    }
}
