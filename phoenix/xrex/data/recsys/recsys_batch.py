# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import enum
import logging
import time
import typing
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, TypeVar

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy import typing as npt

from xrex.data.recsys.ads_head_masking import build_trained_candidate_mask
from xrex.data.recsys.constants import (
    CONVERSION_DELAY_COLUMNS,
    CONVERSION_DELAY_NONE,
    CONVERSION_KEEP_APP,
    CONVERSION_KEEP_WEB,
    action_type_map,
)
from xrex.data.recsys.feature_config import (
    ADS_PRODUCT_KEY_HASH_BIAS,
    ADS_PRODUCT_KEY_HASH_BIAS_2,
    ADS_PRODUCT_KEY_HASH_MODULUS,
    ADS_PRODUCT_KEY_HASH_SCALE,
    ADS_PRODUCT_KEY_HASH_SCALE_2,
    ADS_PRODUCT_KEY_TABLE_SIZE,
    AUTHOR_NSFW_BIT,
    BOOL_FEATURES,
    CATEGORICAL_FEATURES,
    FLOAT_FEATURES,
    INT64_FEATURES,
    STALE_POST_14D_TTL_SEC,
    STALE_POST_30D_TTL_SEC,
    BoolFeature,
    CategoricalFeature,
    FloatFeature,
    Int64Feature,
    UserBoolFeature,
    UserCategoricalFeature,
    UserFloatFeature,
    UserInt64Feature,
)
from xrex.data.recsys.negative_word_match import clear_word_match, compute_word_match
from xrex.models.recsys_embedding import HashKeys, HashTable

if TYPE_CHECKING:
    from xrex.data.recsys.sequence_packing import (
        SequencePackedLayout,
    )

rank_logger = logging.getLogger("rank")
rank_logger.setLevel(logging.INFO)

TWITTER_EPOCH_MS = 1288834974657

EXPLORATION_REACH_TARGET_FRACTION = 0.03
EXPLORATION_WINDOW_HOURS = 24.0
EXPLORATION_HALF_LIFE_HOURS = 8.0


def compute_post_unexplored_labels(
    follower_counts: npt.NDArray[np.float64],
    view_counts: npt.NDArray[np.float64],
    post_creation_ts_sec: np.ndarray,
    impression_ts_sec: np.ndarray,
    in_reply_to_post_ids: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    is_original = in_reply_to_post_ids == 0
    has_valid_ts = (impression_ts_sec > 0) & (post_creation_ts_sec > 0)
    age_hours = (
        impression_ts_sec.astype(np.float64) - post_creation_ts_sec.astype(np.float64)
    ) / 3600.0
    in_exploration_window = has_valid_ts & (age_hours <= EXPLORATION_WINDOW_HOURS)
    decay = np.log(2.0) / EXPLORATION_HALF_LIFE_HOURS
    saturation = 1.0 - np.exp(-decay * EXPLORATION_WINDOW_HOURS)
    clamped_age = np.clip(age_hours, 0.0, EXPLORATION_WINDOW_HOURS)
    exploration_view_target = (
        EXPLORATION_REACH_TARGET_FRACTION
        * follower_counts
        * (1.0 - np.exp(-decay * clamped_age))
        / saturation
    )
    under_explored = view_counts < exploration_view_target
    return is_original & in_exploration_window & under_explored


EMBEDDING_CONFIG = {
    "v1": ("embeddingV1Seq", 1536),
    "v3": ("embeddingV3Seq", 1024),
    "v5": ("embeddingV5Seq", 1024),
    "v6": ("embeddingV6Seq", 1024),
    "v8": ("embeddingV8Seq", 1024),
}

EmbeddingType = Literal["v1", "v3", "v5", "v6", "v8"]


class CandidateNegativeFilter(enum.Enum):
    NONE = "none"
    VQV_DWELL_10S = "vqv_dwell_10s"


class CandidateNegativeMode(enum.Enum):
    FILTER = "filter"
    PARTITION = "partition"


def _apply_candidate_negative_filter(
    filter_type: CandidateNegativeFilter,
    actions: npt.NDArray[np.bool_],
    continuous_actions: npt.NDArray[np.float32],
) -> npt.NDArray[np.bool_]:
    if filter_type == CandidateNegativeFilter.VQV_DWELL_10S:
        vqv_idx = action_type_map["ClientTweetVideoQualityView"]
        dwell_idx = 1
        has_vqv = actions[:, vqv_idx]
        dwell_secs = continuous_actions[:, dwell_idx]
        return has_vqv & (dwell_secs >= 10.0)
    return np.ones(actions.shape[0], dtype=np.bool_)


POST_CATEGORICAL_FEATURE_SIZE = 0 if len(CategoricalFeature) == 0 else max(CategoricalFeature) + 1
POST_BOOL_FEATURE_SIZE = 0 if len(BoolFeature) == 0 else max(BoolFeature) + 1
POST_FLOAT_FEATURE_SIZE = 0 if len(FloatFeature) == 0 else max(FloatFeature) + 1
POST_INT64_FEATURE_SIZE = 0 if len(Int64Feature) == 0 else max(Int64Feature) + 1

_INT32_MAX = 2**31 - 1
_ENGAGEMENT_COUNT_INT64_INDICES = (
    Int64Feature.favCountSeq.value,
    Int64Feature.replyCountSeq.value,
    Int64Feature.repostCountSeq.value,
    Int64Feature.quoteCountSeq.value,
    Int64Feature.viewCountSeq.value,
)

POST_CATEGORICAL_FEATURE_LIST = CATEGORICAL_FEATURES
POST_BOOL_FEATURE_LIST = BOOL_FEATURES
POST_FLOAT_FEATURE_LIST = FLOAT_FEATURES
POST_INT64_FEATURE_LIST = INT64_FEATURES

USER_CATEGORICAL_FEATURE_SIZE = (
    0 if len(UserCategoricalFeature) == 0 else max(UserCategoricalFeature) + 1
)
USER_BOOL_FEATURE_SIZE = 0 if len(UserBoolFeature) == 0 else max(UserBoolFeature) + 1
USER_FLOAT_FEATURE_SIZE = 0 if len(UserFloatFeature) == 0 else max(UserFloatFeature) + 1
USER_INT64_FEATURE_SIZE = 0 if len(UserInt64Feature) == 0 else max(UserInt64Feature) + 1


NUM_USER_INSTALLED_APPS = 64


class PostEmbeddingTable:
    post_ids: np.ndarray
    embeddings: np.ndarray
    dim: int

    def __init__(self, directory: str):
        self.post_ids = np.load(f"{directory}/post_ids.npy", mmap_mode="r")
        self.embeddings = np.load(f"{directory}/embeddings.npy", mmap_mode="r")
        self.dim = self.embeddings.shape[1]
        rank_logger.info(
            f"PostEmbeddingTable: loaded {len(self.post_ids):,} embeddings "
            f"(dim={self.dim}) from {directory}"
        )

    def lookup(
        self, query_ids: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        flat = query_ids.ravel()
        idx = np.searchsorted(self.post_ids, flat)
        idx = np.clip(idx, 0, len(self.post_ids) - 1)
        found = self.post_ids[idx] == flat
        result = np.zeros((len(flat), self.dim), dtype=np.float32)
        if found.any():
            result[found] = self.embeddings[idx[found]]
        return result.reshape(*query_ids.shape, self.dim), found.reshape(query_ids.shape)


_T = TypeVar("_T", bound=np.generic)


def _list_values(col: pa.Array) -> pa.Array:
    if pa.types.is_fixed_size_list(col.type):
        list_size = col.type.list_size
        return col.values.slice(col.offset * list_size, len(col) * list_size)
    return col.values


def _nested_list_values(col: pa.Array) -> pa.Array:
    while (
        pa.types.is_fixed_size_list(col.type)
        or pa.types.is_list(col.type)
        or pa.types.is_large_list(col.type)
    ):
        if pa.types.is_fixed_size_list(col.type):
            col = _list_values(col)
        else:
            col = col.flatten()
    return col


def _col(
    rb: pa.RecordBatch,
    col: str,
    batch_size: int,
    t: type[_T],
) -> npt.NDArray[_T]:
    values = _list_values(rb.column(col))
    if values.null_count:
        values = values.fill_null(False if pa.types.is_boolean(values.type) else 0)
    arr = values.to_numpy(zero_copy_only=False).reshape(batch_size, -1).astype(t)
    return typing.cast(npt.NDArray[_T], arr)


def _col_null_filled(
    rb: pa.RecordBatch,
    col: str,
    batch_size: int,
    t: type[_T],
) -> npt.NDArray[_T]:
    values = _list_values(rb.column(col))
    if values.null_count:
        values = values.fill_null(0)
    arr = values.to_numpy(zero_copy_only=False).reshape(batch_size, -1).astype(t)
    return typing.cast(npt.NDArray[_T], arr)


def _bool_col_as_categorical(
    rb: pa.RecordBatch, col: str, batch_size: int
) -> npt.NDArray[np.int16]:
    values = _list_values(rb.column(col))
    valid = values.is_valid().to_numpy(zero_copy_only=False)
    filled = values.fill_null(False).to_numpy(zero_copy_only=False).astype(np.int16)
    arr = np.where(valid, filled + 1, 0).astype(np.int16).reshape(batch_size, -1)
    return typing.cast(npt.NDArray[np.int16], arr)


class PostSeq(TypedDict):
    impr_ts: npt.NDArray[np.int32] | None
    actions: npt.NDArray[np.bool_] | None
    post_hashes: npt.NDArray[np.int32]
    auth_hashes: npt.NDArray[np.int32]
    ip_hashes: npt.NDArray[np.int32]
    post_ids: npt.NDArray[np.int64] | None
    product_surface: npt.NDArray[np.int32]
    client_app_id: npt.NDArray[np.int32]
    continuous_actions: npt.NDArray[np.float32]
    promoted_ids: npt.NDArray[np.int64] | None
    line_item_objective: npt.NDArray[np.int16] | None
    trained_candidate_mask: NotRequired[npt.NDArray[np.bool_] | None]
    value_label_valid: NotRequired[npt.NDArray[np.bool_]]
    value_baseline_mean_usd: NotRequired[npt.NDArray[np.float32]]
    conversion_delay_ms: NotRequired[npt.NDArray[np.int32]]
    safety_label_mask: npt.NDArray[np.int64] | None
    embedding: npt.NDArray[np.float32] | jax.Array | None
    search_query_embeddings: npt.NDArray[np.float32] | None
    categorical_features: npt.NDArray[np.int16]
    int64_features: npt.NDArray[np.int64]
    float_features: npt.NDArray[np.float32]
    bool_features: npt.NDArray[np.bool_]
    post_creation_ts_sec: npt.NDArray[np.int32]
    post_sids: npt.NDArray[np.uint16] | None


def empty_feature_arrays(
    batch_size: int,
    seq_len: int,
) -> dict[str, np.ndarray]:
    return {
        "categorical_features": np.zeros(
            (batch_size, seq_len, POST_CATEGORICAL_FEATURE_SIZE), dtype=np.int16
        ),
        "bool_features": np.zeros((batch_size, seq_len, POST_BOOL_FEATURE_SIZE), dtype=np.bool_),
        "float_features": np.zeros(
            (batch_size, seq_len, POST_FLOAT_FEATURE_SIZE), dtype=np.float32
        ),
        "int64_features": np.zeros((batch_size, seq_len, POST_INT64_FEATURE_SIZE), dtype=np.int64),
    }


def empty_user_feature_arrays(batch_size: int) -> dict[str, typing.Any]:
    return {
        "user_categorical_features": np.zeros(
            (batch_size, USER_CATEGORICAL_FEATURE_SIZE), dtype=np.int16
        ),
        "user_bool_features": np.zeros((batch_size, USER_BOOL_FEATURE_SIZE), dtype=np.bool_),
        "user_float_features": np.zeros((batch_size, USER_FLOAT_FEATURE_SIZE), dtype=np.float32),
        "user_int64_features": np.zeros((batch_size, USER_INT64_FEATURE_SIZE), dtype=np.int64),
    }


def _extract_user_features(
    record_batch: pa.RecordBatch,
    batch_size: int,
) -> dict[str, typing.Any]:
    user_categorical = np.zeros((batch_size, USER_CATEGORICAL_FEATURE_SIZE), dtype=np.int16)
    user_bool = np.zeros((batch_size, USER_BOOL_FEATURE_SIZE), dtype=np.bool_)
    user_float = np.zeros((batch_size, USER_FLOAT_FEATURE_SIZE), dtype=np.float32)
    user_int64 = np.zeros((batch_size, USER_INT64_FEATURE_SIZE), dtype=np.int64)

    schema_names = record_batch.schema.names

    for feat in UserCategoricalFeature:
        if feat.name in schema_names:
            user_categorical[:, feat.value] = (
                record_batch.column(feat.name).to_numpy(zero_copy_only=False).astype(np.int16)
            )

    for feat in UserBoolFeature:
        if feat.name in schema_names:
            user_bool[:, feat.value] = (
                record_batch.column(feat.name).to_numpy(zero_copy_only=False).astype(np.bool_)
            )

    for feat in UserFloatFeature:
        if feat.name in schema_names:
            user_float[:, feat.value] = (
                record_batch.column(feat.name).to_numpy(zero_copy_only=False).astype(np.float32)
            )

    for feat in UserInt64Feature:
        if feat.name in schema_names:
            user_int64[:, feat.value] = (
                record_batch.column(feat.name).to_numpy(zero_copy_only=False).astype(np.int64)
            )

    return {
        "user_categorical_features": user_categorical,
        "user_bool_features": user_bool,
        "user_float_features": user_float,
        "user_int64_features": user_int64,
    }


class RecsysFeaturesBatch(TypedDict):
    user_hashes: np.ndarray
    user_ip_hashes: np.ndarray
    history_seq: PostSeq
    candidate_seq: PostSeq
    user_categorical_features: npt.NDArray[np.int16]
    user_bool_features: npt.NDArray[np.bool_]
    user_float_features: npt.NDArray[np.float32]
    user_int64_features: npt.NDArray[np.int64]
    user_installed_apps_multihot: npt.NDArray[np.bool_]
    num_positive_candidates: npt.NDArray[np.int32] | None
    sample_weights: NotRequired[npt.NDArray[np.float32] | None]
    sample_source: NotRequired[npt.NDArray[np.int8] | None]
    packing_layout: NotRequired[SequencePackedLayout | None]


LEXICAL_TEXT_COLUMN = "lexicalTextSeq"
LEXICAL_QUERY_COLUMN = "lexicalQuerySeq"
LEXICAL_AUTHOR_COLUMN = "lexicalAuthorSeq"

VALUE_LABEL_DTYPES = {
    "value_label_valid": np.bool_,
    "value_baseline_mean_usd": np.float32,
}


def _read_value_labels(
    record_batch: pa.RecordBatch, shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {
        key: np.zeros(shape, dtype=dtype) for key, dtype in VALUE_LABEL_DTYPES.items()
    }
    for key, column, arrow_type in (
        ("value_label_valid", "valueLabelValidSeq", pa.bool_()),
        ("value_baseline_mean_usd", "valueBaselineMeanUsdSeq", pa.float32()),
    ):
        if column not in record_batch.schema.names:
            continue
        col = record_batch.column(column)
        if col.type != pa.list_(arrow_type, shape[1]):
            raise ValueError(f"{column} must have type {pa.list_(arrow_type, shape[1])}")
        values = col.values.slice(col.offset * shape[1], shape[0] * shape[1])
        values = values.fill_null(False if key == "value_label_valid" else 0.0)
        labels[key] = (
            values.to_numpy(zero_copy_only=False).reshape(shape).astype(VALUE_LABEL_DTYPES[key])
        )
        labels[key][col.is_null().to_numpy(zero_copy_only=False)] = 0
    return labels


def _read_conversion_delays(record_batch: pa.RecordBatch, shape: tuple[int, int]) -> np.ndarray:
    rows, seq_len = shape
    out = np.full((rows, seq_len, len(CONVERSION_DELAY_COLUMNS)), CONVERSION_DELAY_NONE, np.int32)
    for k, column in enumerate(CONVERSION_DELAY_COLUMNS):
        if column not in record_batch.schema.names:
            continue
        col = record_batch.column(column)
        if col.type != pa.list_(pa.int64(), seq_len):
            raise ValueError(f"{column} must have type {pa.list_(pa.int64(), seq_len)}")
        values = col.values.slice(col.offset * seq_len, rows * seq_len)
        values = values.fill_null(CONVERSION_DELAY_NONE)
        delays = values.to_numpy(zero_copy_only=False).reshape(shape)
        out[:, :, k] = np.clip(delays, CONVERSION_DELAY_NONE, _INT32_MAX)
        out[col.is_null().to_numpy(zero_copy_only=False), :, k] = CONVERSION_DELAY_NONE
    return out


def empty_conversion_delays(batch_size: int, candidate_seq_len: int) -> np.ndarray:
    return np.full(
        (batch_size, candidate_seq_len, len(CONVERSION_DELAY_COLUMNS)),
        CONVERSION_DELAY_NONE,
        np.int32,
    )


def _extend_conversion_delays(post_seq: PostSeq, shape: tuple[int, int]) -> np.ndarray:
    out = empty_conversion_delays(*shape)
    values = post_seq.get("conversion_delay_ms")
    if isinstance(values, np.ndarray):
        out[:, : values.shape[1]] = values
    return out


def _scatter_candidate_slots(slots: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    rank = np.cumsum(candidate_mask, axis=1) - 1
    keep = candidate_mask & (rank < slots.shape[1])
    gathered = np.take_along_axis(slots, np.clip(rank, 0, slots.shape[1] - 1), axis=1)
    return np.where(keep, gathered, np.zeros((), dtype=slots.dtype))


def _read_string_slots(
    record_batch: pa.RecordBatch, column: str, candidate_mask: np.ndarray
) -> np.ndarray | None:
    if column not in record_batch.schema.names:
        return None
    col = record_batch.column(column)
    width = col.type.list_size
    values = col.values.slice(col.offset * width, candidate_mask.shape[0] * width)
    slots = np.asarray(values.to_numpy(zero_copy_only=False), dtype=object)
    return _scatter_candidate_slots(slots.reshape(candidate_mask.shape[0], width), candidate_mask)


def _extend_value_labels(post_seq: PostSeq, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    labels = {key: np.zeros(shape, dtype=dtype) for key, dtype in VALUE_LABEL_DTYPES.items()}
    for key in labels:
        values = post_seq.get(key)
        if isinstance(values, np.ndarray):
            labels[key][:, : values.shape[1]] = values
    return labels


def _extract_feature_columns(
    record_batch: pa.RecordBatch,
    feature_list: list[str],
    batch_size: int,
    dtype: type[np.generic],
) -> dict[str, npt.NDArray]:
    out: dict[str, npt.NDArray] = {}
    for col_name in feature_list:
        if col_name in record_batch.schema.names:
            out[col_name] = _col(record_batch, col_name, batch_size, dtype)
    return out


def _stack_features(
    raw_cols: dict[str, npt.NDArray],
    feature_enum: type[enum.IntEnum],
    indices: npt.NDArray[np.intp],
    user_idx: int,
    dest: npt.NDArray,
    dest_slice: tuple,
) -> None:
    for feat in feature_enum:
        if feat.name in raw_cols:
            dest[dest_slice + (feat.value,)] = raw_cols[feat.name][user_idx, indices]


def from_record_batch(
    record_batch: pa.RecordBatch,
    max_history_post_action_pairs: int,
    max_candidate_post_action_pairs: int,
    num_negatives_per_example: int,
    output_vocab_size: int,
    num_continuous_actions: int,
    hash_table: HashTable,
    include_candidate_post_ids: bool = False,
    num_global_negatives_per_example: int = 0,
    global_post_ids: np.ndarray | None = None,
    global_author_ids: np.ndarray | None = None,
    max_candidates_for_embeddings: int = 64,
    embedding_type: EmbeddingType | None = None,
    offline_embedding_table: PostEmbeddingTable | None = None,
    filter_candidates_require_embedding: bool = False,
    search_query_embedding_dim: int = 0,
    candidate_negative_filter: CandidateNegativeFilter | None = None,
    candidate_negative_mode: CandidateNegativeMode = CandidateNegativeMode.FILTER,
    global_post_sids: np.ndarray | None = None,
    sid_num_levels: int = 0,
    compute_post_unexplored_label: bool = False,
    zero_stale_post_14d_candidate_counts: bool = False,
    stale_post_30d: bool = False,
    search_negative_clear_word_match: bool = False,
    ads_head_masking: bool = False,
) -> RecsysFeaturesBatch:
    start = time.time()
    batch_size = record_batch.num_rows

    user_ids = record_batch.column("userId").to_numpy(zero_copy_only=False)
    tweet_ids = _col(record_batch, "tweetIdSeq", batch_size, np.int64)
    author_ids = _col(record_batch, "authorIdSeq", batch_size, np.int64)
    impression_timestamps_ms = _col(record_batch, "impressedTimeMsSeq", batch_size, np.int64)
    product_surface = _col(record_batch, "productSurfaceSeq", batch_size, np.int32)
    new_event_mask = _col(record_batch, "newEventMask", batch_size, np.bool_)
    padding_mask = _col(record_batch, "paddingMask", batch_size, np.bool_)
    promoted_ids = _col(record_batch, "promotedIdSeq", batch_size, np.int64)
    if "lineItemObjectiveSeq" in record_batch.schema.names:
        line_item_objective = _col(record_batch, "lineItemObjectiveSeq", batch_size, np.int16)
    else:
        line_item_objective = np.zeros_like(promoted_ids, dtype=np.int16)
    if "safetyLabelMaskSeq" in record_batch.schema.names:
        safety_label_mask = _col(record_batch, "safetyLabelMaskSeq", batch_size, np.int64)
    else:
        safety_label_mask = np.zeros_like(promoted_ids, dtype=np.int64)

    action_col = record_batch.column("actionNameMultiHotSeqSeq")
    assert action_col.type.value_type.list_size == output_vocab_size, (
        f"output_vocab_size ({output_vocab_size}) != record_batch action vocab size ({action_col.type.value_type.list_size})"
    )
    flat_action = _nested_list_values(action_col).to_numpy(zero_copy_only=False)
    actions = flat_action.reshape(batch_size, -1, output_vocab_size).astype(np.bool_)
    actions = typing.cast(npt.NDArray[np.bool_], actions)

    if "continuousActionValuesSeqSeq" in record_batch.schema.names:
        continuous_values_col = record_batch.column("continuousActionValuesSeqSeq")
        flat_values = _nested_list_values(continuous_values_col).to_numpy(zero_copy_only=False)
        assert continuous_values_col.type.value_type.list_size == num_continuous_actions, (
            f"num_continuous_actions ({num_continuous_actions}) != record_batch continuous action size ({continuous_values_col.type.value_type.list_size})"
        )
        continuous_actions = flat_values.reshape(batch_size, -1, num_continuous_actions).astype(
            np.float32
        )
    else:
        seq_len = actions.shape[1]
        continuous_actions = np.zeros(
            (batch_size, seq_len, num_continuous_actions), dtype=np.float32
        )

    value_labels = _read_value_labels(record_batch, (batch_size, actions.shape[1]))
    conversion_delays = _read_conversion_delays(record_batch, (batch_size, actions.shape[1]))
    lexical_strings = {
        column: slots
        for column in (LEXICAL_TEXT_COLUMN, LEXICAL_QUERY_COLUMN, LEXICAL_AUTHOR_COLUMN)
        if (slots := _read_string_slots(record_batch, column, new_event_mask & padding_mask))
        is not None
    }

    if "clientAppIdSeq" in record_batch.schema.names:
        client_app_id = _col(record_batch, "clientAppIdSeq", batch_size, np.int32)
    else:
        seq_len = actions.shape[1]
        client_app_id = np.zeros((batch_size, seq_len), dtype=np.int32)
    if "conversionKeepBits" in record_batch.schema.names:
        conversion_keep_bits = _col(record_batch, "conversionKeepBits", batch_size, np.uint8)
        null_rows = ~record_batch.column("conversionKeepBits").is_valid().to_numpy(
            zero_copy_only=False
        )
        conversion_keep_bits[null_rows] = CONVERSION_KEEP_WEB | CONVERSION_KEEP_APP
    else:
        conversion_keep_bits = None
    sample_source = (
        record_batch.column("sample_source")
        .fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.int8)
        .reshape(-1, 1)
        if "sample_source" in record_batch.schema.names
        else np.zeros((batch_size, 1), dtype=np.int8)
    )

    post_creation_ts_sec = (((tweet_ids >> 22) + TWITTER_EPOCH_MS) // 1000).astype(np.int32)
    post_creation_ts_sec = np.where(tweet_ids == 0, 0, post_creation_ts_sec)

    if compute_post_unexplored_label:
        post_unexplored_idx = action_type_map["ServerTweetPostUnexplored"]
        if post_unexplored_idx >= output_vocab_size:
            raise ValueError(
                f"compute_post_unexplored_label: ServerTweetPostUnexplored index "
                f"{post_unexplored_idx} >= output_vocab_size {output_vocab_size}"
            )
        gate_columns = ("authorFollowerCountSeq", "viewCountSeq", "inReplyToPostIdSeq")
        missing_columns = [c for c in gate_columns if c not in record_batch.schema.names]
        if missing_columns:
            raise ValueError(
                f"compute_post_unexplored_label: missing columns {missing_columns} in record batch"
            )
        post_unexplored = compute_post_unexplored_labels(
            follower_counts=_col_null_filled(
                record_batch, "authorFollowerCountSeq", batch_size, np.float64
            ),
            view_counts=_col_null_filled(record_batch, "viewCountSeq", batch_size, np.float64),
            post_creation_ts_sec=post_creation_ts_sec,
            impression_ts_sec=impression_timestamps_ms // 1000,
            in_reply_to_post_ids=_col_null_filled(
                record_batch, "inReplyToPostIdSeq", batch_size, np.int64
            ),
        )
        actions[:, :, post_unexplored_idx] = np.where(
            new_event_mask, post_unexplored, actions[:, :, post_unexplored_idx]
        )

    embedding_raw: npt.NDArray[np.float32] | None = None
    embedding_dim: int = 0

    if embedding_type is not None:
        col_name, embedding_dim = EMBEDDING_CONFIG[embedding_type]
        if col_name not in record_batch.schema.names:
            raise ValueError(
                f"Embedding column '{col_name}' (embedding_type={embedding_type}) "
                f"not found in record batch. Available columns: {record_batch.schema.names}"
            )
        try:
            emb_col = record_batch.column(col_name)
            flat_values = _nested_list_values(emb_col)
            embedding_raw = (
                flat_values.to_numpy(zero_copy_only=False)
                .reshape(batch_size, max_candidates_for_embeddings, embedding_dim)
                .astype(np.float32)
            )
        except Exception as e:
            raise ValueError(f"Failed to extract {col_name}: {e}") from e

    search_query_embeddings: np.ndarray | None = None
    if search_query_embedding_dim > 0 and "searchQueryEmbeddingSeq" in record_batch.schema.names:
        search_query_col = record_batch.column("searchQueryEmbeddingSeq")
        flat_search_query = _nested_list_values(search_query_col).to_numpy(zero_copy_only=False)
        max_candidates_in_data = search_query_col.type.list_size
        actual_embedding_dim = search_query_col.type.value_type.list_size
        assert actual_embedding_dim == search_query_embedding_dim, (
            f"search_query_embedding_dim ({search_query_embedding_dim}) != record_batch embedding dim ({actual_embedding_dim})"
        )
        search_query_embeddings = flat_search_query.reshape(
            batch_size, max_candidates_in_data, search_query_embedding_dim
        ).astype(np.float32)

    semantic_ids_full_raw: np.ndarray | None = None
    if "semanticIdSeq" in record_batch.schema.names:
        sid_col = record_batch.column("semanticIdSeq")
        flat_sid = _nested_list_values(sid_col).to_numpy(zero_copy_only=False)
        sid_seq_len = sid_col.type.list_size
        upstream_sid_dim = sid_col.type.value_type.list_size
        assert sid_seq_len == tweet_ids.shape[1], (
            f"semanticIdSeq seq_len ({sid_seq_len}) must match tweetIdSeq seq_len "
            f"({tweet_ids.shape[1]}) for 1:1 alignment"
        )
        semantic_ids_full_raw = flat_sid.reshape(batch_size, sid_seq_len, upstream_sid_dim).astype(
            np.int32
        )

    hist_raw_categorical = _extract_feature_columns(
        record_batch, POST_CATEGORICAL_FEATURE_LIST, batch_size, np.int16
    )
    hist_raw_bool = _extract_feature_columns(
        record_batch, POST_BOOL_FEATURE_LIST, batch_size, np.bool_
    )
    hist_raw_float = _extract_feature_columns(
        record_batch, POST_FLOAT_FEATURE_LIST, batch_size, np.float32
    )
    hist_raw_int64 = _extract_feature_columns(
        record_batch, POST_INT64_FEATURE_LIST, batch_size, np.int64
    )

    cand_raw_categorical = _extract_feature_columns(
        record_batch, POST_CATEGORICAL_FEATURE_LIST, batch_size, np.int16
    )
    cand_raw_bool = _extract_feature_columns(
        record_batch, POST_BOOL_FEATURE_LIST, batch_size, np.bool_
    )
    cand_raw_float = _extract_feature_columns(
        record_batch, POST_FLOAT_FEATURE_LIST, batch_size, np.float32
    )
    cand_raw_int64 = _extract_feature_columns(
        record_batch, POST_INT64_FEATURE_LIST, batch_size, np.int64
    )

    _timezone_enum = hist_raw_categorical.get(
        "timezoneSeq",
        np.zeros((batch_size, impression_timestamps_ms.shape[1]), dtype=np.int16),
    )
    _local_hour, _local_dow = compute_local_time_features(impression_timestamps_ms, _timezone_enum)
    hist_raw_categorical["localHourOfDaySeq"] = _local_hour
    hist_raw_categorical["localDayOfWeekSeq"] = _local_dow
    cand_raw_categorical["localHourOfDaySeq"] = _local_hour
    cand_raw_categorical["localDayOfWeekSeq"] = _local_dow

    _author_is_nsfw = ((safety_label_mask >> AUTHOR_NSFW_BIT) & 1).astype(np.int16)
    hist_raw_categorical["authorIsNsfwSeq"] = _author_is_nsfw
    cand_raw_categorical["authorIsNsfwSeq"] = _author_is_nsfw

    _exact_phrase_col = CategoricalFeature.exactPhraseSeq.name
    if _exact_phrase_col in record_batch.schema.names:
        _exact_phrase = _bool_col_as_categorical(record_batch, _exact_phrase_col, batch_size)
        hist_raw_categorical[_exact_phrase_col] = _exact_phrase
        cand_raw_categorical[_exact_phrase_col] = _exact_phrase

    apps_col = record_batch.column("installedAppsMultiHot")
    user_installed_apps_multihot = (
        _list_values(apps_col)
        .to_numpy(zero_copy_only=False)
        .reshape(batch_size, -1)
        .astype(np.bool_)
    )

    hist_shape_2d = (batch_size, max_history_post_action_pairs)
    hist_shape_3d = (batch_size, max_history_post_action_pairs, output_vocab_size)
    hist_shape_3d_continuous = (batch_size, max_history_post_action_pairs, num_continuous_actions)

    history_post_ids = np.zeros(hist_shape_2d, dtype=tweet_ids.dtype)
    history_auth_ids = np.zeros(hist_shape_2d, dtype=author_ids.dtype)
    history_impr_ts = np.zeros(hist_shape_2d, dtype=np.int32)
    history_product_surface = np.zeros(hist_shape_2d, dtype=np.int32)
    history_client_app_id = np.zeros(hist_shape_2d, dtype=np.int32)
    history_post_creation_ts_sec = np.zeros(hist_shape_2d, dtype=np.int32)
    history_safety_label_mask = np.zeros(hist_shape_2d, dtype=np.int64)
    history_actions = np.zeros(hist_shape_3d, dtype=actions.dtype)
    history_continuous_actions = np.zeros(hist_shape_3d_continuous, dtype=np.float32)

    hist_categorical_features = np.zeros(
        (batch_size, max_history_post_action_pairs, POST_CATEGORICAL_FEATURE_SIZE),
        dtype=np.int16,
    )
    hist_bool_features = np.zeros(
        (batch_size, max_history_post_action_pairs, POST_BOOL_FEATURE_SIZE), dtype=np.bool_
    )
    hist_float_features = np.zeros(
        (batch_size, max_history_post_action_pairs, POST_FLOAT_FEATURE_SIZE),
        dtype=np.float32,
    )
    hist_int64_features = np.zeros(
        (batch_size, max_history_post_action_pairs, POST_INT64_FEATURE_SIZE), dtype=np.int64
    )

    cand_shape_2d = (batch_size, max_candidate_post_action_pairs)
    cand_shape_3d = (batch_size, max_candidate_post_action_pairs, output_vocab_size)
    cand_shape_3d_continuous = (batch_size, max_candidate_post_action_pairs, num_continuous_actions)

    candidate_post_ids = np.zeros(cand_shape_2d, dtype=tweet_ids.dtype)
    candidate_auth_ids = np.zeros(cand_shape_2d, dtype=author_ids.dtype)
    candidate_impr_ts = np.zeros(cand_shape_2d, dtype=np.int32)
    candidate_product_surface = np.zeros(cand_shape_2d, dtype=np.int32)
    candidate_client_app_id = np.zeros(cand_shape_2d, dtype=np.int32)
    candidate_trained_actions_bits = np.full(
        cand_shape_2d, CONVERSION_KEEP_WEB | CONVERSION_KEEP_APP, dtype=np.uint8
    )
    candidate_value_labels: dict[str, np.ndarray] = {
        key: np.zeros(cand_shape_2d, dtype=dtype) for key, dtype in VALUE_LABEL_DTYPES.items()
    }
    candidate_conversion_delays = np.full(
        (*cand_shape_2d, len(CONVERSION_DELAY_COLUMNS)), CONVERSION_DELAY_NONE, dtype=np.int32
    )
    candidate_lexical_strings = {
        column: np.zeros(cand_shape_2d, dtype=object) for column in lexical_strings
    }
    candidate_post_creation_ts_sec = np.zeros(cand_shape_2d, dtype=np.int32)
    candidate_actions = np.zeros(cand_shape_3d, dtype=actions.dtype)
    candidate_continuous_actions = np.zeros(
        cand_shape_3d_continuous, dtype=continuous_actions.dtype
    )
    candidate_promoted_ids = np.zeros(cand_shape_2d, dtype=promoted_ids.dtype)
    candidate_line_item_objective = np.zeros(cand_shape_2d, dtype=np.int16)
    candidate_safety_label_mask = np.zeros(cand_shape_2d, dtype=np.int64)
    candidate_search_query_embeddings: np.ndarray | None = None
    if search_query_embedding_dim > 0:
        candidate_search_query_embeddings = np.zeros(
            (batch_size, max_candidate_post_action_pairs, search_query_embedding_dim),
            dtype=np.float32,
        )

    cand_categorical_features = np.zeros(
        (batch_size, max_candidate_post_action_pairs, POST_CATEGORICAL_FEATURE_SIZE),
        dtype=np.int16,
    )
    cand_bool_features = np.zeros(
        (batch_size, max_candidate_post_action_pairs, POST_BOOL_FEATURE_SIZE),
        dtype=np.bool_,
    )
    cand_float_features = np.zeros(
        (batch_size, max_candidate_post_action_pairs, POST_FLOAT_FEATURE_SIZE),
        dtype=np.float32,
    )
    cand_int64_features = np.zeros(
        (batch_size, max_candidate_post_action_pairs, POST_INT64_FEATURE_SIZE),
        dtype=np.int64,
    )

    history_post_sids: np.ndarray | None = (
        np.zeros((batch_size, max_history_post_action_pairs, sid_num_levels), dtype=np.uint16)
        if sid_num_levels > 0
        else None
    )
    candidate_post_sids: np.ndarray | None = (
        np.zeros((batch_size, max_candidate_post_action_pairs, sid_num_levels), dtype=np.uint16)
        if sid_num_levels > 0
        else None
    )

    candidate_embedding: npt.NDArray[np.float32] | None = None
    if embedding_type is not None and embedding_dim > 0:
        candidate_embedding = np.zeros(
            (batch_size, max_candidate_post_action_pairs, embedding_dim), dtype=np.float32
        )
        if embedding_raw is not None:
            copy_len = min(max_candidates_for_embeddings, max_candidate_post_action_pairs)
            candidate_embedding[:, :copy_len, :] = embedding_raw[:, :copy_len, :]

    for user_idx in range(batch_size):
        user_new_event_mask = new_event_mask[user_idx]
        user_padding_mask = padding_mask[user_idx]
        user_history_indices = np.where(~user_new_event_mask & user_padding_mask)[0]
        num_history = min(len(user_history_indices), max_history_post_action_pairs)

        if num_history > 0:
            hslice = (user_idx, slice(0, num_history))
            dslice = (user_idx, user_history_indices[-num_history:])

            history_post_ids[*hslice] = tweet_ids[*dslice]
            history_auth_ids[*hslice] = author_ids[*dslice]
            if history_post_sids is not None and semantic_ids_full_raw is not None:
                _src = semantic_ids_full_raw[*dslice]
                _w = min(_src.shape[-1], history_post_sids.shape[-1])
                history_post_sids[*hslice, :_w] = (_src[..., :_w] + 1).astype(np.uint16)
            ts_values = impression_timestamps_ms[*dslice] / 1000
            ts_values = np.nan_to_num(ts_values, nan=0.0, posinf=0.0, neginf=0.0)
            ts_values = np.clip(ts_values, -2147483648, 2147483647)
            history_impr_ts[*hslice] = ts_values.astype(np.int32)
            history_product_surface[*hslice] = product_surface[*dslice]
            history_client_app_id[*hslice] = client_app_id[*dslice]
            history_post_creation_ts_sec[*hslice] = post_creation_ts_sec[*dslice]
            history_safety_label_mask[*hslice] = safety_label_mask[*dslice]
            history_actions[*hslice, :] = actions[*dslice, :]
            history_continuous_actions[*hslice, :] = continuous_actions[*dslice, :]

            src_indices = user_history_indices[-num_history:]
            _stack_features(
                hist_raw_categorical,
                CategoricalFeature,
                src_indices,
                user_idx,
                hist_categorical_features,
                hslice,
            )
            _stack_features(
                hist_raw_bool,
                BoolFeature,
                src_indices,
                user_idx,
                hist_bool_features,
                hslice,
            )
            _stack_features(
                hist_raw_float,
                FloatFeature,
                src_indices,
                user_idx,
                hist_float_features,
                hslice,
            )
            _stack_features(
                hist_raw_int64,
                Int64Feature,
                src_indices,
                user_idx,
                hist_int64_features,
                hslice,
            )

    num_positive_per_user = np.full(batch_size, max_candidate_post_action_pairs, dtype=np.int32)

    for user_idx in range(batch_size):
        user_new_event_mask = new_event_mask[user_idx]
        user_padding_mask = padding_mask[user_idx]
        all_candidate_indices = np.where(user_new_event_mask & user_padding_mask)[0]

        if (
            candidate_negative_filter is not None
            and candidate_negative_filter != CandidateNegativeFilter.NONE
            and len(all_candidate_indices) > 0
        ):
            is_positive = _apply_candidate_negative_filter(
                candidate_negative_filter,
                actions[user_idx, all_candidate_indices, :],
                continuous_actions[user_idx, all_candidate_indices, :],
            )
            pos_indices = all_candidate_indices[is_positive]
            neg_indices = all_candidate_indices[~is_positive]
            if candidate_negative_mode == CandidateNegativeMode.FILTER:
                user_candidate_indices = pos_indices
            else:
                user_candidate_indices = np.concatenate([pos_indices, neg_indices])
            num_pos = min(len(pos_indices), max_candidate_post_action_pairs)
            num_positive_per_user[user_idx] = num_pos
        else:
            user_candidate_indices = all_candidate_indices

        num_candidates = min(len(user_candidate_indices), max_candidate_post_action_pairs)

        if num_candidates > 0:
            cslice = (user_idx, slice(0, num_candidates))
            dslice = (user_idx, user_candidate_indices[:num_candidates])

            candidate_post_ids[*cslice] = tweet_ids[*dslice]
            candidate_auth_ids[*cslice] = author_ids[*dslice]
            if candidate_post_sids is not None and semantic_ids_full_raw is not None:
                _src = semantic_ids_full_raw[*dslice]
                _w = min(_src.shape[-1], candidate_post_sids.shape[-1])
                candidate_post_sids[*cslice, :_w] = (_src[..., :_w] + 1).astype(np.uint16)
            ts_values = impression_timestamps_ms[*dslice] / 1000
            ts_values = np.nan_to_num(ts_values, nan=0.0, posinf=0.0, neginf=0.0)
            ts_values = np.clip(ts_values, -2147483648, 2147483647)
            candidate_impr_ts[*cslice] = ts_values.astype(np.int32)
            candidate_product_surface[*cslice] = product_surface[*dslice]
            candidate_client_app_id[*cslice] = client_app_id[*dslice]
            if conversion_keep_bits is not None:
                candidate_trained_actions_bits[*cslice] = conversion_keep_bits[*dslice]
            candidate_post_creation_ts_sec[*cslice] = post_creation_ts_sec[*dslice]
            candidate_actions[*cslice] = actions[*dslice, :]
            candidate_continuous_actions[*cslice] = continuous_actions[*dslice, :]
            for key, values in value_labels.items():
                candidate_value_labels[key][*cslice] = values[*dslice]
            candidate_conversion_delays[*cslice] = conversion_delays[*dslice, :]
            for column, values in lexical_strings.items():
                candidate_lexical_strings[column][*cslice] = values[*dslice]
            candidate_promoted_ids[*cslice] = promoted_ids[*dslice]
            candidate_line_item_objective[*cslice] = line_item_objective[*dslice]
            candidate_safety_label_mask[*cslice] = safety_label_mask[*dslice]
            if (
                candidate_search_query_embeddings is not None
                and search_query_embeddings is not None
            ):
                candidate_search_query_embeddings[user_idx, :num_candidates, :] = (
                    search_query_embeddings[user_idx, :num_candidates, :]
                )

            if (
                candidate_negative_filter is not None
                and candidate_negative_filter != CandidateNegativeFilter.NONE
            ):
                cand_offset = all_candidate_indices[0] if len(all_candidate_indices) > 0 else 0
                emb_indices = user_candidate_indices[:num_candidates] - cand_offset

                if candidate_embedding is not None and embedding_raw is not None:
                    candidate_embedding[user_idx, :num_candidates, :] = 0
                    valid_emb = emb_indices < max_candidates_for_embeddings
                    if valid_emb.any():
                        candidate_embedding[user_idx, np.where(valid_emb)[0], :] = embedding_raw[
                            user_idx, emb_indices[valid_emb], :
                        ]

                if (
                    candidate_search_query_embeddings is not None
                    and search_query_embeddings is not None
                ):
                    sqe_limit = search_query_embeddings.shape[1]
                    valid_sqe = emb_indices < sqe_limit
                    candidate_search_query_embeddings[user_idx, :num_candidates, :] = 0
                    if valid_sqe.any():
                        candidate_search_query_embeddings[user_idx, np.where(valid_sqe)[0], :] = (
                            search_query_embeddings[user_idx, emb_indices[valid_sqe], :]
                        )

            src_indices = user_candidate_indices[:num_candidates]
            _stack_features(
                cand_raw_categorical,
                CategoricalFeature,
                src_indices,
                user_idx,
                cand_categorical_features,
                cslice,
            )
            _stack_features(
                cand_raw_bool,
                BoolFeature,
                src_indices,
                user_idx,
                cand_bool_features,
                cslice,
            )
            _stack_features(
                cand_raw_float,
                FloatFeature,
                src_indices,
                user_idx,
                cand_float_features,
                cslice,
            )
            _stack_features(
                cand_raw_int64,
                Int64Feature,
                src_indices,
                user_idx,
                cand_int64_features,
                cslice,
            )

    if offline_embedding_table is not None:
        table_embs, found_mask = offline_embedding_table.lookup(candidate_post_ids)

        if filter_candidates_require_embedding:
            not_found = ~found_mask
            candidate_post_ids[not_found] = 0
            candidate_auth_ids[not_found] = 0
            candidate_impr_ts[not_found] = 0
            candidate_product_surface[not_found] = 0
            candidate_client_app_id[not_found] = 0
            candidate_post_creation_ts_sec[not_found] = 0
            candidate_actions[not_found] = 0
            candidate_continuous_actions[not_found] = 0
            for values in candidate_value_labels.values():
                values[not_found] = 0
            candidate_conversion_delays[not_found] = CONVERSION_DELAY_NONE
            candidate_promoted_ids[not_found] = 0
            candidate_line_item_objective[not_found] = 0
            candidate_safety_label_mask[not_found] = 0
            table_embs[not_found] = 0.0

        candidate_embedding = table_embs

    _ip_idx = Int64Feature.ipAddressSeq.value
    history_ip_ids = hist_int64_features[:, :, _ip_idx]
    candidate_ip_ids = cand_int64_features[:, :, _ip_idx]

    for _ec_idx in _ENGAGEMENT_COUNT_INT64_INDICES:
        np.clip(
            hist_int64_features[:, :, _ec_idx],
            0,
            _INT32_MAX,
            out=hist_int64_features[:, :, _ec_idx],
        )
        np.clip(
            cand_int64_features[:, :, _ec_idx],
            0,
            _INT32_MAX,
            out=cand_int64_features[:, :, _ec_idx],
        )

    _dpa_idx = Int64Feature.firstDpaProductKey.value

    def _hash_dpa_keys(keys: npt.NDArray[np.int64], scale: int, bias: int) -> npt.NDArray:
        hashed = (
            (keys * scale + bias) % ADS_PRODUCT_KEY_HASH_MODULUS % (ADS_PRODUCT_KEY_TABLE_SIZE - 1)
        ) + 1
        return np.where(keys == 0, 0, hashed)

    _dpa_idx2 = Int64Feature.firstDpaProductKeyHash2.value
    for _dpa_arr in (hist_int64_features, cand_int64_features):
        _dpa_keys = _dpa_arr[:, :, _dpa_idx].copy()
        _dpa_arr[:, :, _dpa_idx2] = _hash_dpa_keys(
            _dpa_keys, ADS_PRODUCT_KEY_HASH_SCALE_2, ADS_PRODUCT_KEY_HASH_BIAS_2
        )
        _dpa_arr[:, :, _dpa_idx] = _hash_dpa_keys(
            _dpa_keys, ADS_PRODUCT_KEY_HASH_SCALE, ADS_PRODUCT_KEY_HASH_BIAS
        )

    if zero_stale_post_14d_candidate_counts:
        ttl_sec = np.int64(STALE_POST_30D_TTL_SEC if stale_post_30d else STALE_POST_14D_TTL_SEC)
        original_age_sec = candidate_impr_ts.astype(
            np.int64
        ) - candidate_post_creation_ts_sec.astype(np.int64)
        stale_post_14d = (
            (candidate_impr_ts > 0)
            & (candidate_post_creation_ts_sec > 0)
            & (original_age_sec > ttl_sec)
        )
        for _ec_idx in _ENGAGEMENT_COUNT_INT64_INDICES:
            cand_int64_features[:, :, _ec_idx] = np.where(
                stale_post_14d, 0, cand_int64_features[:, :, _ec_idx]
            )
        if cand_bool_features.shape[2] > BoolFeature.isStalePost14d.value:
            cand_bool_features[:, :, BoolFeature.isStalePost14d.value] = stale_post_14d

    history_seq = PostSeq(
        impr_ts=history_impr_ts,
        actions=history_actions,
        post_hashes=hash_table.get_item_hash(history_post_ids),
        auth_hashes=hash_table.get_author_hash(history_auth_ids),
        ip_hashes=hash_table.get_ip_hash(history_ip_ids),
        product_surface=history_product_surface,
        client_app_id=history_client_app_id,
        post_ids=None,
        continuous_actions=history_continuous_actions,
        promoted_ids=None,
        line_item_objective=None,
        safety_label_mask=history_safety_label_mask,
        embedding=None,
        search_query_embeddings=None,
        categorical_features=hist_categorical_features,
        bool_features=hist_bool_features,
        float_features=hist_float_features,
        int64_features=hist_int64_features,
        post_creation_ts_sec=history_post_creation_ts_sec,
        post_sids=history_post_sids,
    )
    candidate_trained_mask = build_trained_candidate_mask(
        candidate_actions,
        sample_source,
        candidate_trained_actions_bits,
        ads_head_masking=ads_head_masking,
    )
    candidate_seq = PostSeq(
        impr_ts=candidate_impr_ts,
        actions=candidate_actions,
        post_hashes=hash_table.get_item_hash(candidate_post_ids),
        auth_hashes=hash_table.get_author_hash(candidate_auth_ids),
        ip_hashes=hash_table.get_ip_hash(candidate_ip_ids),
        product_surface=candidate_product_surface,
        client_app_id=candidate_client_app_id,
        post_ids=candidate_post_ids if include_candidate_post_ids else None,
        trained_candidate_mask=candidate_trained_mask,
        **candidate_value_labels,
        conversion_delay_ms=candidate_conversion_delays,
        continuous_actions=candidate_continuous_actions,
        promoted_ids=candidate_promoted_ids,
        line_item_objective=candidate_line_item_objective,
        safety_label_mask=candidate_safety_label_mask,
        embedding=candidate_embedding,
        search_query_embeddings=candidate_search_query_embeddings,
        categorical_features=cand_categorical_features,
        bool_features=cand_bool_features,
        float_features=cand_float_features,
        int64_features=cand_int64_features,
        post_creation_ts_sec=candidate_post_creation_ts_sec,
        post_sids=candidate_post_sids,
    )

    candidate_seq_with_negatives = apply_negative_sampling(
        user_ids,
        candidate_seq,
        num_negatives_per_example,
        batch_size,
        max_candidate_post_action_pairs,
        sample_source=sample_source,
        ads_head_masking=ads_head_masking,
        clear_negative_word_match=search_negative_clear_word_match,
        lexical_text=candidate_lexical_strings.get(LEXICAL_TEXT_COLUMN),
        lexical_query=candidate_lexical_strings.get(LEXICAL_QUERY_COLUMN),
        lexical_author=candidate_lexical_strings.get(LEXICAL_AUTHOR_COLUMN),
    )

    candidate_seq_with_negatives = apply_global_negative_sampling(
        global_post_ids,
        global_author_ids,
        hash_table,
        candidate_seq_with_negatives,
        num_global_negatives_per_example,
        batch_size,
        global_post_sids=global_post_sids,
        sample_source=sample_source,
        ads_head_masking=ads_head_masking,
    )

    user_ip_ids = np.zeros(batch_size, dtype=np.int64)
    for user_idx in range(batch_size):
        for ip_arr in (candidate_ip_ids, history_ip_ids):
            row = ip_arr[user_idx]
            nonzero = np.nonzero(row)[0]
            if len(nonzero) > 0:
                user_ip_ids[user_idx] = row[nonzero[-1]]
                break

    rank_logger.debug(f"from_record_batch took: {time.time() - start}")
    batch = RecsysFeaturesBatch(
        user_hashes=hash_table.get_user_hash(user_ids),
        user_ip_hashes=hash_table.get_ip_hash(user_ip_ids),
        history_seq=history_seq,
        candidate_seq=candidate_seq_with_negatives,
        **_extract_user_features(record_batch, batch_size),
        user_installed_apps_multihot=user_installed_apps_multihot,
        sample_weights=(
            record_batch.column("sampleWeight")
            .to_numpy(zero_copy_only=False)
            .astype(np.float32)
            .reshape(-1, 1)
            if "sampleWeight" in record_batch.schema.names
            else np.ones((batch_size, 1), dtype=np.float32)
        ),
        sample_source=sample_source,
    )
    batch["num_positive_candidates"] = (
        num_positive_per_user.reshape(-1, 1)
        if candidate_negative_filter is not None
        and candidate_negative_filter != CandidateNegativeFilter.NONE
        else None
    )
    return batch


def apply_negative_sampling(
    user_ids: npt.NDArray[np.int64],
    post_seq: PostSeq,
    num_negatives_per_example: int,
    batch_size: int,
    max_candidate_post_action_pairs: int,
    *,
    sample_source: npt.NDArray[np.integer] | None = None,
    ads_head_masking: bool = False,
    clear_negative_word_match: bool = False,
    lexical_text: np.ndarray | None = None,
    lexical_query: np.ndarray | None = None,
    lexical_author: np.ndarray | None = None,
) -> PostSeq:
    if num_negatives_per_example == 0:
        return post_seq

    impr_ts = post_seq["impr_ts"]
    actions = post_seq["actions"]
    post_hashes = post_seq["post_hashes"]
    auth_hashes = post_seq["auth_hashes"]
    ip_hashes = post_seq["ip_hashes"]
    post_ids = post_seq["post_ids"]
    product_surface = post_seq["product_surface"]
    client_app_id = post_seq["client_app_id"]
    trained_candidate_mask = post_seq.get("trained_candidate_mask")
    post_creation_ts_sec = post_seq["post_creation_ts_sec"]
    continuous_actions = post_seq["continuous_actions"]
    promoted_ids = post_seq["promoted_ids"]
    line_item_objective = post_seq["line_item_objective"]
    safety_label_mask = post_seq["safety_label_mask"]
    _emb = post_seq.get("embedding")
    embedding: npt.NDArray[np.float32] | None = (
        np.asarray(_emb, dtype=np.float32) if _emb is not None else None
    )
    search_query_embeddings = post_seq["search_query_embeddings"]
    categorical_features = post_seq["categorical_features"]
    bool_features = post_seq["bool_features"]
    float_features = post_seq["float_features"]
    int64_features = post_seq["int64_features"]
    _post_sids = post_seq.get("post_sids")

    if num_negatives_per_example >= batch_size:
        raise ValueError(
            f"num_negatives_per_example ({num_negatives_per_example}) must be less than batch_size ({batch_size})"
        )

    has_search_query = search_query_embeddings is not None
    num_neg_blocks = 2 if has_search_query else 1
    total_candidate_slots = max_candidate_post_action_pairs * (
        1 + num_neg_blocks * num_negatives_per_example
    )

    assert actions is not None, "actions cannot be None for negative sampling"
    assert impr_ts is not None, "impr_ts cannot be None for negative sampling"

    new_impr_ts = np.zeros((batch_size, total_candidate_slots), dtype=impr_ts.dtype)
    new_actions = np.zeros(
        (batch_size, total_candidate_slots, actions.shape[2]), dtype=actions.dtype
    )
    num_continuous_actions = continuous_actions.shape[2]
    new_continuous_actions = np.zeros(
        (batch_size, total_candidate_slots, num_continuous_actions), dtype=np.float32
    )
    new_post_hashes = np.zeros(
        (batch_size, total_candidate_slots, post_hashes.shape[2]), dtype=post_hashes.dtype
    )
    new_auth_hashes = np.zeros(
        (batch_size, total_candidate_slots, auth_hashes.shape[2]), dtype=auth_hashes.dtype
    )
    new_ip_hashes = np.zeros(
        (batch_size, total_candidate_slots, ip_hashes.shape[2]), dtype=ip_hashes.dtype
    )
    new_product_surface = np.zeros((batch_size, total_candidate_slots), dtype=product_surface.dtype)
    new_client_app_id = np.zeros((batch_size, total_candidate_slots), dtype=client_app_id.dtype)
    new_trained_mask = np.ones(
        (batch_size, total_candidate_slots, actions.shape[-1]), dtype=np.bool_
    )
    new_post_creation_ts_sec = np.zeros(
        (batch_size, total_candidate_slots), dtype=post_creation_ts_sec.dtype
    )

    new_categorical_features = np.zeros(
        (batch_size, total_candidate_slots, categorical_features.shape[2]),
        dtype=categorical_features.dtype,
    )
    new_bool_features = np.zeros(
        (batch_size, total_candidate_slots, bool_features.shape[2]), dtype=bool_features.dtype
    )
    new_float_features = np.zeros(
        (batch_size, total_candidate_slots, float_features.shape[2]), dtype=float_features.dtype
    )
    new_int64_features = np.zeros(
        (batch_size, total_candidate_slots, int64_features.shape[2]), dtype=int64_features.dtype
    )
    new_post_sids = (
        np.zeros((batch_size, total_candidate_slots, _post_sids.shape[-1]), dtype=np.uint16)
        if _post_sids is not None
        else None
    )
    positive_slice = slice(0, max_candidate_post_action_pairs)
    new_impr_ts[:, positive_slice] = impr_ts
    new_actions[:, positive_slice, :] = actions
    new_post_hashes[:, positive_slice, :] = post_hashes
    new_auth_hashes[:, positive_slice, :] = auth_hashes
    new_ip_hashes[:, positive_slice, :] = ip_hashes
    new_product_surface[:, positive_slice] = product_surface
    new_client_app_id[:, positive_slice] = client_app_id
    if trained_candidate_mask is not None:
        new_trained_mask[:, positive_slice] = trained_candidate_mask
    new_post_creation_ts_sec[:, positive_slice] = post_creation_ts_sec
    if new_post_sids is not None and _post_sids is not None:
        new_post_sids[:, positive_slice, :] = _post_sids
    new_continuous_actions[:, positive_slice, :] = continuous_actions
    if categorical_features.shape[2] > 0:
        new_categorical_features[:, positive_slice, :] = categorical_features
    if bool_features.shape[2] > 0:
        new_bool_features[:, positive_slice, :] = bool_features
    if float_features.shape[2] > 0:
        new_float_features[:, positive_slice, :] = float_features
    if int64_features.shape[2] > 0:
        new_int64_features[:, positive_slice, :] = int64_features

    if post_ids is not None:
        new_post_ids = np.zeros((batch_size, total_candidate_slots), dtype=post_ids.dtype)
        new_post_ids[:, positive_slice] = post_ids
    else:
        new_post_ids = None
    if promoted_ids is not None:
        new_promoted_ids = np.zeros((batch_size, total_candidate_slots), dtype=promoted_ids.dtype)
        new_promoted_ids[:, positive_slice] = promoted_ids
    else:
        new_promoted_ids = None
    if line_item_objective is not None:
        new_line_item_objective = np.zeros(
            (batch_size, total_candidate_slots), dtype=line_item_objective.dtype
        )
        new_line_item_objective[:, positive_slice] = line_item_objective
    else:
        new_line_item_objective = None
    if safety_label_mask is not None:
        new_safety_label_mask = np.zeros(
            (batch_size, total_candidate_slots), dtype=safety_label_mask.dtype
        )
        new_safety_label_mask[:, positive_slice] = safety_label_mask
    else:
        new_safety_label_mask = None
    if search_query_embeddings is not None:
        search_query_embedding_dim = search_query_embeddings.shape[2]
        new_search_query_embeddings = np.zeros(
            (batch_size, total_candidate_slots, search_query_embedding_dim), dtype=np.float32
        )
        new_search_query_embeddings[:, positive_slice, :] = search_query_embeddings
    else:
        new_search_query_embeddings = None

    new_embedding: npt.NDArray[np.float32] | None = None
    if embedding is not None:
        new_embedding = np.zeros(
            (batch_size, total_candidate_slots, embedding.shape[2]), dtype=embedding.dtype
        )
        new_embedding[:, positive_slice, :] = embedding

    all_indices = np.arange(batch_size)
    neg_user_indices = np.array(
        [
            np.random.choice(
                [i for i in all_indices if user_ids[i] != user_ids[user_idx]],
                size=num_negatives_per_example,
                replace=False,
            )
            for user_idx in range(batch_size)
        ]
    )

    if has_search_query:
        q = search_query_embeddings[:, 0, :]
        q_norm = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
        q_normed = q / q_norm
        query_dissimilar = np.ones((batch_size, num_negatives_per_example), dtype=np.bool_)
        for i in range(num_negatives_per_example):
            neg_q = q_normed[neg_user_indices[:, i]]
            cos_sim = (q_normed * neg_q).sum(axis=1)
            query_dissimilar[:, i] = cos_sim < 0.9
    else:
        query_dissimilar = None

    lexical = (
        (lexical_text, lexical_query, lexical_author)
        if clear_negative_word_match
        and lexical_text is not None
        and lexical_query is not None
        and lexical_author is not None
        else None
    )
    new_lexical_text = np.zeros((batch_size, total_candidate_slots), dtype=object)
    new_lexical_query = np.zeros((batch_size, total_candidate_slots), dtype=object)
    new_lexical_author = np.zeros((batch_size, total_candidate_slots), dtype=object)

    def _copy_neg_features(curr_user_idx, start_slot, end_slot, post_src, query_src):
        new_post_hashes[curr_user_idx, start_slot:end_slot, :] = post_hashes[post_src]
        new_auth_hashes[curr_user_idx, start_slot:end_slot, :] = auth_hashes[post_src]
        new_ip_hashes[curr_user_idx, start_slot:end_slot, :] = ip_hashes[post_src]
        new_impr_ts[curr_user_idx, start_slot:end_slot] = impr_ts[post_src]
        new_product_surface[curr_user_idx, start_slot:end_slot] = product_surface[post_src]
        new_client_app_id[curr_user_idx, start_slot:end_slot] = client_app_id[post_src]
        new_post_creation_ts_sec[curr_user_idx, start_slot:end_slot] = post_creation_ts_sec[
            post_src
        ]
        new_actions[
            curr_user_idx, start_slot:end_slot, action_type_map["ClientTweetRecapNotDwelled"]
        ] = 1
        if post_ids is not None and new_post_ids is not None:
            new_post_ids[curr_user_idx, start_slot:end_slot] = post_ids[post_src]
        if promoted_ids is not None and new_promoted_ids is not None:
            new_promoted_ids[curr_user_idx, start_slot:end_slot] = promoted_ids[post_src]
        if line_item_objective is not None and new_line_item_objective is not None:
            new_line_item_objective[curr_user_idx, start_slot:end_slot] = line_item_objective[
                post_src
            ]
        if safety_label_mask is not None and new_safety_label_mask is not None:
            new_safety_label_mask[curr_user_idx, start_slot:end_slot] = safety_label_mask[post_src]
        if embedding is not None and new_embedding is not None:
            new_embedding[curr_user_idx, start_slot:end_slot, :] = embedding[post_src]
        if search_query_embeddings is not None and new_search_query_embeddings is not None:
            new_search_query_embeddings[curr_user_idx, start_slot:end_slot, :] = (
                search_query_embeddings[query_src, 0:1, :]
            )
        if categorical_features.shape[2] > 0:
            new_categorical_features[curr_user_idx, start_slot:end_slot, :] = categorical_features[
                post_src
            ]
        if bool_features.shape[2] > 0:
            new_bool_features[curr_user_idx, start_slot:end_slot, :] = bool_features[post_src]
        if float_features.shape[2] > 0:
            new_float_features[curr_user_idx, start_slot:end_slot, :] = float_features[post_src]
        if int64_features.shape[2] > 0:
            new_int64_features[curr_user_idx, start_slot:end_slot, :] = int64_features[post_src]
        if new_post_sids is not None and _post_sids is not None:
            new_post_sids[curr_user_idx, start_slot:end_slot, :] = _post_sids[post_src]
        if lexical is not None:
            new_lexical_text[curr_user_idx, start_slot:end_slot] = lexical[0][post_src]
            new_lexical_query[curr_user_idx, start_slot:end_slot] = lexical[1][query_src, 0:1]
            new_lexical_author[curr_user_idx, start_slot:end_slot] = lexical[2][post_src]

    for i in range(num_negatives_per_example):
        neg_user_idx = neg_user_indices[:, i]
        start_slot = max_candidate_post_action_pairs * (i + 1)
        end_slot = start_slot + max_candidate_post_action_pairs

        if query_dissimilar is not None:
            row_mask = query_dissimilar[:, i]
            if not row_mask.any():
                continue
            neg_user_idx = neg_user_idx[row_mask]
        else:
            row_mask = None

        curr_user_idx = row_mask if row_mask is not None else slice(None)
        _copy_neg_features(
            curr_user_idx, start_slot, end_slot, post_src=neg_user_idx, query_src=curr_user_idx
        )

    if has_search_query:
        block2_start = max_candidate_post_action_pairs * (1 + num_negatives_per_example)
        for i in range(num_negatives_per_example):
            neg_user_idx = neg_user_indices[:, i]
            start_slot = block2_start + max_candidate_post_action_pairs * i
            end_slot = start_slot + max_candidate_post_action_pairs

            if query_dissimilar is not None:
                row_mask = query_dissimilar[:, i]
                if not row_mask.any():
                    continue
                neg_user_idx = neg_user_idx[row_mask]
            else:
                row_mask = None

            curr_user_idx = row_mask if row_mask is not None else slice(None)
            _copy_neg_features(
                curr_user_idx, start_slot, end_slot, post_src=curr_user_idx, query_src=neg_user_idx
            )

    negative_slots = new_post_hashes[:, :, 0] != 0
    negative_slots[:, :max_candidate_post_action_pairs] = False
    if has_search_query and clear_negative_word_match:
        clear_word_match(new_categorical_features, new_float_features, negative_slots)
    if has_search_query and lexical is not None:
        compute_word_match(
            new_categorical_features,
            new_float_features,
            negative_slots,
            new_lexical_text,
            new_lexical_query,
            new_lexical_author,
        )

    negative_slice = slice(max_candidate_post_action_pairs, total_candidate_slots)
    new_trained_mask[:, negative_slice] = build_trained_candidate_mask(
        new_actions[:, negative_slice].astype(np.bool_, copy=False),
        sample_source,
        np.full(
            (batch_size, total_candidate_slots - max_candidate_post_action_pairs),
            CONVERSION_KEEP_WEB | CONVERSION_KEEP_APP,
            dtype=np.uint8,
        ),
        ads_head_masking=ads_head_masking,
    )
    return PostSeq(
        impr_ts=new_impr_ts,
        actions=new_actions,
        post_hashes=new_post_hashes,
        auth_hashes=new_auth_hashes,
        ip_hashes=new_ip_hashes,
        product_surface=new_product_surface,
        client_app_id=new_client_app_id,
        post_ids=new_post_ids,
        trained_candidate_mask=new_trained_mask,
        **_extend_value_labels(post_seq, (batch_size, total_candidate_slots)),
        conversion_delay_ms=_extend_conversion_delays(
            post_seq, (batch_size, total_candidate_slots)
        ),
        continuous_actions=new_continuous_actions,
        promoted_ids=new_promoted_ids,
        line_item_objective=new_line_item_objective,
        safety_label_mask=new_safety_label_mask,
        embedding=new_embedding,
        search_query_embeddings=new_search_query_embeddings,
        categorical_features=new_categorical_features,
        bool_features=new_bool_features,
        float_features=new_float_features,
        int64_features=new_int64_features,
        post_creation_ts_sec=new_post_creation_ts_sec,
        post_sids=new_post_sids,
    )


def apply_global_negative_sampling(
    global_post_ids: npt.NDArray | None,
    global_author_ids: npt.NDArray | None,
    hash_table: HashTable,
    post_seq: PostSeq,
    num_global_negatives_per_example: int,
    batch_size: int,
    global_post_sids: npt.NDArray | None = None,
    *,
    sample_source: npt.NDArray[np.integer] | None = None,
    ads_head_masking: bool = False,
) -> PostSeq:
    if num_global_negatives_per_example == 0:
        return post_seq

    impr_ts = post_seq["impr_ts"]
    actions = post_seq["actions"]
    continuous_actions = post_seq["continuous_actions"]
    post_hashes = post_seq["post_hashes"]
    auth_hashes = post_seq["auth_hashes"]
    ip_hashes = post_seq["ip_hashes"]
    post_ids = post_seq["post_ids"]
    product_surface = post_seq["product_surface"]
    client_app_id = post_seq["client_app_id"]
    post_creation_ts_sec = post_seq["post_creation_ts_sec"]
    promoted_ids = post_seq["promoted_ids"]
    line_item_objective = post_seq["line_item_objective"]
    safety_label_mask = post_seq["safety_label_mask"]
    _emb = post_seq.get("embedding")
    embedding: npt.NDArray[np.float32] | None = (
        np.asarray(_emb, dtype=np.float32) if _emb is not None else None
    )
    search_query_embeddings = post_seq["search_query_embeddings"]
    categorical_features = post_seq["categorical_features"]
    bool_features = post_seq["bool_features"]
    float_features = post_seq["float_features"]
    int64_features = post_seq["int64_features"]
    assert global_post_ids is not None
    assert global_author_ids is not None

    original_candidate_slots = post_hashes.shape[1]
    expanded_candidate_slots = original_candidate_slots + num_global_negatives_per_example
    new_post_hashes = np.zeros(
        (batch_size, expanded_candidate_slots, post_hashes.shape[2]), dtype=post_hashes.dtype
    )
    new_auth_hashes = np.zeros(
        (batch_size, expanded_candidate_slots, auth_hashes.shape[2]), dtype=auth_hashes.dtype
    )
    new_ip_hashes = np.zeros(
        (batch_size, expanded_candidate_slots, ip_hashes.shape[2]), dtype=ip_hashes.dtype
    )
    new_product_surface = np.zeros(
        (batch_size, expanded_candidate_slots),
        dtype=product_surface.dtype,
    )
    new_client_app_id = np.zeros(
        (batch_size, expanded_candidate_slots),
        dtype=client_app_id.dtype,
    )
    _gn_trained_mask = post_seq.get("trained_candidate_mask")
    new_post_creation_ts_sec = np.zeros(
        (batch_size, expanded_candidate_slots),
        dtype=post_creation_ts_sec.dtype,
    )
    num_continuous_actions = continuous_actions.shape[2]
    new_continuous_actions = np.zeros(
        (batch_size, expanded_candidate_slots, num_continuous_actions),
        dtype=continuous_actions.dtype,
    )

    new_categorical_features = np.zeros(
        (batch_size, expanded_candidate_slots, categorical_features.shape[2]),
        dtype=categorical_features.dtype,
    )
    new_bool_features = np.zeros(
        (batch_size, expanded_candidate_slots, bool_features.shape[2]), dtype=bool_features.dtype
    )
    new_float_features = np.zeros(
        (batch_size, expanded_candidate_slots, float_features.shape[2]),
        dtype=float_features.dtype,
    )
    new_int64_features = np.zeros(
        (batch_size, expanded_candidate_slots, int64_features.shape[2]),
        dtype=int64_features.dtype,
    )

    original_slice = slice(0, original_candidate_slots)
    expanded_slice = slice(original_candidate_slots, expanded_candidate_slots)
    new_post_hashes[:, original_slice, :] = post_hashes
    new_auth_hashes[:, original_slice, :] = auth_hashes
    new_ip_hashes[:, original_slice, :] = ip_hashes
    new_product_surface[:, original_slice] = product_surface
    new_client_app_id[:, original_slice] = client_app_id
    new_post_creation_ts_sec[:, original_slice] = post_creation_ts_sec
    new_continuous_actions[:, original_slice, :] = continuous_actions
    if categorical_features.shape[2] > 0:
        new_categorical_features[:, original_slice, :] = categorical_features
    if bool_features.shape[2] > 0:
        new_bool_features[:, original_slice, :] = bool_features
    if float_features.shape[2] > 0:
        new_float_features[:, original_slice, :] = float_features
    if int64_features.shape[2] > 0:
        new_int64_features[:, original_slice, :] = int64_features

    _gn_post_sids = post_seq.get("post_sids")
    new_gn_post_sids: np.ndarray | None = None
    if _gn_post_sids is not None:
        new_gn_post_sids = np.zeros(
            (batch_size, expanded_candidate_slots, _gn_post_sids.shape[-1]), dtype=np.uint16
        )
        new_gn_post_sids[:, original_slice, :] = _gn_post_sids

    if post_ids is not None:
        new_post_ids = np.zeros((batch_size, expanded_candidate_slots), dtype=post_ids.dtype)
        new_post_ids[:, original_slice] = post_ids
    else:
        new_post_ids = None
    if promoted_ids is not None:
        new_promoted_ids = np.zeros(
            (batch_size, expanded_candidate_slots), dtype=promoted_ids.dtype
        )
        new_promoted_ids[:, original_slice] = promoted_ids
    else:
        new_promoted_ids = None
    if line_item_objective is not None:
        new_line_item_objective = np.zeros(
            (batch_size, expanded_candidate_slots), dtype=line_item_objective.dtype
        )
        new_line_item_objective[:, original_slice] = line_item_objective
    else:
        new_line_item_objective = None
    if safety_label_mask is not None:
        new_safety_label_mask = np.zeros(
            (batch_size, expanded_candidate_slots), dtype=safety_label_mask.dtype
        )
        new_safety_label_mask[:, original_slice] = safety_label_mask
    else:
        new_safety_label_mask = None
    if impr_ts is not None:
        new_impr_ts = np.zeros((batch_size, expanded_candidate_slots), dtype=impr_ts.dtype)
        new_impr_ts[:, original_slice] = impr_ts
    else:
        new_impr_ts = None
    if actions is not None:
        new_actions = np.zeros(
            (batch_size, expanded_candidate_slots, actions.shape[2]), dtype=actions.dtype
        )
        new_actions[:, original_slice, :] = actions
        new_actions[:, expanded_slice, action_type_map["ClientTweetRecapNotDwelled"]] = 1
        new_gn_trained_mask = np.ones(
            (batch_size, expanded_candidate_slots, actions.shape[-1]), dtype=np.bool_
        )
        if _gn_trained_mask is not None:
            new_gn_trained_mask[:, original_slice] = _gn_trained_mask
        new_gn_trained_mask[:, expanded_slice] = build_trained_candidate_mask(
            new_actions[:, expanded_slice].astype(np.bool_, copy=False),
            sample_source,
            np.full(
                (batch_size, num_global_negatives_per_example),
                CONVERSION_KEEP_WEB | CONVERSION_KEEP_APP,
                dtype=np.uint8,
            ),
            ads_head_masking=ads_head_masking,
        )
    else:
        new_actions = None
        new_gn_trained_mask = None
        if _gn_trained_mask is not None:
            new_gn_trained_mask = np.ones(
                (batch_size, expanded_candidate_slots, _gn_trained_mask.shape[-1]), dtype=np.bool_
            )
            new_gn_trained_mask[:, original_slice] = _gn_trained_mask
    if search_query_embeddings is not None:
        search_query_embedding_dim = search_query_embeddings.shape[2]
        new_search_query_embeddings = np.zeros(
            (batch_size, expanded_candidate_slots, search_query_embedding_dim), dtype=np.float32
        )
        new_search_query_embeddings[:, original_slice, :] = search_query_embeddings
        new_search_query_embeddings[:, expanded_slice, :] = search_query_embeddings[:, 0:1, :]
    else:
        new_search_query_embeddings = None

    new_embedding: npt.NDArray[np.float32] | None = None
    if embedding is not None:
        new_embedding = np.zeros(
            (batch_size, expanded_candidate_slots, embedding.shape[2]), dtype=embedding.dtype
        )
        new_embedding[:, original_slice, :] = embedding

    user_ctx = categorical_features[:, 0:1, :]
    num_cat = categorical_features.shape[2]
    global_ctx = np.broadcast_to(
        user_ctx, (batch_size, num_global_negatives_per_example, num_cat)
    ).copy()
    new_categorical_features[:, expanded_slice, :] = global_ctx

    random_negative_indices = np.random.choice(
        len(global_post_ids), size=(batch_size, num_global_negatives_per_example), replace=True
    )
    random_post_ids = global_post_ids[random_negative_indices]
    random_auth_ids = global_author_ids[random_negative_indices]
    random_post_hashes = hash_table.get_item_hash(random_post_ids)
    random_auth_hashes = hash_table.get_author_hash(random_auth_ids)
    new_post_hashes[:, expanded_slice] = random_post_hashes
    new_auth_hashes[:, expanded_slice] = random_auth_hashes
    if new_post_ids is not None:
        new_post_ids[:, expanded_slice] = random_post_ids

    if (
        new_gn_post_sids is not None
        and global_post_sids is not None
        and global_post_sids.shape[0] == len(global_post_ids)
    ):
        sampled_sids_raw = global_post_sids[random_negative_indices]
        target_dim = new_gn_post_sids.shape[-1]
        if sampled_sids_raw.shape[-1] >= target_dim:
            sampled_sids_raw = sampled_sids_raw[..., :target_dim]
        else:
            pad = target_dim - sampled_sids_raw.shape[-1]
            sampled_sids_raw = np.pad(
                sampled_sids_raw,
                ((0, 0), (0, 0), (0, pad)),
                constant_values=-1,
            )
        new_gn_post_sids[:, expanded_slice, :] = (sampled_sids_raw + 1).astype(np.uint16)

    new_product_surface[:, expanded_slice] = product_surface[:, 0:1]

    return PostSeq(
        impr_ts=new_impr_ts,
        actions=new_actions,
        post_hashes=new_post_hashes,
        auth_hashes=new_auth_hashes,
        ip_hashes=new_ip_hashes,
        product_surface=new_product_surface,
        client_app_id=new_client_app_id,
        post_ids=new_post_ids,
        trained_candidate_mask=new_gn_trained_mask,
        **_extend_value_labels(post_seq, (batch_size, expanded_candidate_slots)),
        conversion_delay_ms=_extend_conversion_delays(
            post_seq, (batch_size, expanded_candidate_slots)
        ),
        continuous_actions=new_continuous_actions,
        promoted_ids=new_promoted_ids,
        line_item_objective=new_line_item_objective,
        safety_label_mask=new_safety_label_mask,
        embedding=new_embedding,
        search_query_embeddings=new_search_query_embeddings,
        categorical_features=new_categorical_features,
        bool_features=new_bool_features,
        float_features=new_float_features,
        int64_features=new_int64_features,
        post_creation_ts_sec=new_post_creation_ts_sec,
        post_sids=new_gn_post_sids,
    )


_TZ_ENUM_TO_UTC_OFFSET: np.ndarray = np.array(
    [
        0.0,
        -5.0,
        -6.0,
        -8.0,
        -7.0,
        -3.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        3.0,
        3.0,
        9.0,
        5.5,
        8.0,
        9.0,
        7.0,
        7.0,
        4.0,
        3.0,
        8.0,
        8.0,
        11.0,
        -10.0,
        -7.0,
        -5.0,
        -6.0,
        -3.0,
        1.0,
        0.0,
    ],
    dtype=np.float64,
)


def compute_local_time_features(
    timestamp_ms: np.ndarray,
    timezone_enum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tz_clipped = np.clip(timezone_enum, 0, len(_TZ_ENUM_TO_UTC_OFFSET) - 1)
    offset_hours = _TZ_ENUM_TO_UTC_OFFSET[tz_clipped]
    local_epoch_sec = (timestamp_ms / 1000.0) + (offset_hours * 3600.0)

    local_sec_of_day = np.mod(local_epoch_sec, 86400.0)
    hour = np.clip((local_sec_of_day / 3600.0).astype(np.int16), 0, 23)

    local_days = np.floor(local_epoch_sec / 86400.0).astype(np.int64)
    dow = np.mod(local_days + 3, 7).astype(np.int16)

    valid = timestamp_ms > 0
    return (
        np.where(valid, hour + 1, 0).astype(np.int16),
        np.where(valid, dow + 1, 0).astype(np.int16),
    )


def main():
    path = "example_batch.parquet"
    pf = pq.ParquetFile(path)
    arrow_schema = pf.schema_arrow
    excluded_columns = ["productSurfaceSeq", "firstPageSeq"]
    valid_columns = [name for name in arrow_schema.names if name not in excluded_columns]

    it = pf.iter_batches(16, columns=valid_columns)
    rb = next(it)

    user_vocab_size = 100_000
    item_vocab_size = 100_000
    author_vocab_size = 100_000
    INPUT_VOCAB_K = 512
    output_vocab_k = 64

    input_vocab_size = (
        user_vocab_size + item_vocab_size + author_vocab_size + len(action_type_map) + 1
    )

    input_vocab_size = (input_vocab_size + INPUT_VOCAB_K - 1) // INPUT_VOCAB_K * INPUT_VOCAB_K
    output_vocab_size = (
        (len(action_type_map) + output_vocab_k - 1) // output_vocab_k * output_vocab_k
    )

    mparams = {}
    mparams["user_vocab_size"] = user_vocab_size
    mparams["item_vocab_size"] = item_vocab_size
    mparams["author_vocab_size"] = author_vocab_size

    mparams["input_vocab_size"] = input_vocab_size
    mparams["output_vocab_size"] = output_vocab_size
    mparams["num_continuous_actions"] = 8

    input_vocab_size = (
        user_vocab_size + item_vocab_size + author_vocab_size + len(action_type_map) + 1
    )

    input_vocab_size = (input_vocab_size + INPUT_VOCAB_K - 1) // INPUT_VOCAB_K * INPUT_VOCAB_K
    output_vocab_size = (
        (len(action_type_map) + output_vocab_k - 1) // output_vocab_k * output_vocab_k
    )

    mparams["user_hash_scales"] = [196_742_702, 1_852_108_266]
    mparams["user_biases"] = [193_5840_681, 16_7407_236]
    mparams["user_modulus"] = 2_859_568_897
    mparams["item_hash_scales"] = [2_161_410_491, 1_754_358_832]
    mparams["item_biases"] = [1_935_840_681, 167_407_236]
    mparams["item_modulus"] = 2_361_375_383
    mparams["author_hash_scales"] = [371_965_780, 328_930_218]
    mparams["author_biases"] = [139_686_260, 37_755_056]
    mparams["author_modulus"] = 631_860_353

    hash_table = HashTable(
        hash_keys=HashKeys(
            user_id_table_size=mparams["user_vocab_size"],
            user_hash_scales=mparams["user_hash_scales"],
            user_biases=mparams["user_biases"],
            user_modulus=mparams["user_modulus"],
            item_id_table_size=mparams["item_vocab_size"],
            item_hash_scales=mparams["item_hash_scales"],
            item_biases=mparams["item_biases"],
            item_modulus=mparams["item_modulus"],
            author_id_table_size=mparams["author_vocab_size"],
            author_hash_scales=mparams["author_hash_scales"],
            author_biases=mparams["author_biases"],
            author_modulus=mparams["author_modulus"],
        ),
        output_vocab_size=mparams["output_vocab_size"],
    )

    from_record_batch(
        rb,
        1000,
        64,
        4,
        mparams["output_vocab_size"],
        mparams["num_continuous_actions"],
        hash_table,
        include_candidate_post_ids=True,
    )


if __name__ == "__main__":
    main()
