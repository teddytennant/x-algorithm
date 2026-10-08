# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec as P

from xrex.data.recsys.ads_head_masking import EARLY_RELABEL_STREAM_ID
from xrex.utils.sharding import with_sharding_constraint_unless_manual

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


def multihot_loss_weights(padding_mask: jax.Array, raw_weights: jax.Array | None) -> jax.Array:
    mask = padding_mask.astype(jnp.int32)
    return mask if raw_weights is None else mask * raw_weights


def multihot_loss_compute(
    logits: jax.Array,
    raw_targets: jax.Array,
    padding_mask: jax.Array,
    loss_mask: jax.Array,
    raw_weights: jax.Array | None = None,
    one_hot_targets_sharding=P(None),
    normalizer: jax.Array | None = None,
):
    logits = logits.astype(jnp.float32)

    one_hot_targets = with_sharding_constraint_unless_manual(raw_targets, one_hot_targets_sharding)

    assert logits.shape == one_hot_targets.shape

    mask_3d = jnp.expand_dims(padding_mask, axis=-1) * loss_mask
    mask = padding_mask.astype(jnp.int32)

    bce_per_element = optax.sigmoid_binary_cross_entropy(
        logits, one_hot_targets.astype(logits.dtype)
    )

    masked_bce = bce_per_element * mask_3d

    if raw_weights is not None:
        masked_bce = masked_bce * jnp.expand_dims(raw_weights, axis=-1)
    if normalizer is None:
        normalizer = jnp.sum(multihot_loss_weights(padding_mask, raw_weights))

    cross_entropy_loss = jnp.sum(masked_bce) / (normalizer + 1e-10)

    return (
        cross_entropy_loss,
        mask,
    )


def continuous_loss_weights(
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    mask_negatives: bool,
    raw_weights: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    loss_mask = valid_mask & (~negative_sample_mask) if mask_negatives else valid_mask
    return loss_mask, (loss_mask if raw_weights is None else loss_mask * raw_weights)


def continuous_loss_compute(
    gt_raw: jax.Array,
    pred_raw: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    norm_scale: float,
    loss_type: str = "mse",
    mask_negatives: bool = True,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    gt_raw = gt_raw.astype(jnp.float32)
    pred_raw = pred_raw.astype(jnp.float32)

    gt_clamped = jnp.clip(gt_raw, 0.0, norm_scale)
    gt_norm = gt_clamped / norm_scale
    pred_norm = pred_raw

    pred_in_original_units = pred_raw * norm_scale

    loss_mask, weights = continuous_loss_weights(
        valid_mask, negative_sample_mask, mask_negatives, raw_weights
    )
    if normalizer is None:
        normalizer = jnp.sum(weights)

    if loss_type == "mse":
        errors = (pred_norm - gt_norm) ** 2
    elif loss_type == "mae":
        errors = jnp.abs(pred_norm - gt_norm)
    elif loss_type == "huber":
        delta = 1.0
        abs_diff = jnp.abs(pred_norm - gt_norm)
        errors = jnp.where(abs_diff <= delta, 0.5 * abs_diff**2, delta * (abs_diff - 0.5 * delta))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    loss = jnp.sum(errors * weights) / jnp.maximum(normalizer, 1.0)

    return loss, gt_clamped, pred_in_original_units, loss_mask, errors


def cread_log_thresholds(
    norm_scale: float,
    log_min_threshold: float,
    num_thresholds: int,
) -> tuple[float, ...]:
    if num_thresholds < 2:
        raise ValueError(f"num_thresholds must be at least 2, got {num_thresholds}")
    if not 0.0 < log_min_threshold < norm_scale:
        raise ValueError(
            f"log_min_threshold ({log_min_threshold}) must be in (0, norm_scale={norm_scale})"
        )
    ratio = norm_scale / log_min_threshold
    thresholds = [
        log_min_threshold * ratio ** (k / (num_thresholds - 1)) for k in range(num_thresholds - 1)
    ]
    return tuple(thresholds) + (norm_scale,)


def cread_loss_compute(
    gt_raw: jax.Array,
    cread_logits: jax.Array,
    restored_pred: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    thresholds: tuple[float, ...],
    norm_scale: float,
    restoration_weight: float,
    mask_negatives: bool = True,
    sentinel_values: tuple[float, ...] = (),
    raw_weights: jax.Array | None = None,
    restoration_loss_type: str = "mae",
    huber_delta: float = 0.1,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    gt_raw = gt_raw.astype(jnp.float32)
    logits = cread_logits.astype(jnp.float32)
    restored = restored_pred.astype(jnp.float32)

    num_thresholds = logits.shape[-1]
    assert num_thresholds == len(thresholds), (
        f"cread_logits has {num_thresholds} thresholds, expected {len(thresholds)}: {thresholds}"
    )

    gt_clamped = jnp.clip(gt_raw, 0.0, norm_scale)

    threshold_arr = jnp.asarray(thresholds, dtype=jnp.float32)
    targets = (gt_raw[..., None] > threshold_arr).astype(jnp.float32)

    per_element_cls = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets), axis=-1)
    restore_residual = jnp.abs(restored - gt_clamped) / norm_scale
    if restoration_loss_type == "mae":
        per_element_restore = restore_residual
    elif restoration_loss_type == "huber":
        per_element_restore = jnp.where(
            restore_residual <= huber_delta,
            0.5 * restore_residual**2 / huber_delta,
            restore_residual - 0.5 * huber_delta,
        )
    else:
        raise ValueError(f"Unknown restoration_loss_type: {restoration_loss_type}")
    per_element_loss = per_element_cls + restoration_weight * per_element_restore

    if mask_negatives:
        loss_mask = valid_mask.astype(jnp.bool_) & (~negative_sample_mask)
    else:
        loss_mask = valid_mask.astype(jnp.bool_)
    for sentinel in sentinel_values:
        loss_mask = loss_mask & (gt_raw != sentinel)

    weights = loss_mask if raw_weights is None else loss_mask * raw_weights
    if normalizer is None:
        normalizer = jnp.sum(weights)
    num_loss_samples = jnp.maximum(normalizer, 1.0)
    cls_loss = jnp.sum(per_element_cls * weights) / num_loss_samples
    restoration_loss = jnp.sum(per_element_restore * weights) / num_loss_samples
    loss = cls_loss + restoration_weight * restoration_loss

    return loss, gt_clamped, restored, loss_mask, per_element_loss, cls_loss, restoration_loss


def purchase_value_valid_mask(
    label_valid: jax.Array,
    padding_mask: jax.Array,
    negative_sample_mask: jax.Array,
    sample_source: jax.Array,
    has_click: jax.Array,
    has_purchase: jax.Array,
    keeper_mask: jax.Array,
) -> jax.Array:
    return (
        label_valid.astype(jnp.bool_)
        & padding_mask.astype(jnp.bool_)
        & ~negative_sample_mask.astype(jnp.bool_)
        & (sample_source == EARLY_RELABEL_STREAM_ID)
        & has_click.astype(jnp.bool_)
        & has_purchase.astype(jnp.bool_)
        & keeper_mask.astype(jnp.bool_)
    )


def purchase_value_weights(
    raw_ratio: jax.Array,
    baseline_mean_usd: jax.Array,
    valid_mask: jax.Array,
    raw_weights: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    ratio = raw_ratio.astype(jnp.float32)
    baseline = baseline_mean_usd.astype(jnp.float32)
    valid = (
        valid_mask.astype(jnp.bool_)
        & jnp.isfinite(ratio)
        & (ratio > 0)
        & jnp.isfinite(baseline)
        & (baseline > 0)
    )
    weights = jnp.ones_like(ratio) if raw_weights is None else raw_weights.astype(jnp.float32)
    valid = valid & jnp.isfinite(weights) & (weights > 0)
    return valid, jnp.where(valid, weights, 0.0)


def purchase_value_loss_compute(
    raw_ratio: jax.Array,
    pred_ratio: jax.Array,
    baseline_mean_usd: jax.Array,
    valid_mask: jax.Array,
    delta: float = 1.0,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if not 0 < delta < float("inf"):
        raise ValueError("purchase value Huber delta must be finite and positive")
    ratio = raw_ratio.astype(jnp.float32)
    baseline = baseline_mean_usd.astype(jnp.float32)
    pred = pred_ratio.astype(jnp.float32)
    valid, weights = purchase_value_weights(raw_ratio, baseline_mean_usd, valid_mask, raw_weights)
    target = jnp.where(valid, ratio, 0.0)
    error = jnp.where(valid, jnp.where(valid, pred, 0.0) - target, 0.0)
    abs_error = jnp.abs(error)
    quadratic = jnp.minimum(abs_error, delta)
    errors = 0.5 * quadratic**2 + delta * (abs_error - quadratic)
    weight_sum = jnp.sum(weights)
    if normalizer is None:
        normalizer = weight_sum
    loss = jnp.sum(errors * weights) / jnp.where(normalizer > 0, normalizer, 1.0)
    usd_abs_error = abs_error * jnp.where(valid, baseline, 0.0)
    sums = jnp.stack(
        [
            jnp.sum(errors * weights),
            jnp.sum(abs_error * weights),
            jnp.sum(target * weights),
            jnp.sum(jnp.where(valid, baseline, 0.0) * weights),
            weight_sum,
            jnp.sum(valid).astype(jnp.float32),
            jnp.sum(jnp.abs(1.0 - target) * weights),
            jnp.sum(usd_abs_error * weights),
            jnp.sum(jnp.where(valid, pred, 0.0) * weights),
        ]
    )
    return loss, sums


def purchase_value_stats(sums: jax.Array) -> dict[str, jax.Array]:
    denominator = jnp.where(sums[4] > 0, sums[4], 1.0)
    return {
        "purchase-value_delayed_website_clicked-loss": sums[0] / denominator,
        "purchase-value_delayed_website_clicked-valid-count": sums[5],
        "purchase-value_delayed_website_clicked-weight-sum": sums[4],
        "purchase-value_delayed_website_clicked-ratio-mae": sums[1] / denominator,
        "purchase-value_delayed_website_clicked-target-ratio": sums[2] / denominator,
        "purchase-value_delayed_website_clicked-baseline-mean-usd": sums[3] / denominator,
        "purchase-value_delayed_website_clicked-usd-mae": sums[7] / denominator,
        "purchase-value_delayed_website_clicked-calib": sums[8] / (sums[2] + 1e-12),
    }


def purchase_value_smoothed_stats(
    sums: jax.Array,
    slice_count: jax.Array,
    rce_ema: dict[str, jax.Array],
    batch_size: jax.Array,
    smoothing_windows: tuple[int, ...],
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    batch_stat = jnp.concatenate([sums, slice_count.astype(jnp.float32)[None]])
    a = jnp.minimum(1.0, batch_size / jnp.array(smoothing_windows, dtype=jnp.float32))[:, None]
    old = jnp.stack(
        [
            rce_ema.get(f"purchase_value/{ws}", jnp.zeros((10,), dtype=jnp.float32))
            for ws in smoothing_windows
        ]
    )
    raw_updated = (1.0 - a) * old + a * batch_stat[None, :]
    updated = jnp.where(
        jnp.isnan(raw_updated),
        jnp.where(jnp.isnan(old), batch_stat[None, :], old),
        raw_updated,
    )
    weight = jnp.maximum(updated[:, 4], 1e-12)
    has_labels = updated[:, 5] > 0
    stats: dict[str, jax.Array] = {}
    new_ema: dict[str, jax.Array] = {}
    for i, ws in enumerate(smoothing_windows):
        new_ema[f"purchase_value/{ws}"] = updated[i]
        prefix = "purchase-value_delayed_website_clicked-smoothed"
        for name, col in (
            ("loss", 0),
            ("ratio-mae", 1),
            ("target-ratio", 2),
            ("baseline-mean-usd", 3),
            ("usd-mae", 7),
        ):
            stats[f"{prefix}-{name}-{ws}"] = jnp.where(
                has_labels[i], updated[i, col] / weight[i], 0.0
            )
        stats[f"{prefix}-mae-vs-prior-{ws}"] = jnp.where(
            updated[i, 6] > 0, updated[i, 1] / jnp.maximum(updated[i, 6], 1e-12), 0.0
        )
        stats[f"{prefix}-calib-{ws}"] = jnp.where(
            updated[i, 2] > 0, updated[i, 8] / jnp.maximum(updated[i, 2], 1e-12), 0.0
        )
        stats[f"{prefix}-ratio-valid-{ws}"] = updated[i, 5] / jnp.maximum(updated[i, 9], 1.0)
        stats[f"{prefix}-valid-count-{ws}"] = updated[i, 5]
    return stats, new_ema


def binary_threshold_loss_compute(
    gt_raw: jax.Array,
    logit: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    threshold: float,
    mask_negatives: bool = True,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    logit = logit.astype(jnp.float32)
    gt_binary = (gt_raw.astype(jnp.float32) > threshold).astype(jnp.float32)
    per_element_loss = optax.sigmoid_binary_cross_entropy(logit, gt_binary)

    loss_mask, weights = continuous_loss_weights(
        valid_mask, negative_sample_mask, mask_negatives, raw_weights
    )
    if normalizer is None:
        normalizer = jnp.sum(weights)
    loss = jnp.sum(per_element_loss * weights) / jnp.maximum(normalizer, 1.0)

    pred_prob = jax.nn.sigmoid(logit)
    return loss, gt_binary, pred_prob, loss_mask, per_element_loss


def tweedie_loss_compute(
    gt_raw: jax.Array,
    pred_raw: jax.Array,
    valid_mask: jax.Array,
    negative_sample_mask: jax.Array,
    p: float = 1.5,
    norm_scale: float = 300.0,
    mask_negatives: bool = True,
    raw_weights: jax.Array | None = None,
    normalizer: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    gt = jnp.clip(gt_raw.astype(jnp.float32), 0.0, norm_scale)
    pred = jnp.maximum(pred_raw.astype(jnp.float32), 1e-6)

    if abs(p - 1.0) < 1e-8:
        deviance = -gt * jnp.log(pred) + pred
    elif abs(p - 2.0) < 1e-8:
        deviance = gt / pred + jnp.log(pred)
    else:
        log_pred = jnp.log(pred)
        deviance = -gt * jnp.exp((1.0 - p) * log_pred) / (1.0 - p) + jnp.exp(
            (2.0 - p) * log_pred
        ) / (2.0 - p)

    loss_mask, weights = continuous_loss_weights(
        valid_mask, negative_sample_mask, mask_negatives, raw_weights
    )
    if normalizer is None:
        normalizer = jnp.sum(weights)
    loss = jnp.sum(deviance * weights) / jnp.maximum(normalizer, 1.0)

    return loss, gt, pred, loss_mask, deviance
