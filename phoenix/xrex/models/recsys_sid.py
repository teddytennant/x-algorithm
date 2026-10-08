# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

import functools
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from numpy import typing as npt

from xrex.models.layers import get_parameter


@functools.lru_cache(maxsize=2)
def _load_sid_decoder(
    path: str,
) -> tuple[npt.NDArray, tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]]:
    import safetensors.numpy

    tensors = safetensors.numpy.load_file(path)
    return tensors["stages"], (
        tensors["dec_w0"],
        tensors["dec_b0"],
        tensors["dec_w1"],
        tensors["dec_b1"],
    )


def sid_prefix2_lookup(
    sids: jnp.ndarray,
    sid_codebook_size: int,
    embed_dim: int,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    name_prefix: str,
) -> jnp.ndarray:
    sids = sids.astype(jnp.int32)
    first, second = sids[..., 0], sids[..., 1]
    prefix2 = jnp.where((first > 0) & (second > 0), (first - 1) * sid_codebook_size + second, 0)
    table = get_parameter(
        f"{name_prefix}_sid_prefix2",
        [sid_codebook_size * sid_codebook_size + 1, embed_dim],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_out"),
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(embed_dim),
    )
    table = table.at[0].set(0.0)
    return jnp.take(table, prefix2, axis=0)


def sid_prefix3_lookup(
    sids: jnp.ndarray,
    sid_codebook_size: int,
    embed_dim: int,
    rows: int,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    name_prefix: str,
) -> jnp.ndarray:
    codes = sids[..., :3].astype(jnp.uint32)
    exact = ((codes[..., 0] - 1) * sid_codebook_size + (codes[..., 1] - 1)) * sid_codebook_size + (
        codes[..., 2] - 1
    )
    hashed = (exact * jnp.uint32(2654435761)) % jnp.uint32(rows) + 1
    prefix3 = jnp.where((codes > 0).all(axis=-1), hashed, 0).astype(jnp.int32)
    table = get_parameter(
        f"{name_prefix}_sid_prefix3",
        [rows + 1, embed_dim],
        dtype=jnp.float32,
        init=hk.initializers.VarianceScaling(1.0, mode="fan_out"),
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(embed_dim),
    )
    table = table.at[0].set(0.0)
    return jnp.take(table, prefix3, axis=0)


def reconstruct_entity_sid(
    sids: jnp.ndarray,
    target_dim: int,
    decoder_path: str,
    lr_multiplier_func: Callable[[int], float],
    embed_init_scale: float,
    fprop_dtype: jnp.dtype,
    name_prefix: str,
) -> jnp.ndarray:
    stages_np, (w0_np, b0_np, w1_np, b1_np) = _load_sid_decoder(decoder_path)
    num_levels, codebook_size, _ = stages_np.shape
    assert sids.shape[-1] == num_levels, (
        f"post_sids have {sids.shape[-1]} levels but the decoder at {decoder_path!r} "
        f"expects {num_levels}; set sid_num_levels to match the artifact."
    )

    stages = jnp.asarray(stages_np)
    w0, b0 = jnp.asarray(w0_np), jnp.asarray(b0_np)
    w1, b1 = jnp.asarray(w1_np), jnp.asarray(b1_np)

    codes = sids.astype(jnp.int32) - 1
    missing = sids[..., 0] == 0
    codes = jnp.clip(codes, 0, codebook_size - 1)
    quant = stages[jnp.arange(num_levels), codes].sum(-2)
    h = jax.nn.relu(quant @ w0 + b0)
    recon = h @ w1 + b1
    recon = recon * jax.lax.rsqrt((recon**2).sum(-1, keepdims=True) + 1e-12)
    recon = jnp.where(missing[..., None], 0.0, recon)
    recon = jax.lax.stop_gradient(recon)

    recon_dim = recon.shape[-1]
    embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
    proj = get_parameter(
        f"{name_prefix}_sid_recon_proj",
        [recon_dim, target_dim],
        dtype=jnp.float32,
        init=lambda shape, dtype: embed_init(list(reversed(shape)), dtype).T,
        pspec=P(None, None),
        lr_multiplier=lr_multiplier_func(recon_dim),
    )
    return jnp.dot(recon, proj).astype(fprop_dtype)
