# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import logging

import jax
import jax.numpy as jnp

_log = logging.getLogger(__name__)

TARGET = "xrex_mol_epilogue"
SUPPORTED_KL_H = ((4, 32), (8, 16), (8, 32), (16, 16))
MAX_USERS = 512

try:
    import xrex_cuda_kernels.mol_epilogue_api as mol_epilogue_api
except ModuleNotFoundError:
    mol_epilogue_api = None
    _log.warning(
        "xrex.cuda.mol_epilogue: xrex_cuda_kernels.mol_epilogue_api not installed; the MoL "
        "serving kernel is unavailable"
    )
else:
    jax.ffi.register_ffi_target(TARGET, fn=mol_epilogue_api.mol_epilogue(), platform="CUDA")


def available() -> bool:
    return mol_epilogue_api is not None and jax.default_backend() == "gpu"


def mol_epilogue(
    dots: jax.Array,
    norms: jax.Array,
    ubias_t: jax.Array,
    ibias: jax.Array,
    w1: jax.Array,
    b1: jax.Array,
    w2: jax.Array,
    b2: jax.Array,
    *,
    num_users: int,
    num_components: int,
) -> jax.Array:
    if not available():
        raise RuntimeError(
            f"CUDA FFI target {TARGET!r} is unavailable (xrex_cuda_kernels.mol_epilogue_api "
            f"installed: {mol_epilogue_api is not None}, backend: {jax.default_backend()})"
        )
    kl, hidden = w1.shape
    if dots.shape != (kl * num_users, dots.shape[1]):
        raise ValueError(f"dots must be [KL*B, M] with KL={kl}, B={num_users}; got {dots.shape}")
    if (kl, hidden) not in SUPPORTED_KL_H:
        raise ValueError(f"unsupported (KL, H)=({kl}, {hidden}); compiled: {SUPPORTED_KL_H}")
    if num_users > MAX_USERS:
        raise ValueError(f"num_users {num_users} exceeds the kernel limit {MAX_USERS}")
    call = jax.ffi.ffi_call(
        TARGET,
        jax.ShapeDtypeStruct((num_users, dots.shape[1]), jnp.float32),
        vmap_method="sequential",
    )
    return call(
        dots.astype(jnp.bfloat16),
        norms.astype(jnp.float32),
        ubias_t.astype(jnp.float32),
        ibias.astype(jnp.float32),
        w1.astype(jnp.float32),
        b1.reshape(-1).astype(jnp.float32),
        w2.astype(jnp.float32),
        b2.reshape(-1).astype(jnp.float32),
        num_users=num_users,
        num_components=num_components,
    )
