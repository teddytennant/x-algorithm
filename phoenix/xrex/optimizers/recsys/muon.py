# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from optax import contrib as optax_contrib

from optax._src import numerics as _optax_numerics
from optax.contrib._muon import _DEFAULT_NS_COEFFS, orthogonalize_via_newton_schulz

from xrex.models.model_utils import Parameter

_VECTOR_LEAF_NAMES = frozenset({"b", "bias"})


def _leaf_name(path) -> str:
    if not path:
        return ""
    last = path[-1]
    key = getattr(last, "key", None) or getattr(last, "name", None)
    return str(key).lower() if key is not None else ""


def _is_matrix_group(path, x, adam_patterns: tuple[str, ...]) -> bool:
    if not hasattr(x, "ndim") or x.ndim < 2:
        return False
    if _leaf_name(path) in _VECTOR_LEAF_NAMES:
        return False
    name = jax.tree_util.keystr(path).lower()
    return not any(s in name for s in adam_patterns)


def muon_dimension_numbers(
    params, adam_patterns: tuple[str, ...], split_patterns: tuple[str, ...] = ()
):
    assignments: list[tuple[str, str, tuple[int, ...]]] = []
    saw_masked = False

    def _label(path, p):
        nonlocal saw_masked
        name = jax.tree_util.keystr(path).lower()
        x = p.x if isinstance(p, Parameter) else p
        if not hasattr(x, "ndim"):
            saw_masked = True
            return None
        if isinstance(p, Parameter):
            assert p.weight_decay_mask == 0.0, (
                f"optim='muon' ignores weight_decay_mask, but {name} has "
                f"weight_decay_mask={p.weight_decay_mask}; use "
                "muon_matrix_weight_decay / adam_embedding_weight_decay."
            )
        is_muon = _is_matrix_group(path, x, adam_patterns)
        assignments.append(("muon" if is_muon else "adam", name, tuple(x.shape)))
        if not is_muon:
            return None
        return optax_contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)

    dim_nums = jax.tree.map_with_path(_label, params, is_leaf=lambda v: isinstance(v, Parameter))
    if not saw_masked and any(g == "muon" for g, _, _ in assignments):
        for pattern in split_patterns:
            assert any(g == "muon" and pattern in name for g, name, _ in assignments), (
                "muon_split_fused pattern %r matched no muon matrix" % pattern
            )
    return dim_nums


def _cast_muon_group(updates, adam_patterns: tuple[str, ...], dtype):
    def _cast(path, u):
        x = u.x if isinstance(u, Parameter) else u
        if not _is_matrix_group(path, x, adam_patterns):
            return u
        return jax.tree.map(lambda a: a.astype(dtype), u)

    return jax.tree.map_with_path(_cast, updates, is_leaf=lambda v: isinstance(v, Parameter))


def embedding_decay_mask(emb_patterns: tuple[str, ...]):
    def mask_fn(params):
        def _m(path, p):
            x = p.x if isinstance(p, Parameter) else p
            if not hasattr(x, "ndim"):
                return False
            name = jax.tree_util.keystr(path).lower()
            if "norm" in name or _leaf_name(path) in _VECTOR_LEAF_NAMES:
                return False
            return any(s in name for s in emb_patterns)

        return jax.tree.map_with_path(_m, params, is_leaf=lambda v: isinstance(v, Parameter))

    return mask_fn


def parse_split_fused(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        pattern, _, dim = item.partition(":")
        out[pattern.strip().lower()] = int(dim)
    return out


def _split_block_dim(path, split_spec: dict[str, int]):
    name = jax.tree_util.keystr(path).lower()
    for pattern, dim in split_spec.items():
        if pattern in name:
            return dim
    return None


def _shard_groups(mesh: Mesh, group_size: int) -> list[list[int]]:
    assert mesh.size % group_size == 0, (
        f"muon_ns_shard_group_size={group_size} must divide the mesh {dict(mesh.shape)}"
    )
    devices = mesh.devices.reshape(-1, group_size)
    host = lambda d: getattr(d, "slice_index", d.process_index)
    assert all(len({host(d) for d in row}) == 1 for row in devices), (
        "a Newton-Schulz shard group must be one host (its NVLink domain)"
    )
    return np.arange(mesh.size).reshape(-1, group_size).tolist()


def _newton_schulz(
    ns_fn, x: jax.Array, dim_num, mesh: Mesh | None, groups: list[list[int]] | None
) -> jax.Array:
    red, out = dim_num.reduction_axis % x.ndim, dim_num.output_axis % x.ndim
    batch_axes = [a for a in range(x.ndim) if a not in (red, out)]
    num_matrices = math.prod(x.shape[a] for a in batch_axes)
    if groups is None or num_matrices < len(groups[0]):
        return ns_fn(x, dimension_numbers=dim_num)
    members = len(groups[0])
    per_member = -(-num_matrices // members)

    perm = batch_axes + [red, out]
    xm = jnp.transpose(x, perm).reshape(num_matrices, x.shape[red], x.shape[out])
    xm = jnp.pad(xm, ((0, per_member * members - num_matrices), (0, 0), (0, 0)))
    dn = optax_contrib.MuonDimensionNumbers(reduction_axis=-2, output_axis=-1)

    def body(xm):
        member = jax.lax.axis_index(mesh.axis_names) % members
        mine = jax.lax.dynamic_slice_in_dim(xm, member * per_member, per_member)
        return jax.lax.all_gather(
            ns_fn(mine, dimension_numbers=dn),
            mesh.axis_names,
            axis=0,
            tiled=True,
            axis_index_groups=groups,
        )

    om = shard_map(body, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False)(xm)
    om = om[:num_matrices].reshape([x.shape[a] for a in perm])
    return jnp.transpose(om, np.argsort(perm))


def scale_by_muon_shaped(
    cfg,
    dim_nums_for_partition,
    adam_patterns: tuple[str, ...],
    split_spec: dict[str, int],
    mesh: Mesh | None = None,
) -> optax.GradientTransformation:
    beta = cfg.muon_beta
    ns_steps = cfg.muon_ns_steps
    preconditioning = cfg.muon_preconditioning
    consistent_rms = cfg.muon_consistent_rms
    ns_dtype = jnp.dtype(cfg.muon_ns_dtype) if cfg.muon_ns_dtype else None
    mu_dtype = ns_dtype or cfg.mu_dtype
    ns_coeffs = jnp.asarray(_DEFAULT_NS_COEFFS)

    ns_fn = functools.partial(
        orthogonalize_via_newton_schulz,
        ns_coeffs=ns_coeffs,
        ns_steps=ns_steps,
        preconditioning=preconditioning,
        eps=1e-8,
    )
    groups = (
        _shard_groups(mesh, cfg.muon_ns_shard_group_size)
        if cfg.muon_ns_shard_group_size > 1
        else None
    )

    def _orthogonalize(path, x, dim_num):
        block = _split_block_dim(path, split_spec)
        if block is not None:
            assert x.shape[-1] % block == 0, (
                f"muon_split_fused: block {block} does not divide output axis "
                f"{x.shape[-1]} of {jax.tree_util.keystr(path)}"
            )
        if block is not None and x.shape[-1] > block:
            n_blocks = x.shape[-1] // block
            xs = x.reshape(x.shape[:-1] + (n_blocks, block))
            dn = optax_contrib.MuonDimensionNumbers(reduction_axis=-3, output_axis=-1)
            o = _newton_schulz(ns_fn, xs, dn, mesh, groups)
            fan_in, fan_out = x.shape[-2], block
            o = o.reshape(x.shape)
        else:
            o = _newton_schulz(ns_fn, x, dim_num, mesh, groups)
            fan_in, fan_out = x.shape[-2], x.shape[-1]
        if consistent_rms is not None:
            return o * (jnp.sqrt(jnp.maximum(fan_in, fan_out)) * consistent_rms)
        return o * jnp.sqrt(jnp.maximum(1, fan_out / fan_in))

    def init_fn(params):
        mu = optax.tree.zeros_like(params, dtype=mu_dtype)
        return optax_contrib.MuonState(count=jnp.zeros([], jnp.int32), mu=mu, ns_coeffs=ns_coeffs)

    def update_fn(updates, state, params=None):
        del params
        dim_nums = dim_nums_for_partition(updates)
        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count_inc = _optax_numerics.safe_increment(state.count)
        mu_hat = jax.tree.map(
            lambda m, g: beta * m + (1 - beta) * g,
            optax.tree.bias_correction(mu, beta, _optax_numerics.safe_increment(count_inc)),
            optax.tree.bias_correction(updates, beta, count_inc),
        )
        new_updates = jax.tree_util.tree_map_with_path(
            lambda path, x, dn: _orthogonalize(path, x, dn),
            mu_hat,
            dim_nums,
            is_leaf=lambda v: isinstance(v, optax_contrib.MuonDimensionNumbers),
        )
        mu = optax.tree.cast(mu, mu_dtype)
        return new_updates, optax_contrib.MuonState(
            count=count_inc, mu=mu, ns_coeffs=state.ns_coeffs
        )

    return optax.GradientTransformation(init_fn, update_fn)


def make_muon_optimizer(cfg, mesh: Mesh | None = None):
    assert cfg.muon_ns_shard_group_size <= 1 or mesh is not None
    adam_patterns = tuple(s.strip().lower() for s in cfg.muon_adam_patterns.split(",") if s.strip())
    emb_patterns = tuple(
        s.strip().lower() for s in cfg.adam_embedding_decay_patterns.split(",") if s.strip()
    )
    ns_dtype = jnp.dtype(cfg.muon_ns_dtype) if cfg.muon_ns_dtype else None
    split_spec = parse_split_fused(getattr(cfg, "muon_split_fused", ""))
    dim_nums_fn = functools.partial(
        muon_dimension_numbers,
        adam_patterns=adam_patterns,
        split_patterns=tuple(split_spec),
    )

    _is_dim_nums = lambda x: isinstance(x, optax_contrib.MuonDimensionNumbers)

    def param_labels(params):
        dim_nums = dim_nums_fn(params)
        populate = lambda dn, x: jax.tree.map(lambda _: "muon" if dn is not None else "adam", x)
        return jax.tree.map(
            populate, dim_nums, params, is_leaf=lambda x: x is None or _is_dim_nums(x)
        )

    def dim_nums_for_partition(params):
        dim_nums = dim_nums_fn(params)
        mask = jax.tree.map(lambda label: label == "muon", param_labels(params))
        is_leaf = lambda x: (
            x is None or _is_dim_nums(x) or isinstance(x, optax.transforms.MaskedNode)
        )
        populate = lambda dn, submask: jax.tree.map(
            lambda m: dn if m else optax.transforms.MaskedNode(), submask
        )
        return jax.tree.map(populate, dim_nums, mask, is_leaf=is_leaf)

    def optimizer(learning_rate, b1, b2, weight_decay):
        del weight_decay
        mu_dtype = ns_dtype or cfg.mu_dtype
        muon_chain = optax.chain(
            scale_by_muon_shaped(cfg, dim_nums_for_partition, adam_patterns, split_spec, mesh),
            optax.add_decayed_weights(cfg.muon_matrix_weight_decay),
            optax.scale_by_learning_rate(learning_rate),
        )
        adam = optax.adamw(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=1e-8,
            eps_root=0.0,
            mu_dtype=mu_dtype,
            weight_decay=cfg.adam_embedding_weight_decay,
            mask=embedding_decay_mask(emb_patterns),
            nesterov=False,
        )
        base = optax.transforms.partition({"muon": muon_chain, "adam": adam}, param_labels)

        def wrapped_update(updates, state, params=None):
            if ns_dtype is not None:
                updates = _cast_muon_group(updates, adam_patterns, ns_dtype)
            return base.update(updates, state, params)

        return optax.GradientTransformation(base.init, wrapped_update)

    return optimizer
