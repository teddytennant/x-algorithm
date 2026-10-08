# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from typing import Union

import jax
from jax.sharding import Mesh

AxisNames = Union[str, list[str], tuple[str, ...]]


def get_flatten_axis_index(axis_names: AxisNames):
    if isinstance(axis_names, list):
        axis_names = tuple(axis_names)
    return jax.lax.axis_index(axis_names)


def get_flatten_axis_size(axis_names: AxisNames) -> int:
    if isinstance(axis_names, str):
        return jax.lax.axis_size(axis_names)
    if len(axis_names) == 1:
        return jax.lax.axis_size(axis_names[0])
    size = 1
    for name in axis_names:
        size *= jax.lax.axis_size(name)
    return size


def get_flatten_replica_groups(mesh: Mesh, axis_names: tuple[str]) -> tuple[int, list[int]]:
    from jax._src.interpreters.pxla import axis_groups
    from jax._src.sharding_impls import AxisEnv

    env = AxisEnv(
        nreps=mesh.devices.size, names=tuple(mesh.axis_names), sizes=tuple(mesh.devices.shape)
    )
    groups = axis_groups(env, axis_names)
    group_size = len(groups[0]) if len(groups) > 0 else 0
    flatten = [int(x) for g in groups for x in g]

    return group_size, tuple(flatten)


def get_context_id(group_size: int, flatten_replicas: tuple[int, ...]) -> int:
    h = group_size
    for r in flatten_replicas:
        h ^= r + 0x9E3779B9 + (h << 6) + (h >> 2)

    return h % (2**31)
