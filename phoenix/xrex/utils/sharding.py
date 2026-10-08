# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from collections.abc import Callable
from typing import Any, TypeVar

import jax
from jax.sharding import AxisType, Mesh

T = TypeVar("T")


def with_sharding_constraint_unless_manual(x: T, shardings) -> T:
    context_mesh = jax.sharding.get_abstract_mesh()
    manual = {
        axis
        for axis, kind in zip(context_mesh.axis_names, context_mesh.axis_types)
        if kind == AxisType.Manual
    }

    if not manual:
        return jax.lax.with_sharding_constraint(x, shardings)

    assert len(manual) == len(context_mesh.axis_names), "partially manual region"
    return x


def shard_map_unless_manual(
    f: Callable[..., Any],
    /,
    *,
    out_specs,
    in_specs,
    mesh: Mesh,
    check_vma: bool = True,
) -> Callable[..., Any]:
    context_mesh = jax.sharding.get_abstract_mesh()
    manual = {
        axis
        for axis, kind in zip(context_mesh.axis_names, context_mesh.axis_types)
        if kind == AxisType.Manual
    }

    if not manual:
        return jax.shard_map(
            f, out_specs=out_specs, in_specs=in_specs, mesh=mesh, check_vma=check_vma
        )

    assert len(manual) == len(context_mesh.axis_names), "partially manual region"
    return f
