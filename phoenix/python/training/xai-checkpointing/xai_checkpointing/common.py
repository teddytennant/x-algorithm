# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import functools
import os
import queue
import time
from typing import TypeVar

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

T = TypeVar("T")
PyTree = dict[str, "PyTree[T]"] | list["PyTree[T]"] | T | None


@functools.cache
def _affinity_numa_node() -> int | None:
    try:
        cpus = os.sched_getaffinity(0)
        nodes = set()
        for entry in os.listdir("/sys/devices/system/node"):
            if not (entry.startswith("node") and entry[4:].isdigit()):
                continue
            with open(f"/sys/devices/system/node/{entry}/cpulist") as f:
                cpulist = f.read().strip()
            node_cpus = set()
            for part in filter(None, cpulist.split(",")):
                lo, _, hi = part.partition("-")
                node_cpus.update(range(int(lo), int(hi or lo) + 1))
            if cpus & node_cpus:
                nodes.add(int(entry[4:]))
    except (OSError, ValueError):
        return None
    return nodes.pop() if len(nodes) == 1 else None


def node_lock_path(base: str, scope_env: str) -> str:
    if os.getenv(scope_env, "node").lower() == "numa":
        node = _affinity_numa_node()
        if node is not None:
            return f"{base}.numa{node}"
    return base


def _ready(futures, timeout, q=queue.SimpleQueue()):
    deadline = time.time() + timeout

    for future in futures:
        future.add_done_callback(q.put)

    for _ in futures:
        yield q.get(timeout=deadline - time.time())


class _Array:
    __slots__ = ("__array_interface__",)


def _unsafe_jax2np(array: jax.Array) -> np.ndarray:
    a = _Array()
    a.__array_interface__ = np.array(array, copy=False).__array_interface__

    ptr, _ = a.__array_interface__["data"]
    a.__array_interface__["data"] = (ptr, False)

    writeable_npy_value = np.array(a, copy=False).view(array.dtype)

    array._npy_value = writeable_npy_value

    return writeable_npy_value


def pbroadcast(x, axis_name, source):
    axis_name = tuple(axis_name)
    masked = jnp.where(jax.lax.axis_index(axis_name) == source, x, jnp.zeros_like(x))
    return jax.lax.psum(masked, axis_name)


def get_replicated_axes(shardings: PyTree[jax.sharding.Sharding]) -> PyTree[P]:
    def _get_axes(sharding):
        if isinstance(sharding, jax.sharding.SingleDeviceSharding):
            return jax.sharding.PartitionSpec()

        spec, _ = jax.tree.flatten(tuple(sharding.spec))
        axes = [axis for axis in sharding.mesh.axis_names if axis not in spec]
        return jax.sharding.PartitionSpec(*axes)

    return jax.tree.map(_get_axes, shardings)


def get_sources(shardings: PyTree[jax.sharding.Sharding], replicated_axes: PyTree[P]):
    counter = 0

    def get_source(sharding, axes):
        if isinstance(sharding, jax.sharding.SingleDeviceSharding):
            return jnp.zeros((), jnp.int32)

        nonlocal counter
        axis_size = np.prod([sharding.mesh.shape[axis] for axis in axes])
        src = jnp.array(counter % axis_size, dtype=jnp.int32)
        counter += 1
        return src

    return jax.tree.map(get_source, shardings, replicated_axes)


def get_load_mask(axes: PyTree[P], srcs: PyTree[jax.Array], mesh: jax.sharding.Mesh):
    @functools.partial(jax.jit)
    @functools.partial(jax.shard_map, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False)
    def load_mask():
        return jax.tree.map(
            lambda axis_name, src: jax.lax.axis_index(tuple(axis_name)) == src, axes, srcs
        )

    return load_mask()
