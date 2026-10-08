# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import base64
import contextlib
import ctypes
import fcntl
import functools
import gc
import json
import logging
import math
import os
import pathlib
import re
import time
from collections.abc import Iterable
from typing import Any, Callable

import jax
import tensorstore as ts
from jax.experimental.shard_map import shard_map

from xai_checkpointing import (
    common,
    fix_jax,
)
from xai_checkpointing.dek import adopt_tree_dek
from xai_checkpointing.encrypted_kvstore import at_dir, use_encrypted_kvstore
from xai_checkpointing.kms_client import shared_decrypt_client
from xai_checkpointing.tree_util import has_subtree, tree_to_dict

import orbax.checkpoint as ocp

rank_logger = logging.getLogger("rank")

PyTree = common.PyTree

_NODE_SERIALIZE_ENV = "XAI_RESTORE_NODE_SERIALIZE"
_NODE_LOCK_FILE_ENV = "XAI_RESTORE_NODE_LOCK_FILE"
_ENCRYPTION_MAGIC = b"XAIENC01"


def _is_encrypted_tree(tree: pathlib.Path) -> bool:
    try:
        with (tree / "_METADATA").open("rb") as f:
            return f.read(len(_ENCRYPTION_MAGIC)) == _ENCRYPTION_MAGIC
    except FileNotFoundError:
        return False


def _names_from_tree_metadata(metadata_json: dict[str, Any]) -> list[str]:
    return [
        ".".join(str(k["key"]) for k in v["key_metadata"])
        for v in metadata_json["tree_metadata"].values()
        if not v["value_metadata"]["skip_deserialize"]
    ]


def _read_encrypted_file(
    path: pathlib.Path, key: str, kvstore_base: dict[str, Any], ts_context: ts.Context
) -> bytes:
    kv = ts.KvStore.open(at_dir(kvstore_base, path), context=ts_context).result()
    return kv.read(key).result().value


def _encrypted_tspec_transform(kvstore_base: dict[str, Any]) -> Callable:
    def transform(tspec: dict[str, Any]) -> dict[str, Any]:
        return {**tspec, "kvstore": use_encrypted_kvstore(tspec["kvstore"], kvstore_base)}

    return transform


def _prepare_checkpoint_read(
    path: pathlib.Path, kms_client: object | None
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str], ts.Context]:
    ts_context = ts.Context(
        {
            "file_io_concurrency": {"limit": 128},
            "cache_pool#ocdbt": {"total_bytes_limit": 100000000},
        }
    )
    if not _is_encrypted_tree(path):
        with (path / "_METADATA").open() as f:
            metadata_json = json.load(f)
        checkpoint_names = list(
            tree_to_dict(ocp.StandardCheckpointer().metadata(path), keep_none=False).keys()
        )
        return None, metadata_json, checkpoint_names, ts_context

    if kms_client is None:
        kms_client = shared_decrypt_client()
    raw = adopt_tree_dek(path, kms_client)
    kvstore_base = {
        "driver": "xai_encrypted",
        "base": {"driver": "file", "path": path.as_posix() + "/"},
        "dek_b64": base64.b64encode(raw).decode(),
    }
    metadata_json = json.loads(_read_encrypted_file(path, "_METADATA", kvstore_base, ts_context))
    return kvstore_base, metadata_json, _names_from_tree_metadata(metadata_json), ts_context


def _restore_node_serialize_enabled() -> bool:
    return os.getenv(_NODE_SERIALIZE_ENV, "1").lower() not in ("0", "", "false")


def _node_lock_path() -> str:
    path = os.getenv(_NODE_LOCK_FILE_ENV)
    if path:
        return path
    base = (
        "/dev/shm/xai_restore_node_lock"
        if os.path.isdir("/dev/shm")
        else "/tmp/xai_restore_node_lock"
    )
    return common.node_lock_path(base, "XAI_RESTORE_NODE_LOCK_SCOPE")


class _NodeBatchLock:
    def __init__(self, path: str):
        self._path = path
        self._fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)

    def __enter__(self):
        t0 = time.time()
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 1.0:
            rank_logger.info(
                "restore node-serialize: waited %.1fs for node lock %s", waited, self._path
            )
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self):
        try:
            os.close(self._fd)
        except OSError:
            pass


def _release_batch_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _build_read_plan(
    checkpoint_names: Iterable[str],
    host_state: dict[str, jax.Array],
    load_mask: dict[str, jax.Array],
    rename: Callable[[str], str] | None,
) -> list[tuple[str, str, list[bool], int]]:
    plan: list[tuple[str, str, list[bool], int]] = []
    for checkpoint_name in checkpoint_names:
        name = checkpoint_name
        if rename is not None:
            name = rename(checkpoint_name)

        if host_state.get(name) is None:
            if not has_subtree(name, host_state):
                rank_logger.warning(
                    "Not loading %r from checkpoint because it's not in the initialized state", name
                )
            continue

        if not load_mask:
            mask = [True for _ in host_state[name].addressable_shards]
        else:
            mask = [shard.data.item() for shard in load_mask[name].addressable_shards]
        if any(mask):
            array = host_state[name]
            shard_nbytes = array.dtype.itemsize * math.prod(array.sharding.shard_shape(array.shape))
            plan.append((checkpoint_name, name, mask, shard_nbytes * sum(mask)))

    return plan


def _pack_read_batches(
    plan: list[tuple[str, str, list[bool], int]],
    concurrent_bytes: int | None,
) -> list[list[tuple[str, str, list[bool], int]]]:
    batches: list[list[tuple[str, str, list[bool], int]]] = []
    if concurrent_bytes:
        cur: list[tuple[str, str, list[bool], int]] = []
        cur_bytes = 0
        for item in plan:
            nbytes = item[3]
            if cur and cur_bytes + nbytes > concurrent_bytes:
                batches.append(cur)
                cur, cur_bytes = [], 0
            cur.append(item)
            cur_bytes += nbytes
        if cur:
            batches.append(cur)
    elif plan:
        batches.append(plan)
    return batches


def _convert_domains(domains: dict[str, Any], host_state: dict[str, jax.Array]) -> dict[str, Any]:
    for name, domain in domains.items():
        assert host_state.get(name) is not None, f"Cannot restrict domain of skipped tensor {name}"
        if isinstance(domain, dict):
            domains[name] = ts.IndexDomain(json=domain)
        elif isinstance(domain, ts.DimExpression):
            domains[name] = ts.IndexDomain(shape=host_state[name].shape)[domain]
        elif not isinstance(domain, ts.IndexDomain):
            raise ValueError(f"Unknown domain: {domain} for {name!r}")
    return domains


def _open_tensors(
    plan: list[tuple[str, str, list[bool], int]],
    path: pathlib.Path,
    use_zarr3: bool,
    ts_context: ts.Context,
    arrays: dict[str, jax.Array],
    domains: dict[str, Any],
    tspec_transform: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, ts.TensorStore]:
    futures = {}
    for checkpoint_name, _, _, _ in plan:
        info = ocp.type_handlers.ParamInfo(
            name=checkpoint_name,
            path=path / checkpoint_name,
            parent_dir=path,
            is_ocdbt_checkpoint=True,
            use_zarr3=use_zarr3,
        )
        tspec = ocp.type_handlers.get_json_tspec_read(info, use_ocdbt=True)
        if tspec_transform is not None:
            tspec = tspec_transform(tspec)
        futures[checkpoint_name] = ts.open(ts.Spec(tspec), open=True, context=ts_context)

    stores = {}
    for checkpoint_name, name, _, _ in plan:
        t = stores[checkpoint_name] = futures[checkpoint_name].result()
        if domains.get(name) is None and tuple(t.shape) != tuple(arrays[name].shape):
            raise ValueError(
                f"Tensor {name!r}: checkpoint has shape {tuple(t.shape)}, "
                f"but initialized state has shape {tuple(arrays[name].shape)}. "
                f"Use 'no_loading' to skip this tensor or 'domains' to load a partial slice."
            )
    return stores


def _read_into_shards(
    t: ts.TensorStore,
    array: jax.Array,
    mask: list[bool],
    restricted_domain: ts.IndexDomain | None = None,
):
    memory_kind = array.sharding.memory_kind
    is_cpu = all(d.platform == "cpu" for d in array.sharding.addressable_devices)
    assert memory_kind == "pinned_host" or is_cpu, (
        f"expected pinned_host memory, got {memory_kind!r}"
    )
    devices_indices_map = array.sharding.addressable_devices_indices_map(array.shape)

    for shard, should_load in zip(array.addressable_shards, mask, strict=True):
        if not should_load:
            continue

        index = devices_indices_map[shard.device]
        shard_domain = ts.IndexTransform(input_shape=array.shape)[index].domain

        domain = shard_domain
        if restricted_domain:
            domain = restricted_domain.intersect(shard_domain)

        src = t[domain]

        a = common._unsafe_jax2np(shard.data)
        assert (
            shard.data.addressable_data(0).unsafe_buffer_pointer()
            == a.__array_interface__["data"][0]
        )

        dest = ts.array(a)[ts.d[:].translate_to[shard_domain.origin]][domain]
        yield dest.write(src).commit


def _drain_read_futures(
    futures: dict[Any, tuple[str, str, int]],
    arrays: dict[str, jax.Array],
    path,
    timeout: float,
    *,
    log_loaded: bool = False,
) -> None:
    for future in common._ready(list(futures), timeout=timeout):
        try:
            future.result()
        except Exception:
            checkpoint_name, name, _ = futures.pop(future, (None, None, None))
            tensor = arrays.get(name) if name is not None else None
            extra = (
                f" (in checkpoint: {checkpoint_name})"
                if checkpoint_name is not None and checkpoint_name != name
                else ""
            )
            detail = f"{tensor.shape=}, {tensor.dtype=}" if tensor is not None else "missing tensor"
            rank_logger.exception(
                "Checkpoint error from loading %s. Error loading from %s into %s.%s",
                path,
                name,
                detail,
                extra,
            )
            raise

        checkpoint_name, name, _ = futures.pop(future, (None, None, None))
        if log_loaded and name is not None:
            if checkpoint_name != name:
                rank_logger.debug("Loaded %s (name in checkpoint: %s)", name, checkpoint_name)
            else:
                rank_logger.debug("Loaded %s", name)


def load_checkpoint(
    path: str,
    host_state: PyTree[jax.Array],
    load_mask: PyTree[jax.Array] | None,
    rename: Callable[[str], str] | None = None,
    domains: dict[str, Any] | None = None,
    tag: str | None = None,
    timeout: float = 900.0,
    concurrent_gb: float | None = None,
    kms_client: object | None = None,
):
    if tag is None:
        tag = "orbax-ckpt"

    host_state = tree_to_dict(host_state)
    load_mask = tree_to_dict(load_mask)
    domains = _convert_domains(tree_to_dict(domains), host_state)

    start = time.time()
    rank_logger.info("Restoring checkpoint from %s", path)

    path = pathlib.Path(path) / tag

    kvstore_base, metadata_json, checkpoint_names, ts_context = _prepare_checkpoint_read(
        path, kms_client
    )
    use_zarr3 = metadata_json["use_zarr3"]

    plan = _build_read_plan(checkpoint_names, host_state, load_mask, rename)

    concurrent_bytes = int(concurrent_gb * 10**9) if concurrent_gb else None
    total_bytes = sum(nbytes for *_, nbytes in plan)
    max_leaf = max((nbytes for *_, nbytes in plan), default=0)

    batches = _pack_read_batches(plan, concurrent_bytes)
    num_batches = len(batches)

    if concurrent_bytes:
        rank_logger.info(
            "load_checkpoint: restore read throttle ACTIVE concurrent_gb=%s "
            "(batch limit %.2fGiB): total=%.2fGiB in %d tensors, max_leaf=%.2fGiB, "
            "num_batches=%d (issue+drain one read batch at a time to bound host transient)",
            concurrent_gb,
            concurrent_bytes / (1 << 30),
            total_bytes / (1 << 30),
            len(plan),
            max_leaf / (1 << 30),
            num_batches,
        )
        if max_leaf > concurrent_bytes:
            rank_logger.warning(
                "load_checkpoint: largest tensor loads %.2fGiB > concurrent_gb %.2fGiB; "
                "it is issued as its own batch but still reads in one shot.",
                max_leaf / (1 << 30),
                concurrent_bytes / (1 << 30),
            )
    else:
        rank_logger.info(
            "load_checkpoint: no restore read throttle (concurrent_gb=None); issuing "
            "reads for all %d tensors (%.2fGiB) at once",
            len(plan),
            total_bytes / (1 << 30),
        )

    futures: dict[Any, tuple[str, str, int]] = {}

    node_lock: _NodeBatchLock | None = None
    if _restore_node_serialize_enabled():
        node_lock = _NodeBatchLock(_node_lock_path())
        rank_logger.info(
            "restore node-serialize ACTIVE (flock per batch) lock=%s pid=%d num_batches=%d",
            node_lock._path,
            os.getpid(),
            num_batches,
        )

    with contextlib.ExitStack() as stack:
        if node_lock is not None:
            stack.callback(node_lock.close)
        tspec_transform = None
        if kvstore_base is not None:
            tspec_transform = _encrypted_tspec_transform(kvstore_base)

        stores = _open_tensors(
            plan, path, use_zarr3, ts_context, host_state, domains, tspec_transform
        )
        for batch_index, batch in enumerate(batches, start=1):
            with node_lock if node_lock is not None else contextlib.nullcontext():
                for checkpoint_name, name, mask, _nbytes in batch:
                    t = stores[checkpoint_name]
                    for s, future in enumerate(
                        _read_into_shards(t, host_state[name], mask, domains.get(name))
                    ):
                        futures[future] = (checkpoint_name, name, s)

                inflight_bytes = sum(nbytes for *_, nbytes in batch)
                rank_logger.info(
                    "load_checkpoint: draining read batch %d (%.2fGiB in %d futures)",
                    batch_index,
                    inflight_bytes / (1 << 30),
                    len(futures),
                )
                _drain_read_futures(futures, host_state, path, timeout, log_loaded=True)
            _release_batch_memory()

    rank_logger.info("Loading checkpoint took %.2f sec", time.time() - start)


_AUTO_WINDOW_TARGET_BATCHES = 16
_AUTO_WINDOW_MAX_BYTES = 32 * 10**9


def _auto_window_bytes(plan: list[tuple[str, str, list[bool], int]]) -> int:
    total = sum(nbytes for *_, nbytes in plan)
    max_leaf = max((nbytes for *_, nbytes in plan), default=0)
    return int(min(max(max_leaf, total / _AUTO_WINDOW_TARGET_BATCHES), _AUTO_WINDOW_MAX_BYTES))


def _stage_to_host(arrays: dict[str, jax.Array]) -> dict[str, jax.Array]:
    staged = {}
    for name, array in arrays.items():
        is_cpu = all(d.platform == "cpu" for d in array.sharding.addressable_devices)
        kind = "unpinned_host" if is_cpu else "pinned_host"
        staged[name] = jax.device_put(array, array.sharding.with_memory_kind(kind))
    jax.block_until_ready(list(staged.values()))
    return staged


def copy_aliased_arrays(tree):
    seen: set[int] = set()

    def _copy(x):
        if not isinstance(x, jax.Array):
            return x
        if id(x) in seen:
            copied = jax.device_put(x, x.sharding, may_alias=False)
            assert copied is not x
            return copied
        seen.add(id(x))
        return x

    return jax.tree.map(_copy, tree)


def load_checkpoint_streamed(
    path: str,
    device_state: dict[str, jax.Array],
    load_mask: PyTree[jax.Array] | None,
    rename: Callable[[str], str] | None = None,
    domains: dict[str, Any] | None = None,
    tag: str | None = None,
    timeout: float = 900.0,
    window_gb: float | None = None,
    window_cap_gb: float | None = None,
    on_replaced: Callable[[list[tuple[jax.Array, jax.Array]]], None] | None = None,
    kms_client: Any | None = None,
) -> list[tuple[jax.Array, jax.Array]]:
    if tag is None:
        tag = "orbax-ckpt"

    load_mask = tree_to_dict(load_mask)
    domains = _convert_domains(tree_to_dict(domains), device_state)

    src_path = path
    start = time.time()

    path = pathlib.Path(path) / tag
    kvstore_base, metadata_json, checkpoint_names, ts_context = _prepare_checkpoint_read(
        path, kms_client
    )
    use_zarr3 = metadata_json["use_zarr3"]

    plan = _build_read_plan(checkpoint_names, device_state, load_mask, rename)
    plan.sort(key=lambda item: item[3], reverse=True)

    auto_window = window_gb is None
    window_bytes = _auto_window_bytes(plan) if auto_window else int(window_gb * 10**9)
    total_bytes = sum(nbytes for *_, nbytes in plan)
    max_leaf = max((nbytes for *_, nbytes in plan), default=0)
    if window_cap_gb is not None:
        window_bytes = min(window_bytes, max(max_leaf, int(window_cap_gb * 10**9)))

    batches = _pack_read_batches(plan, window_bytes)
    num_batches = len(batches)

    rank_logger.info(
        "Restoring %s (streamed): window=%.2fGiB%s total=%.2fGiB tensors=%d "
        "max_leaf=%.2fGiB batches=%d",
        src_path,
        window_bytes / (1 << 30),
        " (auto)" if auto_window else "",
        total_bytes / (1 << 30),
        len(plan),
        max_leaf / (1 << 30),
        num_batches,
    )
    if num_batches <= 1 and len(plan) > 1:
        rank_logger.warning(
            "load_checkpoint_streamed: window %.2fGiB covers the full %.2fGiB state "
            "(one batch); set restore_window_gb between %.2fGiB and the total to bound RSS",
            window_bytes / (1 << 30),
            total_bytes / (1 << 30),
            max_leaf / (1 << 30),
        )
    elif max_leaf > window_bytes:
        rank_logger.warning(
            "load_checkpoint_streamed: largest tensor %.2fGiB > window %.2fGiB; "
            "that leaf still needs its full size in pinned memory",
            max_leaf / (1 << 30),
            window_bytes / (1 << 30),
        )

    replaced: list[tuple[jax.Array, jax.Array]] = []
    futures: dict[Any, tuple[str, str, int]] = {}

    node_lock: _NodeBatchLock | None = None
    if _restore_node_serialize_enabled():
        node_lock = _NodeBatchLock(_node_lock_path())
        rank_logger.info(
            "restore node-serialize ACTIVE (flock per batch, streamed) lock=%s pid=%d "
            "num_batches=%d",
            node_lock._path,
            os.getpid(),
            num_batches,
        )

    with contextlib.ExitStack() as stack:
        if node_lock is not None:
            stack.callback(node_lock.close)
        tspec_transform = None if kvstore_base is None else _encrypted_tspec_transform(kvstore_base)
        stores = _open_tensors(
            plan, path, use_zarr3, ts_context, device_state, domains, tspec_transform
        )

        for batch in batches:
            with node_lock if node_lock is not None else contextlib.nullcontext():
                staging = _stage_to_host({name: device_state[name] for _, name, _, _ in batch})

                for checkpoint_name, name, mask, _nbytes in batch:
                    t = stores[checkpoint_name]
                    for s, future in enumerate(
                        _read_into_shards(t, staging[name], mask, domains.get(name))
                    ):
                        futures[future] = (checkpoint_name, name, s)

                _drain_read_futures(futures, staging, path, timeout)

                new_arrays = {
                    name: jax.device_put(staging[name], device_state[name].sharding)
                    for _, name, _, _ in batch
                }
                jax.block_until_ready(list(new_arrays.values()))
                batch_replaced = []
                for _, name, _, _ in batch:
                    batch_replaced.append((device_state[name], new_arrays[name]))
                    device_state[name] = new_arrays[name]
                if on_replaced is not None:
                    on_replaced(batch_replaced)
                else:
                    replaced.extend(batch_replaced)

                del staging, new_arrays, batch_replaced
            if node_lock is not None:
                _release_batch_memory()

    if node_lock is None:
        _release_batch_memory()
    rank_logger.info("Loading checkpoint (streamed) took %.2f sec", time.time() - start)
    return replaced


def broadcast_replicated(
    state: PyTree[jax.Array],
    axes: PyTree[jax.sharding.PartitionSpec],
    srcs: PyTree[jax.Array],
    mesh: jax.sharding.Mesh,
):
    if all(math.prod(mesh.shape[axis] for axis in spec) == 1 for spec in jax.tree.leaves(axes)):
        return jax.block_until_ready(state)

    shardings = jax.tree.map(lambda t: t.sharding, state)
    pspecs = jax.tree.map(lambda s: s.spec, shardings)
    donate_argnums = () if os.getenv("XAI_CHECKPOINT_BROADCAST_DONATE", "1") == "0" else (0,)

    @functools.partial(
        jax.jit, donate_argnums=donate_argnums, in_shardings=(shardings,), out_shardings=shardings
    )
    @functools.partial(shard_map, mesh=mesh, in_specs=(pspecs,), out_specs=pspecs, check_rep=False)
    def broadcast(tree):
        return jax.tree.map(common.pbroadcast, tree, axes, srcs)

    return jax.block_until_ready(broadcast(state))


def rename_tensor(name: str, rename_patterns: dict[str, str]):
    for pattern, repl in rename_patterns.items():
        result, count = re.subn(pattern, repl, name, flags=re.X)
        if count:
            return result
    return name
