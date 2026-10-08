# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import base64
import json
import math
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import jax
import numpy as np
import tensorstore as ts
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from xai_checkpointing import checksum, common
from xrex.utils import ocdbt

_NAMES = ("emb_table", "emb_table_state.row_sum_sq.table", "emb_table_state.last_step.table")
_BASE = 65521


def _synchronized(fn):
    error = None
    result = None
    try:
        result = fn()
    except Exception as exc:
        error = exc
    failed = multihost_utils.process_allgather(np.array([error is not None], np.int32))
    if np.any(failed):
        raise ValueError(
            f"Row checkpoint restore failed: {error or 'failure on another rank'}"
        ) from error
    return result


def _contribution(data, offset, total):
    raw = memoryview(data.view(np.uint8)).cast("B")
    adler = zlib.adler32(raw)
    size = len(raw)
    a = ((adler & 65535) - 1) % _BASE
    return a, ((adler >> 16) - size + (total - offset - size) * a) % _BASE


def _resize_adler(adler, delta_bytes):
    a = adler & 65535
    return (((adler >> 16) + delta_bytes * a) % _BASE) << 16 | a


def _read(root, source, offset, size):
    if len(source) == 2:
        data = source[1][offset : offset + size]
    else:
        with (root / source[1]).open("rb") as f:
            f.seek(source[2] + offset)
            data = f.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated raw chunk: {source[0]}")
    return data


def _prepare(
    path,
    arrays,
    load_mask,
    rename,
    tag,
    on_replaced,
    logical_rows,
    verify_checksums,
    max_bytes,
    row_sharded,
    allow_conversion,
):
    root = Path(path) / (tag or "orbax-ckpt")
    counts = np.zeros((len(_NAMES), 3), np.int32)
    names = [name for name in _NAMES if arrays.get(name) is not None]
    if not names:
        return root, {}, counts
    metadata_file = root / "_METADATA"
    encrypted = (root / "_DEK").exists() or (
        metadata_file.exists() and metadata_file.read_bytes().startswith(b"XAIENC01")
    )
    if encrypted:
        if row_sharded:
            from xai_checkpointing import load

            base, metadata, source_names, context = load._prepare_checkpoint_read(root, None)
            plan = []
            for source_name in source_names:
                name = rename(source_name) if rename else source_name
                if name in names:
                    plan.append((source_name, name, [], 0))
            stores = load._open_tensors(
                plan,
                root,
                metadata["use_zarr3"],
                context,
                arrays,
                {},
                load._encrypted_tspec_transform(base) if base is not None else None,
            )
            for source_name, name, *_ in plan:
                store = stores[source_name]
                chunk = store.chunk_layout.read_chunk.shape
                if name == "emb_table" and chunk[0] == store.shape[0] and chunk[1] < store.shape[1]:
                    raise ValueError("Encrypted column-to-row checkpoint conversion is unsupported")
        return root, {}, counts
    base = {"driver": "file", "path": str(root) + "/"}
    kv = ts.KvStore.open(
        {"driver": "ocdbt", "base": base} if (root / "manifest.ocdbt").exists() else base
    ).result()
    probe_names = names
    if rename is not None:
        probe_names = [
            ".".join(str(k["key"]) for k in value["key_metadata"])
            for value in json.loads(metadata_file.read_text())["tree_metadata"].values()
        ]
    plans = {}
    for source_name in probe_names:
        name = rename(source_name) if rename else source_name
        if name not in names and source_name not in names:
            continue
        array = arrays[name if name in names else source_name]
        value = kv.read(source_name + "/zarr.json").result()
        zarr3 = value.state != "missing"
        if not zarr3:
            value = kv.read(source_name + "/.zarray").result()
        if value.state == "missing":
            continue
        metadata = json.loads(value.value)
        shape = tuple(metadata["shape"])
        chunk = (
            tuple(metadata["chunk_grid"]["configuration"]["chunk_shape"])
            if zarr3
            else tuple(metadata["chunks"])
        )
        column_chunked = (
            zarr3 and "emb_table" in (name, source_name) and len(shape) == 2 and chunk[1] < shape[1]
        )
        row_chunked = (
            zarr3
            and "emb_table" in (name, source_name)
            and len(shape) == 2
            and chunk[0] < shape[0]
            and chunk[1] == shape[1]
        )
        if shape == array.shape and not (column_chunked if row_sharded else row_chunked):
            continue
        saved_file = Path(path) / "checksums.0.json"
        sharding_file = root / "_sharding"
        encoded = base64.b64encode(name.encode()).decode()
        stored_sharding = (
            json.loads(sharding_file.read_text()).get(encoded) if sharding_file.exists() else None
        )
        saved = (
            json.loads(saved_file.read_text())
            if saved_file.exists() and (verify_checksums or not stored_sharding)
            else {}
        )
        if stored_sharding:
            entry = json.loads(stored_sharding)
            mesh = dict(zip(entry.get("axis_names", []), entry.get("shape", []), strict=True))
        else:
            mesh = saved.get("shardings", {}).get(name, {}).get("mesh") or {}
        source_spec = (
            entry.get("partition_spec", [])
            if stored_sharding
            else saved.get("shardings", {}).get(name, {}).get("spec", [])
        )
        row_axes = source_spec[0] if source_spec else None
        source_row_sharded = row_axes == "expert" or (
            isinstance(row_axes, list) and "expert" in row_axes
        )
        if shape == array.shape and (
            source_row_sharded
            if row_sharded
            else mesh.get("expert", 1) == 1 or not source_row_sharded
        ):
            continue
        if not allow_conversion:
            raise ValueError(f"{name}: row layout conversion requires a complete local checkpoint")
        if source_name != name or (rename and rename(name) != name):
            raise ValueError(f"Renames involving handled row tensor {name} are unsupported")
        if not zarr3 or not (root / "manifest.ocdbt").exists():
            raise ValueError(f"{name}: changed row restore requires raw Zarr3 OCDBT")
        if logical_rows <= 0 or len(shape) != (2 if name == "emb_table" else 1):
            raise ValueError(f"{name}: invalid row geometry")
        if len(chunk) != len(shape) or any(type(n) is not int or n <= 0 for n in shape + chunk):
            raise ValueError(f"{name}: nonpositive shape/chunk geometry")
        if shape[1:] != array.shape[1:] or array.shape[0] < logical_rows:
            raise ValueError(f"{name}: width changed or destination truncates logical rows")
        if any(c > n for c, n in zip(chunk[1:], shape[1:], strict=True)):
            raise ValueError(f"{name}: column chunk exceeds source width")
        dtype = np.dtype(ts.dtype(metadata["data_type"]).numpy_dtype)
        row_bytes = math.prod(shape[1:]) * dtype.itemsize
        if dtype != array.dtype or max_bytes < row_bytes:
            raise ValueError(f"{name}: dtype changed or window smaller than one row")
        codecs = metadata["codecs"]
        raw_codec = [{"name": "bytes", "configuration": {"endian": "little"}}]
        if len(codecs) != 1 or codecs[0]["name"] != "sharding_indexed":
            raise ValueError(f"{name}: requires single-inner-chunk indexed sharding")
        cfg = codecs[0]["configuration"]
        if (
            tuple(cfg["chunk_shape"]) != chunk
            or cfg["codecs"] != raw_codec
            or cfg["index_codecs"] != raw_codec + [{"name": "crc32c"}]
            or cfg.get("index_location", "end") != "end"
        ):
            raise ValueError(f"{name}: requires bytes-only codec and a single end index")
        if metadata["fill_value"] != 0 or (
            metadata["chunk_key_encoding"]["name"] != "default"
            or metadata["chunk_key_encoding"].get("configuration", {}).get("separator", "/") != "/"
        ):
            raise ValueError(f"{name}: requires zero fill and default chunk keys")
        source_ep = mesh.get("expert", 1)
        if (
            type(source_ep) is not int
            or source_ep <= 0
            or shape[0]
            not in (logical_rows, ((logical_rows + source_ep - 1) // source_ep) * source_ep)
        ):
            raise ValueError(f"{name}: source rows are not logical rows or source-EP padding")
        if verify_checksums:
            checksum.check_internal_consistency(saved, str(saved_file), names=[name])
            if name not in saved.get("global_checksums", {}):
                raise ValueError(f"{name}: missing stored global checksum")
        sources = ocdbt.load_shard_sources(str(root), prefix=name + "/")
        raw_metadata = next(s for s in sources if s[0] == name + "/zarr.json")
        size = len(raw_metadata[1]) if len(raw_metadata) == 2 else raw_metadata[3] + 20
        if json.loads(_read(root, raw_metadata, 0, size)) != metadata:
            raise ValueError(f"{name}: inconsistent raw metadata")
        chunks = {}
        raw_size = math.prod(chunk) * dtype.itemsize
        for source in sources:
            if not source[0].startswith(name + "/c/"):
                continue
            index = tuple(map(int, source[0][len(name) + 3 :].split("/")))
            if len(index) != len(shape) or any(
                i < 0 or i * c >= n for i, c, n in zip(index, chunk, shape, strict=True)
            ):
                raise ValueError(f"{name}: invalid chunk index")
            size = len(source[1]) - 20 if len(source) == 2 else source[3]
            if size != raw_size or struct.unpack("<QQ", _read(root, source, size, 16)) != (0, size):
                raise ValueError(f"{name}: invalid raw size or end index")
            if index in chunks:
                raise ValueError(f"{name}: duplicate raw chunk")
            chunks[index] = source
        if not isinstance(array.sharding, NamedSharding):
            raise ValueError(f"{name}: destination requires named sharding")
        if row_sharded:
            indices = array.sharding.devices_indices_map(array.shape)
            for index in indices.values():
                normalized = tuple(s.indices(n) for s, n in zip(index, array.shape, strict=True))
                if normalized[0][2] != 1 or any(
                    s != (0, n, 1) for s, n in zip(normalized[1:], array.shape[1:], strict=True)
                ):
                    raise ValueError(f"{name}: destination must contain contiguous full-width rows")
        elif array.shape[0] != logical_rows:
            raise ValueError(f"{name}: column destination must retain logical rows")
        mask = (
            {s.device: bool(s.data.item()) for s in load_mask[name].addressable_shards}
            if load_mask
            else None
        )
        shards = array.addressable_shards
        loaded = sum(mask is None or mask[s.device] for s in shards)
        counts[_NAMES.index(name)] = (1, loaded, len(shards))
        if (
            on_replaced is None
            and array.sharding.memory_kind != "pinned_host"
            and any(s.device.platform != "cpu" for s in shards)
        ):
            raise ValueError(f"{name}: GPU restore requires on_replaced")
        plans[name] = (metadata, chunks, saved.get("global_checksums", {}).get(name))
    return root, plans, counts


def _window(root, metadata, chunks, start, stop, pool):
    shape = metadata["shape"]
    chunk = metadata["chunk_grid"]["configuration"]["chunk_shape"]
    dtype = np.dtype(ts.dtype(metadata["data_type"]).numpy_dtype)
    width = math.prod(shape[1:])
    chunk_width = math.prod(chunk[1:])
    block = np.zeros((stop - start, width), dtype)

    def read_column(column):
        col = column * chunk_width
        for row in range(start // chunk[0], (min(stop, shape[0]) + chunk[0] - 1) // chunk[0]):
            lo, hi = max(start, row * chunk[0]), min(stop, shape[0], (row + 1) * chunk[0])
            if hi <= lo:
                continue
            source = chunks.get((row, column) if len(shape) == 2 else (row,))
            if source is not None:
                data = _read(
                    root,
                    source,
                    (lo - row * chunk[0]) * chunk_width * dtype.itemsize,
                    (hi - lo) * chunk_width * dtype.itemsize,
                )
                values = np.frombuffer(data, dtype).reshape(hi - lo, chunk_width)
                block[lo - start : hi - start, col : col + chunk_width] = values[
                    :, : min(chunk_width, width - col)
                ]

    list(pool.map(read_column, range((width + chunk_width - 1) // chunk_width)))
    return block.reshape((stop - start, *shape[1:]))


@partial(jax.jit, donate_argnums=(0,))
def _put_rows(destination, block, start):
    return jax.lax.dynamic_update_slice(destination, block, (start,) + (0,) * (block.ndim - 1))


def restore_row_embeddings(
    path,
    arrays,
    load_mask,
    rename,
    tag,
    on_replaced,
    *,
    logical_rows,
    row_sharded=True,
    verify_checksums=True,
    max_bytes=64 << 20,
    allow_conversion=True,
) -> dict[str, int]:
    root, plans, counts = _synchronized(
        lambda: _prepare(
            path,
            arrays,
            load_mask,
            rename,
            tag,
            on_replaced,
            logical_rows,
            verify_checksums,
            max_bytes,
            row_sharded,
            allow_conversion,
        )
    )
    gathered = multihost_utils.process_allgather(counts).reshape(-1, len(_NAMES), 3)
    if np.any(gathered[:, :, 0] != gathered[0, :, 0]):
        raise ValueError("Row restore plans differ between ranks")
    counts = gathered.sum(axis=0)
    if np.any((counts[:, 1] != 0) & (counts[:, 1] != counts[:, 2])):
        raise ValueError(
            "Row restore partial global load masks are unsupported; load all or skip all"
        )
    expected = {}

    def restore_one(name, array, metadata, chunks, stored):
        row_bytes = math.prod(array.shape[1:]) * array.dtype.itemsize
        source_rows = metadata["shape"][0]
        total = source_rows * row_bytes

        def restore_local():
            contributions = np.zeros(2, np.uint32)
            restored = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                for shard in array.addressable_shards:
                    start, end, _ = shard.index[0].indices(array.shape[0])
                    destination = (
                        shard.data
                        if on_replaced and row_sharded
                        else common._unsafe_jax2np(shard.data)
                    )
                    stop = max(end, source_rows) if end == array.shape[0] else end
                    sums = [0, 0]
                    for lo in range(start, stop, max_bytes // row_bytes):
                        hi = min(stop, lo + max_bytes // row_bytes)
                        block = _window(root, metadata, chunks, lo, hi, pool)
                        if np.any(block[max(0, logical_rows - lo) :].view(np.uint8)):
                            raise ValueError(f"{name}: nonzero source padding")
                        if lo < source_rows:
                            part = _contribution(
                                block[: min(hi, source_rows) - lo], lo * row_bytes, total
                            )
                            sums = [(a + b) % _BASE for a, b in zip(sums, part, strict=True)]
                        if lo < end:
                            block = block[: min(hi, end) - lo]
                            if on_replaced and row_sharded:
                                destination = _put_rows(
                                    destination,
                                    jax.device_put(block, shard.data.sharding),
                                    lo - start,
                                )
                                destination.block_until_ready()
                            else:
                                destination[lo - start : min(hi, end) - start] = block
                    if shard.replica_id == 0:
                        contributions = (contributions + np.array(sums, np.uint32)) % _BASE
                    if on_replaced and row_sharded:
                        restored.append(destination)
            replacement = (
                jax.make_array_from_single_device_arrays(array.shape, array.sharding, restored)
                if on_replaced and row_sharded
                else None
            )
            return contributions, replacement

        contributions, replacement = _synchronized(restore_local)
        gathered = multihost_utils.process_allgather(contributions).reshape(-1, 2)

        def finish():
            a, b = gathered.sum(axis=0, dtype=np.uint64).tolist()
            actual = ((total + b) % _BASE) << 16 | (1 + a) % _BASE
            if verify_checksums and actual != stored:
                raise ValueError(f"{name}: source checksum does not match")
            if on_replaced and row_sharded:
                on_replaced([(array, replacement)])
            return _resize_adler(actual, (array.shape[0] - source_rows) * row_bytes)

        return _synchronized(finish)

    def restore_column(name, original, plan):
        mesh = original.sharding.mesh
        ep = mesh.shape["expert"]
        rows = ((logical_rows + ep - 1) // ep) * ep
        shape = (rows, *original.shape[1:])
        kind = (
            "pinned_host"
            if any(d.platform == "gpu" for d in mesh.devices.flat)
            else "unpinned_host"
        )
        row_sharding = NamedSharding(mesh, P("expert"), memory_kind=kind)
        staging = _synchronized(
            lambda: jax.make_array_from_callback(
                shape,
                row_sharding,
                lambda index: np.zeros(
                    tuple(
                        s.stop - (s.start or 0) if s.stop is not None else n
                        for s, n in zip(index, shape, strict=True)
                    ),
                    original.dtype,
                ),
            )
        )
        value = restore_one(name, staging, *plan)
        target = original.sharding
        device_rows = row_sharding.with_memory_kind("device")
        device_target = target.with_memory_kind("device")
        reshard = _synchronized(
            lambda: jax.jit(
                lambda data: data,
                in_shardings=device_rows,
                out_shardings=device_target,
                donate_argnums=(0,),
            )
            .lower(jax.ShapeDtypeStruct(shape, original.dtype, sharding=device_rows))
            .compile()
        )
        loaded = _synchronized(lambda: jax.device_put(staging, device_rows).block_until_ready())
        converted = reshard(loaded)
        converted.block_until_ready()
        trim = jax.jit(lambda data: data[:logical_rows], out_shardings=device_target)
        converted = _synchronized(lambda: trim(converted).block_until_ready())
        converted = _synchronized(lambda: jax.device_put(converted, target).block_until_ready())
        if on_replaced:
            _synchronized(lambda: on_replaced([(original, converted)]))
        else:

            def copy_back():
                sources = {shard.device: shard.data for shard in converted.addressable_shards}
                for shard in original.addressable_shards:
                    np.copyto(common._unsafe_jax2np(shard.data), np.asarray(sources[shard.device]))

            _synchronized(copy_back)
        row_bytes = math.prod(original.shape[1:]) * original.dtype.itemsize
        return _resize_adler(value, (logical_rows - rows) * row_bytes)

    for name, plan in plans.items():
        if counts[_NAMES.index(name), 1]:
            original = arrays[name]
            expected[name] = (
                restore_one(name, original, *plan)
                if row_sharded
                else restore_column(name, original, plan)
            )
        arrays[name] = None
    return expected
