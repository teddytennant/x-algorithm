# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import concurrent.futures
import dataclasses
import json
import logging
import os
import pathlib
import threading
import time

import jax
import orbax.checkpoint as ocp
import tensorstore as ts
from orbax.checkpoint._src.handlers import pytree_checkpoint_handler as _pytree
from orbax.checkpoint._src.handlers.base_pytree_checkpoint_handler import (
    BasePyTreeCheckpointHandler,
)
from orbax.checkpoint._src.metadata import checkpoint as _step_metadata
from orbax.checkpoint._src.metadata import sharding as sharding_metadata
from orbax.checkpoint._src.multihost import multihost

from xai_checkpointing import save as checkpointing_save
from xai_checkpointing.encrypted_kvstore import at_dir, encrypted_kvstore, use_encrypted_kvstore

rank_logger = logging.getLogger("rank")

_ENVELOPE_MAGIC = b"XAIENC01"
_SPECIAL_CASE_NAMES = ("_DEK", "_DEK.claim")
_SWEEP_THREADS = 64


def encrypt_write(base, directory, name: str, data: bytes) -> None:
    kv = ts.KvStore.open(at_dir(base, pathlib.Path(str(directory)))).result()
    kv.write(name, data).result()


def decrypt_read(base, directory, name: str) -> bytes:
    kv = ts.KvStore.open(at_dir(base, pathlib.Path(str(directory)))).result()
    result = kv.read(name).result()
    if result.state == "missing":
        raise FileNotFoundError(f"{name} does not exist at {directory}")
    return result.value


def _require_array_leaves(state) -> None:
    bad = [
        f"{jax.tree_util.keystr(key_path)}: {type(leaf).__name__}"
        for key_path, leaf in jax.tree_util.tree_leaves_with_path(state)
        if not isinstance(leaf, jax.Array)
    ]
    if bad:
        raise ValueError(
            "encrypted orbax saves support only jax.Array leaves (add support in "
            f"xai_checkpointing/orbax_encrypted.py if needed); offending leaves: {bad}"
        )


class EncryptedPyTreeCheckpointHandler(BasePyTreeCheckpointHandler):
    def __init__(self, kms_client, encryption_chunk_size: int, **kwargs):
        super().__init__(**kwargs)
        self._kms_client = kms_client
        self._encryption_chunk_size = encryption_chunk_size
        self._base_lock = threading.Lock()
        self._base_dir: str | None = None
        self._base: dict | None = None

    def _base_for(self, directory) -> dict:
        with self._base_lock:
            key = str(directory)
            if self._base_dir != key:
                self._base_dir = key
                self._base = encrypted_kvstore(
                    pathlib.Path(key), self._kms_client, self._encryption_chunk_size
                )
            return self._base

    def set_kms_client(self, kms_client) -> None:
        self._kms_client = kms_client

    def _write_metadata_file(self, directory, param_infos, save_args, use_zarr3=False):
        def _save_fn():
            if multihost.is_primary_host(self._primary_host):
                metadata = ocp._src.metadata.tree.InternalTreeMetadata.build(
                    param_infos, save_args=save_args, use_zarr3=use_zarr3
                )
                encrypt_write(
                    self._base_for(directory),
                    directory,
                    "_METADATA",
                    json.dumps(metadata.to_json()).encode(),
                )
            return 0

        return self._thread_pool.submit(_save_fn)

    def _read_metadata_file(self, directory):
        return ocp._src.metadata.tree.InternalTreeMetadata.from_json(
            json.loads(decrypt_read(self._base_for(directory), directory, "_METADATA"))
        )

    def finalize(self, directory):
        path = pathlib.Path(str(directory))
        start = time.monotonic()
        checkpointing_save.finalize_ts(
            path,
            world_size=jax.process_count(),
            ts_context=checkpointing_save._get_ts_context(),
            encrypted_base=self._base_for(directory),
        )
        grafted = time.monotonic()
        assert_all_enveloped(path)
        rank_logger.info(
            "Encrypted orbax finalize done at %s (graft %.1fs, sweep %.1fs)",
            path,
            grafted - start,
            time.monotonic() - grafted,
        )


class EncryptedCheckpointMetadataStore:
    def __init__(self, base_for):
        self._base_for = base_for

    def is_blocking_writer(self) -> bool:
        return True

    def write(
        self,
        checkpoint_path: str | os.PathLike,
        checkpoint_metadata: _step_metadata.StepMetadata,
    ) -> None:
        directory = pathlib.Path(str(checkpoint_path))
        encrypt_write(
            self._base_for(directory),
            directory,
            "_CHECKPOINT_METADATA",
            json.dumps(dataclasses.asdict(checkpoint_metadata)).encode(),
        )

    def read(self, checkpoint_path: str | os.PathLike) -> _step_metadata.StepMetadata | None:
        directory = pathlib.Path(str(checkpoint_path))
        kv = ts.KvStore.open(at_dir(self._base_for(directory), directory)).result()
        result = kv.read("_CHECKPOINT_METADATA").result()
        if result.state == "missing":
            return None
        return _step_metadata.StepMetadata.from_dict(json.loads(result.value.decode()))

    def update(self, checkpoint_path: str | os.PathLike, **kwargs) -> None:
        self.write(
            checkpoint_path,
            dataclasses.replace(
                self.read(checkpoint_path) or _step_metadata.StepMetadata(), **kwargs
            ),
        )

    def wait_until_finished(self) -> None:
        return None

    def close(self) -> None:
        return None


def _encrypted_array_handler(array_handler_cls, base_for, array_handler_args=()):
    class EncryptedArrayHandler(array_handler_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._base_for = base_for

        def _get_json_tspec_write(self, info, *args, **kwargs):
            spec = super()._get_json_tspec_write(info, *args, **kwargs)
            spec["kvstore"] = use_encrypted_kvstore(
                spec["kvstore"], self._base_for(info.parent_dir)
            )
            return spec

        def _get_json_tspec_read(self, info, *args, **kwargs):
            spec = super()._get_json_tspec_read(info, *args, **kwargs)
            spec["kvstore"] = use_encrypted_kvstore(
                spec["kvstore"], self._base_for(info.parent_dir)
            )
            return spec

        async def _serialize_sharding(self, sharding, info, sharding_metadata_txn):
            if info.parent_dir is None:
                raise ValueError("parent_dir cannot be None")
            tspec = ocp.type_handlers.get_sharding_tensorstore_spec(
                info.parent_dir.as_posix(), info.name
            )
            scoped = at_dir(self._base_for(info.parent_dir), info.parent_dir)
            tspec["kvstore"] = {**scoped, "path": scoped.get("path", "") + "_sharding"}
            if multihost.is_primary_host(self._primary_host):
                t = await ts.open(tspec, open=True, context=info.ts_context)
                serialized_sharding = None
                sharding_metadata_value = sharding_metadata.from_jax_sharding(sharding)
                if sharding_metadata_value is not None:
                    serialized_sharding = sharding_metadata_value.to_serialized_string()
                if serialized_sharding is not None:
                    await t.with_transaction(sharding_metadata_txn).write(serialized_sharding)

    return EncryptedArrayHandler(*array_handler_args)


def encrypted_checkpointer(
    kms_client,
    encryption_chunk_size: int,
    timeout_secs: int,
    array_handler_cls,
    array_handler_args=(),
    *,
    checkpoint_handler_kwargs,
):
    impl_ref: list[EncryptedPyTreeCheckpointHandler] = []

    def base_for(directory) -> dict:
        return impl_ref[0]._base_for(directory)

    array_handler = _encrypted_array_handler(array_handler_cls, base_for, array_handler_args)
    registry = _ArraysOnlyRegistry(
        ocp.type_handlers.create_type_handler_registry((jax.Array, array_handler))
    )
    impl = EncryptedPyTreeCheckpointHandler(
        kms_client,
        encryption_chunk_size,
        use_ocdbt=True,
        use_zarr3=checkpoint_handler_kwargs.get("use_zarr3", False),
        save_concurrent_bytes=_pytree._concurrent_bytes(
            checkpoint_handler_kwargs.get("save_concurrent_gb")
        ),
        restore_concurrent_bytes=_pytree._concurrent_bytes(
            checkpoint_handler_kwargs.get("restore_concurrent_gb")
        ),
        type_handler_registry=registry,
    )
    impl_ref.append(impl)
    return ocp.AsyncCheckpointer(
        ocp.PyTreeCheckpointHandler(
            handler_impl=impl, type_handler_registry=registry, **checkpoint_handler_kwargs
        ),
        timeout_secs,
        checkpoint_metadata_store=EncryptedCheckpointMetadataStore(impl._base_for),
    )


class _ArraysOnlyRegistry:
    def __init__(self, inner):
        self._inner = inner

    def get(self, ty):
        if not (isinstance(ty, type) and issubclass(ty, jax.Array)):
            raise ValueError(
                "encrypted orbax saves support only jax.Array leaves (add support in "
                f"xai_checkpointing/orbax_encrypted.py if needed); offending type: {ty}"
            )
        return self._inner.get(ty)

    def has(self, ty):
        return self._inner.has(ty)

    def add(self, *args, **kwargs):
        return self._inner.add(*args, **kwargs)


_ENCRYPTED_CHECKPOINTER = None
_ENCRYPTED_SAVE_CONCURRENT_GB: int | None = None
_ENCRYPTED_TIMEOUT_SECS: int | None = None
_ENCRYPTED_CHUNK_SIZE: int | None = None
_ENCRYPTED_ARRAY_HANDLER_CLS = None


def wait_until_finished() -> None:
    if _ENCRYPTED_CHECKPOINTER is not None:
        _ENCRYPTED_CHECKPOINTER.wait_until_finished()


def get_encrypted_checkpointer(
    kms_client,
    encryption_chunk_size: int,
    timeout_secs: int,
    save_concurrent_gb: int | None,
    array_handler_cls,
):
    global \
        _ENCRYPTED_CHECKPOINTER, \
        _ENCRYPTED_SAVE_CONCURRENT_GB, \
        _ENCRYPTED_TIMEOUT_SECS, \
        _ENCRYPTED_CHUNK_SIZE, \
        _ENCRYPTED_ARRAY_HANDLER_CLS
    if _ENCRYPTED_CHECKPOINTER is None:
        checkpoint_handler_kwargs: dict = {"use_zarr3": True}
        if save_concurrent_gb is not None:
            checkpoint_handler_kwargs["save_concurrent_gb"] = save_concurrent_gb
            checkpoint_handler_kwargs["restore_concurrent_gb"] = save_concurrent_gb
        _ENCRYPTED_SAVE_CONCURRENT_GB = save_concurrent_gb
        _ENCRYPTED_TIMEOUT_SECS = timeout_secs
        array_handler_args = (
            (int(save_concurrent_gb) * 10**9,) if save_concurrent_gb is not None else ()
        )
        _ENCRYPTED_CHUNK_SIZE = encryption_chunk_size
        _ENCRYPTED_ARRAY_HANDLER_CLS = array_handler_cls
        _ENCRYPTED_CHECKPOINTER = encrypted_checkpointer(
            kms_client,
            encryption_chunk_size,
            timeout_secs,
            array_handler_cls,
            array_handler_args,
            checkpoint_handler_kwargs=checkpoint_handler_kwargs,
        )
    else:
        _ENCRYPTED_CHECKPOINTER._handler._handler_impl.set_kms_client(kms_client)
        if save_concurrent_gb is not None and _ENCRYPTED_SAVE_CONCURRENT_GB != save_concurrent_gb:
            rank_logger.warning(
                "get_encrypted_checkpointer(save_concurrent_gb=%s) ignored; checkpointer already "
                "created with save_concurrent_gb=%s.",
                save_concurrent_gb,
                _ENCRYPTED_SAVE_CONCURRENT_GB,
            )
        if _ENCRYPTED_TIMEOUT_SECS is not None and timeout_secs != _ENCRYPTED_TIMEOUT_SECS:
            rank_logger.warning(
                "get_encrypted_checkpointer(timeout_secs=%s) ignored; checkpointer already "
                "created with timeout_secs=%s.",
                timeout_secs,
                _ENCRYPTED_TIMEOUT_SECS,
            )
        if _ENCRYPTED_CHUNK_SIZE is not None and encryption_chunk_size != _ENCRYPTED_CHUNK_SIZE:
            rank_logger.warning(
                "get_encrypted_checkpointer(encryption_chunk_size=%s) ignored; checkpointer already "
                "created with encryption_chunk_size=%s.",
                encryption_chunk_size,
                _ENCRYPTED_CHUNK_SIZE,
            )
        if (
            _ENCRYPTED_ARRAY_HANDLER_CLS is not None
            and array_handler_cls is not _ENCRYPTED_ARRAY_HANDLER_CLS
        ):
            rank_logger.warning(
                "get_encrypted_checkpointer(array_handler_cls=%s) ignored; checkpointer already "
                "created with array_handler_cls=%s.",
                array_handler_cls,
                _ENCRYPTED_ARRAY_HANDLER_CLS,
            )
    return _ENCRYPTED_CHECKPOINTER


def _raise(error: OSError) -> None:
    raise error


def _files_under(directory: str) -> list[str]:
    return [
        os.path.join(root, name)
        for root, _, names in os.walk(directory, onerror=_raise)
        for name in names
    ]


def _is_enveloped(file: str) -> bool:
    with open(file, "rb") as fh:
        return fh.read(8) == _ENVELOPE_MAGIC


def assert_all_enveloped(path: pathlib.Path) -> None:
    if not (path / "_DEK").is_file():
        raise RuntimeError(f"encrypted save left no master _DEK at {path}; not committing")
    with os.scandir(path) as scan:
        top_level = list(scan)
    files = [entry.path for entry in top_level if not entry.is_dir()]
    subdirs = [entry.path for entry in top_level if entry.is_dir() and not entry.is_symlink()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=_SWEEP_THREADS) as pool:
        for nested in pool.map(_files_under, subdirs):
            files.extend(nested)
        files = [file for file in files if os.path.basename(file) not in _SPECIAL_CASE_NAMES]
        enveloped = list(pool.map(_is_enveloped, files))
    offenders = sorted(file for file, ok in zip(files, enveloped, strict=True) if not ok)
    if offenders:
        raise RuntimeError(f"encrypted save left plaintext at rest; not committing: {offenders}")


def extend_base(path: str, kms_client, encryption_chunk_size: int) -> dict:
    return encrypted_kvstore(pathlib.Path(path) / "orbax-ckpt", kms_client, encryption_chunk_size)
