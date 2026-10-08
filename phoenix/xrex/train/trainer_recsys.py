# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import collections
import concurrent.futures
import enum
import functools
import gc
import itertools
import json
import logging
import math
import operator
import os
import pathlib
import shutil
import signal
import sys
import time
import typing
from dataclasses import dataclass, field, replace
from typing import Iterator, NamedTuple, Sequence

import grpc
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import psutil
from jax import shard_map
from jax.experimental import multihost_utils
from jax.sharding import (
    NamedSharding,
)
from jax.sharding import PartitionSpec as P
from jax.tree_util import DictKey, GetAttrKey, SequenceKey

import xai_recsys_engine
from xai_checkpointing.common import _unsafe_jax2np
from xai_checkpointing.tree_util import tree_to_dict
from xai_configlib import configclass
from xai_proto import copy_pb2, copy_pb2_grpc
from xrex.data.parquet_recsys import DataPosition, PhoenixDataset

if typing.TYPE_CHECKING:
    from xrex.cuda.async_emb import (
        async_emb,
    )

    AsyncEmbContextHandle: typing.TypeAlias = async_emb.AsyncEmbContextHandle
else:
    try:
        from xrex.cuda.async_emb import async_emb
    except ImportError:
        async_emb = None

    AsyncEmbContextHandle = typing.Any

from xrex.data.recsys.recsys_batch import RecsysFeaturesBatch
from xrex.data.recsys.sequence_packing import (
    FixedLengthDistribution,
    LengthDistribution,
    pack_batch,
)
from xrex.data.retrieval_dataset import RetrievalDataset
from xrex.data.rust_kafka_recsys import RustKafkaDataset
from xrex.data.streaming.kafkaloader import PhoenixKafkaDataset, report_training_metrics
from xrex.eval.eval_utils import report_forward_eval_results
from xrex.eval.metrics_recsys import merge_report_extras
from xrex.eval.recsys_eval import RecsysTwoTowerEval, run_recsys_evals
from xrex.models.compress_token_ids import compress_token_ids
from xrex.models.model_utils import Parameter, unwrap_tree
from xrex.models.recsys_embedding import (
    EmbTable,
    RecsysEmbeddings,
    RecsysEmbeddingsParameter,
    get_recsys_embed_param_to_jax_array,
)
from xrex.models.recsys_model import (
    CandidateInputs,
    RecsysAggregatedModel,
    RecsysAggregatedModelConfig,
)
from xrex.models.sharding_context import make_legacy_sharding_context
from xrex.optimizers.optim import InjectHyperparamsState, apply_updates
from xrex.optimizers.recsys import RecsysEmbeddingOptimConfig
from xrex.optimizers.recsys.protocol import AsyncEmbOptimizer
from xrex.train.misc import (
    CheckpointConfig,
    PostEmbeddings,
    RecsysTrainingState,
)
from xrex.train.trainer import (
    RecsysTwoTowerModelConfig,
    Trainer,
    TrainerContext,
)
from xrex.utils import recsys_async_emb
from xrex.utils.aot import JittedOrCompiled
from xrex.utils.checkpointing import wait_until_finished
from xrex.utils.metrics import norm_metrics
from xrex.utils.utils import (
    cast_bfloat16,
)

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")


OUT_PATH = "/dev/shm"

_DATA_POSITION_FILENAME = "data_position.json"
_CONFIG_FILENAME = "config.json"


@dataclass(frozen=True)
class FetchedBatch:
    batch: typing.Any
    offsets: dict[int, int]
    data_position: typing.Any


@dataclass
class BatchPipelineState:
    current: FetchedBatch | None = None
    reserve: FetchedBatch | None = None
    exhausted: bool = False


class IncrementalState(NamedTuple):
    unique_tokens: npt.NDArray[np.int32]
    token_count: int
    soft_step: int
    checkpoint_path: str


class Store(enum.Enum):
    FS = 0
    S3 = 1
    GCS = 2

    @staticmethod
    def from_path(path):
        if path.startswith("gs://"):
            return Store.GCS
        if path.startswith("s3://"):
            return Store.S3
        return Store.FS


def path_to_dotted(path):
    parts = []
    for p in path:
        if isinstance(p, DictKey):
            parts.append(str(p.key))
        elif isinstance(p, GetAttrKey):
            parts.append(p.name)
        elif isinstance(p, SequenceKey):
            parts.append(str(p.idx))
    return ".".join(parts)


def pbroadcast(x, axis_name, source):
    axis_name = tuple(axis_name) if not isinstance(axis_name, tuple) else axis_name
    masked = jnp.where(jax.lax.axis_index(axis_name) == source, x, jnp.zeros_like(x))
    return jax.lax.psum(masked, axis_name)


def _num_users(data: RecsysFeaturesBatch) -> int:
    return math.prod(data["user_hashes"].shape[:-1])


def _step_token_ids(token_ids: Sequence[jax.Array], data_axis) -> jax.Array:
    return jax.lax.with_sharding_constraint(jnp.stack(token_ids), P(None, data_axis, None))


def _pad_vocab_for_ep(emb: jax.Array, ep: int, name: str = "emb_table") -> jax.Array:
    V = emb.shape[0]
    target_V = -(-V // ep) * ep
    if target_V == V:
        return emb
    rank_logger.info("Padded %s rows %d -> %d for ep=%d save", name, V, target_V, ep)
    return jnp.pad(emb, ((0, target_V - V),) + ((0, 0),) * (emb.ndim - 1))


def _write_all_bytes(path: str, mv: memoryview, chunk_size: int = 1 << 30) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    total_size = len(mv)
    written = 0
    with open(path, "wb", buffering=0) as f:
        while written < total_size:
            chunk_end = min(written + chunk_size, total_size)
            chunk = mv[written:chunk_end]
            while len(chunk) > 0:
                n = f.write(chunk)
                if n is None or n <= 0:
                    raise OSError(
                        f"failed writing checkpoint shard {path}: "
                        f"wrote {written} of {total_size} bytes"
                    )
                written += n
                chunk = chunk[n:]

    if written != total_size:
        raise OSError(
            f"incomplete write for checkpoint shard {path}: wrote {written} of {total_size} bytes"
        )


def _read_checkpoint_kafka_config(ctx) -> tuple[str | None, int | None]:
    if ctx.checkpoint is None:
        return None, None
    config_path = os.path.join(ctx.checkpoint.path, _CONFIG_FILENAME)
    if not os.path.isfile(config_path):
        return None, None
    try:
        with open(config_path) as f:
            config = json.load(f)
        dataset_config = config.get("dataset", config)
        topic = dataset_config.get("topic_name")
        partitions = dataset_config.get("num_kafka_partitions")
        return topic, partitions
    except (json.JSONDecodeError, OSError) as e:
        rank_logger.warning("Failed to read kafka config from %s: %s", config_path, e)
        return None, None


@configclass
class RecsysCheckpointConfig(CheckpointConfig):
    keep_emb_opt_state: bool = False


@configclass
class RecsysTrainer(Trainer):
    offsets_to_commit: dict[int, int] = field(default_factory=dict)
    store_load: str | None = None
    exclude_nodes_file: str | None = None
    enable_metrics_file: bool = False
    emb_optim_config: RecsysEmbeddingOptimConfig = field(default_factory=RecsysEmbeddingOptimConfig)

    stop_at_data_end: bool = False

    empty_history_augmentation_rate: float = 0.0

    empty_history_user_dropout_rate: float = 0.0

    split_home_checkpoint: bool = False

    export_stablehlo_bundle: bool = False
    export_bundle_bs_per_device: str = "1,2,4"
    export_bundle_history_seq_len: int = 0
    export_bundle_candidate_seq_len: int = 0

    smoothing_windows: list[int] = field(default_factory=lambda: [1_048_576, 4_194_304])

    reset_data_position: bool = False

    seqpack_distribution: LengthDistribution | None = None

    seqpack_fixed_length: bool = False

    use_async_emb: bool = False

    use_row_emb: bool = False
    row_emb_recv_factor: float = 2.0
    num_microbatch: int = 1

    _async_emb_context: AsyncEmbContextHandle | None = field(default=None, init=False, repr=False)
    _emb_hash_vocab: int = field(default=0, init=False, repr=False)
    _first_step_embedding_lookup_start_jit: typing.Any = field(default=None, init=False, repr=False)
    _async_emb_step_jit: typing.Any = field(default=None, init=False, repr=False)
    _async_emb_lookup_pin: jax.Array | None = field(default=None, init=False, repr=False)
    _batch_pipeline: BatchPipelineState = field(
        default_factory=BatchPipelineState, init=False, repr=False
    )

    _data_position: DataPosition | None = field(default=None, init=False, repr=False)
    _seqpack_rng: np.random.Generator | None = field(default=None, init=False, repr=False)
    _history_user_dropout_rng: np.random.Generator | None = field(
        default=None, init=False, repr=False
    )
    _last_history_user_dropout_count: int = field(default=0, init=False, repr=False)
    _last_history_user_dropout_bsz: int = field(default=0, init=False, repr=False)
    _retrieval_post_emb_built: bool = field(default=False, init=False, repr=False)
    dense_params_ema_decay: float = 0.0
    dense_params_ema_in_checkpoint: bool = False
    _dense_params_ema: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_init_jit: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_update_jit: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_gap_jit: typing.Any = field(default=None, init=False, repr=False)
    _dense_params_ema_as_params_jit: typing.Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.using_seqpack and self.seqpack_fixed_length:
            assert isinstance(self.model_config, RecsysAggregatedModelConfig)
            hl = self.model_config.history_seq_len
            self.seqpack_distribution = FixedLengthDistribution(min_len=hl, max_len=hl, mean_len=hl)
        if self.export_stablehlo_bundle and not self.checkpoint_config.copy_port:
            raise ValueError("export_stablehlo_bundle requires checkpoint_config.copy_port")

    state: RecsysTrainingState = field(init=False, repr=False, compare=False)
    _pending_shmem_ckpt_write_s: float | None = field(default=None, init=False, repr=False)
    _pending_checksum_s: float | None = field(default=None, init=False, repr=False)
    _pending_gc_collect_s: float | None = field(default=None, init=False, repr=False)
    _stablehlo_bundle_files: list | None = field(default=None, init=False, repr=False)

    _engine = None
    _shmem_write_pool = None
    _shmem_write_future = None
    _incremental_state = None
    _last_disk_checkpoint_ts = 0.0

    train_dataset: typing.Any = field(init=False, repr=False, compare=False, default=None)
    _eval_dataset: typing.Any = field(init=False, repr=False, compare=False, default=None)
    _emb_optim: typing.Any = field(init=False, repr=False, compare=False, default=None)
    loss_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    loss_fn_eval: typing.Any = field(init=False, repr=False, compare=False, default=None)
    microbatch_loss_fns: typing.Any = field(init=False, repr=False, compare=False, default=None)
    forward_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    two_tower_forward_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    two_tower_forward_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    mol_side_table_fn: typing.Any = field(init=False, repr=False, compare=False, default=None)
    mol_side_table_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    candidate_tower_forward_fn: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    state_shape: typing.Any = field(init=False, repr=False, compare=False, default=None)
    stats_shape: typing.Any = field(init=False, repr=False, compare=False, default=None)
    state_sharding: typing.Any = field(init=False, repr=False, compare=False, default=None)
    data_sharding: typing.Any = field(init=False, repr=False, compare=False, default=None)
    init_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    update_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    forward_jit: typing.Any = field(init=False, repr=False, compare=False, default=None)
    candidate_tower_forward_jit: typing.Any = field(
        init=False, repr=False, compare=False, default=None
    )
    host_state: typing.Any = field(init=False, repr=False, compare=False, default=None)

    def create_runtime(self, ctx):
        if os.environ.get("JAX_COORDINATOR_PORT"):
            ctx.coordinator_port = int(os.environ["JAX_COORDINATOR_PORT"])
        super().create_runtime(ctx)

    def example_data(self, bs: int) -> RecsysFeaturesBatch:
        assert isinstance(self.dataset, PhoenixDataset)
        batch = self.dataset.example_data(bs)
        if self.using_seqpack:
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            batch = pack_batch(
                batch=batch,
                num_devices_per_process=self.parallel_config.num_devices_per_process,
                num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                dist=self.seqpack_distribution,
                rng=np.random.default_rng(0),
            )
            if self.using_fa4:
                batch = self.add_block_sparse_layout(batch)
        return batch

    def prepare_data(self, batch):
        return super().prepare_data(self.transform_batch(batch))

    def transform_batch(self, batch: RecsysFeaturesBatch) -> RecsysFeaturesBatch:
        if self.using_seqpack:
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            batch = pack_batch(
                batch=batch,
                num_devices_per_process=self.parallel_config.num_devices_per_process,
                num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                dist=self.seqpack_distribution,
                rng=self._seqpack_rng,
            )
            if self.using_fa4:
                batch = self.add_block_sparse_layout(batch)

        return batch

    def _purchase_value_ema_keys(self) -> dict[str, jax.Array]:
        if not (
            isinstance(self.model_config, RecsysAggregatedModelConfig)
            and self.model_config.purchase_value_enabled
        ):
            return {}
        return {
            f"purchase_value/{ws}": jnp.zeros((10,), dtype=jnp.float32)
            for ws in self.model_config.purchase_value_smoothing_windows
        }

    def _conversion_delay_slice_ema_keys(self, state_size: int) -> dict[str, jax.Array]:
        if not isinstance(self.model_config, RecsysAggregatedModelConfig):
            return {}
        return {
            f"{s.head}/{s.name}/{ws}": jnp.zeros((state_size,), dtype=jnp.float32)
            for s in self.model_config.conversion_delay_slices()
            for ws in self.smoothing_windows
        }

    def init(self, batch: RecsysFeaturesBatch, rng: jax.Array) -> RecsysTrainingState:
        assert isinstance(self.dataset, PhoenixDataset)
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        if self.using_seqpack:
            assert self.seqpack_distribution is not None
            assert self.seqpack_distribution.max_len == self.dataset.history_seq_len
            self._seqpack_rng = np.random.default_rng(self.rng_seed)

        rng, init_rng = jax.random.split(rng)
        if self.rng_seed == 0:
            init_rng = jax.random.PRNGKey(0), [jax.random.PRNGKey(0)] * 1000

        is_device = self.model_config.get_embed_memory_kind() == "device"
        emb_table, emb_table_state = self.model_config.make_embedding_table(
            init_rng,
            self.optim,
            self.dataset.input_vocab_size if is_device else 1,
            self.init_opt_state,
        )
        if self.use_row_emb:
            rows = (
                math.ceil(emb_table.x.shape[0] / self.parallel_config.ep) * self.parallel_config.ep
            )
            emb_table = replace(
                emb_table,
                x=jnp.pad(emb_table.x, ((0, rows - emb_table.x.shape[0]), (0, 0))),
                pspec=P("expert", None),
            )
        if self.emb_optim_config._active() == "rowwise_adagrad":
            sparse_emb_optim = self.emb_optim_config.make_optimizer(self.optim)
            emb_table_state = sparse_emb_optim.init({"table": emb_table})
            if self.use_row_emb:
                logical_rows = self.dataset.input_vocab_size
                emb_table_state = emb_table_state._replace(
                    row_sum_sq={
                        "table": emb_table_state.row_sum_sq["table"].at[logical_rows:].set(0)
                    }
                )
        emb_table = replace(emb_table, x=emb_table.x.at[0, :].set(0))
        post_embeddings = PostEmbeddings(
            post_ids=jnp.arange(1).astype(jnp.int32),
            author_ids=jnp.arange(1).astype(jnp.int32),
            embeddings=Parameter(x=jnp.empty((1, 1), dtype=jnp.bfloat16), pspec=P(None, None)),
            dataset_types=jnp.array([RetrievalDataset.PAD.value], dtype=jnp.int32),
        )
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            post_embeddings = self.model_config.candidate_tower_config.make_post_embeddings()
            side_width = self.model_config.mol_side_table_width
            if side_width:
                emb = post_embeddings.embeddings
                post_embeddings = post_embeddings._replace(
                    mol_side_table=Parameter(
                        x=jnp.zeros((emb.x.shape[0], side_width), dtype=jnp.float32),
                        pspec=P(None, None),
                    )
                )

        packing_layout = batch.get("packing_layout")
        batch = jax.tree.map(jnp.ones_like, batch)
        batch["packing_layout"] = packing_layout

        recsys_embeddings = self.get_recsys_embeddings(batch, emb_table)

        initial_params = self.loss_fn.init(init_rng, batch, recsys_embeddings)

        if self.init_opt_state:
            initial_opt_state = self.optim.init(initial_params)
        else:
            initial_opt_state = None

        if self.precision_level == 0:
            initial_params = jax.tree.map(cast_bfloat16, initial_params)
        if self.precision_level < 2:
            initial_opt_state = jax.tree.map(cast_bfloat16, initial_opt_state)

        if isinstance(self.dataset, PhoenixKafkaDataset):
            self.dataset.ensure_partition_count()
        rep = self.dataset.num_kafka_partitions or 1

        rce_ema = None
        if self.smoothing_windows and isinstance(self.model_config, RecsysAggregatedModelConfig):
            from xrex.data.recsys.constants import engagement_to_ids
            from xrex.models.recsys_model import build_metric_masks

            eng_names = list(engagement_to_ids(self.model_config.metric_group).keys())
            dummy = jnp.zeros((1, 1))
            dummy_3d = jnp.zeros((1, 1, self.model_config.model_config.output_vocab_size))
            mask_keys = list(
                build_metric_masks(
                    dummy,
                    dummy_3d,
                    dummy,
                    dummy,
                    trained_candidate_mask=dummy_3d,
                    ads_head_masking=self.model_config.ads_head_masking,
                    enable_platform_metrics=self.model_config.enable_platform_metrics,
                    metric_mask_keys=self.model_config.metric_mask_keys,
                ).keys()
            )
            rce_ema = {
                f"{e}/{m}/{ws}": jnp.zeros((3,), dtype=jnp.float32)
                for e in eng_names
                for m in mask_keys
                for ws in self.smoothing_windows
            }
            rce_ema.update(self._purchase_value_ema_keys())
            rce_ema.update(self._conversion_delay_slice_ema_keys(3))

        calib_ema = None
        if rce_ema is not None:
            calib_ema = {
                f"{e}/{m}/{ws}": jnp.zeros((2,), dtype=jnp.float32)
                for e in eng_names
                for m in mask_keys
                for ws in self.smoothing_windows
            }
            calib_ema.update(self._conversion_delay_slice_ema_keys(2))

        state = RecsysTrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            rng=rng,
            step=jnp.array(0),
            emb_table=emb_table,
            emb_table_state=emb_table_state,
            offset_keys=jnp.array([-1] * rep),
            offset_values=jnp.array([-1] * rep),
            post_embeddings=post_embeddings,
            rce_ema=rce_ema,
            calib_ema=calib_ema,
        )
        return state

    def _lookup(self, embedding_table_param: Parameter, token_ids: jax.Array) -> Parameter:
        data_axis = tuple(self.model_config.model_config.data_axis)
        token_ndim = token_ids.ndim
        if self.use_row_emb:
            token_ids = self._row_embedding_token_ids(token_ids)

            @shard_map(
                mesh=self.mesh,
                in_specs=(P("expert", None), P(data_axis)),
                out_specs=P(data_axis, *((None,) * token_ndim)),
                check_vma=False,
            )
            def lookup_rows(table, ids):
                ids = jax.lax.all_gather(ids, "expert", axis=0, tiled=True)
                local_ids = ids - jax.lax.axis_index("expert") * table.shape[0]
                values = jnp.take(table, local_ids, axis=0, mode="clip")
                values = jnp.where(
                    ((local_ids >= 0) & (local_ids < table.shape[0]))[..., None], values, 0
                )
                return jax.lax.psum_scatter(values, "expert", scatter_dimension=0, tiled=True)

            return replace(embedding_table_param, x=lookup_rows(embedding_table_param.x, token_ids))

        in_token_spec = P(data_axis, *((None,) * (token_ndim - 1)))
        out_spec = P(data_axis, *((None,) * token_ndim))

        @shard_map(
            mesh=self.mesh,
            in_specs=(P(None, "expert"), in_token_spec),
            out_specs=out_spec,
        )
        def _gather(table: jax.Array, ids: jax.Array) -> jax.Array:
            bs_per_device = ids.shape[0]

            all_ids = jax.lax.all_gather(ids, axis_name="expert", axis=0, tiled=True)
            ep = all_ids.shape[0] // bs_per_device

            result = table[all_ids]

            result = result.reshape(ep, bs_per_device, *result.shape[1:])

            result = jax.lax.all_to_all(
                result,
                axis_name="expert",
                split_axis=0,
                concat_axis=result.ndim - 1,
                tiled=True,
            )

            return result.reshape(result.shape[1:])

        new_x = _gather(embedding_table_param.x, token_ids)
        return replace(embedding_table_param, x=new_x)

    def _get_embedding_hash_leaves(self, data: RecsysFeaturesBatch) -> list[jax.Array]:
        use_ip = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_ip_address
        )
        use_user_embedding = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_user_embedding
        )
        use_post_embedding = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_post_embedding
        )
        leaves: list[jax.Array] = []
        if use_user_embedding:
            leaves.append(data["user_hashes"])
        if use_post_embedding:
            leaves.append(data["history_seq"]["post_hashes"])
        leaves.append(data["history_seq"]["auth_hashes"])
        if use_post_embedding:
            leaves.append(data["candidate_seq"]["post_hashes"])
        leaves.append(data["candidate_seq"]["auth_hashes"])
        if use_ip:
            leaves.append(data["user_ip_hashes"])
        return leaves

    def get_recsys_embeddings(
        self, data: RecsysFeaturesBatch, emb_table: Parameter
    ) -> RecsysEmbeddingsParameter:
        data_axis = tuple(self.model_config.model_config.data_axis)

        hash_leaves = self._get_embedding_hash_leaves(data)

        users = _num_users(data)
        flat_hashes = [x.reshape(users, -1) for x in hash_leaves]
        all_hashes = jax.lax.with_sharding_constraint(
            jnp.concatenate(flat_hashes, axis=1), P(data_axis)
        )
        all_embeddings = self._lookup(emb_table, all_hashes)
        all_embeddings = replace(
            all_embeddings,
            x=jax.lax.with_sharding_constraint(all_embeddings.x, P(data_axis, None, None)),
        )
        return self._unflatten_emb_lookup(data, all_embeddings)

    def _unflatten_emb_lookup(
        self, data: RecsysFeaturesBatch, table: Parameter
    ) -> RecsysEmbeddingsParameter:
        cfg = self.model_config
        has_emb_flags = isinstance(cfg, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
        hash_leaves = self._get_embedding_hash_leaves(data)
        lengths = [x.reshape(_num_users(data), -1).shape[1] for x in hash_leaves]
        splits = jnp.split(table.x, np.cumsum(lengths[:-1]), axis=-2)

        if self.using_seqpack:
            splits = [
                s.reshape(*s.shape[:-3], *leaf.shape, s.shape[-1])
                for s, leaf in zip(splits, hash_leaves)
            ]

        parts = iter(splits)
        segment = lambda on: replace(table, x=next(parts)) if on else None

        return RecsysEmbeddingsParameter(
            user_embeddings=segment(has_emb_flags and cfg.use_user_embedding),
            history_post_embeddings=segment(has_emb_flags and cfg.use_post_embedding),
            history_author_embeddings=segment(True),
            candidate_post_embeddings=segment(has_emb_flags and cfg.use_post_embedding),
            candidate_author_embeddings=segment(True),
            user_ip_embeddings=segment(has_emb_flags and cfg.use_ip_address),
        )

    def _flatten_emb_grads(self, grads: RecsysEmbeddingsParameter, users: int) -> jax.Array:
        cfg = self.model_config
        has_emb_flags = isinstance(cfg, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
        enabled = [
            (grads.user_embeddings, has_emb_flags and cfg.use_user_embedding),
            (grads.history_post_embeddings, has_emb_flags and cfg.use_post_embedding),
            (grads.history_author_embeddings, True),
            (grads.candidate_post_embeddings, has_emb_flags and cfg.use_post_embedding),
            (grads.candidate_author_embeddings, True),
            (grads.user_ip_embeddings, has_emb_flags and cfg.use_ip_address),
        ]
        segments = [p.x for p, on in enabled if on and p is not None]
        width = segments[0].shape[-1]
        return jnp.concatenate([s.reshape(s.shape[0], users, -1, width) for s in segments], axis=2)

    def local_global_hack(self, batch):
        batch = multihost_utils.global_array_to_host_local_array(
            batch,
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )

        batch = multihost_utils.host_local_array_to_global_array(
            batch,
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )
        return batch

    def _maybe_inject_global_neg_embeddings(
        self, batch: RecsysFeaturesBatch
    ) -> RecsysFeaturesBatch:
        return batch

    def _empty_history_augmentation_period(self) -> int:
        rate = self.empty_history_augmentation_rate
        return max(1, int(1.0 / rate)) if rate > 0 else 0

    def _apply_per_history_user_dropout(self, batch: RecsysFeaturesBatch, rate: float) -> int:
        if self._history_user_dropout_rng is None:
            self._history_user_dropout_rng = np.random.default_rng(
                self.rng_seed + self.data_rank + 1
            )

        bsz = batch["user_hashes"].shape[0]
        drop_mask = self._history_user_dropout_rng.random(bsz) < rate
        n_dropped = int(drop_mask.sum())
        if n_dropped == 0:
            return 0

        for arr in typing.cast("dict[str, np.ndarray | None]", batch["history_seq"]).values():
            if arr is not None:
                arr[drop_mask] = 0
        if batch.get("user_hashes") is not None:
            batch["user_hashes"][drop_mask] = 0
        if batch.get("user_ip_hashes") is not None:
            batch["user_ip_hashes"][drop_mask] = 0
        return n_dropped

    def dataset_with_prepare(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        assert isinstance(self.dataset, PhoenixDataset)
        aug_period = self._empty_history_augmentation_period()
        history_user_drop_rate = self.empty_history_user_dropout_rate
        microbatches: list[tuple[RecsysFeaturesBatch, dict[int, int] | None]] = []
        for i, (batch, offsets) in enumerate(dataset):
            batch = self._maybe_inject_global_neg_embeddings(batch)

            if history_user_drop_rate > 0.0:
                if i % self.num_microbatch == 0:
                    self._last_history_user_dropout_bsz = 0
                    self._last_history_user_dropout_count = 0
                self._last_history_user_dropout_bsz += batch["user_hashes"].shape[0]
                self._last_history_user_dropout_count += self._apply_per_history_user_dropout(
                    batch, history_user_drop_rate
                )

            microbatches.append((self.transform_batch(batch), offsets))
            if len(microbatches) == self.num_microbatch:
                for packed, packed_offsets in microbatches:
                    yield self.local_global_hack(super().prepare_data(packed)), packed_offsets
                microbatches = []

            if aug_period > 0 and i % aug_period == 0:
                for key in list(batch["history_seq"]):
                    if batch["history_seq"][key] is not None:
                        batch["history_seq"][key] = np.zeros_like(batch["history_seq"][key])
                if batch.get("user_hashes") is not None:
                    batch["user_hashes"] = np.zeros_like(batch["user_hashes"])
                if batch.get("user_ip_hashes") is not None:
                    batch["user_ip_hashes"] = np.zeros_like(batch["user_ip_hashes"])
                yield self.local_global_hack(self.prepare_data(batch)), offsets

    def dataset_without_prepare(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
    ) -> Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]]:
        assert isinstance(self.dataset, PhoenixDataset)
        for batch, offsets in dataset:
            if self.using_seqpack:
                assert isinstance(
                    self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
                )
                batch = pack_batch(
                    batch=batch,
                    num_devices_per_process=self.parallel_config.num_devices_per_process,
                    num_user_prefix_tokens=self.model_config.num_user_prefix_tokens,
                    dist=self.seqpack_distribution,
                    rng=np.random.default_rng(0),
                )
                if self.using_fa4:
                    batch = self.add_block_sparse_layout(batch)
            yield batch, offsets

    def dataset_thread_iterator(
        self,
        dataset: Iterator[tuple[RecsysFeaturesBatch, dict[int, int] | None]],
        stop=object(),
        prepare_data=True,
    ):
        assert isinstance(self.dataset, PhoenixDataset)
        if prepare_data:
            it = self.dataset_with_prepare(dataset)
            if self.use_async_emb:
                it = (
                    (tuple(batch for batch, _ in group), group[-1][1])
                    for group in itertools.batched(it, self.num_microbatch, strict=False)
                    if len(group) == self.num_microbatch
                )
        else:
            it = self.dataset_without_prepare(dataset)
        future = self.dataloading_thread.submit(next, it, stop)
        while True:
            result = future.result()
            if result is stop:
                return
            assert isinstance(result, tuple)
            future = self.dataloading_thread.submit(next, it, stop)

            batch, offsets = result

            if offsets:
                self.offsets_to_commit = offsets

            yield batch

    def _report_save_timing(self, metrics) -> None:
        if (
            self._pending_checksum_s is not None
            or self._pending_shmem_ckpt_write_s is not None
            or self._pending_gc_collect_s is not None
        ):
            ckpt_s = self._pending_checksum_s
            shmem_s = self._pending_shmem_ckpt_write_s
            gc_s = self._pending_gc_collect_s
            rank_logger.info(
                "save_checkpoint timings: checksum=%.2fs, shmem_write=%.2fs, gc_collect=%.2fs",
                ckpt_s or 0,
                shmem_s or 0,
                gc_s or 0,
            )
            if ckpt_s is not None:
                metrics["save_checkpoint/checksum_s"] = ckpt_s
                self._pending_checksum_s = None
            if shmem_s is not None:
                metrics["save_checkpoint/background_shmem_write_s"] = shmem_s
                self._pending_shmem_ckpt_write_s = None
            if gc_s is not None:
                metrics["save_checkpoint/gc_collect_s"] = gc_s
                self._pending_gc_collect_s = None

    def handle_metrics(
        self, soft_step: int, metrics: dict[str, float | int]
    ) -> dict[str, float | int]:
        metrics = super().handle_metrics(soft_step, metrics)

        if self.empty_history_user_dropout_rate > 0.0 and self._last_history_user_dropout_bsz > 0:
            metrics["train/empty_history_user_dropout_frac"] = (
                self._last_history_user_dropout_count / self._last_history_user_dropout_bsz
            )

        self._report_save_timing(metrics)

        if isinstance(self.dataset, PhoenixKafkaDataset) and self.dataset.export_metrics:
            report_training_metrics(
                training_name=self.dataset.name,
                metrics_dict=metrics,
            )

        return metrics

    def create_dataset(self, ctx: TrainerContext):
        assert isinstance(self.dataset, PhoenixDataset)
        if self.use_mock_gpu_client:
            logger.info("Using mock GPU client. Skipping dataset creation.")
            return

        data_rank, data_world_size = self.data_rank, self.data_world_size
        elapsed_samples = self.elapsed_samples
        if ctx.dataset_metadata_fn is not None:
            dataset_metadata = ctx.dataset_metadata_fn(self.data_rank, self.data_world_size)
            data_rank = dataset_metadata.data_rank
            data_world_size = dataset_metadata.data_world_size
            elapsed_samples = dataset_metadata.elapsed_samples

        if isinstance(self.dataset, RustKafkaDataset) and self.dataset.redistribution_reader_count(
            data_world_size
        ):
            if self.evals or self.empty_history_augmentation_rate > 0:
                raise ValueError("Redistribution requires one consumed batch per training step")

        assert elapsed_samples >= self.dataloader_offset, (
            f"elapsed_samples {elapsed_samples} < dataloader_offset {self.dataloader_offset}"
        )

        resume_position = self._data_position
        self._data_position = None
        should_reset_data = (
            self.reset_data_position
            and ctx.checkpoint is not None
            and ctx.checkpoint.is_manual_load()
        )
        if should_reset_data:
            resume_position = None
            rank_logger.info(
                "create_dataset: reset_data_position=True with manual load, ignoring saved data position"
            )
        rank_logger.info("create_dataset: resume_position=%s", resume_position)

        skip_rows = 0 if should_reset_data else max(0, elapsed_samples - self.dataloader_offset)
        loader_batch_size = self.read_bsz_per_process // self.num_microbatch
        train_dataset = self.dataset.make(
            batch_size=loader_batch_size,
            shard_index=data_rank,
            num_shards=data_world_size,
            server_hosts=ctx.ip_addrs,
            run_server=ctx.rank
            % (
                self.parallel_config.num_devices_per_node
                // self.parallel_config.num_devices_per_process
            )
            == 0,
            skip_rows=skip_rows,
            resume_position=resume_position,
        )
        self._eval_dataset = train_dataset

        if self.stop_at_data_end:
            assert isinstance(self.state, RecsysTrainingState)
            step = self.state.step.item()
            data_end = self.dataset.compute_max_steps(
                num_shards=data_world_size,
                batch_size=loader_batch_size,
                current_step=step,
                resume_position=resume_position,
            )
            if data_end is not None:
                remaining_batches = data_end - step + 1
                data_end = step + remaining_batches // self.num_microbatch - 1
                if self.max_steps is None or data_end < self.max_steps:
                    self.max_steps: int | None = data_end

        self.train_dataset = self.dataset_thread_iterator(train_dataset)

    def create_embedding_init_data(self) -> RecsysEmbeddingsParameter:
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)
        use_ip = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_ip_address
        )

        if self.using_seqpack:
            batch = self.example_data(self.read_bsz_per_process // self.num_microbatch)

            assert self.batch_size % self.bs_per_device == 0, (self.batch_size, self.bs_per_device)
            assert batch["user_hashes"].shape[1] == self.bs_per_device // self.num_microbatch, (
                batch["user_hashes"].shape[1],
                self.bs_per_device,
                self.num_microbatch,
            )
            assert self.num_devices == self.batch_size // self.bs_per_device, (
                self.num_devices,
                self.batch_size,
                self.bs_per_device,
            )

            emb_size = self.model_config.emb_table_width

            return RecsysEmbeddingsParameter(
                user_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (self.num_devices, *batch["user_hashes"].shape[1:], emb_size), jnp.bfloat16
                    )
                ),
                history_post_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["history_seq"]["post_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                history_author_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["history_seq"]["auth_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                candidate_post_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["candidate_seq"]["post_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                candidate_author_embeddings=EmbTable(
                    x=jax.ShapeDtypeStruct(
                        (
                            self.num_devices,
                            *batch["candidate_seq"]["auth_hashes"].shape[1:],
                            emb_size,
                        ),
                        jnp.bfloat16,
                    )
                ),
                user_ip_embeddings=(
                    EmbTable(
                        x=jax.ShapeDtypeStruct(
                            (self.num_devices, *batch["user_ip_hashes"].shape[1:], emb_size),
                            jnp.bfloat16,
                        )
                    )
                    if use_ip
                    else None
                ),
            )

        batch_size = self.batch_size // self.num_microbatch
        emb_size = self.model_config.emb_table_width
        user_emb_len = self.model_config.hash_table.num_user_hashes
        history_post_len = (
            self.model_config.hash_table.num_item_hashes * self.dataset.history_seq_len
        )
        history_author_len = (
            self.model_config.hash_table.num_author_hashes * self.dataset.history_seq_len
        )
        num_neg_blocks = 2 if self.dataset.search_query_embedding_dim > 0 else 1
        total_candidate_seq_len = (
            self.dataset.candidate_seq_len
            * (1 + num_neg_blocks * self.dataset.num_negatives_per_example)
            + self.dataset.num_global_negatives_per_example
        )
        candidate_post_len = self.model_config.hash_table.num_item_hashes * total_candidate_seq_len
        candidate_author_len = (
            self.model_config.hash_table.num_author_hashes * total_candidate_seq_len
        )
        if use_ip:
            user_ip_len = self.model_config.hash_table.num_ip_hashes
        else:
            user_ip_len = 0

        return RecsysEmbeddingsParameter(
            user_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, user_emb_len, emb_size), jnp.bfloat16)
            ),
            history_post_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, history_post_len, emb_size), jnp.bfloat16)
            ),
            history_author_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, history_author_len, emb_size), jnp.bfloat16),
            ),
            candidate_post_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, candidate_post_len, emb_size), jnp.bfloat16),
            ),
            candidate_author_embeddings=EmbTable(
                x=jax.ShapeDtypeStruct((batch_size, candidate_author_len, emb_size), jnp.bfloat16),
            ),
            user_ip_embeddings=(
                EmbTable(
                    x=jax.ShapeDtypeStruct((batch_size, user_ip_len, emb_size), jnp.bfloat16),
                )
                if use_ip
                else None
            ),
        )

    def _row_embedding_token_ids(self, ids: jax.Array) -> jax.Array:
        rows = self.dataset.input_vocab_size
        return jnp.clip(jnp.where(ids < 0, ids + rows, ids), 0, rows - 1)

    def get_flattened_token_ids(self, data: RecsysFeaturesBatch) -> jax.Array:
        use_ip = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_ip_address
        )
        use_user_embedding = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_user_embedding
        )
        use_post_embedding = (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_post_embedding
        )
        segments: list[jax.Array] = []
        if use_user_embedding:
            segments.append(data["user_hashes"])
        if use_post_embedding:
            segments.append(data["history_seq"]["post_hashes"])
        segments.append(data["history_seq"]["auth_hashes"])
        if use_post_embedding:
            segments.append(data["candidate_seq"]["post_hashes"])
        segments.append(data["candidate_seq"]["auth_hashes"])
        if use_ip:
            segments.append(data["user_ip_hashes"])
        users = _num_users(data)
        ids = jnp.concatenate([x.reshape(users, -1) for x in segments], axis=1)
        return self._row_embedding_token_ids(ids) if self.use_row_emb else ids

    def _segment_sum(
        self,
        emb_gradients_p: RecsysEmbeddingsParameter,
        inverse_indices: jax.Array,
        num_unique: int,
    ) -> jax.Array:
        if self.using_seqpack:
            segments: list[jax.Array] = []
            if emb_gradients_p.user_embeddings is not None:
                segments.append(emb_gradients_p.user_embeddings.x)
            if emb_gradients_p.history_post_embeddings is not None:
                segments.append(emb_gradients_p.history_post_embeddings.x)
            segments.append(emb_gradients_p.history_author_embeddings.x)
            if emb_gradients_p.candidate_post_embeddings is not None:
                segments.append(emb_gradients_p.candidate_post_embeddings.x)
            segments.append(emb_gradients_p.candidate_author_embeddings.x)
            if emb_gradients_p.user_ip_embeddings is not None:
                segments.append(emb_gradients_p.user_ip_embeddings.x)
            emb_width = segments[0].shape[-1]
            emb_gradients: jax.Array = jnp.concatenate(
                [x.reshape(self.batch_size, -1, emb_width) for x in segments], axis=1
            ).reshape(-1, emb_width)
        else:
            grad_segments: list[jax.Array] = []
            if emb_gradients_p.user_embeddings is not None:
                grad_segments.append(emb_gradients_p.user_embeddings.x)
            if emb_gradients_p.history_post_embeddings is not None:
                grad_segments.append(emb_gradients_p.history_post_embeddings.x)
            grad_segments.append(emb_gradients_p.history_author_embeddings.x)
            if emb_gradients_p.candidate_post_embeddings is not None:
                grad_segments.append(emb_gradients_p.candidate_post_embeddings.x)
            grad_segments.append(emb_gradients_p.candidate_author_embeddings.x)
            if emb_gradients_p.user_ip_embeddings is not None:
                grad_segments.append(emb_gradients_p.user_ip_embeddings.x)
            emb_width = grad_segments[0].shape[-1]
            emb_gradients = jnp.concatenate(
                grad_segments,
                axis=1,
            ).reshape((-1, emb_width))

        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        @shard_map(
            mesh=self.mesh,
            in_specs=(
                P("replica", "expert"),
                P("replica"),
            ),
            out_specs=P(None, "expert"),
            check_vma=False,
        )
        def _seg_sum(grads: jax.Array, tokens: jax.Array):
            acc = jax.ops.segment_sum(
                grads.astype(jnp.float32),
                tokens,
                num_segments=num_unique,
            ).astype(jnp.bfloat16)
            return jax.lax.psum(acc, axis_name="replica")

        return _seg_sum(emb_gradients, inverse_indices)

    def update(
        self,
        state: RecsysTrainingState,
        data: RecsysFeaturesBatch,
        lr: float,
    ):
        assert state.emb_table is not None
        assert state.emb_table_state is not None
        assert state.opt_state is not None
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )

        token_ids = self.get_flattened_token_ids(data)
        assert isinstance(self.dataset, PhoenixDataset)
        hash_vocab = getattr(self.dataset, "hash_vocab_size", None) or state.emb_table.x.shape[0]
        num_unique = min(token_ids.shape[0] * token_ids.shape[1], hash_vocab) + 1
        unique_tokens, inverse_indices = compress_token_ids(
            token_ids,
            fill_size=num_unique,
            fill_value=hash_vocab,
            output_sharding=P(("stage", *self.model_config.model_config.data_axis)),
        )

        rng, new_rng = jax.random.split(state.rng)

        fprop_params = state.params
        if self.precision_level < 3:
            fprop_params = jax.tree.map(cast_bfloat16, state.params)

        emb_table = state.emb_table
        recsys_embeddings = self.get_recsys_embeddings(data, emb_table)
        recsys_embeddings, emb_carry = self._emb_optim.transform_embeddings(
            recsys_embeddings, token_ids, state.emb_table_state
        )

        rce_ema = state.rce_ema
        calib_ema = state.calib_ema
        rce_alpha = None
        if rce_ema is not None and self.smoothing_windows:
            rce_alpha = jnp.array(
                [min(1.0, self.batch_size / w) for w in self.smoothing_windows],
                dtype=jnp.float32,
            )

        loss_and_grad_fn = jax.value_and_grad(self.loss_fn.apply, argnums=(0, 3), has_aux=True)
        (loss, stats), (gradients, emb_gradients) = loss_and_grad_fn(
            fprop_params, rng, data, recsys_embeddings, rce_ema, rce_alpha, calib_ema
        )

        if self.precision_level >= 2:
            gradients = jax.tree.map(lambda x: x.astype(jnp.float32), gradients)
        emb_gradients = jax.tree.map(lambda x: x.astype(jnp.bfloat16), emb_gradients)

        updates, new_opt_state = self.optim.update(gradients, state.opt_state, params=fprop_params)
        new_opt_state = typing.cast(InjectHyperparamsState, new_opt_state)
        new_params: Parameter = apply_updates(state.params, updates, lr)

        valid_step, grad_norm = self.is_valid_step(gradients)

        segment_sum_result = self._segment_sum(emb_gradients, inverse_indices, num_unique)

        emb_gradients = replace(
            emb_gradients.history_author_embeddings,
            x=segment_sum_result,
        )

        emb_valid_step, emb_grad_norm = self.is_valid_step(emb_gradients)
        keep_step = valid_step & emb_valid_step

        def _update(updated, original):
            return jnp.where(keep_step, updated, original)

        new_params = jax.tree.map(_update, new_params, state.params)
        new_opt_state = jax.tree.map(_update, new_opt_state, state.opt_state)

        examples = np.prod(data["user_hashes"].shape[:-1])

        metrics = {
            "step": state.step,
            "loss": loss,
            "examples_per_batch": examples,
            "valid_step": keep_step,
            "global_grad_norm": grad_norm,
            "learning_rate": lr * new_opt_state.hyperparams["learning_rate"],
            "weight_decay": new_opt_state.hyperparams["weight_decay"],
            "b1": new_opt_state.hyperparams["b1"],
            "b2": new_opt_state.hyperparams["b2"],
        }
        if self.track_norm_metrics:
            metrics.update(norm_metrics(new_params, new_opt_state, gradients, updates))

        new_rce_ema = stats.pop("_rce_ema", state.rce_ema)
        new_calib_ema = stats.pop("_calib_ema", state.calib_ema)
        metrics.update(**stats)

        new_emb_table, emb_new_opt_state, emb_optim_metrics = self._emb_optim.sparse_update(
            grads=emb_gradients,
            full_state=state.emb_table_state,
            full_emb_table=emb_table,
            unique_tokens=unique_tokens,
            num_unique=num_unique,
            lr=lr,
            valid_step=keep_step,
            carry=emb_carry,
        )

        metrics["emb_grad_norm"] = emb_grad_norm
        metrics["emb_valid_step"] = emb_valid_step
        metrics.update(emb_optim_metrics)

        new_state = RecsysTrainingState(
            params=new_params,
            opt_state=new_opt_state,
            rng=new_rng,
            step=state.step + 1,
            emb_table=new_emb_table,
            emb_table_state=emb_new_opt_state,
            offset_keys=state.offset_keys,
            offset_values=state.offset_values,
            post_embeddings=state.post_embeddings,
            rce_ema=new_rce_ema,
            calib_ema=new_calib_ema,
        )

        assert isinstance(self.dataset, PhoenixDataset)
        metrics["offset_zero_values"] = jnp.sum(state.offset_values == 0)
        metrics = jax.tree.map(lambda x: x.astype(jnp.float32), metrics)

        metric_keys, metric_values = zip(*metrics.items())
        metrics = {
            metric_keys: jnp.stack([jnp.float32(v) for v in metric_values], axis=0),
            ("valid_step",): keep_step,
        }

        return new_state, metrics, {}

    def _run_microbatches(
        self,
        microbatch_loss_fn: typing.Callable[..., typing.Any],
        params: Parameter,
        batches: RecsysFeaturesBatch,
        embeddings: RecsysEmbeddingsParameter,
        rngs: jax.Array,
        candidate_inputs: CandidateInputs,
        loss_normalizers: dict[str, jax.Array],
    ):
        ctx = self._async_emb_context
        assert ctx is not None
        rows = batches["user_hashes"].shape[1]
        leaves = jax.tree.leaves((batches, candidate_inputs))
        assert all(x.shape[1] == rows for x in leaves), "a leaf without rows"
        assert all(ctx.mesh.shape[a] == 1 for a in ctx.mesh.axis_names if a not in ctx.data_axis)
        rows_spec = P(None, ctx.data_axis)

        @shard_map(
            mesh=ctx.mesh,
            in_specs=(P(), rows_spec, rows_spec, P(), rows_spec, P()),
            out_specs=(P(), P(), P(), rows_spec, P(ctx.data_axis)),
            check_vma=False,
        )
        def run(params, batches, embeddings, rngs, candidate_inputs, loss_normalizers):
            def body(gradients, loss_normalizers, rng, batch, embeddings, candidate_inputs):
                rng = jax.random.fold_in(rng, jax.lax.axis_index(ctx.data_axis))
                (loss_k, (stats, metric_inputs)), (gradients_k, emb_gradients) = jax.value_and_grad(
                    functools.partial(microbatch_loss_fn, loss_normalizers=loss_normalizers),
                    argnums=(0, 3),
                    has_aux=True,
                )(params, rng, batch, embeddings, candidate_inputs=candidate_inputs)
                gradients = jax.tree.map(
                    lambda a, g: a + g.astype(jnp.float32), gradients, gradients_k
                )
                return gradients, (loss_k, stats, emb_gradients, metric_inputs)

            inputs = (rngs, batches, embeddings, candidate_inputs)
            inputs, loss_normalizers = jax.lax.optimization_barrier((inputs, loss_normalizers))
            gradients = jax.tree.map(lambda p: jnp.zeros(p.shape, jnp.float32), params)
            outputs = []
            for k in range(rngs.shape[0]):
                if k > 0:
                    inputs, gradients, outputs[-1] = jax.lax.optimization_barrier(
                        (inputs, gradients, outputs[-1])
                    )
                microbatch = jax.tree.map(operator.itemgetter(k), inputs)
                gradients, output = body(gradients, loss_normalizers, *microbatch)
                outputs.append(output)
            losses, stats, emb_gradients, metric_inputs = jax.tree.map(
                lambda *xs: jnp.stack(xs), *outputs
            )
            gradients = jax.tree.map(lambda g, p: g.astype(p.dtype), gradients, params)
            loss_sum, stats = jax.tree.map(lambda x: x.sum(0), (losses, stats))
            joined = jax.tree.map(
                lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]),
                (metric_inputs, batches),
            )
            return (
                *jax.lax.psum((gradients, loss_sum, stats), ctx.data_axis),
                emb_gradients,
                joined,
            )

        return run(params, batches, embeddings, rngs, candidate_inputs, loss_normalizers)

    def async_emb_step(
        self,
        state: RecsysTrainingState,
        data: tuple[RecsysFeaturesBatch, ...],
        lr: float,
        next_step_data: tuple[RecsysFeaturesBatch, ...],
        prev_step_lookup_pin: jax.Array,
    ):
        assert state.emb_table is not None
        assert state.emb_table_state is not None
        assert state.opt_state is not None
        ctx = self._async_emb_context
        assert ctx is not None

        rng, new_rng = jax.random.split(state.rng)
        fprop_params = state.params
        if self.precision_level < 3:
            fprop_params = jax.tree.map(cast_bfloat16, state.params)

        rce_alpha = None
        if state.rce_ema is not None and self.smoothing_windows:
            rce_alpha = jnp.array(
                [min(1.0, self.batch_size / w) for w in self.smoothing_windows],
                dtype=jnp.float32,
            )

        emb_table = state.emb_table
        token_ids = _step_token_ids([self.get_flattened_token_ids(b) for b in data], ctx.data_axis)
        prefetched_embeddings = recsys_async_emb.lookup_done(
            ctx, prev_step_lookup_pin, token_ids.shape
        )

        def dedup_tokens(token_ids: jax.Array) -> tuple[jax.Array, jax.Array]:
            unique_tokens, segment_ids = compress_token_ids(
                token_ids, fill_size=ctx.num_unique, fill_value=self._emb_hash_vocab
            )
            segment_ids = jax.lax.with_sharding_constraint(
                segment_ids.reshape(token_ids.shape), P(None, ctx.data_axis, None)
            )
            return unique_tokens, segment_ids

        unique_tokens = segment_ids = None
        if self.num_microbatch == 1 and not self.use_row_emb:
            fenced_token_ids, _ = jax.lax.optimization_barrier((token_ids, prefetched_embeddings))
            unique_tokens, segment_ids = dedup_tokens(fenced_token_ids)

        gate = prefetched_embeddings[0, :, 0, :1]

        if self.num_microbatch > 1:
            loss_inputs_fn, microbatch_loss_fn, metrics_fn = self.microbatch_loss_fns.apply
            data_shards = math.prod(self.mesh.shape[a] for a in ctx.data_axis)
            candidate_inputs, loss_normalizers = loss_inputs_fn(
                {}, None, data, self.num_microbatch * data_shards
            )
            batches = jax.tree.map(lambda *x: jnp.stack(x), *data)
            gate = recsys_async_emb.depend(
                ctx, gate, jnp.stack(jax.tree.leaves(loss_normalizers)).sum(), on_sharded=False
            )

        update_start_pin, updating_table, updating_emb_state, emb_optim_metrics = (
            self._emb_optim.gradient_update_start(
                ctx, emb_table.x, state.emb_table_state, gate=gate
            )
        )

        embeddings = self._unflatten_emb_lookup(
            data[0],
            replace(
                emb_table,
                x=jax.lax.with_sharding_constraint(
                    prefetched_embeddings, P(None, ctx.data_axis, None, None)
                ),
            ),
        )
        embeddings, _ = self._emb_optim.transform_embeddings(
            embeddings, token_ids, state.emb_table_state
        )
        next_step_token_ids = _step_token_ids(
            [self.get_flattened_token_ids(b) for b in next_step_data], ctx.data_axis
        ).astype(jnp.int32)

        if self.num_microbatch == 1:
            embeddings = jax.tree.map(lambda x: x[0], embeddings)
            candidate_authors = embeddings.candidate_author_embeddings
            embeddings = replace(
                embeddings,
                candidate_author_embeddings=replace(
                    candidate_authors,
                    x=recsys_async_emb.depend(
                        ctx, candidate_authors.x, update_start_pin, on_sharded=False
                    ),
                ),
            )

            def loss_fn(params, embeddings):
                return self.loss_fn.apply(
                    params, rng, data[0], embeddings, state.rce_ema, rce_alpha, state.calib_ema
                )

            loss, loss_vjp, stats = jax.vjp(loss_fn, fprop_params, embeddings, has_aux=True)

            (
                emb_grad_norm,
                emb_valid_step,
                emb_update_pending,
                updated_emb_state,
                grad_update_done_pin,
            ) = self._emb_optim.gradient_update_done(ctx, updating_emb_state, loss)
            updated_emb_state, grad_update_done_pin = jax.lax.optimization_barrier(
                (updated_emb_state, grad_update_done_pin)
            )

            next_step_token_ids, _ = jax.lax.optimization_barrier((next_step_token_ids, loss))
            new_emb_table, next_step_lookup_pin = recsys_async_emb.lookup_start(
                ctx, next_step_token_ids, updating_table, grad_update_done_pin
            )
            loss_cotangent, next_step_lookup_pin = jax.lax.optimization_barrier(
                (jnp.ones_like(loss), next_step_lookup_pin)
            )
            gradients, emb_gradients = loss_vjp(loss_cotangent)
            emb_gradients = jax.tree.map(lambda x: x[None], emb_gradients)
        else:
            new_emb_table, next_step_lookup_pin = recsys_async_emb.lookup_start(
                ctx, next_step_token_ids, updating_table, update_start_pin
            )
            loss_normalizers = recsys_async_emb.depend(
                ctx, loss_normalizers, next_step_lookup_pin, x_sharded=False
            )
            gradients, loss, stats, emb_gradients, (metric_inputs, batch) = self._run_microbatches(
                microbatch_loss_fn,
                fprop_params,
                batches,
                embeddings,
                jax.random.split(rng, self.num_microbatch),
                candidate_inputs,
                loss_normalizers,
            )
            if not self.use_row_emb:
                token_ids, gradients = jax.lax.optimization_barrier((token_ids, gradients))
                unique_tokens, segment_ids = dedup_tokens(token_ids)
                gradients, unique_tokens = jax.lax.optimization_barrier((gradients, unique_tokens))
            (
                emb_grad_norm,
                emb_valid_step,
                emb_update_pending,
                updated_emb_state,
                grad_update_done_pin,
            ) = self._emb_optim.gradient_update_done(ctx, updating_emb_state, loss)
            updated_emb_state, grad_update_done_pin = jax.lax.optimization_barrier(
                (updated_emb_state, grad_update_done_pin)
            )
            stats = metrics_fn(
                {},
                None,
                metric_inputs,
                batch,
                stats,
                state.rce_ema,
                rce_alpha,
                tuple(self.smoothing_windows) if state.rce_ema is not None else None,
                state.calib_ema,
            )
        emb_valid_step = emb_valid_step | ~emb_update_pending
        if self.precision_level >= 2:
            gradients = jax.tree.map(lambda x: x.astype(jnp.float32), gradients)

        updates, new_opt_state = self.optim.update(gradients, state.opt_state, params=fprop_params)
        new_opt_state = typing.cast(InjectHyperparamsState, new_opt_state)
        new_params: Parameter = apply_updates(state.params, updates, lr)

        valid_step, grad_norm = self.is_valid_step(gradients)

        emb_gradients = jax.tree.map(lambda x: x.astype(jnp.bfloat16), emb_gradients)
        deferred_emb_valid_step, deferred_emb_grad_norm = self.is_valid_step(emb_gradients)
        keep_step = valid_step & deferred_emb_valid_step

        def _update(updated, original):
            return jnp.where(keep_step, updated, original)

        new_params = jax.tree.map(_update, new_params, state.params)
        new_opt_state = jax.tree.map(_update, new_opt_state, state.opt_state)

        metrics = {
            "step": state.step,
            "loss": loss,
            "examples_per_batch": sum(np.prod(batch["user_hashes"].shape[:-1]) for batch in data),
            "valid_step": keep_step,
            "global_grad_norm": grad_norm,
            "learning_rate": lr * new_opt_state.hyperparams["learning_rate"],
            "weight_decay": new_opt_state.hyperparams["weight_decay"],
            "b1": new_opt_state.hyperparams["b1"],
            "b2": new_opt_state.hyperparams["b2"],
            "emb_grad_norm": emb_grad_norm,
            "emb_valid_step": emb_valid_step,
            "deferred_emb_grad_norm": deferred_emb_grad_norm,
            "deferred_emb_valid_step": deferred_emb_valid_step,
        }
        metrics.update(emb_optim_metrics)
        if self.track_norm_metrics:
            metrics.update(norm_metrics(new_params, new_opt_state, gradients, updates))

        new_rce_ema = stats.pop("_rce_ema", state.rce_ema)
        new_calib_ema = stats.pop("_calib_ema", state.calib_ema)
        metrics.update(**stats)
        metrics["offset_zero_values"] = jnp.sum(state.offset_values == 0)
        metrics = jax.tree.map(lambda x: x.astype(jnp.float32), metrics)

        metric_keys, metric_values = zip(*metrics.items())
        metrics = {
            metric_keys: jnp.stack([jnp.float32(v) for v in metric_values], axis=0),
            ("valid_step",): keep_step,
        }

        stage_pin = recsys_async_emb.stage_update(
            self._async_emb_context,
            self._flatten_emb_grads(emb_gradients, _num_users(data[0])),
            segment_ids,
            unique_tokens,
            keep_step,
            grad_update_done_pin,
        )
        next_step_lookup_pin = jnp.minimum(next_step_lookup_pin, stage_pin)

        new_state = RecsysTrainingState(
            params=new_params,
            opt_state=new_opt_state,
            rng=new_rng,
            step=state.step + 1,
            emb_table=replace(emb_table, x=new_emb_table),
            emb_table_state=updated_emb_state,
            offset_keys=state.offset_keys,
            offset_values=state.offset_values,
            post_embeddings=state.post_embeddings,
            rce_ema=new_rce_ema,
            calib_ema=new_calib_ema,
        )
        return new_state, metrics, {}, next_step_lookup_pin

    def async_emb_update(
        self,
        state: RecsysTrainingState,
        data: tuple[RecsysFeaturesBatch, ...],
        lr: float,
    ):
        assert self._batch_pipeline.reserve is not None
        try:
            if self._async_emb_lookup_pin is None:
                state, self._async_emb_lookup_pin = self._first_step_embedding_lookup_start_jit(
                    state, data
                )

            state, metrics, extras, self._async_emb_lookup_pin = self._async_emb_step_jit(
                state, data, lr, self._batch_pipeline.reserve.batch, self._async_emb_lookup_pin
            )
        except Exception:
            assert self._async_emb_context is not None
            recsys_async_emb.kernel_api(self._async_emb_context).abort(
                self._async_emb_context.context_id
            )
            raise

        return state, metrics, extras

    @property
    def using_seqpack(self) -> bool:
        return (
            isinstance(self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig))
            and self.model_config.use_seqpack
        )

    @property
    def using_fa4(self) -> bool:
        if not self.using_seqpack:
            return False
        if isinstance(self.model_config, RecsysAggregatedModelConfig):
            attn_config = self.model_config.model_config.attn_config
        elif isinstance(self.model_config, RecsysTwoTowerModelConfig):
            attn_config = self.model_config.user_tower_config.model_config.attn_config
        else:
            return False
        return getattr(attn_config, "attn_impl", None) == "cutedsl_ranker_varlen_attn"

    def add_block_sparse_layout(self, batch: RecsysFeaturesBatch) -> RecsysFeaturesBatch:
        from dataclasses import replace

        from xrex.cutedsl.ranker_attention_varlen_fa4 import build_block_sparse_layout

        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        layout = batch["packing_layout"]
        assert layout is not None, "FA4 path requires a packed batch"
        bs_per_device = layout.cu_seqlens.shape[1] - 1
        transformer_candidate_seq_len = layout.candidate_positions.shape[1] // bs_per_device
        _mc = (
            self.model_config.user_tower_config
            if isinstance(self.model_config, RecsysTwoTowerModelConfig)
            else self.model_config
        )
        cand_slot_lens = getattr(layout, "cand_slot_lens", None)
        block_sparse = build_block_sparse_layout(
            cu_seqlens=layout.cu_seqlens,
            transformer_candidate_seq_len=transformer_candidate_seq_len,
            max_history_seq_len=(_mc.num_user_prefix_tokens + _mc.history_seq_len),
            packed_seq_len=int(layout.segment_ids.shape[1]),
            padding_mask=layout.padding_mask,
            num_user_prefix_tokens=_mc.num_user_prefix_tokens,
            candidate_slot_lens=(
                np.asarray(cand_slot_lens) if cand_slot_lens is not None else None
            ),
        )
        return {**batch, "packing_layout": replace(layout, block_sparse=block_sparse)}

    @property
    def _seqpack_block_size(self) -> int:
        if isinstance(self.model_config, RecsysAggregatedModelConfig):
            attn_config = self.model_config.model_config.attn_config
        elif isinstance(self.model_config, RecsysTwoTowerModelConfig):
            attn_config = self.model_config.user_tower_config.model_config.attn_config
        else:
            raise ValueError(f"Unexpected model_config type {type(self.model_config)}")
        attn_impl = getattr(attn_config, "attn_impl", None)
        if attn_impl == "cutedsl_ranker_varlen_attn":
            from xrex.cutedsl.ranker_attention_varlen_fa4 import get_device_tuning_config

            return get_device_tuning_config().block_q
        if attn_impl == "pallas_ranker_varlen_attn":
            from xrex.pallas.ranker_attention_varlen import get_device_tuning_config

            return get_device_tuning_config().block_q
        raise ValueError(f"Unexpected attn_impl {attn_impl!r} for seqpack block size")

    def _row_emb_state_sharding(self, state_sharding):
        emb_state = state_sharding.emb_table_state
        if emb_state is None:
            return state_sharding
        row_sharding = NamedSharding(self.mesh, P("expert"))
        emb_state = emb_state._replace(row_sum_sq={**emb_state.row_sum_sq, "table": row_sharding})
        if emb_state.last_step is not None:
            emb_state = emb_state._replace(last_step={**emb_state.last_step, "table": row_sharding})
        return state_sharding._replace(emb_table_state=emb_state)

    def _create_async_emb_executables(self, init_data, lr_shape, compiler_options) -> None:
        if async_emb is None and not self.use_row_emb:
            raise RuntimeError(
                "use_async_emb requires the async_emb kernels (xrex.cuda.async_emb), "
                "which failed to import: no compiled binding was found, or the "
                "installed extension does not match this environment's NCCL."
            )

        data_axis = ("stage", *self.model_config.model_config.data_axis)
        assert int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8")) > 1
        assert os.environ.get("NCCL_RUNTIME_CONNECT") == "0"
        assert os.environ.get("NCCL_LAUNCH_ORDER_IMPLICIT") == "1"
        assert self.parallel_config.num_devices_per_process == 1
        assert "expert" in data_axis
        assert all(self.mesh.shape[a] == 1 for a in data_axis if a != "expert")
        assert isinstance(self._emb_optim, AsyncEmbOptimizer)

        tokens_per_example = jax.eval_shape(self.get_flattened_token_ids, init_data).shape[1]
        tokens_per_batch = self.batch_size * tokens_per_example
        data_shards = math.prod(self.mesh.shape[a] for a in data_axis)
        emb_width = self.state_shape.emb_table.x.shape[1]

        self._emb_hash_vocab = (
            getattr(self.dataset, "hash_vocab_size", None) or self.dataset.input_vocab_size
        )
        assert self.use_row_emb or self._emb_hash_vocab >= self.state_shape.emb_table.x.shape[0]
        assert self._emb_hash_vocab < 2**31
        num_unique_tokens = min(tokens_per_batch, self._emb_hash_vocab) + 1

        if self.use_row_emb:
            try:
                from xrex.cuda.row_emb import row_emb
            except ImportError as e:
                raise RuntimeError(f"use_row_emb requires the row_emb kernels: {e}") from e

            self._async_emb_context = row_emb.make_context_handle(
                self.mesh,
                ("expert",),
                data_axis=data_axis,
                tokens_per_batch=tokens_per_batch,
                emb_width=emb_width,
                vocab_rows=self.state_shape.emb_table.x.shape[0],
                recv_factor=self.row_emb_recv_factor,
                num_devices_per_node=self.parallel_config.num_devices_per_node,
            )
        else:
            self._async_emb_context = async_emb.make_context_handle(
                self.mesh,
                ("expert",),
                data_axis=data_axis,
                tokens_per_batch=tokens_per_batch,
                emb_width=emb_width,
                num_unique=num_unique_tokens,
                num_devices_per_node=self.parallel_config.num_devices_per_node,
            )

        row_sharding = NamedSharding(self.mesh, P(data_axis, None))

        def first_step_embedding_lookup_start(state, data):
            token_ids = _step_token_ids(
                [self.get_flattened_token_ids(b) for b in data], data_axis
            ).astype(jnp.int32)
            emb_table, lookup_pin = recsys_async_emb.lookup_start(
                self._async_emb_context,
                token_ids,
                state.emb_table.x,
                gate=token_ids[0, :1, :1].astype(jnp.float32),
            )
            return (state._replace(emb_table=replace(state.emb_table, x=emb_table)), lookup_pin)

        lookup_pin_shape = jax.ShapeDtypeStruct((data_shards, 1), jnp.float32)
        step_data = (init_data,) * self.num_microbatch

        self._first_step_embedding_lookup_start_jit = JittedOrCompiled(
            jax.jit(
                first_step_embedding_lookup_start,
                in_shardings=(self.state_sharding, self.data_sharding),
                out_shardings=(self.state_sharding, row_sharding),
                donate_argnums=(0,),
            )
        )
        self.register_jit_function(
            self._first_step_embedding_lookup_start_jit,
            self.state_shape,
            step_data,
            compiler_options=compiler_options,
        )

        self._async_emb_step_jit = JittedOrCompiled(
            jax.jit(
                self.async_emb_step,
                in_shardings=(
                    self.state_sharding,
                    self.data_sharding,
                    None,
                    self.data_sharding,
                    row_sharding,
                ),
                out_shardings=(self.state_sharding, None, None, row_sharding),
                donate_argnums=(0, 4),
            )
        )
        self.register_jit_function(
            self._async_emb_step_jit,
            self.state_shape,
            step_data,
            lr_shape,
            step_data,
            lookup_pin_shape,
            compiler_options=compiler_options,
        )

        self.update_jit = self.async_emb_update

    def create_executable(self) -> None:
        if self.num_microbatch < 1 or self.bs_per_device % self.num_microbatch:
            raise ValueError(
                f"num_microbatch={self.num_microbatch} must divide "
                f"bs_per_device={self.bs_per_device}"
            )
        if self.num_microbatch > 1:
            if not self.use_async_emb:
                raise ValueError("num_microbatch > 1 requires use_async_emb=True")
            if type(self.model_config) is not RecsysAggregatedModelConfig:
                raise ValueError("num_microbatch > 1 supports only the ranker")
            if self.empty_history_augmentation_rate > 0:
                raise ValueError(
                    "num_microbatch > 1 does not support empty_history_augmentation_rate"
                )

        self._init_shmem_write_pool()
        self._free_ports()
        self._init_incremental_state()

        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)

        if self.use_row_emb:
            if not self.use_async_emb:
                raise ValueError("use_row_emb requires use_async_emb=True")
            if self.emb_optim_config._active() != "rowwise_adagrad":
                raise ValueError("use_row_emb requires the rowwise_adagrad embedding optimizer")

        if isinstance(self.model_config, RecsysAggregatedModelConfig):
            ctx = self.model_config.context_features
        elif isinstance(self.model_config, RecsysTwoTowerModelConfig):
            ctx = self.model_config.user_tower_config.context_features
        else:
            ctx = None
        if ctx is not None and ctx.enabled:
            num_cat = max(
                (f.index + 1 for f in ctx.categorical_features if f.embedding_dim > 0),
                default=0,
            )
            num_cat = max(num_cat, 16) if num_cat > 0 else 0
            object.__setattr__(self.dataset, "num_context_categorical_features", num_cat)

        self.create_optim()
        self._emb_optim = self.emb_optim_config.make_optimizer(self.optim)

        @hk.transform
        def loss_fn(
            batch: RecsysFeaturesBatch,
            recsys_embeddings: RecsysEmbeddingsParameter,
            rce_ema: dict[str, jax.Array] | None = None,
            rce_alpha: jax.Array | None = None,
            calib_ema: dict[str, jax.Array] | None = None,
        ):
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            smoothing_windows = tuple(self.smoothing_windows) if rce_ema is not None else None
            if type(model) is RecsysAggregatedModel:
                loss, (stats, metric_inputs) = model.loss(batch, recsys_embeddings, True)
                return loss, model.metrics(
                    metric_inputs, batch, stats, rce_ema, rce_alpha, smoothing_windows, calib_ema
                )
            metrics_state_kwargs = {}
            if rce_ema is not None:
                metrics_state_kwargs = {
                    "rce_ema": rce_ema,
                    "rce_alpha": rce_alpha,
                    "smoothing_windows": smoothing_windows,
                    "calib_ema": calib_ema,
                }
            return model.loss(None, batch, recsys_embeddings, True, **metrics_state_kwargs)

        @hk.transform
        def loss_fn_eval(batch: RecsysFeaturesBatch, recsys_embeddings: RecsysEmbeddingsParameter):
            assert isinstance(
                self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
            )
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            if type(model) is RecsysAggregatedModel:
                loss, (stats, metric_inputs) = model.loss(batch, recsys_embeddings, False)
                return loss, model.metrics(metric_inputs, batch, stats)
            return model.loss(None, batch, recsys_embeddings, is_training=False)

        @hk.transform
        def user_forward_fn(
            batch: RecsysFeaturesBatch, recsys_embeddings: RecsysEmbeddingsParameter
        ):
            assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
            return self.model_config.make(
                sharding_context=make_legacy_sharding_context(self.mesh),
            ).user_embeddings_only(batch, recsys_embeddings, is_training=False)

        self.loss_fn = loss_fn
        self.loss_fn_eval = loss_fn_eval
        self.user_forward_fn: typing.Any = user_forward_fn

        if self.num_microbatch > 1:

            def microbatch_loss_fns():
                assert isinstance(self.model_config, RecsysAggregatedModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                return model.loss, (model.loss_inputs_and_normalizers, model.loss, model.metrics)

            self.microbatch_loss_fns = hk.multi_transform(microbatch_loss_fns)

        if isinstance(self.model_config, RecsysTwoTowerModelConfig):

            @hk.transform
            def two_tower_forward_fn(
                batch: RecsysFeaturesBatch,
                recsys_embeddings: RecsysEmbeddings,
                post_embeddings: jax.Array,
                dataset_types: jax.Array,
                top_k: int,
                target_dataset_types,
                eval_bs_per_device: int = 0,
            ):
                assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                results = model.forward(
                    batch,
                    recsys_embeddings,
                    post_embeddings,
                    dataset_types,
                    top_k,
                    target_dataset_types,
                    eval_bs_per_device=eval_bs_per_device,
                )
                top_k_indices, top_k_scores = results[0]
                return top_k_indices, top_k_scores

            @hk.transform
            def candidate_tower_forward_fn(
                post_author_embedding: jax.Array,
                post_sids: jax.Array,
                post_hashes: jax.Array,
                head_index: int = 0,
            ):
                assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                assert model.candidate_tower is not None
                return model.build_candidate_embeddings_from_lookups(
                    post_author_embedding,
                    post_sids,
                    post_hashes,
                    head_index=head_index,
                )

            @hk.transform
            def mol_side_table_fn(post_table: jax.Array):
                assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
                model = self.model_config.make(
                    sharding_context=make_legacy_sharding_context(self.mesh)
                )
                return model.mol_side_table_rows(post_table)

            self.two_tower_forward_fn = two_tower_forward_fn
            self.candidate_tower_forward_fn = candidate_tower_forward_fn
            self.mol_side_table_fn = mol_side_table_fn

        @hk.transform
        def forward_fn(
            batch: RecsysFeaturesBatch,
            recsys_embeddings: RecsysEmbeddings,
        ):
            assert isinstance(self.model_config, (RecsysAggregatedModelConfig))
            model = self.model_config.make(sharding_context=make_legacy_sharding_context(self.mesh))
            logits = model.forward(batch, recsys_embeddings)
            return logits

        self.forward_fn = forward_fn

        if self.using_seqpack:
            rng = jax.ShapeDtypeStruct((2,), jnp.uint32)
            init_data = typing.cast(
                RecsysFeaturesBatch,
                super().prepare_data(
                    self.example_data(self.read_bsz_per_process // self.num_microbatch)
                ),
            )
        else:
            rng, init_data = self._rng_and_init_data()
            if self.num_microbatch > 1:
                assert isinstance(self.dataset, PhoenixDataset)
                init_data = self.dataset.example_data_shape(self.batch_size // self.num_microbatch)
            init_data = typing.cast(RecsysFeaturesBatch, init_data)
        emb_table_init_data = self.create_embedding_init_data()

        lr_shape = jax.ShapeDtypeStruct((), dtype=jnp.float32)
        with self.mesh:
            self.state_shape = jax.eval_shape(self.init, init_data, rng)
            _, self.stats_shape = jax.eval_shape(
                self.loss_fn.apply,
                self.state_shape.params,
                rng,
                init_data,
                emb_table_init_data,
            )

        self.state_sharding = self.get_sharding(self.state_shape)
        if self.use_row_emb:
            self.state_sharding = self._row_emb_state_sharding(self.state_sharding)
        self.data_sharding = NamedSharding(
            self.mesh,
            P(("stage", *self.model_config.model_config.data_axis), ("seq", "model")),
        )

        compiler_options = None if not self.xla_flag_overrides else self.xla_flag_overrides

        def init(r):
            return self.init(init_data, r)

        self.init_jit = JittedOrCompiled(
            jax.jit(init, in_shardings=None, out_shardings=self.state_sharding)
        )
        self.register_jit_function(self.init_jit, rng)

        if self.use_async_emb:
            self._create_async_emb_executables(init_data, lr_shape, compiler_options)
        else:
            self.update_jit = JittedOrCompiled(
                jax.jit(
                    self.update,
                    in_shardings=(
                        self.state_sharding,
                        self.data_sharding,
                        None,
                    ),
                    out_shardings=(self.state_sharding, None, None),
                    donate_argnums=(0,),
                )
            )
            self.register_jit_function(
                self.update_jit,
                self.state_shape,
                init_data,
                lr_shape,
                compiler_options=compiler_options,
            )

        if self.dense_params_ema_decay > 0.0:
            self.update_jit = self._with_dense_params_ema(self.update_jit)

        self.forward_jit = JittedOrCompiled(
            jax.jit(
                self.forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=self.data_sharding,
            )
        )
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            self.candidate_tower_forward_jit = JittedOrCompiled(
                jax.jit(
                    self.candidate_tower_forward_fn.apply,
                    in_shardings=(
                        self.state_sharding.params,
                        None,
                        self.data_sharding,
                        self.data_sharding,
                        self.data_sharding,
                    ),
                    out_shardings=self.data_sharding,
                    static_argnums=(5,),
                )
            )

        self.create_extra_executables()
        self.maybe_create_eval_executables()

    def maybe_create_eval_executables(self):
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            self.two_tower_forward_jit = jax.jit(
                self.two_tower_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                    self.data_sharding,
                    None,
                ),
                out_shardings=(
                    None,
                    None,
                ),
                static_argnums=[6, 7, 8],
            )

            self.loss_fn_eval_jit = jax.jit(
                self.loss_fn_eval.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=(None, None),
            )

            self.user_forward_jit = jax.jit(
                self.user_forward_fn.apply,
                in_shardings=(
                    self.state_sharding.params,
                    None,
                    self.data_sharding,
                    self.data_sharding,
                ),
                out_shardings=self.data_sharding,
            )

    def _is_copy_port_binder(self) -> bool:
        hostnames = self.ctx.hostnames
        if not hostnames:
            return self.ctx.rank == 0
        return hostnames.index(hostnames[self.ctx.rank]) == self.ctx.rank

    def _copy_port_channel(self, port: int) -> grpc.Channel:
        cc = self.checkpoint_config
        if not cc.copy_port_tls_cert:
            return grpc.insecure_channel(f"127.0.0.1:{port}")
        if not cc.copy_port_tls_ca or not cc.copy_port_tls_server_name:
            raise ValueError(
                "copy_port_tls_ca and copy_port_tls_server_name are required for the "
                "loopback client when copy_port TLS is enabled"
            )
        if cc.copy_port_tls_client_ca and not (
            cc.copy_port_tls_client_cert and cc.copy_port_tls_client_key
        ):
            raise ValueError(
                "copy_port_tls_client_ca (mTLS) requires copy_port_tls_client_cert/key "
                "for the loopback client"
            )

        def read(p: str) -> bytes | None:
            return pathlib.Path(p).read_bytes() if p else None

        creds = grpc.ssl_channel_credentials(
            root_certificates=read(cc.copy_port_tls_ca),
            private_key=read(cc.copy_port_tls_client_key),
            certificate_chain=read(cc.copy_port_tls_client_cert),
        )
        options = (("grpc.ssl_target_name_override", cc.copy_port_tls_server_name),)
        return grpc.secure_channel(f"127.0.0.1:{port}", creds, options=options)

    def _free_ports(self):
        if port := self.checkpoint_config.copy_port:
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr and (conn.laddr.port == port or conn.laddr.port == port + 1):
                    try:
                        process = psutil.Process(conn.pid)
                        os.kill(process.pid, signal.SIGKILL)
                    except psutil.NoSuchProcess:
                        pass

    def _init_shmem_write_pool(self) -> None:
        self._shmem_write_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ckpt-shmem-write"
        )

    def _init_incremental_state(self) -> None:
        self._incremental_state = IncrementalState(
            unique_tokens=np.empty(0, dtype=np.int32),
            token_count=0,
            soft_step=0,
            checkpoint_path="",
        )

    def _copy_port_emb_host_spec(self) -> P | None:
        if not self.checkpoint_config.copy_port:
            return None
        if getattr(self.state_shape, "emb_table", None) is None:
            return None
        ep = self.mesh.shape["expert"]
        vocab = self.state_shape.emb_table.x.shape[0]
        if vocab % ep != 0:
            rank_logger.warning(
                "copy_port emb host row-shard disabled: vocab %d %% ep %d != 0 "
                "(falling back to the save-time device reshard + second host copy)",
                vocab,
                ep,
            )
            return None
        return P("expert", None)

    def adjust_host_sharding(self, host_sharding):
        spec = self._copy_port_emb_host_spec()
        if spec is None:
            return host_sharding
        emb_host = host_sharding.emb_table
        rank_logger.info(
            "copy_port: host emb_table layout %s -> %s (offload writes the flat-file "
            "row shards directly; no save-time duplicate host copy)",
            emb_host.spec,
            spec,
        )
        return host_sharding._replace(
            emb_table=NamedSharding(self.mesh, spec, memory_kind=emb_host.memory_kind)
        )

    def _finalize_checkpoint_load(
        self,
        ctx: TrainerContext,
        res: tuple[bool, int, int],
    ) -> tuple[bool, int, int]:
        self._data_position = None
        if res[0] and ctx.checkpoint is not None:
            ckpt_path = ctx.checkpoint.path
            pos_path = os.path.join(ckpt_path, _DATA_POSITION_FILENAME)
            rank_logger.info("Looking for data position at %s", pos_path)
            if os.path.isfile(pos_path):
                with open(pos_path) as f:
                    self._data_position = json.loads(f.read())
                rank_logger.info(
                    "Restored data position from filesystem checkpoint: %s, "
                    "current read_bsz_per_process=%d, data_world_size=%d",
                    self._data_position,
                    self.read_bsz_per_process,
                    self.data_world_size,
                )
            else:
                rank_logger.info("No data_position.json found at %s", pos_path)

        assert isinstance(self.state, RecsysTrainingState)
        if self.state.offset_keys is not None:
            assert self.state.offset_values is not None
            k = np.array(self.state.offset_keys).tolist()
            v = np.array(self.state.offset_values).tolist()
            if k[0] != -1 and isinstance(self.dataset, PhoenixKafkaDataset):
                ckpt_topic, ckpt_partitions = _read_checkpoint_kafka_config(ctx)
                current_topic = self.dataset.topic_name
                current_partitions = self.dataset.num_kafka_partitions

                skip_reason: str | None = None
                if ckpt_topic is not None and ckpt_topic != current_topic:
                    skip_reason = (
                        f"topic mismatch: checkpoint={ckpt_topic!r} != current={current_topic!r}"
                    )
                elif (
                    ckpt_partitions is not None
                    and current_partitions is not None
                    and ckpt_partitions != current_partitions
                ):
                    skip_reason = (
                        f"partition count changed: checkpoint={ckpt_partitions} "
                        f"!= current={current_partitions}"
                    )

                if skip_reason is not None:
                    rank_logger.warning(
                        "Skipping Kafka offset restore: %s. Consumer will start from %s.",
                        skip_reason,
                        self.dataset.auto_offset_reset,
                    )
                else:
                    if ckpt_topic is None:
                        rank_logger.info(
                            "No kafka config in checkpoint config.json (legacy). "
                            "Applying Kafka offsets as-is."
                        )
                    else:
                        rank_logger.info(
                            "Checkpoint kafka config matches: topic=%r, partitions=%s",
                            ckpt_topic,
                            ckpt_partitions,
                        )
                    seek_to_offset = dict(zip(k, v))
                    self.dataset.seek_to_offset = seek_to_offset
                    rank_logger.info(
                        "Restored Kafka offsets for %d partitions", len(seek_to_offset)
                    )

        if self.smoothing_windows and isinstance(self.model_config, RecsysAggregatedModelConfig):
            from xrex.data.recsys.constants import engagement_to_ids
            from xrex.models.recsys_model import build_metric_masks

            eng_names = list(engagement_to_ids(self.model_config.metric_group).keys())
            dummy = jnp.zeros((1, 1))
            dummy_3d = jnp.zeros((1, 1, self.model_config.model_config.output_vocab_size))
            mask_keys = list(
                build_metric_masks(
                    dummy,
                    dummy_3d,
                    dummy,
                    dummy,
                    trained_candidate_mask=dummy_3d,
                    ads_head_masking=self.model_config.ads_head_masking,
                    enable_platform_metrics=self.model_config.enable_platform_metrics,
                    metric_mask_keys=self.model_config.metric_mask_keys,
                ).keys()
            )
            loaded_rce = self.state.rce_ema
            reconciled_rce: dict[str, jax.Array] = {}
            for e in eng_names:
                for m in mask_keys:
                    for ws in self.smoothing_windows:
                        key = f"{e}/{m}/{ws}"
                        if loaded_rce and key in loaded_rce and loaded_rce[key].shape == (3,):
                            reconciled_rce[key] = loaded_rce[key]
                        else:
                            reconciled_rce[key] = jnp.zeros((3,), dtype=jnp.float32)
            for key, zeros in {
                **self._purchase_value_ema_keys(),
                **self._conversion_delay_slice_ema_keys(3),
            }.items():
                loaded = loaded_rce.get(key) if loaded_rce else None
                reconciled_rce[key] = (
                    loaded if loaded is not None and loaded.shape == zeros.shape else zeros
                )

            loaded_calib = self.state.calib_ema
            reconciled_calib: dict[str, jax.Array] = {}
            for e in eng_names:
                for m in mask_keys:
                    for ws in self.smoothing_windows:
                        key = f"{e}/{m}/{ws}"
                        if loaded_calib and key in loaded_calib and loaded_calib[key].shape == (2,):
                            reconciled_calib[key] = loaded_calib[key]
                        else:
                            reconciled_calib[key] = jnp.zeros((2,), dtype=jnp.float32)
            for key, zeros in self._conversion_delay_slice_ema_keys(2).items():
                loaded = loaded_calib.get(key) if loaded_calib else None
                reconciled_calib[key] = (
                    loaded if loaded is not None and loaded.shape == zeros.shape else zeros
                )

            self.state = self.state._replace(
                rce_ema=reconciled_rce,
                calib_ema=reconciled_calib,
            )

        if (
            res[0]
            and self.step_count_start is not None
            and ctx.checkpoint is not None
            and ctx.checkpoint.is_manual_load()
        ):
            rank_logger.info(
                "step_count_start=%d with manual load: resetting step and elapsed counters",
                self.step_count_start,
            )
            self.state = self.state._replace(step=jnp.array(self.step_count_start))
            res = (True, 0, 0)

        return res

    def purge_opt_state_on_load(self, host_state):
        if getattr(self.checkpoint_config, "keep_emb_opt_state", False):
            rank_logger.info("Not loading dense optimizer state (keeping emb_table_state)")
            return host_state._replace(opt_state=None)
        return super().purge_opt_state_on_load(host_state)

    def warm_start_staging_spec(self):
        if getattr(self.checkpoint_config, "keep_emb_opt_state", False):
            return (lambda tree: tree._replace(opt_state=None)), {"opt_state"}
        return super().warm_start_staging_spec()

    def restore_checkpoint_arrays(self, path, arrays, load_mask, rename, tag, on_replaced):
        if not self.use_row_emb and not isinstance(self.model_config, RecsysAggregatedModelConfig):
            return {}
        from xrex.train.row_embedding_restore import restore_row_embeddings

        return restore_row_embeddings(
            path,
            arrays,
            load_mask,
            rename,
            tag,
            on_replaced,
            logical_rows=self.dataset.input_vocab_size,
            row_sharded=self.use_row_emb,
            verify_checksums=self.checkpoint_config.verify_checksums,
        )

    def maybe_load_checkpoint(self, ctx: TrainerContext, tag=None):
        assert isinstance(
            self.model_config, (RecsysAggregatedModelConfig, RecsysTwoTowerModelConfig)
        )
        assert isinstance(self.dataset, PhoenixDataset)

        if self.store_load:
            raise ValueError(
                f"store_load={self.store_load!r} is not supported by the trainer; "
                "restore from a filesystem checkpoint with load=."
            )

        res = super().maybe_load_checkpoint(ctx, tag)
        return self._finalize_checkpoint_load(ctx, res)

    def next_data_batch(self):
        if self._shmem_write_future is not None and self._shmem_write_future.done():
            self._pending_shmem_ckpt_write_s = self._shmem_write_future.result()
            self._shmem_write_future = None

        if not self.use_async_emb:
            return super().next_data_batch()

        if self._batch_pipeline.exhausted:
            raise StopIteration

        assert isinstance(self.dataset, PhoenixDataset)
        dataset, load = self.dataset, super().next_data_batch

        def fetch() -> FetchedBatch:
            batch = load()
            return FetchedBatch(batch, dict(self.offsets_to_commit), dataset.get_data_position())

        if self._batch_pipeline.reserve is None:
            self._batch_pipeline.reserve = fetch()

        self._batch_pipeline.current = self._batch_pipeline.reserve

        try:
            self._batch_pipeline.reserve = fetch()
        except StopIteration:
            self._batch_pipeline.exhausted = True

        self.offsets_to_commit = self._batch_pipeline.current.offsets

        assert self._async_emb_context is not None
        ready_step = recsys_async_emb.kernel_api(self._async_emb_context).wait_step_ready(
            self._async_emb_context.context_id
        )
        assert ready_step != 0 or self._async_emb_lookup_pin is None
        return self._batch_pipeline.current.batch

    def _checkpoint_data_position(self) -> DataPosition | None:
        assert isinstance(self.dataset, PhoenixDataset)
        if not self.use_async_emb or self._batch_pipeline.current is None:
            return self.dataset.get_data_position()
        return self._batch_pipeline.current.data_position

    def _maybe_build_stablehlo_bundle(self) -> list | None:
        if not self.export_stablehlo_bundle:
            return None
        if self._stablehlo_bundle_files is None:
            from xrex.train.recsys_bundle_export import build_bundle

            try:
                start = time.perf_counter()
                self._stablehlo_bundle_files = build_bundle(self)
                rank_logger.info(
                    "Built StableHLO bundle (%d files, %.1fs); it will be included in "
                    "every copy_port checkpoint publish",
                    len(self._stablehlo_bundle_files),
                    time.perf_counter() - start,
                )
            except Exception:
                rank_logger.exception(
                    "StableHLO bundle export failed; disabling for the rest of this run "
                    "(copy_port checkpoints continue without export/)"
                )
                self._stablehlo_bundle_files = []
        if self._engine is None:
            return None
        return self._stablehlo_bundle_files or None

    def _write_stablehlo_bundle_files(self, prefix: str, bundle_files: list | None) -> None:
        from xrex.train.recsys_bundle_export import MANIFEST_NAME, restamp_manifest

        try:
            for bundle_file in bundle_files or ():
                data = bundle_file.data
                if bundle_file.name == MANIFEST_NAME:
                    data = restamp_manifest(data)
                bundle_path = f"{OUT_PATH}/.{prefix}/{bundle_file.name}"
                os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
                _write_all_bytes(bundle_path, memoryview(data))
        except OSError:
            rank_logger.exception(
                "StableHLO bundle write failed; disabling for the rest of this run "
                "(this publish continues without export/)"
            )
            self._stablehlo_bundle_files = []
            for subdir in {f.name.split("/", 1)[0] for f in bundle_files or ()}:
                shutil.rmtree(f"{OUT_PATH}/.{prefix}/{subdir}", ignore_errors=True)

    def _write_shmem_checkpoint(
        self,
        write_items: list[tuple[str, typing.Any, npt.NDArray]],
        checksums: dict,
        data_pos: typing.Any,
        prefix: str,
        stub: typing.Any,
        bundle_files: list | None = None,
    ) -> float:
        start = time.perf_counter()
        proc_idx = jax.process_index()
        proc_count = jax.process_count()
        has_engine = self._engine is not None
        devices_per_node = self.parallel_config.num_devices_per_node
        use_rdma = os.environ.get("XAI_RECSYS_RDMA") == "1"

        for key, value, arr in write_items:
            shard = value.addressable_shards[0]
            sharded = [(x.start, x.stop, x.step) != (None,) * 3 for x in shard.index]
            path = f"{OUT_PATH}/.{prefix}/{key}/c"
            if any(sharded):
                err = NotImplementedError(f"unsupported sharding in tensor {key}: {shard.index}")
                if sharded not in ([False, True], [True, False]):
                    raise err
                i = sharded.index(True)
                if shard.index[i].step is not None:
                    raise err
                delta = shard.index[i].stop - shard.index[i].start
                if shard.index[i].start % delta != 0 or value.shape[i] % delta != 0:
                    raise err
                shard_idx = shard.index[i].start // delta
                num_shards = value.shape[i] // delta
                if proc_count % num_shards != 0:
                    raise err
                path += (f"/{shard_idx}/0", f"/0/{shard_idx}")[i]
                tensor_written = proc_idx % (proc_count // num_shards) == 0
            else:
                suffix = "/0" * value.ndim
                path += suffix
                tensor_written = has_engine
            if tensor_written:
                os.makedirs(path[: path.rindex("/")], exist_ok=True)
                mv = memoryview(np.ndarray(shape=arr.nbytes, dtype=np.uint8, buffer=arr))
                _write_all_bytes(path, mv)
                actual_size = os.path.getsize(path)
                if actual_size != arr.nbytes:
                    raise RuntimeError(
                        f"bad checkpoint shard size for {key} at {path}: "
                        f"wanted {arr.nbytes}, got {actual_size}"
                    )
                if use_rdma and any(sharded) and not key.startswith("emb_table_state"):
                    d = proc_idx * devices_per_node // proc_count
                    request = copy_pb2.RegisterRequest(
                        path=f"{OUT_PATH}/.",
                        name=path[len(OUT_PATH) + 2 :],
                        rdma_device_indexes=bytes([d]),
                    )
                    try:
                        stub.Register(request)
                    except Exception as e:
                        raise RuntimeError(repr(e))

        from jax._src.distributed import global_state as _jax_dist

        assert _jax_dist.client is not None, "JAX distributed not initialized"
        _jax_dist.client.wait_at_barrier(f"recsys-save-checkpoint2-{prefix}", timeout_in_ms=300_000)

        if has_engine:
            with open(f"{OUT_PATH}/.{prefix}/checksums.0.json", "w") as f:
                json.dump(
                    {
                        "global_checksums": checksums,
                        "created_timestamp": time.time(),
                    },
                    f,
                )
            self._write_stablehlo_bundle_files(prefix, bundle_files)
            if data_pos is not None:
                with open(f"{OUT_PATH}/.{prefix}/{_DATA_POSITION_FILENAME}", "w") as f:
                    json.dump(data_pos, f)
                rank_logger.info(
                    "Saved data position (engine) to %s/.%s: %s", OUT_PATH, prefix, data_pos
                )
            name = prefix[: prefix.index("/")]
            all_names = os.listdir(OUT_PATH)
            old_entries = sorted(
                x for x in all_names if x.startswith("elapsed_samples") and x < name
            )
            keep_old = max(self.checkpoint_config.shm_max_entries - 1, 0)
            expired = old_entries if keep_old == 0 else old_entries[:-keep_old]
            for x in expired + [
                x
                for x in all_names
                if (x.startswith("elapsed_samples") and x >= name)
                or (x.startswith(".elapsed_samples") and x != f".{name}")
            ]:
                shutil.rmtree(f"{OUT_PATH}/{x}", ignore_errors=True)
            suffix = prefix[prefix.index("/") + 1 :]
            for x in os.listdir(f"{OUT_PATH}/.{name}"):
                if x != suffix:
                    shutil.rmtree(f"{OUT_PATH}/.{name}/{x}", ignore_errors=True)
            os.rename(f"{OUT_PATH}/.{name}", f"{OUT_PATH}/{name}")

        return time.perf_counter() - start

    def save_checkpoint(self, *args, **kwargs):
        if not self.dense_params_ema_in_checkpoint or self._dense_params_ema is None:
            return self._save_checkpoint_current_params(*args, **kwargs)
        live_params = self.state.params
        self.state = self.state._replace(
            params=self._dense_params_ema_as_params_jit(self._dense_params_ema, live_params)
        )
        try:
            return self._save_checkpoint_current_params(*args, **kwargs)
        finally:
            self.state = self.state._replace(params=live_params)

    def _save_checkpoint_current_params(self, *args, **kwargs):
        dataset = self.dataset
        assert isinstance(dataset, PhoenixDataset), f"Got {type(dataset)}"
        assert isinstance(self.state, RecsysTrainingState), f"Got {type(self.state)}"

        if self.offsets_to_commit or (
            isinstance(dataset, RustKafkaDataset)
            and dataset.redistribution_reader_count(jax.process_count())
        ):
            items_list = list(self.offsets_to_commit.items())

            n_partitions = dataset.num_kafka_partitions
            assert n_partitions is not None, "num_kafka_partitions must be set for Kafka offsets"
            max_size = (n_partitions + jax.process_count() - 1) // jax.process_count()
            if isinstance(dataset, RustKafkaDataset) and dataset.redistribution_reader_count(
                jax.process_count()
            ):
                max_size = n_partitions

            items_padded = items_list + [(-1, 0)] * (max_size - len(items_list))

            items_array = np.array(items_padded, dtype=np.int64)

            all_offsets = multihost_utils.process_allgather(items_array, tiled=True)

            if np.any(all_offsets[:, 0] >= 0):
                u = collections.defaultdict(int)
                for k, v in all_offsets:
                    u[k] = max(v, u[k])

                n = dataset.num_kafka_partitions
                assert n is not None
                _k = np.arange(n, dtype=all_offsets.dtype)
                _v = np.array([u.get(k, 0) for k in range(n)], dtype=all_offsets.dtype)

                offset_keys: jax.Array = multihost_utils.host_local_array_to_global_array(
                    _k, self.mesh, P()
                )
                offsets_values: jax.Array = multihost_utils.host_local_array_to_global_array(
                    _v, self.mesh, P()
                )

                self.state = self.state._replace(offset_keys=offset_keys)
                self.state = self.state._replace(offset_values=offsets_values)

        self.maybe_build_retrieval_post_embeddings()
        self.maybe_build_gen_recs_post_embeddings()

        force_disk_save = bool(kwargs.pop("force_disk_save", False))
        state_step = int(np.asarray(self.state.step).item())
        if self.max_steps is not None and state_step > self.max_steps:
            force_disk_save = True
        if self.max_samples is not None and self.elapsed_samples > self.max_samples:
            force_disk_save = True

        should_persist_disk = True
        disk_every_s = self.checkpoint_config.checkpoint_disk_every_s
        now = time.time()
        port = self.checkpoint_config.copy_port
        elapsed_since_disk_ckpt = -1.0
        if port and disk_every_s > 0:
            if force_disk_save:
                should_persist_disk = True
            elif jax.process_index() == 0 and self._last_disk_checkpoint_ts > 0:
                elapsed_since_disk_ckpt = now - self._last_disk_checkpoint_ts
                should_persist_disk = elapsed_since_disk_ckpt >= disk_every_s

            should_persist_disk = bool(
                multihost_utils.broadcast_one_to_all(
                    np.array([1 if should_persist_disk else 0], dtype=np.int32)
                )[0]
            )
            if not should_persist_disk and jax.process_index() == 0:
                rank_logger.info(
                    "Skipping disk checkpoint save at step=%s, elapsed_samples=%s; "
                    "copy_port publish continues (checkpoint_disk_every_s=%ss, %.1fs since last disk save)",
                    state_step,
                    self.elapsed_samples,
                    disk_every_s,
                    elapsed_since_disk_ckpt,
                )

        if port:
            if self._engine is None and self._is_copy_port_binder():
                cc = self.checkpoint_config
                self._engine = xai_recsys_engine.RecsysPredictorServer(
                    port,
                    port + 1,
                    1,
                    0,
                    copy_max_entries=cc.shm_max_entries,
                    tls_cert_path=cc.copy_port_tls_cert or None,
                    tls_key_path=cc.copy_port_tls_key or None,
                    tls_client_ca_path=cc.copy_port_tls_client_ca or None,
                )
            multihost_utils.sync_global_devices("recsys-copy-port-bind")

            checkpoint_path = self.get_checkpoint_path(self.ctx)
            prefix = "/".join(checkpoint_path.split("/")[-2:])
            wait_until_finished()
            if self._shmem_write_future is not None:
                self._pending_shmem_ckpt_write_s = self._shmem_write_future.result()
                self._shmem_write_future = None
            multihost_utils.sync_global_devices("recsys-save-checkpoint1")
            stub = copy_pb2_grpc.CopyStub(self._copy_port_channel(port))

            if not hasattr(self, "host_state") or self.host_state is None:
                self.host_state = jax.device_put(self.state, self.host_sharding)
            else:
                self.state, self.host_state = self.offload_state(self.state, self.host_state)
            jax.block_until_ready(self.host_state)

            assert isinstance(self.state, RecsysTrainingState)
            host_state = unwrap_tree(self.host_state)
            host_state = host_state.purge_opt_state()
            host_state = tree_to_dict(host_state)

            ep = self.mesh.shape["expert"]

            if self._copy_port_emb_host_spec() is not None:
                pass
            else:
                emb = _pad_vocab_for_ep(self.state.emb_table.x, ep, "emb_table")
                sharding = NamedSharding(self.mesh, P("expert", None))
                emb_table = jax.lax.with_sharding_constraint(emb, sharding.spec)
                host_state["emb_table"] = jax.device_put(
                    emb_table,
                    sharding.with_memory_kind("pinned_host"),
                )

            pe_key = "post_embeddings.embeddings"
            if (
                isinstance(self.model_config, RecsysTwoTowerModelConfig)
                and pe_key in host_state
                and host_state[pe_key] is not None
            ):
                pe_padded = _pad_vocab_for_ep(host_state[pe_key], ep, pe_key)
                if pe_padded is not host_state[pe_key]:
                    host_state[pe_key] = jax.device_put(
                        pe_padded,
                        pe_padded.sharding.with_memory_kind("pinned_host"),
                    )

            start_checksum = time.perf_counter()
            checksums = {}

            for key in host_state:
                value = host_state[key]
                if value is None:
                    continue
                shards = value.addressable_shards
                if len(shards) != 1:
                    continue
                shard = shards[0]
                sharded = [(x.start, x.stop, x.step) != (None,) * 3 for x in shard.index]
                shard_data = _unsafe_jax2np(shard.data)
                shard_bytes = np.ndarray(shape=shard_data.nbytes, dtype=np.uint8, buffer=shard_data)

                if not any(sharded):
                    checksums[key] = xai_recsys_engine.adler32_parallel(
                        np.ascontiguousarray(shard_bytes)
                    )
                elif key in ("emb_table", "post_embeddings.embeddings"):
                    local_checksum = xai_recsys_engine.adler32_parallel(
                        np.ascontiguousarray(shard_bytes)
                    )
                    local_len = shard_bytes.nbytes

                    all_checksums = multihost_utils.process_allgather(
                        np.array([local_checksum], dtype=np.uint32), tiled=True
                    )
                    local_len_parts = np.array([local_len], dtype=np.uint64).view(np.uint32)
                    all_lens_split = multihost_utils.process_allgather(
                        local_len_parts.reshape(1, 2), tiled=True
                    )

                    i = sharded.index(True)
                    delta = shard.index[i].stop - shard.index[i].start
                    num_shards = value.shape[i] // delta
                    n = jax.process_count()
                    processes_per_shard = n // num_shards

                    shard_info = []
                    for proc_idx in range(n):
                        proc_shard_idx = proc_idx // processes_per_shard
                        if proc_idx % processes_per_shard == 0:
                            ck = int(all_checksums[proc_idx])
                            len_lo = int(all_lens_split[proc_idx, 0])
                            len_hi = int(all_lens_split[proc_idx, 1])
                            length = len_lo | (len_hi << 32)
                            shard_info.append((proc_shard_idx, ck, length))
                    shard_info.sort(key=lambda x: x[0])

                    combined = shard_info[0][1]
                    for _, ck, length in shard_info[1:]:
                        combined = xai_recsys_engine.adler32_combine_py(combined, ck, length)
                    checksums[key] = combined
                else:
                    i = sharded.index(True)
                    delta = shard.index[i].stop - shard.index[i].start
                    shard_idx = shard.index[i].start // delta
                    num_shards = value.shape[i] // delta
                    total_cols = value.shape[1]
                    shard_width = delta
                    num_rows = value.shape[0]

                    local_partial = xai_recsys_engine.adler32_shard_partial(
                        np.ascontiguousarray(shard_bytes),
                        shard_width * shard_data.dtype.itemsize,
                        shard_idx,
                        total_cols * shard_data.dtype.itemsize,
                        128,
                    )

                    all_partials_flat = multihost_utils.process_allgather(
                        local_partial.astype(np.uint32), tiled=True
                    )
                    all_partials_2d = all_partials_flat.reshape(-1, 2)

                    n = jax.process_count()
                    processes_per_shard = n // num_shards
                    unique_partials = []
                    for proc_idx in range(n):
                        if proc_idx % processes_per_shard == 0:
                            proc_shard_idx = proc_idx // processes_per_shard
                            unique_partials.append((proc_shard_idx, all_partials_2d[proc_idx]))

                    unique_partials.sort(key=lambda x: x[0])
                    partials_array = np.stack([p[1] for p in unique_partials], axis=0).astype(
                        np.uint32
                    )

                    total_len = num_rows * total_cols * shard_data.dtype.itemsize
                    checksums[key] = xai_recsys_engine.adler32_combine_partials(
                        partials_array, total_len
                    )

            self._pending_checksum_s = time.perf_counter() - start_checksum

            write_items: list[tuple[str, typing.Any, npt.NDArray]] = []
            for key in host_state:
                value = host_state[key]
                if value is None:
                    continue
                shards = value.addressable_shards
                if len(shards) != 1:
                    continue
                write_items.append((key, value, _unsafe_jax2np(shards[0].data)))

            data_pos = self._checkpoint_data_position() if self._engine is not None else None

            bundle_files = self._maybe_build_stablehlo_bundle()

            self._shmem_write_future = self._shmem_write_pool.submit(
                self._write_shmem_checkpoint,
                write_items,
                checksums,
                data_pos,
                prefix,
                stub,
                bundle_files,
            )

        if port and not should_persist_disk:
            return None

        result = super().save_checkpoint(*args, **kwargs)
        self._last_disk_checkpoint_ts = now
        data_pos = self._checkpoint_data_position()
        if data_pos is not None:
            ckpt_path = self.get_checkpoint_path(self.ctx)
            pos_path = os.path.join(ckpt_path, _DATA_POSITION_FILENAME)
            os.makedirs(ckpt_path, exist_ok=True)
            with open(pos_path, "w") as f:
                json.dump(data_pos, f)
            rank_logger.info("Saved data position to %s: %s", pos_path, data_pos)

        gc_start = time.perf_counter()
        gc.collect()
        self._pending_gc_collect_s = time.perf_counter() - gc_start

        return result

    def run(self) -> None:
        gc.disable()
        rank_logger.info("Disabled Python cyclic GC for the recsys training loop")
        try:
            super().run()
        finally:
            exception_type = sys.exc_info()[0]
            if (
                exception_type is not None
                and not issubclass(exception_type, StopIteration)
                and self._async_emb_context is not None
            ):
                recsys_async_emb.kernel_api(self._async_emb_context).abort(
                    self._async_emb_context.context_id
                )

            shutdown = getattr(self.dataset, "shutdown", None)
            if callable(shutdown):
                rank_logger.info("Stopping dataset background threads via shutdown()")
                try:
                    shutdown()
                except Exception:
                    rank_logger.exception("dataset.shutdown() failed during teardown")

    def two_int32_to_int64(self, two_int32: jax.Array | np.ndarray) -> npt.NDArray[np.int64]:
        low_64 = np.asarray(two_int32[:, 0], dtype=np.int64) & 0xFFFFFFFF
        high_64 = np.asarray(two_int32[:, 1], dtype=np.int64) << 32
        return low_64 | high_64

    def int64_to_two_int32(self, post_ids: npt.NDArray[np.int64]) -> jax.Array:
        low_32 = jnp.asarray(post_ids & 0xFFFFFFFF, dtype=jnp.int32)
        high_32 = jnp.asarray(post_ids >> 32, dtype=jnp.int32)
        return jnp.stack([low_32, high_32], axis=1)

    def _with_dense_params_ema(self, update_fn):
        decay = self.dense_params_ema_decay
        params_sharding = self.state_sharding.params

        def ema_init(params):
            return jax.tree.map(lambda p: jnp.copy(p.astype(jnp.float32)), params)

        def ema_update(ema, params):
            return jax.tree.map(
                lambda e, p: decay * e + (1.0 - decay) * p.astype(jnp.float32), ema, params
            )

        def ema_gap(ema, params):
            pairs = zip(jax.tree.leaves(ema), jax.tree.leaves(params))
            gap_sq = sum(jnp.sum(jnp.square(e - p.astype(jnp.float32))) for e, p in pairs)
            norm_sq = sum(
                jnp.sum(jnp.square(p.astype(jnp.float32))) for p in jax.tree.leaves(params)
            )
            return jnp.sqrt(gap_sq), jnp.sqrt(norm_sq)

        self._dense_params_ema_init_jit = jax.jit(
            ema_init, in_shardings=(params_sharding,), out_shardings=params_sharding
        )
        self._dense_params_ema_update_jit = jax.jit(
            ema_update,
            in_shardings=(params_sharding, params_sharding),
            out_shardings=params_sharding,
            donate_argnums=(0,),
        )
        self._dense_params_ema_gap_jit = jax.jit(
            ema_gap, in_shardings=(params_sharding, params_sharding)
        )
        self._dense_params_ema_as_params_jit = jax.jit(
            lambda ema, params: jax.tree.map(lambda e, p: jnp.copy(e).astype(p.dtype), ema, params),
            in_shardings=(params_sharding, params_sharding),
            out_shardings=params_sharding,
        )

        def update_with_dense_params_ema(state, *args):
            state, metrics, extras = update_fn(state, *args)
            if self._dense_params_ema is None:
                self._dense_params_ema = self._dense_params_ema_init_jit(state.params)
                num_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(state.params))
                rank_logger.info(
                    f"dense params EMA started: decay {decay}, {num_params:,} params "
                    f"({num_params * 4 / 2**30:.2f} GiB in float32)"
                )
            else:
                self._dense_params_ema = self._dense_params_ema_update_jit(
                    self._dense_params_ema, state.params
                )
            return state, metrics, extras

        return update_with_dense_params_ema

    def eval(self, soft_step: int):
        if self._dense_params_ema is None:
            return self._eval_current_params(soft_step)
        live_params = self.state.params
        gap, norm = self._dense_params_ema_gap_jit(self._dense_params_ema, live_params)
        self.state = self.state._replace(
            params=jax.tree.map(lambda e, p: e.astype(p.dtype), self._dense_params_ema, live_params)
        )
        try:
            metrics = dict(self._eval_current_params(soft_step))
        finally:
            self.state = self.state._replace(params=live_params)
        metrics["dense_params_ema/relative_gap"] = float(gap) / max(float(norm), 1e-12)
        return metrics

    def _eval_current_params(self, soft_step: int):
        if isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return self.eval_two_tower(soft_step)

        raise ValueError("Ranking model eval_every_n is not supported yet.")

    def _split_home_checkpoint(self) -> bool:
        return bool(self.split_home_checkpoint) or bool(
            getattr(self.model_config, "split_home_checkpoint", False)
        )

    def maybe_build_retrieval_post_embeddings(self):
        if not isinstance(self.model_config, RecsysTwoTowerModelConfig):
            return

        assert isinstance(self.state, RecsysTrainingState)
        assert self.state.emb_table is not None

        if self.model_config.checkpoint_dataset_names is not None:
            valid_names = set(RetrievalDataset.__members__.keys())
            for name in self.model_config.checkpoint_dataset_names:
                if name not in valid_names:
                    raise ValueError(
                        f"Unknown dataset name '{name}' in checkpoint_dataset_names. "
                        f"Valid names: {sorted(valid_names)}"
                    )
            target_datasets = [
                RetrievalDataset[name] for name in self.model_config.checkpoint_dataset_names
            ]
            if self._split_home_checkpoint():
                target_datasets = RetrievalDataset.expand_home_to_cold_hot(target_datasets)
            rank_logger.info(
                f"Loading configured retrieval datasets: {[ds.name for ds in target_datasets]}"
            )
        else:
            eval_target_types: set[RetrievalDataset] = set()
            for eval_module in self.evals:
                if isinstance(eval_module.eval_conf, RecsysTwoTowerEval):
                    eval_target_types.add(eval_module.eval_conf.target_dataset_type)
            target_datasets = (
                list(eval_target_types) if eval_target_types else [RetrievalDataset.HOME]
            )
            if self._split_home_checkpoint():
                target_datasets = RetrievalDataset.expand_home_to_cold_hot(target_datasets)
            rank_logger.info(
                f"Loading configured retrieval datasets: {[ds.name for ds in target_datasets]}"
            )

        _user_cfg = self.model_config.user_tower_config
        _use_post_sid = _user_cfg.use_post_sid
        _sid_num_levels = _user_cfg.sid_num_levels
        max_posts = self.model_config.candidate_tower_config.max_posts
        if jax.process_index() == 0:
            post_ids, author_ids, dataset_types, post_sids_raw = RetrievalDataset.load_datasets(
                target_datasets,
                max_posts=max_posts,
                read_post_sid=_use_post_sid,
                sid_num_levels=_sid_num_levels,
                cold_start_max_age_seconds=float(
                    getattr(self.model_config, "cold_start_max_age_seconds", 0.0) or 0.0
                ),
            )
        else:
            post_ids = np.zeros(max_posts, dtype=np.int64)
            author_ids = np.zeros(max_posts, dtype=np.int64)
            dataset_types = np.zeros(max_posts, dtype=np.int32)
            post_sids_raw = None
        if post_sids_raw is None:
            post_sids_raw = np.full((max_posts, _sid_num_levels), -1, dtype=np.int32)
        post_ids_i32 = post_ids.view(np.int32)
        author_ids_i32 = author_ids.view(np.int32)
        post_ids_i32, author_ids_i32, dataset_types, post_sids_raw = (
            multihost_utils.broadcast_one_to_all(
                (post_ids_i32, author_ids_i32, dataset_types, post_sids_raw)
            )
        )
        post_ids = np.asarray(post_ids_i32, dtype=np.int32).view(np.int64)
        author_ids = np.asarray(author_ids_i32, dtype=np.int32).view(np.int64)
        dataset_types = np.asarray(dataset_types, dtype=np.int32)
        post_sids_raw = np.asarray(post_sids_raw, dtype=np.int32)
        total_samples = len(post_ids)
        shard_size = total_samples // self.data_world_size
        start_idx = self.data_rank * shard_size
        end_idx = (
            (self.data_rank + 1) * shard_size
            if self.data_rank < self.data_world_size - 1
            else total_samples
        )
        post_ids_shard = post_ids[start_idx:end_idx]
        author_ids_shard = author_ids[start_idx:end_idx]
        dataset_types_shard = dataset_types[start_idx:end_idx]

        _hash_table = self.model_config.candidate_tower_config.hash_table
        post_hashes_shard = _hash_table.get_item_hash(post_ids_shard)
        author_hashes_shard = _hash_table.get_author_hash(author_ids_shard)
        if _user_cfg.use_post_embedding:
            combined_hashes_shard = jnp.concatenate(
                [post_hashes_shard, author_hashes_shard], axis=1
            )
        else:
            combined_hashes_shard = author_hashes_shard
        post_sids_raw_shard = post_sids_raw[start_idx:end_idx]
        if _use_post_sid:
            post_sids_u16_shard = (post_sids_raw_shard + 1).astype(np.uint16)
        else:
            post_sids_u16_shard = np.zeros_like(post_sids_raw_shard, dtype=np.uint16)

        rng, _new_rng = jax.random.split(self.state.rng)

        combined_hashes_np = np.asarray(combined_hashes_shard)
        post_hashes_np = np.asarray(post_hashes_shard)
        _shard_rows = combined_hashes_np.shape[0]
        if total_samples % self.data_world_size == 0:
            _chunk_rows = min(_shard_rows, 65536)
        else:
            _chunk_rows = _shard_rows

        def _forward_chunked(head_index: int) -> jax.Array:
            outs: list[np.ndarray] = []
            for _start in range(0, _shard_rows, _chunk_rows):
                _sl = slice(_start, min(_start + _chunk_rows, _shard_rows))
                _hashes_jax = jax.make_array_from_process_local_data(
                    self.data_sharding, combined_hashes_np[_sl]
                )
                _pae = self._lookup(self.state.emb_table, _hashes_jax)
                _sids_jax = jax.make_array_from_process_local_data(
                    self.data_sharding, post_sids_u16_shard[_sl]
                )
                _ph_jax = jax.make_array_from_process_local_data(
                    self.data_sharding, post_hashes_np[_sl]
                )
                _out = self.candidate_tower_forward_jit(
                    self.state.params,
                    rng,
                    _pae.x,
                    _sids_jax,
                    _ph_jax,
                    head_index,
                )
                _local_shards = sorted(_out.addressable_shards, key=lambda s: s.index[0].start or 0)
                outs.append(np.concatenate([np.asarray(s.data) for s in _local_shards], axis=0))
                del _out, _pae
            return jax.make_array_from_process_local_data(
                self.data_sharding, np.concatenate(outs, axis=0)
            )

        head_dataset_mapping = getattr(self.model_config, "head_dataset_mapping", None)
        num_heads = self.model_config.candidate_tower_config.num_candidate_heads
        candidate_embeddings = _forward_chunked(0)
        if head_dataset_mapping is not None and num_heads > 1:
            dataset_to_head: dict[int, int] = {}
            for ds_name, head_idx in head_dataset_mapping.items():
                dataset_to_head[RetrievalDataset[ds_name].value] = head_idx
            head_indices_shard = np.array(
                [dataset_to_head.get(int(dt), 0) for dt in dataset_types_shard],
                dtype=np.int32,
            ).reshape(-1, 1)
            post_head_jax = jax.make_array_from_process_local_data(
                self.data_sharding,
                head_indices_shard,
            )
            for h in range(1, num_heads):
                emb_h = _forward_chunked(h)
                mask_h = post_head_jax == h
                candidate_embeddings = jnp.where(mask_h, emb_h, candidate_embeddings)
                del emb_h

        global_post_ids = multihost_utils.host_local_array_to_global_array(
            self.int64_to_two_int32(post_ids), self.mesh, P(None)
        )
        global_author_ids = multihost_utils.host_local_array_to_global_array(
            self.int64_to_two_int32(author_ids), self.mesh, P(None)
        )
        combined_dataset_types = multihost_utils.process_allgather(
            dataset_types_shard, tiled=True
        ).reshape(-1, 1)

        original_embeddings = self.state.post_embeddings.embeddings
        mol_side_table = self.state.post_embeddings.mol_side_table
        if mol_side_table is not None:
            if self.mol_side_table_jit is None:
                self.mol_side_table_jit = jax.jit(
                    self.mol_side_table_fn.apply,
                    in_shardings=(self.state_sharding.params, None, self.data_sharding),
                    out_shardings=NamedSharding(self.mesh, P(None, None)),
                )
            t_side = time.time()
            side_rows = self.mol_side_table_jit(self.state.params, None, candidate_embeddings)
            mol_side_table = replace(mol_side_table, x=side_rows, pspec=P(None, None))
            rank_logger.info(
                "MoL serving side table %s built in %.1fs",
                tuple(side_rows.shape),
                time.time() - t_side,
            )
        post_embeddings = PostEmbeddings(
            post_ids=global_post_ids,
            author_ids=global_author_ids,
            embeddings=replace(
                original_embeddings, x=candidate_embeddings, pspec=self.data_sharding.spec
            ),
            dataset_types=combined_dataset_types,
            mol_side_table=mol_side_table,
        )
        self.state = self.state._replace(post_embeddings=post_embeddings)

    def maybe_build_gen_recs_post_embeddings(self) -> None:
        pass

    def eval_two_tower(self, soft_step: int):
        assert isinstance(self.state, RecsysTrainingState)
        assert isinstance(self.model_config, RecsysTwoTowerModelConfig)
        rank_logger.info("Running evals at %s", soft_step)
        rng, _new_rng = jax.random.split(self.state.rng)

        def _two_tower_eval_forward_fn(
            batch, post_embeddings, dataset_types, top_k, target_dataset_types, eval_bs_per_device=0
        ):
            recsys_embeddings_parameter = self.get_recsys_embeddings(
                batch,
                self.state.emb_table,
            )
            recsys_embeddings = get_recsys_embed_param_to_jax_array(recsys_embeddings_parameter)
            recsys_embeddings = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), recsys_embeddings
            )
            return self.two_tower_forward_jit(
                self.state.params,
                rng,
                batch,
                recsys_embeddings,
                post_embeddings,
                dataset_types,
                top_k,
                target_dataset_types,
                eval_bs_per_device,
            )

        _eval_user_cfg = self.model_config.user_tower_config

        head_dataset_mapping = getattr(self.model_config, "head_dataset_mapping", None)

        def _make_candidate_tower_eval_fn(head_idx: int = 0):
            def _fn(candidate_embeddings, post_sids=None, post_hashes=None):
                if post_sids is None:
                    post_sids_local = jnp.zeros(
                        (candidate_embeddings.shape[0], _eval_user_cfg.sid_num_levels),
                        dtype=jnp.uint16,
                    )
                else:
                    post_sids_local = post_sids
                if post_hashes is None:
                    post_hashes_local = jnp.zeros(
                        (
                            candidate_embeddings.shape[0],
                            _eval_user_cfg.hash_table.hash_keys.num_item_hashes,
                        ),
                        dtype=jnp.int64,
                    )
                else:
                    post_hashes_local = post_hashes
                return self.candidate_tower_forward_jit(
                    self.state.params,
                    rng,
                    candidate_embeddings,
                    post_sids_local,
                    post_hashes_local,
                    head_idx,
                )

            return _fn

        _candidate_tower_eval_forward_fn = _make_candidate_tower_eval_fn(0)

        def _loss_fn(batch: RecsysFeaturesBatch):
            recsys_embeddings_param: RecsysEmbeddingsParameter = self.get_recsys_embeddings(
                batch,
                self.state.emb_table,
            )
            recsys_embeddings_param = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), recsys_embeddings_param
            )
            return self.loss_fn_eval_jit(self.state.params, rng, batch, recsys_embeddings_param)

        def _user_emb_fn(batch: RecsysFeaturesBatch):
            recsys_embeddings_param = self.get_recsys_embeddings(batch, self.state.emb_table)
            recsys_embeddings_param = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, self.data_sharding), recsys_embeddings_param
            )
            return self.user_forward_jit(self.state.params, rng, batch, recsys_embeddings_param)

        all_eval_metrics = {}
        if self.state.post_embeddings is None or self.state.post_embeddings.post_ids is None:
            return all_eval_metrics

        if not self._retrieval_post_emb_built:
            rank_logger.info(
                "Building retrieval post-embedding table for the first time "
                "(eval would otherwise score against an uninitialized table)."
            )
            self.maybe_build_retrieval_post_embeddings()
            self._retrieval_post_emb_built = True

        for eval_module in self.evals:
            eval_name, eval_conf, _, eval_bs_per_device, _ = (
                eval_module.eval_name,
                eval_module.eval_conf,
                eval_module.eval_dataset,
                eval_module.eval_bs_per_device,
                eval_module.eval_seq_len,
            )
            if not isinstance(eval_conf, RecsysTwoTowerEval):
                continue

            eval_head_idx = 0
            if head_dataset_mapping is not None:
                eval_head_idx = head_dataset_mapping.get(eval_conf.target_dataset_type.name, 0)
            eval_cand_fn = _make_candidate_tower_eval_fn(eval_head_idx)

            eval_results = run_recsys_evals(
                evals=[(eval_name, eval_conf)],
                train_dataset=self.dataset_thread_iterator(self._eval_dataset, prepare_data=False),
                forward_fn=_two_tower_eval_forward_fn,
                loss_fn=_loss_fn,
                mesh=self.mesh,
                rng=rng,
                candidate_tower_forward_fn=eval_cand_fn,
                all_post_embeddings=self.state.post_embeddings.embeddings.x,
                all_post_ids=self.two_int32_to_int64(self.state.post_embeddings.post_ids),
                dataset_types=self.state.post_embeddings.dataset_types,
                eval_bs_per_device=eval_bs_per_device,
                user_emb_fn=_user_emb_fn,
            )

            eval_metrics = report_forward_eval_results(results=eval_results)
            eval_metrics = merge_report_extras(eval_results, eval_metrics)
            for k, v in (getattr(eval_conf, "train_metrics", None) or {}).items():
                eval_metrics[f"{eval_name}/train/{k}"] = v
            rank_logger.info(f"eval_metrics: {eval_metrics}")
            all_eval_metrics.update(eval_metrics)
        return all_eval_metrics
