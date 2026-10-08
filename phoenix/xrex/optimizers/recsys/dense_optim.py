# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
from __future__ import annotations

from typing import Optional

import optax
from jax.sharding import Mesh

from xai_configlib import configclass
from xrex.optimizers.optim import (
    OptimConfig,
    _instantiate,
    inject_hyperparams,
    scale_by_lr_multiplier,
)
from xrex.optimizers.recsys.muon import make_muon_optimizer


@configclass
class RecsysDenseOptimConfig(OptimConfig):
    muon_beta: float = 0.95
    muon_ns_steps: int = 5
    muon_adam_patterns: str = "embed,_emb,logits,vocab,norm,table"
    muon_consistent_rms: Optional[float] = None
    muon_ns_dtype: str = "bfloat16"
    muon_preconditioning: str = "frobenius"
    muon_split_fused: str = ""
    muon_ns_shard_group_size: int = 1
    muon_matrix_weight_decay: float = 0.0
    adam_embedding_weight_decay: float = 0.0
    adam_embedding_decay_patterns: str = "embed,_emb,logits,vocab,table"

    def make(self, mesh: Mesh | None = None):
        if self.optim != "muon":
            return super().make(mesh)

        optimizer = make_muon_optimizer(self, mesh)

        @inject_hyperparams
        def schedule_optim(learning_rate, b1, b2, weight_decay):
            return optax.chain(
                optax.clip_by_global_norm(self.clip_by_global_norm),
                optimizer(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay),
                scale_by_lr_multiplier(),
            )

        schedule_args = map(_instantiate, (self.learning_rate, self.b1, self.b2, self.weight_decay))
        return schedule_optim(*schedule_args)
