# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import enum

import jax


class RematType(enum.IntEnum):
    WHOLE = 0
    SAVE_GB300_RECSYS = 29
    SAVE_H100_RECSYS = 32
    SAVE_H100_QKV_ATTN_FFN = 34


def custom_remat_policy(policy: RematType):
    if policy == RematType.WHOLE:
        return jax.checkpoint_policies.save_only_these_names(
            "scalar_stats",
        )
    elif policy == RematType.SAVE_GB300_RECSYS:
        return jax.checkpoint_policies.save_only_these_names(
            "attn_outputs",
            "dense_outputs",
            "dense_outputs_individual",
            "attn",
            "gate_up_proj",
            "dense_up_proj",
            "query_heads",
            "key_heads",
            "value_heads",
            "scalar_stats",
        )
    elif policy == RematType.SAVE_H100_RECSYS:
        return jax.checkpoint_policies.save_only_these_names(
            "attn_outputs",
            "dense_outputs",
            "dense_outputs_individual",
            "attn",
            "gate_up_proj",
            "dense_up_proj",
            "query_heads_rope",
            "key_heads_rope",
            "cutedsl_attn_outputs",
            "query_heads",
            "key_heads",
            "value_heads",
            "scalar_stats",
        )
    elif policy == RematType.SAVE_H100_QKV_ATTN_FFN:
        return jax.checkpoint_policies.save_only_these_names(
            "query_heads",
            "key_heads",
            "value_heads",
            "query_heads_rope",
            "key_heads_rope",
            "cutedsl_attn_outputs",
            "attn_outputs",
            "dense_up_proj",
            "scalar_stats",
        )
    else:
        raise NotImplementedError(f"Unknown remat policy: {policy}")
