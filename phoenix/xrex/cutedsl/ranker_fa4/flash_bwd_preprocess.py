# BSD 3-Clause License
#
# Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Derived from flash-attention (https://github.com/Dao-AILab/flash-attention);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import math
import operator
from functools import partial
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import Arch, BaseDSL

from xrex.cutedsl.quack import copy_utils, layout_utils
from xrex.cutedsl.quack.cute_dsl_utils import ParamsBase
from xrex.cutedsl.ranker_fa4 import utils
from xrex.cutedsl.ranker_fa4.seqlen_info import SeqlenInfo
from xrex.cutedsl.ranker_fa4.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FlashAttentionBackwardPreprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        tile_m: int = 128,
        num_threads: int = 256,
        use_padded_offsets: bool = True,
    ):
        self.use_pdl = BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a
        self.dtype = dtype
        self.tile_m = tile_m
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.num_threads = num_threads
        self.use_padded_offsets = use_padded_offsets

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < tile_m:
            return False
        return True

    def _setup_attributes(self):
        gmem_k_block_size = (
            128
            if self.head_dim_v_padded % 128 == 0
            else (
                64
                if self.head_dim_v_padded % 64 == 0
                else (32 if self.head_dim_v_padded % 32 == 0 else 16)
            )
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = gmem_k_block_size // num_copy_elems
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // Float32.width
        assert (
            self.tile_m * self.head_dim_padded // num_copy_elems_dQaccum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            Float32, self.num_threads, num_copy_elems_dQaccum
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mdLSE: Optional[cute.Tensor],
        stream: cuda.CUstream = None,
    ):
        if const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mPdPsum.element_type not in [Float32]):
            raise TypeError("PdPsum tensor must be Float32")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if const_expr(mLSE is not None):
            assert mLSElog2 is not None, "If mLSE is provided, mLSElog2 must also be provided"
            if const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
            if const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")
        if const_expr(mdLSE is not None):
            if const_expr(mdLSE.element_type not in [Float32]):
                raise TypeError("dLSE tensor must be Float32")

        self._setup_attributes()

        transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mPdPsum = layout_utils.select(mPdPsum, transpose)
        if const_expr(mLSE is not None):
            mLSE = layout_utils.select(mLSE, transpose)
            mLSElog2 = layout_utils.select(mLSElog2, transpose)
        if const_expr(mdLSE is not None):
            mdLSE = layout_utils.select(mdLSE, transpose)
        if const_expr(mdQaccum is not None):
            mdQaccum = layout_utils.select(mdQaccum, transpose)

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_head = mO.shape[1]
            num_batch = mCuSeqlensQ.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_head = mO.shape[2]
            num_batch = mO.shape[0]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[1], self.tile_m),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[2],
            total_q=mO.shape[0],
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            mCuSeqlensQ,
            mSeqUsedQ,
            mdLSE,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mdLSE: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if work_tile.is_valid_tile:
            seqlen = SeqlenInfo.create(
                batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m
            )
            mO_cur = seqlen.offset_batch(mO, batch_idx, dim=0)[None, head_idx, None]
            mdO_cur = seqlen.offset_batch(mdO, batch_idx, dim=0)[None, head_idx, None]
            stats_use_padded_offsets = self.use_padded_offsets
            if const_expr(mdQaccum is not None):
                stats_use_padded_offsets = True
            mPdPsum_cur = seqlen.offset_batch(
                mPdPsum, batch_idx, dim=2, padded=stats_use_padded_offsets
            )[None, head_idx]
            headdim_v = mO_cur.shape[cute.rank(mO_cur) - 1]
            seqlen_q = seqlen.seqlen
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
            seqlen_limit = seqlen_q - m_block * self.tile_m

            lse = None
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch(mLSE, batch_idx, dim=2)[None, head_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_limit:
                    lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = None
            if const_expr(self.check_hdim_v_oob):
                tOpO = copy_utils.predicate_k(tOcO, limit=headdim_v)
            copy = partial(copy_utils.copy, pred=tOpO)

            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)
            if const_expr(self.check_hdim_v_oob):
                tOrO.fill(0.0)
                tOrdO.fill(0.0)
            assert tOgO.shape == tOgdO.shape
            for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                    copy(tOgO[None, m, None], tOrO[None, m, None])
                    copy(tOgdO[None, m, None], tOrdO[None, m, None])
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
            )
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            pdpsum = utils.warp_reduce(pdpsum, operator.add, width=threads_per_row)
            PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            PdP_sum.store(pdpsum)

            gdLSE = None
            if const_expr(mdLSE is not None):
                mdLSE_cur = seqlen.offset_batch(mdLSE, batch_idx, dim=2)[None, head_idx]
                gdLSE = cute.local_tile(mdLSE_cur, (self.tile_m,), (m_block,))

            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    PdPsum_val = 0.0
                    if row < seqlen_limit:
                        PdPsum_val = PdP_sum[m]
                        if const_expr(mdLSE is not None):
                            PdPsum_val -= gdLSE[row]
                    gPdPsum[row] = PdPsum_val

            if const_expr(mdQaccum is not None):
                mdQaccum_cur = seqlen.offset_batch(
                    mdQaccum,
                    batch_idx,
                    dim=2,
                    padded=True,
                    multiple=self.head_dim_padded,
                )[None, head_idx]
                blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            if const_expr(mLSE is not None):
                mLSElog2_cur = seqlen.offset_batch(
                    mLSElog2, batch_idx, dim=2, padded=stats_use_padded_offsets
                )[None, head_idx]
                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.tile_m:
                    gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0
