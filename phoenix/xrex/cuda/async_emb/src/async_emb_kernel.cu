#include <cstdint>

#include "async_emb_kernel.hpp"
#include "cuda_error_utils.hpp"

namespace xai::kernels::async_emb {

namespace {

constexpr int kThreads = 256;
constexpr int kSideGrid = 2048;
constexpr int kMainGrid = 1024;

int grid_for(int64_t work, int max_blocks) {
  int64_t blocks = (work + kThreads - 1) / kThreads;
  if (blocks > max_blocks) {
    blocks = max_blocks;
  }
  return static_cast<int>(blocks < 1 ? 1 : blocks);
}

__device__ __forceinline__ void copy_row(
    __nv_bfloat16* dst, const __nv_bfloat16* src, int64_t width, int lane, int num_lanes
) {
  if (width % 8 == 0) {
    const int4* src_vec = reinterpret_cast<const int4*>(src);
    int4* dst_vec = reinterpret_cast<int4*>(dst);
    for (int64_t unit = lane; unit < width / 8; unit += num_lanes) {
      dst_vec[unit] = src_vec[unit];
    }
  } else if (width % 4 == 0) {
    const int2* src_vec = reinterpret_cast<const int2*>(src);
    int2* dst_vec = reinterpret_cast<int2*>(dst);
    for (int64_t unit = lane; unit < width / 4; unit += num_lanes) {
      dst_vec[unit] = src_vec[unit];
    }
  } else {
    for (int64_t col = lane; col < width; col += num_lanes) {
      dst[col] = src[col];
    }
  }
}

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float sums[kThreads];
  sums[threadIdx.x] = value;
  __syncthreads();
  for (int span = blockDim.x / 2; span > 0; span >>= 1) {
    if (threadIdx.x < span) {
      sums[threadIdx.x] += sums[threadIdx.x + span];
    }
    __syncthreads();
  }
  return sums[0];
}

}

__global__ void lookup_dispatch(
    const int32_t* token_ids_all,
    const __nv_bfloat16* table_shard,
    __nv_bfloat16* send_slices,
    int64_t tokens_per_rank,
    int64_t shard_width,
    int64_t vocab_rows,
    int nranks,
    int64_t lanes_per_row
) {
  const int64_t total = tokens_per_rank * nranks * lanes_per_row;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  for (int64_t item = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; item < total;
       item += grid_stride) {
    const int64_t token = item / lanes_per_row;
    const int lane = static_cast<int>(item - token * lanes_per_row);
    int64_t row = token_ids_all[token];
    if (row < 0) {
      row += vocab_rows;
    }
    if (row < 0) {
      row = 0;
    }
    if (row >= vocab_rows) {
      row = vocab_rows - 1;
    }
    const __nv_bfloat16* src = table_shard + row * shard_width;
    const int dst_rank = static_cast<int>(token / tokens_per_rank);
    const int64_t slot = token - int64_t(dst_rank) * tokens_per_rank;
    __nv_bfloat16* dst =
        send_slices + dst_rank * (tokens_per_rank * shard_width) + slot * shard_width;
    copy_row(dst, src, shard_width, lane, static_cast<int>(lanes_per_row));
  }
}

void launch_lookup_dispatch(
    const int32_t* token_ids_all,
    const __nv_bfloat16* table_shard,
    __nv_bfloat16* send_slices,
    const SliceLayout& slices,
    int64_t vocab_rows,
    cudaStream_t stream
) {
  const int64_t lanes = slices.shard_width <= 16 ? 1 : 32;
  const int64_t work = slices.tokens_per_rank * int64_t(slices.nranks) * lanes;
  lookup_dispatch<<<grid_for(work, kSideGrid), kThreads, 0, stream>>>(
      token_ids_all,
      table_shard,
      send_slices,
      slices.tokens_per_rank,
      slices.shard_width,
      vocab_rows,
      slices.nranks,
      lanes
  );
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void lookup_combine(
    const __nv_bfloat16* recv_slices,
    __nv_bfloat16* embeddings_out,
    int64_t tokens_per_rank,
    int64_t shard_width,
    int nranks
) {
  const int64_t emb_width = int64_t(nranks) * shard_width;
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  if (shard_width % 8 == 0) {
    const int64_t units_per_slice = shard_width / 8;
    const int64_t units_total = tokens_per_rank * emb_width / 8;
    const int4* src = reinterpret_cast<const int4*>(recv_slices);
    int4* dst = reinterpret_cast<int4*>(embeddings_out);
    for (int64_t unit = thread_id; unit < units_total; unit += grid_stride) {
      const int64_t slice_unit = unit % units_per_slice;
      const int64_t token_rank = unit / units_per_slice;
      const int rank = static_cast<int>(token_rank % nranks);
      const int64_t token = token_rank / nranks;
      dst[token * (emb_width / 8) + int64_t(rank) * units_per_slice + slice_unit] =
          src[int64_t(rank) * (tokens_per_rank * units_per_slice) + token * units_per_slice +
              slice_unit];
    }
  } else {
    const int64_t total = tokens_per_rank * emb_width;
    for (int64_t elem = thread_id; elem < total; elem += grid_stride) {
      const int64_t col = elem % emb_width;
      const int64_t token = elem / emb_width;
      const int rank = static_cast<int>(col / shard_width);
      const int64_t slice_col = col - int64_t(rank) * shard_width;
      embeddings_out[elem] = recv_slices
          [int64_t(rank) * (tokens_per_rank * shard_width) + token * shard_width + slice_col];
    }
  }
}

void launch_lookup_combine(
    const __nv_bfloat16* recv_slices,
    __nv_bfloat16* embeddings_out,
    const SliceLayout& slices,
    cudaStream_t stream
) {
  const int64_t work = slices.tokens_per_rank * int64_t(slices.nranks) * slices.shard_width /
                       (slices.shard_width % 8 == 0 ? 8 : 1);
  lookup_combine<<<grid_for(work, kMainGrid), kThreads, 0, stream>>>(
      recv_slices, embeddings_out, slices.tokens_per_rank, slices.shard_width, slices.nranks
  );
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void grad_dispatch(
    const __nv_bfloat16* grads,
    __nv_bfloat16* send_slices,
    int64_t tokens_per_rank,
    int64_t shard_width,
    int nranks
) {
  const int64_t emb_width = int64_t(nranks) * shard_width;
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  if (shard_width % 8 == 0) {
    const int64_t units_per_slice = shard_width / 8;
    const int64_t units_total = tokens_per_rank * emb_width / 8;
    const int4* src = reinterpret_cast<const int4*>(grads);
    for (int64_t unit = thread_id; unit < units_total; unit += grid_stride) {
      const int64_t slice_unit = unit % units_per_slice;
      const int64_t token_rank = unit / units_per_slice;
      const int rank = static_cast<int>(token_rank % nranks);
      const int64_t token = token_rank / nranks;
      int4* dst =
          reinterpret_cast<int4*>(send_slices + int64_t(rank) * (tokens_per_rank * shard_width));
      dst[token * units_per_slice + slice_unit] =
          src[token * (emb_width / 8) + int64_t(rank) * units_per_slice + slice_unit];
    }
  } else {
    const int64_t total = tokens_per_rank * emb_width;
    for (int64_t elem = thread_id; elem < total; elem += grid_stride) {
      const int64_t col = elem % emb_width;
      const int64_t token = elem / emb_width;
      const int rank = static_cast<int>(col / shard_width);
      const int64_t slice_col = col - int64_t(rank) * shard_width;
      __nv_bfloat16* dst = send_slices + int64_t(rank) * (tokens_per_rank * shard_width);
      dst[token * shard_width + slice_col] = grads[elem];
    }
  }
}

void launch_grad_dispatch(
    const __nv_bfloat16* grads,
    __nv_bfloat16* send_slices,
    const SliceLayout& slices,
    cudaStream_t stream
) {
  const int64_t work = slices.tokens_per_rank * int64_t(slices.nranks) * slices.shard_width /
                       (slices.shard_width % 8 == 0 ? 8 : 1);
  grad_dispatch<<<grid_for(work, kMainGrid), kThreads, 0, stream>>>(
      grads, send_slices, slices.tokens_per_rank, slices.shard_width, slices.nranks
  );
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void grad_segment_sum(
    const __nv_bfloat16* routed_grads,
    const int32_t* segment_ids_all,
    float* grad_accum,
    float* row_sq_sums,
    int64_t n_tokens_global,
    int64_t shard_width,
    int64_t num_unique
) {
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  if (thread_id == 0) {
    row_sq_sums[num_unique] = 0.f;
  }
  const int64_t total = n_tokens_global * shard_width;
  for (int64_t elem = thread_id; elem < total; elem += grid_stride) {
    const int64_t token = elem / shard_width;
    const int64_t col = elem - token * shard_width;
    const int32_t segment = __ldg(segment_ids_all + token);
    if (segment < 0 || segment >= num_unique) {
      continue;
    }
    atomicAdd(
        grad_accum + int64_t(segment) * shard_width + col,
        __bfloat162float(__ldg(routed_grads + elem))
    );
  }
}

__global__ void grad_row_stats(
    const float* grad_accum, float* row_sq_sums, int64_t num_unique, int64_t shard_width
) {
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  float thread_total = 0.f;
  for (int64_t row = thread_id; row < num_unique; row += grid_stride) {
    float sq_sum = 0.f;
    const float* row_vals = grad_accum + row * shard_width;
    for (int64_t col = 0; col < shard_width; ++col) {
      const float grad = __bfloat162float(__float2bfloat16(row_vals[col]));
      sq_sum += grad * grad;
    }
    row_sq_sums[row] = sq_sum;
    thread_total += sq_sum;
  }
  const float block_total = block_sum(thread_total);
  if (threadIdx.x == 0) {
    atomicAdd(row_sq_sums + num_unique, block_total);
  }
}

void launch_grad_segment_sum(
    const __nv_bfloat16* routed_grads,
    const int32_t* segment_ids_all,
    float* grad_accum,
    float* row_sq_sums,
    const SliceLayout& slices,
    int64_t num_unique,
    cudaStream_t stream
) {
  const int64_t n_tokens_global = slices.tokens_per_rank * int64_t(slices.nranks);
  grad_segment_sum<<<
      grid_for(n_tokens_global * slices.shard_width, kSideGrid),
      kThreads,
      0,
      stream>>>(
      routed_grads,
      segment_ids_all,
      grad_accum,
      row_sq_sums,
      n_tokens_global,
      slices.shard_width,
      num_unique
  );
  XAI_CUDA_LAUNCH_CHECK();
  grad_row_stats<<<grid_for(num_unique, kSideGrid), kThreads, 0, stream>>>(
      grad_accum, row_sq_sums, num_unique, slices.shard_width
  );
  XAI_CUDA_LAUNCH_CHECK();
}

__global__ void rowwise_adagrad_apply(
    const float* grad_accum,
    const float* row_sq_sums,
    const int32_t* unique_tokens,
    float* row_state,
    int32_t* last_step,
    __nv_bfloat16* table_shard,
    UpdateScalars* scalars,
    AdagradParams adagrad,
    int64_t num_unique,
    int64_t shard_width,
    int64_t vocab_rows
) {
  const float norm = sqrtf(__ldg(row_sq_sums + num_unique));
  const bool apply = isfinite(norm) && __ldg(&scalars->pending) != 0;
  const bool lazy = last_step != nullptr;
  const int32_t clock = lazy ? __ldg(&scalars->step) : 0;
  const int64_t thread_id = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (thread_id == 0) {
    scalars->norm = norm;
    scalars->valid = apply ? 1 : 0;
  }
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int64_t total = num_unique * shard_width;
  const int64_t grid_stride = gridDim.x * int64_t(blockDim.x);
  for (int64_t warp_base = thread_id - lane; warp_base < total; warp_base += grid_stride) {
    const int64_t elem = warp_base + lane;
    const bool in_range = elem < total;
    const int64_t row = in_range ? elem / shard_width : 0;
    const int col = in_range ? static_cast<int>(elem - row * shard_width) : 0;
    int64_t token = __ldg(unique_tokens + row);
    if (token < 0) {
      token += vocab_rows;
    }
    const bool ok = in_range && apply && token >= 0 && token < vocab_rows;
    float accum = 0.f;
    float wd = 1.f;
    if (ok && col == 0) {
      float accum_factor = adagrad.decay;
      if (lazy) {
        const float delta = fmaxf(0.f, float(clock - last_step[token]));
        accum_factor = expf(-adagrad.accum_decay_rate * delta);
        wd = __ldg(row_sq_sums + row) > 0.f ? expf(-adagrad.weight_decay_rate * delta) : 1.f;
        last_step[token] = clock;
      } else {
        wd = __ldg(row_sq_sums + row) > 0.f ? adagrad.weight_decay_factor : 1.f;
      }
      accum = row_state[token] * accum_factor + __ldg(row_sq_sums + row) * adagrad.inv_emb_width;
    }
    accum = __shfl_sync(0xffffffffu, accum, lane - col);
    wd = __shfl_sync(0xffffffffu, wd, lane - col);
    if (ok) {
      const float step = (-adagrad.lr) * (1.f / (sqrtf(accum) + adagrad.eps));
      const float grad = __bfloat162float(__float2bfloat16(__ldg(grad_accum + elem)));
      const float delta = __bfloat162float(__float2bfloat16(step * grad));
      __nv_bfloat16* cell = table_shard + token * shard_width + col;
      const float decayed = __bfloat162float(__float2bfloat16(__bfloat162float(*cell) * wd));
      *cell = __float2bfloat16(decayed + delta);
      if (col == 0) {
        row_state[token] = accum;
      }
    }
  }
}

void launch_rowwise_adagrad_apply(
    const float* grad_accum,
    const float* row_sq_sums,
    const int32_t* unique_tokens,
    float* row_state,
    int32_t* last_step,
    __nv_bfloat16* table_shard,
    int64_t vocab_rows,
    UpdateScalars* scalars,
    const AdagradParams& adagrad,
    int64_t num_unique,
    int64_t shard_width,
    cudaStream_t stream
) {
  rowwise_adagrad_apply<<<grid_for(num_unique * shard_width, kSideGrid), kThreads, 0, stream>>>(
      grad_accum,
      row_sq_sums,
      unique_tokens,
      row_state,
      last_step,
      table_shard,
      scalars,
      adagrad,
      num_unique,
      shard_width,
      vocab_rows
  );
  XAI_CUDA_LAUNCH_CHECK();
}

}
