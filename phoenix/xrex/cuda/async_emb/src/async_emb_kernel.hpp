#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace xai::kernels::async_emb {

struct SliceLayout {
  int64_t tokens_per_rank;
  int64_t shard_width;
  int nranks;
};

struct AdagradParams {
  float lr;
  float eps;
  float decay;
  float inv_emb_width;
  float weight_decay_factor;
  float accum_decay_rate;
  float weight_decay_rate;
};

struct UpdateScalars {
  float norm;
  int32_t valid;
  int32_t pending;
  int32_t step;
};

static_assert(sizeof(UpdateScalars) == 4 * sizeof(int32_t));

void launch_lookup_dispatch(
    const int32_t* token_ids_all,
    const __nv_bfloat16* table_shard,
    __nv_bfloat16* send_slices,
    const SliceLayout& slices,
    int64_t vocab_rows,
    cudaStream_t stream
);

void launch_lookup_combine(
    const __nv_bfloat16* recv_slices,
    __nv_bfloat16* embeddings_out,
    const SliceLayout& slices,
    cudaStream_t stream
);

void launch_grad_dispatch(
    const __nv_bfloat16* grads,
    __nv_bfloat16* send_slices,
    const SliceLayout& slices,
    cudaStream_t stream
);

void launch_grad_segment_sum(
    const __nv_bfloat16* routed_grads,
    const int32_t* segment_ids_all,
    float* grad_accum,
    float* row_sq_sums,
    const SliceLayout& slices,
    int64_t num_unique,
    cudaStream_t stream
);

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
);

}
