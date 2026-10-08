
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "top_k_by_key_radix_select_kernel.hpp"
#include "xrex/cuda/xla_utils/cuda_error_utils.hpp"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ uint32_t ordered16(uint16_t raw) {
  return (raw & 0x8000u) ? (uint32_t)(uint16_t)~raw : (uint32_t)(raw | 0x8000u);
}

__global__ void hist_hi_kernel(
    const uint16_t* __restrict__ keys, int64_t n, uint32_t* __restrict__ hist
) {
  __shared__ uint32_t sh[kWarps][256];
  const int row = blockIdx.y;
  const int warp = threadIdx.x / 32;
  for (int i = threadIdx.x; i < kWarps * 256; i += blockDim.x) ((uint32_t*)sh)[i] = 0;
  __syncthreads();
  const uint16_t* krow = keys + (int64_t)row * n;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    atomicAdd(&sh[warp][ordered16(krow[i]) >> 8], 1u);
  }
  __syncthreads();
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    uint32_t s = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += sh[w][b];
    if (s)
      atomicAdd(&hist[row * 256 + b], s);
  }
}

__global__ void scan_hi_kernel(
    const uint32_t* __restrict__ hist,
    int64_t k,
    int32_t* __restrict__ hi_bin,
    int32_t* __restrict__ k2
) {
  const int row = blockIdx.x;
  const uint32_t* h = hist + row * 256;
  int64_t cum = 0;
  int b = 256;
  while (b > 0) {
    --b;
    cum += h[b];
    if (cum >= k)
      break;
  }
  hi_bin[row] = b;
  k2[row] = (int32_t)(k - (cum - h[b]));
}

__global__ void collect_hi_kernel(
    const uint16_t* __restrict__ keys,
    int64_t n,
    int64_t kk,
    const int32_t* __restrict__ hi_bin,
    uint16_t* __restrict__ out_keys,
    int32_t* __restrict__ out_vals,
    int32_t* __restrict__ out_count,
    uint32_t* __restrict__ hist_lo
) {
  __shared__ uint32_t sh[kWarps][256];
  const int row = blockIdx.y;
  const int warp = threadIdx.x / 32;
  const uint32_t H = (uint32_t)hi_bin[row];
  for (int i = threadIdx.x; i < kWarps * 256; i += blockDim.x) ((uint32_t*)sh)[i] = 0;
  __syncthreads();
  const uint16_t* krow = keys + (int64_t)row * n;
  uint16_t* okrow = out_keys + (int64_t)row * kk;
  int32_t* ovrow = out_vals + (int64_t)row * kk;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint16_t raw = krow[i];
    const uint32_t t = ordered16(raw);
    const uint32_t hi = t >> 8;
    if (hi > H) {
      const int pos = atomicAdd(&out_count[row], 1);
      if (pos < kk) {
        okrow[pos] = raw;
        ovrow[pos] = (int32_t)i;
      }
    } else if (hi == H) {
      atomicAdd(&sh[warp][t & 0xFFu], 1u);
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    uint32_t s = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += sh[w][b];
    if (s)
      atomicAdd(&hist_lo[row * 256 + b], s);
  }
}

__global__ void scan_lo_kernel(
    const uint32_t* __restrict__ hist_lo,
    const int32_t* __restrict__ k2,
    int32_t* __restrict__ lo_bin,
    int32_t* __restrict__ ties_needed
) {
  const int row = blockIdx.x;
  const uint32_t* h = hist_lo + row * 256;
  const int64_t kk2 = k2[row];
  int64_t cum = 0;
  int b = 256;
  while (b > 0) {
    --b;
    cum += h[b];
    if (cum >= kk2)
      break;
  }
  lo_bin[row] = b;
  ties_needed[row] = (int32_t)(kk2 - (cum - h[b]));
}

__global__ void collect_rest_kernel(
    const uint16_t* __restrict__ keys,
    int64_t n,
    int64_t kk,
    const int32_t* __restrict__ hi_bin,
    const int32_t* __restrict__ lo_bin,
    const int32_t* __restrict__ ties_needed,
    uint16_t* __restrict__ out_keys,
    int32_t* __restrict__ out_vals,
    int32_t* __restrict__ out_count,
    int32_t* __restrict__ tie_count
) {
  const int row = blockIdx.y;
  const uint32_t H = (uint32_t)hi_bin[row];
  const uint32_t L = (uint32_t)lo_bin[row];
  const int need = ties_needed[row];
  const uint16_t* krow = keys + (int64_t)row * n;
  uint16_t* okrow = out_keys + (int64_t)row * kk;
  int32_t* ovrow = out_vals + (int64_t)row * kk;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint16_t raw = krow[i];
    const uint32_t t = ordered16(raw);
    if ((t >> 8) != H)
      continue;
    const uint32_t lo = t & 0xFFu;
    if (lo > L) {
      const int pos = atomicAdd(&out_count[row], 1);
      if (pos < kk) {
        okrow[pos] = raw;
        ovrow[pos] = (int32_t)i;
      }
    } else if (lo == L) {
      const int tp = atomicAdd(&tie_count[row], 1);
      if (tp < need) {
        const int pos = atomicAdd(&out_count[row], 1);
        if (pos < kk) {
          okrow[pos] = raw;
          ovrow[pos] = (int32_t)i;
        }
      }
    }
  }
}


__global__ void hist_hi_vec_kernel(
    const uint16_t* __restrict__ keys,
    int64_t n8,
    int64_t n,
    uint32_t* __restrict__ hist
) {
  __shared__ uint32_t sh[kWarps][256];
  const int row = blockIdx.y;
  const int warp = threadIdx.x / 32;
  for (int i = threadIdx.x; i < kWarps * 256; i += blockDim.x) ((uint32_t*)sh)[i] = 0;
  __syncthreads();
  const uint4* krow = reinterpret_cast<const uint4*>(keys + (int64_t)row * n);
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n8;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 v = krow[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      atomicAdd(&sh[warp][ordered16((uint16_t)(words[w] & 0xFFFFu)) >> 8], 1u);
      atomicAdd(&sh[warp][ordered16((uint16_t)(words[w] >> 16)) >> 8], 1u);
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    uint32_t s = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += sh[w][b];
    if (s)
      atomicAdd(&hist[row * 256 + b], s);
  }
}

__global__ void collect_hi_vec_kernel(
    const uint16_t* __restrict__ keys,
    int64_t n8,
    int64_t n,
    int64_t kk,
    const int32_t* __restrict__ hi_bin,
    uint16_t* __restrict__ out_keys,
    int32_t* __restrict__ out_vals,
    int32_t* __restrict__ out_count,
    uint32_t* __restrict__ hist_lo
) {
  __shared__ uint32_t sh[kWarps][256];
  const int row = blockIdx.y;
  const int warp = threadIdx.x / 32;
  const uint32_t H = (uint32_t)hi_bin[row];
  for (int i = threadIdx.x; i < kWarps * 256; i += blockDim.x) ((uint32_t*)sh)[i] = 0;
  __syncthreads();
  const uint4* krow = reinterpret_cast<const uint4*>(keys + (int64_t)row * n);
  uint16_t* okrow = out_keys + (int64_t)row * kk;
  int32_t* ovrow = out_vals + (int64_t)row * kk;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n8;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 v = krow[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint16_t raw = (uint16_t)(half ? (words[w] >> 16) : (words[w] & 0xFFFFu));
        const uint32_t t = ordered16(raw);
        const uint32_t hi = t >> 8;
        if (hi > H) {
          const int pos = atomicAdd(&out_count[row], 1);
          if (pos < kk) {
            okrow[pos] = raw;
            ovrow[pos] = (int32_t)(i * 8 + 2 * w + half);
          }
        } else if (hi == H) {
          atomicAdd(&sh[warp][t & 0xFFu], 1u);
        }
      }
    }
  }
  __syncthreads();
  for (int b = threadIdx.x; b < 256; b += blockDim.x) {
    uint32_t s = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += sh[w][b];
    if (s)
      atomicAdd(&hist_lo[row * 256 + b], s);
  }
}

__global__ void collect_rest_vec_kernel(
    const uint16_t* __restrict__ keys,
    int64_t n8,
    int64_t n,
    int64_t kk,
    const int32_t* __restrict__ hi_bin,
    const int32_t* __restrict__ lo_bin,
    const int32_t* __restrict__ ties_needed,
    uint16_t* __restrict__ out_keys,
    int32_t* __restrict__ out_vals,
    int32_t* __restrict__ out_count,
    int32_t* __restrict__ tie_count
) {
  const int row = blockIdx.y;
  const uint32_t H = (uint32_t)hi_bin[row];
  const uint32_t L = (uint32_t)lo_bin[row];
  const int need = ties_needed[row];
  const uint4* krow = reinterpret_cast<const uint4*>(keys + (int64_t)row * n);
  uint16_t* okrow = out_keys + (int64_t)row * kk;
  int32_t* ovrow = out_vals + (int64_t)row * kk;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n8;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 v = krow[i];
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint16_t raw = (uint16_t)(half ? (words[w] >> 16) : (words[w] & 0xFFFFu));
        const uint32_t t = ordered16(raw);
        if ((t >> 8) != H)
          continue;
        const uint32_t lo = t & 0xFFu;
        if (lo > L) {
          const int pos = atomicAdd(&out_count[row], 1);
          if (pos < kk) {
            okrow[pos] = raw;
            ovrow[pos] = (int32_t)(i * 8 + 2 * w + half);
          }
        } else if (lo == L) {
          const int tp = atomicAdd(&tie_count[row], 1);
          if (tp < need) {
            const int pos = atomicAdd(&out_count[row], 1);
            if (pos < kk) {
              okrow[pos] = raw;
              ovrow[pos] = (int32_t)(i * 8 + 2 * w + half);
            }
          }
        }
      }
    }
  }
}

}

ffi::Error top_k_by_key_bf16_radix_select(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch_allocator,
    ffi::Buffer<ffi::DataType::BF16> keys,
    int64_t k,
    ffi::Result<ffi::Buffer<ffi::DataType::BF16>> top_k_keys,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> top_k_values
) {
  auto dims = keys.dimensions();
  const int64_t rows = (dims.size() == 1) ? 1 : dims[0];
  const int64_t n = (dims.size() == 1) ? dims[0] : dims[1];
  if (k <= 0 || k > n) {
    return ffi::Error::InvalidArgument("k must be positive and <= n");
  }
  if (n > INT32_MAX) {
    return ffi::Error::InvalidArgument("n exceeds int32 index range");
  }

  const uint16_t* keys_ptr = reinterpret_cast<const uint16_t*>(keys.typed_data());
  uint16_t* ok = reinterpret_cast<uint16_t*>(top_k_keys->typed_data());
  int32_t* ov = reinterpret_cast<int32_t*>(top_k_values->typed_data());

  auto alloc = [&](size_t bytes) -> void* {
    auto r = scratch_allocator.Allocate(bytes);
    return r.has_value() ? r.value() : nullptr;
  };
  uint32_t* hist_hi = (uint32_t*)alloc(rows * 256 * sizeof(uint32_t));
  uint32_t* hist_lo = (uint32_t*)alloc(rows * 256 * sizeof(uint32_t));
  int32_t* misc = (int32_t*)alloc(rows * 6 * sizeof(int32_t));
  if (hist_hi == nullptr || hist_lo == nullptr || misc == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInternal, "scratch allocation failed");
  }
  int32_t* hi_bin = misc + 0 * rows;
  int32_t* k2 = misc + 1 * rows;
  int32_t* lo_bin = misc + 2 * rows;
  int32_t* ties = misc + 3 * rows;
  int32_t* out_count = misc + 4 * rows;
  int32_t* tie_count = misc + 5 * rows;

  cudaMemsetAsync(hist_hi, 0, rows * 256 * sizeof(uint32_t), stream);
  cudaMemsetAsync(hist_lo, 0, rows * 256 * sizeof(uint32_t), stream);
  cudaMemsetAsync(out_count, 0, rows * sizeof(int32_t), stream);
  cudaMemsetAsync(tie_count, 0, rows * sizeof(int32_t), stream);

  if ((n % 8) == 0) {
    const int64_t n8 = n / 8;
    const int xblocks = (int)std::min<int64_t>((n8 + kThreads - 1) / kThreads, 2048);
    const dim3 grid((unsigned)xblocks, (unsigned)rows);
    hist_hi_vec_kernel<<<grid, kThreads, 0, stream>>>(keys_ptr, n8, n, hist_hi);
    scan_hi_kernel<<<(unsigned)rows, 1, 0, stream>>>(hist_hi, k, hi_bin, k2);
    collect_hi_vec_kernel<<<grid, kThreads, 0, stream>>>(
        keys_ptr, n8, n, k, hi_bin, ok, ov, out_count, hist_lo
    );
    scan_lo_kernel<<<(unsigned)rows, 1, 0, stream>>>(hist_lo, k2, lo_bin, ties);
    collect_rest_vec_kernel<<<grid, kThreads, 0, stream>>>(
        keys_ptr, n8, n, k, hi_bin, lo_bin, ties, ok, ov, out_count, tie_count
    );
  } else {
    const int xblocks = (int)std::min<int64_t>((n + kThreads - 1) / kThreads, 2048);
    const dim3 grid((unsigned)xblocks, (unsigned)rows);
    hist_hi_kernel<<<grid, kThreads, 0, stream>>>(keys_ptr, n, hist_hi);
    scan_hi_kernel<<<(unsigned)rows, 1, 0, stream>>>(hist_hi, k, hi_bin, k2);
    collect_hi_kernel<<<grid, kThreads, 0, stream>>>(
        keys_ptr, n, k, hi_bin, ok, ov, out_count, hist_lo
    );
    scan_lo_kernel<<<(unsigned)rows, 1, 0, stream>>>(hist_lo, k2, lo_bin, ties);
    collect_rest_kernel<<<grid, kThreads, 0, stream>>>(
        keys_ptr, n, k, hi_bin, lo_bin, ties, ok, ov, out_count, tie_count
    );
  }

  XAI_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  return ffi::Error::Success();
}
