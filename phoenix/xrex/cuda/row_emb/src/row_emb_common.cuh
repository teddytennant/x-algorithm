#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "row_emb_kernel.hpp"

namespace xai::kernels::row_emb {

constexpr int kWarpThreads = 32;

template <class T>
constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}
template <class T>
constexpr T align_up(T v, T a) {
  return (v + a - 1) / a * a;
}

inline int grid_1d(int64_t n, int threads, int max_blocks = 1 << 30) {
  const int64_t blocks = ceil_div<int64_t>(n > 0 ? n : 1, threads);
  return int(blocks < max_blocks ? blocks : max_blocks);
}

class Carver {
 public:
  static constexpr size_t kAlign = 256;
  explicit Carver(void* base) : base_(static_cast<char*>(base)) {}
  template <class T>
  T* take(size_t count) {
    char* p = base_ ? base_ + off_ : nullptr;
    off_ += align_up(count * sizeof(T), kAlign);
    return reinterpret_cast<T*>(p);
  }
  size_t bytes() const { return off_; }

 private:
  char* base_;
  size_t off_ = 0;
};

struct bf16x8 {
  uint4 raw;
};
__device__ __forceinline__ bf16x8 load8(const bf16* p) {
  return {*reinterpret_cast<const uint4*>(p)};
}
__device__ __forceinline__ void store8(bf16* p, const bf16x8& v) {
  *reinterpret_cast<uint4*>(p) = v.raw;
}
__device__ __forceinline__ void unpack8(const bf16x8& v, float (&f)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&(&v.raw.x)[i]));
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ bf16x8 pack8(const float (&f)[8]) {
  bf16x8 v;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_bfloat162 t = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    (&v.raw.x)[i] = *reinterpret_cast<const uint32_t*>(&t);
  }
  return v;
}
__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = kWarpThreads / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

}
