#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "mol_epilogue_kernel.hpp"
#include "xrex/cuda/xla_utils/cuda_error_utils.hpp"

namespace {

constexpr int kMaxUsers = 512;
constexpr int kThreads = 128;
constexpr int kMinBlocksPerSm = 6;
constexpr int U = 2;

struct Params {
  const __nv_bfloat16* dots;
  const float* norms;
  const float* ubias;
  const float* ibias;
  const float* w1;
  const float* b1;
  const float* w2;
  const float* b2;
  float* out;
  int num_users;
  int num_components;
  long long num_posts;
};

template <int KL, int H, int U>
__device__ __forceinline__ void score_users(
    const Params& p,
    const volatile float* vw1,
    const volatile float* vw2,
    const float* s_b1,
    const float* s_b2,
    const float* s_ubias,
    const float* inv_norm,
    const float* ib,
    int b0,
    long long m
) {
  const long long M = p.num_posts;
  const int B = p.num_users;
  float cos[U][KL];
#pragma unroll
  for (int c = 0; c < KL; ++c) {
    const long long row = static_cast<long long>(c) * B + b0;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      cos[u][c] = __bfloat162float(p.dots[(row + u) * M + m]) * inv_norm[c];
    }
  }
  float hid[U][H];
#pragma unroll
  for (int h = 0; h < H; ++h) {
    float a[U];
#pragma unroll
    for (int u = 0; u < U; ++u) a[u] = s_b1[h];
#pragma unroll
    for (int c = 0; c < KL; ++c) {
      const float w = vw1[c * H + h];
#pragma unroll
      for (int u = 0; u < U; ++u) a[u] = fmaf(w, cos[u][c], a[u]);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) hid[u][h] = a[u] * __frcp_rn(1.0f + __expf(-a[u]));
  }
  float logit[U][KL];
  float mx[U];
#pragma unroll
  for (int u = 0; u < U; ++u) mx[u] = -3.0e38f;
#pragma unroll
  for (int c = 0; c < KL; ++c) {
    float a[U];
#pragma unroll
    for (int u = 0; u < U; ++u) a[u] = s_b2[c] + ib[c] + s_ubias[c * B + b0 + u];
#pragma unroll
    for (int h = 0; h < H; ++h) {
      const float w = vw2[h * KL + c];
#pragma unroll
      for (int u = 0; u < U; ++u) a[u] = fmaf(w, hid[u][h], a[u]);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      logit[u][c] = a[u];
      mx[u] = fmaxf(mx[u], a[u]);
    }
  }
#pragma unroll
  for (int u = 0; u < U; ++u) {
    float den = 0.0f, num = 0.0f;
#pragma unroll
    for (int c = 0; c < KL; ++c) {
      const float e = __expf(logit[u][c] - mx[u]);
      den += e;
      num = fmaf(e, cos[u][c], num);
    }
    p.out[static_cast<long long>(b0 + u) * M + m] = num * __frcp_rn(den);
  }
}

template <int KL, int H>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm) mol_epilogue_kernel(Params p) {
  __shared__ float s_w1[KL * H];
  __shared__ float s_w2[H * KL];
  __shared__ float s_b1[H];
  __shared__ float s_b2[KL];
  __shared__ float s_ubias[KL * kMaxUsers];
  for (int i = threadIdx.x; i < KL * H; i += blockDim.x) {
    s_w1[i] = p.w1[i];
    s_w2[i] = p.w2[i];
  }
  if (threadIdx.x < H)
    s_b1[threadIdx.x] = p.b1[threadIdx.x];
  if (threadIdx.x < KL)
    s_b2[threadIdx.x] = p.b2[threadIdx.x];
  for (int i = threadIdx.x; i < KL * p.num_users; i += blockDim.x) s_ubias[i] = p.ubias[i];
  __syncthreads();

  const long long m = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (m >= p.num_posts)
    return;
  const long long M = p.num_posts;
  const int B = p.num_users;

  float inv_norm[KL];
  float ib[KL];
#pragma unroll
  for (int c = 0; c < KL; ++c) {
    inv_norm[c] = 1.0f / p.norms[static_cast<long long>(c % p.num_components) * M + m];
    ib[c] = p.ibias[static_cast<long long>(c) * M + m];
  }

  const volatile float* vw1 = s_w1;
  const volatile float* vw2 = s_w2;
  int b = 0;
  for (; b + U <= B; b += U) {
    score_users<KL, H, U>(p, vw1, vw2, s_b1, s_b2, s_ubias, inv_norm, ib, b, m);
  }
  for (; b < B; ++b) {
    score_users<KL, H, 1>(p, vw1, vw2, s_b1, s_b2, s_ubias, inv_norm, ib, b, m);
  }
}

}

ffi::Error mol_epilogue(
    cudaStream_t stream,
    ffi::Buffer<ffi::DataType::BF16> dots,
    ffi::Buffer<ffi::DataType::F32> norms,
    ffi::Buffer<ffi::DataType::F32> ubias,
    ffi::Buffer<ffi::DataType::F32> ibias,
    ffi::Buffer<ffi::DataType::F32> w1,
    ffi::Buffer<ffi::DataType::F32> b1,
    ffi::Buffer<ffi::DataType::F32> w2,
    ffi::Buffer<ffi::DataType::F32> b2,
    ffi::Result<ffi::Buffer<ffi::DataType::F32>> out,
    int64_t num_users,
    int64_t num_components
) {
  const auto dots_dims = dots.dimensions();
  const auto w1_dims = w1.dimensions();
  if (dots_dims.size() != 2 || w1_dims.size() != 2) {
    return ffi::Error::InvalidArgument("dots must be [KL*B, M] and w1 [KL, H]");
  }
  const int64_t kl = w1_dims[0];
  const int64_t hidden = w1_dims[1];
  const int64_t rows = dots_dims[0];
  const int64_t M = dots_dims[1];
  if (rows != kl * num_users)
    return ffi::Error::InvalidArgument("dots rows != KL * num_users");
  if (num_users > kMaxUsers)
    return ffi::Error::InvalidArgument("too many users for the kernel");
  if (norms.dimensions().size() != 2 || norms.dimensions()[0] != num_components ||
      norms.dimensions()[1] != M) {
    return ffi::Error::InvalidArgument("norms must be [L, M]");
  }
  if (ibias.dimensions().size() != 2 || ibias.dimensions()[0] != kl || ibias.dimensions()[1] != M) {
    return ffi::Error::InvalidArgument("ibias must be [KL, M]");
  }
  if (ubias.dimensions().size() != 2 || ubias.dimensions()[0] != kl ||
      ubias.dimensions()[1] != num_users) {
    return ffi::Error::InvalidArgument("ubias must be [KL, B]");
  }
  Params p{
      reinterpret_cast<const __nv_bfloat16*>(dots.typed_data()),
      norms.typed_data(),
      ubias.typed_data(),
      ibias.typed_data(),
      w1.typed_data(),
      b1.typed_data(),
      w2.typed_data(),
      b2.typed_data(),
      out->typed_data(),
      static_cast<int>(num_users),
      static_cast<int>(num_components),
      M
  };
  const unsigned blocks = static_cast<unsigned>((M + kThreads - 1) / kThreads);
  if (kl == 4 && hidden == 32) {
    mol_epilogue_kernel<4, 32><<<blocks, kThreads, 0, stream>>>(p);
  } else if (kl == 8 && hidden == 16) {
    mol_epilogue_kernel<8, 16><<<blocks, kThreads, 0, stream>>>(p);
  } else if (kl == 8 && hidden == 32) {
    mol_epilogue_kernel<8, 32><<<blocks, kThreads, 0, stream>>>(p);
  } else if (kl == 16 && hidden == 16) {
    mol_epilogue_kernel<16, 16><<<blocks, kThreads, 0, stream>>>(p);
  } else {
    return ffi::Error::InvalidArgument(
        "unsupported (KL, H); compiled: (4,32), (8,16), (8,32), (16,16)"
    );
  }
  XAI_RETURN_IF_CUDA_ERROR(cudaGetLastError());
  return ffi::Error::Success();
}
