#include <cuda_runtime.h>
#include <stdint.h>

#define MOD_ADLER 65521U

#define BLOCK_DIM 1024

constexpr __device__ uint16_t mod_adler(uint64_t d) {
#ifdef __CUDA_ARCH__
  uint64_t q = __umul64hi(d, 0x800780708697e2e7) >> 15;
  return d - q * MOD_ADLER;
#else
  return d % MOD_ADLER;
#endif
}

constexpr __device__ uint16_t mod_adler(uint32_t d) {
#ifdef __CUDA_ARCH__
  uint64_t q = __umulhi(d, 0x80078071) >> 15;
  return d - q * MOD_ADLER;
#else
  return d % MOD_ADLER;
#endif
}

__global__ void xai_adler32_kernel(
    const uint8_t* data, size_t length, uint32_t* sum1, uint32_t* sum2
) {
  __shared__ uint32_t s_sum1[BLOCK_DIM];
  __shared__ uint32_t s_sum2[BLOCK_DIM];

  uint32_t local_sum1 = 0;
  uint64_t local_sum2 = 0;

  for (size_t index = (threadIdx.x + blockIdx.x * blockDim.x) * 16; index < (length / 16) * 16;
       index += (gridDim.x * blockDim.x) * 16) {
    uint4 u32 = *(uint4*)&data[index];
    const uint8_t* u8 = reinterpret_cast<const uint8_t*>(&u32);

#pragma unroll
    for (int i = 0; i < 16; ++i) {
      local_sum1 += u8[i];
      local_sum2 += (length - index - i) * u8[i];
    }
    local_sum2 = mod_adler(local_sum2);
  }

  for (size_t index = (length / 16) * 16 + threadIdx.x + blockIdx.x * blockDim.x; index < length;
       index += gridDim.x * blockDim.x) {
    uint8_t val = data[index];
    local_sum1 += val;
    local_sum2 += mod_adler((length - index) * val);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    local_sum1 += 1;
    local_sum2 += length;
  }

  s_sum1[threadIdx.x] = mod_adler(local_sum1);
  s_sum2[threadIdx.x] = mod_adler(local_sum2);
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum1[threadIdx.x] += s_sum1[threadIdx.x + s];
      s_sum2[threadIdx.x] += s_sum2[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(sum1, mod_adler(s_sum1[0]));
    atomicAdd(sum2, mod_adler(s_sum2[0]));
  }
}

template <size_t N>
__global__ void xai_global_adler32_shard_kernel(
    const uint8_t* data,
    const uint32_t* global_shape,
    const uint32_t* shard_shape,
    const uint32_t* shard_offsets,
    uint32_t* sums
) {
  __shared__ uint32_t s_sum1[BLOCK_DIM];
  __shared__ uint32_t s_sum2[BLOCK_DIM];

  uint32_t local_sum1 = 0;
  uint64_t local_sum2 = 0;

  size_t shard_numel = 1;
  size_t global_numel = 1;

#pragma unroll
  for (int d = 0; d < N; ++d) {
    shard_numel *= shard_shape[d];
    global_numel *= global_shape[d];
  }

  for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < shard_numel;
       index += gridDim.x * blockDim.x) {
    uint64_t global_index = 0;
    size_t stride = 1;
    size_t temp = index;

    size_t c_d;
    size_t global_c_d;

#pragma unroll
    for (int d = N - 1; d >= 0; --d) {
      c_d = temp % shard_shape[d];
      global_c_d = c_d + shard_offsets[d];
      global_index += global_c_d * stride;
      stride *= global_shape[d];
      temp /= shard_shape[d];
    }

    uint8_t val = data[index];
    local_sum1 += val;
    local_sum2 += ((global_numel - global_index) * val) % MOD_ADLER;
  }

  s_sum1[threadIdx.x] = local_sum1 % MOD_ADLER;
  s_sum2[threadIdx.x] = local_sum2 % MOD_ADLER;
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum1[threadIdx.x] += s_sum1[threadIdx.x + s];
      s_sum2[threadIdx.x] += s_sum2[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(sums + 0, s_sum1[0] % MOD_ADLER);
    atomicAdd(sums + 1, s_sum2[0] % MOD_ADLER);
  }
}

typedef union {
  uint8_t u8[16];
  uint4 u128;
} Pack128;

__global__ void xai_adler32_fast_kernel(
    const uint8_t* data,
    const uint32_t* scales,
    size_t length,
    uint32_t* sum1,
    uint32_t* sum2,
    int divider_log2
) {
  __shared__ uint32_t s_sum1[BLOCK_DIM];
  __shared__ uint32_t s_sum2[BLOCK_DIM];

  uint64_t local_sum1 = 0;
  uint64_t local_sum2 = 0;
  int divider = 1 << divider_log2;

  int accumulation_count = 0;
  for (size_t index = (threadIdx.x + blockIdx.x * blockDim.x) * 16; index < (length / 16) * 16;
       index += (gridDim.x * blockDim.x) * 16) {
    Pack128 load;
    load.u128 = *(uint4*)&data[index];

    if (((index + 15) >> divider_log2) == (index >> divider_log2)) {
      uint64_t scale = (uint64_t)(scales[index >> divider_log2]) * divider;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        size_t index_ = index + i;
        uint64_t cur_scale = scale - (index_ & (divider - 1));
        local_sum1 += load.u8[i];
        local_sum2 += cur_scale * load.u8[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        size_t index_ = index + i;
        uint64_t scale =
            (uint64_t)(scales[index_ >> divider_log2]) * divider - (index_ & (divider - 1));
        local_sum1 += load.u8[i];
        local_sum2 += scale * load.u8[i];
      }
    }
    accumulation_count++;
    if (accumulation_count == 1024) {
      local_sum1 = mod_adler(local_sum1);
      local_sum2 = mod_adler(local_sum2);
      accumulation_count = 0;
    }
  }

  for (size_t index = (length / 16) * 16 + threadIdx.x + blockIdx.x * blockDim.x; index < length;
       index += gridDim.x * blockDim.x) {
    uint64_t scale = (uint64_t)(scales[index >> divider_log2]) * divider - (index & (divider - 1));
    uint8_t val = data[index];
    local_sum1 += val;
    local_sum2 += scale * val;
  }

  s_sum1[threadIdx.x] = mod_adler(local_sum1);
  s_sum2[threadIdx.x] = mod_adler(local_sum2);
  __syncthreads();

  for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_sum1[threadIdx.x] += s_sum1[threadIdx.x + s];
      s_sum2[threadIdx.x] += s_sum2[threadIdx.x + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(sum1, mod_adler(s_sum1[0]));
    atomicAdd(sum2, mod_adler(s_sum2[0]));
  }
}