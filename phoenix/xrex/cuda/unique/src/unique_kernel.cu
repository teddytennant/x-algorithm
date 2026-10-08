#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdio>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <iostream>
#include <stdexcept>
#include <string>

#include "unique_kernel.hpp"
#include "xrex/cuda/xla_utils/cuda_error_utils.hpp"
#include "xrex/cuda/xla_utils/thrust_allocator.hpp"

using xrex_cuda::ThrustAllocator;

namespace {

void radix_sort_into(
    cudaStream_t stream,
    ffi::ScratchAllocator& scratch_allocator,
    int num_items,
    int* d_keys,
    int* d_keys_alt,
    int* d_values = nullptr,
    int* d_values_alt = nullptr
) {
  cub::DoubleBuffer<int> keys(d_keys, d_keys_alt);
  cub::DoubleBuffer<int> values(d_values, d_values_alt);
  const bool sort_pairs = d_values != nullptr;

  auto sort = [&](void* d_temp_storage, size_t& temp_storage_bytes) {
    return sort_pairs
               ? cub::DeviceRadixSort::SortPairs(
                     d_temp_storage, temp_storage_bytes, keys, values, num_items, 0, 32, stream
                 )
               : cub::DeviceRadixSort::SortKeys(
                     d_temp_storage, temp_storage_bytes, keys, num_items, 0, 32, stream
                 );
  };

  size_t temp_storage_bytes = 0;
  XAI_CUDA_CHECK(sort(nullptr, temp_storage_bytes));
  auto d_temp_storage = scratch_allocator.Allocate(temp_storage_bytes);
  if (!d_temp_storage) {
    throw std::runtime_error(
        "unique: failed to allocate " + std::to_string(temp_storage_bytes) +
        " bytes of sort scratch"
    );
  }
  XAI_CUDA_CHECK(sort(*d_temp_storage, temp_storage_bytes));

  if (keys.Current() != d_keys) {
    XAI_CUDA_CHECK(cudaMemcpyAsync(
        d_keys, keys.Current(), sizeof(int) * num_items, cudaMemcpyDeviceToDevice, stream
    ));
  }
  if (sort_pairs && values.Current() != d_values) {
    XAI_CUDA_CHECK(cudaMemcpyAsync(
        d_values, values.Current(), sizeof(int) * num_items, cudaMemcpyDeviceToDevice, stream
    ));
  }
}

}

ffi::Error unique_impl_int32(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch_allocator,
    ffi::BufferR1<ffi::DataType::S32> arr,
    const bool return_inverse,
    const int64_t size,
    const int64_t fill_value,
    ffi::Result<ffi::BufferR1<ffi::DataType::S32>> unique_vals,
    ffi::Result<ffi::BufferR1<ffi::DataType::S32>> unique_inverse,
    ffi::Result<ffi::BufferR1<ffi::DataType::S32>> d_keys,
    ffi::Result<ffi::BufferR1<ffi::DataType::S32>> d_indices,
    ffi::Result<ffi::BufferR1<ffi::DataType::S32>> d_temp_buffer
)
{
  const int N = arr.element_count();

  ThrustAllocator thrust_allocator(&scratch_allocator);
  auto policy = thrust::cuda::par(thrust_allocator).on(stream);

  const auto* arr_ptr = arr.typed_data();
  auto* d_unique_vals = unique_vals->typed_data();

  thrust::device_ptr<int> d_keys_ptr(d_keys->typed_data());

  thrust::copy(policy, arr_ptr, arr_ptr + N, d_keys_ptr);

  int num_unique;

  if (return_inverse) {
    thrust::device_ptr<int> d_indices_ptr(d_indices->typed_data());
    thrust::device_ptr<int> d_temp_buffer_ptr(d_temp_buffer->typed_data());

    thrust::sequence(policy, d_indices_ptr, d_indices_ptr + N);
    radix_sort_into(
        stream,
        scratch_allocator,
        N,
        d_keys->typed_data(),
        d_temp_buffer->typed_data(),
        d_indices->typed_data(),
        unique_inverse->typed_data()
    );

    auto unique_end = thrust::unique_copy(policy, d_keys_ptr, d_keys_ptr + N, d_unique_vals);
    num_unique = unique_end - d_unique_vals;

    thrust::adjacent_difference(
        policy, d_keys_ptr, d_keys_ptr + N, d_temp_buffer_ptr, thrust::not_equal_to<int>()
    );
    thrust::fill_n(policy, d_temp_buffer_ptr, 1, 1);

    thrust::inclusive_scan(policy, d_temp_buffer_ptr, d_temp_buffer_ptr + N, d_temp_buffer_ptr);

    thrust::transform(
        policy, d_temp_buffer_ptr, d_temp_buffer_ptr + N, d_temp_buffer_ptr, [] __device__(int x) {
          return x - 1;
        }
    );

    auto* d_inverse = unique_inverse->typed_data();
    thrust::scatter(policy, d_temp_buffer_ptr, d_temp_buffer_ptr + N, d_indices_ptr, d_inverse);

  } else {
    radix_sort_into(
        stream, scratch_allocator, N, d_keys->typed_data(), d_temp_buffer->typed_data()
    );
    auto unique_end = thrust::unique_copy(policy, d_keys_ptr, d_keys_ptr + N, d_unique_vals);
    num_unique = unique_end - d_unique_vals;
  }

  thrust::fill(
      policy, d_unique_vals + num_unique, d_unique_vals + size, static_cast<int>(fill_value)
  );

  XAI_CUDA_LAUNCH_CHECK();
  return ffi::Error::Success();
}
