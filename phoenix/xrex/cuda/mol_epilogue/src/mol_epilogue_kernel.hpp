#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

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
);
