#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <string>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

#define XAI_RETURN_IF_CUDA_ERROR(stmt)                       \
  do {                                                       \
    cudaError_t _err = (stmt);                               \
    if (_err != cudaSuccess) {                               \
      return ffi::Error::Internal(cudaGetErrorString(_err)); \
    }                                                        \
  } while (0)

#define XAI_CUDA_CHECK(condition)                                                                 \
  do {                                                                                            \
    cudaError_t error = condition;                                                                \
    if (error != cudaSuccess) {                                                                   \
      std::string error_message = std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + \
                                  std::string(cudaGetErrorString(error));                         \
      throw std::runtime_error("cuda error: " + error_message);                                   \
    }                                                                                             \
  } while (0)

#define XAI_CUDA_LAUNCH_CHECK() XAI_CUDA_CHECK(cudaGetLastError())

#define XAI_CHECK(cond, msg)                                                       \
  if (!(cond)) {                                                                   \
    std::string error_message =                                                    \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + std::string(msg); \
    return ffi::Error(XLA_FFI_Error_Code_INTERNAL, error_message);                 \
  }

#define XAI_ASSERT(cond, msg)                                                      \
  if (!(cond)) {                                                                   \
    std::string error_message =                                                    \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + std::string(msg); \
    throw std::runtime_error(error_message);                                       \
  }

#define XAI_CUDA_FFI_CHECK(condition)                                                \
  do {                                                                               \
    cudaError_t error = condition;                                                   \
    if (error != cudaSuccess) {                                                      \
      return ffi::Error(                                                             \
          XLA_FFI_Error_Code_INTERNAL,                                               \
          (std::stringstream() << "Error at " << __FILE__ << ":" << __LINE__ << ": " \
                               << cudaGetErrorString(error))                         \
              .str()                                                                 \
      );                                                                             \
    }                                                                                \
  } while (0)

#define XAI_CUDA_FFI_LAUNCH_CHECK() XAI_CUDA_FFI_CHECK(cudaGetLastError())

#define XAI_CUDA_ENTRY_CHECK()                                                          \
  do {                                                                                  \
    auto cuda_error = cudaGetLastError();                                               \
    std::string __msg__ = "CUDA is already in an error state due to a previous ops. " + \
                          std::string(cudaGetErrorString(cuda_error));                  \
    XAI_CHECK(cudaGetLastError() == cudaSuccess, __msg__);                              \
  } while (0)