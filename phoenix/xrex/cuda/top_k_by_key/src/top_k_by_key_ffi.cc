#include <cstdint>
#include <type_traits>

#include "mol_epilogue_kernel.hpp"
#include "top_k_by_key_async_kernel.hpp"
#include "top_k_by_key_kernel.hpp"
#include "top_k_by_key_radix_select_kernel.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#define XAI_TOPK_FFI_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

XLA_FFI_DEFINE_HANDLER(
    kTopKByKey,
    top_k_by_key_bf16,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()
        .Attr<int64_t>("k")
        .Attr<double>("heuristic_pivot_ratio")
        .Ret<ffi::Buffer<ffi::DataType::BF16>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
);

XLA_FFI_DEFINE_HANDLER(
    kTopKByKeyAsync,
    top_k_by_key_bf16_async,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()
        .Attr<int64_t>("k")
        .Ret<ffi::Buffer<ffi::DataType::BF16>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
);

XLA_FFI_DEFINE_HANDLER(
    kTopKByKeyRadixSelect,
    top_k_by_key_bf16_radix_select,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()
        .Attr<int64_t>("k")
        .Ret<ffi::Buffer<ffi::DataType::BF16>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
);

XLA_FFI_DEFINE_HANDLER(
    kXrexMolEpilogue,
    mol_epilogue,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Attr<int64_t>("num_users")
        .Attr<int64_t>("num_components")
);

}

XAI_TOPK_FFI_EXPORT int32_t xai_topk_ffi_api_version(void) {
  return XLA_FFI_API_MAJOR * 1000 + XLA_FFI_API_MINOR;
}

XAI_TOPK_FFI_EXPORT XLA_FFI_Error* xai_topk_ffi_top_k_by_key(XLA_FFI_CallFrame* call_frame) {
  return kTopKByKey(call_frame);
}

XAI_TOPK_FFI_EXPORT XLA_FFI_Error* xai_topk_ffi_top_k_by_key_async(XLA_FFI_CallFrame* call_frame) {
  return kTopKByKeyAsync(call_frame);
}

XAI_TOPK_FFI_EXPORT XLA_FFI_Error* xai_topk_ffi_top_k_by_key_radix_select(
    XLA_FFI_CallFrame* call_frame
) {
  return kTopKByKeyRadixSelect(call_frame);
}

XAI_TOPK_FFI_EXPORT XLA_FFI_Error* xai_topk_ffi_xrex_mol_epilogue(XLA_FFI_CallFrame* call_frame) {
  return kXrexMolEpilogue(call_frame);
}

static_assert(
    std::is_invocable_r_v<XLA_FFI_Error*, decltype(xai_topk_ffi_top_k_by_key), XLA_FFI_CallFrame*>,
    "exported handlers must have the XLA_FFI_Handler signature"
);
