#include "nanobind/nanobind.h"
#include "top_k_by_key_radix_select_kernel.hpp"

namespace nb = nanobind;

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    top_k_by_key_radix_select,
    top_k_by_key_bf16_radix_select,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::DataType::BF16>>()
        .Attr<int64_t>("k")
        .Ret<ffi::Buffer<ffi::DataType::BF16>>()
        .Ret<ffi::Buffer<ffi::DataType::S32>>()
);

template <typename T>
nb::capsule encapsulate_ffi_call(T* fn) {
  static_assert(
      std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
      "Encapsulated function must be an XLA FFI handler"
  );
  return nb::capsule(reinterpret_cast<void*>(fn));
}

NB_MODULE(top_k_by_key_radix_select_api, m) {
  m.doc() = "Radix-select top-k by key (BF16, unordered output)";
  m.def("top_k_by_key_radix_select", []() {
    return encapsulate_ffi_call(top_k_by_key_radix_select);
  });
}
