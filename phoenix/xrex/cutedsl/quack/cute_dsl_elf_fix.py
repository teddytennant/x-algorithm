# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from quack-kernels (https://github.com/Dao-AILab/quack);
# modified by X.AI Corp.

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import struct

_PATCHED = False


def _fix_elf_dup_text_flags(data: bytes) -> bytes:
    if len(data) < 64 or data[4] != 2 or data[5] != 1:
        return data
    e_shoff = struct.unpack_from("<Q", data, 40)[0]
    e_shentsize = struct.unpack_from("<H", data, 58)[0]
    e_shnum = struct.unpack_from("<H", data, 60)[0]
    e_shstrndx = struct.unpack_from("<H", data, 62)[0]
    if not e_shoff or not e_shnum or e_shstrndx >= e_shnum:
        return data
    shstr_hdr = e_shoff + e_shstrndx * e_shentsize
    shstr_off = struct.unpack_from("<Q", data, shstr_hdr + 24)[0]
    text_secs: list[tuple[int, int]] = []
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        ni = struct.unpack_from("<I", data, sh)[0]
        ns = shstr_off + ni
        if ns + 6 <= len(data) and data[ns : ns + 6] == b".text\x00":
            text_secs.append((i, sh))
    if len(text_secs) <= 1:
        return data
    result = bytearray(data)
    for _, sh in text_secs[1:]:
        struct.pack_into("<Q", result, sh + 8, 0x6)
    return bytes(result)


def patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        from cutlass.base_dsl.common import DSLRuntimeError
        from cutlass.base_dsl.export import external_binary_module as _ebm
    except Exception:
        return

    cls = _ebm.ExternalBinaryModule
    orig_init = cls.__init__

    def patched_init(self, file_path: str, enable_tvm_ffi: bool = False) -> None:
        self.enable_tvm_ffi = enable_tvm_ffi
        assert self.load_provider is not None, "Load provider is not set for ExternalBinaryModule."
        shared_libs = self.load_provider.dsl._get_dsl().get_shared_libs()
        object_file_content = bytes()
        if file_path.endswith(".so"):
            shared_libs.append(file_path)
        else:
            try:
                with open(file_path, "rb") as f:
                    object_file_content = f.read()
            except Exception as e:
                raise DSLRuntimeError(f"Failed to read object file {file_path}: {e}")

        useJitLink = not enable_tvm_ffi
        if not useJitLink and object_file_content:
            object_file_content = _fix_elf_dup_text_flags(object_file_content)
            useJitLink = True
        self.engine = self.load_provider.execution_engine_constructor(
            object_file_content, shared_libs, useJitLink
        )

    patched_init.__wrapped__ = orig_init
    cls.__init__ = patched_init
    _PATCHED = True
