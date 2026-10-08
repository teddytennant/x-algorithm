# Copyright (c) 2025-2026, QuACK team.
#
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
import os

_PATCHED = False


def patch() -> None:
    global _PATCHED
    if _PATCHED or os.getenv("QUACK_DISABLE_MLIR_THREADING_PATCH", "0") == "1":
        return
    try:
        from cutlass._mlir import ir
    except Exception:
        return

    orig_context = ir.Context
    if getattr(orig_context, "_quack_mlir_threading_patch", False):
        _PATCHED = True
        return

    class SingleThreadedContext(orig_context):
        _quack_mlir_threading_patch = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_multithreading(False)

    SingleThreadedContext.__name__ = orig_context.__name__
    SingleThreadedContext.__qualname__ = orig_context.__qualname__
    SingleThreadedContext.__module__ = orig_context.__module__
    ir.Context = SingleThreadedContext
    _PATCHED = True
