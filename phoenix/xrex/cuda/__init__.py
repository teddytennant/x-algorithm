# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
"""CUDA kernels used by ``xrex``.

``adler32`` (checkpoint integrity), ``unique`` (training embedding compressor) and
``top_k_by_key`` (retrieval serving) each have a pure-JAX (or zlib) reference path and a
compiled path. Each subpackage imports ``xrex_cuda_kernels.<api>`` at import time: on success
it registers the kernel as an XLA FFI target; on ``ModuleNotFoundError`` it logs a warning and
runs the reference, so a venv silently stuck on the reference path is visible.

``async_emb``, ``row_emb`` and ``fa3`` are compiled-only: a missing extension raises a named
``ImportError`` that callers treat as the feature being unavailable (``use_async_emb``,
``use_row_emb``, ``attn_impl="flash_attn"``). For every kernel, a present-but-broken extension
raises its own error.

FFI target names are prefixed ``xrex_`` except top-k's ``top_k_by_key*``, which serving
matches by name.

The ``xrex-cuda-kernels`` package is a bazel wheel defined by ``xrex/cuda/wheel/BUILD``.
Installing ``xrex/cuda/wheel`` with ``pip``/``uv`` runs that build through
``wheel/backend.py`` (needs bazelisk and git; the CUDA toolchain is fetched hermetically).
``wheel_inventory_test.py`` fails if a kernel lacks its build, wheel entry or loader import.
"""
