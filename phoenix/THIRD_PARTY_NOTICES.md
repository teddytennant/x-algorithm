# Third-party notices

## flash-attention (BSD-3-Clause)

The files under `xrex/cutedsl/ranker_fa4/` are a CuTeDSL reimplementation of
kernels from the flash-attention project, modified by X.AI Corp.:

    https://github.com/Dao-AILab/flash-attention

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Every shipped file under `xrex/cutedsl/ranker_fa4/` carries this notice
stamped at the top of the file, and the repository-level `NOTICE` file
enumerates every attributed file.

## NVIDIA CUTLASS (BSD-3-Clause)

`xrex/cutedsl/ranker_fa4/mma_sm100_desc.py` is a Python port of CUTLASS C++
sources, and `xrex/cutedsl/ranker_fa4/flash_fwd_sm100.py` is based on a CUTLASS
example, from the NVIDIA CUTLASS project:

    https://github.com/NVIDIA/cutlass

Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## jax-triton (Apache-2.0)

`xrex/pallas/ranker_attention.py` is derived from the jax-triton project:

    https://github.com/jax-ml/jax-triton

Copyright 2023 The jax_triton Authors. Licensed under the Apache License,
Version 2.0; see https://www.apache.org/licenses/LICENSE-2.0.

## QuACK (Apache-2.0)

The files under `xrex/cutedsl/quack/` are the torch-free subset of the
quack-kernels project that `xrex/cutedsl/ranker_fa4/` uses, modified by
X.AI Corp.:

    https://github.com/Dao-AILab/quack

Copyright (c) 2025-2026, the QuACK authors (per-file holders are retained in
each file's header). Licensed under the Apache License, Version 2.0; see
`xrex/cutedsl/quack/LICENSE` and https://www.apache.org/licenses/LICENSE-2.0.
