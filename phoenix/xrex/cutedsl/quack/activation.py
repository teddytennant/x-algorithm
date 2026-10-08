# Copyright (c) 2025, Tri Dao.
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
import math
from functools import partial
from typing import Tuple

import cutlass.cute as cute
from cutlass import Boolean, Float32, const_expr
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

F32_or_F32x2 = Float32 | Tuple[Float32, Float32]


sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
)


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def sigmoid_tanh(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return 0.5 + 0.5 * tanh(0.5 * x)
    else:
        x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x)
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return cute.arch.fma_packed_f32x2(tanh_x_half, (0.5, 0.5), (0.5, 0.5))


@dsl_user_op
def sigmoid(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.rcp_approx(1.0 + cute.math.exp(-x, fastmath=True), loc=loc, ip=ip)
    else:
        log2_e = math.log2(math.e)
        neg_x = cute.arch.mul_packed_f32x2(x, (-log2_e, -log2_e))
        exp_neg_x = (
            cute.math.exp2(neg_x[0], fastmath=True),
            cute.math.exp2(neg_x[1], fastmath=True),
        )
        denom = cute.arch.add_packed_f32x2(exp_neg_x, (1.0, 1.0))
        return cute.arch.rcp_approx(denom[0]), cute.arch.rcp_approx(denom[1])


@dsl_user_op
def dsigmoid_from_output(out: Float32, dout: Float32, *, loc=None, ip=None) -> Float32:
    return dout * (out - out * out)


@dsl_user_op
def relu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0))
    else:
        return cute.arch.fmax(x[0], Float32(0.0)), cute.arch.fmax(x[1], Float32(0.0))


@dsl_user_op
@cute.jit
def drelu(
    x: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        x_pos = Boolean(x > 0)
        return dout if x_pos else Float32(0.0), cute.arch.fmax(x, Float32(0.0))
    else:
        x0_pos = Boolean(x[0] > 0)
        x1_pos = Boolean(x[1] > 0)
        dx = (dout[0] if x0_pos else Float32(0.0), dout[1] if x1_pos else Float32(0.0))
        return dx, relu(x)


@dsl_user_op
def relu_sq(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * x
    else:
        relu_x = (cute.arch.fmax(x[0], Float32(0.0)), cute.arch.fmax(x[1], Float32(0.0)))
        return cute.arch.mul_packed_f32x2(relu_x, x)


@dsl_user_op
@cute.jit
def drelu_sq(
    x: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        relu_x = relu(x)
        relu_sq_out = relu_x * x
        dx = 2.0 * (dout * relu_x)
        return dx, relu_sq_out
    else:
        relu_x = relu(x)
        relu_sq_out = cute.arch.mul_packed_f32x2(relu_x, x)
        dx = cute.arch.mul_packed_f32x2((2.0, 2.0), cute.arch.mul_packed_f32x2(dout, relu_x))
        return dx, relu_sq_out


@dsl_user_op
def gelu_tanh_approx(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi
    if const_expr(not isinstance(x, tuple)):
        return 0.5 * (x * (1.0 + tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x)))))
    else:
        x_sq = cute.arch.mul_packed_f32x2(x, x)
        x_sq_scaled = cute.arch.fma_packed_f32x2(
            x_sq, (sqrt_2_over_pi_coeff, sqrt_2_over_pi_coeff), (sqrt_2_over_pi, sqrt_2_over_pi)
        )
        z = cute.arch.mul_packed_f32x2(x, x_sq_scaled)
        tanh_z = (tanh(z[0]), tanh(z[1]))
        x_tanh_z = cute.arch.fma_packed_f32x2(tanh_z, x, x)
        return cute.arch.mul_packed_f32x2((0.5, 0.5), x_tanh_z)


@dsl_user_op
def dgelu_tanh_approx(
    x: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi
    sqrt_2_over_pi_coeff_3 = 3.0 * sqrt_2_over_pi_coeff

    if const_expr(not isinstance(x, tuple)):
        x_sq = x * x
        tanh_z = tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq))
        half_tanh_z_plus_one = 0.5 + 0.5 * tanh_z
        gelu_out = x * half_tanh_z_plus_one

        sech2_z = 1 - tanh_z * tanh_z
        dz_dx = sqrt_2_over_pi + sqrt_2_over_pi_coeff_3 * x_sq
        dgelu = half_tanh_z_plus_one + x * (0.5 * (sech2_z * dz_dx))

        dx = dout * dgelu
        return dx, gelu_out
    else:
        x_sq = cute.arch.mul_packed_f32x2(x, x)
        x_sq_scaled = cute.arch.fma_packed_f32x2(
            x_sq, (sqrt_2_over_pi_coeff, sqrt_2_over_pi_coeff), (sqrt_2_over_pi, sqrt_2_over_pi)
        )
        z = cute.arch.mul_packed_f32x2(x, x_sq_scaled)
        tanh_z = (tanh(z[0]), tanh(z[1]))
        half_tanh_z_plus_one = cute.arch.fma_packed_f32x2(tanh_z, (0.5, 0.5), (0.5, 0.5))
        gelu_out = cute.arch.mul_packed_f32x2(x, half_tanh_z_plus_one)

        sech2_z = cute.arch.fma_packed_f32x2(tanh_z, (-tanh_z[0], -tanh_z[1]), (1.0, 1.0))
        dz_dx = cute.arch.fma_packed_f32x2(
            x_sq, (sqrt_2_over_pi_coeff_3, sqrt_2_over_pi_coeff_3), (sqrt_2_over_pi, sqrt_2_over_pi)
        )
        sech2_dz_dx = cute.arch.mul_packed_f32x2(sech2_z, dz_dx)
        x_sech2_dz_dx = cute.arch.mul_packed_f32x2(x, sech2_dz_dx)
        dgelu = cute.arch.fma_packed_f32x2(x_sech2_dz_dx, (0.5, 0.5), half_tanh_z_plus_one)

        dx = cute.arch.mul_packed_f32x2(dout, dgelu)
        return dx, gelu_out


@dsl_user_op
@cute.jit
def softplus(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        use_linear = Boolean(x > 20.0)
        return (
            cute.math.log(Float32(cute.math.exp(x, fastmath=True)) + 1.0, fastmath=True)
            if not use_linear
            else x
        )
    else:
        log2_e = math.log2(math.e)
        x_log2e = cute.arch.mul_packed_f32x2(x, (log2_e, log2_e))
        x_exp = (cute.math.exp(x_log2e[0], fastmath=True), cute.math.exp(x_log2e[1], fastmath=True))
        x_exp_p1 = cute.arch.add_packed_f32x2(x_exp, (1.0, 1.0))
        log_x_exp_p1 = (
            cute.math.log2(x_exp_p1[0], fastmath=True),
            cute.math.log2(x_exp_p1[1], fastmath=True),
        )
        ln2 = math.log(2.0)
        softplus_x = cute.arch.mul_packed_f32x2(log_x_exp_p1, (ln2, ln2))
        use_linear_0 = Boolean(x[0] > 20.0)
        use_linear_1 = Boolean(x[1] > 20.0)
        return (
            softplus_x[0] if not use_linear_0 else x[0],
            softplus_x[1] if not use_linear_1 else x[1],
        )


@dsl_user_op
@cute.jit
def dsoftplus_from_output(out: Float32, dout: Float32, *, loc=None, ip=None) -> Float32:
    use_linear = Boolean(out > 20.0)
    dx = dout - dout * cute.math.exp(-out, fastmath=True)
    return dx if not use_linear else dout


@dsl_user_op
def silu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return x * sigmoid(x)
    else:
        return cute.arch.mul_packed_f32x2(x, sigmoid(x))


@dsl_user_op
def silu_tanh(x: F32_or_F32x2, *, already_halved: bool = False, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x if const_expr(not already_halved) else x
        return x_half * tanh(x_half) + x_half
    else:
        x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x) if const_expr(not already_halved) else x
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return cute.arch.fma_packed_f32x2(x_half, tanh_x_half, x_half)


@dsl_user_op
def dsilu(
    x: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        silu_x = x * sigmoid_x
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x + silu_x) * dout
        return d_silu_x_dout, silu_x
    else:
        sigmoid_x = sigmoid(x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_x)
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x = cute.arch.add_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, silu_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x, dout
        )
        return d_silu_x_dout, silu_x


@dsl_user_op
def dsilu_tanh(
    x: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    already_halved: bool = False,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        if const_expr(not already_halved):
            x_half = 0.5 * x
            tanh_x_half = tanh(x_half)
            sigmoid_x = 0.5 * tanh_x_half + 0.5
            silu_x = x_half * tanh_x_half + x_half
        else:
            tanh_x = tanh(x)
            sigmoid_x = 0.5 * tanh_x + 0.5
            silu_x = x * tanh_x + x
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x + silu_x) * dout
        return d_silu_x_dout, silu_x
    else:
        if const_expr(not already_halved):
            x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x)
            tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
            sigmoid_x = cute.arch.fma_packed_f32x2(tanh_x_half, (0.5, 0.5), (0.5, 0.5))
            silu_x = cute.arch.fma_packed_f32x2(x_half, tanh_x_half, x_half)
        else:
            tanh_x = (tanh(x[0]), tanh(x[1]))
            sigmoid_x = cute.arch.fma_packed_f32x2(tanh_x, (0.5, 0.5), (0.5, 0.5))
            silu_x = cute.arch.fma_packed_f32x2(x, tanh_x, x)
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x = cute.arch.add_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, silu_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x_plus_silu_x, dout
        )
        return d_silu_x_dout, silu_x


@dsl_user_op
def swiglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return silu(x) * y
    else:
        return cute.arch.mul_packed_f32x2(silu(x), y)


@dsl_user_op
def swiglu_tanh(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return silu_tanh(x) * y
    else:
        return cute.arch.mul_packed_f32x2(silu_tanh(x), y)


@dsl_user_op
def dswiglu(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        silu_x = x * sigmoid_x
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x_dout
        dx = d_silu_x_dout * y
        dy = silu_x_dout
        swiglu_out = silu_x * y
        return dx, dy, swiglu_out
    else:
        sigmoid_x = sigmoid(x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        d_silu_x_dout = cute.arch.fma_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, dout, silu_x_dout
        )
        dx = cute.arch.mul_packed_f32x2(d_silu_x_dout, y)
        dy = silu_x_dout
        swiglu_out = cute.arch.mul_packed_f32x2(silu_x, y)
        return dx, dy, swiglu_out


@dsl_user_op
def dswiglu_tanh(
    x: F32_or_F32x2,
    y: F32_or_F32x2,
    dout: F32_or_F32x2,
    *,
    already_halved: bool = False,
    loc=None,
    ip=None,
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        if const_expr(not already_halved):
            sigmoid_x = sigmoid_tanh(x)
            silu_x = x * sigmoid_x
        else:
            tanh_x = tanh(x)
            sigmoid_x = 0.5 * tanh_x + 0.5
            silu_x = x * tanh_x + x
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x_dout
        dx = d_silu_x_dout * y
        dy = silu_x_dout
        swiglu_out = silu_x * y
        return dx, dy, swiglu_out
    else:
        if const_expr(not already_halved):
            sigmoid_x = sigmoid_tanh(x)
            silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_x)
        else:
            tanh_x = (tanh(x[0]), tanh(x[1]))
            sigmoid_x = cute.arch.fma_packed_f32x2(tanh_x, (0.5, 0.5), (0.5, 0.5))
            silu_x = cute.arch.fma_packed_f32x2(x, tanh_x, x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        sigmoid_x_minus_silu_x_sigmoid_x = cute.arch.fma_packed_f32x2(
            sigmoid_x, (-silu_x[0], -silu_x[1]), sigmoid_x
        )
        d_silu_x_dout = cute.arch.fma_packed_f32x2(
            sigmoid_x_minus_silu_x_sigmoid_x, dout, silu_x_dout
        )
        dx = cute.arch.mul_packed_f32x2(d_silu_x_dout, y)
        dy = silu_x_dout
        swiglu_out = cute.arch.mul_packed_f32x2(silu_x, y)
        return dx, dy, swiglu_out


@dsl_user_op
def swiglu_oai(
    x: F32_or_F32x2, y: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        sigmoid_alpha_x = sigmoid(alpha * x)
        silu_x = x * sigmoid_alpha_x
        return silu_x * y + silu_x
    else:
        alpha_x = cute.arch.mul_packed_f32x2((alpha, alpha), x)
        sigmoid_alpha_x = sigmoid(alpha_x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_alpha_x)
        return cute.arch.fma_packed_f32x2(silu_x, y, silu_x)


@dsl_user_op
def swiglu_oai_tanh(
    x: F32_or_F32x2, y: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x
        silu_x = x_half * tanh(alpha * x_half) + x_half
        return silu_x * y + silu_x
    else:
        x_half = cute.arch.mul_packed_f32x2((0.5, 0.5), x)
        alpha_x_half = cute.arch.mul_packed_f32x2((alpha, alpha), x_half)
        tanh_alpha_x_half = (tanh(alpha_x_half[0]), tanh(alpha_x_half[1]))
        silu_x = cute.arch.fma_packed_f32x2(x_half, tanh_alpha_x_half, x_half)
        return cute.arch.fma_packed_f32x2(silu_x, y, silu_x)


@dsl_user_op
def dswiglu_oai(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        sigmoid_alpha_x = sigmoid(alpha * x)
        silu_x = x * sigmoid_alpha_x
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sigmoid_alpha_x + alpha * (silu_x - silu_x * sigmoid_alpha_x)) * dout
        dx = d_silu_x_dout * y + d_silu_x_dout
        dy = silu_x_dout
        swiglu_out = silu_x * y + silu_x
        return dx, dy, swiglu_out
    else:
        alpha_x = cute.arch.mul_packed_f32x2((alpha, alpha), x)
        sigmoid_alpha_x = sigmoid(alpha_x)
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_alpha_x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        silu_x_minus_product = cute.arch.fma_packed_f32x2(
            silu_x, (-sigmoid_alpha_x[0], -sigmoid_alpha_x[1]), silu_x
        )
        sigmoid_plus_alpha_diff = cute.arch.fma_packed_f32x2(
            (alpha, alpha), silu_x_minus_product, sigmoid_alpha_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(sigmoid_plus_alpha_diff, dout)
        dx = cute.arch.fma_packed_f32x2(d_silu_x_dout, y, d_silu_x_dout)
        dy = silu_x_dout
        swiglu_out = cute.arch.fma_packed_f32x2(silu_x, y, silu_x)
        return dx, dy, swiglu_out


@dsl_user_op
def dswiglu_oai_tanh(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        alpha_x_half = (0.5 * alpha) * x
        sigmoid_alpha_x = 0.5 + 0.5 * tanh(alpha_x_half)
        silu_x = x * sigmoid_alpha_x
        silu_x_dout = silu_x * dout
        d_silu_x_dout = (sigmoid_alpha_x + alpha * (silu_x - silu_x * sigmoid_alpha_x)) * dout
        dx = d_silu_x_dout * y + d_silu_x_dout
        dy = silu_x_dout
        swiglu_out = silu_x * y + silu_x
        return dx, dy, swiglu_out
    else:
        alpha_x_half = cute.arch.mul_packed_f32x2(((0.5 * alpha), (0.5 * alpha)), x)
        tanh_alpha_x_half = (tanh(alpha_x_half[0]), tanh(alpha_x_half[1]))
        sigmoid_alpha_x = cute.arch.fma_packed_f32x2(tanh_alpha_x_half, (0.5, 0.5), (0.5, 0.5))
        silu_x = cute.arch.mul_packed_f32x2(x, sigmoid_alpha_x)
        silu_x_dout = cute.arch.mul_packed_f32x2(silu_x, dout)
        silu_x_minus_product = cute.arch.fma_packed_f32x2(
            silu_x, (-sigmoid_alpha_x[0], -sigmoid_alpha_x[1]), silu_x
        )
        sigmoid_plus_alpha_diff = cute.arch.fma_packed_f32x2(
            (alpha, alpha), silu_x_minus_product, sigmoid_alpha_x
        )
        d_silu_x_dout = cute.arch.mul_packed_f32x2(sigmoid_plus_alpha_diff, dout)
        dx = cute.arch.fma_packed_f32x2(d_silu_x_dout, y, d_silu_x_dout)
        dy = silu_x_dout
        swiglu_out = cute.arch.fma_packed_f32x2(silu_x, y, silu_x)
        return dx, dy, swiglu_out


@dsl_user_op
def glu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        return sigmoid_x * y
    else:
        sigmoid_x = sigmoid(x)
        return cute.arch.mul_packed_f32x2(sigmoid_x, y)


@dsl_user_op
def dglu(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)
        sigmoid_x_dout = sigmoid_x * dout
        glu_out = sigmoid_x * y
        dx = (y - glu_out) * sigmoid_x_dout
        dy = sigmoid_x_dout
        return dx, dy, glu_out
    else:
        sigmoid_x = sigmoid(x)
        sigmoid_x_dout = cute.arch.mul_packed_f32x2(sigmoid_x, dout)
        glu_out = cute.arch.mul_packed_f32x2(sigmoid_x, y)
        y_minus_glu_out = sub_packed_f32x2(y, glu_out)
        dx = cute.arch.mul_packed_f32x2(y_minus_glu_out, sigmoid_x_dout)
        dy = sigmoid_x_dout
        return dx, dy, glu_out


@dsl_user_op
def reglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * y
    else:
        relu_x = relu(x)
        return cute.arch.mul_packed_f32x2(relu_x, y)


@dsl_user_op
@cute.jit
def dreglu(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        x_pos = Boolean(x > 0)
        relu_x = cute.arch.fmax(x, Float32(0.0))
        dx = (dout * y) if x_pos else Float32(0.0)
        dy = dout * relu_x
        reglu_out = relu_x * y
        return dx, dy, reglu_out
    else:
        x0_pos = Boolean(x[0] > 0)
        x1_pos = Boolean(x[1] > 0)
        relu_x = relu(x)
        dout_y = cute.arch.mul_packed_f32x2(dout, y)
        dx = ((dout_y[0] if x0_pos else Float32(0.0)), (dout_y[1] if x1_pos else Float32(0.0)))
        dy = cute.arch.mul_packed_f32x2(dout, relu_x)
        reglu_out = cute.arch.mul_packed_f32x2(relu_x, y)
        return dx, dy, reglu_out


@dsl_user_op
def geglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return gelu_tanh_approx(x) * y
    else:
        return cute.arch.mul_packed_f32x2(gelu_tanh_approx(x), y)


@dsl_user_op
def dgeglu(
    x: F32_or_F32x2, y: F32_or_F32x2, dout: F32_or_F32x2, *, loc=None, ip=None
) -> Tuple[F32_or_F32x2, F32_or_F32x2, F32_or_F32x2]:
    if const_expr(not isinstance(x, tuple)):
        dgelu_x_dout, gelu_x = dgelu_tanh_approx(x, dout)
        dx = dgelu_x_dout * y
        dy = gelu_x * dout
        geglu_out = gelu_x * y
        return dx, dy, geglu_out
    else:
        dgelu_x_dout, gelu_x = dgelu_tanh_approx(x, dout)
        dx = cute.arch.mul_packed_f32x2(dgelu_x_dout, y)
        dy = cute.arch.mul_packed_f32x2(gelu_x, dout)
        geglu_out = cute.arch.mul_packed_f32x2(gelu_x, y)
        return dx, dy, geglu_out


act_fn_map = {
    None: None,
    "silu": silu,
    "silu-tanh": silu_tanh,
    "relu": relu,
    "relu_sq": relu_sq,
    "gelu_tanh_approx": gelu_tanh_approx,
}

dact_fn_map = {
    None: None,
    "silu": dsilu,
    "silu-tanh": dsilu_tanh,
    "relu": drelu,
    "relu_sq": drelu_sq,
    "gelu_tanh_approx": dgelu_tanh_approx,
}

gate_fn_map = {
    "swiglu": swiglu,
    "swiglu-tanh": swiglu_tanh,
    "swiglu_oai": swiglu_oai,
    "swiglu_oai-tanh": swiglu_oai_tanh,
    "reglu": reglu,
    "geglu": geglu,
    "glu": glu,
}

dgate_fn_map = {
    "swiglu": dswiglu,
    "swiglu-tanh": dswiglu_tanh,
    "swiglu_oai": dswiglu_oai,
    "swiglu_oai-tanh": dswiglu_oai_tanh,
    "reglu": dreglu,
    "geglu": dgeglu,
    "glu": dglu,
}
