# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Temporal-RoPE rotation of FadeMem keys where attention reads them.

FadeMem stores keys without their temporal phase. Every chunk re-applies the
phase for that chunk's history layout. Rotating into a persistent copy kept a
second full-capacity key buffer (about 0.33 GiB); rotating inside the compiled
denoise and refresh passes keeps only one layer's rotated keys alive at a time.

The rotation must match :func:`rotate_half_reference` bit for bit, so it is an
opaque custom op: Inductor would otherwise fuse it and contract the multiply
and add into an FMA, which rounds differently. On CUDA a Triton kernel computes
the same float32 operations in the same order in inline PTX: on Blackwell,
Triton packs the arithmetic into ``f32x2`` instructions, and ptxas contracts
those into FMAs even with ``enable_fp_fusion=False``; it never contracts the
scalar ``.rn`` instructions.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None


def rotate_half_reference(key: Tensor, cos: Tensor, sin: Tensor, sign: int) -> Tensor:
    """Apply ``x * cos + sign * rotate_half(x) * sin`` in float32.

    Args:
        key: ``[..., S, D]`` keys.
        cos: ``[S, D]`` float32 table.
        sin: ``[S, D]`` float32 table.
        sign: ``1`` applies the phase, ``-1`` removes it.

    Returns:
        The rotated keys in ``key``'s dtype.
    """
    x = key.float()
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos[None, None] + sign * rotated * sin[None, None]).to(key.dtype)


if triton is not None:

    @triton.jit
    def _mul_sub(a, b, c, d):
        """Return ``a * b - c * d`` with each operation rounded on its own."""
        return tl.inline_asm_elementwise(
            "{ .reg .f32 t, u; mul.rn.f32 t, $1, $2; mul.rn.f32 u, $3, $4; "
            "sub.rn.f32 $0, t, u; }",
            "=f,f,f,f,f",
            [a, b, c, d],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _mul_add(a, b, c, d):
        """Return ``a * b + c * d`` with each operation rounded on its own."""
        return tl.inline_asm_elementwise(
            "{ .reg .f32 t, u; mul.rn.f32 t, $1, $2; mul.rn.f32 u, $3, $4; "
            "add.rn.f32 $0, t, u; }",
            "=f,f,f,f,f",
            [a, b, c, d],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _rotate_kernel(
        key_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        seq,
        heads,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_ts,
        HALF: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        batch_head = tl.program_id(1)
        batch = batch_head // heads
        head = batch_head % heads
        cols = tl.arange(0, HALF)
        mask = rows[:, None] < seq
        src = (
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + rows[:, None] * stride_ks
            + cols[None, :]
        )
        x1 = tl.load(src, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(src + HALF, mask=mask, other=0.0).to(tl.float32)
        table = rows[:, None] * stride_ts + cols[None, :]
        c1 = tl.load(cos_ptr + table, mask=mask, other=0.0)
        c2 = tl.load(cos_ptr + table + HALF, mask=mask, other=0.0)
        s1 = tl.load(sin_ptr + table, mask=mask, other=0.0)
        s2 = tl.load(sin_ptr + table + HALF, mask=mask, other=0.0)
        # Same operations and order as the reference: x * cos + (rotated * sin),
        # where rotated = cat(-x2, x1); x1 * c1 + (-x2 * s1) == x1 * c1 - x2 * s1
        # exactly in IEEE arithmetic.
        o1 = _mul_sub(x1, c1, x2, s1)
        o2 = _mul_add(x2, c2, x1, s2)
        dst = out_ptr + batch_head * seq * 2 * HALF + rows[:, None] * 2 * HALF
        tl.store(dst + cols[None, :], o1.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(
            dst + HALF + cols[None, :], o2.to(out_ptr.dtype.element_ty), mask=mask
        )


def _triton_supported(key: Tensor, cos: Tensor, sin: Tensor) -> bool:
    half = key.shape[-1] // 2
    return (
        triton is not None
        and key.is_cuda
        and key.ndim == 4
        and key.dtype in (torch.bfloat16, torch.float16)
        and key.stride(-1) == 1
        and key.shape[-1] % 2 == 0
        and half >= 16
        and half & (half - 1) == 0
        and cos.dtype == sin.dtype == torch.float32
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.shape == sin.shape == key.shape[-2:]
    )


@torch.library.custom_op("cosmos3_h_dreams::rotate_temporal", mutates_args=())
def rotate_temporal(key: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply temporal RoPE to ``[B, H, S, D]`` keys; equal to the reference.

    Args:
        key: Phase-free keys, possibly a strided view of the FadeMem store.
        cos: ``[S, D]`` float32 table for this chunk's history layout.
        sin: ``[S, D]`` float32 table.

    Returns:
        A new contiguous tensor of rotated keys.
    """
    if not _triton_supported(key, cos, sin):
        return rotate_half_reference(key, cos, sin, 1).contiguous()
    batch, heads, seq, dim = key.shape
    out = torch.empty((batch, heads, seq, dim), dtype=key.dtype, device=key.device)
    block = 32
    with torch.cuda.device(key.device):
        _rotate_kernel[(triton.cdiv(seq, block), batch * heads)](
            key,
            cos,
            sin,
            out,
            seq,
            heads,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            cos.stride(0),
            HALF=dim // 2,
            BLOCK_S=block,
            enable_fp_fusion=False,
        )
    return out


@rotate_temporal.register_fake
def _(key: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    del cos, sin
    return key.new_empty(key.shape)
