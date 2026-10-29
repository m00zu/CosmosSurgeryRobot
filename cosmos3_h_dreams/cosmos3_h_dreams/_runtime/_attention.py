# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoT joint attention: the understanding/generation visibility rule.

Cosmos3's MoT layer fuses the understanding (text) and generation (vision) token
streams in one masked attention with this visibility rule:

* an **understanding** query attends causally to understanding keys only
  (text self-attention; it does not see generation tokens);
* a **generation** query attends to *all* understanding keys (full cross-
  attention) **and** to generation keys up to its own temporal frame
  (temporal-causal self-attention — our checkpoint is ``three_way`` +
  ``video_temporal_causal``). Within a frame, spatial tokens see each other.

The dense ``[S, S]`` boolean mask provides a correctness reference for SDPA.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # pragma: no cover - exercised only on older PyTorch builds
    varlen_attn = None


def varlen_attention_available() -> bool:
    """Whether this PyTorch build exposes fused variable-length attention."""
    return varlen_attn is not None


def sdpa_supports_native_gqa() -> bool:
    """Whether this PyTorch build exposes SDPA's ``enable_gqa`` keyword."""
    return "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")


ATTENTION_KERNELS = ("auto", "default", "sage")
"""``attention_kernel`` setting: ``auto`` uses SageAttention on RTX 50 (sm_120) when a
stream-fixed build is installed, ``default`` always uses PyTorch SDPA, ``sage``
requires SageAttention."""

SAGE_INSTALL_HINT = "install it with scripts/install_sageattention.sh"

_unmasked_kernel: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
"""Replaces SDPA for mask-free calls; set by :func:`select_attention_kernel`."""


@dataclass(frozen=True)
class AttentionChoice:
    kernel: str
    """``"default"`` or ``"sage"``."""
    reason: str


@torch.library.custom_op("cosmos3_h_dreams::sage_attention", mutates_args=())
def sage_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """SageAttention2 as an op Inductor treats as opaque; K/V may have fewer heads."""
    from sageattention import sageattn

    q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
    return sageattn(q, k, v, tensor_layout="HND", is_causal=False)


@sage_attention.register_fake
def _(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def sage_self_test(device: torch.device) -> str | None:
    """``None`` if SageAttention is accurate and CUDA-graph safe, else the problem.

    Upstream SageAttention launches on the legacy default stream, so graph
    capture misses its kernels and replays read stale memory (NaN video once
    the FadeMem cache fills). A captured replay must equal an eager call.
    """
    gen = torch.Generator(device=device).manual_seed(0)

    def rand(heads: int, tokens: int) -> Tensor:
        return torch.randn(
            1, heads, tokens, 128, device=device, dtype=torch.bfloat16, generator=gen
        )

    q, k, v = rand(32, 256), rand(8, 1024), rand(8, 1024)
    try:
        eager = sage_attention(q, k, v)
        reference = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            sage_attention(q, k, v)
        torch.cuda.current_stream(device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = sage_attention(q, k, v)
        captured.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - any failure means "don't use it"
        return f"self-test raised {type(exc).__name__}: {exc}"
    if not torch.equal(captured, eager):
        return "a CUDA-graph replay differs from an eager call (build lacks the stream fix)"
    cosine = F.cosine_similarity(
        eager.float().flatten(), reference.float().flatten(), dim=0
    )
    if not bool(torch.isfinite(eager).all()) or float(cosine) < 0.99:
        return f"output too far from SDPA (cosine {float(cosine):.4f})"
    return None


def select_attention_kernel(choice: str, device: str | torch.device) -> AttentionChoice:
    """Pick the kernel for mask-free attention; call before the model compiles.

    Raises:
        ValueError: unknown ``choice``.
        RuntimeError: ``choice="sage"`` and SageAttention is missing or fails
            :func:`sage_self_test`.
    """
    global _unmasked_kernel
    if choice not in ATTENTION_KERNELS:
        raise ValueError(f"attention_kernel must be one of {ATTENTION_KERNELS}, got {choice!r}")
    _unmasked_kernel = None
    if choice == "default":
        return AttentionChoice("default", "set to default")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        if choice == "sage":
            raise RuntimeError("attention_kernel: sage needs a CUDA device")
        return AttentionChoice("default", "no CUDA device")
    major, minor = torch.cuda.get_device_capability(device)
    if choice == "auto" and major != 12:
        return AttentionChoice(
            "default", f"sm_{major}{minor}: SageAttention is only faster on RTX 50 (sm_120)"
        )
    try:
        import sageattention  # noqa: F401
    except ImportError:
        if choice == "sage":
            raise RuntimeError(
                f"attention_kernel: sage but SageAttention is not installed; {SAGE_INSTALL_HINT}"
            ) from None
        return AttentionChoice("default", f"SageAttention not installed ({SAGE_INSTALL_HINT})")
    problem = sage_self_test(device)
    if problem is not None:
        if choice == "sage":
            raise RuntimeError(
                f"attention_kernel: sage but SageAttention is unusable: {problem}; {SAGE_INSTALL_HINT}"
            )
        return AttentionChoice("default", f"SageAttention unusable: {problem}")
    _unmasked_kernel = sage_attention
    return AttentionChoice("sage", f"sm_{major}{minor}, self-test passed")


def _sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    if attn_mask is None and _unmasked_kernel is not None:
        return _unmasked_kernel(q, k, v)
    if enable_gqa:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=True
        )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def build_mot_visibility_mask(
    is_gen: Tensor,
    order: Tensor,
    frame: Tensor,
) -> Tensor:
    """Boolean ``[S, S]`` visibility mask (``True`` = query row may attend to key col).

    Args:
        is_gen: ``[S]`` bool — ``True`` for generation (vision) tokens, ``False``
            for understanding (text) tokens.
        order: ``[S]`` long — causal order index for understanding self-attention
            (a text query sees text keys with ``order <= its own``).
        frame: ``[S]`` long — temporal frame index for generation tokens (a gen
            query sees gen keys with ``frame <= its own``); ignored for und keys.

    Returns:
        ``[S, S]`` bool mask.
    """
    q_gen = is_gen[:, None]  # [S,1]
    k_gen = is_gen[None, :]  # [1,S]

    # Understanding query (text): sees understanding keys with order <= own; no gen.
    und_causal = (order[None, :] <= order[:, None]) & (~k_gen)
    und_row = (~q_gen) & und_causal

    # Generation query (vision): sees all und keys (full cross) + gen keys with
    # frame <= own (temporal-causal self).
    gen_sees_und = ~k_gen
    gen_sees_gen = k_gen & (frame[None, :] <= frame[:, None])
    gen_row = q_gen & (gen_sees_und | gen_sees_gen)

    return und_row | gen_row


def mot_joint_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    *,
    enable_gqa: bool = False,
) -> Tensor:
    """Masked scaled-dot-product attention over a packed und+gen sequence.

    Args:
        q: ``[B, Hq, S, d]``.
        k, v: ``[B, Hq, S, d]`` when pre-expanded, or ``[B, Hkv, S, d]`` when
            ``enable_gqa=True``.
        mask: ``[S, S]`` bool visibility mask from :func:`build_mot_visibility_mask`.
        enable_gqa: pass unexpanded grouped K/V into PyTorch SDPA.

    Returns:
        ``[B, H, S, d]`` attention output.
    """
    return _sdpa(q, k, v, attn_mask=mask[None, None], enable_gqa=enable_gqa)


def cached_joint_attention(
    q: Tensor,
    k_all: Tensor,
    v_all: Tensor,
    n_cached: int,
    within_mask: Tensor | None,
    *,
    cached_mask: Tensor | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    """Incremental (KV-cache) attention for the current chunk's query tokens.

    The current chunk attends to **all** cached keys (clean text + past clean
    frames are always visible under the temporal-causal rule, since they precede
    the current frame) plus its own tokens, gated within the current block by
    ``within_mask`` (temporal-causal among current-chunk frames; ``None`` = full).

    Args:
        q: ``[B, Hq, s_cur, d]`` current-chunk queries.
        k_all, v_all: ``[B, Hq, n_cached + s_cur, d]`` cached ++ current K/V
            when pre-expanded, or ``[B, Hkv, n_cached + s_cur, d]`` when
            ``enable_gqa=True``.
        n_cached: number of cached (visible) key positions.
        within_mask: ``[s_cur, s_cur]`` bool over the current block, or ``None``.
        cached_mask: Optional ``[n_cached]`` bool validity mask. This lets a
            filling cache expose a fixed-size, right-padded history without
            allowing attention to read the unwritten slots.
        enable_gqa: pass unexpanded grouped K/V into PyTorch SDPA.

    Returns:
        ``[B, Hq, s_cur, d]`` attention output for the current tokens.
    """
    if cached_mask is not None:
        cached_mask = cached_mask.to(device=q.device, dtype=torch.bool)
        if cached_mask.shape != (n_cached,):
            raise ValueError(
                f"cached_mask must have shape {(n_cached,)}, "
                f"got {tuple(cached_mask.shape)}"
            )
        s_cur = q.shape[-2]
        if within_mask is None:
            current_mask = torch.ones(s_cur, dtype=torch.bool, device=q.device)
            mask = torch.cat((cached_mask, current_mask))[None, None, None]
        else:
            cached = cached_mask.unsqueeze(0).expand(s_cur, -1)
            mask = torch.cat((cached, within_mask), dim=1)[None, None]
        return _sdpa(q, k_all, v_all, attn_mask=mask, enable_gqa=enable_gqa)
    if within_mask is None:
        # All cached keys are visible and the current chunk is a single frame
        # (full self): the mask would be all-True, i.e. no mask. Passing
        # ``attn_mask=None`` lets SDPA dispatch to FlashAttention / cuDNN instead
        # of the much slower dense-mask (math/efficient) kernel — the main
        # per-step win on the len_t=1 hot path.
        return _sdpa(q, k_all, v_all, enable_gqa=enable_gqa)
    s_cur = q.shape[-2]
    s_all = k_all.shape[-2]
    mask = torch.ones((s_cur, s_all), dtype=torch.bool, device=q.device)
    mask[:, n_cached:] = within_mask
    return _sdpa(q, k_all, v_all, attn_mask=mask[None, None], enable_gqa=enable_gqa)


def block_causal_cached_attention(
    q: Tensor, k_all: Tensor, v_all: Tensor, block_tokens: int
) -> Tensor:
    """Unmasked attention for a fully valid cache and a block-causal chunk.

    ``k_all``/``v_all`` are ``[cache | current chunk]`` with native K/V heads.
    Each ``block_tokens`` query block sees the whole cache and the current
    blocks up to its own, so every call is a plain prefix attention: SDPA
    can dispatch to cuDNN/FlashAttention without a mask or varlen packing.
    """
    current = int(q.shape[-2])
    if block_tokens <= 0 or current % block_tokens:
        raise ValueError("block_tokens must be positive and divide the chunk")
    cached = int(k_all.shape[-2]) - current
    gqa = int(q.shape[1]) != int(k_all.shape[1])
    outputs = []
    for start in range(0, current, block_tokens):
        end = cached + start + block_tokens
        outputs.append(
            _sdpa(
                q[..., start : start + block_tokens, :],
                k_all[..., :end, :],
                v_all[..., :end, :],
                enable_gqa=gqa,
            )
        )
    return torch.cat(outputs, dim=2)


def valid_length_cached_attention(
    q: Tensor,
    k_all: Tensor,
    v_all: Tensor,
    valid_cached: Tensor,
    current_tokens: int,
    *,
    causal_block_tokens: int | None = None,
) -> Tensor:
    """FlashAttention over a fixed K/V allocation with a dynamic valid prefix.

    ``k_all``/``v_all`` are laid out as ``[fixed cache | current chunk]``. During
    cache fill, unused cache slots sit between the valid prefix and the current
    chunk. Copy the current K/V into the first unused slots, then tell varlen
    FlashAttention the logical prefix length. Trailing physical slots remain
    allocated for pointer/shape stability but are never visible to attention.

    The cumulative sequence length is a device tensor, so one captured CUDA
    graph can replay while ``valid_cached`` grows from frame zero to saturation.
    """
    if varlen_attn is None:
        raise RuntimeError(
            "valid-length cache attention requires PyTorch varlen FlashAttention"
        )
    if q.shape[0] != 1 or k_all.shape[0] != 1 or v_all.shape[0] != 1:
        raise ValueError("valid-length cache attention currently requires batch=1")
    if valid_cached.numel() != 1:
        raise ValueError("valid_cached must be a scalar tensor")
    current = int(current_tokens)
    if current <= 0 or current > int(k_all.shape[-2]):
        raise ValueError(f"invalid current token count {current}")

    physical_cached = int(k_all.shape[-2]) - current
    valid_cached = valid_cached.to(device=q.device).reshape(())
    destination = valid_cached.to(dtype=torch.long) + torch.arange(
        current, device=q.device
    )
    packed_k = k_all.clone().index_copy_(
        2, destination, k_all[..., physical_cached:, :]
    )
    packed_v = v_all.clone().index_copy_(
        2, destination, v_all[..., physical_cached:, :]
    )
    packed_k = packed_k[0].transpose(0, 1)
    packed_v = packed_v[0].transpose(0, 1)

    def attend(query: Tensor, valid_total: Tensor) -> Tensor:
        query_tokens = int(query.shape[-2])
        cu_seq_q = torch.arange(2, device=q.device, dtype=torch.int32) * query_tokens
        cu_seq_k = torch.stack((torch.zeros_like(valid_total), valid_total))
        out = varlen_attn(
            query[0].transpose(0, 1),
            packed_k,
            packed_v,
            cu_seq_q,
            cu_seq_k,
            query_tokens,
            int(k_all.shape[-2]),
        )
        return out.transpose(0, 1).unsqueeze(0)

    valid_cached = valid_cached.to(dtype=torch.int32)
    if causal_block_tokens is None:
        return attend(q, valid_cached + current)

    block = int(causal_block_tokens)
    if block <= 0 or current % block:
        raise ValueError(
            "causal_block_tokens must be positive and divide current_tokens"
        )
    outputs = []
    for start in range(0, current, block):
        end = start + block
        outputs.append(attend(q[..., start:end, :], valid_cached + end))
    return torch.cat(outputs, dim=2)


def reference_masked_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
    """Explicit-softmax reference for :func:`mot_joint_attention` (test oracle)."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B,H,S,S]
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask[None, None], neg_inf)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
