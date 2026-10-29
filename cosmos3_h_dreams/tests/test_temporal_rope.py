# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FadeMem keys rotated where attention reads them, not in a persistent copy."""

import pytest
import torch
from cosmos3_h_dreams._runtime._fademem import CosmosFadeMem
from cosmos3_h_dreams._runtime._rope import mrope_cos_sin
from cosmos3_h_dreams._runtime._spec import Cosmos3HDreamsSpec, LLMSpec
from cosmos3_h_dreams._runtime._temporal_rope import (
    rotate_half_reference,
    rotate_temporal,
)

_TOKENS = 3
_TEXT = 2
_SPEC = Cosmos3HDreamsSpec(
    llm=LLMSpec(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=96,
        vocab_size=50,
        mrope_section=(4, 2, 2),
    )
)


def _positions(start: float, frames: int) -> torch.Tensor:
    # Fractional temporal ids, like the action tokens' sub-frame times.
    times = start + torch.arange(frames * _TOKENS, dtype=torch.float32) / _TOKENS
    return torch.stack((times, torch.ones_like(times), 2 * torch.ones_like(times)))


def _fademem(capacity: int = 3) -> CosmosFadeMem:
    torch.manual_seed(0)
    cache = [
        {
            "k": torch.randn(1, 2, _TEXT + _TOKENS, 16).to(torch.bfloat16),
            "v": torch.randn(1, 2, _TEXT + _TOKENS, 16).to(torch.bfloat16),
        }
        for _ in range(2)
    ]
    return CosmosFadeMem(
        cache,
        n_text=_TEXT,
        capacity=capacity,
        beta=0.3,
        chunk_frames=1,
        positions=_positions,
        spec=_SPEC,
    )


def _commit(memory: CosmosFadeMem, frame: int) -> None:
    layers = [
        (
            torch.randn(1, 2, _TOKENS, 16).to(torch.bfloat16),
            torch.randn(1, 2, _TOKENS, 16).to(torch.bfloat16),
        )
        for _ in range(2)
    ]
    memory.commit(frame, _positions(frame, 1), layers)


@pytest.mark.ci_cpu
def test_op_matches_the_reference_on_a_strided_store_view() -> None:
    torch.manual_seed(0)
    store = torch.randn(1, 2, 9, 16).to(torch.bfloat16)
    view = store[:, :, :6]
    cos, sin = mrope_cos_sin(_positions(4.0, 2), 16, 1e6, (4, 2, 2))

    out = rotate_temporal(view, cos, sin)

    assert out.is_contiguous()
    assert torch.equal(out, rotate_half_reference(view, cos, sin, 1))


@pytest.mark.ci_cpu
@pytest.mark.parametrize("commits", [0, 1, 4])
def test_fixed_shape_history_rotates_to_the_eager_history(commits: int) -> None:
    memory = _fademem()
    for frame in range(1, commits + 1):
        _commit(memory, frame)
    start = commits + 1

    _, eager = memory.prepare(start, 1)
    _, fixed = memory.prepare(start, 1, fixed_shape=True)

    valid = memory.memory.count * _TOKENS
    assert fixed is memory.stored
    for (stored_k, stored_v), (eager_k, eager_v) in zip(fixed, eager, strict=True):
        rotated = rotate_temporal(stored_k, memory.history_cos, memory.history_sin)
        assert torch.equal(rotated[:, :, :valid], eager_k)
        # Past the valid entries the tables are zero, so padding stays finite.
        assert not rotated[:, :, valid:].any()
        assert torch.equal(stored_v[:, :, :valid], eager_v)


@pytest.mark.ci_cpu
def test_no_rotated_copy_of_the_key_store_is_kept() -> None:
    memory = _fademem(capacity=3)

    tables = 2 * 3 * _TOKENS * 16 * 4
    seed_and_text = 2 * 2 * (_TOKENS + _TEXT) * 2 * 16 * 2
    mask_and_length = memory.mask.numel() + 4
    assert memory.storage_bytes == (
        memory.memory.storage_bytes + tables + seed_and_text + mask_and_length
    )
    assert not hasattr(memory, "prepared")


@pytest.mark.ci_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for Triton")
def test_triton_rotation_is_bit_identical_for_every_bf16_key() -> None:
    values = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    keys = values.view(torch.bfloat16).cuda()
    keys = keys[torch.isfinite(keys.float())]
    keys = keys[: keys.numel() // 128 * 128].view(1, 1, -1, 128)
    torch.manual_seed(0)
    positions = torch.zeros(3, keys.shape[2])
    positions[0] = torch.rand(keys.shape[2]) * 3000
    cos, sin = mrope_cos_sin(positions.cuda(), 128, 5e6, (24, 20, 20))

    out = rotate_temporal(keys, cos, sin)

    expected = rotate_half_reference(keys, cos, sin, 1)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
