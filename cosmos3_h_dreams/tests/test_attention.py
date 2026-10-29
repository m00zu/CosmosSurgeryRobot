# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU check for the steady-state FadeMem refresh attention."""

import pytest
import torch

from cosmos3_h_dreams._runtime._attention import (
    block_causal_cached_attention,
    cached_joint_attention,
)

pytestmark = pytest.mark.ci_cpu


def test_block_causal_prefixes_match_masked_attention_with_native_gqa():
    generator = torch.Generator().manual_seed(0)
    cached, block, blocks = 7, 3, 2
    q = torch.randn(1, 4, block * blocks, 8, generator=generator)
    k = torch.randn(1, 2, cached + block * blocks, 8, generator=generator)
    v = torch.randn(1, 2, cached + block * blocks, 8, generator=generator)
    frame = torch.arange(blocks).repeat_interleave(block)
    within = frame[None, :] <= frame[:, None]
    expected = cached_joint_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        cached,
        within,
    )
    torch.testing.assert_close(block_causal_cached_attention(q, k, v, block), expected)
    with pytest.raises(ValueError):
        block_causal_cached_attention(q, k, v, 4)
