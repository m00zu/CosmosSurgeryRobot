# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choosing the mask-free attention kernel (SDPA or SageAttention)."""

from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from cosmos3_h_dreams._runtime import _attention as attention
from cosmos3_h_dreams.app import Cosmos3HDreamsApplication
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig

pytestmark = pytest.mark.ci_cpu


@pytest.fixture(autouse=True)
def _restore_default_kernel():
    yield
    attention.select_attention_kernel("default", "cpu")


def _fake_gpu(monkeypatch, capability, installed=True, problem=None):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setitem(
        sys.modules, "sageattention", types.ModuleType("sageattention") if installed else None
    )
    monkeypatch.setattr(attention, "sage_self_test", lambda device: problem)


def test_default_and_cpu_use_sdpa():
    assert attention.select_attention_kernel("default", "cpu") == attention.AttentionChoice(
        "default", "set to default"
    )
    assert attention.select_attention_kernel("auto", "cpu").kernel == "default"
    assert attention._unmasked_kernel is None
    with pytest.raises(RuntimeError, match="needs a CUDA device"):
        attention.select_attention_kernel("sage", "cpu")
    with pytest.raises(ValueError, match="attention_kernel must be one of"):
        attention.select_attention_kernel("flash", "cpu")


def test_auto_leaves_hopper_on_sdpa(monkeypatch):
    _fake_gpu(monkeypatch, (9, 0))
    choice = attention.select_attention_kernel("auto", "cuda:0")
    assert choice.kernel == "default" and "sm_90" in choice.reason
    assert attention.select_attention_kernel("sage", "cuda:0").kernel == "sage"


def test_auto_uses_sage_on_rtx50_when_installed_and_healthy(monkeypatch):
    _fake_gpu(monkeypatch, (12, 0))
    choice = attention.select_attention_kernel("auto", "cuda:0")
    assert choice == attention.AttentionChoice("sage", "sm_120, self-test passed")
    assert attention._unmasked_kernel is attention.sage_attention


def test_auto_falls_back_when_missing_or_broken(monkeypatch):
    _fake_gpu(monkeypatch, (12, 0), installed=False)
    choice = attention.select_attention_kernel("auto", "cuda:0")
    assert choice.kernel == "default" and "install_sageattention.sh" in choice.reason
    with pytest.raises(RuntimeError, match="not installed"):
        attention.select_attention_kernel("sage", "cuda:0")

    _fake_gpu(monkeypatch, (12, 0), problem="replay differs")
    choice = attention.select_attention_kernel("auto", "cuda:0")
    assert choice.kernel == "default" and "replay differs" in choice.reason
    assert attention._unmasked_kernel is None
    with pytest.raises(RuntimeError, match="unusable: replay differs"):
        attention.select_attention_kernel("sage", "cuda:0")


def test_only_mask_free_calls_use_the_selected_kernel(monkeypatch):
    calls = []
    monkeypatch.setattr(
        attention, "_unmasked_kernel", lambda q, k, v: calls.append(k.shape) or q
    )
    q = torch.randn(1, 4, 3, 8)
    k = v = torch.randn(1, 2, 5, 8)
    assert attention._sdpa(q, k, v, enable_gqa=True) is q
    mask = torch.ones(3, 5, dtype=torch.bool)[None, None]
    torch.testing.assert_close(
        attention._sdpa(q, k, v, attn_mask=mask, enable_gqa=True),
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
    )
    assert calls == [k.shape]


def test_setting_defaults_to_auto_and_is_validated(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)
    config = Cosmos3HDreamsConfig(checkpoint="/checkpoint", initial_frame=frame, prompt="p")
    assert config.attention_kernel == "auto"
    with pytest.raises(ValueError, match="unknown attention kernel"):
        Cosmos3HDreamsConfig(
            checkpoint="/checkpoint", initial_frame=frame, prompt="p", attention_kernel="x"
        )
    application = Cosmos3HDreamsApplication()
    application.init(
        ["--checkpoint", "/checkpoint", "--initial-frame", str(frame), "--headset",
         "--attention-kernel", "default"]
    )
    assert application.create_session().config.attention_kernel == "default"
