# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
import torch

from flashdreams.infra.cuda_graph import CUDAGraphWrapper

pytestmark = pytest.mark.ci_cpu


def test_failed_capture_does_not_store_invalid_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "capture_attempts": 0,
        "capture_error_mode": "",
        "in_capture": False,
        "replays": 0,
    }

    class FakeGraph:
        def replay(self) -> None:
            state["replays"] += 1

    class FakeGraphContext:
        def __init__(self, graph: FakeGraph, capture_error_mode: str) -> None:
            self.graph = graph
            self.capture_error_mode = capture_error_mode

        def __enter__(self) -> None:
            del self.graph
            state["capture_error_mode"] = self.capture_error_mode
            state["capture_attempts"] += 1
            state["in_capture"] = True

        def __exit__(self, *exc_info: Any) -> bool:
            state["in_capture"] = False
            return False

    def fake_graph(
        graph: FakeGraph,
        *,
        capture_error_mode: str = "global",
    ) -> FakeGraphContext:
        return FakeGraphContext(graph, capture_error_mode)

    def fn(x: torch.Tensor) -> torch.Tensor:
        if state["in_capture"] and state["capture_attempts"] == 1:
            raise RuntimeError("capture failed")
        return x + 1

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", fake_graph)

    wrapper = CUDAGraphWrapper(fn, warmup_iters=0)

    with pytest.raises(RuntimeError, match="capture failed"):
        wrapper(torch.tensor([1]))

    assert wrapper._graph is None
    assert state["replays"] == 0

    out = wrapper(torch.tensor([2]))

    assert out.item() == 3
    assert isinstance(wrapper._graph, FakeGraph)
    assert state["replays"] == 1
    assert state["capture_error_mode"] == "thread_local"


@pytest.mark.parametrize("pool", [None, ("shared", 7)])
def test_capture_uses_the_shared_pool_only_when_given(
    monkeypatch: pytest.MonkeyPatch, pool: Any
) -> None:
    captured: list[dict[str, Any]] = []

    class FakeGraph:
        def replay(self) -> None:
            pass

    class FakeGraphContext:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *exc_info: Any) -> bool:
            return False

    def fake_graph(graph: FakeGraph, **kwargs: Any) -> FakeGraphContext:
        captured.append(kwargs)
        return FakeGraphContext()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", fake_graph)

    CUDAGraphWrapper(lambda x: x + 1, warmup_iters=0, pool=pool)(torch.tensor([1]))

    expected: dict[str, Any] = {"capture_error_mode": "thread_local"}
    if pool is not None:
        expected["pool"] = pool
    assert captured == [expected]


@pytest.mark.parametrize("clone_outputs", [True, False])
def test_replay_returns_static_outputs_only_without_cloning(
    monkeypatch: pytest.MonkeyPatch, clone_outputs: bool
) -> None:
    class FakeGraph:
        def replay(self) -> None:
            pass

    class FakeGraphContext:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *exc_info: Any) -> bool:
            return False

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **_: FakeGraphContext())
    static = [torch.zeros(2), torch.ones(2)]

    wrapper = CUDAGraphWrapper(
        lambda x: [static[0], static[1]], warmup_iters=0, clone_outputs=clone_outputs
    )
    out = wrapper(torch.tensor([1.0]))

    assert isinstance(out, list) and len(out) == 2
    assert (out[0] is static[0]) is not clone_outputs
    assert torch.equal(out[1], static[1])
