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

"""Manual GPU check: model mode switches return CUDA memory to baseline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.manual


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mode_switches_release_cuda_memory():
    from cosmos3_h_dreams.app import Cosmos3HDreamsApplication
    from cosmos3_h_dreams.headset_server import release_cuda_memory
    from cosmos3_h_dreams.session import Cosmos3HDreamsSession
    from dataclasses import replace

    checkpoint = os.environ.get("COSMOS3_H_DREAMS_CHECKPOINT", "data_local/checkpoints/c3-h-d_2-step_model_bf16.safetensors")
    if not Path(checkpoint).exists():
        pytest.skip("checkpoint not available")
    application = Cosmos3HDreamsApplication()
    application.init([
        "--config", "wan-taehv-chunk2-2steps-fademem-compile",
        "--checkpoint", checkpoint,
        "--initial-frame", "assets/example_data/scenes/example_frame.png",
        "--actions", "assets/example_data/scenes/trajectory.npy",
        "--offload-understanding",
    ])
    config = application.create_session().config
    baseline = release_cuda_memory()
    peaks = []
    for fp8 in (False, True, False, True, False):
        session = Cosmos3HDreamsSession(config=replace(config, fp8_generation=fp8))
        session.init()
        peaks.append(torch.cuda.memory_allocated())
        session.close()
        del session
        remaining = release_cuda_memory()
        print(f"fp8={fp8}: loaded {peaks[-1] / 2**30:.2f} GiB, after release {remaining / 2**30:.2f} GiB (baseline {baseline / 2**30:.2f} GiB)")
        assert remaining - baseline < 2**30, f"{(remaining - baseline) / 2**30:.2f} GiB left after fp8={fp8}"
    assert peaks[1] < peaks[0], "FP8 mode should hold less memory than standard"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mode_switch_reloads_with_generation_do_not_grow_cuda_memory():
    """Reload like ``HeadsetServer.switch_mode`` and generate between reloads.

    The server runs ``init``, ``step`` and ``close`` through ``asyncio.to_thread``,
    so steps hop between default-executor threads. Each thread has its own cuBLAS
    handle, each session creates its own cache stream, and PyTorch keeps one
    cuBLAS workspace (32 MiB + 1 MiB on Hopper) per (handle, stream) pair for the
    life of the process. Without ``release_cuda_memory`` clearing them, steady
    memory grew by about 33 MiB per thread per reload.
    """
    import concurrent.futures
    from dataclasses import replace

    from cosmos3_h_dreams.app import Cosmos3HDreamsApplication
    from cosmos3_h_dreams.headset_server import release_cuda_memory
    from cosmos3_h_dreams.session import Cosmos3HDreamsSession
    from flashdreams.runtime.inputs import CanonicalInputWindow, TimeWindow

    checkpoint = os.environ.get("COSMOS3_H_DREAMS_CHECKPOINT", "data_local/checkpoints/c3-h-d_2-step_model_bf16.safetensors")
    if not Path(checkpoint).exists():
        pytest.skip("checkpoint not available")
    application = Cosmos3HDreamsApplication()
    application.init([
        "--config", "wan-taehv-chunk2-2steps-fademem-compile",
        "--checkpoint", checkpoint,
        "--initial-frame", "assets/example_data/scenes/example_frame.png",
        "--actions", "assets/example_data/scenes/trajectory.npy",
        "--offload-understanding",
    ])
    config = application.create_session().config
    release_cuda_memory()
    # Long-lived workers used in turn, as idle to_thread workers are in the server.
    workers = [concurrent.futures.ThreadPoolExecutor(1) for _ in range(8)]
    calls = 0

    def run(fn, *args):
        nonlocal calls
        calls += 1
        return workers[calls % len(workers)].submit(fn, *args).result()

    steady: dict[bool, list[float]] = {False: [], True: []}
    try:
        for reload in range(8):
            fp8 = reload % 2 == 1
            session = Cosmos3HDreamsSession(config=replace(config, fp8_generation=fp8))
            run(session.init)
            # Past FadeMem fill (12 slots, 2 frames per step).
            for _ in range(10):
                window = CanonicalInputWindow(window=TimeWindow(start_s=0.0, end_s=1.0))
                run(session.step, window)
            torch.cuda.synchronize()
            steady[fp8].append(torch.cuda.memory_allocated() / 2**20)
            run(session.close)
            del session
            remaining = release_cuda_memory()
            print(f"reload {reload} fp8={fp8}: steady {steady[fp8][-1]:.0f} MiB, after release {remaining / 2**20:.0f} MiB")
    finally:
        for worker in workers:
            worker.shutdown()
    for fp8, levels in steady.items():
        # Same-mode visits are two reloads apart.
        slope = (levels[-1] - levels[0]) / (2 * (len(levels) - 1))
        print(f"fp8={fp8}: steady {[round(level) for level in levels]} MiB, slope {slope:.1f} MiB/reload")
        assert slope < 32, f"steady memory grows {slope:.1f} MiB per reload (fp8={fp8}): {levels}"
