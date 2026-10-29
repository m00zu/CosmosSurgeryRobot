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

"""Manual GPU check: the GPU monitor's NVML reader on a real GPU."""

from __future__ import annotations

import pytest
import torch

from cosmos3_h_dreams import gpu_monitor

pytestmark = pytest.mark.manual


def test_default_reader_samples_every_gpu_without_cuda_context():
    if gpu_monitor.pynvml is None:
        pytest.skip("nvidia-ml-py is not installed")
    assert not torch.cuda.is_initialized()
    monitor = gpu_monitor.GpuMonitor(device=0)
    try:
        if not monitor.nvml:
            pytest.skip("NVML is not available")
        samples = monitor.sample_once()
        assert monitor.gpus
        for gpu in monitor.gpus:
            rows = [s for s in samples if s.gpu == gpu["index"]]
            assert rows, f"no sample for GPU {gpu['index']}"
            for s in rows:
                assert 0 < s.used_mib <= s.total_mib
        assert not torch.cuda.is_initialized()
    finally:
        monitor.stop()
    assert not torch.cuda.is_initialized()
