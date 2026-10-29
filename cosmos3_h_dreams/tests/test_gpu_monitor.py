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

"""GPU memory sampler: ring buffer, marks, window stats, CSV log and readers."""

from __future__ import annotations

import csv
import logging
import sys
import threading
import time
from types import SimpleNamespace

import pynvml
import pytest
import torch

from cosmos3_h_dreams import gpu_monitor
from cosmos3_h_dreams.gpu_monitor import GpuMonitor, GpuSample

pytestmark = pytest.mark.ci_cpu

HEADER = [
    "time",
    "gpu",
    "used_mib",
    "total_mib",
    "util_pct",
    "temp_c",
    "torch_allocated_mib",
    "torch_reserved_mib",
    "host_pss_mib",
]

REAL_HOST_PSS = gpu_monitor._host_pss_mib


@pytest.fixture(autouse=True)
def _no_host_pss(monkeypatch):
    """The default reader reads this process tree's PSS; keep exact-sample tests deterministic."""
    monkeypatch.setattr(gpu_monitor, "_host_pss_mib", lambda: None)


def _sample(
    t,
    gpu=0,
    used=1000.0,
    total=81559.0,
    util=5.0,
    temp=40.0,
    alloc=None,
    reserved=None,
    host=None,
):
    return GpuSample(t, gpu, used, total, util, temp, alloc, reserved, host)


class ScriptedReader:
    """Return the next scripted list of samples on every call."""

    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = 0

    def __call__(self):
        index = min(self.calls, len(self.rounds) - 1)
        self.calls += 1
        return list(self.rounds[index])


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        self.now += 1.0
        return self.now


def _monitor(rounds, **kwargs):
    reader = ScriptedReader(rounds)
    return GpuMonitor(reader=reader, **kwargs), reader


def test_sample_to_json_has_every_field():
    sample = _sample(
        12.5, gpu=1, used=10.0, total=20.0, util=None, alloc=3.0, reserved=4.0, host=900.0
    )
    assert sample.to_json() == {
        "t": 12.5,
        "gpu": 1,
        "used_mib": 10.0,
        "total_mib": 20.0,
        "util_pct": None,
        "temp_c": 40.0,
        "torch_allocated_mib": 3.0,
        "torch_reserved_mib": 4.0,
        "host_pss_mib": 900.0,
    }


def test_ring_buffer_keeps_last_history_samples_per_gpu():
    rounds = [
        [_sample(float(i), 0, used=100.0 + i), _sample(float(i), 1, used=500.0 + i)]
        for i in range(5)
    ]
    monitor, _ = _monitor(rounds, history=3)
    for _ in rounds:
        monitor.sample_once()
    assert [s.used_mib for s in monitor.samples(gpu=0)] == [102.0, 103.0, 104.0]
    assert [s.used_mib for s in monitor.samples(gpu=1)] == [502.0, 503.0, 504.0]
    assert len(monitor.samples()) == 6


def test_sample_once_returns_what_it_read():
    first = [_sample(1.0), _sample(1.0, gpu=1)]
    monitor, _ = _monitor([first])
    assert monitor.sample_once() == first


def test_samples_filter_by_since_and_gpu():
    rounds = [[_sample(float(i), 0), _sample(float(i), 1)] for i in range(1, 5)]
    monitor, _ = _monitor(rounds)
    for _ in rounds:
        monitor.sample_once()
    newer = monitor.samples(since=2.0)
    assert sorted({s.t for s in newer}) == [3.0, 4.0]  # strictly newer than `since`
    assert {s.gpu for s in newer} == {0, 1}
    only_one = monitor.samples(since=2.0, gpu=1)
    assert [(s.t, s.gpu) for s in only_one] == [(3.0, 1), (4.0, 1)]
    assert monitor.samples(since=4.0) == []
    assert [s.t for s in monitor.samples(gpu=0)] == [1.0, 2.0, 3.0, 4.0]


def test_latest_returns_the_newest_sample_of_that_gpu_or_none():
    rounds = [
        [_sample(1.0, 0, used=100.0), _sample(1.0, 1, used=500.0)],
        [_sample(2.0, 0, used=200.0)],
    ]
    monitor, _ = _monitor(rounds)
    assert monitor.latest(0) is None
    for _ in rounds:
        monitor.sample_once()
    assert monitor.latest(0) == _sample(2.0, 0, used=200.0)
    assert monitor.latest(1) == _sample(1.0, 1, used=500.0)
    assert monitor.latest(7) is None


def test_baseline_is_first_sample_and_peak_is_max_per_gpu():
    used = [(100.0, 700.0), (900.0, 650.0), (300.0, 800.0)]
    rounds = [
        [_sample(float(i), 0, used=a), _sample(float(i), 1, used=b)]
        for i, (a, b) in enumerate(used)
    ]
    monitor, _ = _monitor(rounds)
    assert monitor.baseline() == {} and monitor.peak() == {}
    for _ in rounds:
        monitor.sample_once()
    assert monitor.baseline() == {0: 100.0, 1: 700.0}
    assert monitor.peak() == {0: 900.0, 1: 800.0}


def test_peak_survives_the_ring_buffer_dropping_the_sample():
    rounds = [
        [_sample(float(i), 0, used=u)] for i, u in enumerate([50.0, 900.0, 60.0, 70.0])
    ]
    monitor, _ = _monitor(rounds, history=2)
    for _ in rounds:
        monitor.sample_once()
    assert [s.used_mib for s in monitor.samples()] == [60.0, 70.0]
    assert monitor.peak() == {0: 900.0}
    assert monitor.baseline() == {0: 50.0}


def test_stats_cover_the_device_only_over_the_inclusive_window():
    rows = [
        # t, device used, device allocated, other-gpu used
        (1.0, 100.0, 10.0, 9000.0),
        (2.0, 200.0, None, 9000.0),
        (3.0, 400.0, 30.0, 9000.0),
        (4.0, 800.0, 50.0, 9000.0),
    ]
    rounds = [
        [_sample(t, 0, used=other), _sample(t, 1, used=used, alloc=alloc)]
        for t, used, alloc, other in rows
    ]
    monitor, _ = _monitor(rounds, device=1)
    for _ in rounds:
        monitor.sample_once()
    assert monitor.stats(2.0, 4.0) == {
        "peak_used_mib": 800.0,
        "mean_used_mib": pytest.approx((200.0 + 400.0 + 800.0) / 3),
        "peak_allocated_mib": 50.0,
        "mean_allocated_mib": pytest.approx((30.0 + 50.0) / 2),  # the None is ignored
        "peak_host_pss_mib": None,
        "mean_host_pss_mib": None,
    }
    assert monitor.stats(1.0, 1.0)["peak_used_mib"] == 100.0


def test_stats_are_none_without_samples_in_the_window():
    monitor, _ = _monitor([[_sample(5.0, 0, used=100.0)]])
    none_stats = {
        "peak_used_mib": None,
        "mean_used_mib": None,
        "peak_allocated_mib": None,
        "mean_allocated_mib": None,
        "peak_host_pss_mib": None,
        "mean_host_pss_mib": None,
    }
    assert monitor.stats(0.0, 10.0) == none_stats  # nothing sampled yet
    monitor.sample_once()
    assert monitor.stats(6.0, 10.0) == none_stats
    only_used = monitor.stats(0.0, 10.0)  # samples exist but carry no torch numbers
    assert only_used["peak_used_mib"] == 100.0
    assert only_used["peak_allocated_mib"] is None
    assert only_used["mean_allocated_mib"] is None


def test_marks_are_ordered_and_filter_by_since():
    clock = FakeClock(now=100.0)
    monitor, _ = _monitor([[_sample(1.0)]], clock=clock)
    monitor.mark("mode_switch_start", "low")
    monitor.mark("mode_switch_end", "low")
    monitor.mark("incident")
    assert monitor.marks() == [
        {"t": 101.0, "kind": "mode_switch_start", "label": "low"},
        {"t": 102.0, "kind": "mode_switch_end", "label": "low"},
        {"t": 103.0, "kind": "incident", "label": ""},
    ]
    assert [m["kind"] for m in monitor.marks(since=101.0)] == [
        "mode_switch_end",
        "incident",
    ]
    assert monitor.marks(since=103.0) == []


def test_mark_takes_an_explicit_time():
    clock = FakeClock(now=100.0)
    monitor, _ = _monitor([[_sample(1.0)]], clock=clock)
    monitor.mark("attempt_end", "a1", t=55.5)
    monitor.mark("incident")
    assert monitor.marks() == [
        {"t": 55.5, "kind": "attempt_end", "label": "a1"},
        {"t": 101.0, "kind": "incident", "label": ""},
    ]


def test_marks_keep_the_last_500():
    monitor, _ = _monitor([[_sample(1.0)]], clock=FakeClock())
    for i in range(520):
        monitor.mark("attempt_start", str(i))
    marks = monitor.marks()
    assert len(marks) == 500
    assert marks[0]["label"] == "20" and marks[-1]["label"] == "519"


def test_csv_has_header_and_one_row_per_sample(tmp_path):
    path = tmp_path / "gpu-test.csv"
    rounds = [
        [
            _sample(1.5, 0, used=100.0, util=7.0, temp=41.0, alloc=60.0, reserved=80.0),
            _sample(1.5, 1, used=200.0, util=None),
        ],
        [
            _sample(2.5, 0, used=110.0, alloc=61.0, reserved=81.0),
            _sample(2.5, 1, used=210.0),
        ],
    ]
    monitor, _ = _monitor(rounds, csv_path=path)
    assert monitor.csv_name == "gpu-test.csv"
    assert not path.exists()  # opened lazily on the first sample
    monitor.sample_once()
    # flushed as written: readable before stop()
    with path.open(newline="") as handle:
        assert len(list(csv.reader(handle))) == 3
    monitor.sample_once()
    monitor.stop()
    with path.open(newline="") as handle:
        table = list(csv.reader(handle))
    assert table[0] == HEADER
    assert len(table) == 5
    first, second = table[1], table[2]
    assert first == ["1.500", "0", "100.0", "81559.0", "7.0", "41.0", "60.0", "80.0", ""]
    # no torch numbers on the other GPU, and a missing util stays empty
    assert second[:2] == ["1.500", "1"] and second[4] == "" and second[6:] == ["", "", ""]
    assert monitor.csv_error is None


def test_csv_in_an_unwritable_directory_sets_csv_error_and_sampling_continues(tmp_path):
    path = tmp_path / "missing" / "gpu.csv"
    monitor, _ = _monitor([[_sample(1.0)], [_sample(2.0)]], csv_path=path)
    assert monitor.csv_error is None
    assert len(monitor.sample_once()) == 1
    assert monitor.csv_error
    assert len(monitor.sample_once()) == 1
    assert [s.t for s in monitor.samples()] == [1.0, 2.0]
    assert not path.parent.exists()  # the monitor never creates the directory
    monitor.stop()


@pytest.mark.parametrize(
    "error",
    [OSError("disk full"), ValueError("write to closed file"), csv.Error("bad row")],
    ids=["oserror", "valueerror", "csv-error"],
)
def test_any_csv_write_failure_disables_the_csv_logs_once_and_sampling_continues(
    tmp_path, caplog, error
):
    path = tmp_path / "gpu.csv"
    monitor, _ = _monitor(
        [[_sample(1.0)], [_sample(2.0)], [_sample(3.0)]], csv_path=path
    )
    monitor.sample_once()

    class Broken:
        def write(self, _):
            raise error

        def flush(self):
            pass

        def close(self):
            pass

    monitor._csv_file = Broken()
    monitor._csv_writer = csv.writer(monitor._csv_file)
    with caplog.at_level(logging.WARNING, logger=gpu_monitor.__name__):
        monitor.sample_once()
        monitor.sample_once()
    assert monitor.csv_error == str(error)
    assert len(monitor.samples()) == 3
    disabled = [r for r in caplog.records if "CSV log disabled" in r.getMessage()]
    assert len(disabled) == 1 and str(error) in disabled[0].getMessage()
    monitor.stop()


def test_a_late_write_after_stop_does_not_reopen_the_csv(tmp_path):
    path = tmp_path / "gpu.csv"
    monitor, _ = _monitor([[_sample(1.0)]], csv_path=path)
    monitor.stop()
    monitor.sample_once()
    assert not path.exists()


def test_a_write_already_past_the_first_check_rechecks_under_the_lock(tmp_path):
    path = tmp_path / "gpu.csv"
    monitor, _ = _monitor([[_sample(1.0)]], csv_path=path)
    writer = threading.Thread(target=monitor._write_csv, args=([_sample(1.0)],))
    with monitor._csv_lock:  # stop() is mid-way: it holds the lock
        writer.start()
        time.sleep(0.05)  # the writer passed the unlocked check and now waits
        monitor._csv_on = False
    writer.join(timeout=5.0)
    assert not writer.is_alive()
    assert monitor._csv_file is None and not path.exists()


def test_csv_uses_unix_line_endings(tmp_path):
    path = tmp_path / "gpu.csv"
    monitor, _ = _monitor([[_sample(1.0)]], csv_path=path)
    monitor.sample_once()
    monitor.stop()
    raw = path.read_bytes()
    assert raw.startswith(b"time,gpu,used_mib,")
    assert b"\r" not in raw and raw.endswith(b"\n")


def test_without_a_csv_path_csv_name_is_none():
    monitor, _ = _monitor([[_sample(1.0)]])
    assert monitor.csv_name is None
    monitor.sample_once()
    assert monitor.csv_error is None


def test_injected_reader_reports_nvml_false_and_learns_gpus():
    monitor, _ = _monitor(
        [[_sample(1.0, 0, total=100.0), _sample(1.0, 1, total=200.0)]]
    )
    assert monitor.nvml is False
    monitor.sample_once()
    assert [(g["index"], g["total_mib"]) for g in monitor.gpus] == [
        (0, 100.0),
        (1, 200.0),
    ]
    assert all(g["name"] for g in monitor.gpus)


def test_thread_samples_and_start_stop_are_idempotent(tmp_path):
    path = tmp_path / "gpu.csv"
    monitor, reader = _monitor([[_sample(time.time())]], period_s=0.01, csv_path=path)
    monitor.start()
    thread = monitor._thread
    monitor.start()  # idempotent
    assert monitor._thread is thread
    deadline = time.monotonic() + 5.0
    while len(monitor.samples()) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(monitor.samples()) >= 3
    monitor.stop()
    assert not thread.is_alive()
    monitor.stop()  # idempotent
    calls = reader.calls
    time.sleep(0.05)
    assert reader.calls == calls  # really stopped
    with path.open(newline="") as handle:
        assert len(list(csv.reader(handle))) == calls + 1


def test_thread_survives_a_failing_reader(caplog):
    class Flaky:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("driver hiccup")
            return [_sample(float(self.calls))]

    reader = Flaky()
    monitor = GpuMonitor(reader=reader, period_s=0.01)
    with caplog.at_level(logging.WARNING, logger=gpu_monitor.__name__):
        monitor.start()
        deadline = time.monotonic() + 5.0
        while len(monitor.samples()) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        monitor.stop()
    assert len(monitor.samples()) >= 2
    assert (
        len([r for r in caplog.records if "driver hiccup" in r.getMessage()]) == 1
    )  # logged once


def test_stop_without_start_is_safe(tmp_path):
    monitor, _ = _monitor([[_sample(1.0)]], csv_path=tmp_path / "gpu.csv")
    monitor.stop()
    monitor.stop()


class FakeNvml:
    """Patch pynvml with two scripted GPUs; GPU 1 can be made to fail."""

    def __init__(self, monkeypatch, *, fail_memory_on=()):
        self.fail_memory_on = set(fail_memory_on)
        self.shutdowns = 0
        self.memory_calls = 0
        for name, value in {
            "nvmlInit": lambda: None,
            "nvmlShutdown": self._shutdown,
            "nvmlDeviceGetCount": lambda: 2,
            "nvmlDeviceGetHandleByIndex": lambda i: i,
            "nvmlDeviceGetName": lambda h: f"Fake H100 {h}",
            "nvmlDeviceGetMemoryInfo": self._memory,
            "nvmlDeviceGetUtilizationRates": lambda h: SimpleNamespace(gpu=10 + h),
            "nvmlDeviceGetTemperature": lambda h, kind: 50 + h,
        }.items():
            monkeypatch.setattr(pynvml, name, value)

    def _shutdown(self):
        self.shutdowns += 1

    def _memory(self, handle):
        self.memory_calls += 1
        if handle in self.fail_memory_on:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        return SimpleNamespace(used=(handle + 1) * 2**30, total=80 * 2**30)


@pytest.fixture()
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


def test_default_reader_reads_every_gpu_through_nvml(monkeypatch, no_cuda):
    fake = FakeNvml(monkeypatch)
    monitor = GpuMonitor(device=1, clock=lambda: 42.0)
    assert monitor.nvml is True
    assert monitor.gpus == [
        {"index": 0, "name": "Fake H100 0", "total_mib": 81920.0},
        {"index": 1, "name": "Fake H100 1", "total_mib": 81920.0},
    ]
    first, second = monitor.sample_once()
    assert first == GpuSample(42.0, 0, 1024.0, 81920.0, 10.0, 50.0, None, None)
    assert second == GpuSample(
        42.0, 1, 2048.0, 81920.0, 11.0, 51.0, None, None
    )  # CUDA not initialised
    monitor.stop()
    monitor.stop()
    assert fake.shutdowns == 1


def test_default_reader_adds_torch_numbers_for_the_device_only_when_cuda_is_initialised(
    monkeypatch,
):
    FakeNvml(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "memory_allocated", lambda d: 3 * 2**30 if d == 1 else 1 / 0
    )
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda d: 5 * 2**30 if d == 1 else 1 / 0
    )
    monitor = GpuMonitor(device=1)
    other, device = monitor.sample_once()
    assert (other.torch_allocated_mib, other.torch_reserved_mib) == (None, None)
    assert (device.torch_allocated_mib, device.torch_reserved_mib) == (3072.0, 5120.0)
    assert device.used_mib == 2048.0
    monitor.stop()


def test_torch_counters_are_read_with_torch_device_and_attached_to_the_nvml_device(
    monkeypatch,
):
    """CUDA_VISIBLE_DEVICES=1: NVML index 1 is torch index 0."""
    FakeNvml(monkeypatch)
    read_with = []

    def counter(value):
        def read(index):
            read_with.append(index)
            return value * 2**30

        return read

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", counter(3))
    monkeypatch.setattr(torch.cuda, "memory_reserved", counter(5))
    monitor = GpuMonitor(device=1, torch_device=0)
    assert (monitor.device, monitor.torch_device) == (1, 0)
    other, device = monitor.sample_once()
    assert (other.gpu, other.torch_allocated_mib, other.torch_reserved_mib) == (
        0,
        None,
        None,
    )
    assert (device.gpu, device.torch_allocated_mib, device.torch_reserved_mib) == (
        1,
        3072.0,
        5120.0,
    )
    assert set(read_with) == {0}
    monitor.stop()


def test_torch_device_defaults_to_device():
    assert GpuMonitor(device=1, reader=list).torch_device == 1


def test_one_bad_gpu_is_skipped_and_logged_once(monkeypatch, no_cuda, caplog):
    FakeNvml(monkeypatch, fail_memory_on=[1])
    with caplog.at_level(logging.WARNING, logger=gpu_monitor.__name__):
        monitor = GpuMonitor(device=0)
        for _ in range(3):
            samples = monitor.sample_once()
            assert [s.gpu for s in samples] == [0]
    assert len([r for r in caplog.records if "GPU 1" in r.getMessage()]) == 1
    assert [s.gpu for s in monitor.samples()] == [0, 0, 0]
    monitor.stop()


def test_util_and_temperature_failures_become_none(monkeypatch, no_cuda):
    FakeNvml(monkeypatch)

    def boom(*_):
        raise pynvml.NVMLError(pynvml.NVML_ERROR_NOT_SUPPORTED)

    monkeypatch.setattr(pynvml, "nvmlDeviceGetUtilizationRates", boom)
    monkeypatch.setattr(pynvml, "nvmlDeviceGetTemperature", boom)
    monitor = GpuMonitor()
    sample = monitor.sample_once()[0]
    assert (sample.util_pct, sample.temp_c) == (None, None)
    assert sample.used_mib == 1024.0
    monitor.stop()


def test_nvml_init_failure_falls_back_without_raising(monkeypatch, no_cuda):
    def boom():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

    monkeypatch.setattr(pynvml, "nvmlInit", boom)
    monitor = GpuMonitor()
    assert monitor.nvml is False
    assert monitor.gpus == []
    assert (
        monitor.sample_once() == []
    )  # no NVML and CUDA not initialised: nothing to report
    monitor.stop()


def test_missing_pynvml_falls_back_without_raising(monkeypatch, no_cuda):
    monkeypatch.setattr(gpu_monitor, "pynvml", None)
    monitor = GpuMonitor()
    assert monitor.nvml is False
    assert monitor.sample_once() == []
    monitor.stop()


def test_torch_only_fallback_reports_reserved_as_used(monkeypatch):
    monkeypatch.setattr(
        pynvml,
        "nvmlInit",
        lambda: (_ for _ in ()).throw(pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 4 * 2**30)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(total_memory=80 * 2**30, name="Fake H100"),
    )
    monitor = GpuMonitor(device=0, clock=lambda: 7.0)
    assert monitor.nvml is False
    assert monitor.sample_once() == [
        GpuSample(7.0, 0, 4096.0, 81920.0, None, None, 2048.0, 4096.0)
    ]
    assert monitor.gpus == [{"index": 0, "name": "GPU 0", "total_mib": 81920.0}]
    monitor.stop()


def test_torch_only_fallback_reads_torch_device_and_reports_the_device_index(
    monkeypatch,
):
    monkeypatch.setattr(
        pynvml,
        "nvmlInit",
        lambda: (_ for _ in ()).throw(pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)),
    )
    read_with = []

    def memory(value):
        def read(index):
            read_with.append(index)
            return value * 2**30

        return read

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", memory(2))
    monkeypatch.setattr(torch.cuda, "memory_reserved", memory(4))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: read_with.append(d) or SimpleNamespace(total_memory=80 * 2**30),
    )
    monitor = GpuMonitor(device=1, torch_device=0, clock=lambda: 7.0)
    assert monitor.sample_once() == [
        GpuSample(7.0, 1, 4096.0, 81920.0, None, None, 2048.0, 4096.0)
    ]
    assert set(read_with) == {0}
    monitor.stop()


# -- host RAM ---------------------------------------------------------------------------


def test_host_pss_is_recorded_once_per_round_on_the_device_and_reaches_the_csv(tmp_path):
    path = tmp_path / "gpu.csv"
    rounds = [
        [_sample(1.0, 0, used=50.0), _sample(1.0, 1, used=100.0, host=2048.0)],
        [_sample(2.0, 0, used=50.0), _sample(2.0, 1, used=100.0, host=4096.0)],
        [_sample(3.0, 0, used=50.0), _sample(3.0, 1, used=100.0, host=3072.0)],
    ]
    monitor, _ = _monitor(rounds, device=1, csv_path=path)
    assert monitor.host() == {"pss_mib": None, "peak_mib": None}
    for _ in rounds:
        monitor.sample_once()
    monitor.stop()

    assert monitor.host() == {"pss_mib": 3072.0, "peak_mib": 4096.0}
    stats = monitor.stats(2.0, 3.0)
    assert stats["peak_host_pss_mib"] == 4096.0
    assert stats["mean_host_pss_mib"] == pytest.approx(3584.0)
    with path.open(newline="") as handle:
        table = list(csv.reader(handle))
    assert table[0][-1] == "host_pss_mib"
    assert [row[-1] for row in table[1:]] == ["", "2048.0", "", "4096.0", "", "3072.0"]


def test_host_peak_survives_the_ring_buffer():
    rounds = [[_sample(float(i), 0, host=h)] for i, h in enumerate([10.0, 900.0, 20.0, 30.0])]
    monitor, _ = _monitor(rounds, history=2)
    for _ in rounds:
        monitor.sample_once()
    assert monitor.host() == {"pss_mib": 30.0, "peak_mib": 900.0}


def test_default_reader_attaches_host_pss_to_the_device_sample_only(monkeypatch, no_cuda):
    FakeNvml(monkeypatch)
    monitor = GpuMonitor(device=1, clock=lambda: 42.0, host_reader=lambda: 1234.5)
    monitor.read_host()
    other, device = monitor.sample_once()
    assert other.host_pss_mib is None
    assert device == GpuSample(42.0, 1, 2048.0, 81920.0, 11.0, 51.0, None, None, 1234.5)
    monitor.stop()


def test_samples_carry_the_latest_host_snapshot_without_reading_it(monkeypatch, no_cuda):
    FakeNvml(monkeypatch)
    reads = []
    monitor = GpuMonitor(device=0, host_reader=lambda: reads.append(1) or 100.0 * len(reads))
    assert monitor.sample_once()[0].host_pss_mib is None  # no snapshot yet
    monitor.read_host()
    assert [monitor.sample_once()[0].host_pss_mib for _ in range(3)] == [100.0] * 3
    monitor.read_host()
    assert monitor.sample_once()[0].host_pss_mib == 200.0
    assert len(reads) == 2  # the GPU sampler never reads host memory itself
    monitor.stop()


def test_host_thread_takes_snapshots_until_the_monitor_stops(monkeypatch, no_cuda):
    FakeNvml(monkeypatch)
    reads = []
    monitor = GpuMonitor(
        device=0, period_s=60.0, host_period_s=0.01, host_reader=lambda: reads.append(1) or 5.0
    )
    monitor.start()
    deadline = time.time() + 5
    while len(reads) < 3 and time.time() < deadline:
        time.sleep(0.01)
    monitor.stop()
    stopped_at = len(reads)
    time.sleep(0.1)
    assert stopped_at >= 3
    assert len(reads) == stopped_at
    assert not [t for t in threading.enumerate() if t.name.startswith("gpu-monitor")]


def test_torch_only_fallback_attaches_host_pss(monkeypatch):
    monkeypatch.setattr(
        pynvml,
        "nvmlInit",
        lambda: (_ for _ in ()).throw(pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 2 * 2**30)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 4 * 2**30)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda d: SimpleNamespace(total_memory=80 * 2**30),
    )
    monitor = GpuMonitor(device=0, clock=lambda: 7.0, host_reader=lambda: 512.0)
    monitor.read_host()
    (sample,) = monitor.sample_once()
    assert sample.host_pss_mib == 512.0
    monitor.stop()


def test_a_failing_host_reader_gives_none_and_logs_once(monkeypatch, no_cuda, caplog):
    FakeNvml(monkeypatch)

    def broken():
        raise PermissionError("no /proc")

    with caplog.at_level(logging.WARNING, logger=gpu_monitor.__name__):
        monitor = GpuMonitor(device=0, host_reader=broken)
        for _ in range(3):
            monitor.read_host()
            assert monitor.sample_once()[0].host_pss_mib is None
    assert len([r for r in caplog.records if "host" in r.getMessage()]) == 1
    monitor.stop()


def test_host_pss_reads_this_process_and_its_children():
    import subprocess

    alone = REAL_HOST_PSS()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; b = bytearray(64 << 20); time.sleep(30)"],
        stdout=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 10
        together = REAL_HOST_PSS()
        while together < alone + 48 and time.time() < deadline:
            time.sleep(0.05)
            together = REAL_HOST_PSS()
    finally:
        child.kill()
        child.wait()
    assert alone > 1.0  # MiB; any Python process is larger
    assert together >= alone + 48  # the child's 64 MiB buffer counts


linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="PSS and fork are Linux-only"
)
# The forked children only pause or allocate and then _exit, so they take no locks.
fork_ok = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


@linux_only
@fork_ok
def test_host_pss_snapshot_has_no_overshoot_right_after_a_fork_burst():
    import os
    import signal

    block = bytearray(b"\x01") * (256 << 20)  # 256 MiB of touched pages, before the fork
    before = REAL_HOST_PSS()
    children = []
    try:
        for _ in range(16):
            pid = os.fork()
            if pid == 0:  # child: share the parent's pages, touch nothing, wait to be killed
                signal.pause()
                os._exit(0)
            children.append(pid)
        readings = [REAL_HOST_PSS() for _ in range(5)]  # straight after the burst
    finally:
        for pid in children:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    del block
    assert before > 256  # the block is in the parent's footprint
    # Every snapshot is coherent: the shared block is split, never counted twice.
    assert max(readings) - before < 64, readings


@linux_only
@fork_ok
def test_host_pss_counts_a_zombie_child_as_zero():
    import os

    me = os.getpid()
    ready_r, ready_w = os.pipe()
    go_r, go_w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: 256 MiB of private memory, then exit when told
        private = bytearray(b"\x02") * (256 << 20)
        os.write(ready_w, b"r")
        os.read(go_r, 1)
        del private
        os._exit(0)
    try:
        os.read(ready_r, 1)
        alive = REAL_HOST_PSS()
        alone_alive = gpu_monitor._proc_pss(me) / 2**20
        os.write(go_w, b"g")
        deadline = time.time() + 10
        while time.time() < deadline:  # exited but not reaped: a zombie
            with open(f"/proc/{pid}/stat") as handle:
                if handle.read().rsplit(")", 1)[1].split()[0] == "Z":
                    break
            time.sleep(0.01)
        zombie = REAL_HOST_PSS()
        alone = gpu_monitor._proc_pss(me) / 2**20
    finally:
        os.waitpid(pid, 0)
        for fd in (ready_r, ready_w, go_r, go_w):
            os.close(fd)
    assert alive - alone_alive > 200  # the live child's private memory counts
    assert abs(zombie - alone) < 32  # the zombie has no memory left


def test_host_pss_falls_back_to_this_process_rss_when_pss_is_denied(monkeypatch):
    def denied(pid):
        raise PermissionError(f"/proc/{pid}/smaps_rollup")

    monkeypatch.setattr(gpu_monitor, "_proc_pss", denied)
    rss = gpu_monitor._psutil.Process().memory_info().rss / 2**20
    assert REAL_HOST_PSS() == pytest.approx(rss, rel=0.05)


def test_host_pss_skips_a_child_that_exits_mid_read(monkeypatch):
    import os

    me, real_pss = os.getpid(), gpu_monitor._proc_pss

    def pss(pid):
        if pid != me:
            raise FileNotFoundError(f"/proc/{pid}/smaps_rollup")  # listed, then gone
        return real_pss(pid)

    monkeypatch.setattr(gpu_monitor, "_proc_tree", lambda root: [root, 4242])
    monkeypatch.setattr(gpu_monitor, "_proc_pss", pss)
    assert REAL_HOST_PSS() == pytest.approx(real_pss(me) / 2**20, rel=0.05)
