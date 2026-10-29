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

"""GPU memory sampler for Cosmos3-H-Dreams.

A daemon thread samples every GPU through NVML plus the server device's PyTorch
allocator counters, keeps a ring buffer, marks, a baseline and a peak, and
appends every sample to a CSV log. It never creates a CUDA context.

A second daemon thread takes a snapshot of the server process tree's host memory
(itself plus its children, e.g. ffmpeg and compile workers) every
``host_period_s``, as summed PSS so pages a forked child shares count once. Each
GPU sample of the server device carries the latest snapshot, which may be up to
``host_period_s`` old; the GPU sampler itself never reads host memory.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

try:
    import pynvml as _pynvml
except ImportError:  # the monitor then runs on PyTorch numbers only
    _pynvml = None
pynvml: Any = _pynvml  # Any: None when the package is missing

try:
    import psutil as _psutil  # a flashdreams dependency
except ImportError:  # no RSS fallback when PSS cannot be read
    _psutil = None

logger = logging.getLogger(__name__)

_MIB = 2**20
CSV_HEADER = (
    "time",
    "gpu",
    "used_mib",
    "total_mib",
    "util_pct",
    "temp_c",
    "torch_allocated_mib",
    "torch_reserved_mib",
    "host_pss_mib",
)


@dataclass(frozen=True)
class GpuSample:
    t: float  # time.time()
    gpu: int
    used_mib: float
    total_mib: float
    util_pct: float | None
    temp_c: float | None
    torch_allocated_mib: float | None  # only for the server's device
    torch_reserved_mib: float | None
    host_pss_mib: float | None = None  # PSS of server + children; device sample only

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _torch_cuda() -> Any | None:
    """``torch.cuda`` once CUDA is initialised, else None; never creates a context."""
    torch = sys.modules.get("torch")  # not imported means CUDA cannot be initialised
    try:
        ready = torch is not None and torch.cuda.is_initialized()
        return torch.cuda if ready and torch.cuda.is_available() else None
    except Exception:
        return None


def _proc_tree(root: int) -> list[int]:
    """``root`` and its live descendants, from ``/proc/<pid>/task/*/children``."""
    pids, queue = [], [root]
    while queue:
        pid = queue.pop()
        pids.append(pid)
        try:
            tasks = os.listdir(f"/proc/{pid}/task")
        except OSError:  # exited
            continue
        for task in tasks:
            try:
                with open(f"/proc/{pid}/task/{task}/children", "rb") as handle:
                    queue.extend(int(child) for child in handle.read().split())
            except OSError:  # thread or process exited
                pass
    return pids


def _proc_pss(pid: int) -> int:
    """PSS of ``pid`` in bytes from ``/proc/<pid>/smaps_rollup``; 0 without memory (a zombie)."""
    with open(f"/proc/{pid}/smaps_rollup", "rb") as handle:
        for line in handle:
            if line.startswith(b"Pss:"):
                return int(line.split()[1]) * 1024
    return 0


def _host_pss_mib() -> float | None:
    """One coherent snapshot: the summed PSS of this process and its descendants, in MiB.

    PSS (proportional set size) splits each shared page between the processes
    that map it, so forked workers (e.g. torchinductor's compile pool) add only
    their private pages and the sum is the tree's real footprint. Every live
    process is read in the same pass; one that has exited is skipped. Without a
    readable PSS for this process (not Linux, access denied) its RSS stands in,
    or None without psutil. A pass walks every resident page (about 2.5 ms per
    GiB per process), so it runs on the monitor's host thread, not per sample.
    """
    me = os.getpid()
    try:
        total = _proc_pss(me)
    except (OSError, ValueError):
        return None if _psutil is None else _psutil.Process(me).memory_info().rss / _MIB
    for pid in _proc_tree(me)[1:]:
        try:
            total += _proc_pss(pid)
        except (OSError, ValueError):  # exited between listing and reading
            pass
    return total / _MIB


def _cell(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def _peak_mean(values: list[float]) -> tuple[float | None, float | None]:
    return (max(values), sum(values) / len(values)) if values else (None, None)


class GpuMonitor:
    """Sample GPU memory once per ``period_s``; see the module docstring.

    ``device`` is the server's GPU index (the NVML index when NVML works); the
    PyTorch counters are attached to it only and read with ``torch_device``,
    which defaults to ``device`` but differs when ``CUDA_VISIBLE_DEVICES``
    remaps indices. ``samples(since=t)`` returns samples
    strictly newer than ``t`` so a polling client can pass the last ``t`` it saw.
    A stopped monitor cannot be restarted and no longer writes the CSV.
    """

    def __init__(
        self,
        *,
        device: int = 0,
        torch_device: int | None = None,
        period_s: float = 1.0,
        history: int = 1800,
        csv_path: Path | None = None,
        reader: Callable[[], list[GpuSample]] | None = None,
        clock: Callable[[], float] = time.time,
        host_reader: Callable[[], float | None] | None = None,
        host_period_s: float = 5.0,
    ) -> None:
        self.device = device
        self.torch_device = device if torch_device is None else torch_device
        self.period_s = period_s
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.csv_name = self.csv_path.name if self.csv_path else None
        self.csv_error: str | None = None
        self.nvml = False
        self.gpus: list[dict] = []
        self._clock = clock
        self._history = history
        self._buf: deque[GpuSample] = deque(maxlen=history)
        self._marks: deque[dict] = deque(maxlen=500)
        self._baseline: dict[int, float] = {}
        self._peak: dict[int, float] = {}
        self._host_now: float | None = None
        self._host_peak: float | None = None
        self._host_reader = host_reader
        self.host_period_s = host_period_s
        self._host_pss: float | None = None  # latest snapshot, set by read_host()
        self._host_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._csv_lock = threading.Lock()
        self._csv_on = self.csv_path is not None
        self._csv_file: Any = None
        self._csv_writer: Any = None
        self._warned: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handles: list[Any] = []
        self._nvml_open = False
        self._reader = reader or self._read
        if reader is None:
            self._init_nvml()

    # -- NVML and the default reader ------------------------------------------------
    def _try(self, what: str, read: Callable[[], Any]) -> Any:
        """Run one read; on error log it once per ``what`` and return None."""
        try:
            return read()
        except Exception as exc:
            if what not in self._warned:
                self._warned.add(what)
                logger.warning("GPU monitor: %s failed (%s)", what, exc)
            return None

    def _init_nvml(self) -> None:
        def open_nvml() -> list[Any]:
            if pynvml is None:
                raise ImportError("pynvml is not installed")
            pynvml.nvmlInit()
            self._nvml_open = True
            count = pynvml.nvmlDeviceGetCount()
            return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]

        handles = self._try("NVML init (using PyTorch counters only)", open_nvml)
        if handles is None:
            self._shutdown_nvml()
            return
        self.nvml, self._handles = True, handles
        for index, handle in enumerate(handles):
            name = self._try(
                f"reading GPU {index} name", lambda: pynvml.nvmlDeviceGetName(handle)
            )
            mem = self._try(
                f"reading GPU {index} memory",
                lambda: pynvml.nvmlDeviceGetMemoryInfo(handle),
            )
            if mem is not None:
                self._note_gpu(index, mem.total / _MIB, name)

    def _shutdown_nvml(self) -> None:
        if self._nvml_open:
            self._nvml_open = False
            self._try("NVML shutdown", pynvml.nvmlShutdown)

    def _note_gpu(self, index: int, total_mib: float, name: str | None = None) -> None:
        if all(g["index"] != index for g in self.gpus):
            gpu = {
                "index": index,
                "name": name or f"GPU {index}",
                "total_mib": total_mib,
            }
            self.gpus = sorted([*self.gpus, gpu], key=lambda g: g["index"])

    def _torch_mib(self, cuda: Any) -> tuple[float | None, float | None]:
        d = self.torch_device
        numbers = self._try(
            "reading PyTorch memory counters",
            lambda: (cuda.memory_allocated(d) / _MIB, cuda.memory_reserved(d) / _MIB),
        )
        return numbers or (None, None)

    def _read(self) -> list[GpuSample]:
        """The default reader: every GPU, with host PSS on the server device's sample."""
        samples = self._read_gpus()
        if any(s.gpu == self.device for s in samples):
            with self._lock:
                pss = self._host_pss
            samples = [
                replace(s, host_pss_mib=pss) if s.gpu == self.device else s
                for s in samples
            ]
        return samples

    def _read_gpus(self) -> list[GpuSample]:
        now, cuda, out = self._clock(), _torch_cuda(), []
        if not self.nvml:  # PyTorch numbers only: reserved memory stands in for "used"
            allocated, reserved = self._torch_mib(cuda) if cuda else (None, None)
            props = cuda and self._try(
                "reading PyTorch device properties",
                lambda: cuda.get_device_properties(self.torch_device),
            )
            if reserved is not None and props:
                total = props.total_memory / _MIB
                out.append(
                    GpuSample(
                        now,
                        self.device,
                        reserved,
                        total,
                        None,
                        None,
                        allocated,
                        reserved,
                    )
                )
            return out
        for index, handle in enumerate(
            self._handles
        ):  # one bad GPU never stops the rest
            mem = self._try(
                f"reading GPU {index} memory",
                lambda: pynvml.nvmlDeviceGetMemoryInfo(handle),
            )
            if mem is None:
                continue
            util = self._try(
                f"reading GPU {index} utilization",
                lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            )
            temp = self._try(
                f"reading GPU {index} temperature",
                lambda: float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                ),
            )
            allocated, reserved = (
                self._torch_mib(cuda) if cuda and index == self.device else (None, None)
            )
            out.append(
                GpuSample(
                    now,
                    index,
                    mem.used / _MIB,
                    mem.total / _MIB,
                    util,
                    temp,
                    allocated,
                    reserved,
                )
            )
        return out

    # -- sampling -------------------------------------------------------------------
    def sample_once(self) -> list[GpuSample]:
        """Read, append to the ring buffer, update baseline and peak, write the CSV."""
        samples = self._reader()
        with self._lock:
            for s in samples:
                self._note_gpu(s.gpu, s.total_mib)
            capacity = self._history * max(1, len(self.gpus))
            if self._buf.maxlen != capacity:
                self._buf = deque(self._buf, maxlen=capacity)
            for s in samples:
                self._buf.append(s)
                self._baseline.setdefault(s.gpu, s.used_mib)
                self._peak[s.gpu] = max(self._peak.get(s.gpu, s.used_mib), s.used_mib)
                if s.gpu == self.device and s.host_pss_mib is not None:
                    self._host_now = s.host_pss_mib
                    self._host_peak = max(self._host_peak or 0.0, s.host_pss_mib)
        self._write_csv(samples)
        return samples

    def _write_csv(self, samples: list[GpuSample]) -> None:
        path = self.csv_path
        if path is None or not self._csv_on or not samples:
            return
        with self._csv_lock:
            if not self._csv_on:  # stop() or an earlier failure got here first
                return
            try:
                if self._csv_writer is None:
                    self._csv_file = path.open("a", newline="", encoding="utf-8")
                    self._csv_writer = csv.writer(self._csv_file, lineterminator="\n")
                    if self._csv_file.tell() == 0:
                        self._csv_writer.writerow(CSV_HEADER)
                for s in samples:
                    self._csv_writer.writerow(
                        [
                            f"{s.t:.3f}",
                            s.gpu,
                            _cell(s.used_mib),
                            _cell(s.total_mib),
                            _cell(s.util_pct),
                            _cell(s.temp_c),
                            _cell(s.torch_allocated_mib),
                            _cell(s.torch_reserved_mib),
                            _cell(s.host_pss_mib),
                        ]
                    )
                self._csv_file.flush()
            except (OSError, ValueError, csv.Error) as exc:
                # a broken log must never stop sampling: disable it, say so once
                self.csv_error, self._csv_on = str(exc), False
                logger.warning("GPU monitor: CSV log disabled (%s)", exc)
                self._close_csv()

    def _close_csv(self) -> None:
        file, self._csv_file, self._csv_writer = self._csv_file, None, None
        if file is not None:
            try:
                file.close()
            except (OSError, ValueError):
                pass

    def read_host(self) -> float | None:
        """Take one host memory snapshot (MiB) and keep it for the next samples."""
        read = self._host_reader or _host_pss_mib  # looked up late for tests
        pss = self._try("reading host memory", read)
        with self._lock:
            self._host_pss = pss
        return pss

    def _run_host(self) -> None:
        while not self._stop.is_set():
            self.read_host()
            self._stop.wait(self.host_period_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._try("sampling", self.sample_once)
            self._stop.wait(self.period_s)

    def start(self) -> None:
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(
                target=self._run, name="gpu-monitor", daemon=True
            )
            self._thread.start()
            self._host_thread = threading.Thread(
                target=self._run_host, name="gpu-monitor-host", daemon=True
            )
            self._host_thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        host_thread, self._host_thread = self._host_thread, None
        for joining in (thread, host_thread):
            if joining is not None:
                joining.join(timeout=5.0)
        with self._csv_lock:
            self._csv_on = False
            self._close_csv()
        if thread is None or not thread.is_alive():
            self._shutdown_nvml()

    # -- queries --------------------------------------------------------------------
    def mark(self, kind: str, label: str = "", t: float | None = None) -> None:
        """Add a mark at ``t`` (default: now), e.g. the end of a stats window."""
        with self._lock:
            when = self._clock() if t is None else t
            self._marks.append({"t": when, "kind": kind, "label": label})

    def samples(
        self, since: float | None = None, gpu: int | None = None
    ) -> list[GpuSample]:
        with self._lock:
            return [
                s
                for s in self._buf
                if (since is None or s.t > since) and (gpu is None or s.gpu == gpu)
            ]

    def latest(self, gpu: int) -> GpuSample | None:
        """The newest sample of ``gpu`` still in the ring buffer, or None."""
        with self._lock:
            for s in reversed(self._buf):
                if s.gpu == gpu:
                    return s
        return None

    def marks(self, since: float | None = None) -> list[dict]:
        with self._lock:
            return [dict(m) for m in self._marks if since is None or m["t"] > since]

    def baseline(self) -> dict[int, float]:
        with self._lock:
            return dict(self._baseline)

    def peak(self) -> dict[int, float]:
        with self._lock:
            return dict(self._peak)

    def host(self) -> dict[str, float | None]:
        """Latest and peak host PSS of the server process tree (MiB); None before any reading."""
        with self._lock:
            return {"pss_mib": self._host_now, "peak_mib": self._host_peak}

    def stats(self, t0: float, t1: float) -> dict[str, float | None]:
        """Peak and mean over the device's samples with ``t0 <= t <= t1``.

        Only samples still in the ring buffer count (the last ``history``
        periods); the CSV is the full record.
        """
        rows = [s for s in self.samples(gpu=self.device) if t0 <= s.t <= t1]
        peak_used, mean_used = _peak_mean([s.used_mib for s in rows])
        allocated = [
            s.torch_allocated_mib for s in rows if s.torch_allocated_mib is not None
        ]
        peak_alloc, mean_alloc = _peak_mean(allocated)
        peak_host, mean_host = _peak_mean(
            [s.host_pss_mib for s in rows if s.host_pss_mib is not None]
        )
        return {
            "peak_used_mib": peak_used,
            "mean_used_mib": mean_used,
            "peak_allocated_mib": peak_alloc,
            "mean_allocated_mib": mean_alloc,
            "peak_host_pss_mib": peak_host,
            "mean_host_pss_mib": mean_host,
        }
