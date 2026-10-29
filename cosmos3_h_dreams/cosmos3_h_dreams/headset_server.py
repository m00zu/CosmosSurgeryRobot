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

"""Headset WebXR input and HTTPS/MJPEG video for the Cosmos3-H-Dreams session."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import logging
import math
import ssl
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import torch
from aiohttp import WSMsgType, web
from flashdreams.demo import CanonicalInputWindow
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import StepResult
from flashdreams.serving.realtime.media import (
    encode_rgb_frame_to_jpeg,
    rgb_array_to_uint8_frames,
)

from cosmos3_h_dreams._runtime.hand_client.source import Observation
from cosmos3_h_dreams._runtime.headset_controller import parse_xr_frame

from . import gpu_monitor as _gpu_monitor
from .app import Cosmos3HDreamsApplication
from .config import COSMOS3_H_DREAMS_PRESETS, Cosmos3HDreamsConfig
from .courses import Exercise, load_courses
from .instruments import (
    DEFAULT_INSTRUMENTS,
    Instrument,
    instruments_json,
    mask_controls,
)
from .gpu_monitor import GpuMonitor
from .headset import HeadsetActionSource
from .recording import SessionRecorder
from .safety import SafetyGuard, SafetyLimits, SafetyStop
from .session import Cosmos3HDreamsSession
from .session_config import load_session_config
from .training_routes import MODEL_MODES, add_training_routes

_WEB_DIR = Path(__file__).resolve().parent / "web"

# Empirical left grip-to-pivot offset from a Meta Quest controller X-rotation trace, in metres.
_LEFT_GRIP_PIVOT_OFFSET_M = (0.0, -0.02415, -0.04658)
INPUT_RATE_MIN = 5.0
INPUT_RATE_MAX = 60.0
"""Input rates /train may pick (Hz). /train sends poses at max(30 Hz, this rate); a
configured rate above the maximum (a headset) raises it."""

LATENCY_WINDOW = 20
"""Chunks whose timings ``/healthz`` ``latency`` reports the medians of."""
CATCH_UP_INTERVAL = 0.75
"""Frame interval scale while the next block waits for the playing one."""
CATCH_UP_DROP_AFTER = 0.5
"""Skip every other frame once the next block has waited this many block playbacks."""

HEADSET_PRESETS = {
    "headset-base": {
        "headset_input_fps": 30.0,
        "display_fps": 20,
    },
}


def release_cuda_memory() -> int | None:
    """Collect garbage and return cached CUDA blocks to the driver.

    Also drops PyTorch's cuBLAS workspaces. PyTorch keeps one workspace
    (32 MiB + 1 MiB on Hopper) per (cuBLAS handle, stream) pair until the
    process exits. Every session creates its own cache stream, and the server
    steps it from several ``asyncio.to_thread`` workers, each with its own
    handle, so each reload left about 33 MiB per worker thread behind. Call
    this only while no other thread is running CUDA work.

    Returns:
        Bytes still allocated by PyTorch on the current device, or ``None``
        without CUDA. A process that never touched CUDA reports ``0`` without
        creating a CUDA context.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return None
    if not torch.cuda.is_initialized():
        return 0
    torch.cuda.synchronize()
    # Private API (no public equivalent); later cuBLAS calls recreate what they need.
    clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear_workspaces is not None:
        clear_workspaces()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated()


def _uuid_key(uuid: object) -> str:
    text = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
    text = text.strip().lower()
    return text.removeprefix("gpu-")


def _nvml_index(torch_index: int, *, cuda: Any = None, nvml: Any = None) -> int:
    """Return the NVML index of PyTorch device ``torch_index``, else ``torch_index``.

    Matches ``get_device_properties(...).uuid`` against ``nvmlDeviceGetUUID``.
    """
    cuda = torch.cuda if cuda is None else cuda
    try:
        target = _uuid_key(cuda.get_device_properties(torch_index).uuid)
        if nvml is None:
            return torch_index
        nvml.nvmlInit()
        try:
            for index in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                if _uuid_key(nvml.nvmlDeviceGetUUID(handle)) == target:
                    return index
        finally:
            nvml.nvmlShutdown()
        logging.getLogger(__name__).warning(
            "GPU monitor: no NVML device matches CUDA device %d's UUID; "
            "using index %d, which is wrong if CUDA_VISIBLE_DEVICES renumbers GPUs",
            torch_index,
            torch_index,
        )
    except Exception as error:  # noqa: BLE001 - the monitor is optional
        logging.getLogger(__name__).warning(
            "Cannot map CUDA device %d to its NVML index (%s)", torch_index, error
        )
    return torch_index


def _server_gpu(device: str) -> tuple[int, int]:
    """Return the ``(torch, NVML)`` indices of the model's CUDA device, or ``(0, 0)``.

    Reading the device's UUID initialises CUDA, which the model load does anyway.
    """
    try:
        target = torch.device(device)
    except (RuntimeError, TypeError, ValueError):
        return 0, 0
    if target.type != "cuda" or not torch.cuda.is_available():
        return 0, 0
    index = target.index if target.index is not None else torch.cuda.current_device()
    # CUDA_VISIBLE_DEVICES renumbers PyTorch devices but not NVML ones: match by UUID.
    return index, _nvml_index(index, nvml=_gpu_monitor.pynvml)


def _with_settings(config: Any, **settings: Any) -> Any:
    """Return ``config`` with ``settings`` applied, copying rather than mutating."""
    if is_dataclass(config):
        return replace(config, **settings)
    return SimpleNamespace(**{**vars(config), **settings})


def _with_mode(config: Any, mode: str) -> Any:
    """Return ``config`` set up for model mode ``mode`` (see ``MODEL_MODES``)."""
    return _with_settings(config, **MODEL_MODES[mode])


def _mode_settings(config: Any) -> dict[str, Any]:
    """The mode-defining runtime settings of ``config``."""
    return {
        "fp8_generation": getattr(config, "fp8_generation", False),
        "fp8_scope": getattr(config, "fp8_scope", "all"),
    }


def _steady_refresh_sdpa(config: Any) -> bool:
    """Whether ``config`` refreshes saturated FadeMem memory with plain SDPA."""
    return bool(getattr(config, "steady_refresh_sdpa", True))


def _mode_of(config: Any) -> str:
    settings = _mode_settings(config)
    if not settings["fp8_generation"]:
        return "standard"
    return "balanced" if settings["fp8_scope"] == "mlp" else "low_memory"


class LatestHeadsetFrames:
    """Sample the newest validated pose, optionally limiting acquisition rate."""

    def __init__(
        self,
        *,
        fps: float | None,
        instruments: Mapping[str, Instrument] = DEFAULT_INSTRUMENTS,
    ) -> None:
        self.fps = fps
        """Acquisition rate in Hz, or None to take every browser frame."""
        self.instruments = instruments
        """Tools per side; a control for a tool a side lacks is zeroed on push."""
        self._period_ns = 0 if fps is None else round(1e9 / fps)
        self._lock = threading.Lock()
        self._latest: dict[str, Any] | None = None
        self._version = 0
        self._sampled_version = 0
        self._next_sample_ns = 0
        self._closed = False

    def set_fps(self, fps: float) -> None:
        """Change the acquisition rate; the next sample follows the new period."""
        with self._lock:
            self.fps = fps
            self._period_ns = round(1e9 / fps)
            self._next_sample_ns = 0

    def start(self) -> None:
        """Accept controller frames again; a reloaded session reuses this sampler."""
        with self._lock:
            self._closed = False
            self._latest = None

    def stop(self) -> None:
        """Stop accepting controller frames."""
        with self._lock:
            self._closed = True
            self._latest = None

    def clear(self) -> None:
        """Discard poses across browser disconnects and reconnects."""
        with self._lock:
            self._latest = None
            self._sampled_version = self._version
            self._next_sample_ns = 0

    def push(self, payload: object) -> bool:
        """Validate an ``xr_frame`` and retain its absolute controller poses."""
        frame = parse_xr_frame(payload)
        if frame is None:
            return False
        for side in ("left", "right"):
            controller = frame.get(side)
            if controller is not None and any(
                not math.isfinite(float(controller[field]))
                for field in ("trigger", "grip")
            ):
                raise ValueError(f"{side} controller buttons must be finite")
        mask_controls(frame, self.instruments)
        with self._lock:
            if self._closed:
                return False
            self._latest = frame
            self._version += 1
        return True

    def poll(self) -> Observation | None:
        now_ns = time.perf_counter_ns()
        with self._lock:
            if (
                self._closed
                or self._latest is None
                or self._version == self._sampled_version
                or now_ns < self._next_sample_ns
                or now_ns - int(self._latest["client_capture_ns"]) > 500_000_000
            ):
                return None
            frame = self._latest
            self._sampled_version = self._version
            self._next_sample_ns = now_ns + self._period_ns
        return Observation(frame=frame, capture_ns=now_ns, source="headset-mjpeg")


class MjpegFrames:
    """Convert model frames for Pillow JPEG encoding."""

    def prepare_result_frames(self, result: StepResult) -> tuple[Any, ...]:
        video = result.video_chunk
        return tuple(
            rgb_array_to_uint8_frames(
                video,
                layout=result.layout,
                value_range="minus_one_one" if video.is_floating_point() else "uint8",
                sync_device=True,
            )
        )

class HeadsetServer:
    """Bind one headset browser to the existing model session and video track."""

    def __init__(
        self,
        *,
        session: Cosmos3HDreamsSession,
        frames: LatestHeadsetFrames,
        display_fps: int = 30,
        overlap_playback: bool = True,
        scenes: dict[str, Cosmos3HDreamsConfig] | None = None,
        scene_labels: dict[str, str] | None = None,
        scene_start_jaws_closed: dict[str, tuple[str, ...]] | None = None,
        guard: SafetyGuard | None = None,
        recording_dir: Path | None = None,
        session_factory: Callable[[Cosmos3HDreamsConfig], Any] | None = None,
        gpu_monitor: GpuMonitor | None = None,
        exercises: dict[str, Exercise] | None = None,
        demos_dir: Path | None = None,
        course_scene: str | None = None,
    ) -> None:
        self.scenes = scenes or {}
        self.scene_labels = scene_labels or {}
        # Sides whose jaws hold tissue in each scene's frame; /train starts them closed.
        self.scene_start_jaws_closed = scene_start_jaws_closed or {}
        self.active_scene = next(iter(self.scenes), None)
        self._scene_request: tuple[str, asyncio.Future[None]] | None = None
        self._scene_changed = asyncio.Event()
        self.session: Any = session
        self.session_factory = session_factory
        self._mode = _mode_of(getattr(session, "config", None))
        self._steady_refresh_sdpa = _steady_refresh_sdpa(getattr(session, "config", None))
        self.mode_switching = False
        self.mode_error: str | None = None
        self._switch_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._baseline_cuda_bytes: int | None = None
        self.frames = frames
        self.display_fps = display_fps
        self._configured_input_fps = float(
            getattr(frames, "fps", None)
            or getattr(getattr(session, "config", None), "fps", 30.0)
        )
        self.overlap_playback = overlap_playback
        self._last_publish_s: float | None = None
        self._jpeg: bytes | None = None
        self._jpeg_version = 0
        self._jpeg_changed = asyncio.Condition()
        self.source: HeadsetActionSource | None = None
        self.track: MjpegFrames | None = None
        self.socket: web.WebSocketResponse | None = None
        self._ready = asyncio.Event()
        self._producer: asyncio.Task[None] | None = None
        self._closed = False
        self.runtime_ready = False
        self.guard = guard
        # Only the /train page shows incidents and can resume; see check_safety.
        self.safety_active = False
        self.hold: str | None = None
        self._unheld = asyncio.Event()
        self._unheld.set()
        self._incident: SafetyStop | None = None
        self.recording_dir = recording_dir
        self.recorder: SessionRecorder | None = None
        self.recording_error: str | None = None
        self._trainee: str | None = None
        # Start, end and restart run off the event loop; one at a time.
        self._recording_lock = threading.RLock()
        self._recording_started: float | None = None  # time.time() of the attempt start
        self.gpu_monitor = gpu_monitor
        # Validated at start-up; None when the session YAML has no courses block.
        self.exercises = exercises
        self.demos_dir = demos_dir
        # Every exercise attempt and course demo runs on this scene; None allows any.
        self.course_scene = course_scene
        # Readiness after a reset, for /train's progress bar: the epoch counts resets,
        # scene switches and connections; chunks_shown counts steady chunks queued for
        # display since, so the page starts a demo or exercise once the new video runs.
        self.simulation_epoch = 0
        self.chunks_shown = 0
        self._recalibrated = False  # this epoch has seen its calibrating restart
        # The latest checked window's safety ratios; set on the generation thread.
        self._safety_ratios: dict[str, float] | None = None
        # (chunk_ms, frame_ms, response_ms or None) of recent chunks, for /healthz.
        # Event loop only. Cleared on a load, scene switch, reset or calibration.
        self.latency: deque[tuple[float, float, float | None]] = deque(maxlen=LATENCY_WINDOW)
        # perf_counter() when the next block was ready and began waiting for the
        # playing one to finish, else None; the playing block catches up meanwhile.
        self._next_block_waiting: float | None = None

    def latency_budget_ms(self) -> float | None:
        """The real-time budget of one chunk, or None while no session is loaded."""
        steps = getattr(self.session, "actions_per_step", None)
        if steps is None:
            return None
        return _latency_budget_ms(steps, self.input_fps)

    @property
    def input_fps(self) -> float:
        """Rate poses become action rows (Hz); main() falls back to the model rate."""
        fps = getattr(self.frames, "fps", None)
        if fps is None:
            fps = getattr(getattr(self.session, "config", None), "fps", None)
        return float(fps or self._configured_input_fps)

    def input_rate_health(self) -> dict[str, Any]:
        """``/healthz`` ``input_rate``: the rate, its limits and the frames per chunk."""
        return {
            "fps": self.input_fps,
            "min": INPUT_RATE_MIN,
            "max": max(INPUT_RATE_MAX, self._configured_input_fps),
            "chunk_frames": getattr(self.session, "actions_per_step", None),
        }

    def set_input_rate(self, fps: float) -> None:
        """Sample poses and play frames at ``fps``; the model and its memory are kept.

        Playback keeps its ratio to the input rate (equal in the training configs).
        Raises:
            ValueError: ``fps`` is outside ``min``..``max`` of :meth:`input_rate_health`.
            RuntimeError: an attempt is being recorded (its video has one rate).
        """
        limits = self.input_rate_health()
        if not limits["min"] <= fps <= limits["max"]:
            raise ValueError(
                f"input rate must be {limits['min']:g} to {limits['max']:g} Hz"
            )
        if self.recorder is not None:
            raise RuntimeError("Stop the attempt before changing the input rate")
        old = self.input_fps
        if fps == old:
            return
        self.display_fps = self.display_fps * fps / old
        self.frames.set_fps(fps)
        self.latency.clear()  # its samples were measured against the old budget

    def latency_playback_ms(self) -> float | None:
        """How long one chunk's frames play at ``display_fps``, or None without a session."""
        steps = getattr(self.session, "actions_per_step", None)
        return None if steps is None else steps / self.display_fps * 1e3

    @property
    def mode(self) -> str:
        """The loaded model's mode: ``"standard"`` (bf16), ``"balanced"`` (FP8 MLP)
        or ``"low_memory"`` (FP8 attention and MLP).

        Kept from the last successful load, so it stays valid mid-switch.
        """
        return self._mode

    @property
    def steady_refresh_sdpa(self) -> bool:
        """The loaded model's ``steady_refresh_sdpa`` setting, kept like :attr:`mode`."""
        return self._steady_refresh_sdpa

    def start_recording(self, trainee: str, exercise: str | None = None) -> None:
        """Record a new attempt by ``trainee``, ending any current one; blocking I/O.

        Does nothing without a recording directory. A folder that cannot be
        created, or a model mode switch in progress, is reported in
        ``recording_error`` and leaves ``recorder`` unset.

        Args:
            trainee: The trainee's name as typed.
            exercise: An id in ``exercises`` to tag the attempt with (checked by the caller).
        """
        if self.recording_dir is None:
            return
        with self._recording_lock:
            # The /mode route sets mode_switching before the switch task runs, so
            # this also covers the window before the old session is unloaded.
            if self.mode_switching or self.session is None:
                self.recording_error = (
                    "The model is switching; start again when it is ready"
                )
                return
            self._start_recording(trainee, exercise)

    def _start_recording(self, trainee: str, exercise: str | None = None) -> None:
        """Start a recording without the mode-switch check; blocking I/O."""
        if self.recording_dir is None:
            return
        with self._recording_lock:
            config = self.session.config
            self.end_recording("restart")
            self._trainee = trainee
            try:
                self.recorder = SessionRecorder(
                    self.recording_dir,
                    trainee=trainee,
                    scene=self.active_scene or "",
                    mode=self.mode,
                    text_cache=_text_cache(self.session, ("status", "rehashed")),
                    seed=int(getattr(config, "seed", 0)),
                    # The video holds one frame per action row, so it plays in real time.
                    fps=float(self.input_fps),
                    safety=asdict(self.guard.limits) if self.guard else {},
                    exercise=(self.exercises or {}).get(exercise) if exercise else None,
                )
            except OSError as error:
                logging.getLogger(__name__).exception("Cannot start recording")
                self.recording_error = str(error)
            else:
                self.recording_error = None
                self._recording_started = time.time()
                if self.hold is not None:
                    # Starting does not clear a pause or incident hold: begin held. Checked after
                    # `recorder` is set, so a pause() racing with this start is seen by one of them.
                    self.recorder.begin_hold()
                self._mark("attempt_start", self.recorder.id, t=self._recording_started)

    def _mark(self, kind: str, label: str = "", t: float | None = None) -> None:
        """Add a GPU monitor mark (at ``t``, default now); never raises."""
        if self.gpu_monitor is None:
            return
        try:
            self.gpu_monitor.mark(kind, label, t=t)
        except Exception:  # noqa: BLE001 - the monitor is optional
            logging.getLogger(__name__).exception("GPU monitor mark failed")

    def _vram_stats(self, started: float | None, ended: float) -> dict[str, Any] | None:
        """GPU memory stats from ``started`` to ``ended``, or None; never raises."""
        if self.gpu_monitor is None or started is None:
            return None
        try:
            return self.gpu_monitor.stats(started, ended)
        except Exception:  # noqa: BLE001 - the monitor is optional
            logging.getLogger(__name__).exception("GPU monitor stats failed")
            return None

    def check_safety(self, frames: Any) -> list[Any]:
        """Return the guard's violations for ``frames`` while a /train client is connected.

        The headset page has no incident UI, so its motion is never refused.
        """
        if self.guard is None or not self.safety_active:
            return []
        result = self.guard.check(frames)
        ratios = getattr(result, "ratios", None)
        self._safety_ratios = dict(ratios) if ratios is not None else None
        return result

    async def _report_safety(self) -> None:
        """Send the latest window's safety ratios to /train and log near misses."""
        ratios, self._safety_ratios = self._safety_ratios, None
        if ratios is None or self.socket is None or not self.safety_active:
            return
        recorder = self.recorder
        if recorder is not None:
            await asyncio.to_thread(self._record, recorder.observe_safety, ratios)
        with contextlib.suppress(ConnectionError):
            await self.socket.send_json({"type": "safety", "ratios": ratios})

    def _record(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run a recording step; a failure ends the recording, never generation. Blocking I/O."""
        try:
            call(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            logging.getLogger(__name__).exception("Recording failed")
            self.recording_error = f"Recording failed: {error}"
            try:
                self.end_recording("error")
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).exception("Ending the failed recording failed")

    def end_recording(self, reason: str) -> None:
        """Close the current recording, if any; may wait for ffmpeg to finish."""
        with self._recording_lock:
            recorder, self.recorder = self.recorder, None
            started, self._recording_started = self._recording_started, None
            if recorder is not None:
                ended = time.time()  # one end time: the stats window and the mark agree
                try:
                    recorder.close(reason, vram=self._vram_stats(started, ended))
                finally:
                    self._mark("attempt_end", recorder.id, t=ended)

    def restart_recording(self, reason: str) -> None:
        """End the current recording and start a new one for the same trainee; blocking I/O.

        An exercise attempt only ends (its report is written); no new attempt starts.
        """
        with self._recording_lock:
            if self.mode_switching:
                return  # the switch ends and restarts the recording itself
            if self.recorder is not None and self.recorder.exercise is not None:
                self._trainee = None
                self.end_recording(reason)
                return
            if self.recorder is not None and self._trainee is not None:
                trainee = self._trainee
                self.end_recording(reason)
                self.start_recording(trainee)

    def new_simulation(self) -> None:
        """Start a new readiness epoch: a reset, scene switch or connection."""
        self.simulation_epoch += 1
        self.chunks_shown = 0
        self._recalibrated = False

    def simulation_health(self) -> dict[str, Any]:
        """``/healthz`` readiness: epoch, calibration and chunks shown since."""
        state = getattr(self.source, "calibration_state", None)
        calibrated, _status, fraction = state() if state else (False, "", 0.0)
        return {
            "epoch": self.simulation_epoch,
            "calibrated": calibrated,
            "calibration_fraction": round(float(fraction), 3),
            "chunks_shown": self.chunks_shown,
        }

    def exercise_scene(self, exercise_id: str) -> str | None:
        """The scene a known exercise runs on: its own ``scene``, else ``course_scene``."""
        exercise = (self.exercises or {}).get(exercise_id)
        return (exercise.scene if exercise else None) or self.course_scene

    def can_start(self, exercise: object) -> bool:
        """Whether ``exercise`` (from a ``start`` message) is None or a known, attemptable exercise."""
        if exercise is None:
            return True
        found = (self.exercises or {}).get(exercise) if isinstance(exercise, str) else None
        return found is not None and not found.demo_only

    def log_step(self, index: object, t_client: object) -> None:
        """Log a completed exercise step reported by /train; anything invalid is ignored.

        A small file append, cheap enough for the event loop (as :meth:`pause`).
        """
        recorder = self.recorder
        if (
            recorder is None
            or recorder.exercise is None
            or type(index) is not int
            or not 0 <= index < len(recorder.exercise.steps)
        ):
            return
        valid = type(t_client) in (int, float) and math.isfinite(t_client)
        recorder.event("step", index=index, t_client=float(t_client) if valid else None)

    def pause(self) -> None:
        """Hold generation at the next chunk boundary until :meth:`resume`."""
        if self.hold is None:
            self.hold = "pause"
            self._unheld.clear()
            recorder = self.recorder
            if recorder is not None:
                recorder.begin_hold()  # at once: the time held is not the attempt's
                # A small file append; cheap enough for the event loop.
                recorder.event("pause")

    def resume(self) -> None:
        """Release a pause or incident hold; after an incident, continue from the last safe pose."""
        if self.hold == "incident" and self._incident is not None:
            # Drop the newest pre-hold pose before rewinding, so it cannot follow the anchor.
            self.frames.clear()
            if self.source is not None:
                self.source.rewind(self._incident.frames[0])
            if self.guard is not None:
                self.guard.resume()
            self._incident = None
        previous, self.hold = self.hold, None
        self._unheld.set()
        recorder = self.recorder
        if previous is not None and recorder is not None:
            recorder.end_hold()
            recorder.event("resume", after=previous)

    def _clear_hold(self) -> None:
        """Drop any hold and all safety history for a fresh start."""
        self.hold = None
        self._incident = None
        self._unheld.set()
        recorder = self.recorder
        if recorder is not None:
            recorder.end_hold()
        if self.guard is not None:
            self.guard.reset()

    async def _hold_for_incident(self, stop: SafetyStop) -> None:
        """Hold generation and tell the browser which limits the refused chunk exceeded."""
        logging.getLogger(__name__).warning("Safety stop: %s", stop)
        if self.socket is None or not self.safety_active:
            # The browser left mid-step; the next connection recalibrates anyway.
            return
        self.hold = "incident"
        self._incident = stop
        self._unheld.clear()
        recorder = self.recorder
        if recorder is not None:
            # At once, not when the event below is logged: a Resume handled before that would
            # otherwise end no hold, and the logged incident would then hold for good.
            recorder.begin_hold()
        self._mark(
            "incident", stop.violations[0].message() if stop.violations else ""
        )
        if recorder is not None:
            await asyncio.to_thread(
                self._record,
                recorder.event,
                "incident",
                violations=[v.to_json() for v in stop.violations],
                blocked_frames=list(stop.frames),  # validated frames: plain JSON values
            )
        restore = {
            side: {"position": c["position"], "orientation": c["orientation"]}
            for side in ("left", "right")
            if isinstance(c := stop.frames[0].get(side), dict)
        }
        with contextlib.suppress(ConnectionError):
            await self.socket.send_json(
                {
                    "type": "incident",
                    "violations": [v.to_json() for v in stop.violations],
                    "restore": restore,
                }
            )

    async def start(self) -> None:
        """Start the GPU monitor, load the model and begin the generation loop."""
        if self.gpu_monitor is not None:
            # Before the model loads, so the first sample is the baseline.
            await asyncio.to_thread(self._start_gpu_monitor)
        self._baseline_cuda_bytes = await asyncio.to_thread(release_cuda_memory)
        try:
            await asyncio.to_thread(self.session.init)
        except Exception:
            self.session.model_status = "Model startup failed — check server logs"
            logging.getLogger(__name__).exception(
                "Cosmos3-H-Dreams model startup failed"
            )
            return
        self.frames.clear()
        self.runtime_ready = True
        self._producer = asyncio.create_task(self._generate())

    def _start_gpu_monitor(self) -> None:
        monitor = self.gpu_monitor
        assert monitor is not None
        try:
            if monitor.csv_path is not None:
                monitor.csv_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # the monitor reports the CSV failure itself
        try:
            # Synchronously, so the baseline really is taken before the model loads.
            monitor.sample_once()
        except Exception:  # noqa: BLE001 - the monitor is optional
            logging.getLogger(__name__).exception("GPU monitor baseline sample failed")
        try:
            monitor.start()
        except Exception:  # noqa: BLE001 - the monitor is optional
            logging.getLogger(__name__).exception("GPU monitor failed to start")

    async def close(self) -> None:
        """Stop controller sampling, video delivery, the model session and the GPU monitor."""
        try:
            await self._close()
        finally:
            if self.gpu_monitor is not None:
                await asyncio.to_thread(self.gpu_monitor.stop)

    async def _close(self) -> None:
        self._closed = True
        self.runtime_ready = False
        self._ready.set()
        self._unheld.set()
        async with self._jpeg_changed:
            self._jpeg_changed.notify_all()
        if self.source is not None:
            self.source.close()
        if self._producer is not None:
            self._producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._producer
        if self._switch_task is not None:
            # A switch sees _closed after its current stage and closes what it loaded.
            await asyncio.wait([self._switch_task])
        await asyncio.to_thread(self._record, self.end_recording, "server_stop")
        await self.reset_video()
        if self.socket is not None:
            await self.socket.close()
        if self.session is not None:
            await asyncio.to_thread(self.session.close)

    async def switch_mode(self, mode: str) -> None:
        """Reload the model in another mode at a generation boundary.

        Ends any recording, stops generation, closes the session and frees CUDA
        memory, then loads ``mode``. If that load fails, the previous mode is
        reloaded and ``mode_error`` describes the failure.

        Args:
            mode: A key of ``MODEL_MODES``: ``"standard"``, ``"balanced"`` or
                ``"low_memory"``.
        """
        log = logging.getLogger(__name__)
        previous = self.session.config
        previous_mode = self.mode
        self.mode_switching, self.mode_error = True, None
        self._mark("mode_switch_start", mode)
        # An exercise attempt only ends; it never restarts as a new attempt.
        trainee = (
            self._trainee
            if self.recorder is not None and self.recorder.exercise is None
            else None
        )
        try:
            failure = await self._unload()
            if failure is None and not self._closed:
                failure = await self._try_load(_with_mode(previous, mode))
            if failure is not None:
                self.mode_error = failure
                if self._closed:
                    return
                if await self._try_load(previous) is not None:
                    self.mode_error += "; reloading the previous mode also failed"
                    return
            if self._closed:
                return
            self.scenes = {
                name: _with_settings(config, **_mode_settings(self.session.config))
                for name, config in self.scenes.items()
            }
            # Stop or a disconnect during the switch clears the trainee: do not revive it.
            trainee = self._trainee if trainee is not None else None
            if trainee is not None and self.socket is not None:
                await asyncio.to_thread(self._start_recording, trainee)
        except Exception as error:  # noqa: BLE001
            # Runs as a background task: never let an error go unreported.
            log.exception("Mode switch failed")
            self.mode_error = self.mode_error or f"Switch failed: {error}"
        finally:
            self.mode_switching = False
            switched = self.mode_error is None and self.mode == mode
            if self._closed:  # the server closed mid-switch
                label = "cancelled"
            else:
                label = mode if switched else f"failed: {previous_mode}"
            self._mark("mode_switch_end", label)

    async def _unload(self) -> str | None:
        """Stop generation and close the session; return an error message on failure."""
        log = logging.getLogger(__name__)
        try:
            await asyncio.to_thread(self.end_recording, "mode_switch")
        except Exception:  # noqa: BLE001
            log.exception("Ending the recording for a mode switch failed")
        await self._stop_generation()
        failure = None
        try:
            remaining = await asyncio.to_thread(self._discard_session)
        except Exception as error:  # noqa: BLE001
            log.exception("Unloading the model failed")
            failure = f"Switch failed: {error}"
        else:
            if (
                remaining is not None
                and self._baseline_cuda_bytes is not None
                and remaining - self._baseline_cuda_bytes > 2**30
            ):
                log.warning(
                    "CUDA memory after unloading is %.1f GiB above the pre-load baseline",
                    (remaining - self._baseline_cuda_bytes) / 2**30,
                )
        return failure

    async def _try_load(self, config: Any) -> str | None:
        """Load ``config``; on failure discard the session and return an error message."""
        failure = None
        try:
            await self._load(config)
        except Exception as error:  # noqa: BLE001
            logging.getLogger(__name__).exception("Loading the model failed")
            failure = f"Switch failed: {error}"
        # Outside the except block, so the traceback no longer pins the
        # failed load's frames and the memory they reference.
        if failure is not None:
            try:
                await asyncio.to_thread(self._discard_session)
            except Exception:  # noqa: BLE001
                logging.getLogger(__name__).exception("Closing the failed session failed")
        return failure

    def _discard_session(self) -> int | None:
        """Close the current session and free its memory; blocking.

        Returns:
            The result of :func:`release_cuda_memory`, which runs even if closing fails.
        """
        session, self.session = self.session, None
        try:
            if session is not None:
                session.close()
        finally:
            del session
            remaining = release_cuda_memory()
        return remaining

    async def _stop_generation(self) -> None:
        """Stop the generation loop at the next boundary, keeping the last video frame."""
        self._stopping = True
        self.runtime_ready = False
        self._ready.set()
        self._unheld.set()
        if self.source is not None:
            # Wakes a step blocked waiting for controller frames.
            self.source.request_restart()
        if self._producer is not None:
            await asyncio.wait([self._producer])
        self._producer = None
        self._stopping = False
        self.hold = None
        self._incident = None
        self._unheld.set()

    async def _load(self, config: Any) -> None:
        """Create and initialize a session for ``config``, then restart generation.

        Raises:
            Exception: Whatever session initialization raised.
        """
        assert self.session_factory is not None
        self.session = self.session_factory(config)
        self.session.model_status = "Loading model…"
        await asyncio.to_thread(self.session.init)
        if self._closed:
            return
        self._mode = _mode_of(self.session.config)
        self._steady_refresh_sdpa = _steady_refresh_sdpa(self.session.config)
        if self.guard is not None:
            self.guard.reset()
        self.frames.clear()
        self.latency.clear()
        self.runtime_ready = True
        self._producer = asyncio.create_task(self._generate())
        if self.socket is not None:
            self._ready.set()

    async def reset_video(self) -> None:
        """Clear the current MJPEG image and invalidate in-flight playback."""
        self._ready.clear()
        self._last_publish_s = None
        async with self._jpeg_changed:
            self._jpeg = None
            self._jpeg_version += 1
            self._jpeg_changed.notify_all()
        self.track = None

    async def select_scene(self, name: str) -> None:
        """Request a scene change at the next generation boundary."""
        if not isinstance(name, str) or name not in self.scenes:
            raise web.HTTPBadRequest(reason="Unknown scene")
        if not self.runtime_ready or self._closed:
            raise web.HTTPConflict(reason="Model is not ready")
        if self._scene_request is not None:
            raise web.HTTPConflict(reason="A scene change is already pending")
        if self._producer is None or self._producer.done():
            raise web.HTTPConflict(reason="Generation is not running")
        if name == self.active_scene:
            return
        recorder = self.recorder
        if recorder is not None and recorder.exercise is not None:
            raise web.HTTPConflict(reason="Stop the exercise before changing the scene")
        done = asyncio.get_running_loop().create_future()
        self._scene_request = (name, done)
        self._scene_changed.set()
        if self.source is not None:
            self.source.request_restart()
        await asyncio.shield(done)

    async def _generate(self) -> None:
        playback: asyncio.Task[None] | None = None
        try:
            while not self._closed:
                if self._stopping:
                    return
                if self._scene_request is not None:
                    name, done = self._scene_request
                    if playback is not None:
                        playback.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await playback
                        playback = None
                    try:
                        await asyncio.to_thread(
                            self.session.switch_scene, self.scenes[name]
                        )
                    except Exception:
                        logging.getLogger(__name__).exception("Scene switch failed")
                        done.set_exception(
                            web.HTTPInternalServerError(
                                reason="Scene switch failed; check server logs"
                            )
                        )
                    else:
                        self.frames.clear()
                        self.latency.clear()
                        self.new_simulation()
                        self.active_scene = name
                        async with self._jpeg_changed:
                            self._jpeg = None
                            self._jpeg_version += 1
                            self._jpeg_changed.notify_all()
                        self._last_publish_s = None
                        self._clear_hold()
                        await asyncio.to_thread(
                            self._record, self.restart_recording, "scene_switch"
                        )
                        done.set_result(None)
                    finally:
                        if not done.done():
                            done.set_exception(
                                web.HTTPServiceUnavailable(reason="Generation stopped")
                            )
                        self._scene_request = None
                        self._scene_changed.clear()
                if not self._ready.is_set():
                    waits = [
                        asyncio.create_task(self._ready.wait()),
                        asyncio.create_task(self._scene_changed.wait()),
                    ]
                    try:
                        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for waiter in waits:
                            waiter.cancel()
                        await asyncio.gather(*waits, return_exceptions=True)
                    if self._scene_request is not None:
                        continue
                if self._closed or self._stopping:
                    return
                track = self.track
                if self.socket is None or track is None:
                    self._ready.clear()
                    continue
                if self.hold is not None:
                    waits = [
                        asyncio.create_task(self._unheld.wait()),
                        asyncio.create_task(self._scene_changed.wait()),
                    ]
                    try:
                        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for waiter in waits:
                            waiter.cancel()
                        await asyncio.gather(*waits, return_exceptions=True)
                    continue
                requirements = self.session.next_step_requirements()
                if requirements is None:
                    if playback is not None:
                        await playback
                    return
                window = CanonicalInputWindow(
                    window=TimeWindow(
                        start_s=0.0,
                        end_s=requirements.input_frame_count / self.session.config.fps,
                    )
                )
                cycle_started = time.perf_counter()
                epoch = self.simulation_epoch
                try:
                    result = await asyncio.to_thread(self.session.step, window)
                except SafetyStop as stop:
                    if self._stopping:
                        return
                    if playback is not None:
                        await playback
                        playback = None
                    await self._report_safety()  # the meter shows the ratio that stopped it
                    await self._hold_for_incident(stop)
                    continue
                except RuntimeError:
                    if self._closed or self._stopping:
                        return
                    raise
                if self._stopping:
                    return
                await self._report_safety()
                if self._scene_request is not None:
                    continue
                if result.metadata.get("action_rows") == 0:
                    self.latency.clear()  # calibrating: a (re)start, not a steady chunk
                    # Any calibrating step that ends after a reset belongs to it (the reset
                    # sets its restart request before a later step starts, and a step already
                    # running may be the one that took it).
                    self._recalibrated = True
                    self.chunks_shown = 0
                if result.metadata.get("action_rows") == 0 and playback is not None:
                    playback.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await playback
                    playback = None
                    self._last_publish_s = None
                if self.track is track and self._ready.is_set():
                    frames = await asyncio.to_thread(
                        track.prepare_result_frames, result
                    )
                    # Chunk time: from the step's start, less its wait for the action
                    # window, to uint8 frames on the host. prepare_result_frames waits
                    # for the decode and copy, so this covers the GPU work: rollout,
                    # decode, frame conversion and thread hand-offs. Not the JPEG encode.
                    chunk_s = (
                        time.perf_counter()
                        - cycle_started
                        - (getattr(result, "metrics", None) or {}).get("input_s", 0.0)
                    )
                    if self._scene_request is not None:
                        continue
                    recorder = self.recorder
                    if (
                        recorder is not None
                        and result.metadata.get("action_rows")
                        and self.source is not None
                    ):
                        await asyncio.to_thread(
                            self._record,
                            recorder.add_chunk,
                            frames,
                            self.source.last_actions,
                            self.source.last_frames[1:],
                        )
                    prepared = time.perf_counter()
                    if playback is not None:
                        self._next_block_waiting = prepared
                        try:
                            await playback
                        finally:
                            self._next_block_waiting = None
                    if self.track is track and self._ready.is_set():
                        # Keep only the playing block and one prepared block.
                        publish = self._publish_mjpeg(
                            track, result, frames, cycle_started, prepared, chunk_s
                        )
                        if (
                            result.metadata.get("action_rows")
                            and epoch == self.simulation_epoch
                            and self._recalibrated
                        ):
                            self.chunks_shown += 1
                        if self.overlap_playback:
                            playback = asyncio.create_task(publish)
                        else:
                            await publish
                if result.metadata.get("action_rows") == 0:
                    await asyncio.sleep(0.25)
        except Exception:
            self.session.model_status = "Generation failed — check server logs"
            logging.getLogger(__name__).exception("Cosmos3-H-Dreams generation failed")
        finally:
            self.runtime_ready = False
            if self._scene_request is not None:
                _, done = self._scene_request
                done.set_exception(
                    web.HTTPServiceUnavailable(reason="Generation stopped")
                )
                self._scene_request = None
            if playback is not None:
                playback.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await playback

    async def _publish_mjpeg(
        self,
        track: MjpegFrames,
        result: StepResult,
        frames: tuple[Any, ...],
        cycle_started: float,
        prepared: float,
        chunk_s: float | None = None,
    ) -> None:
        """Publish one block in order, pacing frames at the display rate.

        Adds the block's ``chunk_s`` and response time to :attr:`latency` when its
        first frame is published.
        """
        playback_started = time.perf_counter()
        jpeg_s = 0.0
        dropped = 0
        skipped_last = False
        boundary_s = None
        pose_age = "n/a"
        capture_span = "n/a"
        if result.metadata.get("action_first_capture_ns") is not None:
            capture_span = f"{(result.metadata['action_capture_ns'] - result.metadata['action_first_capture_ns']) / 1e6:.1f}ms"
        for index, frame in enumerate(frames):
            if (
                self.track is not track
                or not self._ready.is_set()
                or self._scene_request is not None
            ):
                return
            # Catch-up. Playback can only fall behind the input: the next block waits
            # for this one, and frame pacing overshoots a little each frame, so a delay
            # gained once (a slow chunk, display slower than the input) is never won
            # back and creeps up until the controller buffer drops poses. While the
            # next block waits, frames are shown at 3/4 of the interval, which is hard
            # to see. If it has waited more than half a block's playback, every other
            # frame is skipped too (never the first), as a display slower than the
            # input needs. With nothing waiting, playback is unchanged.
            waiting = self._next_block_waiting
            if (
                index > 0
                and not skipped_last
                and waiting is not None
                and time.perf_counter() - waiting
                > CATCH_UP_DROP_AFTER * len(frames) / self.display_fps
            ):
                dropped += 1
                skipped_last = True
                continue
            skipped_last = False
            started = asyncio.get_running_loop().time()
            jpeg = await asyncio.to_thread(encode_rgb_frame_to_jpeg, frame)
            jpeg_s += asyncio.get_running_loop().time() - started
            if (
                self.track is not track
                or not self._ready.is_set()
                or self._scene_request is not None
            ):
                return
            # ponytail: retain one JPEG; slow viewers drop frames. Use
            # per-viewer bounded queues if complete playback is needed.
            async with self._jpeg_changed:
                self._jpeg = jpeg
                self._jpeg_version += 1
                self._jpeg_changed.notify_all()
            published = time.perf_counter()
            if index == 0 and result.metadata.get("action_capture_ns") is not None:
                pose_age = f"{(published - result.metadata['action_capture_ns'] / 1e9) * 1e3:.1f}ms"
            if index == 0 and chunk_s is not None and result.metadata.get("action_rows"):
                self.latency.append(
                    (chunk_s * 1e3, chunk_s * 1e3 / len(frames), _response_ms(result, published))
                )
            if index == 0 and self._last_publish_s is not None:
                boundary_s = published - self._last_publish_s
            self._last_publish_s = published
            interval = 1 / self.display_fps
            if self._next_block_waiting is not None:
                interval *= CATCH_UP_INTERVAL
            await asyncio.sleep(
                max(0, interval - (asyncio.get_running_loop().time() - started))
            )
        if self.session.config.profile and result.metadata.get("action_rows"):
            finished = time.perf_counter()
            metrics = result.metrics
            print(
                f"[headset-mjpeg] step={result.step_index} frames={len(frames)} "
                f"input={metrics['input_s'] * 1e3:.1f}ms "
                f"model={metrics['model_wall_s'] * 1e3:.1f}ms "
                f"decode={metrics['decode_s'] * 1e3:.1f}ms "
                f"prepare/dispatch={(prepared - cycle_started - metrics['input_s'] - metrics['total_s']) * 1e3:.1f}ms "
                f"generation={(prepared - cycle_started) * 1e3:.1f}ms "
                f"queue_wait={(playback_started - prepared) * 1e3:.1f}ms "
                f"jpeg={jpeg_s * 1e3:.1f}ms "
                f"publish/pacing={(finished - playback_started - jpeg_s) * 1e3:.1f}ms "
                f"cycle={(finished - cycle_started) * 1e3:.1f}ms "
                f"boundary_interval={boundary_s * 1e3 if boundary_s is not None else 0:.1f}ms "
                f"overlap={self.overlap_playback} pose_age_first_frame={pose_age} "
                f"input_span={capture_span} dropped={dropped}",
                flush=True,
            )


def _text_cache(session: Any, keys: tuple[str, ...] | None = None) -> Any:
    """The session's text cache outcome (only ``keys`` of it, if given), or None."""
    text_cache = getattr(session, "text_cache", None)
    if text_cache is None or keys is None:
        return text_cache
    return {key: text_cache.get(key) for key in keys}


def _scenes_sharing_text_cache_inputs(
    scenes: dict[str, Cosmos3HDreamsConfig],
) -> str | None:
    """None when every scene has the same prompt, height and width; else what differs."""
    items = list(scenes.items())
    if not items:
        return None
    first_name, first = items[0]
    key = (first.prompt, first.height, first.width)
    others = [name for name, c in items[1:] if (c.prompt, c.height, c.width) != key]
    if not others:
        return None
    return (
        f"scene(s) {', '.join(repr(n) for n in others)} differ from scene "
        f"{first_name!r} in prompt, height or width"
    )


def _health_gpu(monitor: GpuMonitor | None) -> dict[str, float | None] | None:
    """The server device's latest memory sample for ``/healthz``, or None."""
    if monitor is None:
        return None
    latest = monitor.latest(monitor.device)
    if latest is None:
        return None
    return {
        "used_mib": latest.used_mib,
        "total_mib": latest.total_mib,
        "peak_mib": monitor.peak().get(monitor.device),
    }


def _response_ms(result: StepResult, published_s: float) -> float | None:
    """Milliseconds from the chunk's first action row reaching the server to ``published_s``.

    The row's ``client_capture_ns`` is, despite its name, the server's
    ``perf_counter_ns`` when :meth:`LatestHeadsetFrames.push` received the pose
    (``parse_xr_frame`` stamps it; the browser's own time travels as
    ``relay_timestamp``). So this spans waiting for the window to fill,
    generation and queueing behind the previous block, on one clock. None
    without live input, or if the stamp is not a plausible server time.
    """
    first_ns = result.metadata.get("action_first_capture_ns")
    if first_ns is None:
        return None
    response_s = published_s - first_ns / 1e9
    return response_s * 1e3 if 0 <= response_s < 60 else None


def _latency_budget_ms(actions_per_step: int, input_fps: float) -> float:
    """Milliseconds one chunk may take to keep up: its action rows arrive at ``input_fps``."""
    return actions_per_step / input_fps * 1e3


def _health_latency(
    samples: Sequence[tuple[float, float, float | None]],
    budget_ms: float | None,
    playback_ms: float | None = None,
) -> dict[str, Any] | None:
    """Medians of ``(chunk_ms, frame_ms, response_ms)`` samples for ``/healthz``, or None.

    ``status`` is ``"behind"`` when the median chunk time reaches ``budget_ms``
    (``reason`` ``"generation"``) or when a chunk's playback takes more than 2 %
    longer than its input (``reason`` ``"playback"``: chunks queue behind
    playback), else ``"tight"`` from 80 % of the budget, else ``"ok"``; both are
    None without a budget. ``response_ms`` is None when no sample has one.
    """
    if not samples:
        return None
    chunk_ms = median(s[0] for s in samples)
    responses = [s[2] for s in samples if s[2] is not None]
    status = reason = None
    if budget_ms is not None:
        if chunk_ms >= budget_ms:
            status, reason = "behind", "generation"
        elif playback_ms is not None and playback_ms > budget_ms * 1.02:
            status, reason = "behind", "playback"
        else:
            status = "tight" if chunk_ms >= 0.8 * budget_ms else "ok"
    return {
        "chunk_ms": round(chunk_ms, 1),
        "frame_ms": round(median(s[1] for s in samples), 1),
        "response_ms": round(median(responses), 1) if responses else None,
        "budget_ms": round(budget_ms, 1) if budget_ms is not None else None,
        "playback_ms": round(playback_ms, 1) if playback_ms is not None else None,
        "status": status,
        "reason": reason,
    }


def create_app(manager: HeadsetServer) -> web.Application:
    """Serve the headset page, controller socket, and MJPEG video."""
    app = web.Application(client_max_size=64 * 1024)

    async def page(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "headset.html").read_text()
        return web.Response(
            text=body,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def script(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "headset.js").read_text()
        return web.Response(
            text=body,
            content_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def scene_script(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "scenes.js").read_text()
        return web.Response(
            text=body,
            content_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def viewer_page(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "viewer.html").read_text()
        return web.Response(
            text=body,
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def viewer_script(_: web.Request) -> web.Response:
        body = (_WEB_DIR / "viewer.js").read_text()
        return web.Response(
            text=body,
            content_type="text/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def socket(request: web.Request) -> web.WebSocketResponse:
        if manager.socket is not None:
            raise web.HTTPConflict(reason="A Headset browser is already connected")
        ws = web.WebSocketResponse(max_msg_size=64 * 1024, heartbeat=5.0)
        await ws.prepare(request)
        manager.socket = ws
        manager.safety_active = request.query.get("client") == "train"
        manager.frames.clear()
        manager.new_simulation()
        if manager.source is not None:
            manager.source.require_released_triggers = not manager.safety_active
            manager.source.reset_calibration()
        manager.track = MjpegFrames()
        manager._ready.set()
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                    if not isinstance(payload, dict):
                        raise TypeError("message must be an object")
                    kind = payload.get("type")
                    if kind == "reset":
                        manager.latency.clear()
                        manager.new_simulation()
                        if manager.source is not None:
                            manager.source.request_restart()
                        manager._clear_hold()
                        await asyncio.to_thread(
                            manager._record, manager.restart_recording, "reset"
                        )
                    elif kind == "start":
                        trainee = payload.get("trainee")
                        exercise = payload.get("exercise")
                        if not manager.can_start(exercise):
                            await ws.send_json(
                                {"type": "error", "message": "Unknown exercise"}
                            )
                        elif exercise is not None and (
                            scene := manager.exercise_scene(exercise)
                        ) not in (None, manager.active_scene):
                            await ws.send_json(
                                {
                                    "type": "error",
                                    "message": f"This exercise runs on the {scene!r} "
                                    "scene; switch to it first",
                                }
                            )
                        elif isinstance(trainee, str):
                            await asyncio.to_thread(
                                manager._record,
                                manager.start_recording,
                                trainee,
                                exercise,
                            )
                    elif kind == "step":
                        manager.log_step(payload.get("index"), payload.get("t_client"))
                    elif kind == "stop":
                        manager._trainee = None
                        await asyncio.to_thread(
                            manager._record, manager.end_recording, "stop"
                        )
                    elif kind == "pause":
                        manager.pause()
                    elif kind == "resume":
                        manager.resume()
                    else:
                        manager.frames.push(payload)
                except (ValueError, TypeError, KeyError):
                    await ws.send_json(
                        {"type": "error", "message": "Invalid controller frame"}
                    )
        finally:
            # Not resume(): a disconnect is not a resume, and the next user starts fresh.
            manager._clear_hold()
            manager.safety_active = False
            manager._trainee = None
            manager.socket = None
            manager.frames.clear()
            if manager.source is not None:
                manager.source.reset_calibration()
            try:
                await manager.reset_video()
            finally:
                # aiohttp may cancel this handler once the browser has gone; a
                # cancelled to_thread that has not started yet would never run.
                await asyncio.shield(
                    asyncio.to_thread(
                        manager._record, manager.end_recording, "disconnect"
                    )
                )
        return ws

    async def video(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        version = -1
        try:
            while not manager._closed:
                async with manager._jpeg_changed:
                    # Periodically check for disconnected idle subscribers.
                    try:
                        await asyncio.wait_for(
                            manager._jpeg_changed.wait_for(
                                lambda version=version: (
                                    manager._closed or manager._jpeg_version != version
                                )
                            ),
                            timeout=1,
                        )
                    except TimeoutError:
                        if request.transport is None or request.transport.is_closing():
                            break
                        continue
                    version, jpeg = manager._jpeg_version, manager._jpeg
                if manager._closed:
                    break
                if jpeg is not None:
                    part = (
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    await asyncio.wait_for(response.write(part), timeout=5)
        except (ConnectionError, TimeoutError):
            pass
        return response

    async def select_scene(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise TypeError
            name = payload["name"]
        except (ValueError, KeyError, TypeError) as error:
            raise web.HTTPBadRequest(reason="Expected a scene name") from error
        await manager.select_scene(name)
        return web.json_response({"active_scene": manager.active_scene})

    async def scene_frame(request: web.Request) -> web.FileResponse:
        scene = manager.scenes.get(request.query.get("name", ""))
        if scene is None:
            raise web.HTTPNotFound(reason="Unknown scene")
        return web.FileResponse(scene.initial_frame)

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {
                "ready": manager.runtime_ready,
                "headset_connected": manager.socket is not None,
                "model_status": manager.session.model_status
                if manager.session is not None
                else "Switching model…"
                if manager.mode_switching
                else "Model startup failed — check server logs",
                "scenes": [
                    {
                        "name": name,
                        "label": manager.scene_labels.get(name, name),
                        "start_jaws_closed": list(
                            manager.scene_start_jaws_closed.get(name, ())
                        ),
                    }
                    for name in manager.scenes
                ],
                "active_scene": manager.active_scene,
                "scene_switching": manager._scene_request is not None,
                "hold": manager.hold,
                "recording": None
                if (r := manager.recorder) is None
                else {
                    "id": r.id,
                    "trainee": r.trainee,
                    "video_ok": r.video_ok,
                    "exercise": r.exercise.id if r.exercise else None,
                },
                "recording_error": manager.recording_error,
                "mode": manager.mode,
                "mode_switching": manager.mode_switching,
                "mode_error": manager.mode_error,
                "steady_refresh_sdpa": manager.steady_refresh_sdpa,
                "input_rate": manager.input_rate_health(),
                "instruments": instruments_json(manager.frames.instruments),
                "gpu": _health_gpu(manager.gpu_monitor),
                "simulation": manager.simulation_health(),
                "latency": _health_latency(
                    manager.latency,
                    manager.latency_budget_ms(),
                    manager.latency_playback_ms(),
                ),
                "text_cache": _text_cache(manager.session),
                "attention": getattr(manager.session, "attention", None),
            },
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_get("/headset", page)
    app.router.add_get("/headset.js", script)
    app.router.add_get("/scenes.js", scene_script)
    app.router.add_get("/viewer", viewer_page)
    app.router.add_get("/viewer.js", viewer_script)
    app.router.add_get("/ws", socket)
    app.router.add_get("/video", video)
    app.router.add_get("/healthz", health)
    app.router.add_post("/scene", select_scene)
    app.router.add_get("/scene/frame", scene_frame)

    add_training_routes(app, manager)

    async def lifecycle(_app: web.Application):
        startup = asyncio.create_task(manager.start())
        try:
            yield
        finally:
            await startup
            await manager.close()

    app.cleanup_ctx.append(lifecycle)
    return app


def main(argv: list[str] | None = None) -> None:
    """Start the independent HTTPS headset flow with the shared model options.

    Args:
        argv: Command-line arguments, defaulting to the process arguments.
    """
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--headset-config", choices=HEADSET_PRESETS)
    selector.add_argument("--session-config", type=Path)
    selected, _ = selector.parse_known_args(argv)
    session_data = (
        load_session_config(selected.session_config)
        if selected.session_config
        else None
    )
    headset_defaults = dict((session_data or {}).get("headset", {}))
    headset_preset = selected.headset_config or headset_defaults.pop("preset", None)
    if headset_preset not in {None, *HEADSET_PRESETS}:
        selector.error(f"unknown headset preset: {headset_preset}")
    parser = argparse.ArgumentParser(
        description="Serve Cosmos3-H-Dreams to a headset browser over HTTPS/MJPEG.",
        epilog=(
            "Model options such as --checkpoint, --initial-frame, --domain-id, "
            "and --fps match flashdreams-run cosmos3-h-dreams."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", help="TLS certificate (required unless --http)")
    parser.add_argument("--key", help="TLS private key (required unless --http)")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve plain HTTP instead of HTTPS (trusted networks only). /train, "
        "/viewer and /dashboard work; /headset needs HTTPS (WebXR) except on localhost.",
    )
    parser.add_argument("--config", choices=COSMOS3_H_DREAMS_PRESETS)
    parser.add_argument("--headset-config", choices=HEADSET_PRESETS)
    parser.add_argument("--display-fps", type=int, default=30)
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument(
        "--headset-input-fps",
        type=float,
        help="Acquire measured action windows at this rate, independently of model --fps",
    )
    parser.add_argument(
        "--overlap-playback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the next MJPEG block while playing the current block",
    )
    parser.add_argument(
        "--gpu-monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample GPU memory for /gpu, /api/gpu and a CSV log in the recording directory",
    )
    parser.add_argument(
        "--gpu-period", type=float, default=1.0, help="GPU sampling period in seconds"
    )
    gpu_defaults = (session_data or {}).get("gpu_monitor")
    if gpu_defaults is not None:
        parser.set_defaults(
            gpu_monitor=gpu_defaults["enabled"], gpu_period=gpu_defaults["period_s"]
        )
    headset_defaults.pop("preset", None)
    parser.set_defaults(**HEADSET_PRESETS.get(headset_preset, {}))
    parser.set_defaults(**headset_defaults)
    network, model_args = parser.parse_known_args(argv)
    if not network.http and (network.cert is None or network.key is None):
        parser.error("--cert and --key are required unless --http is given")
    if not 1 <= network.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if network.display_fps <= 0:
        parser.error("--display-fps must be positive")
    if network.headset_input_fps is not None and (
        not math.isfinite(network.headset_input_fps) or network.headset_input_fps <= 0
    ):
        parser.error("--headset-input-fps must be finite and positive")
    if not math.isfinite(network.gpu_period) or network.gpu_period <= 0:
        parser.error("--gpu-period must be finite and positive")
    courses = (session_data or {}).get("courses")
    instruments = (session_data or {}).get("instruments") or DEFAULT_INSTRUMENTS
    exercises = None
    if courses is not None:
        try:
            exercises = load_courses(courses["dir"], courses["demos_dir"], instruments)
        except ValueError as error:
            parser.error(str(error))
    application = Cosmos3HDreamsApplication()
    model_defaults = (
        ["--config", network.config]
        if network.config is not None
        else []
        if session_data is not None
        else [
            "--domain-id",
            "25",
            "--fps",
            "10",
            "--height",
            "288",
            "--width",
            "512",
        ]
    )
    application.init(["--headset", *model_defaults, *model_args])
    session = application.create_session()
    config = session.config
    input_fps = (
        network.headset_input_fps if network.headset_input_fps is not None else config.fps
    )
    frames = LatestHeadsetFrames(fps=input_fps, instruments=instruments)
    limits = (session_data or {}).get("safety") or SafetyLimits()
    guard = SafetyGuard(limits)
    manager: HeadsetServer

    def source_factory(**kwargs: Any) -> HeadsetActionSource:
        source = HeadsetActionSource(
            **kwargs,
            max_buffered_frames=config.chunk_size * 4 + 1,
            pivot_offsets={"left": _LEFT_GRIP_PIVOT_OFFSET_M},
            print_actions=network.print_actions,
            source_factory=lambda **_source_kwargs: frames,
            guard=manager.check_safety,
        )
        # /train may start with jaws closed on tissue (start_jaws_closed, G).
        source.require_released_triggers = not manager.safety_active
        manager.source = source
        return source

    for name, sides in application.scene_start_jaws_closed.items():
        for side in sides:
            if "jaws" not in instruments[side].tools:
                parser.error(
                    f"scene {name!r}: start_jaws_closed: the "
                    f"{instruments[side].label(side)} has no jaws"
                )
    course_scene = None
    if courses is not None and application.scenes:
        course_scene = courses["scene"] or next(iter(application.scenes))
        if course_scene not in application.scenes:
            parser.error(
                f"courses.scene: no scene named {course_scene!r} "
                f"(scenes: {', '.join(application.scenes)})"
            )
        for exercise in (exercises or {}).values():
            if exercise.scene is not None and exercise.scene not in application.scenes:
                parser.error(
                    f"{exercise.path}: scene: no scene named {exercise.scene!r} "
                    f"(scenes: {', '.join(application.scenes)})"
                )
    if any(scene.actions is not None for scene in application.scenes.values()):
        parser.error("Headset server scenes must use live controller input, not actions")
    if config.text_cache is not None and (
        mismatch := _scenes_sharing_text_cache_inputs(application.scenes)
    ):
        # The cache holds one prompt; with it the understanding weights are dropped, so
        # another prompt could neither be prefilled on a scene switch nor rebuilt cheaply.
        parser.error(
            f"text_cache holds the text K/V of one prompt, but {mismatch}; "
            "drop text_cache or unify the prompts"
        )
    def session_factory(cfg: Cosmos3HDreamsConfig) -> Cosmos3HDreamsSession:
        return Cosmos3HDreamsSession(config=cfg, headset_source_factory=source_factory)

    recording_dir = Path(
        (session_data or {}).get("recording", {}).get("dir")
        or Path.cwd() / "recordings"
    )
    gpu_monitor = None
    if network.gpu_monitor:
        torch_index, nvml_index = _server_gpu(str(config.device))
        gpu_monitor = GpuMonitor(
            device=nvml_index,
            torch_device=torch_index,
            period_s=network.gpu_period,
            history=max(1, round(1800 / network.gpu_period)),  # 30 minutes of samples
            csv_path=recording_dir / f"gpu-{datetime.now():%Y%m%d-%H%M%S}.csv",
        )
    manager = HeadsetServer(
        session=session_factory(config),
        session_factory=session_factory,
        frames=frames,
        display_fps=network.display_fps,
        overlap_playback=network.overlap_playback,
        scenes=application.scenes,
        scene_labels=application.scene_labels,
        scene_start_jaws_closed=application.scene_start_jaws_closed,
        guard=guard,
        recording_dir=recording_dir,
        gpu_monitor=gpu_monitor,
        exercises=exercises,
        demos_dir=courses["demos_dir"] if courses is not None else None,
        course_scene=course_scene,
    )
    if config.normalizer is None:
        print(
            "[headset-mjpeg] no checkpoint normalizer supplied; "
            "live action magnitudes are unverified"
        )
    tls = None
    if not network.http:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(network.cert, network.key)
    scheme = "http" if network.http else "https"
    print(
        f"[headset-mjpeg] action/model rate: {config.fps:g} Hz; "
        f"acquisition: {'browser rate' if input_fps is None else str(input_fps) + ' Hz'}; "
        f"actions: recorded window; "
        f"video presentation: {network.display_fps} fps; "
        f"overlap playback: {network.overlap_playback}"
    )
    print(f"[headset-mjpeg] safety limits: {limits}")
    if exercises is not None:
        print(f"[headset-mjpeg] courses: {len(exercises)} exercises from {courses['dir']}")
    print(
        f"[headset-mjpeg] GPU memory log: {gpu_monitor.csv_path}"
        if gpu_monitor is not None
        else "[headset-mjpeg] GPU monitor: off"
    )
    if network.http:
        print("[headset-mjpeg] plain HTTP: anyone who can reach this port can use the server")
    print(f"Open {scheme}://<workstation-LAN-IP>:{network.port}/headset in the headset browser")
    print(f"Open {scheme}://<workstation-LAN-IP>:{network.port}/viewer on a PC")
    print(f"Open {scheme}://<workstation-LAN-IP>:{network.port}/train for the training ground")
    web.run_app(
        create_app(manager), host=network.host, port=network.port, ssl_context=tls
    )


if __name__ == "__main__":
    main()
