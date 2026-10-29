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

"""Record each training attempt to a folder a dashboard can replay."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .courses import Exercise, build_report
from .safety import ENERGY_ON, GAP_S, SIDES

SESSION_ID = re.compile(r"^\d{8}-\d{6}_[A-Za-z0-9_-]{1,40}(?:-\d+)?\Z")
"""Session folder name: ``<YYYYMMDD-HHMMSS>_<trainee>`` plus ``-<n>`` on a same-second repeat."""
_ACTION_DIM = 28
NEAR_MISS_RATIO = 0.8
"""A safety ratio at or above this (and below 1) is a near miss."""
NEAR_MISS_REARM = 0.6
"""A rule logs another near miss only after its ratio has dropped below this."""


def sanitize_trainee(name: str) -> str:
    """Return the ASCII folder slug of ``name``: ``[A-Za-z0-9_-]`` only (spaces become ``_``), at most 40 characters.

    Falls back to ``"trainee"`` when nothing is left (e.g. a Chinese name).
    """
    return (
        re.sub(r"[^A-Za-z0-9_-]", "", name.strip().replace(" ", "_"))[:40] or "trainee"
    )


def _display_name(name: str) -> str:
    """Return ``name`` without control characters, whitespace collapsed, at most 40 characters (Unicode kept)."""
    printable = "".join(c for c in name if c.isprintable() or c.isspace())
    return " ".join(printable.split())[:40].rstrip() or "anonymous"


class _Auto:
    """Default for ``ffmpeg``: look it up on ``PATH`` when the recorder is created."""


_AUTO = _Auto()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)  # a reader never sees a half-written file


class SessionRecorder:
    """Write one attempt (video, actions, controller frames, events, summary) under ``root / id``.

    Thread-safe: the generation thread adds chunks while the event loop adds events and closes.
    Recording never stops the simulation: a failed file write (e.g. a full disk) is kept in
    ``write_error`` and the session is then never ``complete``.

    Two clocks stamp every event. ``t_video_s`` is the position in the recorded video (frames
    ÷ fps), which the dashboard seeks by. ``t_s`` is *active* time, the measure of how long the
    trainee took: seconds on the ``clock`` since the attempt started, minus the time held. The
    caller holds the attempt with :meth:`begin_hold` when it pauses or stops for an incident, and
    ends the hold with :meth:`end_hold`; the ``pause``, ``incident`` and ``resume`` events only
    log. ``summary.duration_s`` and the exercise report use active time;
    ``summary.video_duration_s`` is the video length. Active time does not depend on how fast the
    GPU generates frames.

    Attributes:
        id: ``<YYYYMMDD-HHMMSS>_<slug of trainee>``, with ``-<n>`` appended if that folder exists.
        path: The session folder.
        trainee: The trainee's display name (Unicode kept, control characters removed, 40 characters).
        video_ok: False when ffmpeg is missing or failed; the other files are still written.
        video_error: Why ``video_ok`` is False, else None.
        write_error: The first file write that failed, else None.
        text_cache: ``{status, rehashed}`` of the session that generated it
            (see ``Cosmos3HDreamsSession.text_cache``), or None.
        exercise: The exercise this attempt is tagged with, or None for free practice.
    """

    def __init__(
        self,
        root: Path,
        *,
        trainee: str,
        scene: str,
        mode: str,
        seed: int,
        fps: float,
        safety: dict[str, float],
        text_cache: dict[str, Any] | None = None,
        exercise: Exercise | None = None,
        now: datetime | None = None,
        ffmpeg: str | None | _Auto = _AUTO,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        now = now or datetime.now()
        if isinstance(ffmpeg, _Auto):
            ffmpeg = shutil.which("ffmpeg")
        self.trainee = _display_name(trainee)
        self.scene, self.mode, self.seed, self.fps, self.safety = (
            scene,
            mode,
            seed,
            fps,
            dict(safety),
        )
        self.text_cache = dict(text_cache) if text_cache is not None else None
        self.started = now.isoformat(timespec="seconds")
        base = f"{now:%Y%m%d-%H%M%S}_{sanitize_trainee(self.trainee)}"
        n = 1
        while True:
            self.id = base if n == 1 else f"{base}-{n}"
            self.path = Path(root) / self.id
            try:
                self.path.mkdir(parents=True)
                break
            except FileExistsError:
                n += 1
        self._ffmpeg = ffmpeg
        self._proc: subprocess.Popen[bytes] | None = None
        self.video_ok = ffmpeg is not None
        self.video_error = None if ffmpeg else "ffmpeg not found"
        self.write_error: str | None = None
        self._frames = 0
        self._actions: list[np.ndarray] = []
        self._last: dict[str, tuple[float, np.ndarray] | None] = dict.fromkeys(SIDES)
        self._on = dict.fromkeys(SIDES, False)
        self._path_m = dict.fromkeys(SIDES, 0.0)
        self._energy_s = dict.fromkeys(SIDES, 0.0)
        self._incidents = 0
        self.ended: str | None = None
        self.end_reason: str | None = None
        self._closed: dict[str, Any] | None = None
        self._vram: dict[str, Any] | None = None
        self.exercise = exercise
        # Kept in memory only for an exercise attempt, to build its report at close.
        self._events: list[dict[str, Any]] = []
        self._poses: list[dict[str, Any]] = []
        self._report: dict[str, Any] | None = None
        self._near_armed: dict[str, bool] = {}
        self._lock = threading.Lock()
        # Active time: time on `clock` since the start, minus the holds (see the class docstring).
        # Its own lock, so that holding never waits for a chunk's video write or for close().
        self._time_lock = threading.Lock()
        self._clock = clock
        self._clock_start = clock()
        self._held_s = 0.0  # finished holds
        self._held_since: float | None = None  # when the current hold began
        self._duration_s: float | None = None  # active seconds, fixed when the attempt closes
        self._log("start", 0.0, {})
        with self._soft_io():
            _write_json(self.path / "session.json", self._snapshot())

    @property
    def video_time_s(self) -> float:
        """Seconds of video recorded so far (frames are counted even when video is off)."""
        return self._frames / self.fps

    @property
    def held(self) -> bool:
        """Whether the attempt is held now: held time is not active time."""
        return self._held_since is not None

    def begin_hold(self) -> None:
        """Stop counting active time, from now; a hold already begun keeps its start. Never blocks on I/O."""
        with self._time_lock:
            if self._held_since is None:
                self._held_since = self._clock()

    def end_hold(self) -> None:
        """Count active time again; does nothing when not held. Never blocks on I/O."""
        with self._time_lock:
            if self._held_since is not None:
                self._held_s += self._clock() - self._held_since
                self._held_since = None

    def add_chunk(
        self,
        frames: Sequence[np.ndarray],
        actions: np.ndarray,
        controller_frames: Sequence[dict[str, Any]],
    ) -> None:
        """Append one generated chunk; ignored after :meth:`close`.

        Args:
            frames: ``uint8`` ``[H, W, 3]`` RGB frames.
            actions: ``float32`` ``[rows, 28]`` model actions.
            controller_frames: Validated controller frames of the chunk, without
                the frame that overlaps the previous chunk. A missing or untracked
                controller breaks path length and energy time, and counts as energy off.
        """
        with self._lock:
            if self._closed is not None:
                return
            if frames and self.video_ok and self._ffmpeg:
                self._write_video(self._ffmpeg, frames)
            self._actions.append(
                np.asarray(actions, dtype=np.float32).reshape(-1, _ACTION_DIM)
            )
            first = self._frames
            self._frames += len(frames)
            lines = []
            for i, frame in enumerate(controller_frames):
                lines.append(json.dumps(frame))
                self._track(frame, (first + i) / self.fps)
            self._append("controller.jsonl", lines)

    def event(self, kind: str, **data: Any) -> None:
        """Log ``kind`` at the current video and active time; ``data`` must be JSON-serialisable.

        An event never holds: a ``pause`` or ``incident`` is held by :meth:`begin_hold`, which
        the caller calls when it holds, not when the event is logged.
        """
        with self._lock:
            if self._closed is None:
                self._log(kind, self.video_time_s, data)

    def observe_safety(self, ratios: Mapping[str, float]) -> None:
        """Log a ``near_miss`` when a rule's ratio first reaches 0.8 while below 1.

        A rule logs again only after its ratio has dropped below 0.6. A ratio of 1
        or more is a safety stop (logged as an incident elsewhere), not a near miss,
        and also waits for the drop below 0.6.
        """
        with self._lock:
            if self._closed is not None:
                return
            for rule, ratio in ratios.items():
                armed = self._near_armed.get(rule, True)
                if armed and NEAR_MISS_RATIO <= ratio < 1:
                    self._near_armed[rule] = False
                    self._log("near_miss", self.video_time_s, {"rule": rule, "ratio": ratio})
                elif ratio >= 1:
                    self._near_armed[rule] = False
                elif ratio < NEAR_MISS_REARM:
                    self._near_armed[rule] = True

    def close(
        self, reason: str, *, vram: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Finish the video, write ``actions.npy`` and ``session.json``, and return the latter.

        Safe to call again: later calls return the same summary, without a second ``end`` event.

        Args:
            reason: Why the attempt ended.
            vram: GPU memory stats over the attempt, stored as ``summary.vram`` when given.
        """
        with self._lock:
            if self._closed is not None:
                return self._closed
            # Before waiting for ffmpeg: finishing the video is not part of the attempt.
            self._duration_s = self._active_s()
            if vram is not None:
                self._vram = dict(vram)
            try:
                # ffmpeg first, so it is reaped even if a later step fails.
                if self._proc is not None:
                    try:
                        self._finish_video(self._proc)
                    except Exception as error:  # noqa: BLE001 - never stop the simulation
                        self._fail_video(error)
                        with contextlib.suppress(Exception):
                            self._proc.kill()
                            self._proc.wait(timeout=5)
                elif self._ffmpeg and self.video_ok:
                    self._fail_video("no frames recorded")
                self._log("end", self.video_time_s, {"reason": reason}, self._duration_s)
                actions = (
                    np.concatenate(self._actions)
                    if self._actions
                    else np.zeros((0, _ACTION_DIM), np.float32)
                )
                with self._soft_io():
                    np.save(self.path / "actions.npy", actions)
                if self.exercise is not None:
                    try:
                        self._report = build_report(
                            self.exercise, self._summary(), self._events, self._poses
                        )
                    except Exception:  # noqa: BLE001 - a report bug must not lose the recording
                        logging.getLogger(__name__).exception("Building the exercise report failed")
            finally:
                self.ended, self.end_reason = _now(), reason
                with self._soft_io():
                    _write_json(self.path / "session.json", self._snapshot())
                self._closed = self._snapshot()  # includes a failure of the write above
            return self._closed

    @contextlib.contextmanager
    def _soft_io(self) -> Iterator[None]:
        """Keep the first ``OSError`` in ``write_error`` instead of raising it."""
        try:
            yield
        except OSError as error:
            self.write_error = self.write_error or str(error)

    def _active_s(self) -> float:
        """Active seconds so far: since the start, minus the holds (a current one up to now)."""
        with self._time_lock:
            now = self._clock()
            held = self._held_s + (0.0 if self._held_since is None else now - self._held_since)
            return max(0.0, now - self._clock_start - held)

    def _summary(self) -> dict[str, Any]:
        duration = self._active_s() if self._duration_s is None else self._duration_s
        summary: dict[str, Any] = {
            "duration_s": round(duration, 3),
            "video_duration_s": round(self.video_time_s, 3),
            "path_length_m": {s: round(v, 4) for s, v in self._path_m.items()},
            "energy_s": {s: round(v, 3) for s, v in self._energy_s.items()},
            "incidents": self._incidents,
        }
        if self._vram is not None:
            summary["vram"] = dict(self._vram)
        return summary

    def _snapshot(self) -> dict[str, Any]:
        snapshot = {
            "id": self.id,
            "trainee": self.trainee,
            "scene": self.scene,
            "mode": self.mode,
            "text_cache": self.text_cache,
            "seed": self.seed,
            "fps": self.fps,
            "safety": self.safety,
            "started": self.started,
            "ended": self.ended,
            "end_reason": self.end_reason,
            "complete": self.ended is not None
            and self.video_ok
            and self.write_error is None,
            "video_ok": self.video_ok,
            "video_error": self.video_error,
            "write_error": self.write_error,
            "summary": self._summary(),
        }
        if self.exercise is not None:
            snapshot["exercise"] = {"id": self.exercise.id, "title": self.exercise.title}
            snapshot["report"] = self._report
        return snapshot

    def _append(self, name: str, lines: Sequence[str]) -> None:
        if lines:
            with (
                self._soft_io(),
                (self.path / name).open("a", encoding="utf-8") as file,
            ):
                file.write("".join(f"{line}\n" for line in lines))

    def _log(
        self,
        kind: str,
        t_video_s: float,
        data: Mapping[str, Any],
        t_s: float | None = None,
    ) -> None:
        """Append an event at ``t_video_s`` and active time ``t_s`` (default: now)."""
        entry = {
            "kind": kind,
            "t_video_s": round(t_video_s, 3),
            "t_s": round(self._active_s() if t_s is None else t_s, 3),
            "wall_time": _now(),
            **data,
        }
        line = json.dumps(entry)
        self._incidents += kind == "incident"
        if self.exercise is not None:
            self._events.append(entry)
        self._append("events.jsonl", [line])

    def _track(self, frame: Mapping[str, Any], t_video_s: float) -> None:
        """Accumulate path length and energy time and log energy edges, as the safety guard measures them."""
        now = int(frame["client_capture_ns"]) / 1e9
        pose: dict[str, Any] = {"t_s": now}
        for side in SIDES:
            controller = frame.get(side)
            on, last = False, None
            if isinstance(controller, Mapping) and controller.get("tracked", True):
                position = np.asarray(controller["position"], dtype=np.float64)
                pose[side] = {"position": position.tolist()}
                on = float(controller.get("grip", 0.0)) >= ENERGY_ON
                previous = self._last[side]
                if previous is not None and 0 < now - previous[0] <= GAP_S:
                    self._path_m[side] += float(np.linalg.norm(position - previous[1]))
                    if on:
                        self._energy_s[side] += now - previous[0]
                last = (now, position)
            self._last[side] = last
            if on != self._on[side]:
                self._on[side] = on
                self._log(
                    "energy_on" if on else "energy_off", t_video_s, {"side": side}
                )
        if self.exercise is not None:
            self._poses.append(pose)

    def _fail_video(self, error: object) -> None:
        self.video_ok, self.video_error = False, str(error)

    def _write_video(self, ffmpeg: str, frames: Sequence[np.ndarray]) -> None:
        try:
            if self._proc is None:
                height, width = frames[0].shape[:2]
                # fmt: off
                command = [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                    "-r", f"{self.fps:g}", "-i", "-",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags", "+faststart", str(self.path / "video.mp4"),
                ]
                # fmt: on
                # ponytail: stderr is only read at close; -loglevel error keeps it far below the pipe buffer.
                self._proc = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
                )
            stdin = self._proc.stdin
            assert stdin is not None  # opened with stdin=PIPE
            for frame in frames:
                stdin.write(frame.tobytes())
        except OSError as error:  # ffmpeg cannot start, or died (BrokenPipeError)
            self._fail_video(error)

    def _finish_video(self, proc: subprocess.Popen[bytes]) -> None:
        try:
            stderr = proc.communicate(timeout=30)[1]
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self._fail_video("ffmpeg did not finish within 30 s")
        else:
            if proc.returncode:
                text = stderr.decode(errors="replace").strip()
                self._fail_video(text or f"ffmpeg exited with status {proc.returncode}")


def _read_session(path: Path) -> dict[str, Any] | None:
    try:
        session = json.loads((path / "session.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(session, dict):
        return None
    try:
        review = json.loads((path / "review.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        review = None
    return {**session, "id": path.name, "review": review if isinstance(review, dict) else None}


def save_review(root: Path, session_id: str, review: Mapping[str, Any]) -> None:
    """Write ``review.json`` into an existing session folder.

    Raises:
        FileNotFoundError: No such session.
        OSError: The write failed.
    """
    path = Path(root) / session_id
    if not SESSION_ID.match(session_id) or not path.is_dir():
        raise FileNotFoundError(session_id)
    _write_json(path / "review.json", dict(review))


def list_sessions(root: Path) -> list[dict[str, Any]]:
    """Return ``session.json`` of every valid session folder under ``root``, newest first."""
    root = Path(root)
    paths = root.iterdir() if root.is_dir() else ()
    sessions = [
        session
        for path in paths
        if SESSION_ID.match(path.name)
        and path.is_dir()
        and (session := _read_session(path))
    ]
    return sorted(sessions, key=lambda session: session["id"], reverse=True)


def load_session(root: Path, session_id: str) -> dict[str, Any] | None:
    """Return ``{session, events, exercise, report, review}`` for one session, or None if there is no such session.

    ``exercise``, ``report`` and ``review`` repeat the session's own fields (None for free practice).
    """
    path = Path(root) / session_id
    if not SESSION_ID.match(session_id) or not path.is_dir():
        return None
    session = _read_session(path)
    if session is None:
        return None
    events = []
    try:
        lines = (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return {
        "session": session,
        "events": events,
        "exercise": session.get("exercise"),
        "report": session.get("report"),
        "review": session.get("review"),
    }
