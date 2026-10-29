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

"""Safety limits that stop live generation on extreme instrument motion or energy use."""

from __future__ import annotations

import argparse
import copy
import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

SIDES = ("left", "right")
ENERGY_ON = 0.5
"""Grip value at or above which energy counts as applied."""
SPAN_S = 0.1
"""Shortest span used to measure motion; frame-to-frame timing jitters."""
GAP_S = 0.25
"""Sample gap treated as a discontinuity (reset, pause, calibration)."""

_LABELS = {
    "speed": "speed",
    "acceleration": "acceleration",
    "rotation": "rotation rate",
    "energy_duration": "energy held",
    "energy_dose": "energy in window",
    "energy_sweep": "distance swept with energy",
}
RULES = tuple(_LABELS)
"""Rule names, in display order."""


@dataclass(frozen=True)
class SafetyLimits:
    """Thresholds above which a chunk of trainee input is refused.

    The defaults sit at about the highest values surgeons reached in the hand
    motion of 40 CMR Versius cholecystectomies (Open-H-Embodiment), measured as
    this guard measures (0.1 s spans, engaged and unclutched): speed max 0.37 m/s
    (99th percentile 0.16), acceleration max 3.2 m/s² (0.65), rotation max
    4.3 rad/s (0.96), one energy activation 99th percentile 8.8 s, energy in 15 s
    max 13.7 s, distance moved while energized 90th percentile 0.09 m.
    """

    max_speed_m_s: float = 0.35
    max_acceleration_m_s2: float = 3.0
    max_rotation_rad_s: float = 4.0
    max_energy_s: float = 10.0
    max_energy_dose_s: float = 14.0
    energy_dose_window_s: float = 15.0
    max_energy_sweep_m: float = 0.10

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SafetyLimits:
        """Build limits from a session YAML ``safety`` block.

        Raises:
            ValueError: An option is unknown or not a positive finite number.
        """
        data = dict(data or {})
        unknown = data.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown safety options: {', '.join(sorted(unknown))}")
        for key, value in data.items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"safety.{key} must be a positive number")
        return cls(**{key: float(value) for key, value in data.items()})


@dataclass(frozen=True)
class Violation:
    """One limit exceeded by one instrument."""

    rule: str
    side: str
    value: float
    limit: float
    unit: str

    def message(self) -> str:
        """Return a trainee-facing description."""
        return (
            f"{self.side.capitalize()} instrument: {_LABELS[self.rule]} "
            f"{self.value:.2f} {self.unit}, limit {self.limit:.2f} {self.unit}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return the websocket and event-log form."""
        return {**asdict(self), "message": self.message()}


class SafetyCheck(list):
    """One window's violations (the list :meth:`SafetyGuard.check` always returned) and its measured maxima.

    Attributes:
        maxima: Per rule in :data:`RULES`, the highest value measured on either instrument.
        ratios: Per rule, the highest value over its limit, rounded to 3 decimals.
    """

    def __init__(
        self,
        violations: Sequence[Violation] = (),
        maxima: Mapping[str, float] | None = None,
        ratios: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(violations)
        self.maxima = {**dict.fromkeys(RULES, 0.0), **(maxima or {})}
        self.ratios = {**dict.fromkeys(RULES, 0.0), **(ratios or {})}


class SafetyStop(Exception):
    """A chunk of input was refused before generation."""

    def __init__(self, violations: list[Violation], frames: Sequence[dict[str, Any]]):
        super().__init__("; ".join(v.message() for v in violations))
        self.violations = violations
        self.frames = tuple(frames)


@dataclass
class _Side:
    history: deque = field(default_factory=lambda: deque(maxlen=64))
    """Recent ``(time_s, position, orientation, speed)`` samples."""
    energy_run_s: float = 0.0
    sweep_m: float = 0.0
    energy_log: deque = field(default_factory=deque)
    """Energized intervals as ``(end_time_s, duration_s)``."""

    def restart(self) -> None:
        """Drop motion history and the current activation; keep the dose log."""
        self.history.clear()
        self.energy_run_s = 0.0
        self.sweep_m = 0.0


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


class SafetyGuard:
    """Check each action window against :class:`SafetyLimits`."""

    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits
        # check() runs on the generation thread; reset()/resume() on the event loop.
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Forget all motion and energy history."""
        with self._lock:
            self._state = {side: _Side() for side in SIDES}

    def resume(self) -> None:
        """Restart motion history and the energy activation; keep the dose window."""
        with self._lock:
            for side in self._state.values():
                side.restart()

    def check(self, frames: Sequence[Mapping[str, Any]]) -> SafetyCheck:
        """Measure one window; commit rolling state only when it is safe.

        Args:
            frames: Validated controller frames in capture order. The first
                frame overlaps the previous window and only anchors timing.

        Returns:
            Violations, at most one per rule and instrument (the largest), as a
            :class:`SafetyCheck` that also carries the window's measured maxima.
        """
        with self._lock:
            return self._check(frames)

    def _check(self, frames: Sequence[Mapping[str, Any]]) -> SafetyCheck:
        state = copy.deepcopy(self._state)
        limits = self.limits
        worst: dict[tuple[str, str], Violation] = {}
        maxima = dict.fromkeys(RULES, 0.0)
        ratios = dict.fromkeys(RULES, 0.0)

        def exceed(rule: str, side: str, value: float, limit: float, unit: str) -> None:
            maxima[rule] = max(maxima[rule], value)
            ratios[rule] = max(ratios[rule], value / limit)
            if value > limit and value > getattr(
                worst.get((rule, side)), "value", -1.0
            ):
                worst[(rule, side)] = Violation(rule, side, value, limit, unit)

        for name in SIDES:
            side = state[name]
            previous = (
                (side.history[-1][0], side.history[-1][1]) if side.history else None
            )
            for frame in frames:
                controller = frame.get(name)
                if not isinstance(controller, Mapping) or not controller.get(
                    "tracked", True
                ):
                    side.restart()
                    previous = None
                    continue
                now = int(frame["client_capture_ns"]) / 1e9
                position = np.asarray(controller["position"], dtype=np.float64)
                orientation = np.asarray(controller["orientation"], dtype=np.float64)
                if side.history and now - side.history[-1][0] > GAP_S:
                    side.restart()
                    previous = None
                if side.history and now <= side.history[-1][0]:
                    continue
                speed = None
                base = next(
                    (h for h in reversed(side.history) if now - h[0] >= SPAN_S), None
                )
                if base is not None:
                    span = now - base[0]
                    speed = float(np.linalg.norm(position - base[1])) / span
                    exceed("speed", name, speed, limits.max_speed_m_s, "m/s")
                    exceed(
                        "rotation",
                        name,
                        _angle(orientation, base[2]) / span,
                        limits.max_rotation_rad_s,
                        "rad/s",
                    )
                    if base[3] is not None:
                        exceed(
                            "acceleration",
                            name,
                            abs(speed - base[3]) / span,
                            limits.max_acceleration_m_s2,
                            "m/s²",
                        )
                if previous is not None:
                    dt = now - previous[0]
                    step = float(np.linalg.norm(position - previous[1]))
                    if float(controller.get("grip", 0.0)) >= ENERGY_ON:
                        side.energy_run_s += dt
                        side.sweep_m += step
                        side.energy_log.append((now, dt))
                    else:
                        side.energy_run_s = 0.0
                        side.sweep_m = 0.0
                while (
                    side.energy_log
                    and side.energy_log[0][0] < now - limits.energy_dose_window_s
                ):
                    side.energy_log.popleft()
                exceed(
                    "energy_duration", name, side.energy_run_s, limits.max_energy_s, "s"
                )
                exceed(
                    "energy_dose",
                    name,
                    sum(d for _, d in side.energy_log),
                    limits.max_energy_dose_s,
                    "s",
                )
                exceed(
                    "energy_sweep", name, side.sweep_m, limits.max_energy_sweep_m, "m"
                )
                side.history.append((now, position, orientation, speed))
                previous = (now, position)
        if not worst:
            self._state = state
        return SafetyCheck(
            sorted(worst.values(), key=lambda v: (v.side, v.rule)),
            {rule: round(value, 4) for rule, value in maxima.items()},
            {rule: round(value, 3) for rule, value in ratios.items()},
        )


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    a, b = rot6d[..., 0:3], rot6d[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    y = b - (x * b).sum(-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def calibrate(
    paths: Sequence[str], *, fps: float, translation_scale: float
) -> dict[str, float]:
    """Return the 99th-percentile value of each rule over un-normalized ``[N, 28]`` trajectories.

    The trajectories must be raw (not normalized) model actions: translation at
    ``translation_scale`` per metre (``headset._TRANSLATION_SCALE``), rot6d
    relative to the reference pose, energy at columns 10 and 24.
    """
    speeds, accels, rates, energy_runs = [], [], [], []
    for path in paths:
        actions = np.load(path).astype(np.float64)
        for xyz, rot, energy in ((0, 3, 10), (14, 17, 24)):
            position = actions[:, xyz : xyz + 3] / translation_scale
            speed = np.linalg.norm(np.diff(position, axis=0), axis=1) * fps
            speeds.extend(speed)
            accels.extend(np.abs(np.diff(speed)) * fps)
            matrices = _rot6d_to_matrix(actions[:, rot : rot + 6])
            relative = np.einsum("nji,njk->nik", matrices[:-1], matrices[1:])
            cos = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
            rates.extend(np.arccos(cos) * fps)
            run = 0
            for on in actions[:, energy] >= ENERGY_ON:
                run = run + 1 if on else 0
                if on:
                    energy_runs.append(run / fps)
    pick = lambda values: float(np.percentile(values, 99)) if len(values) else 0.0  # noqa: E731
    return {
        "max_speed_m_s": pick(speeds),
        "max_acceleration_m_s2": pick(accels),
        "max_rotation_rad_s": pick(rates),
        "max_energy_s": pick(energy_runs),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Print 99th-percentile safety thresholds for recorded trajectories."""
    parser = argparse.ArgumentParser(prog="python -m cosmos3_h_dreams.safety")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "calibrate", help="Suggest thresholds from [N, 28] trajectories"
    )
    command.add_argument("paths", nargs="+")
    command.add_argument("--fps", type=float, default=10.0)
    command.add_argument("--translation-scale", type=float, default=10.0)
    args = parser.parse_args(argv)
    for key, value in calibrate(
        args.paths, fps=args.fps, translation_scale=args.translation_scale
    ).items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
