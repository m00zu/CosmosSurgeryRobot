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

"""Training courses for /train: exercise files, demos, path deviation and scoring.

Pure functions and file I/O only, so every part is CPU-testable. Grading is an
illustrative scheme, not a clinical standard.
"""

from __future__ import annotations

import bisect
import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .instruments import DEFAULT_INSTRUMENTS, Instrument
from .safety import GAP_S, SIDES

FPS = 30
"""Demo frame rate; matches the 30 Hz tick of /train."""
HOME = {"left": (-0.15, 1.0, -0.3), "right": (0.15, 1.0, -0.3)}
"""/train home positions in metres (+x right, +y up, -z forward)."""
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,59}\Z")
"""Exercise ids and recorded demo names: lower-case letters, digits and hyphens."""
NOTES_MAX = 2000
"""Longest instructor review note, in characters."""
_ITEM_ID = re.compile(r"^[a-z0-9_]{1,40}\Z")
_GAP_CAP_S = 1.0
"""A pause in a recorded demo plays back as at most this long."""
_EXERCISE_KEYS = {
    "id",
    "course",
    "order",
    "title",
    "brief",
    "time_limit_s",
    "demo_only",
    "demo",
    "steps",
    "rubric",
    "scene",
}
_ACTIONS = ("move_to", "rotate_by", "jaws", "energy", "home", "wait_s")
_KEYFRAME_KEYS = {*_ACTIONS, "side", "over_s", "caption"}
_CHECKS = {
    "reach": {"side", "center", "radius_m"},
    "active": set(),
    "jaws_closed": {"side", "hold_s"},
    "jaws_open": {"side"},
    "energy": {"side", "hold_s"},
    "home": {"side"},
    "moved": {"side", "min_m"},
}
_AUTO = (
    "no_safety_stops",
    "max_time_s",
    "max_path_deviation_m",
    "max_energy_s",
    "max_near_misses",
    "all_steps",
)


@dataclass(frozen=True)
class Demo:
    """Controller frames at :data:`FPS` that /train plays in place of mouse input.

    Attributes:
        source: ``"scripted"`` (keyframes in an exercise file) or ``"recorded"`` (a saved attempt).
        name: The recorded demo's file name without ``.jsonl``; None when scripted.
        frames: ``[{t_s, active, left: {position, orientation, trigger, grip}, right: {...}}]``.
        markers: ``[{t_s, caption}]`` timeline markers.
        info: The recorded demo's header ``{name, source, created}``; None when scripted.
    """

    source: str
    name: str | None
    frames: list[dict[str, Any]]
    markers: list[dict[str, Any]]
    info: dict[str, Any] | None = None

    @property
    def duration_s(self) -> float:
        """Time of the last frame."""
        return self.frames[-1]["t_s"] if self.frames else 0.0

    def to_json(self) -> dict[str, Any]:
        """Return the ``/api/courses/{id}/demo`` and ``/api/demos/{name}`` body."""
        return {
            "source": self.source,
            "name": self.name,
            "title": (self.info or {}).get("name"),
            "duration_s": self.duration_s,
            "frames": self.frames,
            "markers": self.markers,
        }


@dataclass(frozen=True)
class Exercise:
    """One validated exercise file.

    Attributes:
        steps: ``({text, check: {kind: argument}}, ...)`` in completion order.
        rubric: ``({id, text, points, auto: {kind: limit} | None}, ...)``; ``auto=None`` is an instructor item.
        demo_spec: ``{"scripted": [keyframes]}``, ``{"recorded": name}`` or None.
        demo: The demo compiled at load time, or None.
        scene: The scene this exercise runs on, or None for the ``courses.scene`` default.
        path: The exercise file, for error messages.
    """

    id: str
    course: str
    order: int
    title: str
    brief: str
    time_limit_s: float | None
    demo_only: bool
    steps: tuple[dict[str, Any], ...]
    rubric: tuple[dict[str, Any], ...]
    demo_spec: dict[str, Any] | None
    demo: Demo | None
    path: Path = field(compare=False)
    scene: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return the exercise as listed by ``/api/courses``."""
        return {
            "id": self.id,
            "course": self.course,
            "order": self.order,
            "title": self.title,
            "brief": self.brief,
            "time_limit_s": self.time_limit_s,
            "demo_only": self.demo_only,
            "steps": list(self.steps),
            "step_count": len(self.steps),
            "rubric": list(self.rubric),
            "has_demo": self.demo is not None,
            "demo_source": self.demo.source if self.demo else None,
            "demo_name": self.demo.name if self.demo else None,
            "demo_duration_s": self.demo.duration_s if self.demo else None,
            "scene": self.scene,
        }


# -- validation ----------------------------------------------------------------------


def _fail(where: Path, field_name: str, problem: str) -> ValueError:
    return ValueError(f"{where}: {field_name}: {problem}")


def _keys(
    value: Any, allowed: set[str], where: Path, field_name: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _fail(where, field_name, "must be a mapping")
    unknown = sorted(value.keys() - allowed)
    if unknown:
        raise _fail(where, field_name, f"unknown key {unknown[0]!r}")
    return value


def _one(
    value: Any, kinds: Sequence[str], where: Path, field_name: str
) -> tuple[str, Any]:
    _keys(value, set(kinds), where, field_name)
    if len(value) != 1:
        raise _fail(where, field_name, f"needs exactly one of {', '.join(kinds)}")
    return next(iter(value.items()))


def _number(
    value: Any, where: Path, field_name: str, *, positive: bool = True
) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise _fail(
            where,
            field_name,
            "must be a positive number" if positive else "must be a number >= 0",
        )
    return float(value)


def _text(value: Any, where: Path, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, field_name, "must be a non-empty string")
    return value.strip()


def _side(value: Any, where: Path, field_name: str, *, both: bool = False) -> str:
    allowed = (*SIDES, "both") if both else SIDES
    if value not in allowed:
        raise _fail(where, field_name, f"must be one of {', '.join(allowed)}")
    return value


def _triple(value: Any, where: Path, field_name: str, what: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise _fail(where, field_name, f"must be {what}")
    return [float(v) for v in value]


def _check(value: Any, where: Path, field_name: str) -> dict[str, Any]:
    kind, arg = _one(value, tuple(_CHECKS), where, field_name)
    at = f"{field_name}.{kind}"
    if kind == "active":
        return {"active": _side(arg, where, at)}
    _keys(arg, _CHECKS[kind], where, at)
    if kind == "reach":
        out = {
            "side": _side(arg.get("side"), where, f"{at}.side"),
            "center": _triple(
                arg.get("center"), where, f"{at}.center", "[x, y, z] in metres"
            ),
            "radius_m": _number(arg.get("radius_m"), where, f"{at}.radius_m"),
        }
    elif kind in ("jaws_closed", "energy"):
        out = {
            "side": _side(arg.get("side"), where, f"{at}.side"),
            "hold_s": _number(
                arg.get("hold_s", 0), where, f"{at}.hold_s", positive=False
            ),
        }
    elif kind == "home":
        out = {"side": _side(arg.get("side"), where, f"{at}.side", both=True)}
    elif kind == "jaws_open":
        out = {"side": _side(arg.get("side"), where, f"{at}.side")}
    else:  # moved
        out = {
            "side": _side(arg.get("side", "both"), where, f"{at}.side", both=True),
            "min_m": _number(arg.get("min_m"), where, f"{at}.min_m"),
        }
    return {kind: out}


def _auto(
    value: Any, where: Path, field_name: str, *, has_demo: bool
) -> dict[str, Any]:
    kind, arg = _one(value, _AUTO, where, field_name)
    at = f"{field_name}.{kind}"
    if kind in ("no_safety_stops", "all_steps"):
        if arg is not True:
            raise _fail(where, at, "must be true")
    elif kind == "max_near_misses":
        if type(arg) is not int or arg < 0:
            raise _fail(where, at, "must be an integer >= 0")
    elif kind == "max_energy_s":
        arg = _number(arg, where, at, positive=False)
    else:
        arg = _number(arg, where, at)
        if kind == "max_path_deviation_m" and not has_demo:
            raise _fail(where, at, "needs a demo to compare against")
    return {kind: arg}


def _keyframe(value: Any, where: Path, field_name: str) -> dict[str, Any]:
    _keys(value, _KEYFRAME_KEYS, where, field_name)
    actions = [key for key in _ACTIONS if key in value]
    if len(actions) != 1:
        raise _fail(where, field_name, f"needs exactly one of {', '.join(_ACTIONS)}")
    action = actions[0]
    at = f"{field_name}.{action}"
    raw = value[action]
    out: dict[str, Any] = {}
    if action == "move_to":
        out[action] = _triple(raw, where, at, "[x, y, z] in metres")
    elif action == "rotate_by":
        out[action] = _triple(raw, where, at, "[yaw, pitch, roll] in radians")
    elif action in ("jaws", "energy"):
        choices = ("close", "open") if action == "jaws" else ("on", "off")
        if action == "energy" and isinstance(
            raw, bool
        ):  # YAML 1.1 reads on/off as booleans
            raw = "on" if raw else "off"
        if raw not in choices:
            raise _fail(where, at, f"must be {' or '.join(choices)}")
        out[action] = raw
    elif action == "home":
        if raw is not True:
            raise _fail(where, at, "must be true")
        out[action] = True
    else:
        out[action] = _number(raw, where, at)
    if "side" in value:
        if action == "wait_s":
            raise _fail(where, f"{field_name}.side", "not allowed with wait_s")
        out["side"] = _side(
            value["side"],
            where,
            f"{field_name}.side",
            both=action not in ("move_to", "rotate_by"),
        )
    if "over_s" in value:
        if action not in ("move_to", "rotate_by", "home"):
            raise _fail(
                where,
                f"{field_name}.over_s",
                "only allowed with move_to, rotate_by or home",
            )
        out["over_s"] = _number(value["over_s"], where, f"{field_name}.over_s")
    if "caption" in value:
        out["caption"] = _text(value["caption"], where, f"{field_name}.caption")
    return out


def load_exercise(path: Path, demos_dir: Path) -> Exercise:
    """Load and validate one exercise file, compiling its demo.

    Raises:
        ValueError: ``"<file>: <field>: <problem>"`` for any invalid content.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: cannot read: {error}") from error
    _keys(data, _EXERCISE_KEYS, path, "exercise")
    exercise_id = data.get("id")
    if not isinstance(exercise_id, str) or not SLUG.match(exercise_id):
        raise _fail(
            path, "id", "must be lower-case letters, digits and hyphens (at most 60)"
        )
    order = data.get("order")
    if type(order) is not int or order < 0:
        raise _fail(path, "order", "must be an integer >= 0")
    demo_only = data.get("demo_only", False)
    if type(demo_only) is not bool:
        raise _fail(path, "demo_only", "must be true or false")
    time_limit = data.get("time_limit_s")
    if time_limit is not None:
        time_limit = _number(time_limit, path, "time_limit_s")
    spec = None
    if "demo" in data:
        kind, arg = _one(data["demo"], ("scripted", "recorded"), path, "demo")
        if kind == "scripted":
            if not isinstance(arg, list) or not arg:
                raise _fail(path, "demo.scripted", "must be a non-empty list")
            spec = {
                "scripted": [
                    _keyframe(k, path, f"demo.scripted[{i}]") for i, k in enumerate(arg)
                ]
            }
        else:
            if not isinstance(arg, str) or not SLUG.match(arg):
                raise _fail(path, "demo.recorded", "must be a recorded demo name")
            spec = {"recorded": arg}
    steps: list[dict[str, Any]] = []
    rubric: list[dict[str, Any]] = []
    if demo_only:
        if spec is None:
            raise _fail(path, "demo", "required when demo_only is true")
        for key in ("steps", "rubric", "time_limit_s"):
            if key in data:
                raise _fail(path, key, "not allowed when demo_only is true")
    else:
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise _fail(path, "steps", "must be a non-empty list")
        for i, step in enumerate(raw_steps):
            at = f"steps[{i}]"
            _keys(step, {"text", "check"}, path, at)
            steps.append(
                {
                    "text": _text(step.get("text"), path, f"{at}.text"),
                    "check": _check(step.get("check"), path, f"{at}.check"),
                }
            )
        raw_rubric = data.get("rubric")
        if not isinstance(raw_rubric, list) or not raw_rubric:
            raise _fail(path, "rubric", "must be a non-empty list")
        for i, item in enumerate(raw_rubric):
            at = f"rubric[{i}]"
            _keys(item, {"id", "text", "points", "auto"}, path, at)
            item_id = item.get("id")
            if not isinstance(item_id, str) or not _ITEM_ID.match(item_id):
                raise _fail(
                    path,
                    f"{at}.id",
                    "must be lower-case letters, digits and underscores",
                )
            if any(existing["id"] == item_id for existing in rubric):
                raise _fail(path, f"{at}.id", f"duplicate rubric id {item_id!r}")
            points = item.get("points")
            if type(points) is not int or points <= 0:
                raise _fail(path, f"{at}.points", "must be a positive integer")
            rubric.append(
                {
                    "id": item_id,
                    "text": _text(item.get("text"), path, f"{at}.text"),
                    "points": points,
                    "auto": _auto(
                        item["auto"], path, f"{at}.auto", has_demo=spec is not None
                    )
                    if "auto" in item
                    else None,
                }
            )
    demo = None
    if spec is not None and "scripted" in spec:
        demo = compile_keyframes(spec["scripted"])
    elif spec is not None:
        demo_path = Path(demos_dir) / f"{spec['recorded']}.jsonl"
        if not demo_path.is_file():
            raise _fail(
                path,
                "demo.recorded",
                f"no recorded demo {spec['recorded']!r} in {demos_dir}",
            )
        try:
            demo = load_recorded_demo(demo_path)
        except ValueError as error:
            raise _fail(path, "demo.recorded", str(error)) from error
    return Exercise(
        id=exercise_id,
        course=_text(data.get("course"), path, "course"),
        order=order,
        title=_text(data.get("title"), path, "title"),
        brief=_text(data.get("brief"), path, "brief"),
        time_limit_s=time_limit,
        demo_only=demo_only,
        steps=tuple(steps),
        rubric=tuple(rubric),
        demo_spec=spec,
        demo=demo,
        path=path,
        scene=_text(data["scene"], path, "scene") if "scene" in data else None,
    )


def check_tools(exercise: Exercise, instruments: Mapping[str, Instrument]) -> None:
    """Reject steps or scripted demo keyframes that use a tool an instrument lacks.

    Opening jaws or switching energy off is always allowed; closing jaws, energy
    on and the matching hold steps need the tool on every side they name.

    Raises:
        ValueError: ``"<file>: <field>: the right instrument (Hook) has no jaws"``.
    """

    def need(side: str, tool: str, field_name: str) -> None:
        for one in SIDES if side == "both" else (side,):
            if tool not in instruments[one].tools:
                raise _fail(
                    exercise.path,
                    field_name,
                    f"the {instruments[one].label(one)} has no {tool}",
                )

    for i, step in enumerate(exercise.steps):
        for kind, tool in (("jaws_closed", "jaws"), ("jaws_open", "jaws"), ("energy", "energy")):
            if kind in step["check"]:
                need(step["check"][kind]["side"], tool, f"steps[{i}].check.{kind}")
    active = "right"  # compile_keyframes: a keyframe without side acts on the active one
    for i, keyframe in enumerate((exercise.demo_spec or {}).get("scripted", ())):
        side = keyframe.get("side", active)
        if side in SIDES:
            active = side
        if keyframe.get("jaws") == "close":
            need(side, "jaws", f"demo.scripted[{i}].jaws")
        if keyframe.get("energy") == "on":
            need(side, "energy", f"demo.scripted[{i}].energy")


def load_courses(
    directory: Path,
    demos_dir: Path,
    instruments: Mapping[str, Instrument] = DEFAULT_INSTRUMENTS,
) -> dict[str, Exercise]:
    """Load every ``<directory>/<course-folder>/*.yaml`` exercise, keyed by id in file order.

    Raises:
        ValueError: The directory is missing or empty, an exercise is invalid or
            uses a tool an instrument lacks, or two share an id.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"courses.dir: {directory} is not a directory")
    files = sorted(directory.glob("*/*.yaml"))
    if not files:
        raise ValueError(
            f"courses.dir: no exercise files in {directory}/<course>/*.yaml"
        )
    exercises: dict[str, Exercise] = {}
    for path in files:
        exercise = load_exercise(path, Path(demos_dir))
        check_tools(exercise, instruments)
        if exercise.id in exercises:
            raise _fail(
                path,
                "id",
                f"duplicate id {exercise.id!r} (also in {exercises[exercise.id].path})",
            )
        exercises[exercise.id] = exercise
    return exercises


def courses_json(exercises: Mapping[str, Exercise]) -> list[dict[str, Any]]:
    """Group exercises by course (in file order), each course sorted by ``order`` then id."""
    courses: dict[str, list[Exercise]] = {}
    for exercise in exercises.values():
        courses.setdefault(exercise.course, []).append(exercise)
    return [
        {
            "name": name,
            "exercises": [
                e.to_json() for e in sorted(items, key=lambda e: (e.order, e.id))
            ],
        }
        for name, items in courses.items()
    ]


# -- demos ---------------------------------------------------------------------------


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _axis_angle(axis: Sequence[float], angle: float) -> list[float]:
    s = math.sin(angle / 2)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)]


def _normalize(q: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def _unit_quaternion(values: Any) -> list[float]:
    """Return ``values`` as a unit quaternion.

    Raises:
        ValueError: ``values`` is not four finite numbers of non-zero length.
    """
    q = [float(v) for v in values]
    length = math.hypot(*q)
    if len(q) != 4 or not math.isfinite(length) or length < 1e-9:
        raise ValueError(
            "orientation must be a non-zero quaternion of four finite numbers"
        )
    return [v / length for v in q]


def _nlerp(a: Sequence[float], b: Sequence[float], u: float) -> list[float]:
    sign = -1.0 if sum(x * y for x, y in zip(a, b)) < 0 else 1.0
    return _normalize([x + (sign * y - x) * u for x, y in zip(a, b)])


def _frame_count(seconds: float) -> int:
    return max(1, math.ceil(seconds * FPS - 1e-9))


def _demo_frame(
    index: int, active: str, pose: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    frame: dict[str, Any] = {"t_s": round(index / FPS, 4), "active": active}
    for side in SIDES:
        frame[side] = {
            "position": [round(v, 6) for v in pose[side]["position"]],
            "orientation": [round(v, 6) for v in pose[side]["orientation"]],
            "trigger": float(pose[side]["trigger"]),
            "grip": float(pose[side]["grip"]),
        }
    return frame


def compile_keyframes(keyframes: Sequence[Mapping[str, Any]]) -> Demo:
    """Compile validated scripted keyframes into :data:`FPS` frames starting at the home poses.

    Each keyframe takes at least one frame: ``wait_s`` and motion (``move_to``,
    ``rotate_by``, ``home``; ``over_s`` default 1 s) span their duration with
    smoothstep easing, while ``jaws`` and ``energy`` switch on the next frame.
    ``jaws``/``energy`` keyframes before anything else set the first frame instead,
    so a demo on a scene that starts gripping never opens the jaws. A keyframe
    without ``side`` acts on the active instrument; a left or right ``side`` makes
    that instrument active.
    """
    pose = {
        side: {
            "position": list(HOME[side]),
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "trigger": 0.0,
            "grip": 0.0,
        }
        for side in SIDES
    }
    active = "right"
    frames = [_demo_frame(0, active, pose)]
    markers: list[dict[str, Any]] = []
    for keyframe in keyframes:
        if "caption" in keyframe:
            markers.append({"t_s": frames[-1]["t_s"], "caption": keyframe["caption"]})
        side = keyframe.get("side", active)
        if side in SIDES:
            active = side
        sides = SIDES if side == "both" else (side,)
        motion: dict[
            str, tuple[list[float], list[float], list[float], list[float]]
        ] = {}
        count = 1
        if "wait_s" in keyframe:
            count = _frame_count(keyframe["wait_s"])
        elif "jaws" in keyframe or "energy" in keyframe:
            for s in sides:
                if "jaws" in keyframe:
                    pose[s]["trigger"] = 1.0 if keyframe["jaws"] == "close" else 0.0
                else:
                    pose[s]["grip"] = 1.0 if keyframe["energy"] == "on" else 0.0
            if len(frames) == 1:  # leading: the starting state, not a step
                frames[0] = _demo_frame(0, active, pose)
                continue
        else:
            count = _frame_count(keyframe.get("over_s", 1.0))
            for s in sides:
                p0, q0 = list(pose[s]["position"]), list(pose[s]["orientation"])
                if "move_to" in keyframe:
                    p1, q1 = list(keyframe["move_to"]), q0
                elif "rotate_by" in keyframe:
                    yaw, pitch, roll = keyframe["rotate_by"]
                    turn = _quat_mul(
                        _quat_mul(
                            _axis_angle((0, 1, 0), yaw), _axis_angle((1, 0, 0), pitch)
                        ),
                        _axis_angle((0, 0, 1), roll),
                    )
                    p1, q1 = p0, _normalize(_quat_mul(q0, turn))
                else:  # home
                    p1, q1 = list(HOME[s]), [0.0, 0.0, 0.0, 1.0]
                motion[s] = (p0, p1, q0, q1)
        for i in range(1, count + 1):
            u = (i / count) ** 2 * (
                3 - 2 * i / count
            )  # smoothstep: zero speed at both ends
            for s, (p0, p1, q0, q1) in motion.items():
                pose[s]["position"] = [a + (b - a) * u for a, b in zip(p0, p1)]
                pose[s]["orientation"] = _nlerp(q0, q1, u)
            frames.append(_demo_frame(len(frames), active, pose))
    return Demo("scripted", None, frames, markers)


def compile_demo(exercise: Exercise) -> Demo | None:
    """Return the exercise's demo frames and markers (compiled once, at load time), or None."""
    return exercise.demo


def load_recorded_demo(path: Path) -> Demo:
    """Load ``<name>.jsonl`` (a ``{name, source, created}`` header line, then controller frames).

    Frames are resampled to :data:`FPS` by linear position and normalized-lerp
    orientation interpolation; trigger and grip hold their last value. Gaps (a
    pause or incident) play back as at most 1 s. The active instrument is the
    one that moved more since the previous frame (right until one moves).

    Raises:
        ValueError: The file cannot be read or is not a recorded demo (including a
            tracked controller whose orientation is not a non-zero quaternion).
    """
    path = Path(path)
    try:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        header = json.loads(lines[0]) if lines else None
        samples = [json.loads(line) for line in lines[1:]]
        if (
            not isinstance(header, dict)
            or not isinstance(header.get("name"), str)
            or not samples
        ):
            raise ValueError("missing header or frames")
        times: list[float] = []
        poses: list[dict[str, dict[str, Any]]] = []
        last = {
            side: {
                "position": list(HOME[side]),
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "trigger": 0.0,
                "grip": 0.0,
            }
            for side in SIDES
        }
        t, previous_ns = 0.0, None
        for sample in samples:
            ns = int(sample["client_capture_ns"])
            if previous_ns is not None:
                t += min(max(ns - previous_ns, 0) / 1e9, _GAP_CAP_S)
            previous_ns = ns
            for side in SIDES:
                controller = sample.get(side)
                if isinstance(controller, dict) and controller.get("tracked", True):
                    last[side] = {
                        "position": [float(v) for v in controller["position"]],
                        "orientation": _unit_quaternion(controller["orientation"]),
                        "trigger": float(controller.get("trigger", 0.0)),
                        "grip": float(controller.get("grip", 0.0)),
                    }
            times.append(t)
            poses.append(copy.deepcopy(last))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        ArithmeticError,
    ) as error:
        raise ValueError(f"{path.name} is not a recorded demo ({error})") from error
    frames, active, previous = [], "right", None
    for index in range(int(times[-1] * FPS + 1e-9) + 1):
        t = index / FPS
        j = max(0, bisect.bisect_right(times, t) - 1)
        k = min(j + 1, len(times) - 1)
        u = (t - times[j]) / (times[k] - times[j]) if times[k] > times[j] else 0.0
        u = min(1.0, max(0.0, u))
        pose = {
            side: {
                "position": [
                    a + (b - a) * u
                    for a, b in zip(
                        poses[j][side]["position"], poses[k][side]["position"]
                    )
                ],
                "orientation": _nlerp(
                    poses[j][side]["orientation"], poses[k][side]["orientation"], u
                ),
                "trigger": poses[j][side]["trigger"],
                "grip": poses[j][side]["grip"],
            }
            for side in SIDES
        }
        if previous is not None:
            moved = {
                s: math.dist(pose[s]["position"], previous[s]["position"])
                for s in SIDES
            }
            mover = max(SIDES, key=lambda s: moved[s])
            if moved[mover] > 1e-4:
                active = mover
        frames.append(_demo_frame(index, active, pose))
        previous = pose
    return Demo(
        "recorded",
        path.stem,
        frames,
        [{"t_s": 0.0, "caption": header["name"]}],
        dict(header),
    )


def list_demos(demos_dir: Path | None) -> list[dict[str, Any]]:
    """Return ``[{slug, name, source, created, duration_s}]`` of the readable recorded demos, newest first."""
    directory = Path(demos_dir) if demos_dir is not None else None
    if directory is None or not directory.is_dir():
        return []
    demos = []
    for path in sorted(directory.glob("*.jsonl")):
        if not SLUG.match(path.stem):
            continue
        try:
            demo = load_recorded_demo(path)
        except ValueError:
            continue
        info = demo.info or {}
        demos.append(
            {
                "slug": path.stem,
                "name": info.get("name"),
                "source": info.get("source"),
                "created": info.get("created"),
                "duration_s": demo.duration_s,
            }
        )
    return sorted(demos, key=lambda d: str(d["created"] or ""), reverse=True)


def demo_slug(name: str) -> str:
    """Return the file name stem for a demo name: lower-case ASCII letters and digits joined by hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60].rstrip("-")


def save_demo(
    demos_dir: Path,
    name: Any,
    session_dir: Path,
    session_id: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Copy an attempt's ``controller.jsonl`` to ``demos_dir/<slug>.jsonl`` under a header line.

    Returns:
        ``{slug, name, source, created}``.

    Raises:
        ValueError: The name is empty, too long or has no letter or digit, or the attempt has no frames.
        FileExistsError: A demo with the same slug exists.
        OSError: The file could not be written (a partial file is removed).
    """
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 60:
        raise ValueError("name must be 1 to 60 characters")
    slug = demo_slug(name)
    if not SLUG.match(slug):
        raise ValueError("name needs at least one letter or digit")
    try:
        text = (Path(session_dir) / "controller.jsonl").read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("this attempt has no controller frames")
    info = {
        "name": " ".join(name.split()),
        "source": session_id,
        "created": (now or datetime.now()).isoformat(timespec="seconds"),
    }
    directory = Path(demos_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.jsonl"
    try:
        with path.open(
            "x", encoding="utf-8"
        ) as file:  # "x": a same-slug demo is never overwritten
            file.write("".join(f"{line}\n" for line in [json.dumps(info), *lines]))
    except FileExistsError:
        raise
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return {"slug": slug, **info}


# -- scoring -------------------------------------------------------------------------


def path_deviation_m(
    attempt_frames: Sequence[Mapping[str, Any]],
    demo_frames: Sequence[Mapping[str, Any]],
) -> float | None:
    """Mean over attempt frames and sides of the distance to the nearest demo point of that side.

    Frames are ``{side: {"position": [x, y, z], "tracked"?: bool}}``; a missing or
    untracked side is skipped. Returns None without demo frames or attempt points.
    """
    if not demo_frames:
        return None
    distances = []
    for side in SIDES:
        demo = np.array(
            [
                f[side]["position"]
                for f in demo_frames
                if isinstance(f.get(side), Mapping)
            ],
            dtype=np.float64,
        )
        points = np.array(
            [
                f[side]["position"]
                for f in attempt_frames
                if isinstance(f.get(side), Mapping) and f[side].get("tracked", True)
            ],
            dtype=np.float64,
        )
        if not len(demo) or not len(points):
            continue
        for start in range(0, len(points), 512):  # bounded memory for long attempts
            block = points[start : start + 512]
            distances.append(
                np.linalg.norm(block[:, None, :] - demo[None, :, :], axis=2).min(axis=1)
            )
    return float(np.concatenate(distances).mean()) if distances else None


def jerk_rms(frames: Sequence[Mapping[str, Any]]) -> float | None:
    """Root-mean-square jerk (m/s³) of both instruments; kept in reports, :func:`corrections` scores smoothness.

    Frames are ``{t_s, side: {"position": [x, y, z], "tracked"?: bool}}``. Jerk is the
    third finite difference of position over each run of four tracked samples spaced
    ``0 < dt <= GAP_S``; a gap or a repeated timestamp breaks the run. Returns None
    when no run is long enough.
    """
    # ponytail: raw finite differences amplify mouse and timestamp jitter; compare it only
    # with a demo measured the same way, and low-pass the positions if that proves too noisy.
    squares = []
    for side in SIDES:
        tracked = [
            f
            for f in frames
            if isinstance(f.get(side), Mapping) and f[side].get("tracked", True)
        ]
        if len(tracked) < 4:
            continue
        t = np.array([f["t_s"] for f in tracked], dtype=np.float64)
        p = np.array([f[side]["position"] for f in tracked], dtype=np.float64)
        dt = np.diff(t)
        ok = (dt > 0) & (dt <= GAP_S)
        valid = ok[:-2] & ok[1:-1] & ok[2:]
        if not valid.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.diff(p, axis=0) / dt[:, None]
            tv = (t[1:] + t[:-1]) / 2
            a = np.diff(v, axis=0) / np.diff(tv)[:, None]
            ta = (tv[1:] + tv[:-1]) / 2
            j = np.diff(a, axis=0) / np.diff(ta)[:, None]
        squares.append((j[valid] ** 2).sum(axis=1))
    return float(np.sqrt(np.concatenate(squares).mean())) if squares else None


SMOOTH_RATE_HZ = 30.0
"""Rate positions are resampled to before :func:`corrections`, so any input rate compares."""
SMOOTH_WINDOW = 5
"""Moving-average width in samples, applied twice (~4 Hz low-pass at 30 Hz): removes mouse jitter."""
MOVING_M_S = 0.005
"""Below this speed an instrument is resting; a peak must rise above it to count."""
PEAK_PROMINENCE_M_S = 0.01
"""A speed peak must stand this far above the lower of the dips on either side of it."""


def _speed_peaks(speed: np.ndarray) -> int:
    """Speed maxima above :data:`MOVING_M_S` that rise :data:`PEAK_PROMINENCE_M_S` over both neighbouring dips.

    Each candidate is compared with the lowest speed between it and the nearest higher
    maximum (or the run's end) on each side, the usual peak prominence.
    """
    rising = np.diff(speed) > 0
    maxima = [i for i in range(1, len(speed) - 1) if rising[i - 1] and not rising[i]]
    count = 0
    for i in maxima:
        if speed[i] < MOVING_M_S:
            continue
        left = i - 1
        while left > 0 and speed[left] <= speed[i]:
            left -= 1
        right = i + 1
        while right < len(speed) - 1 and speed[right] <= speed[i]:
            right += 1
        base = max(speed[left : i + 1].min(), speed[i : right + 1].min())
        if speed[i] - base >= PEAK_PROMINENCE_M_S:
            count += 1
    return count


def corrections(frames: Sequence[Mapping[str, Any]]) -> dict[str, float] | None:
    """Count the movements of both instruments: the report's smoothness measure.

    A smooth reach speeds up and slows down once; hesitant motion is a chain of small
    stop-and-go moves, each its own speed peak. Per side, runs of tracked samples spaced
    ``0 < dt <= GAP_S`` are resampled to :data:`SMOOTH_RATE_HZ`, low-passed, and their
    speed peaks counted (:func:`_speed_peaks`).

    Returns:
        ``{"count": peaks, "path_m": smoothed path length}``, or None when no run has
        at least one second of samples.
    """
    count, path, used = 0, 0.0, False
    step = 1.0 / SMOOTH_RATE_HZ
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    for side in SIDES:
        tracked = [
            f
            for f in frames
            if isinstance(f.get(side), Mapping) and f[side].get("tracked", True)
        ]
        if len(tracked) < 2:
            continue
        t = np.array([f["t_s"] for f in tracked], dtype=np.float64)
        p = np.array([f[side]["position"] for f in tracked], dtype=np.float64)
        breaks = np.flatnonzero((np.diff(t) <= 0) | (np.diff(t) > GAP_S)) + 1
        for run_t, run_p in zip(np.split(t, breaks), np.split(p, breaks)):
            if run_t[-1] - run_t[0] < 1.0:
                continue
            grid = np.arange(run_t[0], run_t[-1], step)
            pos = np.stack([np.interp(grid, run_t, run_p[:, k]) for k in range(3)], axis=1)
            for _ in range(2):
                padded = np.pad(pos, ((SMOOTH_WINDOW // 2,) * 2, (0, 0)), mode="edge")
                pos = np.stack(
                    [np.convolve(padded[:, k], kernel, mode="valid") for k in range(3)], axis=1
                )
            moves = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            count += _speed_peaks(moves / step)
            path += float(moves.sum())
            used = True
    return {"count": count, "path_m": round(path, 4)} if used else None


def _step_times(
    events: Sequence[Mapping[str, Any]], count: int
) -> list[dict[str, Any]]:
    """First ``step`` event per valid index, as ``[{index, t_s}]`` in index order.

    The time is the event's active time (its ``t_s``: seconds without pauses and safety stops), not its video time.
    """
    seen: dict[int, float] = {}
    for event in events:
        index = event.get("index")
        if (
            event.get("kind") == "step"
            and type(index) is int
            and 0 <= index < count
            and index not in seen
        ):
            seen[index] = round(float(event.get("t_s", 0.0)), 3)
    return [{"index": index, "t_s": t} for index, t in sorted(seen.items())]


def score(
    exercise: Exercise,
    summary: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    deviation: float | None,
) -> dict[str, Any]:
    """Score the rubric's automatic items.

    When the rubric has an ``all_steps`` item and the attempt did not complete
    every step, every other automatic item is gated: ``passed=False``,
    ``points=0`` and ``gated=True``, with its measured value kept. An attempt
    stopped early then scores only from instructor items.

    Args:
        summary: The recording summary: ``duration_s`` (active seconds, which ``max_time_s``
            limits, whatever the video length) and ``energy_s``.
        events: The attempt's events (``incident``, ``near_miss``, ``step`` with ``t_s``).
        deviation: :func:`path_deviation_m` against the demo, or None.

    Returns:
        ``{items: [{id, passed, value, points, gated}], auto_points, auto_max, instructor_max}``;
        instructor items have ``passed=None, value=None, points=0, gated=False``.
    """
    values = {
        "no_safety_stops": sum(e.get("kind") == "incident" for e in events),
        "max_time_s": float(summary.get("duration_s", 0.0)),
        "max_path_deviation_m": deviation,
        "max_energy_s": float(sum((summary.get("energy_s") or {}).values())),
        "max_near_misses": sum(e.get("kind") == "near_miss" for e in events),
        "all_steps": len(_step_times(events, len(exercise.steps))),
    }
    incomplete = values["all_steps"] < len(exercise.steps)
    gating = incomplete and any(
        item["auto"] is not None and "all_steps" in item["auto"]
        for item in exercise.rubric
    )
    items, auto_points, auto_max, instructor_max = [], 0, 0, 0
    for item in exercise.rubric:
        if item["auto"] is None:
            instructor_max += item["points"]
            items.append(
                {
                    "id": item["id"],
                    "passed": None,
                    "value": None,
                    "points": 0,
                    "gated": False,
                }
            )
            continue
        ((kind, limit),) = item["auto"].items()
        value = values[kind]
        if kind == "no_safety_stops":
            passed = value == 0
        elif kind == "all_steps":
            passed = value == len(exercise.steps)
        else:
            passed = value is not None and value <= limit
        gated = gating and kind != "all_steps"
        passed = passed and not gated
        points = item["points"] if passed else 0
        auto_points += points
        auto_max += item["points"]
        items.append(
            {
                "id": item["id"],
                "passed": passed,
                "value": None if value is None else round(value, 4),
                "points": points,
                "gated": gated,
            }
        )
    return {
        "items": items,
        "auto_points": auto_points,
        "auto_max": auto_max,
        "instructor_max": instructor_max,
    }


def build_report(
    exercise: Exercise,
    summary: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    attempt_frames: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return ``session.json``'s ``report`` for a finished exercise attempt."""
    deviation = (
        path_deviation_m(attempt_frames, exercise.demo.frames)
        if exercise.demo
        else None
    )
    steps = _step_times(events, len(exercise.steps))
    jerk = jerk_rms(attempt_frames)
    demo_jerk = jerk_rms(exercise.demo.frames) if exercise.demo else None
    moves = corrections(attempt_frames)
    demo_moves = corrections(exercise.demo.frames) if exercise.demo else None
    return {
        "steps": steps,
        "steps_done": len(steps),
        "near_misses": sum(e.get("kind") == "near_miss" for e in events),
        "path_deviation_m": None if deviation is None else round(deviation, 4),
        "jerk_m_s3": None if jerk is None else round(jerk, 3),
        "demo_jerk_m_s3": None if demo_jerk is None else round(demo_jerk, 3),
        "corrections": None if moves is None else moves["count"],
        "demo_corrections": None if demo_moves is None else demo_moves["count"],
        **score(exercise, summary, events, deviation),
    }


def validate_review(
    exercise: Exercise, payload: Any, *, now: datetime | None = None
) -> dict[str, Any]:
    """Validate ``{checked: [instructor item ids], notes}`` into the ``review.json`` content.

    Returns:
        ``{checked, notes, points, instructor_max, saved}``.

    Raises:
        ValueError: Not an object, an unknown or automatic item id, or notes over :data:`NOTES_MAX`.
    """
    if not isinstance(payload, dict):
        raise ValueError("expected {checked, notes}")
    checked = payload.get("checked", [])
    notes = payload.get("notes", "")
    if not isinstance(checked, list) or any(
        not isinstance(item, str) for item in checked
    ):
        raise ValueError("checked must be a list of rubric item ids")
    instructor = {
        item["id"]: item["points"] for item in exercise.rubric if item["auto"] is None
    }
    for item_id in checked:
        if item_id not in instructor:
            raise ValueError(f"not an instructor item: {item_id[:40]!r}")
    if not isinstance(notes, str) or len(notes) > NOTES_MAX:
        raise ValueError(f"notes must be text of at most {NOTES_MAX} characters")
    unique = list(dict.fromkeys(checked))
    return {
        "checked": unique,
        "notes": notes,
        "points": sum(instructor[item_id] for item_id in unique),
        "instructor_max": sum(instructor.values()),
        "saved": (now or datetime.now()).isoformat(timespec="seconds"),
    }
