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

"""Session YAML with shared model options, Headset options, and named scenes."""

from __future__ import annotations

from dataclasses import fields
from glob import glob, has_magic
from math import isfinite
from pathlib import Path
from typing import Any, get_args, get_type_hints

import yaml

from .config import Cosmos3HDreamsConfig
from .instruments import parse_instruments
from .safety import SafetyLimits

REPO_ROOT: Path | None = None
"""Optional asset-root override; paths otherwise start at the launch directory."""

SCENE_FIELDS = {"initial_frame", "prompt", "domain_id", "normalizer", "actions"}


def load_session_config(path: Path) -> dict[str, Any]:
    """Load and validate a session file, resolving asset paths from the launch directory."""
    base = REPO_ROOT or Path.cwd()
    path = path.expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid session YAML {path}: {error}") from error
    _mapping(
        data,
        {
            "runtime",
            "headset",
            "scenes",
            "safety",
            "recording",
            "gpu_monitor",
            "courses",
            "instruments",
        },
        "session",
    )
    for section in ("safety", "recording", "gpu_monitor", "courses"):
        if section in data and not isinstance(data[section], dict | None):
            raise ValueError(f"{section} must be a mapping")
    runtime = data.setdefault("runtime", {})
    headset = data.setdefault("headset", {})
    model_fields = {field.name for field in fields(Cosmos3HDreamsConfig)}
    _mapping(runtime, (model_fields - SCENE_FIELDS) | {"preset"}, "runtime")
    _mapping(
        headset,
        {
            "preset",
            "host",
            "port",
            "cert",
            "key",
            "display_fps",
            "headset_input_fps",
            "overlap_playback",
            "print_actions",
        },
        "headset",
    )
    for section in (runtime, headset):
        if "preset" in section and not isinstance(section["preset"], str):
            raise ValueError("preset must be a string")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes must be a non-empty list")
    expanded = []
    for scene in scenes:
        _mapping(scene, SCENE_FIELDS | {"name", "label", "start_jaws_closed"}, "scene")
        initial_frame = scene.get("initial_frame")
        if isinstance(initial_frame, str) and has_magic(initial_frame):
            matches = sorted(
                match
                for match in glob(str(_path(initial_frame, base)), recursive=True)
                if Path(match).is_file()
            )
            if not matches:
                raise ValueError(
                    f"initial_frame pattern matches no files: {initial_frame}"
                )
            expanded.extend(
                {
                    **scene,
                    "initial_frame": match,
                    "name": Path(match).name,
                    "label": Path(match).name,
                }
                for match in matches
            )
        else:
            expanded.append(scene)
    scenes = data["scenes"] = expanded
    names = set()
    hints = get_type_hints(Cosmos3HDreamsConfig)
    for scene in scenes:
        name = scene.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("scene names must be non-empty and unique")
        names.add(name)
        if "initial_frame" not in scene:
            raise ValueError(f"scene {name!r} requires initial_frame")
        if (
            not isinstance(scene.setdefault("label", name), str)
            or not scene["label"].strip()
        ):
            raise ValueError(f"scene {name!r} requires a non-empty label")
        # Sides whose jaws are already closed on tissue in the frame; /train starts them closed.
        held = scene.setdefault("start_jaws_closed", [])
        if (
            not isinstance(held, list)
            or any(side not in ("left", "right") for side in held)
            or len(set(held)) != len(held)
        ):
            raise ValueError(
                f"scene {name!r}: start_jaws_closed must be a list of left and/or right"
            )
    for section in [runtime, *scenes]:
        for key, value in section.items():
            if key not in hints:
                continue
            expected = get_args(hints[key]) or (hints[key],)
            if value is None and type(None) in expected:
                continue
            if Path in expected:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{key} must be a path string")
                section[key] = _path(value, base)
            elif not any(
                type(value) is kind or (kind is float and type(value) is int)
                for kind in expected
            ):
                raise ValueError(f"Invalid type for {key}")
            elif float in expected and not isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
    for key, value in headset.items():
        if key in {"cert", "key"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"headset.{key} must be a path string")
            headset[key] = str(_path(value, base))
        elif key in {"display_fps", "port"} and type(value) is not int:
            raise ValueError(f"headset.{key} must be an integer")
        elif key == "headset_input_fps" and type(value) not in (float, int):
            raise ValueError("headset.headset_input_fps must be a number")
        elif key in {"overlap_playback", "print_actions"} and type(value) is not bool:
            raise ValueError(f"headset.{key} must be a boolean")
        elif key == "host" and not isinstance(value, str):
            raise ValueError(f"headset.{key} must be a string")
    data["safety"] = SafetyLimits.from_mapping(data.get("safety"))
    recording = data.get("recording")
    if recording is None:
        recording = {"dir": "recordings"}
    _mapping(recording, {"dir"}, "recording")
    if not isinstance(recording.get("dir"), str) or not recording["dir"].strip():
        raise ValueError("recording.dir must be a path string")
    data["recording"] = {"dir": _path(recording["dir"], base)}
    data["gpu_monitor"] = _gpu_monitor(data.get("gpu_monitor") or {})
    courses = data.get("courses")
    data["courses"] = None if courses is None else _courses(courses, base)
    data["instruments"] = parse_instruments(data.get("instruments"))
    if "checkpoint" in runtime:
        checkpoint = runtime["checkpoint"]
        # Registry names and URIs retain their meaning; filesystem paths start at the repo root.
        if checkpoint.startswith((".", "/", "~")) or (
            "://" not in checkpoint
            and (
                "/" in checkpoint
                or Path(checkpoint).suffix in {".pt", ".pth", ".json", ".safetensors"}
            )
        ):
            runtime["checkpoint"] = str(_path(checkpoint, base))
    return data


def _gpu_monitor(block: dict[str, Any]) -> dict[str, Any]:
    _mapping(block, {"enabled", "period_s"}, "gpu_monitor")
    enabled = block.get("enabled", True)
    if type(enabled) is not bool:
        raise ValueError("gpu_monitor.enabled must be a boolean")
    period = block.get("period_s", 1.0)
    if type(period) not in (int, float) or not isfinite(period) or period <= 0:
        raise ValueError("gpu_monitor.period_s must be a finite positive number")
    return {"enabled": enabled, "period_s": float(period)}


def _courses(block: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve the optional ``courses`` block: ``dir`` (required), ``demos_dir`` and ``scene``.

    ``scene`` names the scene every exercise and course demo runs on; None means
    the first scene (checked against the scene names at server start).
    """
    _mapping(block, {"dir", "demos_dir", "scene"}, "courses")
    paths = {"dir": block.get("dir"), "demos_dir": block.get("demos_dir", "data_local/demos")}
    for key, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"courses.{key} must be a path string")
    scene = block.get("scene")
    if scene is not None and (not isinstance(scene, str) or not scene.strip()):
        raise ValueError("courses.scene must be a scene name")
    return {**{key: _path(value, base) for key, value in paths.items()}, "scene": scene}


def _mapping(value: Any, allowed: set[str], context: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a mapping with string keys")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unknown {context} options: {', '.join(sorted(unknown))}")


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()
