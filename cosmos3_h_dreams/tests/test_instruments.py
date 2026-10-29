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

"""Course files, demo compilation, path deviation and scoring."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from cosmos3_h_dreams.courses import load_courses
from cosmos3_h_dreams.headset_server import LatestHeadsetFrames
from cosmos3_h_dreams.instruments import (
    DEFAULT_INSTRUMENTS,
    instruments_json,
    parse_instruments,
)

pytestmark = pytest.mark.ci_cpu
PACKAGE = Path(__file__).resolve().parents[1]
GRASPER_HOOK = {
    "left": {"name": "Grasper", "tools": ["jaws"]},
    "right": {"name": "Hook", "tools": ["energy"]},
}
EXERCISE = {
    "id": "tool-test",
    "course": "Test course",
    "order": 1,
    "title": "Tool test",
    "brief": "Use a tool.",
    "demo": {"scripted": [{"side": "left", "jaws": "close"}, {"jaws": "open"}]},
    "steps": [
        {"text": "Grasp", "check": {"jaws_closed": {"side": "left", "hold_s": 1}}}
    ],
    "rubric": [{"id": "smooth", "text": "Smooth", "points": 10}],
}


def test_parse_instruments_defaults_and_errors():
    assert parse_instruments(None) == DEFAULT_INSTRUMENTS
    parsed = parse_instruments(GRASPER_HOOK)
    assert instruments_json(parsed) == GRASPER_HOOK
    unnamed = parse_instruments(
        {"left": {"tools": ["energy", "jaws"]}, "right": {"tools": []}}
    )
    assert instruments_json(unnamed) == {
        "left": {"name": "Left", "tools": ["jaws", "energy"]},
        "right": {"name": "Right", "tools": []},
    }
    for raw, match in [
        ({"left": GRASPER_HOOK["left"]}, "exactly the keys left and right"),
        ({**GRASPER_HOOK, "right": {"tools": ["laser"]}}, r"right\.tools"),
        ({**GRASPER_HOOK, "right": {"tools": ["jaws", "jaws"]}}, r"right\.tools"),
        ({**GRASPER_HOOK, "left": {"name": "", "tools": []}}, r"left\.name"),
        ({**GRASPER_HOOK, "left": {"tools": [], "colour": "red"}}, "allowed keys"),
    ]:
        with pytest.raises(ValueError, match=match):
            parse_instruments(raw)


def test_frames_drop_controls_for_missing_tools():
    frames = LatestHeadsetFrames(fps=None, instruments=parse_instruments(GRASPER_HOOK))
    pressed = {
        "position": [0.0, 1.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "trigger": 1.0,
        "grip": 1.0,
    }
    assert frames.push(
        {"type": "xr_frame", "controllers": {"left": pressed, "right": pressed}}
    )
    frame = frames.poll().frame
    assert (frame["left"]["trigger"], frame["left"]["grip"]) == (1.0, 0.0)
    assert (frame["right"]["trigger"], frame["right"]["grip"]) == (0.0, 1.0)


def _write(folder: Path, data: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "01-tool-test.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: None, None),
        (
            lambda d: d["steps"][0]["check"]["jaws_closed"].update(side="right"),
            r"steps\[0\]\.check\.jaws_closed: the right instrument \(Hook\) has no jaws",
        ),
        (
            lambda d: d["demo"]["scripted"].insert(0, {"side": "both", "energy": "on"}),
            r"demo\.scripted\[0\]\.energy: the left instrument \(Grasper\) has no energy",
        ),
        (
            # A keyframe without side acts on the active instrument, right at the start.
            lambda d: d["demo"]["scripted"].insert(0, {"jaws": "close"}),
            r"demo\.scripted\[0\]\.jaws: the right instrument \(Hook\) has no jaws",
        ),
        (lambda d: d["demo"]["scripted"].insert(0, {"side": "right", "jaws": "open"}), None),
    ],
)
def test_courses_may_only_use_tools_the_instruments_have(tmp_path, change, match):
    data = copy.deepcopy(EXERCISE)
    change(data)
    path = _write(tmp_path / "course", data)
    instruments = parse_instruments(GRASPER_HOOK)
    if match is None:
        assert "tool-test" in load_courses(tmp_path, tmp_path / "demos", instruments)
        return
    with pytest.raises(ValueError, match=match) as error:
        load_courses(tmp_path, tmp_path / "demos", instruments)
    assert str(error.value).startswith(str(path))
    assert "tool-test" in load_courses(tmp_path, tmp_path / "demos")  # default: any tool


@pytest.mark.parametrize("config", ["training_ground.yaml", "laptop_rtx5090.yaml"])
def test_shipped_courses_fit_the_shipped_instruments(tmp_path, config):
    data = yaml.safe_load((PACKAGE / "configs" / config).read_text())
    instruments = parse_instruments(data["instruments"])
    assert instruments_json(instruments) == GRASPER_HOOK
    assert load_courses(PACKAGE / "courses", tmp_path / "demos", instruments)
