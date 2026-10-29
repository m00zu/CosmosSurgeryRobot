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
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import yaml

from cosmos3_h_dreams.courses import (
    FPS,
    HOME,
    build_report,
    compile_demo,
    compile_keyframes,
    corrections,
    jerk_rms,
    courses_json,
    demo_slug,
    list_demos,
    load_courses,
    load_exercise,
    load_recorded_demo,
    path_deviation_m,
    save_demo,
    score,
    validate_review,
)
from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits

pytestmark = pytest.mark.ci_cpu
COURSES = Path(__file__).resolve().parents[1] / "courses"
NOW = datetime(2026, 10, 27, 14, 30, 5)

BASE = {
    "id": "reach-test",
    "course": "Test course",
    "order": 1,
    "title": "Reach test",
    "brief": "Reach the target.",
    "demo": {
        "scripted": [{"side": "right", "move_to": [0.2, 1.0, -0.3], "over_s": 1.0}]
    },
    "steps": [
        {
            "text": "Reach",
            "check": {
                "reach": {"side": "right", "center": [0.2, 1.0, -0.3], "radius_m": 0.01}
            },
        }
    ],
    "rubric": [
        {
            "id": "no_stops",
            "text": "No stops",
            "points": 30,
            "auto": {"no_safety_stops": True},
        },
        {"id": "smooth", "text": "Smooth", "points": 10},
    ],
}


def _write(folder: Path, data: dict, name: str = "01-reach-test.yaml") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(yaml.safe_dump(data))
    return path


def _exercise(tmp_path: Path, data: dict):
    return load_exercise(_write(tmp_path / "course", data), tmp_path / "demos")


def _controller_frames(demo, stride: int):
    """Demo frames as the server's validated controller frames, keeping every ``stride``-th."""
    return [
        {
            **{side: {**frame[side], "tracked": True} for side in ("left", "right")},
            "client_capture_ns": round(frame["t_s"] * 1e9),
        }
        for frame in demo.frames[::stride]
    ]


def _guard_windows(demo, stride: int, rows: int = 4):
    """Check a demo in overlapping windows, as HeadsetActionSource does; return the results."""
    frames = _controller_frames(demo, stride)
    guard = SafetyGuard(SafetyLimits())
    return [
        guard.check(frames[start : start + rows + 1])
        for start in range(0, len(frames) - 1, rows)
    ]


def test_starter_course_loads(tmp_path):
    exercises = load_courses(COURSES, tmp_path / "demos")
    assert list(exercises) == [
        "instrument-basics",
        "grasp-and-hold",
        "controlled-energy",
        "safety-stop",
        "drag-the-gallbladder",
    ]
    drag = exercises["drag-the-gallbladder"]
    assert drag.scene == "chole-15" and exercises["grasp-and-hold"].scene is None
    assert drag.to_json()["scene"] == "chole-15"
    # Its leading "jaws: close" is the starting state: the demo never opens the jaws before "Let go".
    assert drag.demo.frames[0]["left"]["trigger"] == 1.0
    stop = exercises["safety-stop"]
    assert stop.demo_only and stop.steps == () and stop.rubric == ()
    assert all(
        compile_demo(e) is e.demo and e.demo.source == "scripted"
        for e in exercises.values()
    )
    (course,) = courses_json(exercises)
    assert course["name"] == "Starter course"
    assert [e["id"] for e in course["exercises"]] == list(exercises)
    basics = course["exercises"][0]
    assert (
        basics["step_count"] == 4
        and basics["has_demo"]
        and basics["demo_source"] == "scripted"
    )
    assert sum(item["points"] for item in basics["rubric"]) == 100
    assert basics["rubric"][0]["auto"] == {"all_steps": True}
    assert basics["rubric"][4] == {
        "id": "smooth",
        "text": "Smooth movement",
        "points": 20,
        "auto": None,
    }
    assert basics["demo_duration_s"] == pytest.approx(
        exercises["instrument-basics"].demo.duration_s
    )
    json.dumps(course)  # plain JSON values only


@pytest.mark.parametrize("stride", [1, 3])  # 30 Hz input, and 10 Hz
def test_only_the_safety_stop_demo_trips_the_guard(tmp_path, stride):
    exercises = load_courses(COURSES, tmp_path / "demos")
    for exercise_id, exercise in exercises.items():
        results = _guard_windows(exercise.demo, stride)
        if exercise_id == "safety-stop":
            tripped = [v.rule for result in results for v in result]
            assert "speed" in tripped and "acceleration" in tripped
        else:
            assert all(result == [] for result in results), exercise_id


def test_compile_keyframes_timing_easing_markers_and_home():
    demo = compile_keyframes(
        [
            {"wait_s": 1.0},
            {
                "caption": "Reach",
                "side": "right",
                "move_to": [0.25, 1.0, -0.3],
                "over_s": 1.0,
            },
            {"side": "left", "jaws": "close"},
            {"side": "right", "rotate_by": [0.0, 0.0, 0.5], "over_s": 0.5},
            {"caption": "Home", "side": "both", "home": True, "over_s": 0.5},
        ]
    )
    frames = demo.frames
    assert len(frames) == 1 + 30 + 30 + 1 + 15 + 15
    assert [f["t_s"] for f in frames[:3]] == [0.0, round(1 / FPS, 4), round(2 / FPS, 4)]
    assert demo.duration_s == pytest.approx(91 / FPS, abs=1e-4)
    assert frames[0]["left"]["position"] == list(HOME["left"])
    assert frames[30]["right"]["position"] == list(
        HOME["right"]
    )  # the wait holds still
    assert frames[31]["right"]["position"][0] - 0.15 < 0.001  # smoothstep starts slowly
    assert frames[45]["right"]["position"][0] == pytest.approx(
        0.2
    )  # half way at half time
    assert frames[60]["right"]["position"][0] == pytest.approx(0.25)
    assert (frames[61]["active"], frames[61]["left"]["trigger"]) == ("left", 1.0)
    roll = frames[76]["right"]["orientation"]
    assert roll == pytest.approx([0.0, 0.0, math.sin(0.25), math.cos(0.25)], abs=1e-6)
    for side in ("left", "right"):
        assert frames[-1][side]["position"] == pytest.approx(list(HOME[side]))
        assert frames[-1][side]["orientation"] == pytest.approx([0, 0, 0, 1])
    assert demo.markers == [
        {"t_s": 1.0, "caption": "Reach"},
        {"t_s": round(76 / FPS, 4), "caption": "Home"},
    ]
    assert frames[-1]["active"] == "right"  # "both" keeps the active instrument


def _mutate(path, value):
    def apply(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return apply


_DELETE = object()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_mutate(["colour"], "red"), r"exercise: unknown key 'colour'"),
        (
            _mutate(["steps", 0, "check", "reach", "radius"], 0.1),
            r"steps\[0\]\.check\.reach: unknown key 'radius'",
        ),
        (
            _mutate(["rubric", 0, "points"], 0),
            r"rubric\[0\]\.points: must be a positive integer",
        ),
        (
            _mutate(["demo"], {"recorded": "missing-demo"}),
            r"demo\.recorded: no recorded demo 'missing-demo'",
        ),
        (
            _mutate(["demo", "scripted", 0, "jaws"], "close"),
            r"demo\.scripted\[0\]: needs exactly one of",
        ),
        (
            _mutate(["steps", 0, "check"], {"active": "middle"}),
            r"steps\[0\]\.check\.active: must be one of",
        ),
        (
            _mutate(["rubric", 1, "auto"], {"max_path_deviation_m": 0.03}),
            None,
        ),  # valid: there is a demo
        (_mutate(["demo_only"], True), r"steps: not allowed when demo_only is true"),
        (
            _mutate(["rubric", 1, "id"], "no_stops"),
            r"rubric\[1\]\.id: duplicate rubric id",
        ),
        (
            _mutate(["demo", "scripted", 0, "over_s"], -1),
            r"demo\.scripted\[0\]\.over_s: must be a positive",
        ),
        (_mutate(["id"], "Bad Id"), r"id: must be lower-case"),
    ],
)
def test_exercise_validation_names_the_file_and_field(tmp_path, change, match):
    data = copy.deepcopy(BASE)
    change(data)
    path = _write(tmp_path / "course", data)
    if match is None:
        assert load_courses(tmp_path, tmp_path / "demos")["reach-test"].rubric[1][
            "auto"
        ]
        return
    with pytest.raises(ValueError, match=match) as error:
        load_courses(tmp_path, tmp_path / "demos")
    assert str(error.value).startswith(str(path))


def test_course_directory_errors(tmp_path):
    with pytest.raises(ValueError, match="is not a directory"):
        load_courses(tmp_path / "absent", tmp_path)
    with pytest.raises(ValueError, match="no exercise files"):
        load_courses(tmp_path, tmp_path)
    _write(tmp_path / "a", BASE)
    _write(tmp_path / "b", BASE)
    with pytest.raises(
        ValueError, match=r"b/01-reach-test.yaml: id: duplicate id 'reach-test'"
    ):
        load_courses(tmp_path, tmp_path)
    (tmp_path / "b" / "01-reach-test.yaml").write_text("id: [unclosed")
    with pytest.raises(ValueError, match="cannot read"):
        load_courses(tmp_path, tmp_path)
    no_demo = copy.deepcopy(BASE)
    del no_demo["demo"]
    no_demo["rubric"][1]["auto"] = {"max_path_deviation_m": 0.03}
    (tmp_path / "b" / "01-reach-test.yaml").unlink()
    _write(tmp_path / "a", no_demo)
    with pytest.raises(
        ValueError, match=r"rubric\[1\]\.auto\.max_path_deviation_m: needs a demo"
    ):
        load_courses(tmp_path, tmp_path)


def _session_with_frames(folder: Path, count: int) -> Path:
    folder.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "left": {
                    "position": [-0.15, 1.0, -0.3],
                    "orientation": [0, 0, 0, 1],
                    "trigger": 0.0,
                    "grip": 0.0,
                },
                "right": {
                    "position": [0.15 + 0.01 * i, 1.0, -0.3],
                    "orientation": [0, 0, 0, 1],
                    "trigger": 0.0,
                    "grip": 1.0 if i >= 5 else 0.0,
                },
                "client_capture_ns": 1_000_000_000
                + i * 100_000_000
                + (5_000_000_000 if i == count - 1 else 0),
            }
        )
        for i in range(count)
    ]
    (folder / "controller.jsonl").write_text("".join(f"{line}\n" for line in lines))
    return folder


def test_recorded_demos_save_list_compile_and_back_an_exercise(tmp_path):
    demos = tmp_path / "demos"
    session = _session_with_frames(tmp_path / "rec" / "20261027-143005_Dr_Lin", 11)
    saved = save_demo(demos, "  My Demo! ", session, session.name, now=NOW)
    assert saved == {
        "slug": "my-demo",
        "name": "My Demo!",
        "source": session.name,
        "created": "2026-10-27T14:30:05",
    }
    assert json.loads((demos / "my-demo.jsonl").read_text().splitlines()[0]) == {
        "name": "My Demo!",
        "source": session.name,
        "created": "2026-10-27T14:30:05",
    }
    with pytest.raises(FileExistsError):
        save_demo(demos, "my demo", session, session.name)
    with pytest.raises(ValueError, match="letter or digit"):
        save_demo(demos, "!!!", session, session.name)
    with pytest.raises(ValueError, match="1 to 60"):
        save_demo(demos, "x" * 61, session, session.name)
    with pytest.raises(ValueError, match="no controller frames"):
        save_demo(demos, "empty", (tmp_path / "rec" / "none"), "none")
    assert demo_slug("Two-handed  hand-off") == "two-handed-hand-off"

    demo = load_recorded_demo(demos / "my-demo.jsonl")
    # 10 gaps of 0.1 s, the last one (5.1 s) capped at 1 s: 1.9 s at 30 Hz.
    assert demo.duration_s == pytest.approx(1.9, abs=1e-3) and len(demo.frames) == 58
    assert demo.frames[3]["right"]["position"][0] == pytest.approx(
        0.16, abs=1e-6
    )  # t=0.1 s: second sample
    assert demo.frames[1]["right"]["position"][0] == pytest.approx(
        0.15 + 0.01 / 3, abs=1e-6
    )
    assert (
        demo.frames[15]["right"]["grip"] == 1.0
        and demo.frames[14]["right"]["grip"] == 0.0
    )
    assert demo.frames[5]["active"] == "right" and demo.markers == [
        {"t_s": 0.0, "caption": "My Demo!"}
    ]
    assert demo.to_json()["title"] == "My Demo!"
    assert list_demos(demos) == [
        {
            "slug": "my-demo",
            "name": "My Demo!",
            "source": session.name,
            "created": "2026-10-27T14:30:05",
            "duration_s": demo.duration_s,
        }
    ]
    (demos / "broken.jsonl").write_text("not json\n")
    assert [d["slug"] for d in list_demos(demos)] == ["my-demo"]
    assert list_demos(tmp_path / "absent") == [] and list_demos(None) == []
    with pytest.raises(ValueError, match="not a recorded demo"):
        load_recorded_demo(demos / "broken.jsonl")

    data = copy.deepcopy(BASE)
    data["demo"] = {"recorded": "my-demo"}
    exercise = load_exercise(_write(tmp_path / "course", data), demos)
    assert (
        exercise.demo.source == "recorded"
        and exercise.to_json()["demo_name"] == "my-demo"
    )


@pytest.mark.parametrize(
    "orientation",
    [[0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, float("nan")], "wxyz"],
    ids=["all-zero", "short", "long", "nan", "text"],
)
def test_a_demo_file_with_a_bad_orientation_is_skipped_not_fatal(tmp_path, orientation):
    demos = tmp_path / "demos"
    session = _session_with_frames(tmp_path / "rec" / "20261027-143005_Dr_Lin", 11)
    save_demo(demos, "good demo", session, session.name, now=NOW)
    lines = (demos / "good-demo.jsonl").read_text().splitlines()
    header, samples = lines[0], [json.loads(line) for line in lines[1:]]
    samples[4]["right"]["orientation"] = orientation
    (demos / "hand-edited.jsonl").write_text(
        "".join(f"{line}\n" for line in [header, *map(json.dumps, samples)])
    )
    with pytest.raises(ValueError, match="not a recorded demo"):
        load_recorded_demo(demos / "hand-edited.jsonl")
    assert [d["slug"] for d in list_demos(demos)] == ["good-demo"]


def test_a_demo_file_with_an_unreadable_timestamp_is_skipped_not_fatal(tmp_path):
    demos = tmp_path / "demos"
    session = _session_with_frames(tmp_path / "rec" / "20261027-143005_Dr_Lin", 11)
    save_demo(demos, "good demo", session, session.name, now=NOW)
    lines = (demos / "good-demo.jsonl").read_text().splitlines()
    samples = [json.loads(line) for line in lines[1:]]
    samples[2]["client_capture_ns"] = float("inf")
    (demos / "hand-edited.jsonl").write_text(
        "".join(f"{line}\n" for line in [lines[0], *map(json.dumps, samples)])
    )
    assert [d["slug"] for d in list_demos(demos)] == ["good-demo"]


def test_path_deviation_is_the_mean_nearest_demo_distance():
    line = [
        {
            "left": {"position": [-0.15, 1.0, -0.3]},
            "right": {"position": [0.15 + 0.001 * i, 1.0, -0.3]},
        }
        for i in range(101)
    ]
    attempt = [
        {
            "left": {"position": [-0.15, 1.02, -0.3]},
            "right": {"position": [0.2, 1.01, -0.3]},
        },
        {
            "left": {"position": [-0.15, 1.02, -0.3], "tracked": False},
            "right": {"position": [0.22, 1.01, -0.3]},
        },
        {"right": {"position": [0.15, 1.01, -0.3]}},
    ]
    # left 0.02 (one tracked frame), right 0.01 three times: (0.02 + 3 * 0.01) / 4.
    assert path_deviation_m(attempt, line) == pytest.approx(0.0125)
    assert path_deviation_m(attempt, []) is None
    assert path_deviation_m([], line) is None


def _reach_frames(hz, segments, pause_s=0.6):
    """``segments`` 5 cm minimum-jerk reaches along x, each 1 s, with pauses between, at ``hz``."""
    frames, x, t = [], 0.0, 0.0
    for _ in range(segments):
        for tau in np.arange(0, 1, 1 / hz):
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            frames.append({"t_s": t + tau, "right": {"position": [x + 0.05 * s, 1.0, -0.3]}})
        x, t = x + 0.05, t + 1
        for tau in np.arange(0, pause_s, 1 / hz):
            frames.append({"t_s": t + tau, "right": {"position": [x, 1.0, -0.3]}})
        t += pause_s
    return frames


def test_corrections_count_speed_peaks_at_any_rate():
    one, three = corrections(_reach_frames(30, 1)), corrections(_reach_frames(30, 3))
    assert one["count"] == 1 and three["count"] == 3
    assert three["path_m"] == pytest.approx(0.15, abs=0.005)
    assert corrections(_reach_frames(60, 3))["count"] == 3  # the input rate does not matter
    assert corrections(_reach_frames(24, 3))["count"] == 3
    # Mouse-like jitter of 0.2 mm on a held pose is not a movement.
    rng = np.random.default_rng(0)
    still = [{"t_s": i / 30, "right": {"position": [float(rng.normal(0, 2e-4)), 1.0, -0.3]}} for i in range(90)]
    assert corrections(still)["count"] == 0
    # A gap splits runs; runs under one second and untracked samples are ignored.
    assert corrections(_reach_frames(30, 1)[:20]) is None
    untracked = [{**f, "right": {**f["right"], "tracked": False}} for f in _reach_frames(30, 2)]
    assert corrections(untracked) is None


def test_jerk_rms():
    # x = t³ at 30 Hz: third differences are exact, so jerk is 6 m/s³ throughout.
    cubic = [{"t_s": i / 30, "right": {"position": [(i / 30) ** 3, 1.0, -0.3]}} for i in range(10)]
    assert jerk_rms(cubic) == pytest.approx(6.0)
    steady = [{"t_s": i / 30, "left": {"position": [0.1 * i / 30, 1.0, -0.3]}} for i in range(10)]
    assert jerk_rms(steady) == pytest.approx(0.0, abs=1e-6)
    # A gap longer than GAP_S breaks every run of four; untracked samples are skipped.
    gappy = [{"t_s": t, "right": {"position": [t, 1.0, -0.3]}} for t in (0, 0.03, 0.5, 0.53, 1.0, 1.03)]
    assert jerk_rms(gappy) is None
    assert jerk_rms([{**f, "right": {**f["right"], "tracked": False}} for f in cubic]) is None


ALL_KINDS = {
    **BASE,
    "rubric": [
        {
            "id": "no_stops",
            "text": "a",
            "points": 10,
            "auto": {"no_safety_stops": True},
        },
        {"id": "time", "text": "b", "points": 10, "auto": {"max_time_s": 60}},
        {
            "id": "path",
            "text": "c",
            "points": 10,
            "auto": {"max_path_deviation_m": 0.03},
        },
        {"id": "energy", "text": "d", "points": 10, "auto": {"max_energy_s": 3}},
        {"id": "near", "text": "e", "points": 10, "auto": {"max_near_misses": 1}},
        {"id": "steps", "text": "f", "points": 10, "auto": {"all_steps": True}},
        {"id": "smooth", "text": "g", "points": 25},
        {"id": "tidy", "text": "h", "points": 15},
    ],
}


def test_score_every_auto_kind_and_the_totals(tmp_path):
    exercise = _exercise(tmp_path, ALL_KINDS)
    good = score(
        exercise,
        {"duration_s": 42.0, "energy_s": {"left": 1.0, "right": 1.5}},
        [{"kind": "near_miss"}, {"kind": "step", "index": 0, "t_s": 3.0}],
        0.02,
    )
    assert [(i["id"], i["passed"], i["value"], i["points"]) for i in good["items"]] == [
        ("no_stops", True, 0, 10),
        ("time", True, 42.0, 10),
        ("path", True, 0.02, 10),
        ("energy", True, 2.5, 10),
        ("near", True, 1, 10),
        ("steps", True, 1, 10),
        ("smooth", None, None, 0),
        ("tidy", None, None, 0),
    ]
    assert (good["auto_points"], good["auto_max"], good["instructor_max"]) == (
        60,
        60,
        40,
    )
    bad = score(
        exercise,
        {"duration_s": 61.0, "energy_s": {"left": 2.0, "right": 1.5}},
        [{"kind": "incident"}, {"kind": "near_miss"}, {"kind": "near_miss"}],
        None,
    )
    assert [i["passed"] for i in bad["items"][:6]] == [False] * 6
    assert [i["value"] for i in bad["items"][:6]] == [1, 61.0, None, 3.5, 2, 0]
    assert [i["gated"] for i in bad["items"]] == [True] * 5 + [
        False
    ] * 3  # no step done
    assert (bad["auto_points"], bad["auto_max"], bad["instructor_max"]) == (0, 60, 40)


def test_max_time_s_scores_the_active_duration_not_the_video_length(tmp_path):
    exercise = _exercise(tmp_path, ALL_KINDS)  # time limit 60 s
    steps = [{"kind": "step", "index": 0, "t_s": 3.0}]
    energy = {"left": 0.0, "right": 0.0}

    def time_item(summary):
        items = score(exercise, {**summary, "energy_s": energy}, steps, 0.01)["items"]
        return next(item for item in items if item["id"] == "time")

    # A slow GPU: 40 s of active time makes 75 s of video. The limit is on the 40 s.
    fast = time_item({"duration_s": 40.0, "video_duration_s": 75.0})
    assert (fast["passed"], fast["value"], fast["points"]) == (True, 40.0, 10)
    # A fast GPU: 70 s of active time makes only 30 s of video. It still fails.
    slow = time_item({"duration_s": 70.0, "video_duration_s": 30.0})
    assert (slow["passed"], slow["value"], slow["points"]) == (False, 70.0, 0)


GATED = {
    **BASE,
    "steps": [*BASE["steps"], {"text": "Switch", "check": {"active": "left"}}],
    "rubric": [
        {"id": "all_steps", "text": "All", "points": 30, "auto": {"all_steps": True}},
        {
            "id": "no_stops",
            "text": "a",
            "points": 20,
            "auto": {"no_safety_stops": True},
        },
        {"id": "time", "text": "b", "points": 15, "auto": {"max_time_s": 60}},
        {"id": "smooth", "text": "c", "points": 35},
    ],
}
CLEAN = {"duration_s": 10.0, "energy_s": {"left": 0.0, "right": 0.0}}


def _rows(result):
    return [
        (i["id"], i["passed"], i["value"], i["points"], i["gated"])
        for i in result["items"]
    ]


def test_score_gates_auto_items_when_stopped_after_no_steps(tmp_path):
    result = score(_exercise(tmp_path, GATED), CLEAN, [], None)
    assert _rows(result) == [
        ("all_steps", False, 0, 0, False),
        ("no_stops", False, 0, 0, True),  # measured value kept
        ("time", False, 10.0, 0, True),
        ("smooth", None, None, 0, False),
    ]
    assert (result["auto_points"], result["auto_max"], result["instructor_max"]) == (
        0,
        65,
        35,
    )


def test_score_counts_every_item_when_all_steps_are_done(tmp_path):
    steps = [{"kind": "step", "index": i, "t_s": 1.0 + i} for i in range(2)]
    result = score(_exercise(tmp_path, GATED), CLEAN, steps, None)
    assert _rows(result) == [
        ("all_steps", True, 2, 30, False),
        ("no_stops", True, 0, 20, False),
        ("time", True, 10.0, 15, False),
        ("smooth", None, None, 0, False),
    ]
    assert result["auto_points"] == 65


def test_score_never_gates_without_an_all_steps_item(tmp_path):
    ungated = {**GATED, "rubric": GATED["rubric"][1:]}
    result = score(_exercise(tmp_path, ungated), CLEAN, [], None)  # no step done
    assert _rows(result) == [
        ("no_stops", True, 0, 20, False),
        ("time", True, 10.0, 15, False),
        ("smooth", None, None, 0, False),
    ]


def test_build_report_and_review_validation(tmp_path):
    exercise = _exercise(tmp_path, ALL_KINDS)
    events = [
        {"kind": "start", "t_s": 0.0, "t_video_s": 0.0},
        # Step times are active seconds (t_s); the video position (t_video_s) is not used.
        {"kind": "step", "index": 0, "t_s": 4.25, "t_video_s": 9.5},
        {"kind": "step", "index": 0, "t_s": 5.0, "t_video_s": 11.0},  # a repeat keeps the first time
        {"kind": "step", "index": 3, "t_s": 6.0, "t_video_s": 13.0},  # out of range
        {"kind": "near_miss", "rule": "speed", "ratio": 0.85, "t_s": 2.0, "t_video_s": 4.0},
    ]
    attempt = [
        {
            "left": {"position": list(HOME["left"])},
            "right": {"position": [0.2, 1.01, -0.3]},
        }
    ]
    report = build_report(
        exercise,
        {"duration_s": 10.0, "energy_s": {"left": 0, "right": 0}},
        events,
        attempt,
    )
    assert report["steps"] == [{"index": 0, "t_s": 4.25}]
    assert report["steps_done"] == 1 and report["near_misses"] == 1
    assert report["path_deviation_m"] == pytest.approx(0.005)  # (0 + 0.01) / 2
    assert "corrections" in report and "demo_corrections" in report
    assert report["auto_points"] == 60 and report["instructor_max"] == 40

    review = validate_review(
        exercise, {"checked": ["tidy", "smooth", "tidy"], "notes": "Good"}, now=NOW
    )
    assert review == {
        "checked": ["tidy", "smooth"],
        "notes": "Good",
        "points": 40,
        "instructor_max": 40,
        "saved": "2026-10-27T14:30:05",
    }
    assert validate_review(exercise, {})["points"] == 0
    for bad, match in (
        ({"checked": ["no_stops"]}, "not an instructor item"),
        ({"checked": ["nope"]}, "not an instructor item"),
        ({"checked": "smooth"}, "list of rubric item ids"),
        ({"notes": "x" * 2001}, "at most 2000"),
        (["smooth"], "expected"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_review(exercise, bad)


@pytest.mark.parametrize("stride", [1, 3])
def test_safe_starter_demos_stay_under_60_percent_of_every_limit(tmp_path, stride):
    from cosmos3_h_dreams.safety import RULES

    for exercise_id, exercise in load_courses(COURSES, tmp_path / "demos").items():
        if exercise_id == "safety-stop":
            continue
        results = _guard_windows(exercise.demo, stride)
        worst = {rule: max(result.ratios[rule] for result in results) for rule in RULES}
        assert max(worst.values()) < 0.6, (exercise_id, worst)  # the meter stays grey
