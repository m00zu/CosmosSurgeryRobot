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

"""Session recorder file output and listing."""

from __future__ import annotations

import errno
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from cosmos3_h_dreams import recording
from cosmos3_h_dreams.recording import (
    SESSION_ID,
    SessionRecorder,
    list_sessions,
    load_session,
    sanitize_trainee,
)

pytestmark = pytest.mark.ci_cpu
NOW = datetime(2026, 10, 27, 14, 30, 5)


def _controller(x, grip):
    return {
        "position": [x, 1.0, -0.3],
        "orientation": [0, 0, 0, 1],
        "grip": grip,
        "trigger": 0.0,
    }


def _window(start, xs, grips):
    return [
        {
            "left": _controller(0.0, 0.0),
            "right": _controller(x, g),
            "client_capture_ns": (start + i) * 100_000_000,
        }
        for i, (x, g) in enumerate(zip(xs, grips))
    ]


def _recorder(tmp_path, ffmpeg, trainee="Dr Lin!", **extra):
    return SessionRecorder(
        tmp_path,
        trainee=trainee,
        scene="example_frame.png",
        mode="standard",
        seed=0,
        fps=10.0,
        safety={"max_speed_m_s": 0.25},
        now=NOW,
        ffmpeg=ffmpeg,
        **extra,
    )


def _events(recorder):
    return [
        json.loads(line)
        for line in (recorder.path / "events.jsonl").read_text().splitlines()
    ]


class _Clock:
    """A hand-driven stand-in for ``time.monotonic``."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _frames(count):
    return [np.zeros((4, 6, 3), np.uint8)] * count


def _hold(recorder, kind, **data):
    """Hold and log a ``pause`` or ``incident``, as the manager does."""
    recorder.begin_hold()
    recorder.event(kind, **data)


def _release(recorder, after):
    """End the hold and log the ``resume``, as the manager does."""
    recorder.end_hold()
    recorder.event("resume", after=after)


def test_sanitize_trainee():
    assert sanitize_trainee("Dr Lin!") == "Dr_Lin"
    assert sanitize_trainee("  ") == "trainee"
    assert sanitize_trainee("林醫師") == "trainee"
    assert len(sanitize_trainee("x" * 99)) == 40


def test_records_actions_events_and_summary_without_ffmpeg(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=None)
    assert recorder.id == "20261027-143005_Dr_Lin"
    assert json.loads((recorder.path / "session.json").read_text())["complete"] is False
    frames = [np.zeros((4, 6, 3), np.uint8)] * 4
    recorder.add_chunk(
        frames,
        np.ones((4, 28), np.float32),
        _window(0, [0.0, 0.01, 0.02, 0.03], [0, 1, 1, 0]),
    )
    assert recorder.video_time_s == pytest.approx(0.4)
    recorder.event("incident", violations=[{"rule": "speed"}])
    summary = recorder.close("stop")
    assert recorder.close("again") == summary
    assert summary["video_ok"] is False and summary["complete"] is False
    assert summary["summary"]["incidents"] == 1
    assert summary["summary"]["path_length_m"]["right"] == pytest.approx(0.03)
    assert summary["summary"]["energy_s"]["right"] == pytest.approx(0.2)
    assert summary["end_reason"] == "stop"
    assert np.load(recorder.path / "actions.npy").shape == (4, 28)
    kinds = [
        json.loads(line)["kind"]
        for line in (recorder.path / "events.jsonl").read_text().splitlines()
    ]
    assert kinds == ["start", "energy_on", "energy_off", "incident", "end"]
    assert len((recorder.path / "controller.jsonl").read_text().splitlines()) == 4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_writes_playable_video(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=shutil.which("ffmpeg"))
    frames = [np.full((64, 96, 3), i * 20, np.uint8) for i in range(8)]
    recorder.add_chunk(
        frames, np.zeros((8, 28), np.float32), _window(0, [0.0] * 8, [0] * 8)
    )
    summary = recorder.close("stop")
    assert summary["video_ok"] is True and summary["complete"] is True
    assert (recorder.path / "video.mp4").stat().st_size > 0


def test_broken_ffmpeg_keeps_recording_metadata(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=str(tmp_path / "missing-ffmpeg"))
    recorder.add_chunk(
        [np.zeros((4, 6, 3), np.uint8)],
        np.zeros((1, 28), np.float32),
        _window(0, [0.0, 0.0], [0, 0]),
    )
    summary = recorder.close("stop")
    assert summary["video_ok"] is False and summary["video_error"]


def test_failing_ffmpeg_reports_its_stderr(tmp_path):
    fake = tmp_path / "fake-ffmpeg"
    fake.write_text("#!/bin/sh\ncat > /dev/null\necho encoder exploded >&2\nexit 3\n")
    fake.chmod(0o755)
    recorder = _recorder(tmp_path / "sessions", ffmpeg=str(fake))
    recorder.add_chunk(
        [np.zeros((4, 6, 3), np.uint8)],
        np.zeros((1, 28), np.float32),
        _window(0, [0.0], [0]),
    )
    summary = recorder.close("stop")
    assert summary["video_ok"] is False and summary["complete"] is False
    assert "encoder exploded" in summary["video_error"]


def test_list_and_load_validate_ids(tmp_path):
    first = _recorder(tmp_path, ffmpeg=None)
    first.close("stop")
    (tmp_path / "not-a-session").mkdir()
    sessions = list_sessions(tmp_path)
    assert [s["id"] for s in sessions] == [first.id]
    loaded = load_session(tmp_path, first.id)
    assert loaded is not None
    assert loaded["session"]["trainee"] == "Dr Lin!"
    assert loaded["events"][0]["kind"] == "start"
    assert load_session(tmp_path, "../etc") is None
    assert load_session(tmp_path, "not-a-session") is None
    assert list_sessions(tmp_path / "absent") == []


def test_same_second_attempts_get_distinct_ids(tmp_path):
    first, second, third = (_recorder(tmp_path, ffmpeg=None) for _ in range(3))
    assert [first.id, second.id, third.id] == [
        "20261027-143005_Dr_Lin",
        "20261027-143005_Dr_Lin-2",
        "20261027-143005_Dr_Lin-3",
    ]
    assert [s["id"] for s in list_sessions(tmp_path)] == [third.id, second.id, first.id]
    loaded = load_session(tmp_path, second.id)
    assert loaded is not None and loaded["session"]["id"] == second.id


def test_untracked_controllers_and_gaps_are_discontinuities(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=None)
    frames, actions = [np.zeros((4, 6, 3), np.uint8)] * 2, np.zeros((2, 28), np.float32)
    window = _window(0, [0.0, 0.01, 0.02, 0.03], [1, 1, 1, 1])
    window[2]["right"]["tracked"] = False
    window[3]["left"] = None
    recorder.add_chunk(frames, actions, window[:2])
    recorder.add_chunk(
        frames, actions, window[2:]
    )  # no overlap frame: edges and path carry over
    recorder.add_chunk(
        frames, actions, _window(10, [0.5, 0.51], [1, 1])
    )  # 0.7 s capture gap first
    summary = recorder.close("stop")["summary"]
    assert summary["path_length_m"]["right"] == pytest.approx(0.02)
    assert summary["energy_s"]["right"] == pytest.approx(0.2)
    assert (
        summary["path_length_m"]["left"] == 0.0 and summary["energy_s"]["left"] == 0.0
    )
    edges = [
        (e["kind"], e["side"], e["t_video_s"])
        for e in _events(recorder)
        if e["kind"].startswith("energy")
    ]
    assert edges == [
        ("energy_on", "right", 0.0),
        ("energy_off", "right", 0.2),
        ("energy_on", "right", 0.3),
    ]


def test_state_carries_over_between_chunks(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=None)
    frames, actions = [np.zeros((4, 6, 3), np.uint8)] * 2, np.zeros((2, 28), np.float32)
    window = _window(0, [0.0, 0.01, 0.02, 0.03], [1, 1, 1, 1])
    recorder.add_chunk(frames, actions, window[:2])
    recorder.add_chunk(frames, actions, window[2:])
    summary = recorder.close("stop")["summary"]
    assert summary["path_length_m"]["right"] == pytest.approx(0.03)
    assert summary["energy_s"]["right"] == pytest.approx(0.3)
    edges = [
        (e["kind"], e["t_video_s"])
        for e in _events(recorder)
        if e["kind"].startswith("energy")
    ]
    assert edges == [("energy_on", 0.0)]


def test_unicode_trainee_keeps_the_name_and_slugs_the_id(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=None, trainee="林醫師")
    recorder.close("stop")
    assert recorder.trainee == "林醫師"
    assert recorder.id == "20261027-143005_trainee"
    assert SESSION_ID.match(recorder.id)
    assert [s["trainee"] for s in list_sessions(tmp_path)] == ["林醫師"]
    loaded = load_session(tmp_path, recorder.id)
    assert loaded is not None and loaded["session"]["trainee"] == "林醫師"


def test_trainee_display_name_is_cleaned(tmp_path):
    names = {
        "  Dr \t Lin\x00\u200b!\n": "Dr Lin!",
        "林 醫師\u3000A": "林 醫師 A",
        "林" * 50: "林" * 40,
        "": "anonymous",
        "\x07 \u200b": "anonymous",
    }
    for name, expected in names.items():
        assert _recorder(tmp_path, ffmpeg=None, trainee=name).trainee == expected


def test_ffmpeg_default_is_looked_up_when_the_recorder_is_created(
    tmp_path, monkeypatch
):
    def create():
        return SessionRecorder(
            tmp_path,
            trainee="t",
            scene="s",
            mode="m",
            seed=0,
            fps=10.0,
            safety={},
            now=NOW,
        )

    monkeypatch.setattr(shutil, "which", lambda name: "/fake/ffmpeg")
    assert create().video_ok is True
    monkeypatch.setattr(shutil, "which", lambda name: None)
    recorder = create()
    assert recorder.video_ok is False and recorder.video_error == "ffmpeg not found"


def _disk_full(*args, **kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_failed_write_is_recorded_not_raised(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path, ffmpeg=None)
    monkeypatch.setattr(Path, "open", _disk_full)
    recorder.event("pause")
    recorder.add_chunk(
        [np.zeros((4, 6, 3), np.uint8)],
        np.zeros((1, 28), np.float32),
        _window(0, [0.0], [1]),
    )
    monkeypatch.undo()
    recorder.event("resume")
    summary = recorder.close("stop")
    assert "No space left" in summary["write_error"]
    on_disk = json.loads((recorder.path / "session.json").read_text())
    assert (
        on_disk["write_error"] == summary["write_error"]
        and on_disk["complete"] is False
    )
    assert [e["kind"] for e in _events(recorder)] == ["start", "resume", "end"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_full_disk_never_raises_and_ffmpeg_is_reaped(tmp_path, monkeypatch):
    recorder = _recorder(tmp_path, ffmpeg=shutil.which("ffmpeg"))
    frames = [np.full((64, 96, 3), 30, np.uint8)] * 3
    actions = np.zeros((3, 28), np.float32)
    recorder.add_chunk(frames, actions, _window(0, [0.0] * 3, [0] * 3))
    monkeypatch.setattr(Path, "open", _disk_full)
    monkeypatch.setattr(Path, "write_text", _disk_full)
    monkeypatch.setattr(np, "save", _disk_full)
    recorder.add_chunk(frames, actions, _window(3, [0.0] * 3, [0] * 3))
    recorder.event("incident", violations=[])
    summary = recorder.close("stop")
    assert recorder.close("again") is summary
    monkeypatch.undo()
    assert summary["complete"] is False and "No space left" in summary["write_error"]
    assert summary["video_ok"] is True and summary["summary"]["incidents"] == 1
    assert summary["summary"]["video_duration_s"] == pytest.approx(0.6)
    assert summary["summary"]["duration_s"] >= 0  # active time, measured by the real clock
    assert recorder._proc is not None and recorder._proc.returncode == 0  # waited for
    assert (recorder.path / "video.mp4").stat().st_size > 0


def test_any_video_finish_failure_is_reported_and_ffmpeg_is_reaped(tmp_path, monkeypatch):
    fake = tmp_path / "fake-ffmpeg"
    fake.write_text("#!/bin/sh\ncat > /dev/null\n")
    fake.chmod(0o755)
    recorder = _recorder(tmp_path / "sessions", ffmpeg=str(fake))
    recorder.add_chunk(
        [np.zeros((4, 6, 3), np.uint8)],
        np.zeros((1, 28), np.float32),
        _window(0, [0.0], [0]),
    )

    def broken_finish(_proc):
        raise ValueError("bad stderr")

    monkeypatch.setattr(recorder, "_finish_video", broken_finish)
    summary = recorder.close("stop")
    assert summary["complete"] is False and summary["video_ok"] is False
    assert summary["video_error"] == "bad stderr"
    assert recorder._proc.returncode is not None  # killed and waited for
    assert [e["kind"] for e in _events(recorder)] == ["start", "end"]


def test_close_stores_vram_once(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=None)
    vram = {
        "peak_used_mib": 5000.0,
        "mean_used_mib": 4000.0,
        "peak_allocated_mib": None,
        "mean_allocated_mib": None,
    }
    summary = recorder.close("stop", vram=vram)
    assert summary["summary"]["vram"] == vram
    assert recorder.close("again", vram={"peak_used_mib": 1.0}) == summary
    stored = json.loads((recorder.path / "session.json").read_text())
    assert stored["summary"]["vram"] == vram
    assert "vram" not in _recorder(tmp_path, ffmpeg=None).close("stop")["summary"]


def test_session_json_stores_the_text_cache_status_next_to_mode(tmp_path):
    recorder = SessionRecorder(
        tmp_path, trainee="A", scene="s", mode="standard", seed=0, fps=10.0,
        safety={}, now=NOW, ffmpeg=None, text_cache={"status": "hit", "rehashed": False},
    )
    data = json.loads((recorder.path / "session.json").read_text())
    assert data["mode"] == "standard"
    assert data["text_cache"] == {"status": "hit", "rehashed": False}
    assert _recorder(tmp_path, ffmpeg=None).close("stop")["text_cache"] is None


def _course_exercise(tmp_path):
    from cosmos3_h_dreams.courses import load_courses

    courses = Path(__file__).resolve().parents[1] / "courses"
    return load_courses(courses, tmp_path / "demos")["instrument-basics"]


def test_exercise_attempt_writes_its_report_to_session_json(tmp_path):
    exercise = _course_exercise(tmp_path)
    clock = _Clock()
    recorder = SessionRecorder(
        tmp_path / "rec", trainee="Dr Lin", scene="s", mode="standard", seed=0, fps=10.0,
        safety={}, exercise=exercise, now=NOW, ffmpeg=None, clock=clock,
    )
    started = json.loads((recorder.path / "session.json").read_text())
    assert started["exercise"] == {"id": "instrument-basics", "title": "Instrument basics"}
    assert started["report"] is None
    recorder.add_chunk([np.zeros((4, 6, 3), np.uint8)] * 4, np.ones((4, 28), np.float32),
                       _window(0, [0.15, 0.16, 0.17, 0.18], [0, 0, 0, 0]))
    clock.advance(1.5)
    recorder.event("step", index=0, t_client=1.0)
    clock.advance(2.0)
    recorder.event("step", index=1, t_client=2.0)
    _hold(recorder, "incident", violations=[])
    clock.advance(1.0)  # held by the incident: not counted
    summary = recorder.close("stop")
    report = json.loads((recorder.path / "session.json").read_text())["report"]
    assert summary["report"] == report
    # Step times are active seconds (the clock), not the 0.4 s of video recorded.
    assert report["steps"] == [{"index": 0, "t_s": 1.5}, {"index": 1, "t_s": 3.5}]
    assert report["steps_done"] == 2 and report["near_misses"] == 0
    # Left stays at x 0, 0.15 m from its demo path (home); right moves along x 0.15-0.18
    # at z -0.3, 0-0.023 m (mean 0.0115) from the right demo path: (0.15 + 0.0115) / 2.
    assert report["path_deviation_m"] == pytest.approx(0.0808, abs=1e-3)
    # 2 of 4 steps done: all_steps fails and gates the rest (measured values kept).
    assert [(i["id"], i["passed"], i["gated"]) for i in report["items"]] == [
        ("all_steps", False, False), ("no_stops", False, True), ("time", False, True),
        ("path", False, True), ("smooth", None, False),
    ]
    assert report["items"][2]["value"] == 3.5  # the time item scores the active duration
    assert (report["auto_points"], report["auto_max"], report["instructor_max"]) == (0, 80, 20)


def test_free_practice_has_no_exercise_or_report(tmp_path):
    summary = _recorder(tmp_path, ffmpeg=None).close("stop")
    assert "exercise" not in summary and "report" not in summary


def test_report_failure_never_loses_the_recording(tmp_path, monkeypatch):
    def broken(*_args):
        raise RuntimeError("bug")

    monkeypatch.setattr(recording, "build_report", broken)
    recorder = SessionRecorder(
        tmp_path, trainee="Dr Lin", scene="s", mode="standard", seed=0, fps=10.0,
        safety={}, exercise=_course_exercise(tmp_path), now=NOW, ffmpeg=None,
    )
    summary = recorder.close("stop")
    assert summary["end_reason"] == "stop" and summary["report"] is None
    assert (recorder.path / "actions.npy").exists()


def test_review_is_saved_and_read_with_the_session(tmp_path):
    from cosmos3_h_dreams.recording import save_review

    recorder = _recorder(tmp_path, ffmpeg=None)
    recorder.close("stop")
    assert list_sessions(tmp_path)[0]["review"] is None
    save_review(tmp_path, recorder.id, {"checked": ["smooth"], "notes": "", "points": 30})
    assert list_sessions(tmp_path)[0]["review"]["points"] == 30
    loaded = load_session(tmp_path, recorder.id)
    assert loaded["review"] == loaded["session"]["review"]
    assert (loaded["exercise"], loaded["report"]) == (None, None)
    with pytest.raises(FileNotFoundError):
        save_review(tmp_path, "../etc", {})
    (recorder.path / "review.json").write_text("[1]")
    assert load_session(tmp_path, recorder.id)["review"] is None


def test_near_miss_hysteresis(tmp_path):
    recorder = _recorder(tmp_path, ffmpeg=None)
    for speed in (0.5, 0.85, 0.9, 0.7, 0.85, 0.5, 0.82, 1.2, 0.9, 0.55, 0.95):
        recorder.observe_safety({"speed": speed, "energy_dose": 0.0})
    recorder.observe_safety({"speed": 0.1, "energy_dose": 0.8})
    recorder.close("stop")
    recorder.observe_safety({"speed": 0.9})  # ignored after close
    near = [(e["rule"], e["ratio"]) for e in _events(recorder) if e["kind"] == "near_miss"]
    # 0.85 logs; 0.9/0.7/0.85 wait for a drop under 0.6; 0.82 logs; 1.2 is a stop, not a
    # near miss, and 0.9 after it still waits; 0.95 logs after the drop to 0.55.
    assert near == [("speed", 0.85), ("speed", 0.82), ("speed", 0.95), ("energy_dose", 0.8)]


def test_duration_and_step_times_are_active_seconds_not_video_seconds(tmp_path):
    exercise = _course_exercise(tmp_path)
    clock = _Clock()
    recorder = SessionRecorder(
        tmp_path, trainee="Dr Lin", scene="s", mode="standard", seed=0, fps=10.0,
        safety={}, exercise=exercise, now=NOW, ffmpeg=None, clock=clock,
    )
    actions = np.zeros((1, 28), np.float32)
    clock.advance(2.0)
    recorder.add_chunk(_frames(100), actions, _window(0, [0.15], [0]))  # 10 s of video
    recorder.event("step", index=0)
    clock.advance(1.0)
    _hold(recorder, "pause")
    clock.advance(5.0)  # paused: held
    _release(recorder, "pause")
    clock.advance(2.0)
    recorder.add_chunk(_frames(120), actions, _window(0, [0.15], [0]))  # 22 s of video
    recorder.event("step", index=1)
    clock.advance(1.0)
    _hold(recorder, "incident", violations=[])
    clock.advance(4.0)  # stopped by the incident: held
    _release(recorder, "incident")
    clock.advance(6.0)
    closed = recorder.close("stop")

    # 21 s of wall time, 9 s of it held: 12 s of active time over 22 s of video.
    assert closed["summary"]["duration_s"] == 12.0
    assert closed["summary"]["video_duration_s"] == pytest.approx(22.0)
    assert closed["summary"]["incidents"] == 1
    events = _events(recorder)
    assert [(e["kind"], e["t_s"], e["t_video_s"]) for e in events] == [
        ("start", 0.0, 0.0),
        ("step", 2.0, 10.0),
        ("pause", 3.0, 10.0),
        ("resume", 3.0, 10.0),
        ("step", 5.0, 22.0),
        ("incident", 6.0, 22.0),
        ("resume", 6.0, 22.0),
        ("end", 12.0, 22.0),
    ]
    # The report takes the same clock: step times and the time rubric item.
    report = closed["report"]
    assert report["steps"] == [{"index": 0, "t_s": 2.0}, {"index": 1, "t_s": 5.0}]
    assert report["items"][2]["id"] == "time" and report["items"][2]["value"] == 12.0
    on_disk = json.loads((recorder.path / "session.json").read_text())
    assert on_disk["summary"] == closed["summary"]


def test_holds_overlap_and_end_with_the_attempt(tmp_path):
    clock = _Clock()
    recorder = _recorder(tmp_path, ffmpeg=None, clock=clock)
    clock.advance(1.0)
    _hold(recorder, "pause")
    clock.advance(1.0)
    _hold(recorder, "incident", violations=[])  # an incident while paused: still one hold
    clock.advance(2.0)
    _release(recorder, "incident")
    clock.advance(1.0)
    _release(recorder, "pause")  # nothing is held: ignored
    clock.advance(1.0)
    _hold(recorder, "pause")
    clock.advance(4.0)  # still paused when the attempt ends
    summary = recorder.close("stop")["summary"]
    assert summary["duration_s"] == 3.0
    assert [(e["kind"], e["t_s"]) for e in _events(recorder)] == [
        ("start", 0.0),
        ("pause", 1.0),
        ("incident", 1.0),
        ("resume", 1.0),
        ("resume", 2.0),
        ("pause", 3.0),
        ("end", 3.0),
    ]


def test_active_time_is_fixed_when_the_attempt_closes(tmp_path):
    clock = _Clock()
    recorder = _recorder(tmp_path, ffmpeg=None, clock=clock)
    clock.advance(1.2346)
    _hold(recorder, "pause")
    clock.advance(0.5)
    closed = recorder.close("stop")
    clock.advance(100.0)  # after close: neither the summary nor later calls move
    _release(recorder, "pause")
    assert recorder.close("again")["summary"]["duration_s"] == closed["summary"]["duration_s"] == 1.235
    assert _events(recorder)[1]["t_s"] == 1.235  # rounded to 3 decimal places
    assert [e["kind"] for e in _events(recorder)] == ["start", "pause", "end"]
    # No video was recorded: the video length stays 0 and does not follow the clock.
    assert closed["summary"]["video_duration_s"] == 0.0


def test_holds_start_when_begun_not_when_their_event_is_logged(tmp_path):
    clock = _Clock()
    recorder = _recorder(tmp_path, ffmpeg=None, clock=clock)
    clock.advance(3.0)
    recorder.begin_hold()  # the manager holds at once; the incident event is logged by a worker thread
    clock.advance(2.0)
    recorder.end_hold()  # a resume arrives before that event is logged
    assert not recorder.held
    clock.advance(1.0)
    recorder.event("incident", violations=[])  # the late event only logs: it must not hold again
    recorder.event("resume", after="incident")
    assert not recorder.held
    clock.advance(1.0)
    summary = recorder.close("stop")["summary"]
    assert summary["duration_s"] == 5.0  # 7 s of wall time, held from 3 s to 5 s
    assert [(e["kind"], e["t_s"]) for e in _events(recorder)] == [
        ("start", 0.0),
        ("incident", 4.0),
        ("resume", 4.0),
        ("end", 5.0),
    ]


def test_events_alone_do_not_hold(tmp_path):
    clock = _Clock()
    recorder = _recorder(tmp_path, ffmpeg=None, clock=clock)
    for kind in ("pause", "incident"):
        clock.advance(1.0)
        recorder.event(kind)
        assert not recorder.held
    assert recorder.close("stop")["summary"]["duration_s"] == 2.0


def test_an_attempt_can_start_held(tmp_path):
    clock = _Clock()
    recorder = _recorder(tmp_path, ffmpeg=None, clock=clock)
    recorder.begin_hold()  # started while the manager was paused
    assert recorder.held
    clock.advance(4.0)
    _release(recorder, "pause")
    assert not recorder.held
    clock.advance(3.0)
    summary = recorder.close("stop")["summary"]
    assert summary["duration_s"] == 3.0
    assert [(e["kind"], e["t_s"]) for e in _events(recorder)] == [
        ("start", 0.0),
        ("resume", 0.0),
        ("end", 3.0),
    ]
