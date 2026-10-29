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

"""Training ground server routes and generation control."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from cosmos3_h_dreams.headset_server import HeadsetServer, LatestHeadsetFrames, create_app

pytestmark = pytest.mark.ci_cpu


def _idle_session() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(fps=10, profile=False, fp8_generation=False, seed=0),
        model_status="Model ready",
        init=lambda: None,
        close=lambda: None,
        next_step_requirements=lambda: None,
        step=lambda _window: None,
    )


def test_training_page_and_scripts_are_served() -> None:
    async def check() -> None:
        manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            page = await client.get("/train")
            text = await page.text()
            assert page.status == 200
            assert page.headers["Cache-Control"] == "no-store"
            for marker in (
                'src="/train_core.js"', 'src="/train.js"', 'src="/scenes.js"', 'id="overlay"',
                'src="/train_courses.js"', 'id="course-menu"', 'id="course-panel"', 'id="path-guide"',
                'id="playback"', 'id="report"', 'id="resource-charts"', 'id="latency"',
                'src="/xr_view.js"', 'src="/train_vr.js"', 'id="enter-vr"',
            ):
                assert marker in text
            assert text.index('src="/train_courses.js"') < text.index('src="/train.js"')
            assert (text.index('src="/train_courses.js"') < text.index('src="/xr_view.js"')
                    < text.index('src="/train_vr.js"') < text.index('src="/train.js"'))
            for path in ("/train.js", "/train_core.js", "/train_courses.js", "/xr_view.js", "/train_vr.js"):
                script = await client.get(path)
                assert script.status == 200
                assert script.content_type == "text/javascript"
            await manager.close()

    asyncio.run(check())


def test_gpu_page_and_scripts_are_served_no_store() -> None:
    async def check() -> None:
        manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            page = await client.get("/gpu")
            text = await page.text()
            assert page.status == 200
            assert page.headers["Cache-Control"] == "no-store"
            for marker in ('src="/gpu_chart.js"', 'src="/gpu.js"', 'id="chart"'):
                assert marker in text
            for path in ("/gpu.js", "/gpu_chart.js"):
                script = await client.get(path)
                assert script.status == 200
                assert script.content_type == "text/javascript"
                assert script.headers["Cache-Control"] == "no-store"
            train = await (await client.get("/train")).text()
            assert 'id="gpu-chart"' in train and 'id="host-chart"' in train and 'src="/gpu_chart.js"' in train
            await manager.close()

    asyncio.run(check())


def test_dashboard_page_and_script_are_served() -> None:
    async def check() -> None:
        manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            page = await client.get("/dashboard")
            text = await page.text()
            assert page.status == 200
            assert page.headers["Cache-Control"] == "no-store"
            for marker in (
                'id="sessions"', 'id="trainee-filter"', 'id="player"', 'src="/dashboard.js"',
                'id="exercise-filter"', 'id="progress-chart"', 'id="rubric-section"',
                'id="save-demo-section"', 'src="/gpu_chart.js"', 'src="/train_core.js"',
            ):
                assert marker in text
            assert text.index('src="/train_core.js"') < text.index('src="/dashboard.js"')
            script = await client.get("/dashboard.js")
            assert script.status == 200
            assert script.content_type == "text/javascript"
            assert "timelineMarkers" in await script.text()
            await manager.close()

    asyncio.run(check())


def test_incident_holds_generation_until_resume(monkeypatch) -> None:
    from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits, SafetyStop, Violation
    from cosmos3_h_dreams import headset_server as server_module
    import numpy as np

    anchor = {
        "left": {"position": [0.0, 1.0, -0.3], "orientation": [0, 0, 0, 1]},
        "right": {"position": [0.1, 1.0, -0.3], "orientation": [0, 0, 0, 1]},
        "client_capture_ns": 1,
    }
    calls = {"step": 0}
    violation = Violation("speed", "right", 0.9, 0.25, "m/s")

    def step(_window):
        calls["step"] += 1
        if calls["step"] == 1:
            raise SafetyStop([violation], [anchor, anchor])
        return SimpleNamespace(metadata={"action_rows": 1})

    session = SimpleNamespace(
        config=SimpleNamespace(fps=10, profile=False, fp8_generation=False, seed=0),
        model_status="Model ready",
        init=lambda: None,
        close=lambda: None,
        next_step_requirements=lambda: SimpleNamespace(input_frame_count=1),
        step=step,
    )
    monkeypatch.setattr(
        server_module.MjpegFrames, "prepare_result_frames",
        lambda _t, _r: (np.zeros((4, 6, 3), np.uint8),),
    )
    guard = SafetyGuard(SafetyLimits())
    resumed = []
    monkeypatch.setattr(guard, "resume", lambda: resumed.append(True))
    rewound = []

    async def check() -> None:
        manager = HeadsetServer(session=session, frames=LatestHeadsetFrames(fps=30), guard=guard)
        manager.source = SimpleNamespace(
            rewind=rewound.append,
            request_restart=lambda: None,
            reset_calibration=lambda: None,
            close=lambda: None,
        )
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                message = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert message["type"] == "incident"
                assert message["violations"][0]["message"] == violation.message()
                assert message["restore"]["right"]["position"] == [0.1, 1.0, -0.3]
                assert manager.hold == "incident"
                held_calls = calls["step"]
                await asyncio.sleep(0.3)
                assert calls["step"] == held_calls
                health = await (await client.get("/healthz")).json()
                assert health["hold"] == "incident"
                assert manager.frames.push(_xr_frame(0.5))  # a pre-resume pose
                await ws.send_json({"type": "resume"})
                for _ in range(100):
                    if calls["step"] > held_calls:
                        break
                    await asyncio.sleep(0.01)
                assert calls["step"] > held_calls
                assert rewound == [anchor] and resumed == [True]
                assert manager.hold is None
                assert manager.frames.poll() is None  # resume dropped it
                await ws.send_json({"type": "pause"})
                for _ in range(100):
                    if manager.hold == "pause":
                        break
                    await asyncio.sleep(0.01)
                assert manager.hold == "pause"
                await ws.send_json({"type": "resume"})
                for _ in range(100):
                    if manager.hold is None:
                        break
                    await asyncio.sleep(0.01)
                assert manager.hold is None and rewound == [anchor]
            await manager.close()

    asyncio.run(check())


def _xr_frame(x: float) -> dict:
    pose = {"position": [x, 1.0, -0.3], "orientation": [0, 0, 0, 1],
            "buttons": [{"v": 0.0}, {"v": 0.0}], "axes": []}
    return {"type": "xr_frame", "controllers": {"left": pose, "right": pose}}


def _controller_frame(t_ns: int, x: float) -> dict:
    return {
        "left": {"position": [x, 1.0, -0.3], "orientation": [0, 0, 0, 1], "grip": 0.0},
        "right": {"position": [0.1, 1.0, -0.3], "orientation": [0, 0, 0, 1], "grip": 0.0},
        "client_capture_ns": t_ns,
    }


def _recording_server(monkeypatch, tmp_path, step):
    import numpy as np

    from cosmos3_h_dreams import headset_server as server_module
    from cosmos3_h_dreams import recording as recording_module

    monkeypatch.setattr(recording_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        server_module.MjpegFrames,
        "prepare_result_frames",
        lambda _t, _r: tuple(np.zeros((4, 6, 3), np.uint8) for _ in range(2)),
    )
    session = SimpleNamespace(
        config=SimpleNamespace(fps=10, profile=False, fp8_generation=False, seed=7),
        model_status="Model ready",
        text_cache={
            "status": "hit", "rehashed": True, "path": "/c/text.pt", "hash_s": 16.8, "load_s": 3.9,
        },
        init=lambda: None,
        close=lambda: None,
        next_step_requirements=lambda: SimpleNamespace(input_frame_count=1),
        step=step,
    )
    manager = HeadsetServer(
        session=session, frames=LatestHeadsetFrames(fps=30), recording_dir=tmp_path
    )
    manager.source = SimpleNamespace(
        last_actions=np.zeros((2, 28), np.float32),
        last_frames=tuple(_controller_frame(i * 100_000_000, 0.01 * i) for i in range(3)),
        rewind=lambda _frame: None,
        request_restart=lambda: None,
        reset_calibration=lambda: None,
        close=lambda: None,
    )
    return manager


async def _until(predicate, timeout: float = 3.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


def test_recording_lifecycle_and_recordings_api(monkeypatch, tmp_path) -> None:
    manager = _recording_server(
        monkeypatch, tmp_path, lambda _w: SimpleNamespace(metadata={"action_rows": 2})
    )

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin!"})
                await _until(
                    lambda: manager.recorder is not None
                    and manager.recorder.video_time_s > 0
                )
                health = await (await client.get("/healthz")).json()
                assert manager.recorder.fps == 30.0  # the input rate, not the model's 10 fps
                assert health["recording"] == {
                    "id": manager.recorder.id,
                    "trainee": "Dr Lin!",
                    "video_ok": False,
                    "exercise": None,
                }
                assert health["recording_error"] is None
                first = manager.recorder.id
                await ws.send_json({"type": "reset"})
                await _until(
                    lambda: manager.recorder is not None and manager.recorder.id != first
                )
                assert manager.recorder.trainee == "Dr Lin!"
                await ws.send_json({"type": "stop"})
                await _until(lambda: manager.recorder is None)
                health = await (await client.get("/healthz")).json()
                assert health["recording"] is None

                await ws.send_json({"type": "start", "trainee": "林醫師"})
                await _until(lambda: manager.recorder is not None)
                health = await (await client.get("/healthz")).json()
                assert health["recording"]["trainee"] == "林醫師"
                await ws.send_json({"type": "stop"})
                await _until(lambda: manager.recorder is None)
                await ws.send_json({"type": "start", "trainee": 3})
                await asyncio.sleep(0.1)
                assert manager.recorder is None

            sessions = await (await client.get("/api/sessions")).json()
            assert len(sessions) == 3
            assert {s["end_reason"] for s in sessions} == {"reset", "stop"}
            detail = await (await client.get(f"/api/sessions/{first}")).json()
            assert detail["session"]["id"] == first
            assert detail["events"][0]["kind"] == "start"
            assert detail["session"]["seed"] == 7
            assert detail["session"]["mode"] == "standard"
            assert detail["session"]["text_cache"] == {"status": "hit", "rehashed": True}
            for path in (
                "/api/sessions/..%2Fetc",
                "/api/sessions/nope",
                "/recordings/nope/video.mp4",
                f"/recordings/{first}/video.mp4",  # ffmpeg is off, so no video file
            ):
                assert (await client.get(path)).status == 404, path
            (tmp_path / first / "video.mp4").write_bytes(b"0123456789")
            video = await client.get(
                f"/recordings/{first}/video.mp4", headers={"Range": "bytes=2-5"}
            )
            assert video.status == 206 and await video.read() == b"2345"
            await manager.close()

    asyncio.run(check())


def test_recordings_api_is_404_without_recording_dir() -> None:
    async def check() -> None:
        manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            for path in ("/api/sessions", "/api/sessions/x", "/recordings/x/video.mp4"):
                assert (await client.get(path)).status == 404
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await asyncio.sleep(0.1)
                assert manager.recorder is None
            await manager.close()

    asyncio.run(check())


def test_recording_logs_incident_pause_resume_and_disconnect(monkeypatch, tmp_path) -> None:
    import json

    from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits, SafetyStop, Violation

    blocked = [_controller_frame(1, 0.0), _controller_frame(2, 0.5)]
    violation = Violation("speed", "right", 0.9, 0.25, "m/s")
    calls = {"step": 0}

    def step(_window):
        calls["step"] += 1
        if calls["step"] == 3:
            raise SafetyStop([violation], blocked)
        return SimpleNamespace(metadata={"action_rows": 2})

    manager = _recording_server(monkeypatch, tmp_path, step)
    manager.guard = SafetyGuard(SafetyLimits(max_speed_m_s=0.4))

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                # Hold generation until the recording has started.
                manager.pause()
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                assert recorder.safety["max_speed_m_s"] == 0.4
                await ws.send_json({"type": "resume"})
                message = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert message["type"] == "incident"
                await ws.send_json({"type": "resume"})
                await _until(lambda: calls["step"] > 3)
            await _until(lambda: recorder.end_reason is not None)
            assert manager.recorder is None and recorder.end_reason == "disconnect"
            events = [
                json.loads(line)
                for line in (recorder.path / "events.jsonl").read_text().splitlines()
            ]
            kinds = [event["kind"] for event in events]
            assert kinds[0] == "start" and kinds[-1] == "end"
            assert "incident" in kinds
            incident = events[kinds.index("incident")]
            assert incident["violations"] == [violation.to_json()]
            assert incident["blocked_frames"] == blocked
            resumes = [event["after"] for event in events if event["kind"] == "resume"]
            assert resumes == ["pause", "incident"]
            # Every event carries active time next to video time, and it never runs backwards.
            times = [event["t_s"] for event in events]
            assert times == sorted(times) and all("t_video_s" in event for event in events)
            await manager.close()

    asyncio.run(check())


def test_an_attempt_started_while_held_begins_held(monkeypatch, tmp_path) -> None:
    manager = _recording_server(
        monkeypatch, tmp_path, lambda _w: SimpleNamespace(metadata={"action_rows": 2})
    )
    manager.pause()  # paused before the attempt starts, and not cleared by starting
    manager.start_recording("Dr Lin")
    recorder = manager.recorder
    assert recorder is not None and recorder.held
    manager.resume()
    assert not recorder.held
    # The same for an incident hold, which the page's Resume ends the same way.
    manager.hold = "incident"
    manager.restart_recording("reset")  # a free-practice attempt restarts under the hold
    restarted = manager.recorder
    assert restarted is not None and restarted is not recorder and restarted.held
    manager._clear_hold()  # a reset, scene switch or disconnect drops the hold without a resume
    assert not restarted.held
    manager.end_recording("stop")
    assert restarted.end_reason == "stop"


def test_a_resume_right_after_an_incident_does_not_lose_the_hold_end(
    monkeypatch, tmp_path
) -> None:
    import json
    import threading

    from cosmos3_h_dreams.recording import SessionRecorder
    from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits, SafetyStop, Violation

    violation = Violation("speed", "right", 0.9, 0.25, "m/s")
    calls = {"step": 0}

    def step(_window):
        calls["step"] += 1
        if calls["step"] == 3:
            raise SafetyStop([violation], [_controller_frame(1, 0.0), _controller_frame(2, 0.5)])
        return SimpleNamespace(metadata={"action_rows": 2})

    manager = _recording_server(monkeypatch, tmp_path, step)
    manager.guard = SafetyGuard(SafetyLimits(max_speed_m_s=0.4))
    # The incident event is logged on a worker thread; hold it back so that the
    # page's Resume is handled (and logged) first.
    gate = threading.Event()
    log_event = SessionRecorder.event

    def slow_event(self, kind, **data):
        if kind == "incident":
            assert gate.wait(5)
        return log_event(self, kind, **data)

    monkeypatch.setattr(SessionRecorder, "event", slow_event)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            try:
                async with client.ws_connect("/ws?client=train") as ws:
                    manager.pause()
                    await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                    await _until(lambda: manager.recorder is not None)
                    recorder = manager.recorder
                    await ws.send_json({"type": "resume"})
                    await _until(lambda: manager.hold == "incident")
                    assert recorder.held  # held since the manager held, before the event is logged
                    await ws.send_json({"type": "resume"})
                    await _until(lambda: manager.hold is None)
                    assert not recorder.held
                    gate.set()
                    path = recorder.path / "events.jsonl"

                    def logged() -> list[str]:
                        return [json.loads(line)["kind"] for line in path.read_text().splitlines()]

                    await _until(lambda: "incident" in logged())
                    kinds = logged()
                    # The resume was logged before the incident, and the late event did not hold again.
                    assert [k for k in kinds if k in ("incident", "resume")] == [
                        "resume", "resume", "incident",
                    ]
                    assert not recorder.held
            finally:
                gate.set()
            await _until(lambda: recorder.end_reason is not None)
            await manager.close()

    asyncio.run(check())


def _switching_server(monkeypatch, tmp_path, *, fail_fp8: bool = False):
    """A recording server whose sessions come from a recording fake factory."""
    configs = []
    closed = []

    def factory(config):
        configs.append(config)

        def init():
            if fail_fp8 and config.fp8_generation:
                raise RuntimeError("out of memory")

        session = SimpleNamespace(
            config=config,
            model_status="Loading model…",
            init=init,
            next_step_requirements=lambda: SimpleNamespace(input_frame_count=1),
            step=lambda _w: SimpleNamespace(metadata={"action_rows": 2}),
        )
        session.close = lambda: closed.append(session)
        return session

    manager = _recording_server(
        monkeypatch, tmp_path, lambda _w: SimpleNamespace(metadata={"action_rows": 2})
    )
    first = manager.session
    first.close = lambda: closed.append(first)
    manager.session_factory = factory
    return manager, configs, closed


def test_mode_switch_reloads_session_and_restarts_recording(monkeypatch, tmp_path) -> None:
    manager, configs, closed = _switching_server(monkeypatch, tmp_path)
    first = manager.session

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                first_recording = manager.recorder.id
                health = await (await client.get("/healthz")).json()
                assert (health["mode"], health["mode_switching"], health["mode_error"]) == (
                    "standard", False, None,
                )

                response = await client.post("/mode", json={"mode": "low_memory"})
                assert response.status == 202
                assert await response.json() == {"mode_switching": True}
                health = await (await client.get("/healthz")).json()
                assert health["mode_switching"] is True
                for _ in range(300):
                    health = await (await client.get("/healthz")).json()
                    if not health["mode_switching"]:
                        break
                    await asyncio.sleep(0.01)
                assert health["mode"] == "low_memory"
                assert health["mode_switching"] is False
                assert health["mode_error"] is None
                assert first in closed
                assert [c.fp8_generation for c in configs] == [True]
                assert manager.session is not first and manager.runtime_ready
                assert manager.recorder is not None
                assert manager.recorder.id != first_recording
                assert manager.recorder.trainee == "Dr Lin"

                same = await client.post("/mode", json={"mode": "low_memory"})
                assert same.status == 200
                assert await same.json() == {"mode": "low_memory"}
                assert (await client.post("/mode", json={"mode": "turbo"})).status == 400
                assert (await client.post("/mode", data="nope")).status == 400
            sessions = await (await client.get("/api/sessions")).json()
            ended = {s["id"]: s["end_reason"] for s in sessions}
            assert ended[first_recording] == "mode_switch"
            await manager.close()

    asyncio.run(check())


def test_model_modes_map_to_fp8_settings_and_back() -> None:
    from cosmos3_h_dreams.headset_server import _mode_of, _with_mode
    from cosmos3_h_dreams.training_routes import MODEL_MODES

    assert list(MODEL_MODES) == ["standard", "balanced", "low_memory"]
    assert MODEL_MODES["standard"]["fp8_generation"] is False
    assert MODEL_MODES["balanced"] == {"fp8_generation": True, "fp8_scope": "mlp"}
    assert MODEL_MODES["low_memory"] == {"fp8_generation": True, "fp8_scope": "all"}
    base = SimpleNamespace(fps=10, fp8_generation=False)  # older configs lack fp8_scope
    assert _mode_of(base) == "standard"
    for mode in MODEL_MODES:
        assert _mode_of(_with_mode(base, mode)) == mode
    assert base.fp8_generation is False and not hasattr(base, "fp8_scope")


def test_mode_switch_keeps_steady_refresh_sdpa(monkeypatch, tmp_path) -> None:
    manager, configs, _closed = _switching_server(monkeypatch, tmp_path)
    manager.session.config.steady_refresh_sdpa = False
    manager._steady_refresh_sdpa = False  # as loaded from this config
    manager.scenes = {
        name: SimpleNamespace(
            name=name, fp8_generation=False, fp8_scope="all", steady_refresh_sdpa=False
        )
        for name in ("a", "b")
    }

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            health = await (await client.get("/healthz")).json()
            assert health["steady_refresh_sdpa"] is False
            for mode in ("balanced", "standard"):
                assert (await client.post("/mode", json={"mode": mode})).status == 202
                await _until(lambda: not manager.mode_switching)
                health = await (await client.get("/healthz")).json()
                assert (health["mode"], health["mode_error"]) == (mode, None)
                assert health["steady_refresh_sdpa"] is False
            await manager.close()

    asyncio.run(check())
    assert [c.steady_refresh_sdpa for c in configs] == [False, False]
    assert {c.steady_refresh_sdpa for c in manager.scenes.values()} == {False}


def test_healthz_steady_refresh_sdpa_defaults_on() -> None:
    from cosmos3_h_dreams.headset_server import _steady_refresh_sdpa

    assert _steady_refresh_sdpa(None) is True
    assert _steady_refresh_sdpa(SimpleNamespace(fp8_generation=False)) is True
    assert _steady_refresh_sdpa(SimpleNamespace(steady_refresh_sdpa=False)) is False


def test_mode_switch_cycles_through_balanced(monkeypatch, tmp_path) -> None:
    manager, configs, closed = _switching_server(monkeypatch, tmp_path)
    manager.scenes = {
        name: SimpleNamespace(name=name, fp8_generation=False, fp8_scope="all")
        for name in ("a", "b")
    }
    cycle = ["balanced", "low_memory", "standard", "balanced"]

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                for mode in cycle:
                    response = await client.post("/mode", json={"mode": mode})
                    assert response.status == 202
                    await _until(lambda: not manager.mode_switching)
                    health = await (await client.get("/healthz")).json()
                    assert (health["mode"], health["mode_error"]) == (mode, None)
                    await _until(lambda: manager.recorder is not None)
                    assert manager.recorder.mode == mode
                    settings = {
                        (c.fp8_generation, c.fp8_scope) for c in manager.scenes.values()
                    }
                    assert settings == {
                        (manager.session.config.fp8_generation, manager.session.config.fp8_scope)
                    }
            await manager.close()

    asyncio.run(check())
    assert [(c.fp8_generation, c.fp8_scope) for c in configs] == [
        (True, "mlp"), (True, "all"), (False, "all"), (True, "mlp"),
    ]
    assert [s.config for s in closed[: len(cycle)]][1:] == configs[:-1]  # each switch closes the last load


def test_failed_mode_switch_reloads_previous_mode(monkeypatch, tmp_path) -> None:
    manager, configs, closed = _switching_server(monkeypatch, tmp_path, fail_fp8=True)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            response = await client.post("/mode", json={"mode": "low_memory"})
            assert response.status == 202
            await _until(lambda: not manager.mode_switching)
            health = await (await client.get("/healthz")).json()
            assert health["mode"] == "standard"
            assert "out of memory" in health["mode_error"]
            assert [c.fp8_generation for c in configs] == [True, False]
            assert [s.config.fp8_generation for s in closed] == [False, True]
            assert manager.runtime_ready
            assert manager.session.config.fp8_generation is False
            await manager.close()

    asyncio.run(check())


def test_mode_switch_conflicts() -> None:
    async def check() -> None:
        manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            response = await client.post("/mode", json={"mode": "low_memory"})
            assert response.status == 409
            assert "not available" in response.reason
            manager.session_factory = lambda config: _idle_session()
            manager.runtime_ready = False
            assert (await client.post("/mode", json={"mode": "low_memory"})).status == 409
            manager.runtime_ready = True
            manager.mode_switching = True
            assert (await client.post("/mode", json={"mode": "low_memory"})).status == 409
            manager.mode_switching = False
            await manager.close()

    asyncio.run(check())


def test_input_rate_changes_sampling_playback_and_budget() -> None:
    async def check() -> None:
        session = _idle_session()
        session.actions_per_step = 8
        frames = LatestHeadsetFrames(fps=30)
        manager = HeadsetServer(session=session, frames=frames, display_fps=30)
        async with TestClient(TestServer(create_app(manager))) as client:
            health = await (await client.get("/healthz")).json()
            assert health["input_rate"] == {"fps": 30.0, "min": 5.0, "max": 60.0, "chunk_frames": 8}
            manager.latency.append((250.0, 30.0, 300.0))
            response = await client.post("/input-rate", json={"fps": 22})
            assert response.status == 200 and await response.json() == {"fps": 22.0}
            assert frames.fps == 22.0 and frames._period_ns == round(1e9 / 22)
            assert manager.display_fps == 22.0
            assert manager.latency_budget_ms() == pytest.approx(8000 / 22)
            assert not manager.latency  # measured against the old budget
            for bad in ({"fps": 4}, {"fps": 61}, {"fps": "24"}, {"fps": True}, {}, [24]):
                assert (await client.post("/input-rate", json=bad)).status == 400
            manager.recorder = SimpleNamespace()
            response = await client.post("/input-rate", json={"fps": 30})
            assert response.status == 409 and "Stop the attempt" in response.reason
            assert frames.fps == 22.0
            manager.recorder = None
            await manager.close()

    asyncio.run(check())


def test_configured_input_rate_above_60_raises_the_maximum() -> None:
    manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=72))
    assert manager.input_rate_health() == {"fps": 72.0, "min": 5.0, "max": 72.0, "chunk_frames": None}


def test_headset_frames_reopen_for_the_next_session() -> None:
    """A mode switch closes one action source and starts another on the same sampler."""
    pose = {"position": [0.0, 1.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0],
            "buttons": [{"v": 0.0}, {"v": 0.0}], "axes": []}
    xr_frame = {"type": "xr_frame", "controllers": {"left": pose, "right": pose}}
    frames = LatestHeadsetFrames(fps=None)
    assert frames.push(xr_frame)
    frames.stop()
    assert not frames.push(xr_frame)
    frames.start()
    assert frames.push(xr_frame)


def test_start_during_mode_switch_is_refused(monkeypatch, tmp_path) -> None:
    import threading

    manager, configs, _closed = _switching_server(monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    factory = manager.session_factory

    def blocking_factory(config):
        session = factory(config)

        def init():
            entered.set()
            assert release.wait(timeout=5)

        session.init = init
        return session

    manager.session_factory = blocking_factory

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                assert (await client.post("/mode", json={"mode": "low_memory"})).status == 202
                # Before the old session is unloaded, then while the new one loads.
                await ws.send_json({"type": "start", "trainee": "Intruder"})
                await _until(lambda: manager.recording_error is not None)
                assert await asyncio.to_thread(entered.wait, 3)
                manager.recording_error = None
                await ws.send_json({"type": "start", "trainee": "Intruder"})
                await _until(lambda: manager.recording_error is not None)
                assert manager.recorder is None
                assert not ws.closed
                release.set()
                await _until(lambda: not manager.mode_switching)
                assert manager.recorder is not None
                assert manager.recorder.trainee == "Dr Lin"
                assert manager.recorder.mode == "low_memory"
                await ws.send_json({"type": "stop"})
                await _until(lambda: manager.recorder is None)
                assert not ws.closed
            sessions = await (await client.get("/api/sessions")).json()
            assert sorted(s["trainee"] for s in sessions) == ["Dr Lin", "Dr Lin"]
            await manager.close()

    asyncio.run(check())


def test_unload_failure_reloads_previous_mode(monkeypatch, tmp_path) -> None:
    manager, configs, _closed = _switching_server(monkeypatch, tmp_path)

    def broken_close():
        raise RuntimeError("cache flush failed")

    manager.session.close = broken_close

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            assert (await client.post("/mode", json={"mode": "low_memory"})).status == 202
            await _until(lambda: not manager.mode_switching)
            health = await (await client.get("/healthz")).json()
            assert "cache flush failed" in health["mode_error"]
            assert health["mode"] == "standard"
            assert health["model_status"] != "Switching model…"
            assert [c.fp8_generation for c in configs] == [False]
            assert manager.runtime_ready
            await manager.close()

    asyncio.run(check())


def test_double_load_failure_reports_startup_failure(monkeypatch, tmp_path) -> None:
    manager, configs, closed = _switching_server(monkeypatch, tmp_path)

    def failing_factory(config):
        configs.append(config)

        def init():
            raise RuntimeError("no GPU")

        return SimpleNamespace(
            config=config, model_status="Loading model…", init=init,
            close=lambda: closed.append(config),
        )

    manager.session_factory = failing_factory

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            assert (await client.post("/mode", json={"mode": "low_memory"})).status == 202
            await _until(lambda: not manager.mode_switching)
            health = await (await client.get("/healthz")).json()
            assert "also failed" in health["mode_error"]
            assert health["model_status"] == "Model startup failed — check server logs"
            assert health["ready"] is False
            assert len(closed) == 3  # the original session and both failed loads
            await manager.close()

    asyncio.run(check())


def test_safety_stops_only_apply_to_the_training_client(monkeypatch) -> None:
    """The /headset page has no incident UI, so its motion is never refused."""
    import numpy as np

    from cosmos3_h_dreams import headset_server as server_module
    from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits, SafetyStop, Violation

    window = [_controller_frame(1, 0.0), _controller_frame(2, 0.5)]
    guard = SafetyGuard(SafetyLimits())
    monkeypatch.setattr(guard, "check", lambda _frames: [Violation("speed", "left", 5.0, 0.25, "m/s")])
    calls = {"step": 0}
    manager: HeadsetServer

    def step(_window):
        # What HeadsetActionSource does with the guard main() wires in.
        calls["step"] += 1
        violations = manager.check_safety(window)
        if violations:
            raise SafetyStop(violations, window)
        return SimpleNamespace(metadata={"action_rows": 1})

    session = SimpleNamespace(
        config=SimpleNamespace(fps=10, profile=False, fp8_generation=False, seed=0),
        model_status="Model ready",
        init=lambda: None,
        close=lambda: None,
        next_step_requirements=lambda: SimpleNamespace(input_frame_count=1),
        step=step,
    )
    monkeypatch.setattr(
        server_module.MjpegFrames, "prepare_result_frames",
        lambda _t, _r: (np.zeros((4, 6, 3), np.uint8),),
    )

    async def check() -> None:
        nonlocal manager
        manager = HeadsetServer(session=session, frames=LatestHeadsetFrames(fps=30), guard=guard)
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws") as ws:
                await _until(lambda: calls["step"] >= 5)
                assert manager.hold is None
                with pytest.raises(asyncio.TimeoutError):
                    await ws.receive_json(timeout=0.2)
                assert (await (await client.get("/healthz")).json())["hold"] is None
            await _until(lambda: manager.socket is None)
            async with client.ws_connect("/ws?client=train") as ws:
                message = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert message["type"] == "incident"
                assert manager.hold == "incident"
            await _until(lambda: manager.socket is None)
            assert manager.hold is None and not manager.safety_active
            await manager.close()

    asyncio.run(check())


def test_disconnect_during_incident_is_not_a_resume(monkeypatch, tmp_path) -> None:
    import json

    from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits, SafetyStop, Violation

    blocked = [_controller_frame(1, 0.0), _controller_frame(2, 0.5)]
    calls = {"step": 0}

    def step(_window):
        calls["step"] += 1
        if calls["step"] == 2:
            raise SafetyStop([Violation("speed", "right", 0.9, 0.25, "m/s")], blocked)
        return SimpleNamespace(metadata={"action_rows": 2})

    manager = _recording_server(monkeypatch, tmp_path, step)
    manager.guard = SafetyGuard(SafetyLimits())
    rewound = []
    manager.source.rewind = rewound.append
    resets = []
    monkeypatch.setattr(manager.guard, "reset", lambda: resets.append(True))

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                manager.pause()
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                await ws.send_json({"type": "resume"})
                message = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert message["type"] == "incident"
            await _until(lambda: recorder.end_reason is not None)
            assert manager.hold is None and resets and rewound == []
            events = [
                json.loads(line)
                for line in (recorder.path / "events.jsonl").read_text().splitlines()
            ]
            resumes = [event["after"] for event in events if event["kind"] == "resume"]
            assert resumes == ["pause"]
            await manager.close()

    asyncio.run(check())


def test_recording_failure_never_stops_generation(monkeypatch, tmp_path) -> None:
    calls = {"step": 0}

    def step(_window):
        calls["step"] += 1
        return SimpleNamespace(metadata={"action_rows": 2})

    manager = _recording_server(monkeypatch, tmp_path, step)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder

                def broken(*_args):
                    raise ValueError("cannot reshape actions")

                recorder.add_chunk = broken
                await _until(lambda: manager.recording_error is not None)
                assert "cannot reshape actions" in manager.recording_error
                assert manager.recorder is None and recorder.end_reason == "error"
                seen = calls["step"]
                await _until(lambda: calls["step"] > seen + 2)
                assert manager.runtime_ready
                health = await (await client.get("/healthz")).json()
                assert "cannot reshape actions" in health["recording_error"]
            await manager.close()

    asyncio.run(check())


def test_disconnect_during_mode_switch_does_not_revive_the_recording(
    monkeypatch, tmp_path
) -> None:
    import threading

    manager, _configs, _closed = _switching_server(monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    factory = manager.session_factory

    def blocking_factory(config):
        session = factory(config)

        def init():
            entered.set()
            assert release.wait(timeout=5)

        session.init = init
        return session

    manager.session_factory = blocking_factory

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                assert (await client.post("/mode", json={"mode": "low_memory"})).status == 202
                assert await asyncio.to_thread(entered.wait, 3)
            await _until(lambda: manager.socket is None)
            async with client.ws_connect("/ws"):
                # The next person connects before the switch finishes.
                release.set()
                await _until(lambda: not manager.mode_switching)
                assert manager.mode == "low_memory"
                assert manager.recorder is None
            await manager.close()

    asyncio.run(check())


# -- GPU monitor wiring -----------------------------------------------------------


def _monitor(used=1000.0, **kwargs):
    """A GpuMonitor whose reader reports ``used[0]`` MiB on GPU 0 at the current time."""
    import time

    from cosmos3_h_dreams.gpu_monitor import GpuMonitor, GpuSample

    level = [used]

    def reader():
        now = time.time()
        return [
            GpuSample(now, 0, level[0], 81559.0, 5.0, 40.0, level[0] / 2, level[0], level[0] * 2),
            GpuSample(now, 1, 10.0, 81559.0, 0.0, 30.0, None, None),
        ]

    # A long period: the thread samples once at start; tests sample by hand.
    monitor = GpuMonitor(reader=reader, period_s=60.0, **kwargs)
    return monitor, level


def test_gpu_api_shape_since_filter_and_errors() -> None:
    from cosmos3_h_dreams.gpu_monitor import GpuMonitor, GpuSample

    clock = [100.0]
    rounds = iter([[GpuSample(t, 0, used, 81559.0, 1.0, 40.0, None, None, used * 4)] for t, used in ((100.0, 500.0), (101.0, 900.0), (102.0, 700.0))])
    monitor = GpuMonitor(reader=lambda: next(rounds), clock=lambda: clock[0], period_s=60.0)
    for _ in range(3):
        monitor.sample_once()
    clock[0] = 101.5
    monitor.mark("attempt_start", "a1")

    async def check() -> None:
        manager = HeadsetServer(
            session=_idle_session(), frames=LatestHeadsetFrames(fps=30), gpu_monitor=monitor
        )
        manager.gpu_monitor.start = lambda: None  # keep the scripted samples only
        async with TestClient(TestServer(create_app(manager))) as client:
            response = await client.get("/api/gpu")
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            body = await response.json()
            assert set(body) == {
                "now", "device", "nvml", "gpus", "samples", "marks",
                "baseline", "peak", "host", "csv", "csv_error",
            }
            assert isinstance(body["now"], float)
            assert (body["device"], body["nvml"], body["csv"], body["csv_error"]) == (
                0, False, None, None,
            )
            assert body["gpus"] == [{"index": 0, "name": "GPU 0", "total_mib": 81559.0}]
            assert [s["used_mib"] for s in body["samples"]] == [500.0, 900.0, 700.0]
            assert body["samples"][0] == GpuSample(100.0, 0, 500.0, 81559.0, 1.0, 40.0, None, None, 2000.0).to_json()
            assert body["samples"][0]["host_pss_mib"] == 2000.0
            assert body["host"] == {"pss_mib": 2800.0, "peak_mib": 3600.0}
            assert body["marks"] == [{"t": 101.5, "kind": "attempt_start", "label": "a1"}]
            assert body["baseline"] == {"0": 500.0} and body["peak"] == {"0": 900.0}

            later = await (await client.get("/api/gpu?since=101")).json()
            assert [s["t"] for s in later["samples"]] == [102.0]
            assert later["marks"] == body["marks"]
            assert (await (await client.get("/api/gpu?since=102")).json())["samples"] == []
            for bad in ("nope", "nan", "inf", ""):
                assert (await client.get(f"/api/gpu?since={bad}")).status == 400, bad
            only = await (await client.get("/api/gpu?gpu=0&since=100")).json()
            assert [s["t"] for s in only["samples"]] == [101.0, 102.0]
            assert (await (await client.get("/api/gpu?gpu=3")).json())["samples"] == []
            for bad in ("x", "1.5", ""):
                assert (await client.get(f"/api/gpu?gpu={bad}")).status == 400, bad
            await manager.close()

        manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            response = await client.get("/api/gpu")
            assert response.status == 404
            assert await response.json() == {"error": "GPU monitor disabled"}
            assert (await (await client.get("/healthz")).json())["gpu"] is None
            await manager.close()

    asyncio.run(check())


def test_server_starts_and_stops_the_monitor_and_reports_it_in_healthz() -> None:
    monitor, level = _monitor(used=2048.0)
    started = []

    async def check() -> None:
        manager = HeadsetServer(
            session=_idle_session(), frames=LatestHeadsetFrames(fps=30), gpu_monitor=monitor
        )
        assert manager.gpu_monitor is monitor
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: monitor.samples())  # start() began sampling
            started.append(monitor._thread is not None)
            level[0] = 4096.0
            monitor.sample_once()
            level[0] = 3000.0
            monitor.sample_once()
            health = await (await client.get("/healthz")).json()
            assert health["gpu"] == {"used_mib": 3000.0, "total_mib": 81559.0, "peak_mib": 4096.0}
            await manager.close()
        assert monitor._stop.is_set() and monitor._thread is None

    asyncio.run(check())
    assert started == [True]


def test_baseline_is_sampled_before_the_model_loads() -> None:
    monitor, _level = _monitor(used=512.0)
    monitor.start = lambda: None  # no thread: only a synchronous sample can count
    seen = []
    session = _idle_session()
    session.init = lambda: seen.append(len(monitor.samples()))

    async def check() -> None:
        manager = HeadsetServer(
            session=session, frames=LatestHeadsetFrames(fps=30), gpu_monitor=monitor
        )
        async with TestClient(TestServer(create_app(manager))):
            await _until(lambda: seen)
            await manager.close()

    asyncio.run(check())
    assert seen[0] > 0
    assert monitor.baseline()[0] == 512.0


def test_baseline_sample_failure_still_starts_the_monitor(caplog) -> None:
    monitor, _level = _monitor()
    started = []

    def broken():
        raise RuntimeError("NVML gone")

    monitor.sample_once = broken
    monitor.start = lambda: started.append(True)

    async def check() -> None:
        manager = HeadsetServer(
            session=_idle_session(), frames=LatestHeadsetFrames(fps=30), gpu_monitor=monitor
        )
        async with TestClient(TestServer(create_app(manager))):
            await _until(lambda: manager.runtime_ready)
            await manager.close()

    with caplog.at_level(logging.WARNING):
        asyncio.run(check())
    assert started == [True]
    assert "baseline" in caplog.text


def test_attempt_end_mark_matches_the_stats_window(monkeypatch, tmp_path) -> None:
    manager = _recording_server(
        monkeypatch, tmp_path, lambda _w: SimpleNamespace(metadata={"action_rows": 2})
    )
    monitor, _level = _monitor()
    manager.gpu_monitor = monitor
    windows = []
    stats = monitor.stats

    def recording_stats(t0, t1):
        windows.append((t0, t1))
        return stats(t0, t1)

    monkeypatch.setattr(monitor, "stats", recording_stats)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                await ws.send_json({"type": "stop"})
                await _until(lambda: recorder.end_reason is not None)
            await manager.close()

    asyncio.run(check())
    marks = {m["kind"]: m["t"] for m in monitor.marks()}
    assert windows == [(marks["attempt_start"], marks["attempt_end"])]


def test_attempts_and_incidents_add_marks_and_vram_reaches_session_json(
    monkeypatch, tmp_path
) -> None:
    import json

    from cosmos3_h_dreams.safety import SafetyGuard, SafetyLimits, SafetyStop, Violation

    blocked = [_controller_frame(1, 0.0), _controller_frame(2, 0.5)]
    violation = Violation("speed", "right", 0.9, 0.25, "m/s")
    calls = {"step": 0}

    def step(_window):
        calls["step"] += 1
        if calls["step"] == 2:
            raise SafetyStop([violation], blocked)
        return SimpleNamespace(metadata={"action_rows": 2})

    manager = _recording_server(monkeypatch, tmp_path, step)
    manager.guard = SafetyGuard(SafetyLimits())
    monitor, level = _monitor(used=1000.0)
    manager.gpu_monitor = monitor

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: monitor.samples())  # the pre-attempt sample: 1000 MiB
            async with client.ws_connect("/ws?client=train") as ws:
                manager.pause()
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                for used in (3000.0, 5000.0):
                    level[0] = used
                    monitor.sample_once()
                await ws.send_json({"type": "resume"})
                message = await asyncio.wait_for(ws.receive_json(), timeout=3)
                assert message["type"] == "incident"
                await ws.send_json({"type": "stop"})
                await _until(lambda: recorder.end_reason is not None)
            level[0] = 9000.0  # after the attempt: outside its window
            monitor.sample_once()
            marks = [(m["kind"], m["label"]) for m in monitor.marks()]
            assert marks == [
                ("attempt_start", recorder.id),
                ("incident", violation.message()),
                ("attempt_end", recorder.id),
            ]
            session = json.loads((recorder.path / "session.json").read_text())
            assert session["summary"]["vram"] == {
                "peak_used_mib": 5000.0,
                "mean_used_mib": 4000.0,
                "peak_allocated_mib": 2500.0,
                "mean_allocated_mib": 2000.0,
                "peak_host_pss_mib": 10000.0,
                "mean_host_pss_mib": 8000.0,
            }
            await manager.close()

    asyncio.run(check())


def test_recording_without_monitor_has_no_vram(monkeypatch, tmp_path) -> None:
    import json

    manager = _recording_server(
        monkeypatch, tmp_path, lambda _w: SimpleNamespace(metadata={"action_rows": 2})
    )

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                await ws.send_json({"type": "stop"})
                await _until(lambda: recorder.end_reason is not None)
            session = json.loads((recorder.path / "session.json").read_text())
            assert "vram" not in session["summary"]
            await manager.close()

    asyncio.run(check())


def test_broken_monitor_never_breaks_recording(monkeypatch, tmp_path) -> None:
    manager = _recording_server(
        monkeypatch, tmp_path, lambda _w: SimpleNamespace(metadata={"action_rows": 2})
    )
    monitor, _level = _monitor()
    manager.gpu_monitor = monitor

    def broken(*_args, **_kwargs):
        raise RuntimeError("monitor broke")

    monkeypatch.setattr(monitor, "mark", broken)
    monkeypatch.setattr(monitor, "stats", broken)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                await ws.send_json({"type": "stop"})
                await _until(lambda: recorder.end_reason == "stop")
            assert manager.recording_error is None
            await manager.close()

    asyncio.run(check())


def test_mode_switch_adds_marks(monkeypatch, tmp_path) -> None:
    for fail, end_label in ((False, "low_memory"), (True, "failed: standard")):
        manager, _configs, _closed = _switching_server(monkeypatch, tmp_path, fail_fp8=fail)
        monitor, _level = _monitor()
        manager.gpu_monitor = monitor

        async def check() -> None:
            async with TestClient(TestServer(create_app(manager))) as client:
                await _until(lambda: manager.runtime_ready)
                assert (await client.post("/mode", json={"mode": "low_memory"})).status == 202
                await _until(lambda: not manager.mode_switching)
                await manager.close()

        asyncio.run(check())
        assert [(m["kind"], m["label"]) for m in monitor.marks()] == [
            ("mode_switch_start", "low_memory"),
            ("mode_switch_end", end_label),
        ]


def test_server_close_mid_switch_marks_the_switch_cancelled(monkeypatch, tmp_path) -> None:
    import threading

    manager, _configs, _closed = _switching_server(monkeypatch, tmp_path)
    monitor, _level = _monitor()
    manager.gpu_monitor = monitor
    entered, release = threading.Event(), threading.Event()
    factory = manager.session_factory

    def blocking_factory(config):
        session = factory(config)

        def init():
            entered.set()
            assert release.wait(timeout=5)

        session.init = init
        return session

    manager.session_factory = blocking_factory

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            assert (await client.post("/mode", json={"mode": "low_memory"})).status == 202
            assert await asyncio.to_thread(entered.wait, 3)
            closing = asyncio.create_task(manager.close())
            await _until(lambda: manager._closed)
            release.set()
            await closing

    asyncio.run(check())
    assert [(m["kind"], m["label"]) for m in monitor.marks()] == [
        ("mode_switch_start", "low_memory"),
        ("mode_switch_end", "cancelled"),
    ]


def test_nvml_index_warns_when_no_uuid_matches(caplog) -> None:
    from cosmos3_h_dreams.headset_server import _nvml_index

    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUUID=lambda _handle: "GPU-aaaa-0000",
    )
    cuda = SimpleNamespace(get_device_properties=lambda _i: SimpleNamespace(uuid="zzzz-9999"))
    with caplog.at_level(logging.WARNING):
        assert _nvml_index(3, cuda=cuda, nvml=nvml) == 3
    assert len(caplog.records) == 1
    assert "no NVML device" in caplog.text and "3" in caplog.text
    caplog.clear()
    matching = SimpleNamespace(get_device_properties=lambda _i: SimpleNamespace(uuid="aaaa-0000"))
    with caplog.at_level(logging.WARNING):
        assert _nvml_index(0, cuda=matching, nvml=nvml) == 0
    assert caplog.records == []


def test_nvml_index_matches_torch_uuid() -> None:
    from cosmos3_h_dreams.headset_server import _nvml_index

    uuids = [b"GPU-aaaa-0000", "GPU-BBBB-1111", "GPU-cccc-2222"]
    calls = []

    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetCount=lambda: len(uuids),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUUID=lambda handle: uuids[handle],
    )

    def cuda(uuid):
        return SimpleNamespace(get_device_properties=lambda _i: SimpleNamespace(uuid=uuid))

    # CUDA_VISIBLE_DEVICES=1,2: torch device 0 is NVML device 1.
    assert _nvml_index(0, cuda=cuda("bbbb-1111"), nvml=nvml) == 1
    assert calls == ["init", "shutdown"]
    assert _nvml_index(0, cuda=cuda("GPU-cccc-2222"), nvml=nvml) == 2
    assert _nvml_index(3, cuda=cuda("dddd-3333"), nvml=nvml) == 3  # no match
    assert _nvml_index(1, cuda=cuda("aaaa-0000"), nvml=None) == 1  # no NVML

    def broken(_i):
        raise RuntimeError("no device")

    assert _nvml_index(2, cuda=SimpleNamespace(get_device_properties=broken), nvml=nvml) == 2


def test_main_creates_the_gpu_monitor(tmp_path, monkeypatch) -> None:
    import sys

    from cosmos3_h_dreams import headset_server as server_module

    frame = tmp_path / "frame.png"
    frame.touch()
    created = []

    class FakeMonitor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.csv_path = kwargs.get("csv_path")
            created.append(self)

    monkeypatch.setattr(server_module, "GpuMonitor", FakeMonitor)
    monkeypatch.setattr(server_module, "_server_gpu", lambda device: (3, 5))
    monkeypatch.setattr(
        server_module.ssl,
        "SSLContext",
        lambda _protocol: SimpleNamespace(load_cert_chain=lambda *_args: None),
    )
    monkeypatch.setattr(server_module, "create_app", lambda manager: manager)
    managers = []
    monkeypatch.setattr(server_module.web, "run_app", lambda m, **_kw: managers.append(m))
    monkeypatch.chdir(tmp_path)
    base = ["headset_server", "--cert", "c.pem", "--key", "k.pem",
            "--checkpoint", "/checkpoint", "--initial-frame", str(frame)]

    monkeypatch.setattr(sys, "argv", [*base, "--gpu-period", "0.5"])
    server_module.main()
    (monitor,) = created
    assert managers[-1].gpu_monitor is monitor
    assert monitor.kwargs["device"] == 5 and monitor.kwargs["torch_device"] == 3
    assert monitor.kwargs["period_s"] == 0.5
    assert monitor.kwargs["history"] == 3600  # 30 minutes at 0.5 s
    csv_path = monitor.kwargs["csv_path"]
    assert csv_path.parent == tmp_path / "recordings"
    assert csv_path.name.startswith("gpu-") and csv_path.suffix == ".csv"
    assert len(csv_path.stem) == len("gpu-20261027-143005")

    for period, history in (("7", 257), ("4000", 1)):
        monkeypatch.setattr(sys, "argv", [*base, "--gpu-period", period])
        server_module.main()
        assert created[-1].kwargs["history"] == history

    monkeypatch.setattr(sys, "argv", [*base, "--no-gpu-monitor"])
    server_module.main()
    assert len(created) == 3 and managers[-1].gpu_monitor is None

    for bad in ("0", "-1", "nan"):
        monkeypatch.setattr(sys, "argv", [*base, "--gpu-period", bad])
        with pytest.raises(SystemExit):
            server_module.main()


# -- text cache --------------------------------------------------------------------


def test_healthz_reports_the_sessions_text_cache_and_null_without_one() -> None:
    session = _idle_session()
    session.text_cache = {
        "status": "built", "rehashed": False, "path": "/c/t.pt", "hash_s": 16.8, "load_s": 7.9,
    }

    async def check() -> None:
        manager = HeadsetServer(session=session, frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            health = await (await client.get("/healthz")).json()
            assert health["text_cache"] == {
                "status": "built", "rehashed": False, "path": "/c/t.pt",
                "hash_s": 16.8, "load_s": 7.9,
            }
            manager.session = None  # unloaded mid mode switch
            assert (await (await client.get("/healthz")).json())["text_cache"] is None
            manager.session = _idle_session()  # a session without the attribute
            assert (await (await client.get("/healthz")).json())["text_cache"] is None
            await manager.close()

    asyncio.run(check())


def _main_with_scenes(tmp_path, monkeypatch, scenes, runtime_extra=""):
    import sys

    from cosmos3_h_dreams import headset_server as server_module

    (tmp_path / "a.png").touch()
    (tmp_path / "b.png").touch()
    config = tmp_path / "session.yaml"
    config.write_text(
        "runtime:\n"
        "  checkpoint: /checkpoint\n"
        "  text_cache: cache/text.pt\n"
        f"{runtime_extra}"
        "headset:\n  cert: c.pem\n  key: k.pem\n"
        "gpu_monitor:\n  enabled: false\n"
        f"scenes:\n{scenes}"
    )
    monkeypatch.setattr(
        server_module.ssl,
        "SSLContext",
        lambda _protocol: SimpleNamespace(load_cert_chain=lambda *_args: None),
    )
    monkeypatch.setattr(server_module, "create_app", lambda manager: manager)
    managers = []
    monkeypatch.setattr(server_module.web, "run_app", lambda m, **_kw: managers.append(m))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["headset_server", "--session-config", str(config)])
    server_module.main()
    return managers


def test_main_refuses_a_text_cache_with_scenes_of_different_prompts(
    tmp_path, monkeypatch, capsys
) -> None:
    scenes = (
        "  - {name: a, initial_frame: a.png}\n"
        "  - {name: b, initial_frame: b.png, prompt: Another prompt.}\n"
    )
    with pytest.raises(SystemExit):
        _main_with_scenes(tmp_path, monkeypatch, scenes)
    error = capsys.readouterr().err
    assert "text_cache" in error and "'b'" in error and "'a'" in error
    assert "unify the prompts" in error


def test_main_accepts_a_text_cache_when_every_scene_shares_the_prompt(
    tmp_path, monkeypatch
) -> None:
    scenes = (
        "  - {name: a, initial_frame: a.png, prompt: Same.}\n"
        "  - {name: b, initial_frame: b.png, prompt: Same.}\n"
    )
    (manager,) = _main_with_scenes(tmp_path, monkeypatch, scenes)
    assert manager.session.config.text_cache == tmp_path / "cache" / "text.pt"
    assert set(manager.scenes) == {"a", "b"}


# -- training courses --------------------------------------------------------------

COURSES_DIR = Path(__file__).resolve().parents[1] / "courses"


def _main_with_courses(tmp_path, monkeypatch, courses_block):
    import sys

    from cosmos3_h_dreams import headset_server as server_module

    (tmp_path / "a.png").touch()
    config = tmp_path / "session.yaml"
    config.write_text(
        "runtime:\n  checkpoint: /checkpoint\n"
        "headset:\n  cert: c.pem\n  key: k.pem\n"
        "gpu_monitor:\n  enabled: false\n"
        f"{courses_block}"
        "scenes:\n  - {name: a, initial_frame: a.png}\n"
    )
    monkeypatch.setattr(
        server_module.ssl,
        "SSLContext",
        lambda _protocol: SimpleNamespace(load_cert_chain=lambda *_args: None),
    )
    monkeypatch.setattr(server_module, "create_app", lambda manager: manager)
    managers = []
    monkeypatch.setattr(server_module.web, "run_app", lambda m, **_kw: managers.append(m))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["headset_server", "--session-config", str(config)])
    server_module.main()
    return managers


def test_main_loads_courses_and_refuses_invalid_ones(tmp_path, monkeypatch, capsys) -> None:
    import shutil

    (manager,) = _main_with_courses(tmp_path, monkeypatch, "")
    assert manager.exercises is None and manager.demos_dir is None
    shutil.copytree(COURSES_DIR, tmp_path / "courses")
    with pytest.raises(SystemExit):  # the drag exercise's scene is not among this server's scenes
        _main_with_courses(tmp_path, monkeypatch, "courses:\n  dir: courses\n")
    assert "05-drag-the-gallbladder.yaml: scene: no scene named 'chole-15' (scenes: a)" in (
        capsys.readouterr().err
    )
    (tmp_path / "courses" / "starter" / "05-drag-the-gallbladder.yaml").unlink()
    (manager,) = _main_with_courses(
        tmp_path, monkeypatch, "courses:\n  dir: courses\n  demos_dir: demos\n"
    )
    assert list(manager.exercises)[0] == "instrument-basics"
    assert manager.demos_dir == tmp_path / "demos"
    assert manager.course_scene == "a"  # the first scene by default
    with pytest.raises(SystemExit):
        _main_with_courses(tmp_path, monkeypatch, "courses:\n  dir: courses\n  scene: b\n")
    assert "courses.scene: no scene named 'b' (scenes: a)" in capsys.readouterr().err
    broken = tmp_path / "courses" / "starter" / "02-grasp-and-hold.yaml"
    broken.write_text(broken.read_text().replace("points: 20", "points: 0", 1))
    with pytest.raises(SystemExit):
        _main_with_courses(tmp_path, monkeypatch, "courses:\n  dir: courses\n")
    error = capsys.readouterr().err
    assert "02-grasp-and-hold.yaml: rubric[1].points: must be a positive integer" in error


def _course_server(monkeypatch, tmp_path, step=None):
    from cosmos3_h_dreams.courses import load_courses

    manager = _recording_server(
        monkeypatch,
        tmp_path / "recordings",
        step or (lambda _w: SimpleNamespace(metadata={"action_rows": 2})),
    )
    manager.exercises = load_courses(COURSES_DIR, tmp_path / "demos")
    manager.demos_dir = tmp_path / "demos"
    return manager


def test_simulation_readiness_counts_chunks_after_each_reset(monkeypatch, tmp_path) -> None:
    restarts = []

    def step(_window):
        rows = 0 if restarts and restarts[-1] else 2  # the step after a restart calibrates
        if restarts:
            restarts[-1] = False
        return SimpleNamespace(metadata={"action_rows": rows})

    manager = _recording_server(monkeypatch, tmp_path, step)
    manager.source.request_restart = lambda: restarts.append(True)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                connected = manager.simulation_epoch
                assert connected >= 1
                await ws.send_json({"type": "reset"})
                await _until(lambda: manager.simulation_epoch == connected + 1)
                # Chunks count only once this epoch's restart has calibrated.
                await _until(lambda: manager.chunks_shown >= 2)
                simulation = (await (await client.get("/healthz")).json())["simulation"]
                assert simulation["epoch"] == connected + 1
                assert simulation["chunks_shown"] >= 2
                assert simulation["calibrated"] is False  # this fake source has no calibration
                await ws.send_json({"type": "reset"})
                await _until(lambda: manager.simulation_epoch == connected + 2)
                await _until(lambda: manager.chunks_shown >= 1)  # counted again after recalibrating

    asyncio.run(check())


def test_exercises_run_only_on_the_course_scene(monkeypatch, tmp_path) -> None:
    from aiohttp import web

    manager = _course_server(monkeypatch, tmp_path)
    manager.scenes = {"a": object(), "b": object()}
    manager.active_scene = "b"
    manager.course_scene = "a"

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                await ws.send_json(
                    {"type": "start", "trainee": "Dr Lin", "exercise": "instrument-basics"}
                )
                assert await ws.receive_json(timeout=3) == {
                    "type": "error",
                    "message": "This exercise runs on the 'a' scene; switch to it first",
                }
                assert manager.recorder is None
            assert (await (await client.get("/api/courses")).json())["scene"] == "a"
            # The scene is fixed while an exercise attempt runs.
            manager.runtime_ready = True
            producer, manager._producer = manager._producer, SimpleNamespace(done=lambda: False)
            manager.recorder = SimpleNamespace(exercise=object())
            with pytest.raises(web.HTTPConflict, match="Stop the exercise"):
                await manager.select_scene("a")
            manager.recorder, manager._producer = None, producer

    asyncio.run(check())


def test_exercise_attempt_logs_steps_and_writes_the_report(monkeypatch, tmp_path) -> None:
    import json

    manager = _course_server(monkeypatch, tmp_path)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                for exercise in ("nope", "safety-stop", 3):  # unknown, demo only, not a string
                    await ws.send_json({"type": "start", "trainee": "Dr Lin", "exercise": exercise})
                    assert await ws.receive_json(timeout=3) == {
                        "type": "error", "message": "Unknown exercise",
                    }
                assert manager.recorder is None
                await ws.send_json(
                    {"type": "start", "trainee": "Dr Lin", "exercise": "instrument-basics"}
                )
                await _until(
                    lambda: manager.recorder is not None and manager.recorder.video_time_s > 0
                )
                recorder = manager.recorder
                health = await (await client.get("/healthz")).json()
                assert health["recording"]["exercise"] == "instrument-basics"
                for index in (0, 0, 7, "1", True, 1):
                    await ws.send_json({"type": "step", "index": index, "t_client": 12.5})
                await ws.send_json({"type": "stop"})
                await _until(lambda: recorder.end_reason is not None)
            data = json.loads((recorder.path / "session.json").read_text())
            assert data["exercise"] == {"id": "instrument-basics", "title": "Instrument basics"}
            steps = [e for e in json.loads("[" + ",".join(
                (recorder.path / "events.jsonl").read_text().splitlines()) + "]") if e["kind"] == "step"]
            assert [(e["index"], e["t_client"]) for e in steps] == [(0, 12.5), (0, 12.5), (1, 12.5)]
            report = data["report"]
            assert [s["index"] for s in report["steps"]] == [0, 1]
            assert report["steps_done"] == 2 and report["near_misses"] == 0
            # Left at x 0.01/0.02 is 0.16/0.17 m from its demo path at home; right at
            # x 0.1 is 0.05 m from the right path's start (home).
            assert report["path_deviation_m"] == pytest.approx(0.1075, abs=1e-3)
            # 2 of 4 steps: all_steps fails and gates the other automatic items.
            assert [(i["id"], i["passed"], i["points"], i["gated"]) for i in report["items"]] == [
                ("all_steps", False, 0, False), ("no_stops", False, 0, True),
                ("time", False, 0, True), ("path", False, 0, True), ("smooth", None, 0, False),
            ]
            assert (report["auto_points"], report["auto_max"], report["instructor_max"]) == (0, 80, 20)
            await manager.close()

    asyncio.run(check())


def test_reset_ends_an_exercise_attempt_without_a_new_one(monkeypatch, tmp_path) -> None:
    manager = _course_server(monkeypatch, tmp_path)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                await ws.send_json(
                    {"type": "start", "trainee": "Dr Lin", "exercise": "grasp-and-hold"}
                )
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                await ws.send_json({"type": "reset"})
                await _until(lambda: recorder.end_reason is not None)
                await asyncio.sleep(0.1)
                assert recorder.end_reason == "reset" and manager.recorder is None
                assert recorder.close("again")["report"]["steps_done"] == 0
                await ws.send_json({"type": "start", "trainee": "Dr Lin"})  # free practice
                await _until(lambda: manager.recorder is not None)
                free = manager.recorder
                assert free.exercise is None
                await ws.send_json({"type": "step", "index": 0})  # ignored outside an exercise
                await ws.send_json({"type": "reset"})
                await _until(lambda: manager.recorder is not None and manager.recorder is not free)
                assert "report" not in free.close("again")
            await manager.close()

    asyncio.run(check())


def test_mode_switch_ends_an_exercise_attempt_without_a_new_one(monkeypatch, tmp_path) -> None:
    from cosmos3_h_dreams.courses import load_courses

    manager, _configs, _closed = _switching_server(monkeypatch, tmp_path / "recordings")
    manager.exercises = load_courses(COURSES_DIR, tmp_path / "demos")

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            async with client.ws_connect("/ws") as ws:
                await ws.send_json(
                    {"type": "start", "trainee": "Dr Lin", "exercise": "controlled-energy"}
                )
                await _until(lambda: manager.recorder is not None)
                recorder = manager.recorder
                assert (await client.post("/mode", json={"mode": "low_memory"})).status == 202
                await _until(lambda: not manager.mode_switching, timeout=5)
                assert recorder.end_reason == "mode_switch" and manager.recorder is None
            await manager.close()

    asyncio.run(check())


async def _record_attempt(manager, ws, exercise=None):
    """Run one attempt over ``ws`` until it has frames, stop it, return its recorder."""
    message = {"type": "start", "trainee": "Dr Lin"}
    if exercise:
        message["exercise"] = exercise
    await ws.send_json(message)
    await _until(lambda: manager.recorder is not None and manager.recorder.video_time_s > 0)
    recorder = manager.recorder
    await ws.send_json({"type": "stop"})
    await _until(lambda: recorder.end_reason is not None)
    return recorder


def test_courses_and_demo_routes(monkeypatch, tmp_path) -> None:
    manager = _course_server(monkeypatch, tmp_path)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            body = await (await client.get("/api/courses")).json()
            (course,) = body["courses"]
            assert course["name"] == "Starter course" and body["demos"] == []
            ids = [e["id"] for e in course["exercises"]]
            assert ids == [
                "instrument-basics",
                "grasp-and-hold",
                "controlled-energy",
                "safety-stop",
                "drag-the-gallbladder",
            ]
            assert course["exercises"][3]["demo_only"] is True
            demo = await client.get("/api/courses/instrument-basics/demo")
            assert demo.headers["Cache-Control"] == "no-store"
            demo = await demo.json()
            assert demo["source"] == "scripted" and demo["frames"][0]["t_s"] == 0
            assert demo["markers"][0] == {"t_s": 0.0, "caption": "Start at home"}
            for path in ("/api/courses/nope/demo", "/api/demos/nope", "/api/demos/..%2Fx"):
                assert (await client.get(path)).status == 404, path

            async with client.ws_connect("/ws?client=train") as ws:
                recorder = await _record_attempt(manager, ws)
            response = await client.post(f"/api/sessions/{recorder.id}/demo", json={"name": "Demo A"})
            assert response.status == 201
            saved = await response.json()
            assert saved["slug"] == "demo-a" and saved["source"] == recorder.id
            again = await client.post(f"/api/sessions/{recorder.id}/demo", json={"name": "demo a"})
            assert again.status == 409
            assert "already exists" in (await again.json())["error"]
            for payload in ({"name": "!!"}, {"name": ""}, "nope", {}):
                bad = await client.post(f"/api/sessions/{recorder.id}/demo", json=payload)
                assert bad.status == 400, payload
            assert (await client.post("/api/sessions/nope/demo", json={"name": "x"})).status == 404
            (recorder.path / "controller.jsonl").unlink()
            empty = await client.post(f"/api/sessions/{recorder.id}/demo", json={"name": "Empty"})
            assert empty.status == 400 and "no controller frames" in (await empty.json())["error"]

            listed = (await (await client.get("/api/courses")).json())["demos"]
            assert [d["slug"] for d in listed] == ["demo-a"] and listed[0]["name"] == "Demo A"
            played = await (await client.get("/api/demos/demo-a")).json()
            assert played["source"] == "recorded" and played["title"] == "Demo A"
            assert played["frames"][0]["right"]["position"] == [0.1, 1.0, -0.3]
            await manager.close()

    asyncio.run(check())


def test_save_as_demo_write_failure_is_a_500(monkeypatch, tmp_path) -> None:
    from cosmos3_h_dreams import courses as courses_module

    manager = _course_server(monkeypatch, tmp_path)

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                recorder = await _record_attempt(manager, ws)
            monkeypatch.setattr(courses_module.Path, "mkdir", broken)
            response = await client.post(f"/api/sessions/{recorder.id}/demo", json={"name": "Demo"})
            assert response.status == 500
            assert (await response.json())["error"] == "Could not save the demo: disk full"
            await manager.close()

    asyncio.run(check())


def test_review_route_and_session_detail(monkeypatch, tmp_path) -> None:
    manager = _course_server(monkeypatch, tmp_path)

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws?client=train") as ws:
                tagged = await _record_attempt(manager, ws, "instrument-basics")
                free = await _record_attempt(manager, ws)
            url = f"/api/sessions/{tagged.id}"
            detail = await (await client.get(url)).json()
            assert detail["exercise"] == {"id": "instrument-basics", "title": "Instrument basics"}
            assert detail["report"]["instructor_max"] == 20 and detail["review"] is None
            for payload, status in (
                ({"checked": ["no_stops"]}, 400),        # an automatic item
                ({"checked": ["nope"]}, 400),            # unknown
                ({"notes": "x" * 2001}, 400),
                ("nope", 400),
            ):
                response = await client.post(f"{url}/review", json=payload)
                assert response.status == status, payload
            response = await client.post(f"{url}/review", json={"checked": ["smooth"], "notes": "Calm"})
            assert response.status == 200
            review = await response.json()
            assert (review["checked"], review["points"], review["instructor_max"]) == (["smooth"], 20, 20)
            detail = await (await client.get(url)).json()
            assert detail["review"]["notes"] == "Calm" and detail["session"]["review"]["points"] == 20
            listed = {s["id"]: s for s in await (await client.get("/api/sessions")).json()}
            assert listed[tagged.id]["review"]["points"] == 20 and listed[free.id]["review"] is None
            free_detail = await (await client.get(f"/api/sessions/{free.id}")).json()
            assert free_detail["exercise"] is None and free_detail["report"] is None
            response = await client.post(f"/api/sessions/{free.id}/review", json={"checked": []})
            assert response.status == 400
            assert (await client.post("/api/sessions/nope/review", json={})).status == 404
            await manager.close()

    asyncio.run(check())


def test_course_routes_are_404_without_courses() -> None:
    async def check() -> None:
        manager = HeadsetServer(session=_idle_session(), frames=LatestHeadsetFrames(fps=30))
        async with TestClient(TestServer(create_app(manager))) as client:
            for path in ("/api/courses", "/api/courses/instrument-basics/demo", "/api/demos/a"):
                assert (await client.get(path)).status == 404, path
            for path in ("/api/sessions/x/demo", "/api/sessions/x/review"):
                assert (await client.post(path, json={})).status == 404, path
            await manager.close()

    asyncio.run(check())


def test_safety_ratios_reach_only_the_training_client_and_log_near_misses(
    monkeypatch, tmp_path
) -> None:
    import json

    from cosmos3_h_dreams.safety import RULES, SafetyCheck, SafetyGuard, SafetyLimits, SafetyStop

    window = [_controller_frame(1, 0.0), _controller_frame(2, 0.0)]
    manager: HeadsetServer

    def step(_window):
        # What HeadsetActionSource does with the guard main() wires in.
        violations = manager.check_safety(window)
        if violations:
            raise SafetyStop(violations, window)
        return SimpleNamespace(metadata={"action_rows": 2})

    manager = _course_server(monkeypatch, tmp_path, step)
    manager.guard = SafetyGuard(SafetyLimits())
    monkeypatch.setattr(
        manager.guard, "check", lambda _frames: SafetyCheck([], ratios={"speed": 0.85})
    )

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            async with client.ws_connect("/ws") as ws:  # the headset page: no meter
                with pytest.raises(asyncio.TimeoutError):
                    await ws.receive_json(timeout=0.3)
            await _until(lambda: manager.socket is None)
            async with client.ws_connect("/ws?client=train") as ws:
                await ws.send_json({"type": "start", "trainee": "Dr Lin", "exercise": "controlled-energy"})
                message = await ws.receive_json(timeout=3)
                assert message["type"] == "safety"
                assert list(message["ratios"]) == list(RULES)
                assert message["ratios"]["speed"] == 0.85 and message["ratios"]["energy_dose"] == 0.0
                await _until(lambda: manager.recorder is not None and manager.recorder.video_time_s > 0.5)
                recorder = manager.recorder
                await ws.send_json({"type": "stop"})
                await _until(lambda: recorder.end_reason is not None)
            events = [json.loads(line) for line in (recorder.path / "events.jsonl").read_text().splitlines()]
            near = [e for e in events if e["kind"] == "near_miss"]
            assert [(e["rule"], e["ratio"]) for e in near] == [("speed", 0.85)]  # once while it stays high
            report = json.loads((recorder.path / "session.json").read_text())["report"]
            assert report["near_misses"] == 1
            assert next(i for i in report["items"] if i["id"] == "near")["passed"] is False
            await manager.close()

    asyncio.run(check())


def test_latency_medians_status_and_budget() -> None:
    from collections import deque

    from cosmos3_h_dreams.headset_server import (
        LATENCY_WINDOW,
        _health_latency,
        _latency_budget_ms,
    )

    budget = _latency_budget_ms(8, 30.0)  # 8 action rows at 30 Hz
    assert abs(budget - 266.67) < 0.01
    assert _health_latency([], budget) is None
    window = deque(maxlen=LATENCY_WINDOW)
    window.append((900.0, 112.5, None))  # an old slow chunk the window forgets
    for chunk_ms in (80.0, 84.0, 82.0) * 7:
        window.append((chunk_ms, chunk_ms / 8, 300.0))
    assert len(window) == LATENCY_WINDOW == 20
    assert _health_latency(window, budget, 8 / 30 * 1e3) == {
        "chunk_ms": 82.0, "frame_ms": 10.2, "response_ms": 300.0,
        "budget_ms": 266.7, "playback_ms": 266.7, "status": "ok", "reason": None,
    }

    def status(chunk_ms, playback_ms=100.0):
        latency = _health_latency([(chunk_ms, 1.0, None)], 100.0, playback_ms)
        return latency["status"], latency["reason"]

    assert [status(x) for x in (79.9, 80.0, 99.9, 100.0, 250.0)] == [
        ("ok", None), ("tight", None), ("tight", None),
        ("behind", "generation"), ("behind", "generation"),
    ]
    # 8 frames at 20 fps play for 400 ms, longer than the 267 ms their input takes.
    assert status(50.0, 102.0) == ("ok", None)
    assert status(50.0, 102.1) == ("behind", "playback")
    assert status(120.0, 150.0) == ("behind", "generation")
    replay = _health_latency([(50.0, 6.25, None), (60.0, 7.5, None)], None)
    assert replay["response_ms"] is None and replay["status"] is None
    assert replay["reason"] is None and replay["playback_ms"] is None
    assert replay["chunk_ms"] == 55.0


def test_latency_budget_uses_the_acquisition_rate() -> None:
    session = SimpleNamespace(actions_per_step=8, config=SimpleNamespace(fps=10))
    manager = HeadsetServer(session=session, frames=LatestHeadsetFrames(fps=30))
    assert abs(manager.latency_budget_ms() - 266.67) < 0.01
    assert manager.latency_playback_ms() == 8 / 30 * 1e3  # display_fps defaults to 30
    manager.frames = LatestHeadsetFrames(fps=None)  # browser rate: the model rate
    assert manager.latency_budget_ms() == 800.0
    manager.session = None  # mid mode switch
    assert manager.latency_budget_ms() is None
    assert manager.latency_playback_ms() is None


def test_healthz_latency_fills_and_resets_on_mode_switch(monkeypatch, tmp_path) -> None:
    import time

    def step(_window):
        # The first action row reached the server 50 ms ago, on the server's clock.
        first_ns = time.perf_counter_ns() - 50_000_000
        return SimpleNamespace(
            metadata={
                "action_rows": 8,
                "action_first_capture_ns": first_ns,
                "action_capture_ns": first_ns + 30_000_000,
            },
            metrics={"input_s": 0.0},
        )

    manager = _recording_server(monkeypatch, tmp_path, step)
    manager.session.actions_per_step = 8

    def idle_factory(config):  # the reloaded session generates nothing
        session = _idle_session()
        session.config, session.actions_per_step = config, 8
        return session

    manager.session_factory = idle_factory

    async def check() -> None:
        async with TestClient(TestServer(create_app(manager))) as client:
            await _until(lambda: manager.runtime_ready)
            health = await (await client.get("/healthz")).json()
            assert health["latency"] is None  # no chunk yet: nobody is connected
            async with client.ws_connect("/ws"):
                await _until(lambda: len(manager.latency) >= 2)
                latency = (await (await client.get("/healthz")).json())["latency"]
                assert latency["budget_ms"] == 266.7  # 8 rows at the 30 Hz acquisition rate
                assert latency["playback_ms"] == 266.7  # 8 frames at the default 30 fps
                assert (latency["status"], latency["reason"]) == ("ok", None)
                assert 0 <= latency["chunk_ms"] < 266.7
                assert abs(latency["frame_ms"] - latency["chunk_ms"] / 2) < 0.2  # 2 frames
                assert 50 <= latency["response_ms"] < 5000
                assert (await client.post("/mode", json={"mode": "balanced"})).status == 202
                await _until(lambda: not manager.mode_switching)
                health = await (await client.get("/healthz")).json()
                assert health["mode"] == "balanced"
                assert health["latency"] is None
            await manager.close()

    asyncio.run(check())


def test_playback_catches_up_only_while_the_next_block_waits(monkeypatch) -> None:
    import time

    import numpy as np

    from cosmos3_h_dreams import headset_server as server_module

    monkeypatch.setattr(server_module, "encode_rgb_frame_to_jpeg", lambda frame: bytes([frame[0, 0, 0]]))
    manager = HeadsetServer(
        session=_idle_session(), frames=LatestHeadsetFrames(fps=30), display_fps=20
    )
    manager.track = track = object()
    frames = tuple(np.full((2, 2, 3), i, np.uint8) for i in range(8))
    result = SimpleNamespace(metadata={"action_rows": 0})
    real_sleep = asyncio.sleep

    async def play(waiting_since):
        shown, sleeps = [], []

        async def sleep(delay):
            sleeps.append(delay)
            shown.append(manager._jpeg[0])
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", sleep)
        try:
            manager._ready.set()
            manager._next_block_waiting = (
                None if waiting_since is None else time.perf_counter() - waiting_since
            )
            await manager._publish_mjpeg(track, result, frames, 0.0, 0.0)
        finally:
            monkeypatch.setattr(asyncio, "sleep", real_sleep)
        return shown, sleeps

    async def check() -> None:
        # Nothing waiting: every frame, at the display interval.
        shown, sleeps = await play(None)
        assert shown == list(range(8)) and min(sleeps) > 0.04
        # The next block just became ready: every frame, at 3/4 of the interval.
        shown, sleeps = await play(0.0)
        assert shown == list(range(8)) and max(sleeps) <= 0.0375
        # It has waited more than half a block (200 ms): every other frame, never the first.
        shown, _ = await play(1.0)
        assert shown == [0, 2, 4, 6]

    asyncio.run(check())
