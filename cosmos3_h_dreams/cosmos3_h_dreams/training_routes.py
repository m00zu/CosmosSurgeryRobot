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

"""Training ground pages, recordings API, model-mode and input-rate control."""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from .courses import SLUG, courses_json, list_demos, load_recorded_demo, save_demo, validate_review
from .recording import SESSION_ID, list_sessions, load_session, save_review

if TYPE_CHECKING:
    from .headset_server import HeadsetServer

_WEB_DIR = Path(__file__).resolve().parent / "web"
MODEL_MODES = {
    "standard": {"fp8_generation": False, "fp8_scope": "all"},
    "balanced": {"fp8_generation": True, "fp8_scope": "mlp"},
    "low_memory": {"fp8_generation": True, "fp8_scope": "all"},
}
"""``POST /mode`` names and the runtime settings each one loads."""
_CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript"}


def _static(name: str):
    async def handler(_: web.Request) -> web.Response:
        return web.Response(
            text=(_WEB_DIR / name).read_text(),
            content_type=_CONTENT_TYPES[Path(name).suffix],
            headers={"Cache-Control": "no-store"},
        )

    return handler


def _json(data: object, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, headers={"Cache-Control": "no-store"})


def _error(status: int, message: str) -> web.Response:
    return _json({"error": message}, status)


def add_training_routes(app: web.Application, manager: HeadsetServer) -> None:
    """Register the training page, dashboard, recordings and GPU APIs, and mode control."""
    app.router.add_get("/train", _static("train.html"))
    app.router.add_get("/train.js", _static("train.js"))
    app.router.add_get("/train_core.js", _static("train_core.js"))
    app.router.add_get("/train_courses.js", _static("train_courses.js"))
    app.router.add_get("/xr_view.js", _static("xr_view.js"))
    app.router.add_get("/train_vr.js", _static("train_vr.js"))
    app.router.add_get("/dashboard", _static("dashboard.html"))
    app.router.add_get("/dashboard.js", _static("dashboard.js"))
    app.router.add_get("/gpu", _static("gpu.html"))
    app.router.add_get("/gpu.js", _static("gpu.js"))
    app.router.add_get("/gpu_chart.js", _static("gpu_chart.js"))

    def _root() -> Path:
        if manager.recording_dir is None:
            raise web.HTTPNotFound(reason="Recording is off")
        return Path(manager.recording_dir)

    async def sessions(_: web.Request) -> web.Response:
        return web.json_response(
            await asyncio.to_thread(list_sessions, _root()),
            headers={"Cache-Control": "no-store"},
        )

    async def session(request: web.Request) -> web.Response:
        data = await asyncio.to_thread(load_session, _root(), request.match_info["id"])
        if data is None:
            raise web.HTTPNotFound(reason="Unknown session")
        return web.json_response(data, headers={"Cache-Control": "no-store"})

    async def video(request: web.Request) -> web.FileResponse:
        session_id = request.match_info["id"]
        path = _root() / session_id / "video.mp4"
        if not SESSION_ID.match(session_id) or not path.is_file():
            raise web.HTTPNotFound(reason="No video for this session")
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    def _exercises():
        if manager.exercises is None:
            raise web.HTTPNotFound(reason="Courses are off")
        return manager.exercises

    async def courses(_: web.Request) -> web.Response:
        exercises = _exercises()
        demos = await asyncio.to_thread(list_demos, manager.demos_dir)
        return _json(
            {
                "courses": courses_json(exercises),
                "demos": demos,
                "scene": manager.course_scene,
            }
        )

    async def exercise_demo(request: web.Request) -> web.Response:
        exercise = _exercises().get(request.match_info["id"])
        if exercise is None or exercise.demo is None:
            raise web.HTTPNotFound(reason="No demo for this exercise")
        return _json(exercise.demo.to_json())

    async def recorded_demo(request: web.Request) -> web.Response:
        _exercises()
        name = request.match_info["name"]
        if not SLUG.match(name) or manager.demos_dir is None:
            raise web.HTTPNotFound(reason="Unknown demo")
        path = Path(manager.demos_dir) / f"{name}.jsonl"
        if not path.is_file():
            raise web.HTTPNotFound(reason="Unknown demo")
        try:
            demo = await asyncio.to_thread(load_recorded_demo, path)
        except ValueError as error:
            return _error(500, str(error))
        return _json(demo.to_json())

    async def save_as_demo(request: web.Request) -> web.Response:
        _exercises()
        session_id = request.match_info["id"]
        folder = _root() / session_id
        if not SESSION_ID.match(session_id) or not folder.is_dir():
            raise web.HTTPNotFound(reason="Unknown session")
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "Expected {name}")
        name = payload.get("name") if isinstance(payload, dict) else None
        try:
            saved = await asyncio.to_thread(
                save_demo, Path(manager.demos_dir), name, folder, session_id
            )
        except FileExistsError:
            return _error(409, "A demo with that name already exists")
        except ValueError as error:
            return _error(400, str(error))
        except OSError as error:
            return _error(500, f"Could not save the demo: {error}")
        return _json(saved, 201)

    async def review(request: web.Request) -> web.Response:
        exercises = _exercises()
        session_id = request.match_info["id"]
        data = await asyncio.to_thread(load_session, _root(), session_id)
        if data is None:
            raise web.HTTPNotFound(reason="Unknown session")
        tag = data["exercise"] if isinstance(data["exercise"], dict) else {}
        exercise = exercises.get(tag.get("id"))
        if exercise is None:
            return _error(400, "Not an attempt at a known exercise")
        try:
            payload = await request.json()
        except ValueError:
            return _error(400, "Expected {checked, notes}")
        try:
            saved = validate_review(exercise, payload)
        except ValueError as error:
            return _error(400, str(error))
        try:
            await asyncio.to_thread(save_review, _root(), session_id, saved)
        except OSError as error:
            return _error(500, f"Could not save the review: {error}")
        return _json(saved)

    async def mode(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            requested = payload["mode"] if isinstance(payload, dict) else None
        except (ValueError, KeyError) as error:
            raise web.HTTPBadRequest(reason="Expected a mode") from error
        if requested not in MODEL_MODES:
            raise web.HTTPBadRequest(reason="Unknown mode")
        if manager.session_factory is None:
            raise web.HTTPConflict(reason="Mode switching is not available")
        if manager.mode_switching:
            raise web.HTTPConflict(reason="A mode switch is already running")
        if requested == manager.mode:
            return web.json_response({"mode": requested})
        if not manager.runtime_ready or manager._closed:
            raise web.HTTPConflict(reason="Model is not ready")
        manager.mode_switching = True  # visible to /healthz before the task first runs
        manager._switch_task = asyncio.create_task(manager.switch_mode(requested))
        return web.json_response({"mode_switching": True}, status=202)

    async def input_rate(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            fps = payload["fps"] if isinstance(payload, dict) else None
        except (ValueError, KeyError) as error:
            raise web.HTTPBadRequest(reason="Expected {fps}") from error
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps):
            raise web.HTTPBadRequest(reason="Expected {fps}")
        try:
            manager.set_input_rate(float(fps))
        except ValueError as error:
            raise web.HTTPBadRequest(reason=str(error)) from error
        except RuntimeError as error:
            raise web.HTTPConflict(reason=str(error)) from error
        return web.json_response({"fps": manager.input_fps})

    async def gpu(request: web.Request) -> web.Response:
        # In-memory reads under a short lock: fine on the event loop.
        monitor = manager.gpu_monitor
        if monitor is None:
            return web.json_response(
                {"error": "GPU monitor disabled"},
                status=404,
                headers={"Cache-Control": "no-store"},
            )
        since = None
        if "since" in request.query:
            try:
                since = float(request.query["since"])
            except ValueError as error:
                raise web.HTTPBadRequest(reason="since must be a number") from error
            if not math.isfinite(since):
                raise web.HTTPBadRequest(reason="since must be finite")
        only = None
        if "gpu" in request.query:
            try:
                only = int(request.query["gpu"])
            except ValueError as error:
                raise web.HTTPBadRequest(reason="gpu must be an integer") from error
        return web.json_response(
            {
                "now": time.time(),
                "device": monitor.device,
                "nvml": monitor.nvml,
                "gpus": monitor.gpus,
                "samples": [s.to_json() for s in monitor.samples(since=since, gpu=only)],
                "marks": monitor.marks(since=since),
                "baseline": {str(g): v for g, v in monitor.baseline().items()},
                "peak": {str(g): v for g, v in monitor.peak().items()},
                "host": monitor.host(),
                "csv": monitor.csv_name,
                "csv_error": monitor.csv_error,
            },
            headers={"Cache-Control": "no-store"},
        )

    app.router.add_post("/mode", mode)
    app.router.add_post("/input-rate", input_rate)
    app.router.add_get("/api/gpu", gpu)
    app.router.add_get("/api/sessions", sessions)
    app.router.add_get("/api/sessions/{id}", session)
    app.router.add_post("/api/sessions/{id}/demo", save_as_demo)
    app.router.add_post("/api/sessions/{id}/review", review)
    app.router.add_get("/api/courses", courses)
    app.router.add_get("/api/courses/{id}/demo", exercise_demo)
    app.router.add_get("/api/demos/{name}", recorded_demo)
    app.router.add_get("/recordings/{id}/video.mp4", video)
