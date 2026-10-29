# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for session YAML, preset precedence, and scene transitions."""

import asyncio
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml
from cosmos3_h_dreams.app import Cosmos3HDreamsApplication
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig
from cosmos3_h_dreams.headset_server import LatestHeadsetFrames, HeadsetServer
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from cosmos3_h_dreams.session_config import load_session_config

from cosmos3_h_dreams import headset_server
from cosmos3_h_dreams import session as session_module
from cosmos3_h_dreams import session_config as session_config_module

pytestmark = pytest.mark.ci_cpu


def _session_file(tmp_path):
    (tmp_path / "a.png").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs/session.yaml"
    data = {
        "runtime": {"preset": "base", "checkpoint": "./model.pt", "device": "cpu"},
        "headset": {"preset": "headset-base", "cert": "cert.pem", "key": "key.pem"},
        "scenes": [
            {
                "name": "a",
                "label": "Scene A",
                "initial_frame": "a.png",
                "prompt": "First scene.",
            },
            {"name": "b", "initial_frame": "b.png"},
        ],
    }
    path.write_text(yaml.safe_dump(data))
    return path, data


def test_yaml_paths_and_scene_defaults(tmp_path, monkeypatch):
    path, _ = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(path.parent)
    application = Cosmos3HDreamsApplication()
    application.init(["--session-config", str(path), "--headset", "--fps", "12"])
    assert list(application.scenes) == ["a", "b"]
    assert application.scene_labels == {"a": "Scene A", "b": "b"}
    assert application.scenes["a"].initial_frame == tmp_path / "a.png"
    assert application.scenes["b"].initial_frame == tmp_path / "b.png"
    assert application.scenes["a"].checkpoint == str(tmp_path / "model.pt")
    assert application.scenes["a"].prompt == "First scene."
    assert application.scenes["b"].prompt != "First scene."
    assert all(config.fps == 12 for config in application.scenes.values())


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda data: data["runtime"].update(chonk_size=2), "unknown runtime"),
        (lambda data: data["headset"].update(preset=23), "preset must"),
        (lambda data: data["headset"].update(display_fps=True), "integer"),
        (lambda data: data["headset"].update(video_transport="mjpeg"), "unknown headset"),
        (lambda data: data["scenes"].append(dict(data["scenes"][0])), "unique"),
        (lambda data: data["scenes"][0].update(height=288), "unknown scene"),
        (lambda data: data["scenes"][0].update(domain_id=True), "Invalid type"),
        (lambda data: data.update(scenes=[]), "non-empty list"),
    ],
)
def test_yaml_rejects_invalid_schema(tmp_path, change, match):
    path, data = _session_file(tmp_path)
    change(data)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=match):
        load_session_config(path)


@pytest.mark.parametrize(
    "runtime, mode",
    [
        ({}, "standard"),
        ({"fp8_generation": True}, "low_memory"),
        ({"fp8_generation": True, "fp8_scope": "all"}, "low_memory"),
        ({"fp8_generation": True, "fp8_scope": "mlp"}, "balanced"),
        ({"fp8_generation": False, "fp8_scope": "mlp"}, "standard"),
    ],
)
def test_yaml_fp8_settings_select_the_start_up_mode(tmp_path, monkeypatch, runtime, mode):
    path, data = _session_file(tmp_path)
    data["runtime"].update(runtime)
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    application = Cosmos3HDreamsApplication()
    application.init(["--session-config", str(path), "--headset"])
    scope = runtime.get("fp8_scope", "all")
    assert all(config.fp8_scope == scope for config in application.scenes.values())
    assert {headset_server._mode_of(c) for c in application.scenes.values()} == {mode}


def test_yaml_rejects_invalid_fp8_scope(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    data["runtime"].update(fp8_generation=True, fp8_scope=True)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Invalid type for fp8_scope"):
        load_session_config(path)


@pytest.mark.parametrize(
    "runtime, flags, expected",
    [
        ({}, [], True),
        ({"steady_refresh_sdpa": False}, [], False),
        ({"steady_refresh_sdpa": True}, [], True),
        ({"steady_refresh_sdpa": False}, ["--steady-refresh-sdpa"], True),
        ({"steady_refresh_sdpa": True}, ["--no-steady-refresh-sdpa"], False),
    ],
)
def test_yaml_steady_refresh_sdpa_reaches_every_scene(
    tmp_path, monkeypatch, runtime, flags, expected
):
    path, data = _session_file(tmp_path)
    data["runtime"].update(runtime)
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    application = Cosmos3HDreamsApplication()
    application.init(["--session-config", str(path), "--headset", *flags])
    assert {c.steady_refresh_sdpa for c in application.scenes.values()} == {expected}
    # Not a mode-defining setting: every value keeps the start-up mode.
    assert {headset_server._mode_of(c) for c in application.scenes.values()} == {"standard"}


@pytest.mark.parametrize("value", ["false", 0, None])
def test_yaml_rejects_non_boolean_steady_refresh_sdpa(tmp_path, value):
    path, data = _session_file(tmp_path)
    data["runtime"].update(steady_refresh_sdpa=value)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="Invalid type for steady_refresh_sdpa"):
        load_session_config(path)


def test_both_yaml_presets_and_cli_overrides(tmp_path, monkeypatch):
    path, _ = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "headset_server",
            "--session-config",
            str(path),
            "--config",
            "wan-taehv-chunk2-2steps-fademem-compile",
            "--headset-config",
            "headset-base",
            "--display-fps",
            "17",
            "--fps",
            "12",
        ],
    )
    certificates = []
    monkeypatch.setattr(
        headset_server.ssl,
        "SSLContext",
        lambda _: SimpleNamespace(
            load_cert_chain=lambda *args: certificates.append(args)
        ),
    )
    monkeypatch.setattr(headset_server, "create_app", lambda manager: manager)
    monkeypatch.setattr(headset_server, "_server_gpu", lambda _device: (0, 0))
    monkeypatch.setattr(
        headset_server, "GpuMonitor", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    managers = []
    monkeypatch.setattr(
        headset_server.web, "run_app", lambda manager, **_: managers.append(manager)
    )
    headset_server.main()
    (manager,) = managers
    assert certificates == [(str(tmp_path / "cert.pem"), str(tmp_path / "key.pem"))]
    assert manager.display_fps == 17
    assert manager.frames._period_ns == round(1e9 / 30)
    assert all(
        config.fps == 12 and config.chunk_size == 2
        for config in manager.scenes.values()
    )
    assert manager.session.config.decoder == "taehv"
    assert list(manager.scenes) == ["a", "b"]


def test_switch_keeps_model_changes_domain_and_recalibrates(tmp_path, monkeypatch):
    frame = tmp_path / "frame.png"
    frame.touch()
    config = Cosmos3HDreamsConfig(
        checkpoint="model.pt", initial_frame=frame, prompt="A", device="cpu"
    )
    calls = []
    rollout = SimpleNamespace(
        wait_for_pending_cache_update=lambda **_: calls.append("wait")
    )
    model = SimpleNamespace(
        net=SimpleNamespace(config=SimpleNamespace(num_embodiment_domains=40)),
        action_domain_id=config.domain_id,
        new_session=lambda scene: calls.append(scene) or object(),
    )
    session = Cosmos3HDreamsSession(config=config)
    session._model, session._rollout, session._scene = model, rollout, object()
    session._headset_source = SimpleNamespace(
        set_normalizer=lambda path: calls.append(("normalizer", path))
    )
    monkeypatch.setattr(
        session_module, "prepare_scene", lambda config, _: config.prompt
    )
    session._rows_consumed = 20
    new_config = replace(config, prompt="B", domain_id=25)
    session.switch_scene(new_config)
    assert session._model is model and model.action_domain_id == 25
    assert session.config is new_config and session._scene == "B"
    assert session._rows_consumed == 0
    assert calls == ["wait", "B", ("normalizer", None)]
    with pytest.raises(ValueError, match="shared runtime"):
        session.switch_scene(replace(new_config, height=288))
    with pytest.raises(ValueError, match="domain_id"):
        session.switch_scene(replace(new_config, domain_id=40))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_switch_frees_the_old_rollout_despite_reference_cycles(tmp_path, monkeypatch, device):
    import gc
    import weakref

    frame = tmp_path / "frame.png"
    frame.touch()
    config = Cosmos3HDreamsConfig(
        checkpoint="model.pt", initial_frame=frame, prompt="A", device=device
    )

    class _Rollout:
        def __init__(self):
            self.step_fn = self.wait_for_pending_cache_update  # a cycle, as the real one

        def wait_for_pending_cache_update(self, **_):
            return 0.0

    emptied = []
    monkeypatch.setattr(session_module.torch.cuda, "empty_cache", lambda: emptied.append(1))
    model = SimpleNamespace(
        net=SimpleNamespace(config=SimpleNamespace(num_embodiment_domains=40)),
        action_domain_id=config.domain_id,
        new_session=lambda _scene: _Rollout(),
    )
    session = Cosmos3HDreamsSession(config=config)
    session._model, session._rollout, session._scene = model, _Rollout(), object()
    monkeypatch.setattr(session_module, "prepare_scene", lambda config, _: config.prompt)
    old = weakref.ref(session._rollout)
    gc.disable()  # only switch_scene's own collection may free it
    try:
        session.switch_scene(replace(config, prompt="B"))
        assert old() is None
    finally:
        gc.enable()
    assert isinstance(session._rollout, _Rollout)
    assert emptied == ([1] if device == "cuda" else [])


def test_scene_request_wakes_idle_generation_and_handles_failure():
    async def check():
        configs = {"a": object(), "b": object()}
        calls = []

        def switch(config):
            calls.append(config)
            if config is configs["a"]:
                raise ValueError("bad scene")

        manager = HeadsetServer(
            session=SimpleNamespace(switch_scene=switch),
            frames=LatestHeadsetFrames(fps=10),
            scenes=configs,
        )
        manager.runtime_ready = True
        manager._producer = asyncio.create_task(manager._generate())
        manager.source = SimpleNamespace(request_restart=lambda: calls.append("wake"))
        await asyncio.sleep(0)
        try:
            await asyncio.wait_for(manager.select_scene("b"), timeout=2)
            assert calls == ["wake", configs["b"]]
            assert manager.active_scene == "b" and manager._scene_request is None
            with pytest.raises(headset_server.web.HTTPInternalServerError):
                await asyncio.wait_for(manager.select_scene("a"), timeout=2)
            assert manager.active_scene == "b"
            assert manager._scene_request is None
            with pytest.raises(headset_server.web.HTTPBadRequest):
                await manager.select_scene({"name": "a"})
        finally:
            manager._producer.cancel()
            await asyncio.gather(manager._producer, return_exceptions=True)

    asyncio.run(check())


def test_switch_discards_inflight_output_and_flushes_video():
    async def check():
        entered = threading.Event()
        release = threading.Event()
        prepared = []
        manager = None

        def step(_window):
            entered.set()
            assert release.wait(timeout=2)
            return SimpleNamespace(metadata={"action_rows": 1})

        def switch(_config):
            manager._ready.clear()

        manager = HeadsetServer(
            session=SimpleNamespace(
                config=SimpleNamespace(fps=10),
                next_step_requirements=lambda: SimpleNamespace(input_frame_count=1),
                step=step,
                switch_scene=switch,
            ),
            frames=LatestHeadsetFrames(fps=10),
            scenes={"a": object(), "b": object()},
        )
        manager.source = SimpleNamespace(request_restart=release.set)
        manager.track = SimpleNamespace(prepare_result_frames=prepared.append)
        manager.socket = object()
        manager._jpeg = b"old video"
        manager.runtime_ready = True
        manager._ready.set()
        manager._producer = asyncio.create_task(manager._generate())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await asyncio.wait_for(manager.select_scene("b"), timeout=2)
            assert manager.active_scene == "b"
            assert prepared == []
            assert manager._jpeg is None
        finally:
            release.set()
            manager._producer.cancel()
            await asyncio.gather(manager._producer, return_exceptions=True)

    asyncio.run(check())


def test_yaml_resolves_all_assets_from_repo_root(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(path.parent)
    data["scenes"][0].update(normalizer="data/stats.npz", actions="data/actions.npy")
    data["scenes"][1]["initial_frame"] = str(tmp_path / "absolute.png")
    path.write_text(yaml.safe_dump(data))
    resolved = load_session_config(path)
    assert resolved["scenes"][0]["normalizer"] == tmp_path / "data/stats.npz"
    assert resolved["scenes"][0]["actions"] == tmp_path / "data/actions.npy"
    assert resolved["scenes"][1]["initial_frame"] == tmp_path / "absolute.png"
    assert resolved["runtime"]["checkpoint"] == str(tmp_path / "model.pt")
    assert resolved["headset"]["cert"] == str(tmp_path / "cert.pem")
    assert resolved["headset"]["key"] == str(tmp_path / "key.pem")


def test_initial_frame_wildcard_builds_named_scenes(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    monkeypatch.chdir(path.parent)
    (tmp_path / "other.jpg").touch()
    (tmp_path / "directory.png").mkdir()
    data["scenes"] = [
        {"initial_frame": "*.png", "prompt": "Shared prompt.", "domain_id": 25}
    ]
    path.write_text(yaml.safe_dump(data))
    application = Cosmos3HDreamsApplication()
    application.init(["--session-config", str(path), "--headset"])
    assert list(application.scenes) == ["a.png", "b.png"]
    assert application.scene_labels == {"a.png": "a.png", "b.png": "b.png"}
    assert application.scenes["a.png"].initial_frame == tmp_path / "a.png"
    assert application.scenes["b.png"].initial_frame == tmp_path / "b.png"
    assert all(
        config.prompt == "Shared prompt." and config.domain_id == 25
        for config in application.scenes.values()
    )


@pytest.mark.parametrize("pattern", ["missing/*.png", "configs/*.png"])
def test_initial_frame_wildcard_rejects_empty_matches(tmp_path, monkeypatch, pattern):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    data["scenes"] = [{"initial_frame": pattern}]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="matches no files"):
        load_session_config(path)


def test_initial_frame_wildcards_mix_with_explicit_scenes_and_reject_duplicates(
    tmp_path, monkeypatch
):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    data["scenes"] = [data["scenes"][0], {"initial_frame": str(tmp_path / "[ab].png")}]
    path.write_text(yaml.safe_dump(data))
    assert [scene["name"] for scene in load_session_config(path)["scenes"]] == [
        "a",
        "a.png",
        "b.png",
    ]
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/a.png").touch()
    data["scenes"] = [{"initial_frame": "**/*.png"}]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unique"):
        load_session_config(path)


def test_yaml_safety_block_builds_limits(tmp_path):
    from cosmos3_h_dreams.safety import SafetyLimits

    path, data = _session_file(tmp_path)
    assert load_session_config(path)["safety"] == SafetyLimits()
    data["safety"] = {"max_speed_m_s": 0.4}
    path.write_text(yaml.safe_dump(data))
    assert load_session_config(path)["safety"] == SafetyLimits(max_speed_m_s=0.4)
    data["safety"] = {"bogus": 1}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unknown safety"):
        load_session_config(path)


def test_training_ground_example_config_loads(monkeypatch):
    from pathlib import Path

    from cosmos3_h_dreams.safety import SafetyLimits

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo)
    data = load_session_config(repo / "cosmos3_h_dreams/configs/training_ground.yaml")
    assert data["safety"] == SafetyLimits.from_mapping(
        {
            "max_speed_m_s": 0.35,
            "max_acceleration_m_s2": 3.0,
            "max_rotation_rad_s": 4.0,
            "max_energy_s": 10.0,
            "max_energy_dose_s": 14.0,
            "energy_dose_window_s": 15.0,
            "max_energy_sweep_m": 0.10,
        }
    )
    assert data["recording"] == {"dir": repo / "recordings"}
    assert data["courses"] == {
        "dir": repo / "cosmos3_h_dreams/courses", "demos_dir": repo / "data_local/demos",
        "scene": None,
    }
    assert data["scenes"]


@pytest.mark.parametrize(
    "name, mode, steady",
    [("training_ground.yaml", "standard", True), ("laptop_rtx5090.yaml", "balanced", True)],
)
def test_shipped_training_ground_configs_start_in_their_mode(monkeypatch, name, mode, steady):
    from pathlib import Path

    from cosmos3_h_dreams.training_routes import MODEL_MODES

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo)
    path = repo / "cosmos3_h_dreams/configs" / name
    load_session_config(path)
    application = Cosmos3HDreamsApplication()
    application.init(["--session-config", str(path), "--headset"])
    configs = list(application.scenes.values())
    assert configs
    # HeadsetServer derives its start-up mode from the first scene's config.
    assert {headset_server._mode_of(c) for c in configs} == {mode}
    for key, value in MODEL_MODES[mode].items():
        assert {getattr(c, key) for c in configs} == {value}
    assert {c.steady_refresh_sdpa for c in configs} == {steady}


def test_laptop_config_matches_training_ground_except_mode(monkeypatch):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo)
    configs = repo / "cosmos3_h_dreams/configs"
    base = load_session_config(configs / "training_ground.yaml")
    laptop = load_session_config(configs / "laptop_rtx5090.yaml")
    assert laptop["runtime"] == {
        **base["runtime"],
        "fp8_generation": True,
        "fp8_scope": "mlp",
        "steady_refresh_sdpa": True,
    }
    assert {k: v for k, v in laptop.items() if k != "runtime"} == {
        k: v for k, v in base.items() if k != "runtime"
    }


def test_yaml_recording_dir_resolves_from_base(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    assert load_session_config(path)["recording"] == {"dir": tmp_path / "recordings"}
    data["recording"] = {"dir": "out/attempts"}
    path.write_text(yaml.safe_dump(data))
    assert load_session_config(path)["recording"] == {"dir": tmp_path / "out/attempts"}
    for bad, match in (
        ({"dir": "x", "fps": 3}, "unknown recording"),
        ({"dir": ""}, "recording.dir"),
        ({"dir": 3}, "recording.dir"),
        ({}, "recording.dir"),
        ("recordings", "recording must"),
    ):
        data["recording"] = bad
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match=match):
            load_session_config(path)


def test_yaml_gpu_monitor_block(tmp_path):
    path, data = _session_file(tmp_path)
    assert load_session_config(path)["gpu_monitor"] == {"enabled": True, "period_s": 1.0}
    data["gpu_monitor"] = {"enabled": False, "period_s": 2}
    path.write_text(yaml.safe_dump(data))
    assert load_session_config(path)["gpu_monitor"] == {"enabled": False, "period_s": 2.0}
    for bad, match in (
        ({"enabled": True, "rate": 3}, "unknown gpu_monitor"),
        ({"enabled": "yes"}, "gpu_monitor.enabled"),
        ({"period_s": 0}, "gpu_monitor.period_s"),
        ({"period_s": True}, "gpu_monitor.period_s"),
        ({"period_s": float("inf")}, "gpu_monitor.period_s"),
        ("on", "gpu_monitor must"),
    ):
        data["gpu_monitor"] = bad
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match=match):
            load_session_config(path)


def test_yaml_gpu_monitor_defaults_reach_main(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    data["gpu_monitor"] = {"enabled": True, "period_s": 2.5}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    created = []
    monkeypatch.setattr(
        headset_server, "GpuMonitor", lambda **kwargs: created.append(kwargs) or SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(headset_server, "_server_gpu", lambda device: (0, 0))
    monkeypatch.setattr(
        headset_server.ssl, "SSLContext",
        lambda _: SimpleNamespace(load_cert_chain=lambda *args: None),
    )
    monkeypatch.setattr(headset_server, "create_app", lambda manager: manager)
    monkeypatch.setattr(headset_server.web, "run_app", lambda manager, **_: None)
    monkeypatch.setattr(sys, "argv", ["headset_server", "--session-config", str(path)])
    headset_server.main()
    assert created[-1]["period_s"] == 2.5
    assert created[-1]["csv_path"].parent == tmp_path / "recordings"
    monkeypatch.setattr(
        sys, "argv", ["headset_server", "--session-config", str(path), "--gpu-period", "4"]
    )
    headset_server.main()
    assert created[-1]["period_s"] == 4.0
    data["gpu_monitor"] = {"enabled": False}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(sys, "argv", ["headset_server", "--session-config", str(path)])
    headset_server.main()
    assert len(created) == 2


def test_yaml_courses_block(tmp_path, monkeypatch):
    from pathlib import Path

    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    assert load_session_config(path)["courses"] is None
    data["courses"] = {"dir": "courses"}
    path.write_text(yaml.safe_dump(data))
    assert load_session_config(path)["courses"] == {
        "dir": tmp_path / "courses", "demos_dir": tmp_path / "data_local/demos",
        "scene": None,
    }
    data["courses"] = {"dir": "c", "demos_dir": "/abs/demos"}
    path.write_text(yaml.safe_dump(data))
    assert load_session_config(path)["courses"]["demos_dir"] == Path("/abs/demos")
    data["courses"] = {"dir": "c", "scene": "example_frame"}
    path.write_text(yaml.safe_dump(data))
    assert load_session_config(path)["courses"]["scene"] == "example_frame"
    for bad, match in (
        ({"dir": "c", "extra": 1}, "unknown courses"),
        ({}, "courses.dir"),
        ({"dir": "c", "demos_dir": ""}, "courses.demos_dir"),
        ({"dir": "c", "scene": ""}, "courses.scene"),
        ({"dir": "c", "scene": 3}, "courses.scene"),
        ("courses", "courses must"),
    ):
        data["courses"] = bad
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match=match):
            load_session_config(path)


def test_scene_start_jaws_closed(tmp_path, monkeypatch):
    path, data = _session_file(tmp_path)
    monkeypatch.setattr(session_config_module, "REPO_ROOT", tmp_path)
    data["scenes"][0]["start_jaws_closed"] = ["left"]
    path.write_text(yaml.safe_dump(data))
    first, second = load_session_config(path)["scenes"]
    assert first["start_jaws_closed"] == ["left"] and second["start_jaws_closed"] == []
    for bad in (["middle"], ["left", "left"], "left"):
        data["scenes"][0]["start_jaws_closed"] = bad
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match="start_jaws_closed must be a list"):
            load_session_config(path)
