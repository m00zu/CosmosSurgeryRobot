# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved text K/V: file format, checkpoint identity and rollout equivalence."""

import copy
import dataclasses
import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cosmos3_h_dreams import session as session_module
from cosmos3_h_dreams._runtime import checkpoint_load, text_kv_cache
from cosmos3_h_dreams._runtime._hand_kv_session import HandKVSession
from cosmos3_h_dreams._runtime._network import (
    CosmosMoTNetworkConfig,
    _off_generation_path,
)
from cosmos3_h_dreams._runtime._spec import Cosmos3HDreamsSpec, LLMSpec
from cosmos3_h_dreams._runtime.text_kv_cache import TextKV, TextKVMeta
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig
from cosmos3_h_dreams.session import Cosmos3HDreamsSession
from PIL import Image

pytestmark = pytest.mark.ci_cpu

_META = TextKVMeta(
    checkpoint_id="abc",
    checkpoint_stat="m.safetensors:10:20",
    prompt="A prompt.",
    height=480,
    width=832,
    tokenizer="Qwen/Qwen3-VL-8B-Instruct",
    dtype="torch.bfloat16",
)


def _kv(layers=2, n=3):
    torch.manual_seed(0)
    return TextKV(
        text_ids=torch.arange(n),
        k=[torch.randn(1, 2, n, 4) for _ in range(layers)],
        v=[torch.randn(1, 2, n, 4) for _ in range(layers)],
    )


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "text.pt"
    kv = _kv()
    text_kv_cache.save_text_kv(path, kv, _META)

    loaded = text_kv_cache.load_text_kv(path, _META)

    assert loaded is not None
    assert torch.equal(loaded.text_ids, kv.text_ids)
    for got, want in zip(loaded.k + loaded.v, kv.k + kv.v, strict=True):
        assert torch.equal(got, want)
    assert not list(tmp_path.glob("*.tmp"))


def test_two_concurrent_saves_leave_a_valid_file_and_no_temp_file(tmp_path, monkeypatch):
    import threading

    path = tmp_path / "text.pt"
    both_writing = threading.Barrier(2, timeout=10)
    real_save = torch.save
    written = []

    def save_together(obj, target):  # both writers are inside torch.save at once
        written.append(str(target))
        both_writing.wait()
        real_save(obj, target)

    monkeypatch.setattr(text_kv_cache.torch, "save", save_together)
    errors = []

    def writer():
        try:
            text_kv_cache.save_text_kv(path, _kv(), _META)
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(written)) == 2  # each writer has its own temp file
    assert text_kv_cache.load_text_kv(path, _META) is not None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["text.pt"]


def test_a_failed_save_removes_its_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "text.pt"

    def half_write(_obj, target):
        open(target, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(text_kv_cache.torch, "save", half_write)
    with pytest.raises(OSError, match="disk full"):
        text_kv_cache.save_text_kv(path, _kv(), _META)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("checkpoint_id", "other"),
        ("prompt", "Another prompt."),
        ("height", 544),
        ("width", 960),
        ("tokenizer", "other/tokenizer"),
        ("dtype", "torch.float32"),
    ],
)
def test_load_rejects_any_changed_field(tmp_path, capsys, field, value):
    path = tmp_path / "text.pt"
    text_kv_cache.save_text_kv(path, _kv(), _META)

    loaded = text_kv_cache.load_text_kv(path, dataclasses.replace(_META, **{field: value}))

    assert loaded is None
    assert field in capsys.readouterr().out


def test_load_accepts_a_new_checkpoint_stat_and_stores_it(tmp_path):
    """Only whether to re-hash depends on the stat; a copy with the same bytes is a hit."""
    path = tmp_path / "text.pt"
    text_kv_cache.save_text_kv(path, _kv(), _META)
    copied = dataclasses.replace(_META, checkpoint_stat="m.safetensors:10:21")

    assert text_kv_cache.load_text_kv(path, copied) is not None
    assert text_kv_cache._saved_meta(path)["checkpoint_stat"] == "m.safetensors:10:21"
    assert text_kv_cache.load_text_kv(path, _META) is not None  # still valid either way


def test_a_failed_stat_refresh_still_loads(tmp_path, monkeypatch, capsys):
    path = tmp_path / "text.pt"
    text_kv_cache.save_text_kv(path, _kv(), _META)

    def fail(*_args):
        raise OSError("read-only file system")

    monkeypatch.setattr(text_kv_cache, "save_text_kv", fail)
    copied = dataclasses.replace(_META, checkpoint_stat="m.safetensors:10:21")

    assert text_kv_cache.load_text_kv(path, copied) is not None
    assert "read-only" in capsys.readouterr().out


def test_load_returns_none_for_missing_or_corrupt_file(tmp_path):
    assert text_kv_cache.load_text_kv(tmp_path / "missing.pt", _META) is None
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch file")
    assert text_kv_cache.load_text_kv(corrupt, _META) is None


def test_load_rejects_other_format_version(tmp_path, monkeypatch):
    path = tmp_path / "text.pt"
    monkeypatch.setattr(text_kv_cache, "TEXT_KV_FORMAT_VERSION", 0)
    text_kv_cache.save_text_kv(path, _kv(), _META)
    monkeypatch.undo()

    assert text_kv_cache.load_text_kv(path, _META) is None


from safetensors.torch import save_file  # noqa: E402

_UND = "language_model.model.layers.0.input_layernorm.weight"
_GEN = "language_model.model.layers.0.input_layernorm_moe_gen.weight"
_EMBED = "language_model.model.embed_tokens.weight"
_HEAD = "language_model.lm_head.weight"


def _checkpoint(path, **overrides):
    tensors = {
        _UND: torch.ones(4),
        _GEN: torch.ones(4),
        _EMBED: torch.ones(5, 4),
        _HEAD: torch.ones(5, 4),
    }
    tensors.update(overrides)
    save_file(tensors, str(path))
    return path


def test_checkpoint_id_is_the_sha256_of_the_whole_file(tmp_path):
    import hashlib

    path = _checkpoint(tmp_path / "a.safetensors")
    changed = _checkpoint(tmp_path / "b.safetensors", **{_GEN: torch.zeros(4)})

    assert text_kv_cache.checkpoint_id(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert text_kv_cache.checkpoint_id(changed) != text_kv_cache.checkpoint_id(path)


def test_checkpoint_id_hashes_the_shards_an_index_names(tmp_path):
    import hashlib
    import json

    save_file({_UND: torch.ones(4)}, str(tmp_path / "s1.safetensors"))
    save_file({_EMBED: torch.ones(5, 4)}, str(tmp_path / "s2.safetensors"))
    named = {"weight_map": {_UND: "s1.safetensors", _EMBED: "s2.safetensors"}}
    (tmp_path / "other.safetensors.index.json").write_text(json.dumps(named))
    # A default-named index beside it describes a different checkpoint.
    save_file({_UND: torch.zeros(4)}, str(tmp_path / "z1.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {_UND: "z1.safetensors"}})
    )
    shards = (tmp_path / "s1.safetensors").read_bytes() + (tmp_path / "s2.safetensors").read_bytes()

    named_id = text_kv_cache.checkpoint_id(tmp_path / "other.safetensors.index.json")

    assert named_id == hashlib.sha256(shards).hexdigest()
    assert named_id != text_kv_cache.checkpoint_id(tmp_path)


def test_checkpoint_id_rejects_remote_sources():
    with pytest.raises(ValueError, match="local"):
        text_kv_cache.checkpoint_id("s3://bucket/model.safetensors")


_GRID = (2, 3)
_CHUNK = 2
_TCF = 4
_TEXT_IDS = torch.tensor([1, 2, 3, 4])


def _tiny_network():
    spec = Cosmos3HDreamsSpec(
        llm=LLMSpec(
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            intermediate_size=96,
            vocab_size=50,
            mrope_section=(4, 2, 2),
        )
    )
    torch.manual_seed(0)
    net = CosmosMoTNetworkConfig(
        spec=spec, dtype=torch.float32, include_vision_tower=False, action_gen=True
    ).setup()
    with torch.no_grad():
        net.action_modality_embed.normal_()
    return net.eval()


def _rollout(net, *, cache_mode, text_kv=None):
    torch.manual_seed(1)
    init_latent = torch.randn(_GRID[0] * _GRID[1], net.config.spec.vfm.patch_latent_dim)
    actions = torch.randn(2, _CHUNK * _TCF, 64)
    session = HandKVSession(
        net,
        text_ids=_TEXT_IDS,
        text_kv=text_kv,
        init_latent=init_latent,
        grid_hw=_GRID,
        chunk_size=_CHUNK,
        tcf=_TCF,
        t_list=[1.0, 5.0 / 6.0],
        cache_mode=cache_mode,
        window_frames=4,
        fademem_slots=3,
    )
    with torch.no_grad():
        frames = [torch.cat(session.step(actions[0]))]
        session.reset_rollout()
        frames.append(torch.cat(session.step(actions[1])))
    return frames


@pytest.mark.parametrize("cache_mode", ["block", "cat", "fademem"])
def test_saved_text_kv_reproduces_prefilled_rollout(cache_mode):
    reference = _tiny_network()
    expected = _rollout(reference, cache_mode=cache_mode)

    cached = copy.deepcopy(reference)
    kv = text_kv_cache.compute_text_kv(cached, _TEXT_IDS)
    text_kv_cache.drop_understanding(cached)
    actual = _rollout(cached, cache_mode=cache_mode, text_kv=kv)

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_drop_understanding_leaves_only_generation_weights():
    net = _tiny_network()
    text_kv_cache.drop_understanding(net)

    for name, parameter in net.named_parameters():
        expected = "meta" if _off_generation_path(name) else "cpu"
        assert parameter.device.type == expected, name


def test_compute_text_kv_shapes():
    net = _tiny_network()
    kv = text_kv_cache.compute_text_kv(net, _TEXT_IDS)

    assert torch.equal(kv.text_ids, _TEXT_IDS)
    assert len(kv.k) == len(kv.v) == 2
    assert kv.k[0].shape == (1, 2, 4, 16)
    assert kv.k[0].device.type == "cpu"


def test_load_generation_state_dict_skips_understanding_keys(tmp_path):
    path = _checkpoint(tmp_path / "m.safetensors")
    raw = checkpoint_load._load_generation_state_dict(str(path), "cpu")

    assert set(raw) == {_GEN}


@pytest.mark.parametrize("offload", [False, True])
def test_skip_understanding_loads_only_generation_weights(monkeypatch, capsys, offload):
    source = _tiny_network()

    def read_generation(_source, device):
        return {
            k: v.to(device)
            for k, v in source.state_dict().items()
            if not _off_generation_path(k)
        }

    def read_everything(*_args):
        raise AssertionError("skip_understanding must not read the full checkpoint")

    monkeypatch.setattr(checkpoint_load, "_load_generation_state_dict", read_generation)
    monkeypatch.setattr(checkpoint_load, "_load_raw_state_dict", read_everything)
    monkeypatch.setattr(checkpoint_load, "_remap_state_dict", dict)
    monkeypatch.setattr(
        checkpoint_load,
        "CosmosMoTNetworkConfig",
        functools.partial(CosmosMoTNetworkConfig, spec=source.config.spec),
    )

    net = checkpoint_load.load_exported_checkpoint(
        "/checkpoint",
        device="cpu",
        dtype=torch.float32,
        offload_understanding=offload,
        skip_understanding=True,
    )

    out = capsys.readouterr().out
    assert "offloaded to host memory" not in out  # the understanding half was never loaded
    assert "understanding weights skipped" in out
    devices = {name: p.device.type for name, p in net.named_parameters()}
    assert {d for n, d in devices.items() if _off_generation_path(n)} == {"meta"}
    assert {d for n, d in devices.items() if not _off_generation_path(n)} == {"cpu"}
    kv = text_kv_cache.compute_text_kv(source, _TEXT_IDS)
    frames = _rollout(net, cache_mode="fademem", text_kv=kv)
    assert all(torch.isfinite(f).all() for f in frames)


def test_skip_understanding_rejects_remote_checkpoints():
    with pytest.raises(ValueError, match="local"):
        checkpoint_load.load_exported_checkpoint(
            "s3://bucket/model.safetensors", device="cpu", skip_understanding=True
        )


class _FakeRollout:
    accepts_partial_chunks = True
    last_timing: dict[str, float] = {}
    _compile_start_step = 0

    def prewarm_rollout(self, steps, *, profile=False):
        return []

    def wait_for_pending_cache_update(self, *, block=False):
        return 0.0


class _FakeModel:
    device = "cpu"
    native_gqa_attention = False
    action_domain_id = 0

    def __init__(self):
        self.net = SimpleNamespace(config=SimpleNamespace(num_embodiment_domains=40))
        self.text_kv = None
        self.sessions = []

    def new_session(self, scene):
        self.sessions.append(self.text_kv)
        return _FakeRollout()

    def release_session(self, _scene, _rollout):
        pass

    def decode_tensor(self, _latent):
        return torch.zeros((1, 3, 2, 2))


def _init_session(
    tmp_path, monkeypatch, *, text_cache, loaded, outcome="built", hash_s=None, rehashed=False
):
    frame = tmp_path / "frame.png"
    actions = tmp_path / "actions.npy"
    Image.new("RGB", (2, 2)).save(frame)
    np.save(actions, np.zeros((16, 28), dtype=np.float32))
    scene = SimpleNamespace(fps=10.0, replay_actions=torch.zeros((16, 64)), text_ids=_TEXT_IDS)
    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: scene)
    calls = {}

    def resolve_meta(**kwargs):
        calls["meta"] = kwargs
        return text_kv_cache.MetaLookup(meta=_META, hash_s=hash_s, rehashed=rehashed)

    def ensure(net, text_ids, *, cached, meta, path, native_gqa_attention):
        calls["ensure"] = (cached, path)
        return "kv", "hit" if cached is not None else outcome

    monkeypatch.setattr(text_kv_cache, "resolve_meta", resolve_meta)
    monkeypatch.setattr(text_kv_cache, "load_text_kv", lambda path, meta: loaded)
    monkeypatch.setattr(text_kv_cache, "ensure_text_kv", ensure)
    factory_kwargs = {}
    model = _FakeModel()

    def factory(**kwargs):
        factory_kwargs.update(kwargs)
        return model

    session = Cosmos3HDreamsSession(
        config=Cosmos3HDreamsConfig(
            checkpoint="/checkpoint",
            initial_frame=frame,
            actions=actions,
            prompt="A prompt.",
            device="cpu",
            text_cache=text_cache,
        ),
        model_factory=factory,
    )
    session.init()
    return session, model, factory_kwargs, calls


def test_session_without_text_cache_loads_everything(tmp_path, monkeypatch):
    _, model, kwargs, calls = _init_session(tmp_path, monkeypatch, text_cache=None, loaded=None)

    assert kwargs["skip_understanding"] is False
    assert calls == {}
    assert model.sessions == [None]


@pytest.mark.parametrize("loaded", [None, "saved"])
def test_session_with_text_cache_skips_weights_only_when_file_is_valid(tmp_path, monkeypatch, loaded):
    path = tmp_path / "text.pt"
    _, model, kwargs, calls = _init_session(tmp_path, monkeypatch, text_cache=path, loaded=loaded)

    assert kwargs["skip_understanding"] is (loaded is not None)
    assert calls["meta"] == {
        "checkpoint": "/checkpoint", "prompt": "A prompt.", "height": 368, "width": 640, "cache": path
    }
    assert calls["ensure"] == (loaded, path)
    assert model.sessions == ["kv"]


def test_session_reports_text_cache_off(tmp_path, monkeypatch):
    session, *_ = _init_session(tmp_path, monkeypatch, text_cache=None, loaded=None)

    assert session.text_cache["status"] == "off"
    assert session.text_cache["path"] is None and session.text_cache["hash_s"] is None
    assert session.text_cache["rehashed"] is False
    assert session.text_cache["load_s"] >= 0.0


def test_session_has_no_text_cache_status_before_init(tmp_path):
    frame = tmp_path / "f.png"
    Image.new("RGB", (2, 2)).save(frame)
    config = Cosmos3HDreamsConfig(checkpoint="/c", initial_frame=frame, prompt="P")
    assert Cosmos3HDreamsSession(config=config).text_cache is None


@pytest.mark.parametrize(
    ("loaded", "outcome", "hash_s", "rehashed", "status"),
    [
        ("saved", "hit", None, False, "hit"),
        ("saved", "hit", 16.8, True, "hit"),
        (None, "built", 16.8, False, "built"),
        (None, "built-unsaved", 16.8, False, "built-unsaved"),
        (None, "built", 16.8, True, "built"),
    ],
)
def test_session_records_the_text_cache_outcome(
    tmp_path, monkeypatch, loaded, outcome, hash_s, rehashed, status
):
    path = tmp_path / "text.pt"
    session, *_ = _init_session(
        tmp_path, monkeypatch, text_cache=path, loaded=loaded,
        outcome=outcome, hash_s=hash_s, rehashed=rehashed,
    )

    assert session.text_cache["status"] == status
    assert session.text_cache["rehashed"] is rehashed
    assert session.text_cache["path"] == str(path)
    assert session.text_cache["hash_s"] == hash_s
    assert session.text_cache["load_s"] >= 0.0


def test_each_init_reports_its_own_outcome(tmp_path, monkeypatch):
    """A mode switch builds a new session; after the first build it is a hit."""
    path = tmp_path / "text.pt"
    first, *_ = _init_session(tmp_path, monkeypatch, text_cache=path, loaded=None, hash_s=16.8)
    second, *_ = _init_session(tmp_path, monkeypatch, text_cache=path, loaded="saved")

    assert (first.text_cache["status"], second.text_cache["status"]) == ("built", "hit")
    assert second.text_cache["hash_s"] is None


def test_switch_scene_rejects_a_prompt_the_saved_text_kv_does_not_cover(tmp_path, monkeypatch):
    session, model, _, _ = _init_session(
        tmp_path, monkeypatch, text_cache=tmp_path / "text.pt", loaded="saved"
    )
    model.text_kv = TextKV(text_ids=_TEXT_IDS, k=[], v=[])
    actions = torch.zeros((16, 64))
    other = SimpleNamespace(fps=10.0, replay_actions=actions, text_ids=torch.tensor([9, 9, 9, 9]))
    same = SimpleNamespace(fps=10.0, replay_actions=actions, text_ids=_TEXT_IDS.clone())
    config = dataclasses.replace(session.config, prompt="B")

    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: other)
    with pytest.raises(ValueError, match="text cache"):
        session.switch_scene(config)
    assert len(model.sessions) == 1

    monkeypatch.setattr(session_module, "prepare_scene", lambda *_args: same)
    session.switch_scene(config)
    assert len(model.sessions) == 2


def test_expected_meta_identifies_the_checkpoint_prompt_and_resolution(tmp_path):
    path = _checkpoint(tmp_path / "m.safetensors")

    meta = text_kv_cache.expected_meta(
        checkpoint=str(path), prompt="A prompt.", height=368, width=640, cache=tmp_path / "t.pt"
    )

    stat = path.stat()
    assert meta == TextKVMeta(
        checkpoint_id=text_kv_cache.checkpoint_id(path),
        checkpoint_stat=f"m.safetensors:{stat.st_size}:{stat.st_mtime_ns}",
        prompt="A prompt.",
        height=368,
        width=640,
        tokenizer="Qwen/Qwen3-VL-8B-Instruct",
        dtype="torch.bfloat16",
    )


def test_expected_meta_hashes_only_when_the_checkpoint_file_changes(tmp_path, monkeypatch):
    import os

    path = _checkpoint(tmp_path / "m.safetensors")
    cache = tmp_path / "t.pt"
    meta = text_kv_cache.expected_meta(
        checkpoint=str(path), prompt="A prompt.", height=368, width=640, cache=cache
    )
    text_kv_cache.save_text_kv(cache, _kv(), meta)
    hashed = []
    monkeypatch.setattr(text_kv_cache, "checkpoint_id", lambda source: hashed.append(source) or "new")

    again = text_kv_cache.expected_meta(
        checkpoint=str(path), prompt="A prompt.", height=368, width=640, cache=cache
    )
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    touched = text_kv_cache.expected_meta(
        checkpoint=str(path), prompt="A prompt.", height=368, width=640, cache=cache
    )

    assert again == meta and hashed == [str(path)]
    assert touched.checkpoint_id == "new"


def test_resolve_meta_reports_hash_time_and_whether_a_saved_stat_went_stale(
    tmp_path, monkeypatch
):
    import os

    path = _checkpoint(tmp_path / "m.safetensors")
    cache = tmp_path / "t.pt"
    args = dict(checkpoint=str(path), prompt="A prompt.", height=368, width=640, cache=cache)

    missing = text_kv_cache.resolve_meta(**args)
    text_kv_cache.save_text_kv(cache, _kv(), missing.meta)
    unchanged = text_kv_cache.resolve_meta(**args)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    touched = text_kv_cache.resolve_meta(**args)

    assert (missing.rehashed, missing.hash_s is not None) == (False, True)
    assert (unchanged.rehashed, unchanged.hash_s) == (False, None)
    assert unchanged.meta == missing.meta
    assert (touched.rehashed, touched.hash_s is not None) == (True, True)
    assert text_kv_cache.expected_meta(**args) == touched.meta


def test_a_touched_checkpoint_is_rehashed_once_and_still_hits(tmp_path, monkeypatch):
    """A copy, re-download or touch: one re-hash, no rebuild, then no re-hash."""
    import os

    path = _checkpoint(tmp_path / "m.safetensors")
    cache = tmp_path / "t.pt"
    args = dict(checkpoint=str(path), prompt="A prompt.", height=368, width=640, cache=cache)
    text_kv_cache.save_text_kv(cache, _kv(), text_kv_cache.resolve_meta(**args).meta)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    touched = text_kv_cache.resolve_meta(**args)
    assert touched.rehashed and touched.hash_s is not None
    assert text_kv_cache.load_text_kv(cache, touched.meta) is not None

    hashed = []
    monkeypatch.setattr(text_kv_cache, "checkpoint_id", lambda source: hashed.append(source) or "x")
    after = text_kv_cache.resolve_meta(**args)
    assert hashed == [] and after.hash_s is None and not after.rehashed
    assert after.meta == touched.meta


def test_init_publishes_the_text_cache_status_only_after_warmup(tmp_path, monkeypatch):
    seen = []
    original = Cosmos3HDreamsSession._warmup_rollout

    def warmup(self, rollout):
        seen.append(self.text_cache)
        original(self, rollout)

    monkeypatch.setattr(Cosmos3HDreamsSession, "_warmup_rollout", warmup)
    session, *_ = _init_session(tmp_path, monkeypatch, text_cache=tmp_path / "t.pt", loaded="saved")

    assert seen == [None]
    assert session.text_cache["status"] == "hit"


def test_save_text_kv_creates_the_parent_directory(tmp_path):
    path = tmp_path / "data_local" / "text_kv" / "cache.pt"

    text_kv_cache.save_text_kv(path, _kv(), _META)

    assert text_kv_cache.load_text_kv(path, _META) is not None


def test_ensure_text_kv_reports_a_hit(tmp_path):
    cached = _kv(n=3)
    kv, outcome = text_kv_cache.ensure_text_kv(
        None, torch.arange(3), cached=cached, meta=_META,
        path=tmp_path / "t.pt", native_gqa_attention=False,
    )
    assert (kv, outcome) == (cached, "hit")


def test_ensure_text_kv_builds_saves_and_drops(tmp_path):
    net = _tiny_network()
    path = tmp_path / "text.pt"

    kv, outcome = text_kv_cache.ensure_text_kv(
        net, _TEXT_IDS, cached=None, meta=_META, path=path, native_gqa_attention=False
    )

    assert outcome == "built"
    assert text_kv_cache.load_text_kv(path, _META) is not None
    assert net.layers[0].und.input_layernorm.weight.device.type == "meta"
    assert torch.equal(kv.text_ids, _TEXT_IDS)


def test_ensure_text_kv_warns_and_still_drops_when_the_save_fails(
    tmp_path, monkeypatch, capsys
):
    def fail(*_args):
        raise RuntimeError("open file failed: Permission denied")

    monkeypatch.setattr(text_kv_cache, "save_text_kv", fail)
    net = _tiny_network()
    path = tmp_path / "text.pt"

    kv, outcome = text_kv_cache.ensure_text_kv(
        net, _TEXT_IDS, cached=None, meta=_META, path=path, native_gqa_attention=False
    )

    assert outcome == "built-unsaved"
    assert torch.equal(kv.text_ids, _TEXT_IDS)
    assert "could not save" in capsys.readouterr().out
    assert net.layers[0].und.input_layernorm.weight.device.type == "meta"


def test_save_text_kv_leaves_no_temporary_file_when_torch_save_fails(
    tmp_path, monkeypatch
):
    def fail(_payload, file):
        file.write_bytes(b"partial")
        raise RuntimeError("iostream error")

    monkeypatch.setattr(text_kv_cache.torch, "save", fail)
    path = tmp_path / "text.pt"

    with pytest.raises(RuntimeError, match="iostream"):
        text_kv_cache.save_text_kv(path, _kv(), _META)

    assert list(tmp_path.iterdir()) == []


def test_ensure_text_kv_rejects_cached_ids_for_other_tokens(tmp_path):
    cached = _kv(n=3)
    with pytest.raises(ValueError, match="text cache"):
        text_kv_cache.ensure_text_kv(
            None, torch.tensor([9, 9, 9]), cached=cached, meta=_META,
            path=tmp_path / "t.pt", native_gqa_attention=False,
        )
