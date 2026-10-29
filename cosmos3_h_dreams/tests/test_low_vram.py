# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-VRAM generation: staged placement and the FP8 generation path."""

import copy

import pytest
import torch
from cosmos3_h_dreams._runtime._hand_kv_session import HandKVSession
from cosmos3_h_dreams._runtime._network import CosmosMoTNetworkConfig
from cosmos3_h_dreams._runtime._spec import Cosmos3HDreamsSpec, LLMSpec

_GRID = (2, 3)
_CHUNK = 2
_TCF = 4


def _tiny_network(num_layers=2, dtype=torch.float32):
    spec = Cosmos3HDreamsSpec(
        llm=LLMSpec(
            hidden_size=64,
            num_layers=num_layers,
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
        spec=spec, dtype=dtype, include_vision_tower=False, action_gen=True
    ).setup()
    with torch.no_grad():
        net.action_modality_embed.normal_()
    return net.eval()


def _rollout(net, *, cache_mode, text_ids, chunks=2, on_step=None, **session_options):
    torch.manual_seed(1)
    init_latent = torch.randn(_GRID[0] * _GRID[1], net.config.spec.vfm.patch_latent_dim)
    actions = torch.randn(chunks, _CHUNK * _TCF, 64)
    session = HandKVSession(
        net,
        text_ids=text_ids,
        init_latent=init_latent,
        grid_hw=_GRID,
        chunk_size=_CHUNK,
        tcf=_TCF,
        t_list=[1.0, 5.0 / 6.0],
        cache_mode=cache_mode,
        window_frames=4,
        fademem_slots=3,
        **session_options,
    )
    with torch.no_grad():
        frames = []
        for chunk in [*actions[:-1], None, actions[-1]]:
            if chunk is None:
                session.reset_rollout()
                continue
            frames.append(torch.cat(session.step(chunk)))
            if on_step is not None:
                on_step(session)
    return frames


def _understanding_devices(net):
    return {
        name: parameter.device.type
        for name, parameter in net.named_parameters()
        if ".und." in name or name.split(".")[0] in {"embed_tokens", "norm", "lm_head"}
    }


@pytest.mark.ci_cpu
@pytest.mark.parametrize("cache_mode", ["block", "fademem"])
def test_generation_rollout_never_reads_understanding_weights(cache_mode):
    reference = _tiny_network()
    staged = copy.deepcopy(reference)
    # Meta tensors hold no values: any read of an understanding weight would fail.
    staged.place_for_generation("cpu", understanding_device="meta")

    expected = _rollout(reference, cache_mode=cache_mode, text_ids=None)
    actual = _rollout(staged, cache_mode=cache_mode, text_ids=None)

    assert set(_understanding_devices(staged).values()) == {"meta"}
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.ci_cpu
def test_fixed_shape_fademem_refresh_matches_eager_refresh(monkeypatch):
    net = _tiny_network()
    text_ids = torch.tensor([1, 2, 3])
    eager = _rollout(net, cache_mode="fademem", text_ids=text_ids)
    # The compiled path's masked fixed-shape layout is what is under test, so
    # trace it eagerly; the compiled and captured kernels are covered on CUDA.
    compiled = []
    monkeypatch.setattr(
        torch, "compile", lambda fn, **_: compiled.append(fn.__name__) or fn
    )
    fixed = _rollout(net, cache_mode="fademem", text_ids=text_ids, compile_denoise=True)
    assert "refresh" in compiled
    for got, want in zip(fixed, eager, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.ci_cpu
@pytest.mark.parametrize("offload, raw_device", [(False, "meta"), (True, "cpu")])
def test_checkpoint_load_stages_understanding_through_host_memory(
    monkeypatch, offload, raw_device
):
    import functools

    from cosmos3_h_dreams._runtime import checkpoint_load

    source = _tiny_network()
    reads = []

    def read_raw(_source, device, _credentials):
        reads.append(device)
        return {k: v.to(device) for k, v in source.state_dict().items()}

    monkeypatch.setattr(checkpoint_load, "_load_raw_state_dict", read_raw)
    monkeypatch.setattr(checkpoint_load, "_remap_state_dict", dict)
    monkeypatch.setattr(
        checkpoint_load,
        "CosmosMoTNetworkConfig",
        functools.partial(CosmosMoTNetworkConfig, spec=source.config.spec),
    )

    # Meta stands in for the GPU: placing a weight there is observable on CPU.
    net = checkpoint_load.load_exported_checkpoint(
        "/checkpoint", device="meta", dtype=torch.float32, offload_understanding=offload
    )

    assert reads == [raw_device]
    assert net.device.type == "meta"
    expected = {"cpu"} if offload else {"meta"}
    assert set(_understanding_devices(net).values()) == expected


@pytest.mark.ci_cpu
@pytest.mark.parametrize("offload", [False, True])
def test_hand_model_forwards_offload_to_checkpoint_load(monkeypatch, offload):
    from types import SimpleNamespace

    from cosmos3_h_dreams._runtime import _hand_model, checkpoint_load, fetch_checkpoint

    loads = []

    class _Net:
        config = SimpleNamespace(action_gen=True)

        def eval(self):
            return self

    def load(_source, **kwargs):
        loads.append(
            (
                kwargs["offload_understanding"],
                kwargs["fp8_generation"],
                kwargs["fp8_scope"],
                kwargs["fp8_gemm"],
                kwargs["skip_understanding"],
            )
        )
        return _Net()

    monkeypatch.setattr(checkpoint_load, "load_exported_checkpoint", load)
    monkeypatch.setattr(
        fetch_checkpoint, "ensure_checkpoint", lambda source, **_: source
    )
    monkeypatch.setattr(_hand_model, "make_wan22_encode_fn", lambda **_: None)
    monkeypatch.setattr(_hand_model, "make_wan22_chunk_decoder", lambda **_: None)

    model = _hand_model.HandModel(
        checkpoint="/checkpoint",
        device="cpu",
        offload_understanding=offload,
        fp8_generation=not offload,
        fp8_scope="mlp" if offload else "all",
        fp8_gemm="epilogue_scales",
        skip_understanding=offload,
    )

    assert loads == [
        (offload, not offload, "mlp" if offload else "all", "epilogue_scales", offload)
    ]
    assert model.text_kv is None


@pytest.mark.ci_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA memory stats")
@pytest.mark.parametrize("cache_mode", ["block", "fademem"])
def test_text_prefill_holds_one_borrowed_layer_at_a_time(cache_mode):
    import functools

    from cosmos3_h_dreams._runtime._rope import text_mrope_ids

    net = _tiny_network(num_layers=8)
    net.place_for_generation("cuda")
    layer_bytes = sum(
        p.numel() * p.element_size() for p in net.layers[0].und.parameters()
    )
    text_ids = torch.arange(40)
    positions = text_mrope_ids(40, 0.0)[0]
    if cache_mode == "block":
        new_cache = functools.partial(
            net.init_stream_cache, chunk_grid=(_CHUNK, *_GRID), window_frames=4
        )
        prefill = net.prefill_text_stream
    else:  # FadeMem seeds from the growing cache.
        new_cache = net.new_kv_cache
        prefill = net.prefill_text_cache
    prefill(text_ids, positions, new_cache())  # cuBLAS workspace, kernel caches
    cache = new_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    # Sessions are built outside no_grad in production; prefill must not let
    # autograd keep every borrowed layer's device copy alive.
    prefill(text_ids, positions, cache)

    held = torch.cuda.max_memory_allocated() - before
    # One borrowed layer plus small activations, never all eight layers.
    assert held < 4 * layer_bytes


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.ci_gpu
@pytest.mark.skipif(_accelerator() is None, reason="needs a CUDA or MPS device")
@pytest.mark.parametrize("cache_mode", ["block", "fademem"])
def test_text_prefill_borrows_understanding_layers_and_matches_resident_model(
    cache_mode,
):
    device = _accelerator()
    reference = _tiny_network().to(device)
    staged = _tiny_network()
    staged.place_for_generation(device)
    text_ids = torch.tensor([3, 14, 15, 9, 26])

    expected = _rollout(reference, cache_mode=cache_mode, text_ids=text_ids)
    actual = _rollout(staged, cache_mode=cache_mode, text_ids=text_ids)

    assert staged.device.type == device
    assert set(_understanding_devices(staged).values()) == {"cpu"}
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


_GEN_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 matmul needs CUDA"
)


@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    "scope, expected",
    [
        ("all", set(_GEN_PROJECTIONS)),
        ("mlp", {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}),
    ],
)
def test_fp8_scope_selects_generation_projections(monkeypatch, scope, expected):
    from cosmos3_h_dreams._runtime import _network

    class _Marked(torch.nn.Module):
        def __init__(self, linear, device, gemm):
            super().__init__()
            self.device = device
            self.gemm = gemm

    monkeypatch.setattr(_network, "_Fp8Linear", _Marked)
    net = _tiny_network(dtype=torch.bfloat16)
    net.quantize_generation_fp8("cpu", scope, "epilogue_scales")

    for layer in net.layers:
        swapped = {
            name for name in _GEN_PROJECTIONS
            if isinstance(layer.gen.get_submodule(name), _Marked)
        }
        assert swapped == expected
        assert {layer.gen.get_submodule(n).gemm for n in swapped} == {"epilogue_scales"}
        for name in set(_GEN_PROJECTIONS) - expected:
            assert layer.gen.get_submodule(name).weight.dtype == torch.bfloat16
        for name in _GEN_PROJECTIONS:
            assert not isinstance(layer.und.get_submodule(name), _Marked)
    with pytest.raises(ValueError, match="fp8 scope"):
        net.quantize_generation_fp8("cpu", "attn")
    with pytest.raises(ValueError, match="fp8 gemm"):
        net.quantize_generation_fp8("cpu", "all", "rowwise")


@pytest.mark.ci_cpu
def test_fp8_epilogue_scales_match_dequantized_matmul():
    from cosmos3_h_dreams._runtime._network import _Fp8Linear

    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 48, bias=False)
    with torch.no_grad():
        # Per-channel weight scales spanning 1000x: a transposed scale shows up.
        linear.weight.mul_(torch.logspace(-2, 1, 48)[:, None])
    layer = _Fp8Linear(linear.to(torch.bfloat16), "cpu", "epilogue_scales")
    # Per-token activation scales spanning 1000x, with leading dims to reshape.
    x = (torch.randn(2, 5, 64) * torch.logspace(-1, 2, 10).view(2, 5, 1)).to(torch.bfloat16)

    from flashdreams.accelerated.quantization.quantizer import Granularity, quantize

    xq, x_scale = quantize(x, torch.float8_e4m3fn, Granularity.SLICE, axis=-1)
    reference = (xq.float() * x_scale) @ (layer.weight.float() * layer.weight_scale).T
    output = layer(x)
    assert output.shape == (2, 5, 48) and output.dtype == torch.bfloat16
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=1e-3)


@pytest.mark.ci_gpu
@_needs_cuda
def test_fp8_generation_quantizes_only_generation_projections():
    net = _tiny_network(dtype=torch.bfloat16)
    net.quantize_generation_fp8("cuda")

    for layer in net.layers:
        for name in _GEN_PROJECTIONS:
            gen = layer.gen.get_submodule(name)
            und = layer.und.get_submodule(name)
            assert gen.weight.dtype == torch.float8_e4m3fn
            assert gen.weight.device.type == "cuda"
            assert und.weight.dtype == torch.bfloat16
            assert und.weight.device.type == "cpu"


@pytest.mark.ci_gpu
@_needs_cuda
@pytest.mark.parametrize("cache_mode", ["block", "fademem"])
def test_fp8_generation_rollout_tracks_bf16(cache_mode):
    reference = _tiny_network(dtype=torch.bfloat16)
    reference.place_for_generation("cuda")
    fp8 = _tiny_network(dtype=torch.bfloat16)
    fp8.quantize_generation_fp8("cuda")
    fp8.place_for_generation("cuda")
    text_ids = torch.tensor([3, 14, 15, 9, 26])

    expected = _rollout(reference, cache_mode=cache_mode, text_ids=text_ids)
    actual = _rollout(fp8, cache_mode=cache_mode, text_ids=text_ids)

    for got, want in zip(actual, expected, strict=True):
        error = (got.float() - want.float()).norm() / want.float().norm()
        assert error < 0.15


@pytest.mark.ci_gpu
@_needs_cuda
@pytest.mark.parametrize("offload", [False, True])
def test_checkpoint_load_quantizes_generation_from_host_memory(monkeypatch, offload):
    import functools

    from cosmos3_h_dreams._runtime import checkpoint_load

    source = _tiny_network(dtype=torch.bfloat16)
    reads = []

    def read_raw(_source, device, _credentials):
        reads.append(device)
        return {k: v.to(device) for k, v in source.state_dict().items()}

    monkeypatch.setattr(checkpoint_load, "_load_raw_state_dict", read_raw)
    monkeypatch.setattr(checkpoint_load, "_remap_state_dict", dict)
    monkeypatch.setattr(
        checkpoint_load,
        "CosmosMoTNetworkConfig",
        functools.partial(CosmosMoTNetworkConfig, spec=source.config.spec),
    )

    net = checkpoint_load.load_exported_checkpoint(
        "/checkpoint",
        device="cuda",
        offload_understanding=offload,
        fp8_generation=True,
    )

    # Reading to host first keeps the bf16 generation weights off the GPU.
    assert reads == ["cpu"]
    assert net.layers[0].gen.mlp.up_proj.weight.dtype == torch.float8_e4m3fn
    expected = {"cpu"} if offload else {"cuda"}
    assert set(_understanding_devices(net).values()) == expected


@pytest.mark.ci_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA graphs")
def test_compiled_fademem_refresh_replays_like_eager_refresh():
    # bf16, the production dtype, routes the refresh through varlen attention.
    net = _tiny_network(dtype=torch.bfloat16).to("cuda")
    text_ids = torch.tensor([1, 2, 3], device="cuda")

    def memory_after_each_step(**options):
        # Frames of a random tiny model barely depend on the cache, so compare
        # the committed K/V that the refresh writes.
        memories = []
        _rollout(
            net,
            cache_mode="fademem",
            text_ids=text_ids,
            chunks=6,
            on_step=lambda s: memories.append(
                torch.cat([t.flatten() for kv in s.fademem.memory.layers() for t in kv])
            ),
            **options,
        )
        return memories

    eager = memory_after_each_step()
    # Two warmup calls, then capture and replays through fill and merging.
    graphed = memory_after_each_step(compile_denoise=True, cuda_graph_denoise=True)
    for got, want in zip(graphed, eager, strict=True):
        error = (got.float() - want.float()).norm() / want.float().norm()
        assert error < 0.02


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded_on = None

    def record(self, stream) -> None:
        self.recorded_on = stream

    def query(self) -> bool:
        # Still running on the side stream.
        return False


class _FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited: list[_FakeEvent] = []
        self.synchronized = False

    def record_event(self, event: _FakeEvent) -> _FakeEvent:
        event.record(self)
        return event

    def wait_event(self, event: _FakeEvent) -> None:
        self.waited.append(event)

    def synchronize(self) -> None:
        self.synchronized = True


class _FakeTensor:
    def detach(self):
        return self

    def record_stream(self, _stream) -> None:
        pass


@pytest.mark.ci_cpu
def test_rejected_overlapped_commit_still_orders_the_main_stream(monkeypatch):
    import contextlib

    main, side = _FakeStream("main"), _FakeStream("side")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device=None: main)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda _s: contextlib.nullcontext())
    session = HandKVSession.__new__(HandKVSession)
    session.device = torch.device("cuda")
    session._cache_stream = side
    session._pending_cache = []

    def refresh_then_reject(*_args, selection_stream):
        # The refresh graph has been queued on the side stream by now; the
        # FadeMem commit then rejects non-finite descriptors.
        assert selection_stream is main
        raise ValueError("FadeMem descriptors must be finite [N, D] on the cache device")

    session._refresh_chunk = refresh_then_reject

    with pytest.raises(ValueError, match="finite"):
        session._submit_refresh(0, _FakeTensor(), _FakeTensor(), 2)

    assert len(session._pending_cache) == 1
    done = session._pending_cache[0][0]
    assert done.recorded_on is side
    # reset() and switch_scene() wait like this before rewriting the memory.
    session.wait_for_pending_cache_update(block=True)
    assert main.waited == [done]
    assert main.synchronized
    assert session._pending_cache == []


@pytest.mark.ci_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
def test_rejected_fademem_commit_leaves_a_completion_event():
    net = _tiny_network(dtype=torch.bfloat16).to("cuda")
    torch.manual_seed(1)
    session = HandKVSession(
        net,
        text_ids=torch.tensor([1, 2, 3], device="cuda"),
        init_latent=torch.randn(
            _GRID[0] * _GRID[1], net.config.spec.vfm.patch_latent_dim
        ),
        grid_hw=_GRID,
        chunk_size=_CHUNK,
        tcf=_TCF,
        t_list=[1.0, 5.0 / 6.0],
        cache_mode="fademem",
        window_frames=4,
        fademem_slots=3,
        fademem_merge_selection="cosine",
        overlap_cache_update=True,
    )
    assert session._cache_stream is not None
    action = torch.randn(_CHUNK * _TCF, 64)
    with torch.no_grad():
        session.step(action)  # Warm up, so the rejected step below is quick.
    session.wait_for_pending_cache_update(block=True)

    submit, commit = session._submit_refresh, session.fademem.commit

    def poisoned_submit(start, clean_vision, *args):
        # Non-finite descriptors, made on the main stream like a NaN denoise.
        return submit(start, torch.full_like(clean_vision, float("nan")), *args)

    def busy_commit(*args, **kwargs):
        # The refresh queued on the side stream is still running when the
        # commit rejects the chunk on the main stream.
        torch.cuda._sleep(int(2e9))
        return commit(*args, **kwargs)

    session._submit_refresh = poisoned_submit
    session.fademem.commit = busy_commit
    with torch.no_grad(), pytest.raises(ValueError, match="finite"):
        session.step(action)

    assert len(session._pending_cache) == 1
    done = session._pending_cache[0][0]
    session.reset_rollout()
    assert done.query()
    session._submit_refresh, session.fademem.commit = submit, commit
    with torch.no_grad():
        frames = session.step(action)
    assert all(bool(torch.isfinite(frame).all()) for frame in frames)


@pytest.mark.ci_gpu
@_needs_cuda
def test_overlapped_rollouts_reuse_the_given_cache_stream():
    net = _tiny_network(dtype=torch.bfloat16)
    net.place_for_generation("cuda")
    stream = torch.cuda.Stream()
    sessions = [
        HandKVSession(
            net,
            text_ids=torch.arange(5),
            init_latent=torch.randn(_GRID[0] * _GRID[1], net.config.spec.vfm.patch_latent_dim),
            grid_hw=_GRID,
            chunk_size=_CHUNK,
            tcf=_TCF,
            t_list=[1.0, 5.0 / 6.0],
            cache_mode="block",
            window_frames=4,
            overlap_cache_update=True,
            cache_stream=stream,
        )
        for _ in range(2)
    ]
    assert all(session._cache_stream is stream for session in sessions)



@pytest.mark.ci_cpu
@pytest.mark.parametrize(
    "setting, compiled, device_type, dtype, varlen, expected",
    [
        (True, True, "cuda", torch.bfloat16, True, True),
        (True, True, "cuda", torch.float16, True, True),
        (False, True, "cuda", torch.bfloat16, True, False),
        (True, False, "cuda", torch.bfloat16, True, False),
        (True, True, "cpu", torch.bfloat16, True, False),
        (True, True, "cuda", torch.float32, True, False),
        (True, True, "cuda", torch.bfloat16, False, False),
    ],
)
def test_steady_refresh_needs_the_setting_and_a_varlen_refresh(
    monkeypatch, setting, compiled, device_type, dtype, varlen, expected
):
    from cosmos3_h_dreams._runtime import _hand_kv_session

    monkeypatch.setattr(_hand_kv_session, "varlen_attention_available", lambda: varlen)
    assert (
        _hand_kv_session.uses_steady_refresh(
            steady_refresh_sdpa=setting,
            refresh_compiled=compiled,
            device_type=device_type,
            dtype=dtype,
        )
        is expected
    )


@pytest.mark.ci_cpu
@pytest.mark.parametrize("steady_refresh_sdpa", [True, False])
def test_steady_refresh_sdpa_off_never_builds_the_steady_refresh(
    monkeypatch, steady_refresh_sdpa
):
    from cosmos3_h_dreams._runtime import _hand_kv_session

    # Stand in for a CUDA rollout with varlen FlashAttention, so the setting
    # alone decides; the refresh builds are recorded instead of compiled.
    real = _hand_kv_session.uses_steady_refresh
    monkeypatch.setattr(
        _hand_kv_session,
        "uses_steady_refresh",
        lambda **kwargs: real(**{**kwargs, "device_type": "cuda"}),
    )
    monkeypatch.setattr(_hand_kv_session, "varlen_attention_available", lambda: True)
    built = []

    def build(self, *, steady=False):
        built.append(steady)
        return lambda *_args: pytest.fail("the refresh must not run here")

    monkeypatch.setattr(HandKVSession, "_build_refresh_fn", build)
    net = _tiny_network(dtype=torch.bfloat16)
    session = HandKVSession(
        net,
        text_ids=None,
        init_latent=torch.randn(_GRID[0] * _GRID[1], net.config.spec.vfm.patch_latent_dim),
        grid_hw=_GRID,
        chunk_size=_CHUNK,
        tcf=_TCF,
        t_list=[1.0, 5.0 / 6.0],
        cache_mode="fademem",
        window_frames=4,
        fademem_slots=3,
        compile_denoise=True,
        steady_refresh_sdpa=steady_refresh_sdpa,
    )

    assert session.steady_refresh_sdpa is steady_refresh_sdpa
    if steady_refresh_sdpa:
        assert built == [False, True]
        assert session._refresh_steady_fn is not None
    else:
        # Only the varlen refresh: nothing extra to compile, capture or hold,
        # so step() never selects a steady refresh and warmup never waits on one.
        assert built == [False]
        assert session._refresh_steady_fn is None
        assert session._refresh_steady_graph is None
