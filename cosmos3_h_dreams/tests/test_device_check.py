# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hardware test kit's pure parts: arguments, statuses, aggregation, report, packing."""

import json
import sys
import tarfile

import numpy as np
import pytest
from cosmos3_h_dreams import device_check as dc
from cosmos3_h_dreams._device_check_gpu import categorize_trace, downsample, kernel_category

pytestmark = pytest.mark.ci_cpu


def test_arguments_defaults_and_validation():
    args = dc.parse_args(["--out", "x"])
    assert args.modes == ["standard", "balanced", "low_memory"]
    assert args.vram_cap_gb == 24.0 and args.power_limit is None and not args.quick
    assert args.attention_kernel == "auto"
    assert dc.parse_args(["--out", "x", "--attention-kernel", "default"]).attention_kernel == "default"
    assert dc.parse_args(["--out", "x", "--modes", "balanced, low_memory"]).modes == ["balanced", "low_memory"]
    for bad in (["--modes", "turbo"], ["--vram-cap-gb", "-1"], ["--attention-kernel", "flash"], []):
        with pytest.raises(SystemExit):
            dc.parse_args(([] if bad == [] else ["--out", "x"]) + bad)


def test_plan_quick_and_full():
    quick = [c["name"] for c in dc.plan_cases(dc.parse_args(["--out", "x", "--quick"]))]
    assert quick == [
        "performance.standard", "correctness.standard",
        "performance.balanced", "performance.balanced.sdpa_off", "correctness.balanced",
        "performance.low_memory", "correctness.low_memory",
    ]
    full = dc.plan_cases(dc.parse_args(["--out", "x"]))
    perf = [c for c in full if c["kind"] == "performance"]
    assert len(perf) == 6 and all(c["profile_chunks"] for c in perf)
    corr = [c for c in full if c["kind"] == "correctness"]
    assert [c["name"] for c in corr] == [
        "correctness.standard", "correctness.balanced", "correctness.balanced.sdpa_off", "correctness.low_memory"]
    assert all(c["deterministic"] for c in corr) and not any(c["deterministic"] for c in perf)
    skipped = dc.plan_cases(dc.parse_args(["--out", "x", "--skip-correctness", "--modes", "balanced"]))
    assert {c["kind"] for c in skipped} == {"performance"}


def test_realtime_status_thresholds():
    assert round(dc.BUDGET_MS, 1) == 266.7
    assert dc.realtime_status(82, 90) == "keeping up"
    assert dc.realtime_status(150, 199.9) == "keeping up"
    assert dc.realtime_status(150, 230) == "tight"
    assert dc.realtime_status(266, 300) == "tight"
    assert dc.realtime_status(270, 280) == "falling behind"
    assert dc.realtime_status(None, None) == "n/a"


def test_timing_summary():
    t = dc.timing_summary([float(v) for v in range(1, 11)])
    assert t["median"] == 6 and t["p90"] == 10 and t["min"] == 1 and t["n"] == 10


def test_correctness_status():
    assert dc.correctness_status([31.0, 25.0, 12.0], [True] * 3, [40.0] * 3) == ("PASS", [])
    status, problems = dc.correctness_status([18.0], [True], [40.0])
    assert status == "FAIL" and "first chunk" in problems[0]
    assert dc.correctness_status([30.0, 30.0], [True, False], [40.0, 40.0])[0] == "FAIL"
    assert "constant" in dc.correctness_status([30.0], [True], [0.5])[1][0]


def test_compare_correctness_psnr_and_missing_reference():
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, (3, 2, 8, 8, 3), dtype=np.uint8)
    raw = {"finite_per_chunk": [True] * 3, "min_frame_std_per_chunk": [50.0] * 3}
    same = dc.compare_correctness(ref.copy(), ref, raw)
    assert same["psnr_per_chunk"] == [99.0] * 3 and same["correctness"] == "PASS"
    noisy = dc.compare_correctness(np.zeros_like(ref), ref, raw)
    assert noisy["correctness"] == "FAIL"
    assert dc.compare_correctness(ref, None, raw)["correctness"] == "PASS (no reference)"


def test_summarize_ok_and_oom_cases():
    ok = dc.summarize_case({
        "status": "ok", "spec": {"mode": "balanced", "steady_refresh_sdpa": False, "kind": "performance"},
        "free_running_ms": [100.0] * 30, "mem_after_timing": {"smi_process_MiB": 11000, "allocated_GiB": 9.0},
    })
    assert ok["realtime"] == "keeping up" and ok["h100_ms"] == 87.0 and ok["smi_MiB"] == 11000
    assert ok["vs_h100"] == round(100 / 87, 2)
    oom = dc.summarize_case({"status": "oom", "phase": "load", "error": "CUDA out of memory",
                             "spec": {"mode": "standard", "steady_refresh_sdpa": True, "kind": "performance"}})
    assert oom["status"] == "oom" and "ms_per_chunk" not in oom


def _fake_results():
    perf = dc.summarize_case({
        "status": "ok", "spec": {"mode": "balanced", "steady_refresh_sdpa": True, "kind": "performance"},
        "free_running_ms": [90.0] * 30, "timing_s": {"model_load": 5.0, "scene_compile_warmup": 60.0},
        "attention": {"kernel": "sage", "reason": "sm_120, self-test passed"},
        "stage_ms": {"model_step_s": 50.0}, "mem_after_timing": {"smi_process_MiB": 10500, "allocated_GiB": 9.0},
        "vram_cap": {"effective": True, "allocator_cap_GiB": 23.4, "context_MiB": 600},
        "profile": {"gpu_busy_ms_per_chunk": 80, "gpu_idle_ms_per_chunk": 4, "kernel_sum_ms_per_chunk": 95,
                    "categories": [{"category": "GEMM FP8", "ms_per_chunk": 40, "pct": 42, "launches_per_chunk": 300}],
                    "top_kernels": [{"name": "nvjet_e4m3", "ms_per_chunk": 1.0}]},
    })
    oom = dc.summarize_case({"status": "oom", "phase": "load", "error": "CUDA out of memory. Tried to allocate",
                             "spec": {"mode": "standard", "steady_refresh_sdpa": True, "kind": "performance"}})
    corr = dc.summarize_case({"status": "ok", "spec": {"mode": "balanced", "steady_refresh_sdpa": True,
                                                       "kind": "correctness"}})
    corr["correctness_result"] = {"correctness": "PASS", "psnr_per_chunk": [33.1, 30.2], "problems": []}
    results = {
        "args": {"quick": False, "vram_cap_gb": 24.0, "label": "5080 @ 120 W"},
        "environment": {"gpu_name": "NVIDIA GeForce RTX 5090", "compute_capability": "12.0",
                        "nvidia_smi": {"driver_version": "580.1"}},
        "smoke": [{"name": "_scaled_mm FP8 row-wise", "status": "FAIL", "error": "RuntimeError: unsupported | sm_120"},
                  {"name": "bf16 matmul", "status": "PASS", "note": "ok"}],
        "power_limit": {"settable": False, "message": "Insufficient Permissions", "requested_w": 120},
        "cases": {"performance.balanced": perf, "performance.standard": oom, "correctness.balanced": corr},
        "gemm": [{"shape": "q/o 4096x4096", "bf16": {"tflops": 300.0}, "fp8_rowwise": {"error": "x"}}],
        "tasks": {"text_cache_build": {"status": "ok", "wall_seconds": 50.0, "host_peak_rss_GiB": 27.8}},
    }
    results["verdict"] = dc.verdict(results)
    return results


def test_report_renders_fake_results():
    results = _fake_results()
    report = dc.render_report(results)
    assert "RTX 5090" in report and "5080 @ 120 W" in report
    assert "performance.balanced | ok | 90.0" in report
    assert "keeping up" in report and "| 82 |" in report
    assert "performance.standard**: oom in load" in report
    assert "unsupported / sm_120" in report  # table pipes escaped
    assert "Insufficient Permissions" in report
    assert "error [670]" in report and "300.0 [722]" in report
    assert "GEMM FP8 | 40" in report
    assert "Attention kernel: sage (sm_120, self-test passed)" in report
    assert "[33.1, 30.2]" in report
    assert "host peak RSS 27.8 GiB" in report
    assert any("FAILED: _scaled_mm" in v for v in results["verdict"])
    assert any(v.startswith("performance.standard: OOM") for v in results["verdict"])
    json.dumps(results)  # serialisable as results.json
    assert dc.render_report({})  # an empty run still renders


def test_tarball_contents(tmp_path):
    out = tmp_path / "run1"
    (out / "logs").mkdir(parents=True)
    (out / "report.md").write_text("r")
    (out / "logs" / "smoke.log").write_text("l")
    (out / "trace.json").write_bytes(b"0" * (6 * 2**20))
    path, size_mb = dc.make_tarball(out)
    assert path == tmp_path / "run1.tar.gz" and size_mb < 1
    with tarfile.open(path) as tar:
        assert sorted(tar.getnames()) == ["run1/logs/smoke.log", "run1/report.md"]


def test_child_crash_and_oom_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", "/bin/false")
    crashed = dc.run_child({"task": "smoke"}, tmp_path / "a.log", timeout_s=30)
    assert crashed["status"] == "crashed" and "code 1" in crashed["error"]
    script = tmp_path / "fake_python"
    script.write_text("#!/bin/sh\necho 'torch.OutOfMemoryError: CUDA out of memory.'\nexit 1\n")
    script.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(script))
    assert dc.run_child({"task": "case"}, tmp_path / "b.log", timeout_s=30)["status"] == "oom"
    assert "out of memory" in (tmp_path / "b.log").read_text()


def test_kernel_categories_and_trace_breakdown():
    assert kernel_category("nvjet_sm120_tst_128x256_e4m3") == "GEMM FP8"
    assert kernel_category("sm90_xmma_gemm_bf16bf16_bf16f32") == "GEMM bf16"
    assert kernel_category("cudnn_generated_fort_native_sdpa_sm90") == "attention"
    assert kernel_category("_quantize_slices_kernel") == "FP8 activation quantize"
    assert kernel_category("_rotate_kernel") == "temporal RoPE (Triton)"
    assert kernel_category("triton_red_fused_mean_pow_rsqrt_0") == "Inductor Triton (fused norm/act/cat)"
    assert kernel_category("void at::native::vectorized_elementwise_kernel<4>") == "eager elementwise/reduce/copy"
    assert kernel_category("Memcpy DtoH (Device -> Pinned)") == "copies (memcpy H2D/D2H/D2D)"
    k = lambda name, ts, dur, tid=7: {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur,  # noqa: E731
                                     "tid": tid, "args": {"stream": tid}}
    trace = {"traceEvents": [
        k("nvjet_gemm_bf16", 0, 1000), k("cudnn_sdpa", 1000, 500),
        {"ph": "X", "cat": "gpu_user_annotation", "name": "dc_decode", "ts": 2000, "dur": 1000, "tid": 7},
        k("implicit_convolve_sgemm", 2100, 800),
        k("nvjet_gemm_bf16", 2200, 300, tid=9),  # side stream during decode: still GEMM
    ]}
    out = categorize_trace(trace, chunks=1)
    cats = {c["category"]: c["ms_per_chunk"] for c in out["categories"]}
    assert cats == {"GEMM bf16": 1.3, "decode (VAE) + frame conversion": 0.8, "attention": 0.5}
    assert out["gpu_busy_ms_per_chunk"] == 2.3 and out["gpu_idle_ms_per_chunk"] == 0.6


def test_downsample_box_average():
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[:, :4, :4] = 255
    small = downsample(frames)
    assert small.shape == (2, 2, 2, 3) and small[0, 0, 0, 0] == 255 and small[0, 1, 1, 0] == 0
