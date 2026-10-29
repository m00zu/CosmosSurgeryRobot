# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware test kit: can this GPU run Cosmos3-H-Dreams in real time?

One command checks the device's kernels (Triton, FP8 ``_scaled_mm``, SDPA
backends, varlen attention, Inductor), then loads each model mode under a VRAM
cap and measures load, compile, ms per chunk, stage split, memory, a profiler
kernel breakdown and FP8 GEMM throughput, and compares a deterministic rollout
with an H100 reference. It writes ``OUT/report.md``, ``OUT/results.json``, logs
and contact sheets, then packs ``OUT.tar.gz`` to send back. See
``docs/rtx5090-test-kit.md``.

Example::

    PYTHONPATH=cosmos3_h_dreams:flashdreams uv run --no-sync python -m \\
        cosmos3_h_dreams.device_check --checkpoint CKPT --out results/rtx5090 --quick

Every GPU task runs in its own child process (see ``_device_check_gpu``), so
an OOM, a sticky CUDA error or a crash is recorded for that task and the kit
goes on. This process never initialises CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET = "wan-taehv-chunk2-2steps-fademem-compile"
MODES = ("standard", "balanced", "low_memory")
DEFAULT_CHECKPOINT = "data_local/checkpoints/c3-h-d_2-step_model_bf16.safetensors"
DEFAULT_TEXT_CACHE = "data_local/text_kv/c3-h-d_2-step_480x832.pt"
SCENE = "assets/example_data/scenes/example_frame.png"
ACTIONS = "assets/example_data/scenes/trajectory.npy"
REFERENCE = "assets/device_check/h100_reference.npz"
CHECKPOINT_SHA256 = "3e6fce0046a6fe46c65307245df0db041c4a1828e0b3fc76708fb30e7851c97b"
TARBALL_LIMIT_MB = 20

FRAMES_PER_CHUNK = 8
INPUT_HZ = 30.0
BUDGET_MS = FRAMES_PER_CHUNK / INPUT_HZ * 1e3
"""One chunk turns 8 action rows into 8 frames; 30 Hz input fills it every 266.7 ms."""
KEEPING_UP_FRACTION = 0.75
"""p90 at or below 75% of the budget leaves headroom for decode jitter, the
browser stream and the host: "keeping up"."""
FIRST_CHUNK_MIN_PSNR = 20.0
CONSTANT_FRAME_STD = 2.0

H100 = {
    "standard": {"ms": 82.0, "smi_MiB": 15776},
    "balanced": {"ms": 82.0, "smi_MiB": 10600},
    "low_memory": {"ms": 84.0, "smi_MiB": 9184},
}
"""H100 80GB, this preset, offload_understanding, text-cache hit, steady_refresh_sdpa on:
free-running ms/chunk (step + decode + uint8 frames) and nvidia-smi per-process MiB."""
H100_SDPA_OFF_EXTRA_MS = 5.0
"""steady_refresh_sdpa off was about 5 ms per chunk slower on H100."""
H100_GEMM_TFLOPS = {
    "gate/up 12288x4096": {"bf16": 763, "fp8_rowwise": 760, "fp8_rowwise_fastacc": 1016,
                            "fp8_tensorwise": 1224, "fp8_tensorwise_fastacc": 1302},
    "down 4096x12288": {"bf16": 798, "fp8_rowwise": 686, "fp8_rowwise_fastacc": 1039,
                         "fp8_tensorwise": 1213, "fp8_tensorwise_fastacc": 1356},
    "q/o 4096x4096": {"bf16": 722, "fp8_rowwise": 670, "fp8_rowwise_fastacc": 683,
                       "fp8_tensorwise": 1019, "fp8_tensorwise_fastacc": 1250},
}
"""M = 788 microbenchmark on H100 (TFLOP/s)."""
H100_PROFILE_STANDARD = {
    "GEMM": 44.6, "attention (denoise + refresh)": 22.8, "FP8 activation quantize (Low memory)": 5.5,
    "eager elementwise (FadeMem prepare/commit)": 11.7, "decode (VAE)": 6.6,
}
"""REPORT.md 2.2, measured before the perf work (126 ms/chunk); for orientation only."""

QUICK = {"warmup_chunks": 5, "timed_chunks": 15, "stage_chunks": 8, "profile_chunks": 0, "correctness_chunks": 6}
FULL = {"warmup_chunks": 10, "timed_chunks": 30, "stage_chunks": 20, "profile_chunks": 3, "correctness_chunks": 12}


# --------------------------------------------------------------------------- arguments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m cosmos3_h_dreams.device_check",
        description="Cosmos3-H-Dreams hardware test kit (see docs/rtx5090-test-kit.md).",
    )
    p.add_argument("--checkpoint", default=os.environ.get("COSMOS3_H_DREAMS_CHECKPOINT", DEFAULT_CHECKPOINT),
                   help="2-step bf16 safetensors (default: $COSMOS3_H_DREAMS_CHECKPOINT or %(default)s)")
    p.add_argument("--text-cache", default=DEFAULT_TEXT_CACHE,
                   help="prompt K/V cache; built first when missing (needs ~26 GB more host RAM once)")
    p.add_argument("--out", required=False, help="output directory; OUT.tar.gz is written next to it")
    p.add_argument("--modes", default=",".join(MODES), type=_modes,
                   help="comma-separated subset of standard,balanced,low_memory")
    p.add_argument("--vram-cap-gb", type=float, default=24.0,
                   help="cap the whole process at this many GiB (0 = no cap); default 24 = laptop RTX 5090")
    p.add_argument("--power-limit", type=float, default=None,
                   help="try nvidia-smi -pl WATTS for the run and restore it afterwards (recorded if refused)")
    p.add_argument("--quick", action="store_true",
                   help="fewer chunks; no profiler, GEMM microbenchmark or steady_refresh_sdpa-off runs except Balanced")
    p.add_argument("--attention-kernel", choices=("auto", "default", "sage"), default="auto",
                   help="runtime.attention_kernel for every case (auto: SageAttention on RTX 50 when installed)")
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--timeout-min", type=float, default=45.0, help="per child task")
    p.add_argument("--label", default="", help="free text shown in the report (e.g. 'RunPod 5080 @175W')")
    p.add_argument("--write-reference", metavar="NPZ", help=argparse.SUPPRESS)
    p.add_argument("--child-spec", help=argparse.SUPPRESS)
    p.add_argument("--child-result", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child_spec is None and args.out is None and args.write_reference is None:
        p.error("--out is required")
    if args.vram_cap_gb < 0:
        p.error("--vram-cap-gb must be >= 0")
    return args


def _modes(text: str) -> list[str]:
    modes = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise argparse.ArgumentTypeError(f"unknown mode(s) {bad}; choose from {', '.join(MODES)}")
    return modes


def case_name(mode: str, sdpa: bool, kind: str) -> str:
    return f"{kind}.{mode}" + ("" if sdpa else ".sdpa_off")


def plan_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The model runs, in order: per mode a performance run, then its correctness run.

    ``steady_refresh_sdpa`` off is measured for Balanced (the laptop default),
    and in a full run for every mode.
    """
    knobs = QUICK if args.quick else FULL
    cases: list[dict[str, Any]] = []
    for mode in args.modes:
        sdpa_variants = [True, False] if (mode == "balanced" or not args.quick) else [True]
        for sdpa in sdpa_variants:
            cases.append({"kind": "performance", "mode": mode, "steady_refresh_sdpa": sdpa,
                          "deterministic": False, **{k: knobs[k] for k in
                          ("warmup_chunks", "timed_chunks", "stage_chunks", "profile_chunks")}})
        if args.skip_correctness:
            continue
        for sdpa in ([True, False] if mode == "balanced" and not args.quick else [True]):
            cases.append({"kind": "correctness", "mode": mode, "steady_refresh_sdpa": sdpa,
                          "deterministic": True, "chunks": knobs["correctness_chunks"]})
    for case in cases:
        case["name"] = case_name(case["mode"], case["steady_refresh_sdpa"], case["kind"])
    return cases


# --------------------------------------------------------------------------- statuses


def realtime_status(median_ms: float | None, p90_ms: float | None, budget_ms: float = BUDGET_MS) -> str:
    """``keeping up`` / ``tight`` / ``falling behind`` against the 30 Hz chunk budget."""
    if median_ms is None or p90_ms is None or math.isnan(median_ms):
        return "n/a"
    if median_ms > budget_ms:
        return "falling behind"
    if p90_ms <= KEEPING_UP_FRACTION * budget_ms:
        return "keeping up"
    return "tight"


def timing_summary(samples: list[float]) -> dict[str, float]:
    vals = sorted(samples)
    if not vals:
        return {}
    pick = lambda q: vals[min(len(vals) - 1, int(q * len(vals)))]  # noqa: E731
    return {"median": round(pick(0.5), 2), "p90": round(pick(0.9), 2), "mean": round(sum(vals) / len(vals), 2),
            "min": round(vals[0], 2), "max": round(vals[-1], 2), "n": len(vals)}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else round(10 * math.log10(255.0**2 / mse), 2)


def correctness_status(psnr_per_chunk: list[float], finite: list[bool], min_std: list[float]) -> tuple[str, list[str]]:
    """``FAIL`` only for gross failures: NaN/Inf, constant frames, chunk 0 below 20 dB."""
    problems = []
    if not all(finite):
        problems.append(f"non-finite video in chunk(s) {[i for i, f in enumerate(finite) if not f]}")
    flat = [i for i, s in enumerate(min_std) if s < CONSTANT_FRAME_STD]
    if flat:
        problems.append(f"constant frame(s) in chunk(s) {flat}")
    if psnr_per_chunk and psnr_per_chunk[0] < FIRST_CHUNK_MIN_PSNR:
        problems.append(f"first chunk {psnr_per_chunk[0]:.1f} dB < {FIRST_CHUNK_MIN_PSNR:.0f} dB vs H100")
    return ("FAIL" if problems else "PASS"), problems


def h100_ms(mode: str, sdpa: bool) -> float:
    return H100[mode]["ms"] + (0.0 if sdpa else H100_SDPA_OFF_EXTRA_MS)


# --------------------------------------------------------------------------- aggregation


def summarize_case(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn one child's raw result into the report's numbers (pure; no CUDA)."""
    spec = raw.get("spec", {})
    out: dict[str, Any] = {k: raw.get(k) for k in ("status", "phase", "error", "timing_s", "text_cache", "attention",
                                                       "vram_cap", "host_peak_rss_GiB", "task_seconds",
                                                       "allocated_after_release_GiB")}
    out.update(mode=spec.get("mode"), steady_refresh_sdpa=spec.get("steady_refresh_sdpa"), kind=spec.get("kind"))
    if raw.get("free_running_ms"):
        t = timing_summary(raw["free_running_ms"])
        out["ms_per_chunk"] = t
        out["realtime"] = realtime_status(t["median"], t["p90"])
        out["budget_margin_ms"] = round(BUDGET_MS - t["median"], 1)
        out["h100_ms"] = h100_ms(spec["mode"], spec["steady_refresh_sdpa"])
        out["vs_h100"] = round(t["median"] / out["h100_ms"], 2)
    for key in ("stage_ms", "power_during_timing", "profile", "mem_after_init", "mem_steady",
                "mem_after_timing", "mem_after_release", "peak_at_failure_or_end"):
        if key in raw:
            out[key] = raw[key]
    mem = raw.get("mem_after_timing") or raw.get("mem_steady")
    if mem:
        out["smi_MiB"] = mem.get("smi_process_MiB") or mem.get("smi_device_minus_baseline_MiB")
        out["h100_smi_MiB"] = H100.get(spec.get("mode"), {}).get("smi_MiB")
    return out


def compare_correctness(frames: np.ndarray, reference: np.ndarray | None, raw: dict[str, Any]) -> dict[str, Any]:
    """PSNR per chunk against the H100 reference, plus the gross-failure verdict."""
    out: dict[str, Any] = {"finite_per_chunk": raw.get("finite_per_chunk", []),
                           "min_frame_std_per_chunk": [round(s, 1) for s in raw.get("min_frame_std_per_chunk", [])]}
    if reference is None:
        out["psnr_per_chunk"] = []
        out["note"] = "no H100 reference for this case"
    else:
        n = min(len(frames), len(reference))
        out["psnr_per_chunk"] = [psnr(frames[i], reference[i]) for i in range(n)]
    out["correctness"], out["problems"] = correctness_status(
        out["psnr_per_chunk"], out["finite_per_chunk"], raw.get("min_frame_std_per_chunk", []))
    if reference is None and out["correctness"] == "PASS":
        out["correctness"] = "PASS (no reference)"
    return out


def load_reference(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        return {}, {}
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        return {k: data[k] for k in data.files if k != "meta"}, meta


def contact_sheet(frames: np.ndarray, reference: np.ndarray | None, path: Path, label: str) -> None:
    """Frame 0 of each chunk: this device on top, the H100 reference below."""
    from PIL import Image, ImageDraw

    rows = [np.concatenate(list(frames[:, 0]), axis=1)]
    if reference is not None:
        n = min(len(frames), len(reference))
        ref = np.concatenate(list(reference[:n, 0]), axis=1)
        rows[0] = rows[0][:, : ref.shape[1]]
        rows.append(ref)
    image = Image.fromarray(np.concatenate(rows, axis=0))
    draw = ImageDraw.Draw(image)
    draw.text((4, 2), f"{label} (top: this GPU" + (", bottom: H100)" if reference is not None else ")"),
              fill=(255, 255, 0))
    image.save(path)


# --------------------------------------------------------------------------- report


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(results: dict[str, Any]) -> str:
    """Markdown report from ``results.json`` contents (pure; testable without a GPU)."""
    env = results.get("environment") or {}
    smi = env.get("nvidia_smi") or {}
    lines = [
        f"# Cosmos3-H-Dreams device check: {env.get('gpu_name', 'unknown GPU')}",
        "",
        f"{results.get('started_at', '')} · kit `{results.get('git_commit', '?')}` · "
        f"{'quick' if results.get('args', {}).get('quick') else 'full'} run"
        + (f" · {results['args']['label']}" if results.get("args", {}).get("label") else ""),
        "",
        f"Preset `{PRESET}`, 480×832, `offload_understanding`, text cache. One chunk = 8 action rows "
        f"in, 8 frames out. Real-time budget at 30 Hz input: **{BUDGET_MS:.0f} ms per chunk** "
        f"(keeping up: p90 ≤ {KEEPING_UP_FRACTION * BUDGET_MS:.0f} ms; tight: median ≤ {BUDGET_MS:.0f} ms; "
        "otherwise falling behind). H100 columns are the reference numbers in docs/rtx5090-test-kit.md.",
        "",
        "## Verdict",
        "",
    ]
    lines += [f"- {v}" for v in results.get("verdict", [])] or ["- (no verdict: the run did not finish)"]
    lines += ["", "## Environment", "", "| Item | Value |", "|---|---|"]
    rows = [
        ("GPU", f"{env.get('gpu_name')} (sm {env.get('compute_capability')}, {env.get('sm_count')} SMs, "
                f"{env.get('total_memory_GiB')} GiB)"),
        ("Driver / CUDA (torch)", f"{smi.get('driver_version')} / {env.get('torch_cuda')}"),
        ("torch / triton / cuDNN", f"{env.get('torch')} / {env.get('triton')} / {env.get('cudnn')}"),
        ("torch arch list", ", ".join(env.get("torch_arch_list") or [])),
        ("Power limit (current / default / min / max)",
         " / ".join(str(smi.get(k)) for k in ("power.limit", "power.default_limit", "power.min_limit",
                                               "power.max_limit"))),
        ("Max clocks SM / memory", f"{smi.get('clocks.max.sm')} / {smi.get('clocks.max.mem')}"),
        ("PCIe max", f"gen {smi.get('pcie.link.gen.max')} x{smi.get('pcie.link.width.max')}"),
        ("Host", f"{env.get('cpu')} ({env.get('cpu_count')} threads), {env.get('host_ram_GiB')} GiB RAM, "
                 f"container: {env.get('in_container')}"),
        ("Python / platform", f"{env.get('python')} / {env.get('platform')}"),
        ("Checkpoint", results.get("checkpoint_note", "–")),
        ("VRAM cap", f"{results.get('args', {}).get('vram_cap_gb')} GiB requested"),
    ]
    lines += [f"| {a} | {b} |" for a, b in rows]

    power = results.get("power_limit") or {}
    lines += ["", "## Power limit", ""]
    if power:
        lines.append(
            f"`nvidia-smi -pl` permitted: **{power.get('settable')}** ({power.get('message', '')}). "
            f"Requested: {power.get('requested_w') or 'none (card default)'}; applied: {power.get('applied')}; "
            f"limit during the run: {(power.get('after') or {}).get('power.limit', '–')}; "
            f"restore: {power.get('restore', 'n/a')}."
        )
    else:
        lines.append("Not probed.")

    build = (results.get("tasks") or {}).get("text_cache_build")
    if build:
        lines += ["", f"Text cache built in this run: {build.get('status')}, {build.get('wall_seconds')} s, "
                      f"host peak RSS {build.get('host_peak_rss_GiB')} GiB."]

    smoke = results.get("smoke") or []
    lines += ["", "## Smoke tests", "", "| Test | Status | Detail |", "|---|---|---|"]
    for t in smoke:
        detail = t.get("error") or t.get("note") or ""
        lines.append(f"| {t.get('name')} | {t.get('status')} | {str(detail).replace('|', '/')[:220]} |")
    if not smoke:
        lines.append("| (smoke task failed; see logs/smoke.log) | FAIL | |")

    cases = results.get("cases") or {}
    perf = {k: v for k, v in cases.items() if v.get("kind") == "performance"}
    lines += ["", "## Speed and memory", "",
              "| Case | Status | ms/chunk median | p90 | H100 ms | vs H100 | Real time | Margin ms | "
              "Alloc GiB | Reserved GiB | Max alloc GiB | nvidia-smi MiB | H100 MiB | Load s | Compile+warmup s | "
              "Host RSS GiB |",
              "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in perf.items():
        t = c.get("ms_per_chunk") or {}
        m = c.get("mem_after_timing") or c.get("peak_at_failure_or_end") or {}
        timing = c.get("timing_s") or {}
        lines.append(
            f"| {name} | {c.get('status')} | {_fmt(t.get('median'))} | {_fmt(t.get('p90'))} | "
            f"{_fmt(c.get('h100_ms'), 0)} | {_fmt(c.get('vs_h100'), 2)} | {c.get('realtime', '–')} | "
            f"{_fmt(c.get('budget_margin_ms'))} | {_fmt(m.get('allocated_GiB'), 2)} | {_fmt(m.get('reserved_GiB'), 2)} | "
            f"{_fmt(m.get('max_allocated_GiB'), 2)} | {_fmt(c.get('smi_MiB'))} | {_fmt(c.get('h100_smi_MiB'))} | "
            f"{_fmt(timing.get('model_load'))} | {_fmt(timing.get('scene_compile_warmup'))} | "
            f"{_fmt(c.get('host_peak_rss_GiB'))} |"
        )
    kernels = sorted({f"{a['kernel']} ({a['reason']})" for c in cases.values() if (a := c.get("attention"))})
    if kernels:
        lines += ["", "Attention kernel: " + "; ".join(kernels)]
    caps = {(c.get("vram_cap") or {}).get("note") or
            f"allocator capped at {(c.get('vram_cap') or {}).get('allocator_cap_GiB')} GiB "
            f"(+{(c.get('vram_cap') or {}).get('context_MiB')} MiB CUDA context)"
            for c in cases.values() if c.get("vram_cap")}
    if caps:
        lines += ["", "VRAM cap: " + "; ".join(sorted(caps)) + "."]
    lines += ["", "Synced stage medians (ms; `profile=True` adds syncs, for attribution only):", "",
              "| Case | Model step | Setup | Denoise | Cache seed/prune | Output | Refresh submit | Refresh wait "
              "| Decode + frames | Refresh tail |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in perf.items():
        s = c.get("stage_ms") or {}
        lines.append(f"| {name} | " + " | ".join(_fmt(s.get(k)) for k in (
            "model_step_s", "setup_s", "denoise_s", "cache_seed_prune_s", "output_s", "cache_update_submit_s",
            "cache_update_wait_s", "decode_plus_frames", "refresh_tail_after_decode")) + " |")
    lines += ["", "Power and clocks while timing (NVML, 100 ms samples):", "",
              "| Case | Power W median / max | SM MHz median | Mem MHz | Temp °C max | Clock-limit reasons (% of samples) |",
              "|---|---|---:|---:|---:|---|"]
    for name, c in perf.items():
        pw = c.get("power_during_timing") or {}
        lines.append(
            f"| {name} | {_fmt(pw.get('power_w_median'))} / {_fmt(pw.get('power_w_max'))} | "
            f"{_fmt(pw.get('sm_mhz_median'), 0)} | {_fmt(pw.get('mem_mhz_median'), 0)} | {_fmt(pw.get('temp_c_max'), 0)} | "
            f"{pw.get('clock_limit_reasons_pct', '–')} |")

    corr = {k: v for k, v in cases.items() if v.get("kind") == "correctness"}
    lines += ["", "## Correctness against the H100 reference", ""]
    ref_meta = results.get("reference_meta") or {}
    if ref_meta:
        lines.append(
            f"Deterministic rollout, example scene, `trajectory.npy`, seed 0; frames {ref_meta.get('keep_frames')} of "
            f"each chunk at 1/{ref_meta.get('downsample')} resolution. Reference: {ref_meta.get('gpu')}, torch "
            f"{ref_meta.get('torch')}. Cross-hardware output is never bit-identical and drifts as the rollout "
            "goes on; only gross failures fail (NaN/Inf, constant frames, chunk 0 < 20 dB).")
        drift = ref_meta.get("h100_nondeterministic_psnr") or {}
        if drift:
            lines += ["", "For scale, H100 run-to-run (non-deterministic vs deterministic) PSNR per chunk: " +
                      "; ".join(f"{m}: {v}" for m, v in drift.items())]
    lines += ["", "| Case | Status | Verdict | PSNR per chunk (dB) | Problems | Contact sheet |", "|---|---|---|---|---|---|"]
    for name, c in corr.items():
        cc = c.get("correctness_result") or {}
        lines.append(
            f"| {name} | {c.get('status')} | {cc.get('correctness', '–')} | {cc.get('psnr_per_chunk', '–')} | "
            f"{'; '.join(cc.get('problems', [])) or '–'} | {c.get('contact_sheet', '–')} |")

    profiled = {k: v for k, v in perf.items() if isinstance(v.get("profile"), dict) and v["profile"].get("categories")}
    if profiled:
        lines += ["", "## Kernel time per chunk (torch.profiler)", "",
                  "Kernels inside CUDA graphs included. The refresh runs on a side stream and overlaps decode, "
                  "so the kernel sum can exceed the busy time, and decode kernels slow down while they share "
                  "the GPU with it. H100 Standard before the perf work "
                  f"(REPORT.md 2.2, ms): {H100_PROFILE_STANDARD}.", ""]
        for name, c in profiled.items():
            pr = c["profile"]
            lines += [f"**{name}**: GPU busy {pr['gpu_busy_ms_per_chunk']} ms, idle {pr['gpu_idle_ms_per_chunk']} ms, "
                      f"kernel sum {pr['kernel_sum_ms_per_chunk']} ms per chunk.", "",
                      "| Category | ms/chunk | % | Launches/chunk |", "|---|---:|---:|---:|"]
            lines += [f"| {r['category']} | {r['ms_per_chunk']} | {r['pct']} | {r['launches_per_chunk']} |"
                      for r in pr["categories"]]
            lines += ["", "Top kernels: " + "; ".join(f"`{k['name'][:60]}` {k['ms_per_chunk']} ms"
                                                       for k in pr["top_kernels"][:6]), ""]

    gemm = results.get("gemm") or []
    if gemm:
        lines += ["## FP8 `_scaled_mm` microbenchmark (M = 788, TFLOP/s; H100 in brackets)", "",
                  "| Shape | bf16 | FP8 row-wise (current) | + fast accum | per-tensor | per-tensor + fast accum |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in gemm:
            cells = []
            for key in ("bf16", "fp8_rowwise", "fp8_rowwise_fastacc", "fp8_tensorwise", "fp8_tensorwise_fastacc"):
                v = row.get(key) or {}
                ref = H100_GEMM_TFLOPS.get(row["shape"], {}).get(key)
                cells.append(f"{v['tflops']} [{ref}]" if "tflops" in v else f"error [{ref}]")
            lines.append(f"| {row['shape']} | " + " | ".join(cells) + " |")
        lines.append("")

    failures = [(n, c) for n, c in cases.items() if c.get("status") not in ("ok", None)]
    failures += [(n, t) for n, t in (results.get("tasks") or {}).items() if t.get("status") not in ("ok", None)]
    if failures:
        lines += ["## Failures", ""]
        for name, c in failures:
            lines.append(f"- **{name}**: {c.get('status')} in {c.get('phase', '?')}: "
                         f"{str(c.get('error', ''))[:400]} (log: `logs/{name}.log`)")
        lines.append("")
    lines += ["## Files", "", "`results.json` (all numbers), `logs/` (one log per task), `env/nvidia-smi-q.txt`, "
              "`cases/*/contact_sheet.png`, `cases/*/frames_small.npy`.", ""]
    return "\n".join(lines)


def verdict(results: dict[str, Any]) -> list[str]:
    out = []
    smoke = results.get("smoke") or []
    failed = [t["name"] for t in smoke if t.get("status") == "FAIL"]
    count = lambda status: sum(t.get("status") == status for t in smoke)  # noqa: E731
    out.append(f"Smoke tests: {count('PASS')} passed, {len(failed)} failed, {count('N/A')} N/A (no kernel for "
               "those inputs on any GPU), " f"{count('WARN')} warnings"
               + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    for name, c in (results.get("cases") or {}).items():
        if c.get("kind") == "performance":
            if c.get("status") == "ok":
                t = c["ms_per_chunk"]
                out.append(f"{name}: {t['median']:.1f} ms/chunk (p90 {t['p90']:.1f}) -> {c['realtime']}, "
                           f"{c['vs_h100']:.2f}x H100; nvidia-smi {_fmt(c.get('smi_MiB'))} MiB")
            else:
                out.append(f"{name}: {str(c.get('status')).upper()} ({str(c.get('error', ''))[:120]})")
        elif c.get("kind") == "correctness":
            cc = c.get("correctness_result") or {}
            out.append(f"{name}: {cc.get('correctness', str(c.get('status')).upper())}"
                       + (f", chunk 0 {cc['psnr_per_chunk'][0]} dB" if cc.get("psnr_per_chunk") else ""))
    return out


# --------------------------------------------------------------------------- packing


def make_tarball(out_dir: Path, limit_mb: float = TARBALL_LIMIT_MB) -> tuple[Path, float]:
    """``OUT.tar.gz`` next to ``OUT``; large files (traces, >5 MB) are left out."""
    tar_path = out_dir.parent / f"{out_dir.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and path.stat().st_size <= 5 * 2**20 and path.suffix != ".tmp":
                tar.add(path, arcname=str(Path(out_dir.name) / path.relative_to(out_dir)))
    size_mb = tar_path.stat().st_size / 2**20
    if size_mb > limit_mb:
        print(f"[device-check] warning: {tar_path} is {size_mb:.1f} MB (> {limit_mb} MB)", flush=True)
    return tar_path, size_mb


# --------------------------------------------------------------------------- orchestration


def _nvml_index_guess() -> int:
    """NVML index of CUDA device 0 without creating a CUDA context.

    The smoke task re-checks it by UUID; a mismatch is reported there.
    """
    first = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def _device_used_mib(index: int) -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).used / 2**20
    except Exception:  # noqa: BLE001
        return None


def run_child(spec: dict[str, Any], log_path: Path, timeout_s: float) -> dict[str, Any]:
    """Run one GPU task in a fresh interpreter; a crash or timeout becomes a result."""
    result_path = log_path.with_suffix(".result.json")
    result_path.unlink(missing_ok=True)
    env = dict(os.environ)
    if spec.get("deterministic"):
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cmd = [sys.executable, "-m", "cosmos3_h_dreams.device_check",
           "--child-spec", json.dumps(spec), "--child-result", str(result_path)]
    started = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                bufsize=1)

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith("["):
                    print(f"    {line.rstrip()[:200]}", flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        reader.join(timeout=5)
    seconds = round(time.perf_counter() - started, 1)
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        result_path.unlink()
    else:
        tail = log_path.read_text()[-3000:] if log_path.is_file() else ""
        oom = "out of memory" in tail.lower()
        timed_out = proc.returncode is not None and seconds >= timeout_s
        result = {"status": "timeout" if timed_out else ("oom" if oom else "crashed"), "phase": "child",
                  "error": f"child exited with code {proc.returncode} after {seconds} s; log tail: {tail[-800:]}"}
    result["wall_seconds"] = seconds
    return result


class Kit:
    """Runs the tasks in order and rewrites ``results.json`` / ``report.md`` after each one."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out).resolve()
        for sub in ("logs", "env", "cases"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.nvml_index = _nvml_index_guess()
        self.reference, self.reference_meta = load_reference(REPO_ROOT / REFERENCE)
        self.results: dict[str, Any] = {
            "kit": "cosmos3_h_dreams.device_check",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git(["rev-parse", "--short", "HEAD"]),
            "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "args": {k: v for k, v in vars(args).items() if not k.startswith("child")},
            "budget_ms": round(BUDGET_MS, 1),
            "h100": {"modes": H100, "sdpa_off_extra_ms": H100_SDPA_OFF_EXTRA_MS, "gemm_tflops": H100_GEMM_TFLOPS},
            "reference_meta": self.reference_meta,
            "tasks": {},
            "cases": {},
        }
        self.timeout_s = args.timeout_min * 60

    def base_spec(self) -> dict[str, Any]:
        return {"checkpoint": str(Path(self.args.checkpoint).resolve()), "preset": PRESET,
                "scene": str(REPO_ROOT / SCENE), "actions": str(REPO_ROOT / ACTIONS),
                "text_cache": str(Path(self.args.text_cache).resolve()), "nvml_index": self.nvml_index,
                "vram_cap_gb": self.args.vram_cap_gb, "attention_kernel": self.args.attention_kernel}

    def save(self) -> None:
        self.results["verdict"] = verdict(self.results)
        (self.out / "results.json").write_text(json.dumps(self.results, indent=1, default=str))
        (self.out / "report.md").write_text(render_report(self.results))

    def task(self, name: str, spec: dict[str, Any]) -> dict[str, Any]:
        print(f"[device-check] {name} ...", flush=True)
        spec = {**spec, "out_dir": str(self.out / "cases" / name)}
        spec.setdefault("baseline_used_mib", _device_used_mib(self.nvml_index))
        result = run_child(spec, self.out / "logs" / f"{name}.log", self.timeout_s)
        print(f"[device-check] {name}: {result.get('status', 'ok')} ({result['wall_seconds']} s)", flush=True)
        return result

    def run(self) -> Path:
        args = self.args
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            raise SystemExit(f"checkpoint not found: {ckpt} (see docs/rtx5090-test-kit.md, 'Get the checkpoint')")
        self.results["checkpoint_note"] = f"{ckpt.name}, {ckpt.stat().st_size / 1e9:.2f} GB"

        smoke = self.task("smoke", {"task": "smoke"})
        if "environment" in smoke:
            (self.out / "env").mkdir(exist_ok=True)
            for f in ("nvidia-smi-q.txt", "nvidia-smi.txt"):
                src = self.out / "cases" / "smoke" / f
                if src.is_file():
                    src.replace(self.out / "env" / f)
        self.nvml_index = smoke.get("nvml_index", self.nvml_index)
        self.results["environment"] = smoke.get("environment")
        self.results["smoke"] = smoke.get("smoke", [])
        self.results["tasks"]["smoke"] = {k: smoke.get(k) for k in ("status", "error", "wall_seconds")}
        self.save()

        from cosmos3_h_dreams._device_check_gpu import power_limit_probe, restore_power_limit

        power = power_limit_probe(self.nvml_index, args.power_limit)
        self.results["power_limit"] = power
        self.save()
        try:
            self._run_models()
        finally:
            if power.get("applied"):
                power["restore"] = restore_power_limit(self.nvml_index, power["original_w"])
            self.save()
        self.results["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.save()
        tar_path, size_mb = make_tarball(self.out)
        print(f"[device-check] report: {self.out / 'report.md'}", flush=True)
        print(f"[device-check] send back: {tar_path} ({size_mb:.1f} MB)", flush=True)
        return tar_path

    def _run_models(self) -> None:
        args = self.args
        if not args.quick:
            gemm = self.task("gemm", {"task": "gemm"})
            self.results["gemm"] = gemm.get("gemm", [])
            self.results["tasks"]["gemm"] = {k: gemm.get(k) for k in ("status", "error", "wall_seconds")}
            self.save()
        if not Path(args.text_cache).is_file():
            prep = self.task("text_cache_build", {
                **self.base_spec(), "task": "case", "kind": "prep", "mode": "balanced",
                "steady_refresh_sdpa": True, "deterministic": False,
                "overrides": {"compile_denoise": False, "cuda_graph_denoise": False}})
            self.results["tasks"]["text_cache_build"] = {
                k: prep.get(k) for k in ("status", "error", "wall_seconds", "host_peak_rss_GiB", "text_cache")}
            self.save()
        for case in plan_cases(args):
            name = case["name"]
            raw = self.task(name, {**self.base_spec(), "task": "case", **case})
            summary = summarize_case(raw)
            if case["kind"] == "correctness" and raw.get("status") == "ok":
                frames = np.load(self.out / "cases" / name / "frames_small.npy")
                ref = self.reference.get(case["mode"])
                summary["correctness_result"] = compare_correctness(frames, ref, raw)
                sheet = self.out / "cases" / name / "contact_sheet.png"
                contact_sheet(frames, ref, sheet, name)
                summary["contact_sheet"] = str(sheet.relative_to(self.out))
            self.results["cases"][name] = summary
            self.save()


def _git(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", *cmd], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def write_reference(args: argparse.Namespace) -> None:
    """Regenerate the committed H100 reference (maintainers; needs the reference GPU)."""
    args.out = args.out or str(Path(args.write_reference).with_suffix("")) + "_work"
    kit = Kit(args)
    smoke = kit.task("smoke", {"task": "smoke"})
    kit.nvml_index = smoke.get("nvml_index", kit.nvml_index)
    env = smoke.get("environment") or {}
    arrays: dict[str, np.ndarray] = {}
    drift: dict[str, list[float]] = {}
    chunks = FULL["correctness_chunks"]
    for mode in args.modes:
        base = {**kit.base_spec(), "task": "case", "kind": "correctness", "mode": mode,
                "steady_refresh_sdpa": True, "chunks": chunks, "vram_cap_gb": 0}
        det = kit.task(f"ref.{mode}", {**base, "deterministic": True})
        if det.get("status") != "ok":
            raise SystemExit(f"reference run failed for {mode}: {det.get('error')}")
        arrays[mode] = np.load(kit.out / "cases" / f"ref.{mode}" / "frames_small.npy")
        nondet = kit.task(f"ref.{mode}.nondet", {**base, "deterministic": False})
        if nondet.get("status") == "ok":
            other = np.load(kit.out / "cases" / f"ref.{mode}.nondet" / "frames_small.npy")
            drift[mode] = [psnr(other[i], arrays[mode][i]) for i in range(len(other))]
    from cosmos3_h_dreams._device_check_gpu import DOWNSAMPLE, KEEP_FRAMES

    meta = {"gpu": env.get("gpu_name"), "torch": env.get("torch"), "cudnn": env.get("cudnn"),
            "git_commit": _git(["rev-parse", "--short", "HEAD"]), "preset": PRESET, "seed": 0,
            "chunks": chunks, "keep_frames": list(KEEP_FRAMES), "downsample": DOWNSAMPLE,
            "steady_refresh_sdpa": True, "deterministic": True, "h100_nondeterministic_psnr": drift,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = Path(args.write_reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    print(f"[device-check] wrote {path} ({path.stat().st_size / 2**20:.2f} MB): {json.dumps(meta)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.child_spec is not None:
        from cosmos3_h_dreams._device_check_gpu import child_main

        return child_main(args.child_spec, args.child_result)
    if args.write_reference:
        write_reference(args)
        return 0
    Kit(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
