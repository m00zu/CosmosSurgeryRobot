# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU side of :mod:`cosmos3_h_dreams.device_check`; each task runs in a child process.

A child process per task keeps a sticky CUDA error, an OOM or a crash in one
mode from taking the other modes with it, and lets the correctness runs set
``CUBLAS_WORKSPACE_CONFIG`` before cuBLAS starts. Every task writes one JSON
result; the parent never touches CUDA.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

GiB = 2**30
SDPA_SHAPE = {"heads": 32, "kv_heads": 8, "q_tokens": 788, "k_tokens": 5559, "head_dim": 128}
"""One denoise attention call of the 480x832 chunk2 preset (REPORT.md, model shape)."""
GEMM_SHAPES = {
    "gate/up 12288x4096": (12288, 4096),
    "down 4096x12288": (4096, 12288),
    "q/o 4096x4096": (4096, 4096),
}
"""``N x K`` of the generation projections; M is the 788 tokens of one chunk."""
GEMM_M = 788
DOWNSAMPLE = 4
"""Correctness frames are compared at 1/4 resolution (4x4 box average)."""
KEEP_FRAMES = (0, 4)
"""Frames of each 8-frame chunk kept for the correctness comparison."""


def downsample(frames: np.ndarray, factor: int = DOWNSAMPLE) -> np.ndarray:
    """Box-average ``[..., H, W, C]`` uint8 frames by ``factor`` (H, W divisible)."""
    *lead, h, w, c = frames.shape
    x = frames.reshape(*lead, h // factor, factor, w // factor, factor, c).astype(np.float32)
    return np.rint(x.mean(axis=(-4, -2))).astype(np.uint8)


# --------------------------------------------------------------------------- nvml


class Nvml:
    """Best-effort NVML reads for one device; every method returns None on failure."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception:  # noqa: BLE001 - NVML is optional
            self.nv = None

    def _call(self, fn: Callable[[], Any]) -> Any:
        if self.handle is None:
            return None
        try:
            return fn()
        except Exception:  # noqa: BLE001
            return None

    def used_mib(self) -> float | None:
        info = self._call(lambda: self.nv.nvmlDeviceGetMemoryInfo(self.handle))
        return None if info is None else info.used / 2**20

    def process_mib(self, pid: int) -> float | None:
        procs = self._call(lambda: self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle))
        for proc in procs or []:
            if proc.pid == pid and proc.usedGpuMemory is not None:
                return proc.usedGpuMemory / 2**20
        return None

    def sample(self) -> dict[str, Any]:
        nv, h = self.nv, self.handle
        reasons = self._call(lambda: nv.nvmlDeviceGetCurrentClocksEventReasons(h))
        if reasons is None:
            reasons = self._call(lambda: nv.nvmlDeviceGetCurrentClocksThrottleReasons(h))
        power = self._call(lambda: nv.nvmlDeviceGetPowerUsage(h))
        return {
            "power_w": None if power is None else power / 1000,
            "sm_mhz": self._call(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)),
            "mem_mhz": self._call(lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)),
            "temp_c": self._call(lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
            "reasons": reasons,
        }


THROTTLE_BITS = {
    0x1: "gpu_idle",
    0x2: "app_clocks",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
    0x100: "display_clocks",
}


class PowerSampler:
    """Samples power, clocks, temperature and clock-limit reasons every 100 ms."""

    def __init__(self, nvml: Nvml, period_s: float = 0.1) -> None:
        self.nvml, self.period_s = nvml, period_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self.nvml.sample())
            self._stop.wait(self.period_s)

    def __enter__(self) -> "PowerSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"samples": len(self.samples)}
        for key in ("power_w", "sm_mhz", "mem_mhz", "temp_c"):
            vals = sorted(s[key] for s in self.samples if s[key] is not None)
            if vals:
                out[f"{key}_median"] = round(vals[len(vals) // 2], 1)
                out[f"{key}_max"] = round(vals[-1], 1)
        seen: dict[str, int] = defaultdict(int)
        for s in self.samples:
            for bit, name in THROTTLE_BITS.items():
                if s["reasons"] is not None and s["reasons"] & bit:
                    seen[name] += 1
        n = max(1, len(self.samples))
        out["clock_limit_reasons_pct"] = {k: round(100 * v / n) for k, v in seen.items()}
        return out


# --------------------------------------------------------------------------- environment


def _run(cmd: list[str], timeout: float = 60) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as error:  # noqa: BLE001
        return -1, repr(error)


def smi_query(index: int, fields: list[str]) -> dict[str, str]:
    """``nvidia-smi --query-gpu`` for one device; failed fields map to an error text."""
    out: dict[str, str] = {}
    for field in fields:  # one call per field: an unknown field fails the whole query
        code, text = _run(
            ["nvidia-smi", "-i", str(index), f"--query-gpu={field}", "--format=csv,noheader"]
        )
        out[field] = text if code == 0 else f"unavailable ({text.splitlines()[-1] if text else code})"
    return out


SMI_FIELDS = [
    "name", "pci.bus_id", "compute_cap", "memory.total", "driver_version", "vbios_version",
    "power.limit", "power.default_limit", "power.min_limit", "power.max_limit",
    "enforced.power.limit", "power.draw", "clocks.max.sm", "clocks.max.mem",
    "clocks.sm", "clocks.mem", "temperature.gpu", "pcie.link.gen.max", "pcie.link.width.max",
    "persistence_mode", "display_active",
]


def collect_environment(nvml_index: int, out_dir: Path) -> dict[str, Any]:
    """GPU, driver and software versions; writes the full ``nvidia-smi -q`` dump."""
    import torch

    env: dict[str, Any] = {"nvml_index": nvml_index}
    try:
        import triton

        env["triton"] = triton.__version__
    except Exception as error:  # noqa: BLE001
        env["triton"] = f"unavailable ({error})"
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    env.update(
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        torch_arch_list=torch.cuda.get_arch_list(),
        gpu_name=props.name,
        compute_capability=f"{cap[0]}.{cap[1]}",
        sm_count=props.multi_processor_count,
        total_memory_GiB=round(props.total_memory / GiB, 2),
        python=sys.version.split()[0],
        platform=platform.platform(),
        cpu=_cpu_model(),
        cpu_count=os.cpu_count(),
        host_ram_GiB=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GiB, 1),
        in_container=Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        pytorch_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF"),
    )
    env["nvidia_smi"] = smi_query(nvml_index, SMI_FIELDS)
    code, dump = _run(["nvidia-smi", "-q", "-i", str(nvml_index)])
    (out_dir / "nvidia-smi-q.txt").write_text(dump + "\n")
    code, top = _run(["nvidia-smi"])
    (out_dir / "nvidia-smi.txt").write_text(top + "\n")
    env["nvidia_smi_q_ok"] = code == 0
    return env


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def power_limit_probe(index: int, requested_w: float | None) -> dict[str, Any]:
    """Whether ``nvidia-smi -pl`` works here; applies ``requested_w`` when it does.

    Without a request the current limit is written back unchanged, which only
    tests the permission.
    """
    q = smi_query(index, ["power.limit", "power.default_limit", "power.min_limit", "power.max_limit"])
    out: dict[str, Any] = {"before": q, "requested_w": requested_w}
    try:
        current = float(q["power.limit"].split()[0])
    except (ValueError, IndexError):
        out.update(settable=False, message=f"cannot read the power limit: {q['power.limit']}")
        return out
    target = current if requested_w is None else float(requested_w)
    code, text = _run(["nvidia-smi", "-i", str(index), "-pl", f"{target:g}"])
    out["settable"] = code == 0
    out["message"] = text.splitlines()[0] if text else ""
    out["applied"] = code == 0 and requested_w is not None
    out["original_w"] = current
    out["after"] = smi_query(index, ["power.limit", "enforced.power.limit"])
    return out


def restore_power_limit(index: int, watts: float) -> str:
    code, text = _run(["nvidia-smi", "-i", str(index), "-pl", f"{watts:g}"])
    return "restored" if code == 0 else f"restore failed: {text}"


# --------------------------------------------------------------------------- smoke tests


def _bench_ms(fn: Callable[[], Any], iters: int = 20) -> float:
    import torch

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _rel_err(a: Any, b: Any) -> float:
    a, b = a.detach().float(), b.detach().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _kernel_names(fn: Callable[[], Any]) -> list[str]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type.name == "CUDA"]


def sdpa_backend_of(names: list[str]) -> str:
    """Name the SDPA backend from the CUDA kernels one call launched."""
    text = " ".join(names).lower()
    if "cudnn" in text or "sdpa" in text and "fort" in text:
        return "cudnn"
    if "flash" in text:
        return "flash"
    if "fmha" in text or "efficient" in text or "mem_eff" in text:
        return "efficient"
    if "softmax" in text:
        return "math"
    return "unknown: " + ", ".join(sorted(set(names)))[:200]


def _smoke_tests(nvml_index: int) -> list[dict[str, Any]]:
    import torch

    tests: list[dict[str, Any]] = []

    def check(name: str, fn: Callable[[], dict[str, Any] | None]) -> None:
        started = time.perf_counter()
        try:
            detail = fn() or {}
            status = detail.pop("status", "PASS")
        except Exception as error:  # noqa: BLE001 - a failure is a result
            detail = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()[-3000:]}
            status = "FAIL"
        torch.cuda.synchronize() if status != "FAIL" else None
        detail.update(name=name, status=status, seconds=round(time.perf_counter() - started, 2))
        print(f"[smoke] {status:4} {name} {detail.get('error') or detail.get('note') or ''}", flush=True)
        tests.append(detail)

    dev = "cuda"

    def arch() -> dict[str, Any]:
        major, minor = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        want = f"sm_{major}{minor}"
        ok = want in archs or any(a.startswith(f"sm_{major}") and a.endswith("a") for a in archs)
        return {"status": "PASS" if ok else "FAIL", "note": f"device {want}; torch built for {archs}"}

    def nvml_match() -> dict[str, Any]:
        from cosmos3_h_dreams.headset_server import _nvml_index

        try:
            import pynvml
        except ImportError:
            return {"status": "WARN", "note": "pynvml missing; nvidia-smi numbers use index 0"}
        found = _nvml_index(0, nvml=pynvml)
        ok = found == nvml_index
        return {"status": "PASS" if ok else "WARN", "nvml_index": found,
                "note": f"CUDA device 0 is NVML device {found} (kit used {nvml_index})"}

    def bf16_matmul() -> dict[str, Any]:
        a = torch.randn(788, 4096, device=dev, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
        err = _rel_err(a @ b, a.float() @ b.float())
        return {"status": "PASS" if err < 1e-2 else "FAIL", "rel_err": err, "note": f"rel err {err:.2e}"}

    def rotate_temporal() -> dict[str, Any]:
        from cosmos3_h_dreams._runtime import _temporal_rope as tr

        if tr.triton is None:
            return {"status": "FAIL", "note": "triton not importable; the eager fallback would run"}
        key = torch.randn(1, 8, 4400, 128, device=dev, dtype=torch.bfloat16)
        cos = torch.rand(4400, 128, device=dev)
        sin = torch.rand(4400, 128, device=dev)
        assert tr._triton_supported(key, cos, sin), "the Triton path is not taken for this shape"
        got = tr.rotate_temporal(key, cos, sin)
        ref = tr.rotate_half_reference(key, cos, sin, 1)
        diff = float((got.float() - ref.float()).abs().max())
        return {"status": "PASS" if diff == 0 else "FAIL", "max_abs_diff": diff,
                "note": "bit-identical to the eager reference" if diff == 0 else f"max |diff| {diff} (must be 0)"}

    def triton_quantize() -> dict[str, Any]:
        from flashdreams.accelerated.quantization.quantizer import Granularity, quantize

        x = torch.randn(788, 4096, device=dev, dtype=torch.bfloat16)
        q1, s1 = quantize(x, torch.float8_e4m3fn, Granularity.SLICE, use_triton=True)
        q0, s0 = quantize(x, torch.float8_e4m3fn, Granularity.SLICE, use_triton=False)
        err = _rel_err(q1.float() * s1, q0.float() * s0)
        return {"status": "PASS" if err < 1e-2 else "FAIL", "rel_err": err,
                "note": f"Triton vs torch FP8 row quantize, rel err {err:.2e}"}

    def triton_rope() -> dict[str, Any]:
        from flashdreams.core.attention.rope_kernel import apply_rotary_pos_emb

        x = torch.randn(1, 1024, 8, 128, device=dev, dtype=torch.bfloat16)
        raw = torch.rand(1024, 64, device=dev) * 6
        freqs = torch.cat((raw, raw), dim=-1).reshape(1024, 1, 1, 128)  # the shift_t layout
        half = 64
        rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        ref = x.float() * freqs.cos().view(1, 1024, 1, 128) + rot.float() * freqs.sin().view(1, 1024, 1, 128)
        got = apply_rotary_pos_emb(x, freqs, inplace=False)
        err = _rel_err(got, ref)
        return {"status": "PASS" if err < 2e-2 else "FAIL", "rel_err": err,
                "note": f"flashdreams RoPE Triton kernel (not on the Cosmos3 path), rel err {err:.2e}"}

    def scaled_mm(rowwise: bool, fast: bool) -> Callable[[], dict[str, Any]]:
        def run() -> dict[str, Any]:
            x = torch.randn(788, 4096, device=dev, dtype=torch.bfloat16)
            w = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
            if rowwise:
                sx = x.float().abs().amax(1, keepdim=True) / 448
                sw = w.float().abs().amax(1, keepdim=True) / 448
            else:
                sx = (x.float().abs().amax() / 448).reshape(1, 1)
                sw = (w.float().abs().amax() / 448).reshape(1, 1)
            x8 = (x.float() / sx).to(torch.float8_e4m3fn)
            w8 = (w.float() / sw).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(x8, w8.T, sx, sw.T.contiguous(), out_dtype=torch.bfloat16,
                                   use_fast_accum=fast)
            err = _rel_err(out, x.float() @ w.float().T)
            return {"status": "PASS" if err < 0.08 else "FAIL", "rel_err": err, "note": f"rel err {err:.2e}"}

        return run

    def fp8_linear() -> dict[str, Any]:
        from cosmos3_h_dreams._runtime._network import _Fp8Linear

        lin = torch.nn.Linear(4096, 12288, bias=False, dtype=torch.bfloat16, device=dev)
        x = torch.randn(1, 788, 4096, device=dev, dtype=torch.bfloat16)
        fp8 = _Fp8Linear(lin, dev)
        err = _rel_err(fp8(x), lin(x))
        return {"status": "PASS" if err < 0.08 else "FAIL", "rel_err": err,
                "note": f"product FP8 projection vs bf16 Linear, rel err {err:.2e}"}

    def sdpa_inputs(masked: bool) -> tuple[Any, ...]:
        s = SDPA_SHAPE
        q = torch.randn(1, s["heads"], s["q_tokens"], s["head_dim"], device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, s["kv_heads"], s["k_tokens"], s["head_dim"], device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = None
        if masked:
            mask = torch.ones(s["q_tokens"], s["k_tokens"], device=dev, dtype=torch.bool)
            mask[:, -s["q_tokens"] // 2 :] = torch.rand(s["q_tokens"], s["q_tokens"] // 2, device=dev) > 0.2
            mask = mask[None, None]
        return q, k, v, mask

    def sdpa_default(masked: bool) -> Callable[[], dict[str, Any]]:
        def run() -> dict[str, Any]:
            import torch.nn.functional as F

            q, k, v, mask = sdpa_inputs(masked)
            call = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)  # noqa: E731
            backend = sdpa_backend_of(_kernel_names(call))
            ms = _bench_ms(call)
            return {"backend": backend, "ms": round(ms, 3), "note": f"default picks {backend}, {ms:.3f} ms"}

        return run

    def sdpa_backend(name: str, masked: bool) -> Callable[[], dict[str, Any]]:
        def run() -> dict[str, Any]:
            import torch.nn.functional as F

            from cosmos3_h_dreams._runtime._sdpa_backends import sdpa_kernel_context

            q, k, v, mask = sdpa_inputs(masked)
            ref = F.scaled_dot_product_attention(
                q.float(), k.float().repeat_interleave(4, 1), v.float().repeat_interleave(4, 1), attn_mask=mask
            )

            def call() -> Any:
                with sdpa_kernel_context(torch, name):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

            try:
                got = call()
            except RuntimeError as error:
                if "No available kernel" not in str(error):
                    raise
                return {"status": "N/A", "note": "backend has no kernel for these inputs (GQA/mask); not used"}
            err = _rel_err(got, ref)
            ms = _bench_ms(call)
            ok = err < 2e-2
            return {"status": "PASS" if ok else "FAIL", "ms": round(ms, 3), "rel_err": err,
                    "note": f"{ms:.3f} ms, rel err {err:.1e}"}

        return run

    def varlen() -> dict[str, Any]:
        import torch.nn.functional as F

        from cosmos3_h_dreams._runtime._attention import (
            valid_length_cached_attention,
            varlen_attention_available,
        )

        if not varlen_attention_available():
            return {"status": "FAIL", "note": "torch.nn.attention.varlen missing; the FadeMem fill path needs it"}
        s = SDPA_SHAPE
        cur, phys, valid = s["q_tokens"], 4000, 2500
        q = torch.randn(1, s["heads"], cur, s["head_dim"], device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, s["heads"], phys + cur, s["head_dim"], device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        got = valid_length_cached_attention(q, k, v, torch.tensor(valid, device=dev), cur)
        keep = torch.cat((torch.arange(valid), torch.arange(phys, phys + cur))).to(dev)
        ref = F.scaled_dot_product_attention(q.float(), k[:, :, keep].float(), v[:, :, keep].float())
        err = _rel_err(got, ref)
        return {"status": "PASS" if err < 2e-2 else "FAIL", "rel_err": err, "note": f"rel err {err:.1e}"}

    def compile_graph() -> dict[str, Any]:
        def f(x: Any) -> Any:
            return torch.nn.functional.silu(x) * (x * x).sum(-1, keepdim=True).rsqrt()

        x = torch.randn(256, 1024, device=dev)
        compiled = torch.compile(f, mode="default")
        err = _rel_err(compiled(x), f(x))
        graph = torch.cuda.CUDAGraph()
        static = x.clone()
        compiled(static)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            out = compiled(static)
        static.copy_(x * 2)
        graph.replay()
        err2 = _rel_err(out, f(x * 2))
        ok = err < 1e-4 and err2 < 1e-4
        return {"status": "PASS" if ok else "FAIL", "note": f"Inductor + CUDA graph, rel err {max(err, err2):.1e}"}

    check("torch build has kernels for this GPU", arch)
    check("NVML index matches the CUDA device", nvml_match)
    check("bf16 matmul (cuBLAS)", bf16_matmul)
    check("Triton rotate_temporal == eager reference", rotate_temporal)
    check("Triton FP8 quantize (flashdreams)", triton_quantize)
    check("Triton RoPE (flashdreams)", triton_rope)
    for rowwise, fast, label in (
        (True, False, "row-wise (product path)"),
        (True, True, "row-wise + fast accum"),
        (False, False, "per-tensor"),
        (False, True, "per-tensor + fast accum"),
    ):
        check(f"_scaled_mm FP8 {label}", scaled_mm(rowwise, fast))
    check("_Fp8Linear (Balanced/Low memory projection)", fp8_linear)
    check("SDPA default backend, no mask", sdpa_default(False))
    check("SDPA default backend, bool mask", sdpa_default(True))
    for name in ("cudnn", "flash", "efficient", "math"):
        for masked in (False, True):
            check(f"SDPA {name}{', bool mask' if masked else ''}", sdpa_backend(name, masked))
    check("varlen attention (FadeMem fill refresh)", varlen)
    check("torch.compile + CUDA graph", compile_graph)
    return tests


def task_smoke(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    torch.cuda.init()
    from cosmos3_h_dreams.headset_server import _nvml_index

    try:
        import pynvml

        result["nvml_index"] = _nvml_index(0, nvml=pynvml)
    except ImportError:
        result["nvml_index"] = 0
    result["environment"] = collect_environment(result["nvml_index"], out_dir)
    result["smoke"] = _smoke_tests(result["nvml_index"])
    return result


# --------------------------------------------------------------------------- FP8 GEMM microbenchmark


def task_gemm(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import torch

    rows: list[dict[str, Any]] = []
    M = GEMM_M
    for name, (N, K) in GEMM_SHAPES.items():
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        x8, w8 = x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda")
        row_x, row_w = torch.ones(M, 1, device="cuda"), torch.ones(1, N, device="cuda")
        variants = {
            "bf16": lambda: x @ w.T,
            "fp8_rowwise": lambda: torch._scaled_mm(x8, w8.T, row_x, row_w, out_dtype=torch.bfloat16),
            "fp8_rowwise_fastacc": lambda: torch._scaled_mm(
                x8, w8.T, row_x, row_w, out_dtype=torch.bfloat16, use_fast_accum=True),
            "fp8_tensorwise": lambda: torch._scaled_mm(x8, w8.T, one, one, out_dtype=torch.bfloat16),
            "fp8_tensorwise_fastacc": lambda: torch._scaled_mm(
                x8, w8.T, one, one, out_dtype=torch.bfloat16, use_fast_accum=True),
        }
        row: dict[str, Any] = {"shape": name, "M": M, "N": N, "K": K}
        for key, fn in variants.items():
            try:
                ms = _bench_ms(fn, iters=100)
                row[key] = {"ms": round(ms, 4), "tflops": round(2 * M * N * K / ms / 1e9, 1)}
            except Exception as error:  # noqa: BLE001
                row[key] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
        print("[gemm]", json.dumps(row), flush=True)
        rows.append(row)
    return {"gemm": rows}


# --------------------------------------------------------------------------- model runs


def _mem(nvml: Nvml, baseline_mib: float | None) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize()
    used = nvml.used_mib()
    proc = nvml.process_mib(os.getpid())
    return {
        "allocated_GiB": round(torch.cuda.memory_allocated() / GiB, 3),
        "reserved_GiB": round(torch.cuda.memory_reserved() / GiB, 3),
        "max_allocated_GiB": round(torch.cuda.max_memory_allocated() / GiB, 3),
        "smi_process_MiB": None if proc is None else round(proc),
        "smi_device_used_MiB": None if used is None else round(used),
        "smi_device_minus_baseline_MiB": (
            None if used is None or baseline_mib is None else round(used - baseline_mib)
        ),
    }


def _apply_vram_cap(cap_gb: float | None, nvml: Nvml, baseline_mib: float | None) -> dict[str, Any]:
    """Cap PyTorch's allocator so the whole process fits in ``cap_gb`` GiB.

    The CUDA context and library handles are outside the allocator, so their
    measured size is subtracted from the cap.
    """
    import torch

    torch.cuda.init()
    torch.zeros(1, device="cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    used = nvml.used_mib()
    context_mib = None if used is None or baseline_mib is None else used - baseline_mib
    out: dict[str, Any] = {
        "requested_GiB": cap_gb, "device_total_GiB": round(total / GiB, 2),
        "context_MiB": None if context_mib is None else round(context_mib),
    }
    if not cap_gb:
        out.update(effective=False, note="no cap requested")
        return out
    allocator_bytes = cap_gb * GiB - (context_mib or 600) * 2**20
    if allocator_bytes >= total:
        out.update(effective=False, note=f"card has {total / GiB:.1f} GiB <= cap; cap has no effect")
        return out
    fraction = allocator_bytes / total
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    out.update(effective=True, fraction=round(fraction, 4), allocator_cap_GiB=round(allocator_bytes / GiB, 2))
    return out


def _make_config(spec: dict[str, Any]) -> Any:
    from cosmos3_h_dreams.config import COSMOS3_H_DREAMS_PRESETS, Cosmos3HDreamsConfig
    from cosmos3_h_dreams.training_routes import MODEL_MODES

    preset = dict(COSMOS3_H_DREAMS_PRESETS[spec["preset"]])
    preset["device"] = "cuda:0"
    preset["attention_kernel"] = spec.get("attention_kernel", "auto")
    preset.update(spec.get("overrides", {}))
    return Cosmos3HDreamsConfig(
        **preset,
        checkpoint=spec["checkpoint"],
        initial_frame=Path(spec["scene"]),
        actions=Path(spec["actions"]),
        offload_understanding=True,
        text_cache=Path(spec["text_cache"]) if spec.get("text_cache") else None,
        steady_refresh_sdpa=spec["steady_refresh_sdpa"],
        **MODEL_MODES[spec["mode"]],
    )


class _Runner:
    """One loaded session and the real per-chunk path: step, decode, uint8 frames."""

    def __init__(self, session: Any) -> None:
        from flashdreams.serving.realtime.media import rgb_array_to_uint8_frames

        self.session = session
        self.to_frames = rgb_array_to_uint8_frames
        self.acts = session._scene.replay_actions
        self.per = session.actions_per_step
        self.cursor = 0

    def next_actions(self) -> Any:
        import torch

        n = self.acts.shape[0] // self.per * self.per
        start = (self.cursor * self.per) % n
        self.cursor += 1
        return self.acts[start : start + self.per] if n else torch.cat([self.acts] * self.per)[: self.per]

    def chunk(self, profile: bool = False, check: bool = False) -> tuple[list[np.ndarray], dict[str, Any]]:
        import torch

        latents = self.session._rollout.step(self.next_actions(), profile=profile)
        video = self.session._model.decode_tensor(torch.cat(latents, dim=1))
        info: dict[str, Any] = {}
        if check:
            info["finite"] = bool(torch.isfinite(video).all())
        frames = self.to_frames(video, layout="tchw", value_range="minus_one_one")
        return frames, info


def task_case(spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Load one mode, then measure speed and memory, or generate correctness frames."""
    import torch

    deterministic = bool(spec.get("deterministic"))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    nvml = Nvml(spec["nvml_index"])
    baseline = spec.get("baseline_used_mib")
    result: dict[str, Any] = {"spec": {k: v for k, v in spec.items() if k != "nvml_index"}}
    result["vram_cap"] = _apply_vram_cap(spec.get("vram_cap_gb"), nvml, baseline)
    from cosmos3_h_dreams.headset_server import release_cuda_memory
    from cosmos3_h_dreams.session import Cosmos3HDreamsSession

    phase = "load"
    try:
        started = time.perf_counter()
        session = Cosmos3HDreamsSession(config=_make_config(spec))
        session.init()
        init_s = time.perf_counter() - started
        tc = session.text_cache or {}
        load_s = tc.get("load_s") or 0.0
        result["timing_s"] = {
            "init_total": round(init_s, 1),
            "model_load": round(load_s, 1),
            "checkpoint_hash": None if tc.get("hash_s") is None else round(tc["hash_s"], 1),
            "scene_compile_warmup": round(init_s - load_s - (tc.get("hash_s") or 0.0), 1),
        }
        result["text_cache"] = {k: tc.get(k) for k in ("status", "rehashed")}
        result["attention"] = session.attention
        result["mem_after_init"] = _mem(nvml, baseline)
        torch.cuda.reset_peak_memory_stats()
        runner = _Runner(session)
        phase = "run"
        if spec["kind"] == "correctness":
            result.update(_correctness(runner, spec, out_dir))
        elif spec["kind"] == "performance":
            result.update(_performance(runner, spec, nvml, baseline, out_dir))
        phase = "release"
        session.close()
        del runner, session
        result["allocated_after_release_GiB"] = round((release_cuda_memory() or 0) / GiB, 3)
        result["mem_after_release"] = _mem(nvml, baseline)
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="oom", phase=phase, error=str(error)[:2000])
    except Exception as error:  # noqa: BLE001 - recorded, the kit continues
        oom = "out of memory" in str(error).lower()
        result.update(status="oom" if oom else "error", phase=phase,
                      error=f"{type(error).__name__}: {str(error)[:2000]}",
                      traceback=traceback.format_exc()[-4000:])
    try:
        result["peak_at_failure_or_end"] = _mem(nvml, baseline)
    except Exception:  # noqa: BLE001 - the context may be broken
        pass
    result["host_peak_rss_GiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    return result


def _correctness(runner: _Runner, spec: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    kept, finite, stds = [], [], []
    for _ in range(spec["chunks"]):
        frames, info = runner.chunk(check=True)
        arr = np.stack(frames)
        finite.append(info["finite"])
        stds.append(float(arr.reshape(arr.shape[0], -1).std(axis=1).min()))
        kept.append(downsample(arr[list(KEEP_FRAMES)]))
    frames = np.stack(kept)
    np.save(out_dir / "frames_small.npy", frames)
    return {"chunks": spec["chunks"], "finite_per_chunk": finite, "min_frame_std_per_chunk": stds,
            "frames_small_shape": list(frames.shape)}


def _performance(
    runner: _Runner, spec: dict[str, Any], nvml: Nvml, baseline: float | None, out_dir: Path
) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for _ in range(spec["warmup_chunks"]):
        runner.chunk()
    torch.cuda.synchronize()
    out["mem_steady"] = _mem(nvml, baseline)

    wall: list[float] = []
    with PowerSampler(nvml) as sampler:
        last = time.perf_counter()
        for _ in range(spec["timed_chunks"]):
            runner.chunk()
            now = time.perf_counter()
            wall.append((now - last) * 1e3)
            last = now
    out["free_running_ms"] = wall
    out["power_during_timing"] = sampler.summary()

    rollout = runner.session._rollout
    stages: dict[str, list[float]] = defaultdict(list)
    for _ in range(spec["stage_chunks"]):
        latents = rollout.step(runner.next_actions(), profile=True)
        for key in ("model_step_s", "setup_s", "denoise_s", "cache_seed_prune_s", "output_s",
                    "cache_update_submit_s", "cache_update_wait_s"):
            stages[key].append(rollout.last_timing.get(key, float("nan")) * 1e3)
        torch.cuda.synchronize()
        a = time.perf_counter()
        video = runner.session._model.decode_tensor(torch.cat(latents, dim=1))
        runner.to_frames(video, layout="tchw", value_range="minus_one_one")
        b = time.perf_counter()
        rollout.wait_for_pending_cache_update(block=True)
        c = time.perf_counter()
        stages["decode_plus_frames"].append((b - a) * 1e3)
        stages["refresh_tail_after_decode"].append((c - b) * 1e3)
    out["stage_ms"] = {k: _median(v) for k, v in stages.items()}
    out["mem_after_timing"] = _mem(nvml, baseline)

    if spec.get("profile_chunks"):
        try:
            out["profile"] = _profile(runner, spec["profile_chunks"], out_dir)
        except Exception as error:  # noqa: BLE001
            out["profile"] = {"error": f"{type(error).__name__}: {error}"}
    return out


def _median(values: list[float]) -> float:
    vals = sorted(values)
    return round(vals[len(vals) // 2], 2) if vals else float("nan")


# --------------------------------------------------------------------------- profiler breakdown


def kernel_category(name: str) -> str:
    """Category of one CUDA kernel or memcpy, by name (works for sm_90 and sm_120 names)."""
    n = name.lower()
    if n.startswith("memcpy") or "memcpy" in n:
        return "copies (memcpy H2D/D2H/D2D)"
    if "memset" in n:
        return "copies (memcpy H2D/D2H/D2D)"
    if "_quantize" in n or "quantize_" in n or "tensor_scale" in n or "partial_max" in n:
        return "FP8 activation quantize"
    if "_rotate_kernel" in n:
        return "temporal RoPE (Triton)"
    if any(t in n for t in ("flash", "fmha", "attention", "attn", "sdpa")):
        return "attention"
    gemm = any(t in n for t in ("gemm", "nvjet", "cutlass", "xmma", "cublas", "triton_tem", "_mm_", "matmul"))
    if gemm and any(t in n for t in ("f8", "e4m3", "fp8", "qqtst", "_qq", "scaled")):
        return "GEMM FP8"
    if gemm:
        return "GEMM bf16"
    if n.startswith("triton_"):
        return "Inductor Triton (fused norm/act/cat)"
    if any(t in n for t in ("elementwise", "vectorized", "unrolled", "reduce", "copy", "cat", "index",
                              "fill", "scatter", "gather", "where", "compare", "arange", "sort", "topk")):
        return "eager elementwise/reduce/copy"
    return "other"


def categorize_trace(trace: dict[str, Any], chunks: int) -> dict[str, Any]:
    """Per-chunk kernel time and launch count by category from a chrome trace.

    Kernels on the stream and within the GPU span of a ``dc_decode`` or
    ``dc_frames`` annotation count as "decode + frame conversion".
    """
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not gpu:
        return {"error": "no GPU kernels in the trace"}
    windows = [
        (e["ts"], e["ts"] + e["dur"], e.get("tid"))
        for e in events
        if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and e.get("name") in ("dc_decode", "dc_frames")
    ]
    tot: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    names: dict[tuple[str, str], float] = defaultdict(float)
    streams: dict[Any, float] = defaultdict(float)
    for e in gpu:
        mid = e["ts"] + e["dur"] / 2
        in_decode = any(a <= mid <= b and (tid is None or tid == e.get("tid")) for a, b, tid in windows)
        cat = kernel_category(e["name"])
        if in_decode and cat != "copies (memcpy H2D/D2H/D2D)":
            cat = "decode (VAE) + frame conversion"
        tot[cat] += e["dur"]
        cnt[cat] += 1
        names[(cat, e["name"][:100])] += e["dur"]
        streams[e.get("args", {}).get("stream", e.get("tid"))] += e["dur"]
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
    busy, (cs, ce) = 0.0, iv[0]
    for s, e_ in iv[1:]:
        if s > ce:
            busy, cs, ce = busy + ce - cs, s, e_
        else:
            ce = max(ce, e_)
    busy += ce - cs
    span = max(end for _, end in iv) - iv[0][0]
    total = sum(tot.values())
    per = lambda us: round(us / 1e3 / chunks, 2)  # noqa: E731
    return {
        "chunks": chunks,
        "gpu_span_ms_per_chunk": per(span),
        "gpu_busy_ms_per_chunk": per(busy),
        "gpu_idle_ms_per_chunk": per(span - busy),
        "kernel_sum_ms_per_chunk": per(total),
        "categories": [
            {"category": c, "ms_per_chunk": per(v), "pct": round(100 * v / total, 1),
             "launches_per_chunk": round(cnt[c] / chunks)}
            for c, v in sorted(tot.items(), key=lambda x: -x[1])
        ],
        "streams_ms_per_chunk": {str(k): per(v) for k, v in sorted(streams.items(), key=lambda x: -x[1])},
        "top_kernels": [
            {"category": c, "name": n, "ms_per_chunk": per(v)}
            for (c, n), v in sorted(names.items(), key=lambda x: -x[1])[:15]
        ],
    }


def _profile(runner: _Runner, chunks: int, out_dir: Path) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    rollout = runner.session._rollout
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(chunks):
            with record_function("dc_step"):
                latents = rollout.step(runner.next_actions())
            with record_function("dc_decode"):
                video = runner.session._model.decode_tensor(torch.cat(latents, dim=1))
            with record_function("dc_frames"):
                runner.to_frames(video, layout="tchw", value_range="minus_one_one")
        rollout.wait_for_pending_cache_update(block=True)
        torch.cuda.synchronize()
    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    try:
        with open(trace_path) as f:
            return categorize_trace(json.load(f), chunks)
    finally:
        trace_path.unlink(missing_ok=True)  # tens of MB: not worth sending back


# --------------------------------------------------------------------------- entry


TASKS: dict[str, Callable[[dict[str, Any], Path], dict[str, Any]]] = {
    "smoke": task_smoke,
    "gemm": task_gemm,
    "case": task_case,
}


def child_main(spec_json: str, result_path: str) -> int:
    """Run one task and write its JSON result; never raises."""
    spec = json.loads(spec_json)
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        result = TASKS[spec["task"]](spec, out_dir)
    except Exception as error:  # noqa: BLE001
        result = {"status": "error", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()[-4000:]}
    result["task_seconds"] = round(time.perf_counter() - started, 1)
    tmp = Path(result_path).with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str))
    tmp.replace(result_path)
    sys.stdout.flush()
    gc.collect()
    # Skip interpreter teardown: compile workers and CUDA graphs can hang or
    # crash during shutdown, after the result is already safe on disk.
    os._exit(0)


