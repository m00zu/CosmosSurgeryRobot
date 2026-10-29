<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Blackwell optimisation results and H100 handoff

The goal is real time (**267 ms per chunk**: 8 frames at 30 Hz) on a **laptop RTX 5090**
in the Balanced or Low memory mode, without visible quality loss. This page has what was
measured on a desktop RTX 5090 and RTX 5080 on 2026-10-28, what to keep, and how to repeat
the important cases on an H100. The kit is described in [rtx5090-test-kit.md](rtx5090-test-kit.md).

The experiment scripts and raw results (logs, frames, contact sheets) are attached to the
GitHub release `blackwell-experiments-2026-10-28`. The scripts are kept out of the source
tree because they are monkeypatching experiment code that the repository's type check
would reject:

```bash
gh release download blackwell-experiments-2026-10-28 --repo ITRI-BDL-D/CosmosSurgeryRobot --dir data_local
tar -xzf data_local/blackwell_experiments_scripts.tar.gz -C data_local   # -> data_local/blackwell-experiments/
tar -xzf data_local/blackwell_experiments_results.tar.gz -C data_local   # -> data_local/blackwell_exp/
```

| Script | Purpose |
|---|---|
| `bwexp.py` | Runs the kit's cases with patches (SageAttention, simulated NVFP4, skipped refresh and more). The docstring lists all of them. |
| `sage_build.sh` | Builds SageAttention2 with the CUDA-stream fix for the given architectures. |
| `attn_capture.py`, `attn_bench.py` | Capture real Q/K/V, then time and score attention kernels against fp32. |
| `fp4probe.py`, `svdq_probe.py`, `rot_probe.py` | Capture MLP activations, then score NVFP4, SVDQuant and rotation per layer. |

## Summary

| Change | In the repo | 5080 saving | Quality |
|---|---|---:|---|
| FP8 MLP (Balanced), `fp8_gemm: epilogue_scales` | yes, opt-in (`9acb2a0`) | 6–9 % | at the noise floor |
| **SageAttention2++** for mask-free SDPA calls | no, experiment patch | **−37 ms (−13–15 %)** | at the noise floor |
| **Low memory** instead of Balanced (FP8 attention projections) | yes (mode) | **−34 ms** | at the noise floor |
| NVFP4 `down_proj` in the cache refresh only | no, simulated only | ~−6 ms (estimate) | at the noise floor |
| NVFP4 all MLP projections in the refresh only | no, simulated only | ~−18 ms (estimate) | about −1 dB |
| NVFP4 in the denoise passes, SVDQuant, Hadamard rotation, SageAttention3 | no | — | rejected: visible loss |
| Skip the clean refresh (reuse the last denoise step's K/V) | no | −80 ms | rejected: the video breaks |

**Best quality-safe stack: Low memory + `epilogue_scales` + SageAttention2++: 216 ms on an
RTX 5080.** Laptop estimate: 275–310 ms, so still slightly over budget. Balanced +
`epilogue_scales` + SageAttention2++ is 250 ms on the 5080 (laptop estimate 320–360 ms).

## Measurements

"Same pod" means each row was measured on one machine in one session. Timing is the kit's
median and p90 per chunk: the model step plus TAEHV decode, preset
`wan-taehv-chunk2-2steps-fademem-compile`, 24 GiB VRAM cap.

### Speed

| GPU | Mode | without SageAttention | with SageAttention2++ |
|---|---|---:|---:|
| RTX 5090 desktop (575 W) | Low memory + epi | 156.0 (p90 156.6) | **138.4** (p90 139.3) |
| RTX 5090 desktop | Balanced + epi | 170.1 (p90 170.6) | **153.3** (p90 154.2) |
| RTX 5080 desktop (360 W), GPU 0 | Low memory + epi | 253.3 (p90 254.1) | **216.4** (p90 217.5) |
| RTX 5080, GPU 0 / GPU 1 | Balanced + epi | 286.6 / 288.0 (p90 288.7 / 289.4) | **250.0 / 250.7** (p90 251.8 / 251.7) |

Earlier 5090 pod (fused_scales, no SageAttention): Standard 233.6, Balanced 181.4, Low
memory 171.3 ms.

- **5080 power and clocks:** 298 W (Balanced) and 331 W (Balanced + SageAttention), SM
  clock about 2.7 GHz.
- **Why the laptop numbers are estimates:** the laptop 5090 is the same GB203 chip with 82
  SMs vs the 5080's 84. At 150–175 W it probably sustains 1.9–2.1 GHz (from memory, not
  measured), which puts it at about 70–78 % of the 5080. `nvidia-smi -pl` fails in Vast and
  RunPod containers ("Insufficient Permissions"). A VM instance or a real laptop is needed
  to measure it.

### Where the time goes

5090, Balanced + epi + SageAttention2++, per chunk:

| Category | ms |
|---|---:|
| FP8 GEMM (MLP) | 61.0 |
| bf16 GEMM (attention q/k/v/o) | 37.7 |
| attention (SageAttention) | 20.2 |
| other: SageAttention preprocessing (V transpose/pad 6.7, int8 quantization 2.3) | 11.0 |
| Inductor-fused norm/activation | 9.4 |
| eager elementwise | 5.7 |
| TAEHV decode | 5.2 |
| FP8 activation quantization, RoPE, copies | 4.8 |

The FP8 GEMMs run at about 420 TFLOPS and the bf16 GEMMs at about 190 TFLOPS: both are at
the card's peak, so kernel tuning has little left to gain.

5080 stages, Balanced + epi + SageAttention2++:

| Stage | ms |
|---|---:|
| Denoise (2 steps) | 168.6 |
| **Cache seed/prune** (the clean refresh pass) | **79.5** |
| Cache update submit | 11.1 |
| Decode | 9.2 |

The clean refresh is a third full 36-layer forward pass over the finished frames. It
exists only to write their K/V into the FadeMem cache.

### Attention kernels

Real captured Q/K/V on the 5090: 788 queries, 5165–5559 keys, 32 heads, d = 128, bf16.
Times are per call; errors are against an fp32 reference.

| Kernel | µs | min cosine | max relative L2 |
|---|---:|---:|---:|
| FlashAttention-2 bf16 (today) | 521.7 | 0.999998 | 0.18 % |
| **SageAttention2++** (`sageattn`, sm_120 auto: int8 per-warp QK, fp8 PV, fp32+fp16 accumulation) | **314.2** | 0.99839 | 5.7 % |
| int8 per-thread QK, fp8 PV, fp32+fp16 | 338.3 | 0.99825 | 5.9 % |
| int8 per-thread QK, fp8 PV, fp32+fp32 | 366.7 | 0.99823 | 6.0 % |
| int8 QK, fp16 PV | 472.5 | 0.99849 | 5.5 % |
| SageAttention3 FP4 (`sageattn3_blackwell`) | 418.4 | **0.880** | **51 %** |

SageAttention3 is slower than SageAttention2++ at these sizes and far less accurate, so
it's rejected.

### Quality

Frame PSNR against the same-pod Balanced + epi + SageAttention2++ run, over chunks 2–12 (chunk 1
is identical for refresh-only changes). Correctness case, one scene, 12 chunks. Two
identical runs on this GPU are about 31 dB apart, because the kit's deterministic mode is not
bit-reproducible on sm_120.

| Variant | mean dB | worst chunk |
|---|---:|---:|
| *Noise reference: Balanced + epi without SageAttention* | *30.7* | *28.5* |
| Low memory instead of Balanced | 30.6 | 28.0 |
| NVFP4 `down_proj`, refresh only | 31.6 | 28.1 |
| NVFP4 all MLP projections, refresh only | 29.7 | 25.6 |
| NVFP4 `down_proj`, denoise only | 28.6 | 26.4 |
| NVFP4 `down_proj`, both passes | 28.5 | 25.8 |
| NVFP4 all MLP projections, denoise only | 25.9 | 23.2 |
| NVFP4 all MLP projections, both passes | 24.3 | 22.1 |
| Skip the clean refresh | ~11 | 9.9 |

With vs without SageAttention2++, the frames are 30.2 dB apart (Low memory) and 30.9 dB apart
(Balanced), which is at the noise floor.

### 4-bit MLP, per layer

Output SNR on captured activations and real weights, mean over 15 projections (layers 0, 9,
18, 27 and 35 × gate/up/down):

| Method | mean dB | worst dB |
|---|---:|---:|
| FP8 row-wise (today) | 32.7 | 29.4 |
| NVFP4 W4A4 | 22.3 | 18.3 |
| SmoothQuant α = 0.5 | 23.1 | 18.7 |
| SVDQuant α = 0.5, rank 32 / rank 128 | 24.0 / 25.1 | 18.9 / 19.4 |
| Hadamard rotation, block 64 to 4096 | 21.9–22.1 | 17.6 |
| rotation + smoothing + rank 32 | 24.5 | 18.4 |
| NVFP4 W4A8 (fp8 activations) | 24.8 | 21.2 |

Rotation hurts NVFP4: the 16-value block scales already isolate outliers. The gap to FP8
is 4-bit weight resolution across the whole matrix, so closing it would take
quantization-aware fine-tuning. GEMM speed on the 5090 at M = 788: NVFP4 1046 / 770
TFLOPS (gate-up / down) vs FP8 per-tensor 456 / 334, row-wise 393 / 275, bf16 about 199.

## Pitfalls found

1. **SageAttention launches on the legacy default stream.** It uses a bare `<<<grid,
   block>>>` at all 21 launch sites. CUDA graph capture misses those kernels, the replays
   read uninitialized memory, and the output turns to NaN once the FadeMem cache fills (about
   chunk 7). The cache then raises "FadeMem descriptors must be finite". `sage_build.sh`
   moves every launch to `at::cuda::getCurrentCUDAStream()`; after that, a captured replay
   is bit-identical to an eager call. This should be reported upstream.
2. **Installing SageAttention3 downgrades `nvidia-cublas`** (13.4.1.1 → 13.1.0.3). Restore
   with `uv sync --package flashdreams-cosmos3-h-dreams --locked --inexact`. Build anything
   extra with `--no-deps`.
3. **torch 2.11 is built for CUDA 13.0, so extensions need a CUDA 13 `nvcc`.** RunPod images
   often ship only 12.8; install `cuda-toolkit-13-0` from NVIDIA's apt repository.
   SageAttention3's README says Python 3.13, but its `setup.py` accepts 3.8+, and it
   builds on 3.12.
4. **The kit's correctness "PASS" is lenient.** It only rejects non-finite or blank
   output, and the skipped refresh passed at 10 dB. Compare frames against a same-GPU
   baseline instead (snippet below).
5. **Some 5080 hosts show 100 % utilization at idle,** with full clocks and about 60 W
   drawn by no visible process. Timings there are unusable. Check `nvidia-smi
   --query-gpu=utilization.gpu,power.draw,clocks.sm,pstate --format=csv` before running:
   an idle GPU reads P8 at about 180 MHz.
6. **Check the network before setup.** The checkpoint is 30.3 GB; at 11 MB/s it takes 45
   minutes. Run `curl -sL -m 20 -o /dev/null -w "%{speed_download}\n"` against the
   checkpoint URL and look for ≥ 50 MB/s. Set `UV_HTTP_TIMEOUT=600`. uv's 30 s default
   failed on large CUDA wheels.

## H100 runbook

**Questions to answer:**
- Does `epilogue_scales` help on H100? It's unmeasured there.
- Does SageAttention2++ run, speed things up and keep quality on Hopper?

SageAttention's sm_90 path uses `sageattn_qk_int8_pv_fp8_cuda_sm90` with fp32+fp32
accumulation, and the stream fix covers its two launch sites.

```bash
cd <repo root>
export UV_HTTP_TIMEOUT=600 PYTHONPATH=cosmos3_h_dreams:flashdreams
uv sync --package flashdreams-cosmos3-h-dreams --locked
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
# Checkpoint at data_local/checkpoints/c3-h-d_2-step_model_bf16.safetensors,
# sha256 3e6fce0046a6fe46c65307245df0db041c4a1828e0b3fc76708fb30e7851c97b.

# Scripts from the release (see the top of this page); SageAttention2 for Hopper, stream-fixed,
# into .venv only (needs a CUDA 13 nvcc).
data_local/blackwell-experiments/sage_build.sh "$PWD/.venv/bin/python" "9.0" /tmp

X=data_local/blackwell-experiments
# 1. Kernel check on real inputs: capture, then time FlashAttention vs SageAttention.
.venv/bin/python $X/attn_capture.py data_local/h100/qkv
.venv/bin/python $X/attn_bench.py data_local/h100/qkv

# 2. Rollouts: timing and correctness for each variant, one fresh process per case.
#    Run balanced:epi twice (two output dirs) to measure the H100 noise floor.
.venv/bin/python $X/bwexp.py data_local/h100/runs \
  balanced:base,balanced:epi,balanced:epi+sage,low_memory:epi,low_memory:epi+sage,standard:base,standard:sage
.venv/bin/python $X/bwexp.py data_local/h100/runs_repeat balanced:epi
```

`bwexp.py` sets `vram_cap_gb` to 24 by default. Use `BWEXP_VRAM_CAP_GB=0` to measure H100
without the laptop cap. Results land in `<out>/summary.json`, plus logs, frames and contact
sheets under `<out>/cases/`.

To compare frames against a baseline, use chunks 2–12, with the noise floor from the two
`balanced:epi` runs:

```python
import numpy as np
from cosmos3_h_dreams import device_check as dc
base = np.load("data_local/h100/runs/cases/correctness.balanced.epi/frames_small.npy")
for d in ["runs_repeat/cases/correctness.balanced.epi", "runs/cases/correctness.balanced.epi+sage"]:
    b = np.load(f"data_local/h100/{d}/frames_small.npy")
    per = [float(dc.psnr(base[i], b[i])) for i in range(1, len(b))]
    print(d, round(np.mean(per), 1), round(min(per), 1))
```

**Before SageAttention becomes a product feature:**
- **Dependency:** it would be a new runtime dependency, Apache-2.0, built from source. That
  means a product `pyproject.toml` entry plus `uv.lock` and `THIRD-PARTY-NOTICES` updates.
- **Stream fix:** upstream or vendored.
- **Code:** a `torch.library.custom_op` wrapper around `_attention._sdpa`, as `bwexp.py`
  does.
- **Rollout:** opt-in until measured on each GPU.
