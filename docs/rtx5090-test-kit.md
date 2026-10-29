<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RTX 5090 / 5080 hardware test kit

This is a handoff for an agent or a person who has only this repository. It
explains how to rent a consumer Blackwell GPU on RunPod, run one command, and
send back a small tarball. That tarball tells us whether Cosmos3-H-Dreams runs
in real time on a **laptop RTX 5090**, and which optimisation to do next.

The kit is `python -m cosmos3_h_dreams.device_check`
([device_check.py](../cosmos3_h_dreams/cosmos3_h_dreams/device_check.py) and
[_device_check_gpu.py](../cosmos3_h_dreams/cosmos3_h_dreams/_device_check_gpu.py)).
A reference report from an H100 is in
[rtx5090-test-kit-h100-report.md](rtx5090-test-kit-h100-report.md).

## 1. Why

**The target device** is a laptop RTX 5090: 24 GB, consumer Blackwell (GB203
chip, compute capability sm_120), 95–175 W TGP. Nobody rents laptops, so we
use two stand-ins:

| Card | Chip | VRAM | What it stands in for |
|---|---|---|---|
| Desktop RTX 5090 | GB202, sm_120 | 32 GB | **Correctness and memory.** Same architecture and software path. The kit caps VRAM to 24 GiB. It is much faster than the laptop part (more SMs, 575 W), so its speed is an upper bound. |
| Desktop RTX 5080 | GB203, sm_120 | 16 GB | **Speed.** Same chip as the laptop 5090, fewer memory chips. With a power limit of about 175 W and 120 W (`nvidia-smi -pl`) it comes close to laptop clocks. Only Balanced and Low memory are expected to fit in 16 GB. |

**What we need to learn:**

1. Does everything run on sm_120? FA3/FA4 do not exist there. The model uses
   torch SDPA (cuDNN or flash backends), a varlen FlashAttention path, FP8
   `torch._scaled_mm`, and custom Triton kernels. The kit smoke-tests each one
   on its own, so an incompatibility shows up as one clear FAIL line instead of
   a crash deep in a run.
2. Does it keep up in real time? One chunk is 2 latent frames: 8 action rows in,
   8 video frames out. The headset and `/train` sample input at 30 Hz, so a
   chunk must finish in under **8/30 s = 267 ms**.
3. How much VRAM does each mode need? Does Standard fit in 24 GB, and what fits
   in 16 GB?
4. Is the output correct? It is compared with an H100 reference rollout.
5. Where does the GPU time go on sm_120? The kit records a kernel-category
   profile per mode and an FP8 GEMM microbenchmark.

**What the numbers will decide:**

- Which mode the laptop config should start in. It is Balanced today.
- Whether `steady_refresh_sdpa` stays on by default.
- The order of the next optimisations:
  - a fused FP8 activation quantize: about 750 launches per chunk, about 5.5 ms on H100;
  - configurable `_scaled_mm` options: fast accumulation and per-tensor scales;
  - a fused FadeMem prepare/commit: about 680 small kernels per chunk.

  The profile and the GEMM table rank these on sm_120.

## 2. What to rent (RunPod)

| | Desktop RTX 5090 pod | Desktop RTX 5080 pod |
|---|---|---|
| GPU | 1 × RTX 5090 (32 GB) | 1 × RTX 5080 (16 GB) |
| **Host driver** | **≥ 580 (CUDA 13.0)**. In RunPod's filter, allow only CUDA 13.0 or newer | same |
| Host RAM | ≥ 64 GB recommended (measured peaks stay under 28 GiB; see below) | same |
| Disk (volume at `/workspace`) | ≥ 100 GB | same |
| Image / template | Any Ubuntu 22.04 or 24.04 template with `git` and network, for example RunPod's PyTorch template. The kit does not use the image's own Python, torch or CUDA | same |

**Why driver ≥ 580.** `uv.lock` pins **torch 2.11.0** from PyPI. That wheel is
built for **CUDA 13.0**: cuDNN 9.19 (`nvidia-cudnn-cu13`), triton 3.6.0, and
kernels for `sm_75 … sm_120`. uv installs it together with its own CUDA
runtime, so only the host driver matters. CUDA 13.0 needs driver 580 or newer.
The product package has no CUDA 12.8 option in the lock: `--group cuda12` is
rejected for `flashdreams-cosmos3-h-dreams`. If only CUDA 12.8 hosts (driver
570–579) are available, see Troubleshooting.

**Host RAM.** Measured on the H100 host:

- Each model run, with the text cache, peaks at **13.8 GiB RSS**.
- Building the text cache loads the understanding half of the model on the CPU
  and peaks at **27.8 GiB RSS**. The kit does this once, in its own step, when
  the cache file is missing.
- The 30 GB checkpoint is memory-mapped. The page cache uses free RAM on top of
  this, but it can be reclaimed.

The report's speed table has a `Host RSS GiB` column.

**Disk.**

| Item | Size |
|---|---|
| Checkpoint | 30.3 GB |
| Python environment | about 5 GB |
| uv download cache | about 5 GB |
| Decoder, VAE and tokenizer downloads (first run) | about 3 GB |
| Inductor / Triton compile caches | under 1 GB |
| Results | a few MB |

## 3. Set up from scratch

All commands run in a shell on the pod. Put everything on the volume.

```bash
apt-get update && apt-get install -y git build-essential   # Triton compiles a small C helper with gcc
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv --version                                               # must be >= 0.11.8
# Keep the big caches on the volume; RunPod's container disk is small. Repeat these in every new shell.
export UV_CACHE_DIR=/workspace/.cache/uv HF_HOME=/workspace/.cache/huggingface

cd /workspace
git clone -b staged-vram-loading https://github.com/ITRI-BDL-D/CosmosSurgeryRobot
cd CosmosSurgeryRobot
ls cosmos3_h_dreams/cosmos3_h_dreams/device_check.py       # missing? the branch lacks the kit: stop and ask the user
uv sync --package flashdreams-cosmos3-h-dreams --locked    # exactly as README.md; about 5 GB
```

`uv sync` installs a uv-managed Python 3.12 when the image has none.

**Get the checkpoint.** It is not in git. Use the public, ungated Hugging Face
repository [nvidia/Cosmos3-H-Dreams](https://huggingface.co/nvidia/Cosmos3-H-Dreams),
file `checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors` (30.3 GB). No
token or credentials are needed. Its SHA-256 matches the file the H100 numbers
were measured with.

```bash
uv run --no-sync python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('nvidia/Cosmos3-H-Dreams',
      'checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors', local_dir='data_local/hf'))"
mkdir -p data_local/checkpoints
mv data_local/hf/checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors data_local/checkpoints/
sha256sum data_local/checkpoints/c3-h-d_2-step_model_bf16.safetensors
# expect 3e6fce0046a6fe46c65307245df0db041c4a1828e0b3fc76708fb30e7851c97b
```

- If Hugging Face is slow or blocked, ask the user to upload the file to the pod.
- Use the **2-step** checkpoint, not the 4-step one: the preset is a 2-step preset.
- The first model load also downloads public files from Hugging Face: the Wan2.2
  VAE, the TAEHV decoder (`lightx2v/Autoencoders`) and the Qwen3-VL tokenizer.

**Text K/V cache.** The kit uses `data_local/text_kv/c3-h-d_2-step_480x832.pt`,
the same path as the training-ground configs. When that file is missing, the kit
builds it once (task `text_cache_build`, about 1 minute plus downloads; needs the
extra host RAM above). If the user gives you a prebuilt file (6.4 MB, built
from the same checkpoint), put it at that path and the build is skipped. The
file is tied to the checkpoint's bytes, so a copied cache only works with the
checkpoint above.

**SageAttention (optional).** To measure with it, run
`scripts/install_sageattention.sh` before the kit (needs a CUDA 13 `nvcc`; RunPod
images often ship only 12.8, so install `cuda-toolkit-13-0` from NVIDIA's apt
repository first). The kit's cases then use it automatically on an RTX 50 GPU,
and the report prints `Attention kernel: sage (...)` under the speed table.
Without it the report says why SDPA was used. For both numbers on one machine,
run the kit twice: once as usual, once with `--attention-kernel default`.

## 4. Run it

Every command runs from the repository root. A quick pass goes first: 4.8
minutes on an H100 with warm compile caches. On a fresh pod, add the first-time
compile of each mode: 1–2 minutes per mode on an H100, longer on smaller GPUs.

```bash
export KIT="env PYTHONPATH=cosmos3_h_dreams:flashdreams uv run --no-sync python -m cosmos3_h_dreams.device_check"
export CKPT=data_local/checkpoints/c3-h-d_2-step_model_bf16.safetensors
nvidia-smi; nvidia-smi -q -d POWER | head -30        # note the default/min/max power limits
```

### Desktop RTX 5090 (correctness and memory; VRAM capped to 24 GiB)

```bash
$KIT --checkpoint $CKPT --out data_local/device_check/rtx5090_quick --quick --label "RunPod 5090 quick"
# read data_local/device_check/rtx5090_quick/report.md; if the smoke tests or a mode failed, see Troubleshooting
$KIT --checkpoint $CKPT --out data_local/device_check/rtx5090 --label "RunPod 5090 full, 24 GiB cap"
```

The full run took 8 minutes on an H100 with warm caches; expect 15–30 minutes
on a fresh 5090 pod. `--vram-cap-gb 24` is the default.

### Desktop RTX 5080 (speed proxy for the laptop 5090)

```bash
$KIT --checkpoint $CKPT --out data_local/device_check/rtx5080_quick --quick --label "RunPod 5080 quick"
$KIT --checkpoint $CKPT --out data_local/device_check/rtx5080_default --label "RunPod 5080 default power"
# Laptop-like power. The kit tries nvidia-smi -pl, records whether it worked, and restores the limit at the end.
$KIT --checkpoint $CKPT --out data_local/device_check/rtx5080_175w --power-limit 175 --skip-correctness --label "RunPod 5080 @175 W"
$KIT --checkpoint $CKPT --out data_local/device_check/rtx5080_120w --power-limit 120 --skip-correctness --label "RunPod 5080 @120 W"
```

- If 120 W is below the card's `power.min_limit` (shown in the report and by
  `nvidia-smi -q -d POWER`), use the minimum instead.
- If the report says `nvidia-smi -pl permitted: False` (usual in RunPod
  containers), skip the remaining power-limited runs; the default-power run is
  then the speed result. Do not try to work around this.
- Standard needed about 15.8 GB (nvidia-smi) on an H100, so on 16 GB it will
  likely OOM. The kit records that as `oom` for that mode and goes on.

### Options

| Option | Meaning |
|---|---|
| `--modes standard,balanced,low_memory` | Subset of modes |
| `--vram-cap-gb N` | Cap the whole process at N GiB. The CUDA context is subtracted from the PyTorch allocator cap. `0` = no cap. Default 24. On a card with ≤ N GiB the cap does nothing, and the report says so |
| `--power-limit W` | Try `nvidia-smi -pl W` for the run |
| `--quick` | 5 warm-up + 15 timed chunks per mode, 6-chunk correctness; no profiler, no GEMM microbenchmark, `steady_refresh_sdpa` off for Balanced only |
| `--skip-correctness` | Speed and memory only |
| `--text-cache PATH` | Default `data_local/text_kv/c3-h-d_2-step_480x832.pt`; built when missing |
| `--timeout-min M` | Per child task (default 45) |
| `--label TEXT` | Shown in the report |

### What it does, in order

Each step runs in its own Python process. A crash, an OOM or a hang (it
times out) is recorded for that step and the kit goes on.

1. **smoke**: the environment and the smoke tests:
   - environment: GPU, sm, VRAM, driver, CUDA, torch, triton and cuDNN versions, the power limits, clocks, and `nvidia-smi -q`;
   - the torch arch list contains this GPU;
   - bf16 cuBLAS;
   - the Triton `rotate_temporal` kernel is bit-identical to its eager reference;
   - the flashdreams Triton FP8 quantizer and RoPE kernels;
   - `_scaled_mm` FP8: row-wise (the product path), + fast accumulation, per-tensor, per-tensor + fast accumulation;
   - the product's `_Fp8Linear`;
   - which SDPA backend the default picks, without and with a boolean mask;
   - each SDPA backend at the real shape (32 query heads, 8 KV heads, 788 × 5559 tokens, head dim 128);
   - varlen attention;
   - `torch.compile` with a CUDA graph.
2. **Power limit**: checks whether `nvidia-smi -pl` is permitted (it writes back the current value), and applies `--power-limit`.
3. **gemm** (full run only): `_scaled_mm` at M = 788 for gate/up 12288×4096, down 4096×12288 and q/o 4096×4096: bf16, FP8 row-wise, + fast accumulation, per-tensor, per-tensor + fast accumulation.
4. **text_cache_build**, only when the cache file is missing.
5. Per mode:
   - **performance.MODE**:
     - load time, compile + warm-up time;
     - free-running ms per chunk: median, p90 and more. A chunk is `rollout.step`, `model.decode_tensor` and the uint8 frame conversion, as the server runs them;
     - the real-time verdict and the margin to 267 ms;
     - synced stage medians (`rollout.step(profile=True)`);
     - power, clocks and clock-limit reasons while timing;
     - allocated, reserved and max-allocated memory, and nvidia-smi per-process (device used − baseline when the PID is hidden by a container);
     - a 3-chunk torch.profiler kernel-category breakdown (full run only);
     - finally, the server's mode-switch release path (`session.close()` + `release_cuda_memory()`) and the memory left after it.
   - **performance.MODE.sdpa_off**: the same with `steady_refresh_sdpa: false`. Balanced always; every mode in a full run.
   - **correctness.MODE**: a deterministic rollout (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, seed 0, the bundled `example_frame.png` and `trajectory.npy`) of 12 chunks (6 with `--quick`). It is compared with the committed H100 reference ([§ 6](#6-reading-the-report)). It also writes `contact_sheet.png`: this GPU on top, the H100 below. A full run adds `correctness.balanced.sdpa_off`.
6. Writes `OUT/report.md` and `OUT/results.json` (rewritten after every step, so a partial run still has them), `OUT/logs/*.log` and `OUT/env/nvidia-smi-q.txt`, then packs **`OUT.tar.gz`** and prints its path. It is usually 2–5 MB; files over 5 MB are left out.

## 5. What to send back

Send each run's tarball, which the kit prints at the end:

```
[device-check] send back: /workspace/CosmosSurgeryRobot/data_local/device_check/rtx5090.tar.gz (2.3 MB)
```

Send `rtx5090_quick`, `rtx5090`, `rtx5080_quick`, `rtx5080_default`, and the
power-limited runs if `-pl` worked. In your message, also paste the
**Verdict** section of each `report.md` and anything unusual you saw: OOMs,
FAIL lines, the time the first compile took, and the pod's host RAM and driver.
Stop the pods afterwards.

## 6. Reading the report

**Real-time verdict.** Budget is 266.7 ms per chunk (8 frames at 30 Hz input).

| Verdict | Condition |
|---|---|
| `keeping up` | p90 ≤ 200 ms (75 % of the budget; leaves headroom for the stream and host) |
| `tight` | median ≤ 267 ms but p90 > 200 ms |
| `falling behind` | median > 267 ms; input piles up and latency grows |

The measured action-to-frame response on H100 (Balanced, `/train`, playback at
30 fps) is about 0.36 s: up to ~233 ms waiting for the chunk's 8 actions, plus
~72 ms generation, plus hand-offs. On another GPU, add the difference in ms per
chunk. The live server reports the same figures: `/healthz` → `latency`
(`chunk_ms`, `frame_ms`, `response_ms`, `budget_ms`, `playback_ms`, `status`,
`reason`), shown on `/train` under the GPU graph. If you also run the server on
the rented card, note those values too.

**Correctness.** Runs on different hardware are never bit-identical, and an
autoregressive rollout drifts, so later chunks score lower.

- **Fail only for gross problems:**
  - NaN or Inf in the decoded video;
  - a constant frame (a frame's standard deviation below 2);
  - **chunk 0 below 20 dB PSNR** against the H100.
- **Comparison method:** frames 0 and 4 of each chunk, at 1/4 resolution
  (120×208, a 4×4 box average).
- **For scale:** on the H100, a non-deterministic run against the
  deterministic reference gives these PSNRs:
  - Standard: 52 dB at chunk 0, about 37 dB by chunk 12;
  - Balanced: 34 dB, about 28 dB;
  - Low memory: 34 dB, about 28 dB.

  A healthy sm_120 run should land in the same range. Values in the 20s are
  plausible for FP8 modes.
- **Contact sheets.** Look at `cases/correctness.*/contact_sheet.png`. It shows
  frame 0 of each chunk: this GPU on top, the H100 below. Instruments should
  move the same way. Noise, a grey or black frame, colour garbage or a frozen
  image means a broken kernel.
- **Steady refresh off.** `correctness.balanced.sdpa_off` is compared with the
  Balanced reference, which was made with `steady_refresh_sdpa` on. On H100 the
  two refresh paths differ by about 38.5 dB, so expect slightly lower numbers.

**Other sections:**

| Section | What to look at |
|---|---|
| Smoke tests | Any FAIL is the most important finding. N/A means the backend has no kernel for those inputs (flash with a mask, efficient with GQA), which is also true on H100 and does not matter |
| Speed and memory | The ms/chunk, `vs H100`, nvidia-smi and H100 MiB columns |
| Synced stages | `Denoise` is the two CUDA-graph forwards. `Refresh submit` / `Refresh tail` is the FadeMem refresh. `Setup` is the FadeMem prepare |
| Kernel time per chunk | Rank the planned work. `FP8 activation quantize` (time and launches) is the fused-quantize opportunity; `eager elementwise/reduce/copy` launches are the FadeMem prepare/commit fusion; `GEMM FP8` vs `GEMM bf16` and the GEMM table show whether fast accumulation or per-tensor scales pay off on sm_120 |
| Power and clocks | `sw_power_cap` in most samples means the card is power-limited, as the laptop will be |

## 7. Reference numbers (H100)

H100 80GB setup:

- preset `wan-taehv-chunk2-2steps-fademem-compile`, 480×832;
- `offload_understanding`, text-cache hit;
- full chunk path, steady state;
- nvidia-smi per-process VRAM.

| Configuration | ms/chunk | VRAM (nvidia-smi) |
|---|---:|---:|
| Original `main`: headset-server loop, frame conversion excluded; understanding model on GPU | 128 | ≈ 32 GB |
| Before the perf work (`abb35d5`), Standard | 147.8 | 17.8 GB |
| **Now, Standard** (bf16) | **82** | **15.8 GB** (15776 MiB) |
| **Now, Balanced** (FP8 MLP, bf16 attention; laptop default) | **82** | **10.6 GB** (10600 MiB) |
| **Now, Low memory** (FP8 everywhere) | **84** | **9.2 GB** (9184 MiB) |

Notes:

- The GPU is idle about 4 ms per chunk, so the run is GPU-bound.
- `steady_refresh_sdpa` (default on) is about 5 ms per chunk faster than the
  varlen refresh. It is not bit-identical to it: about 38.5 dB after an
  autoregressive rollout, while bf16 nondeterminism alone drifts to 31.7 dB.
- **Precision** (LPIPS against deterministic bf16, lower is better):

  | Comparison | LPIPS |
  |---|---:|
  | bf16 nondeterminism floor | 0.056 |
  | Balanced | 0.129 |
  | Low memory | 0.143 |

- **FP8 GEMM at M = 788 on H100** (TFLOP/s): see the table in
  [device_check.py](../cosmos3_h_dreams/cosmos3_h_dreams/device_check.py)
  (`H100_GEMM_TFLOPS`), which the report prints in brackets.
  - Row-wise FP8 is no faster than bf16.
  - Fast accumulation adds about 30 % on the big shapes.
  - Per-tensor scales reach about 1.6× bf16.

The kit's own H100 run, with this branch and the same command as above, is in
[rtx5090-test-kit-h100-report.md](rtx5090-test-kit-h100-report.md).

## 8. Troubleshooting (likely sm_120 problems)

**`torch build has kernels for this GPU` FAILs, or `no kernel image is available`.**
The installed torch lacks sm_120. Check `uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"`. It must be 2.11.0 with `sm_120`. Re-run `uv sync --package flashdreams-cosmos3-h-dreams --locked`.

**`CUDA driver version is insufficient` or `torch.cuda.is_available()` is False.**
The host driver is older than 580. Pick another pod: filter for CUDA 13.0 or newer.

- **Last resort (untested):** reinstall a CUDA 12.8 build into the environment:
  `uv pip install --reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`.
  Then run only with `uv run --no-sync`, because a sync would restore the locked wheel.
- Mention in the results that you did this: cuDNN and cuBLAS versions then differ.

**Triton errors** (`ptxas`, `unsupported target`, a `cuda_utils` compile failure, or `gcc` not found).

- Install `build-essential`.
- Triton 3.6.0 supports sm_120. A Triton FAIL in the smoke tests with an otherwise healthy torch is a real finding: report it.
- Balanced and Low memory need the FP8 quantizer kernel; every mode needs `rotate_temporal`.

**`_scaled_mm` FP8 FAILs** (for example `only supported on sm89/sm90`, or a row-wise scaling error).

- Balanced and Low memory cannot run; Standard still can. That is a key finding: report the exact error from `results.json` (`smoke[].error`).
- If only the row-wise variant fails and per-tensor passes, say so explicitly. It decides the next FP8 work.

**SDPA.**

- `flash` with a mask and `efficient` with GQA show N/A on every GPU.
- If the default backend is `math` or `efficient` on sm_120, cuDNN is not
  available there. Speed will suffer; record it.
- A varlen attention FAIL breaks the FadeMem fill path. Note that
  `steady_refresh_sdpa` only replaces the steady-state refresh.

**OOM.**

- Expected for Standard on the 16 GB 5080. The case shows `oom` with the
  phase (load / run) and the kit goes on.
- An OOM for Balanced under the 24 GiB cap on the 5090 is a real finding.
- An OOM while *building the text cache* means GPU memory, not host RAM. Host
  RAM exhaustion instead kills the child, shown as `crashed` with exit code -9:
  rent a pod with more RAM, or ask the user for a prebuilt cache file.

**Power limit.**

- `Insufficient Permissions` or `not supported` is normal in containers. The
  kit records it and runs at the default limit. Do not retry with `sudo` tricks.
- The 175 W and 120 W runs are optional.

**Long first compile.** Each mode compiles on first use; later runs reuse
`~/.cache/flashdreams/torchinductor`. On a slow host this can take several
minutes per mode. Raise `--timeout-min` if a case shows `timeout` during
`load`.

**A case `crashed`.** Read `OUT/logs/<case>.log`: it holds the child's full
output, including any segfault or `illegal memory access`. Send the tarball
anyway.

## 9. Current state of the code (branch contents)

**Branches.** `staged-vram-loading` carries the perf work (the `perf-overlap`
merge, `9475f50`) and this kit: the overlapped FadeMem refresh, GPU-side frame
conversion, expandable segments, the shared graph pool, scene-switch memory
release, the Balanced mode, `steady_refresh_sdpa` and the text K/V cache.

**Model modes** (`/train` → Settings → Mode, `POST /mode`; YAML `runtime`):

| Mode | YAML | What it does |
|---|---|---|
| `standard` | `fp8_generation: false` | bf16 everywhere |
| `balanced` | `fp8_generation: true`, `fp8_scope: mlp` | FP8 E4M3 MLP gate/up/down, bf16 attention |
| `low_memory` | `fp8_generation: true` | FP8 MLP and attention q/k/v/o |

FP8 uses per-channel weight scales, per-token activation scales and FP32
accumulation, through the row-wise `_scaled_mm`.

**`steady_refresh_sdpa`** (default `true`; CLI `--[no-]steady-refresh-sdpa`):

- Once FadeMem memory is full, it refreshes the memory with a second compiled
  CUDA graph that uses plain SDPA (cuDNN on Hopper) instead of varlen
  FlashAttention.
- About 5 ms per chunk faster on H100, and not bit-identical.
- `false` skips that graph: less compile time and memory.

**Laptop config**:
[cosmos3_h_dreams/configs/laptop_rtx5090.yaml](../cosmos3_h_dreams/configs/laptop_rtx5090.yaml).

- Starts in Balanced, with `steady_refresh_sdpa: true`, `offload_understanding`
  and the text cache.
- Run it with `uv run --no-sync cosmos3-h-dreams-run headset --session-config
  cosmos3_h_dreams/configs/laptop_rtx5090.yaml` and open `/train`.

**Latency readout.** `/train` shows generation time per chunk and per frame,
the action-to-frame response, and a status: keeping up / near the limit /
falling behind (generation or playback). The training-ground and laptop configs
play at 30 fps so playback keeps pace with the 30 Hz input; playback catches up
(shorter frame intervals, then skipped frames) when a chunk is waiting. The
Quest `headset-base` preset still plays at 20 fps and reports "falling behind
(playback)".

**Planned next work**, to be ranked by the results:

1. Fuse the FP8 activation quantize into one kernel per projection, or into the
   preceding norm/activation.
2. Make `_scaled_mm` configurable: fast accumulation and per-tensor activation
   scales, with a quality check against bf16.
3. Fuse and graph-capture the FadeMem prepare/commit: about 680 small eager
   kernels per chunk.

## 10. Maintainers: regenerating the reference

The committed reference is `assets/device_check/h100_reference.npz` (3.8 MB):

- 12 chunks per mode, frames 0 and 4 of each chunk, at 1/4 resolution, uint8;
- the generation metadata, including the H100 run-to-run PSNR curves.

Regenerate it on the reference GPU only, and only when the model output is
meant to change:

```bash
env PYTHONPATH=cosmos3_h_dreams:flashdreams uv run --no-sync python -m cosmos3_h_dreams.device_check \
  --checkpoint $CKPT --write-reference assets/device_check/h100_reference.npz
```
