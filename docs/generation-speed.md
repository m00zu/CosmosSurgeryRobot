<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Video generation speed

Measured 2026-10-28 with the device test kit (`python -m cosmos3_h_dreams.device_check`,
see [rtx5090-test-kit.md](rtx5090-test-kit.md)): 480×832 video, the 2-step model,
Balanced/Low memory with `fp8_gemm: epilogue_scales` (the default from this date)
unless the row says otherwise. VRAM is what `nvidia-smi` reports.

## How to read the numbers

- The model works in **chunks**. One chunk turns 8 controller samples into
  **8 new video frames**.
- **ms per chunk** is the time to generate those 8 frames (median of 30 chunks).
- **Frames per second generated** = 8000 / ms per chunk.
- **Real-time budget**: `/train` sends controller input at 30 Hz, so 8 samples
  arrive every **267 ms**. A GPU keeps up when a chunk takes less than that;
  **× real time** = 267 / ms per chunk.

## Results

| GPU | Mode | ms per chunk | Frames/s generated | × real time | VRAM (GiB) |
|---|---|---:|---:|---:|---:|
| H100 80 GB | Standard (bf16) | 82.0 | 98 | 3.3× | 15.4 |
| H100 80 GB | Balanced (FP8 MLP) | **73.4** | **109** | **3.6×** | 10.4 |
| H100 80 GB | Low memory (FP8 all) | **73.4** | **109** | **3.6×** | 9.0 |
| RTX 5090 desktop 32 GB | Standard (bf16) | 233.6 | 34 | 1.1× | 15.4 |
| RTX 5090 desktop 32 GB | Balanced (FP8 MLP) | 170.7 | 47 | 1.6× | 10.4 |
| RTX 5090 desktop 32 GB | Low memory (FP8 all) | 156.9 | 51 | 1.7× | 9.0 |
| RTX 5090 desktop 32 GB | Low memory + SageAttention2++ | 138.4 | 58 | 1.9× | — |
| RTX 5080 desktop 16 GB | Balanced | 286.6 | 28 | 0.93× | — |
| RTX 5080 desktop 16 GB | Low memory | 253.3 | 32 | 1.05× | — |
| RTX 5080 desktop 16 GB | Low memory + SageAttention2++ | 216.4 | 37 | 1.2× | — |
| RTX 5090 laptop 24 GB | Low memory + SageAttention2++ | ~275–310 (estimate) | ~26–29 | ~0.9–1.0× | ~9.0 |

- **H100**: two runs on different GPUs agreed within 1 ms (Balanced 73.4 / 72.9,
  Low memory 73.4 / 72.6). Before `epilogue_scales` the same runs measured
  82.4 (Balanced) and 84.2 ms (Low memory): about 11–13 % faster.
- **RTX 5090 desktop**: RunPod, 575 W card at its power limit throughout, release
  `device-check-rtx5090-2026-10-28`. Before `epilogue_scales`: Balanced 181.4,
  Low memory 171.3 ms. Standard does not use FP8 and is unchanged.
- **SageAttention2++** is an experiment patch, not in the product; see
  [blackwell-optimization.md](blackwell-optimization.md) for the RTX 5090/5080
  runs. The 5080 rows are from that page.
- **RTX 5090 laptop**: not measured. It is the 5080's chip (GB203, 82 vs 84 SMs)
  at lower power, estimated at 70–78 % of the 5080: just short of real time with
  the best stack.
- **H100 re-check, 2026-10-28** (GPU 1, no VRAM cap, `bwexp.py` from the
  `blackwell-experiments-2026-10-28` release): Standard 82.1, Balanced 72.7 /
  72.9 (two runs), Low memory 72.8 ms. Two identical runs give bit-identical
  frames on H100, so frame differences below are real, not noise.

### SageAttention on H100: slower

| | ms per chunk | Frame PSNR vs Balanced (chunks 2–12, mean / worst) |
|---|---:|---:|
| Balanced | 72.7 | 99 / 99 (repeat run, bit-identical) |
| Balanced + SageAttention2 (sm90 kernel) | 97.7 | 28.6 / 25.0 |
| Low memory | 72.8 | 27.1 / 23.9 |
| Standard (bf16) | 82.1 | 29.9 / 25.9 |

On real attention inputs (788 queries, about 5.5k keys) one call takes 152 µs
with PyTorch's default kernel on H100 (cuDNN) and about 430 µs with
SageAttention2, which is also less accurate (cosine 0.9977 vs 0.999998 against
fp32). On the RTX 5090 the default is FlashAttention-2 at 522 µs, which
SageAttention beats. Turn SageAttention on for Blackwell consumer cards only.
SageAttention changes frames about as much as FP8 does (28.6 vs 29.9 dB);
both pass the kit's check.

## Response time

From a mouse movement to the matching frame on screen, `/train` on an H100 at
30 Hz input measured about **0.36 s** (one chunk to fill with input, one to
generate, then playback).

## Short version for slides

- H100: **8 frames every 73 ms**, about **110 frames per second**, **3.6× faster
  than real time**.
- Desktop RTX 5090: **8 frames every 157–171 ms**, about **50 frames per
  second**, **1.6–1.7× real time**; 138 ms (1.9×) with SageAttention.
- Desktop RTX 5080: **216 ms** with Low memory + SageAttention, **1.2× real
  time**. Laptop RTX 5090: estimated **0.9–1.0×**, just short.
- SageAttention speeds up the RTX 5090/5080 by 13–15 % but **slows the H100 by
  34 %**: H100's default attention kernel is already about 3× faster.
