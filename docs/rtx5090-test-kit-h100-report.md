<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3-H-Dreams device check: NVIDIA H100 80GB HBM3

2026-10-28T01:26:41+00:00 · kit `9475f50 + this kit` · full run · H100 80GB reference run (GPU 1 of a DGX H100, 24 GiB cap active)

Preset `wan-taehv-chunk2-2steps-fademem-compile`, 480×832, `offload_understanding`, text cache. One chunk = 8 action rows in, 8 frames out. Real-time budget at 30 Hz input: **267 ms per chunk** (keeping up: p90 ≤ 200 ms; tight: median ≤ 267 ms; otherwise falling behind). H100 columns are the reference numbers in docs/rtx5090-test-kit.md.

## Verdict

- Smoke tests: 20 passed, 0 failed, 3 N/A (no kernel for those inputs on any GPU), 0 warnings
- performance.standard: 82.0 ms/chunk (p90 84.9) -> keeping up, 1.00x H100; nvidia-smi 15818 MiB
- performance.standard.sdpa_off: 87.5 ms/chunk (p90 88.8) -> keeping up, 1.01x H100; nvidia-smi 15734 MiB
- correctness.standard: PASS, chunk 0 99.0 dB
- performance.balanced: 82.3 ms/chunk (p90 83.1) -> keeping up, 1.00x H100; nvidia-smi 10640 MiB
- performance.balanced.sdpa_off: 88.6 ms/chunk (p90 89.5) -> keeping up, 1.02x H100; nvidia-smi 10576 MiB
- correctness.balanced: PASS, chunk 0 99.0 dB
- correctness.balanced.sdpa_off: PASS, chunk 0 99.0 dB
- performance.low_memory: 84.3 ms/chunk (p90 84.8) -> keeping up, 1.00x H100; nvidia-smi 9224 MiB
- performance.low_memory.sdpa_off: 90.7 ms/chunk (p90 91.7) -> keeping up, 1.02x H100; nvidia-smi 9142 MiB
- correctness.low_memory: PASS, chunk 0 99.0 dB

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (sm 9.0, 132 SMs, 79.18 GiB) |
| Driver / CUDA (torch) | 580.159.03 / 13.0 |
| torch / triton / cuDNN | 2.11.0+cu130 / 3.6.0 / 91900 |
| torch arch list | sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| Power limit (current / default / min / max) | 700.00 W / 700.00 W / 200.00 W / 700.00 W |
| Max clocks SM / memory | 1980 MHz / 2619 MHz |
| PCIe max | gen 5 x16 |
| Host | Intel(R) Xeon(R) Platinum 8480C (224 threads), 2015.5 GiB RAM, container: False |
| Python / platform | 3.12.13 / Linux-6.8.0-124-generic-x86_64-with-glibc2.39 |
| Checkpoint | c3-h-d_2-step_model_bf16.safetensors, 30.35 GB, sha256 3e6fce00…c97b |
| VRAM cap | 24.0 GiB requested |

## Power limit

`nvidia-smi -pl` permitted: **False** (Failed to set power management limit for GPU 00000000:52:00.0: Insufficient Permissions). Requested: none (card default); applied: False; limit during the run: 700.00 W; restore: n/a.

## Smoke tests

| Test | Status | Detail |
|---|---|---|
| torch build has kernels for this GPU | PASS | device sm_90; torch built for ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120'] |
| NVML index matches the CUDA device | PASS | CUDA device 0 is NVML device 1 (kit used 1) |
| bf16 matmul (cuBLAS) | PASS | rel err 1.66e-03 |
| Triton rotate_temporal == eager reference | PASS | bit-identical to the eager reference |
| Triton FP8 quantize (flashdreams) | PASS | Triton vs torch FP8 row quantize, rel err 0.00e+00 |
| Triton RoPE (flashdreams) | PASS | flashdreams RoPE Triton kernel (not on the Cosmos3 path), rel err 2.44e-03 |
| _scaled_mm FP8 row-wise (product path) | PASS | rel err 3.75e-02 |
| _scaled_mm FP8 row-wise + fast accum | PASS | rel err 3.75e-02 |
| _scaled_mm FP8 per-tensor | PASS | rel err 3.75e-02 |
| _scaled_mm FP8 per-tensor + fast accum | PASS | rel err 3.75e-02 |
| _Fp8Linear (Balanced/Low memory projection) | PASS | product FP8 projection vs bf16 Linear, rel err 3.68e-02 |
| SDPA default backend, no mask | PASS | default picks cudnn, 0.154 ms |
| SDPA default backend, bool mask | PASS | default picks cudnn, 0.459 ms |
| SDPA cudnn | PASS | 0.155 ms, rel err 2.3e-03 |
| SDPA cudnn, bool mask | PASS | 0.457 ms, rel err 2.3e-03 |
| SDPA flash | PASS | 0.262 ms, rel err 2.3e-03 |
| SDPA flash, bool mask | N/A | backend has no kernel for these inputs (GQA/mask); not used |
| SDPA efficient | N/A | backend has no kernel for these inputs (GQA/mask); not used |
| SDPA efficient, bool mask | N/A | backend has no kernel for these inputs (GQA/mask); not used |
| SDPA math | PASS | 3.672 ms, rel err 1.7e-03 |
| SDPA math, bool mask | PASS | 4.284 ms, rel err 1.7e-03 |
| varlen attention (FadeMem fill refresh) | PASS | rel err 2.3e-03 |
| torch.compile + CUDA graph | PASS | Inductor + CUDA graph, rel err 6.1e-08 |

## Speed and memory

| Case | Status | ms/chunk median | p90 | H100 ms | vs H100 | Real time | Margin ms | Alloc GiB | Reserved GiB | Max alloc GiB | nvidia-smi MiB | H100 MiB | Load s | Compile+warmup s | Host RSS GiB |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| performance.standard | ok | 82.0 | 84.9 | 82 | 1.00 | keeping up | 184.7 | 14.11 | 14.70 | 14.33 | 15818 | 15776 | 6.0 | 20.7 | 13.8 |
| performance.standard.sdpa_off | ok | 87.5 | 88.8 | 87 | 1.01 | keeping up | 179.2 | 14.00 | 14.62 | 14.22 | 15734 | 15776 | 6.0 | 16.7 | 13.8 |
| performance.balanced | ok | 82.3 | 83.1 | 82 | 1.00 | keeping up | 184.3 | 9.05 | 9.64 | 9.27 | 10640 | 10600 | 6.2 | 26.8 | 13.8 |
| performance.balanced.sdpa_off | ok | 88.6 | 89.5 | 87 | 1.02 | keeping up | 178.1 | 8.94 | 9.58 | 9.16 | 10576 | 10600 | 6.4 | 20.6 | 13.8 |
| performance.low_memory | ok | 84.3 | 84.8 | 84 | 1.00 | keeping up | 182.4 | 7.65 | 8.26 | 7.87 | 9224 | 9184 | 6.2 | 33.7 | 13.8 |
| performance.low_memory.sdpa_off | ok | 90.7 | 91.7 | 89 | 1.02 | keeping up | 176.0 | 7.53 | 8.18 | 7.75 | 9142 | 9184 | 6.5 | 26.9 | 13.8 |

VRAM cap: allocator capped at 23.39 GiB (+622 MiB CUDA context).

Synced stage medians (ms; `profile=True` adds syncs, for attribution only):

| Case | Model step | Setup | Denoise | Cache seed/prune | Output | Refresh submit | Refresh wait | Decode + frames | Refresh tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| performance.standard | 79.8 | 1.5 | 50.4 | 27.9 | 0.0 | 5.5 | 0.0 | 4.8 | 0.0 |
| performance.standard.sdpa_off | 85.3 | 1.5 | 49.4 | 34.3 | 0.0 | 5.1 | 0.0 | 4.8 | 0.0 |
| performance.balanced | 81.1 | 1.5 | 51.7 | 27.9 | 0.0 | 5.6 | 0.0 | 4.6 | 0.0 |
| performance.balanced.sdpa_off | 87.4 | 1.5 | 51.7 | 34.1 | 0.0 | 5.5 | 0.0 | 4.6 | 0.0 |
| performance.low_memory | 83.2 | 1.5 | 53.1 | 28.6 | 0.0 | 5.7 | 0.0 | 4.5 | 0.0 |
| performance.low_memory.sdpa_off | 89.6 | 1.5 | 53.2 | 34.9 | 0.0 | 5.6 | 0.0 | 4.6 | 0.0 |

Power and clocks while timing (NVML, 100 ms samples):

| Case | Power W median / max | SM MHz median | Mem MHz | Temp °C max | Clock-limit reasons (% of samples) |
|---|---|---:|---:|---:|---|
| performance.standard | 691.2 / 700.9 | 1755 | 2619 | 51 | {'sw_power_cap': 100} |
| performance.standard.sdpa_off | 692.6 / 703.3 | 1800 | 2619 | 52 | {'sw_power_cap': 100} |
| performance.balanced | 688.9 / 692.5 | 1935 | 2619 | 51 | {'sw_power_cap': 100} |
| performance.balanced.sdpa_off | 674.0 / 680.7 | 1965 | 2619 | 51 | {'sw_power_cap': 85} |
| performance.low_memory | 673.4 / 677.2 | 1980 | 2619 | 50 | {'sw_power_cap': 19} |
| performance.low_memory.sdpa_off | 648.1 / 658.2 | 1980 | 2619 | 50 | {} |

## Correctness against the H100 reference

Deterministic rollout, example scene, `trajectory.npy`, seed 0; frames [0, 4] of each chunk at 1/4 resolution. Reference: NVIDIA H100 80GB HBM3, torch 2.11.0+cu130. Cross-hardware output is never bit-identical and drifts as the rollout goes on; only gross failures fail (NaN/Inf, constant frames, chunk 0 < 20 dB).

For scale, H100 run-to-run (non-deterministic vs deterministic) PSNR per chunk: standard: [52.04, 47.47, 45.54, 42.9, 41.31, 39.91, 37.71, 36.09, 38.35, 38.49, 38.12, 37.36]; balanced: [33.62, 39.94, 34.87, 35.19, 32.62, 29.18, 28.11, 27.32, 29.15, 29.06, 27.88, 27.71]; low_memory: [33.94, 30.5, 29.72, 30.0, 29.27, 26.73, 26.09, 25.58, 27.12, 26.87, 27.9, 28.11]

| Case | Status | Verdict | PSNR per chunk (dB) | Problems | Contact sheet |
|---|---|---|---|---|---|
| correctness.standard | ok | PASS | [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0] | – | cases/correctness.standard/contact_sheet.png |
| correctness.balanced | ok | PASS | [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0] | – | cases/correctness.balanced/contact_sheet.png |
| correctness.balanced.sdpa_off | ok | PASS | [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 42.51, 35.84, 34.87, 33.89, 33.59] | – | cases/correctness.balanced.sdpa_off/contact_sheet.png |
| correctness.low_memory | ok | PASS | [99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0] | – | cases/correctness.low_memory/contact_sheet.png |

## Kernel time per chunk (torch.profiler)

Kernels inside CUDA graphs included. The refresh runs on a side stream and overlaps decode, so the kernel sum can exceed the busy time, and decode kernels slow down while they share the GPU with it. H100 Standard before the perf work (REPORT.md 2.2, ms): {'GEMM': 44.6, 'attention (denoise + refresh)': 22.8, 'FP8 activation quantize (Low memory)': 5.5, 'eager elementwise (FadeMem prepare/commit)': 11.7, 'decode (VAE)': 6.6}.

**performance.standard**: GPU busy 78.2 ms, idle 3.71 ms, kernel sum 89.73 ms per chunk.

| Category | ms/chunk | % | Launches/chunk |
|---|---:|---:|---:|
| GEMM bf16 | 45.45 | 50.7 | 767 |
| attention | 21.1 | 23.5 | 358 |
| decode (VAE) + frame conversion | 11.76 | 13.1 | 446 |
| Inductor Triton (fused norm/act/cat) | 6.13 | 6.8 | 656 |
| eager elementwise/reduce/copy | 3.0 | 3.3 | 591 |
| temporal RoPE (Triton) | 1.26 | 1.4 | 107 |
| copies (memcpy H2D/D2H/D2D) | 0.98 | 1.1 | 253 |
| other | 0.04 | 0.0 | 4 |

Top kernels: `nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_TNT` 22.87 ms; `nvjet_sm90_tst_128x200_64x5_2x1_v_bz_coopA_TNT` 18.71 ms; `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_` 17.84 ms; `nvjet_sm90_tst_128x72_64x8_2x4_h_bz_TNT` 3.72 ms; `triton_poi_fused__scaled_dot_product_cudnn_attention__to_cop` 2.88 ms; `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloa` 2.74 ms

**performance.standard.sdpa_off**: GPU busy 84.44 ms, idle 3.74 ms, kernel sum 92.01 ms per chunk.

| Category | ms/chunk | % | Launches/chunk |
|---|---:|---:|---:|
| GEMM bf16 | 44.92 | 48.8 | 767 |
| attention | 28.78 | 31.3 | 498 |
| decode (VAE) + frame conversion | 7.5 | 8.2 | 446 |
| Inductor Triton (fused norm/act/cat) | 5.98 | 6.5 | 726 |
| eager elementwise/reduce/copy | 2.96 | 3.2 | 661 |
| temporal RoPE (Triton) | 1.23 | 1.3 | 107 |
| copies (memcpy H2D/D2H/D2D) | 0.6 | 0.7 | 183 |
| other | 0.04 | 0.0 | 4 |

Top kernels: `nvjet_sm90_tst_192x208_64x4_2x1_v_bz_coopB_TNT` 22.65 ms; `nvjet_sm90_tst_128x200_64x5_2x1_v_bz_coopA_TNT` 18.48 ms; `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits` 13.56 ms; `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_` 11.81 ms; `nvjet_sm90_tst_128x72_64x8_2x4_h_bz_TNT` 3.63 ms; `triton_poi_fused__scaled_dot_product_cudnn_attention__to_cop` 2.88 ms

**performance.balanced**: GPU busy 79.12 ms, idle 4.76 ms, kernel sum 87.11 ms per chunk.

| Category | ms/chunk | % | Launches/chunk |
|---|---:|---:|---:|
| GEMM FP8 | 32.92 | 37.8 | 321 |
| attention | 19.97 | 22.9 | 358 |
| GEMM bf16 | 11.21 | 12.9 | 446 |
| decode (VAE) + frame conversion | 8.19 | 9.4 | 446 |
| Inductor Triton (fused norm/act/cat) | 5.97 | 6.9 | 656 |
| FP8 activation quantize | 3.67 | 4.2 | 321 |
| eager elementwise/reduce/copy | 2.85 | 3.3 | 591 |
| temporal RoPE (Triton) | 1.25 | 1.4 | 107 |
| copies (memcpy H2D/D2H/D2D) | 1.04 | 1.2 | 467 |
| other | 0.03 | 0.0 | 4 |

Top kernels: `nvjet_sm90_qqtst_128x128_128x6_1x1_h_bz_coopA_algo2_ovscale_` 32.92 ms; `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_` 16.73 ms; `nvjet_sm90_tst_128x200_64x5_2x1_v_bz_coopA_TNT` 7.46 ms; `_quantize_slices_kernel` 3.67 ms; `nvjet_sm90_tst_128x72_64x8_2x4_h_bz_TNT` 3.61 ms; `triton_poi_fused__scaled_dot_product_cudnn_attention__to_cop` 2.87 ms

**performance.balanced.sdpa_off**: GPU busy 85.9 ms, idle 4.84 ms, kernel sum 92.32 ms per chunk.

| Category | ms/chunk | % | Launches/chunk |
|---|---:|---:|---:|
| GEMM FP8 | 33.04 | 35.8 | 321 |
| attention | 27.1 | 29.4 | 498 |
| GEMM bf16 | 11.15 | 12.1 | 446 |
| decode (VAE) + frame conversion | 6.44 | 7.0 | 446 |
| Inductor Triton (fused norm/act/cat) | 6.02 | 6.5 | 726 |
| FP8 activation quantize | 3.65 | 4.0 | 321 |
| eager elementwise/reduce/copy | 2.89 | 3.1 | 661 |
| temporal RoPE (Triton) | 1.22 | 1.3 | 107 |
| copies (memcpy H2D/D2H/D2D) | 0.79 | 0.9 | 397 |
| other | 0.03 | 0.0 | 4 |

Top kernels: `nvjet_sm90_qqtst_128x128_128x6_1x1_h_bz_coopA_algo2_ovscale_` 33.04 ms; `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits` 12.44 ms; `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_` 11.28 ms; `nvjet_sm90_tst_128x200_64x5_2x1_v_bz_coopA_TNT` 7.45 ms; `_quantize_slices_kernel` 3.65 ms; `nvjet_sm90_tst_128x72_64x8_2x4_h_bz_TNT` 3.54 ms

**performance.low_memory**: GPU busy 81.18 ms, idle 5.1 ms, kernel sum 87.82 ms per chunk.

| Category | ms/chunk | % | Launches/chunk |
|---|---:|---:|---:|
| GEMM FP8 | 44.19 | 50.3 | 751 |
| attention | 19.72 | 22.5 | 358 |
| decode (VAE) + frame conversion | 6.9 | 7.9 | 446 |
| Inductor Triton (fused norm/act/cat) | 6.09 | 6.9 | 656 |
| FP8 activation quantize | 5.75 | 6.5 | 751 |
| eager elementwise/reduce/copy | 2.8 | 3.2 | 591 |
| temporal RoPE (Triton) | 1.23 | 1.4 | 107 |
| copies (memcpy H2D/D2H/D2D) | 0.97 | 1.1 | 467 |
| GEMM bf16 | 0.15 | 0.2 | 16 |
| other | 0.03 | 0.0 | 4 |

Top kernels: `nvjet_sm90_qqtst_128x128_128x6_1x1_h_bz_coopA_algo2_ovscale_` 44.19 ms; `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_` 16.53 ms; `_quantize_slices_kernel` 5.75 ms; `triton_poi_fused__scaled_dot_product_cudnn_attention__to_cop` 2.82 ms; `void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloa` 1.54 ms; `_rotate_kernel` 1.23 ms

**performance.low_memory.sdpa_off**: GPU busy 88.0 ms, idle 4.98 ms, kernel sum 93.62 ms per chunk.

| Category | ms/chunk | % | Launches/chunk |
|---|---:|---:|---:|
| GEMM FP8 | 44.33 | 47.4 | 751 |
| attention | 26.66 | 28.5 | 498 |
| Inductor Triton (fused norm/act/cat) | 6.18 | 6.6 | 726 |
| FP8 activation quantize | 5.68 | 6.1 | 751 |
| decode (VAE) + frame conversion | 5.62 | 6.0 | 446 |
| eager elementwise/reduce/copy | 2.99 | 3.2 | 663 |
| temporal RoPE (Triton) | 1.21 | 1.3 | 107 |
| copies (memcpy H2D/D2H/D2D) | 0.77 | 0.8 | 397 |
| GEMM bf16 | 0.15 | 0.2 | 16 |
| other | 0.03 | 0.0 | 4 |

Top kernels: `nvjet_sm90_qqtst_128x128_128x6_1x1_h_bz_coopA_algo2_ovscale_` 44.33 ms; `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits` 12.14 ms; `cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wgmma_f16_` 11.18 ms; `_quantize_slices_kernel` 5.68 ms; `triton_poi_fused__scaled_dot_product_cudnn_attention__to_cop` 2.83 ms; `_rotate_kernel` 1.21 ms

## FP8 `_scaled_mm` microbenchmark (M = 788, TFLOP/s; H100 in brackets)

| Shape | bf16 | FP8 row-wise (current) | + fast accum | per-tensor | per-tensor + fast accum |
|---|---:|---:|---:|---:|---:|
| gate/up 12288x4096 | 763.8 [763] | 795.8 [760] | 1136.2 [1016] | 1274.2 [1224] | 1377.7 [1302] |
| down 4096x12288 | 798.8 [798] | 709.8 [686] | 1175.3 [1039] | 1156.4 [1213] | 1491.1 [1356] |
| q/o 4096x4096 | 731.5 [722] | 714.9 [670] | 890.3 [683] | 991.0 [1019] | 1252.1 [1250] |

## Files

`results.json` (all numbers), `logs/` (one log per task), `env/nvidia-smi-q.txt`, `cases/*/contact_sheet.png`, `cases/*/frames_small.npy`.
