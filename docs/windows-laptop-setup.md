<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Training ground on a Windows laptop (RTX 5090)

This page is for the person who owns the laptop. It explains what is needed,
what gets downloaded and installed, and how to use the training ground every
day. The setup itself can be done by Codex following
[windows-laptop-setup-codex.md](windows-laptop-setup-codex.md); a few steps
need you (administrator rights, a restart, passwords, a GitHub login), and
Codex will ask for them.

## How it runs on Windows

The training ground is Linux software. On Windows it runs inside **WSL 2**
(Windows Subsystem for Linux) with **Ubuntu 24.04**: a lightweight Linux that
Windows starts on demand and that uses the laptop's NVIDIA GPU directly. You
keep using Windows as usual; the server runs in an Ubuntu terminal window, and
you open the training ground in Chrome or Edge on Windows at
`http://localhost:8443/train`.

Native Windows (without WSL) is not supported: the repository is only tested
on Linux, and parts of it (GPU compilation, the optional SageAttention build)
are Linux-only in practice.

```
Windows 11                                  WSL 2: Ubuntu 24.04
┌───────────────────────────┐               ┌──────────────────────────────────┐
│ Chrome / Edge             │  localhost    │ ~/CosmosSurgeryRobot             │
│ http://localhost:8443/train ─────────────▶│ training-ground server (Python)  │
│                           │               │   └─ RTX 5090 via the Windows    │
│ NVIDIA driver (Windows)   │◀──────────────│      NVIDIA driver               │
└───────────────────────────┘     GPU       └──────────────────────────────────┘
```

## 1. Requirements

| Item | Requirement |
|---|---|
| Windows | Windows 11 (or Windows 10 22H2), 64-bit, with administrator rights for the one-time WSL install |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU (24 GB). Other RTX 50 laptops work if they have at least 12 GB |
| NVIDIA driver | The normal **Windows** Game Ready or Studio driver, version **580 or newer**. Nothing NVIDIA is installed inside Ubuntu |
| RAM | 32 GB minimum, 64 GB recommended (see "Memory" below) |
| Disk | **80 GB free on the drive that holds WSL** (normally C:) |
| Network | Internet for the setup (about 45 GB of downloads, mostly the model); GitHub access to the private repository `ITRI-BDL-D/CosmosSurgeryRobot` |
| Accounts | A GitHub account that has been added to the `ITRI-BDL-D` organisation (ask the repository owner). No Hugging Face account is needed |
| Browser | Current Chrome or Edge on Windows |
| Power | Plugged in, Windows power mode **Best performance** while training |

**Memory.** The first start builds a small "text cache" once and needs about
28 GB of RAM for that step; afterwards the server uses about 14 GB. WSL may
use at most half of the laptop's RAM unless you raise the limit, which the
setup does in `C:\Users\<you>\.wslconfig`:

| Laptop RAM | WSL memory | WSL swap |
|---|---|---|
| 32 GB | 24 GB | 32 GB (the first start then uses swap and is slower, once) |
| 64 GB or more | 40 GB | 16 GB |

**GPU memory.** The model needs 9–16 GB of the 24 GB depending on the model
mode; Windows itself and open programs use some too.

| Mode | GPU memory | Quality |
|---|---:|---|
| Standard | 15.8 GB | Best |
| Balanced | 10.6 GB | Near Standard (the laptop config starts here) |
| Low memory | 9.2 GB | Slightly lower; also the fastest on RTX 50 GPUs |

## 2. What gets downloaded and installed

| What | Source | Download | On disk | Where |
|---|---|---:|---:|---|
| WSL 2 and Ubuntu 24.04 | Microsoft | ~0.6 GB | ~2 GB | WSL virtual disk (`ext4.vhdx` in your user profile) |
| Ubuntu packages: `git`, `curl`, `ffmpeg` (saves attempt videos), `openssl`, `build-essential` (C compiler for GPU kernels) | Ubuntu mirrors | ~0.3 GB | ~1 GB | Ubuntu system |
| `uv` (Python package manager) and Python 3.12 | astral.sh | ~0.1 GB | ~0.2 GB | `~/.local` |
| This repository, branch `main` | GitHub (private) | ~40 MB | ~40 MB | `~/CosmosSurgeryRobot` |
| Python packages (PyTorch 2.11 with CUDA 13 libraries and the rest of `uv.lock`) | PyPI, download.pytorch.org | ~5 GB | 4.7 GB + ~5 GB download cache | `~/CosmosSurgeryRobot/.venv`, `~/.cache/uv` (cache can be cleaned) |
| **Model checkpoint** `c3-h-d_2-step_model_bf16.safetensors` | Hugging Face `nvidia/Cosmos3-H-Dreams` (public) | **30.3 GB** | 30.3 GB | `~/CosmosSurgeryRobot/data_local/checkpoints/` |
| On the first start: Wan2.2 VAE (2.7 GB), TAEHV video decoder (22 MB), Qwen3-VL tokenizer (11 MB) | Hugging Face | 2.7 GB | 2.7 GB | `~/.cache/huggingface` |
| Optional: CUDA 13 toolkit (compiler only, for SageAttention) | NVIDIA (WSL-Ubuntu repository) | ~3 GB | ~5 GB | `/usr/local/cuda-13.x` |
| Optional: SageAttention2 (faster attention on RTX 50; built from source, 20–40 min) | GitHub `thu-ml/SageAttention` | small | ~0.1 GB | inside `.venv` |
| Created by the server | — | — | text cache 6 MB, compiler caches ~0.1 GB, recordings ~15 MB per minute | `data_local/text_kv/`, `~/.cache/torchinductor`, `recordings/` |

Total: about 45 GB downloaded and 55–60 GB on disk, which is why 80 GB free is
asked for. If the model file is already on a USB drive or another disk, it can
be copied in instead of downloaded (the setup asks).

**What is not installed:** no NVIDIA driver inside Ubuntu (WSL uses the
Windows driver), no system-wide Python, nothing in Windows besides WSL and the
`.wslconfig` file.

## 3. Setup, in short

Codex follows [windows-laptop-setup-codex.md](windows-laptop-setup-codex.md),
step by step, and stops at each check. Expect 1–2 hours, mostly downloads and
the optional SageAttention build. You will be asked to:

1. update the NVIDIA driver if it is older than 580;
2. run `wsl --install -d Ubuntu-24.04` in an **administrator** PowerShell and
   restart Windows;
3. choose a Linux user name and password when Ubuntu first opens;
4. type your Linux password when Codex runs `sudo` commands (or run those
   commands yourself);
5. sign in to GitHub (`gh auth login`) so the private repository can be
   cloned.

Codex itself works best when run **inside the Ubuntu terminal** (install it
there), so it can run Linux commands directly.

## 4. Every day

Open **Ubuntu** from the Start menu (or Windows Terminal → Ubuntu) and run:

```bash
cd ~/CosmosSurgeryRobot
scripts/start_training_ground.sh
```

(This runs `uv run --no-sync cosmos3-h-dreams-run headset --session-config
cosmos3_h_dreams/configs/laptop_rtx5090.yaml --http` with the compiler cache
kept in your home folder. Extra options are passed on, for example `--port 9001`.)

Then open **http://localhost:8443/train** in Chrome or Edge. The page shows
"Loading model…" and "Compiling model…" until the video appears: about a
minute on a normal start, several minutes the very first time.

Stop the server with **Ctrl+C** in the Ubuntu window. Closing the window also
stops it. Shut WSL down completely (frees its memory) with `wsl --shutdown` in
PowerShell.

| Page | For |
|---|---|
| `http://localhost:8443/train` | Training: instruments, courses, report cards |
| `http://localhost:8443/dashboard` | Instructors: replay and grade recorded attempts |
| `http://localhost:8443/gpu` | GPU memory and RAM charts |

How to use the training page (controls, courses, safety stops, dashboard) is
described in [training-ground-setup.md](training-ground-setup.md#8-using-the-training-ground).

## 5. Speed on a laptop

The model makes video in chunks of 8 frames. To keep up with the default
30 Hz input, a chunk must be ready within 267 ms. Measured and estimated
times per chunk:

| GPU | Best setting | ms per chunk | Keeps up at 30 Hz? |
|---|---|---:|---|
| H100 (server) | any mode | 73 | yes (3.7×) |
| RTX 5090 desktop | Low memory + SageAttention | 138 | yes |
| RTX 5080 desktop (same chip as the laptop 5090) | Low memory + SageAttention | 216 | yes |
| **RTX 5090 laptop** | Low memory + SageAttention | **~275–310 (estimate, not measured yet)** | **just short** |

So on the laptop:

1. **Install SageAttention** (the setup does this; 13–15 % faster on RTX 50).
2. **Use Low memory mode**: ⚙ Settings → Model mode → Low memory (it is
   both smaller and faster on RTX 50 GPUs).
3. **If the speed readout under the video says "falling behind", lower the
   input rate**: ⚙ Settings → Input rate, for example 24 Hz (333 ms per chunk).
   The page suggests a rate when it falls behind. Motion is slightly less
   smooth, but the video keeps up.
4. Keep the laptop plugged in and in **Best performance** power mode; close
   games, video calls and other GPU-heavy programs.

Falling behind is not harmful: the video pauses briefly after each chunk and
some movements are skipped, but the delay does not grow.

**Please measure it.** The laptop numbers above are estimates. The setup's
last (optional) step runs the device test kit for about 30 minutes and makes a
small `.tar.gz` report; send it to the repository owner so the numbers can be
replaced with real ones.

## 6. Letting others connect (optional)

By default only the laptop itself can open the page (`localhost`). To let a
colleague's PC or a Meta Quest headset connect over the network, WSL must
share the laptop's network (**mirrored networking**, Windows 11 22H2 or
newer) and Windows Firewall must allow the port. The headset page also needs
HTTPS with a certificate. Ask Codex for "section 9 of the Codex protocol" or
see [training-ground-setup.md](training-ground-setup.md#5-https-certificate).

## 7. Troubleshooting

| Problem | What to do |
|---|---|
| `nvidia-smi` in Ubuntu says "command not found" or shows no GPU | Update the **Windows** NVIDIA driver to 580+, then `wsl --shutdown` in PowerShell and open Ubuntu again. Never install an NVIDIA driver inside Ubuntu |
| The page does not open at `localhost:8443` | Check the Ubuntu window: the server prints `Open http://<workstation-LAN-IP>:8443/train` when it listens. Try `http://127.0.0.1:8443/train` |
| The first start is killed or Ubuntu closes during "Loading model" | Not enough memory for WSL: raise `memory=`/`swap=` in `C:\Users\<you>\.wslconfig`, run `wsl --shutdown`, start again |
| "falling behind" | Section 5: Low memory, SageAttention, lower input rate, power mode |
| `attention: default (SageAttention not installed …)` in the server log | SageAttention is missing (for example after a `uv sync`). Run `scripts/install_sageattention.sh` again |
| Disk full | Clean the download cache: `uv cache clean`. The WSL disk can be moved to another drive with `wsl --manage Ubuntu-24.04 --move D:\WSL` (recent WSL versions) |
| After a Windows or driver update the first start is slow again | Normal: the GPU code is compiled again once |
