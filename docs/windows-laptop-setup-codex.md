<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Setup protocol for a coding agent: training ground on a Windows RTX 5090 laptop

This file is written for a coding agent (Codex or similar) that sets up the
Cosmos3-H-Dreams training ground on a Windows laptop with an RTX 5090 Laptop
GPU. The human-readable overview, with sizes and reasons, is
[windows-laptop-setup.md](windows-laptop-setup.md). Read that page's sections 1
and 2 once before starting.

## 0. Rules for the agent

1. **Target layout.** Everything Linux runs inside **WSL 2, Ubuntu 24.04**. The
   repository lives at `~/CosmosSurgeryRobot` on the Linux file system, never
   under `/mnt/c/...` (Windows-drive access from WSL is many times slower and
   the 30 GB model load would crawl).
2. **Where you run.** Run inside the Ubuntu (WSL) shell. If you were started
   in Windows PowerShell instead, run Linux commands as
   `wsl -d Ubuntu-24.04 -- bash -lc '<command>'`, and do steps 1–2 together
   with the user.
3. **The human does these, you ask and wait:** installing or updating the
   Windows NVIDIA driver, `wsl --install` (administrator PowerShell) and the
   restart after it, creating the Ubuntu user, anything that prompts for the
   Linux `sudo` password (ask the user to run that block, or to type the
   password), and `gh auth login`. Show the exact command and what to look for.
4. **One step at a time.** Each step ends with a **Check**. Do not continue
   until the check passes. If it fails, consult section 11, try the listed fix
   once, and otherwise stop and report the exact command and its full output.
5. **Never:** install an NVIDIA driver inside Ubuntu (`nvidia-driver-*`,
   `cuda-drivers`, or the `cuda` metapackage); edit `uv.lock` or any
   `pyproject.toml`; run `uv sync` without `--locked`; `pip install` into the
   environment (the only exception is `scripts/install_sageattention.sh`);
   commit, push, or change git configuration beyond what `gh auth login`
   does; store tokens or passwords in files.
6. **Long commands** (downloads, the SageAttention build, the first server
   start) can take 5–40 minutes. Run them so their output is kept (a log
   file), and report progress rather than assuming failure.
7. Every command after step 5 runs from `~/CosmosSurgeryRobot`.

## 1. Windows prerequisites (human, with your guidance)

**1a. NVIDIA driver.** In Windows PowerShell:

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

**Check:** the name contains `RTX 5090 Laptop GPU` (or another RTX 50 GPU with
at least 12 GB) and the driver version is **580 or higher**. Otherwise the
user installs the current Game Ready or Studio driver from nvidia.com (or the
NVIDIA app) and restarts.

**1b. WSL and Ubuntu.** In an **administrator** PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if asked. Ubuntu then opens and asks for a new Linux user name
and password (the user chooses them; this password is the `sudo` password).
If WSL was already installed:

```powershell
wsl --update
wsl -l -v
```

**Check:** `wsl -l -v` lists `Ubuntu-24.04` with `VERSION 2`.

**1c. WSL memory.** Ask how much RAM the laptop has (Task Manager →
Performance → Memory, or `systeminfo | findstr /C:"Total Physical Memory"`).
Create or edit `C:\Users\<user>\.wslconfig` (in PowerShell:
`notepad $env:USERPROFILE\.wslconfig`):

```ini
[wsl2]
# 32 GB laptop: memory=24GB and swap=32GB. 64 GB or more: memory=40GB and swap=16GB.
memory=40GB
swap=16GB
```

Then, in PowerShell, `wsl --shutdown`, and open Ubuntu again.

**Check:** inside Ubuntu, `free -g` shows a `Mem: total` close to the
`memory=` value.

## 2. Check the WSL environment

```bash
grep -i microsoft /proc/version && lsb_release -ds
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi | head -4
free -g
df -h ~
```

**Check:**
- `/proc/version` mentions Microsoft (this is WSL) and Ubuntu is 24.04.
- `nvidia-smi` works inside Ubuntu and shows the RTX 5090 Laptop GPU and
  driver 580+; its header shows `CUDA Version: 13.0` or higher. (In WSL
  `nvidia-smi` comes from `/usr/lib/wsl/lib`; that is correct.)
- `free -g`: total memory as configured in 1c.
- `df -h ~`: **80 GB or more available.** If not, stop and tell the user (they
  can free space on C: or move WSL with
  `wsl --manage Ubuntu-24.04 --move D:\WSL` in PowerShell).

## 3. System packages (sudo)

Ask the user to run this, or to enter the password:

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg openssl build-essential gh
```

**Check:**

```bash
git --version && ffmpeg -version | head -1 && gcc --version | head -1 && gh --version | head -1
```

All four print versions. (`ffmpeg` writes the attempt videos; `gcc` is used
by Triton to build GPU kernels at run time; `gh` is the GitHub CLI for the
private repository.)

## 4. uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

**Check:** `uv 0.11.8` or newer. uv installs Python 3.12 by itself in step 6;
do not install Python with apt.

## 5. Clone the repository (private)

The repository `ITRI-BDL-D/CosmosSurgeryRobot` is private. Ask the user to
sign in (interactive; choose GitHub.com, HTTPS, and log in with a browser):

```bash
gh auth login
gh auth setup-git
```

Then:

```bash
gh repo clone ITRI-BDL-D/CosmosSurgeryRobot ~/CosmosSurgeryRobot
cd ~/CosmosSurgeryRobot
git log --oneline -1
```

**Check:** the clone succeeds and `git branch --show-current` prints
`main`. A "Repository not found" error means the user's GitHub
account has no access yet: stop and ask them to get added to `ITRI-BDL-D`.

## 6. Python environment

```bash
cd ~/CosmosSurgeryRobot
uv sync --package flashdreams-cosmos3-h-dreams --locked
```

About 5 GB of downloads (PyTorch 2.11 with CUDA 13 libraries); 5–20 minutes.

**Check:**

```bash
uv run --no-sync python -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Expect `2.11.0+cu130 13.0 True`, the RTX 5090 Laptop GPU and capability
`(12, 0)`.

## 7. Model checkpoint (30.3 GB)

Ask the user whether they already have
`c3-h-d_2-step_model_bf16.safetensors` on a Windows drive or USB disk.

**If yes**, copy it in (Windows drives appear under `/mnt/<letter>/`; copy,
do not symlink, because reading from `/mnt` is slow):

```bash
mkdir -p data_local/checkpoints
cp /mnt/d/path/to/c3-h-d_2-step_model_bf16.safetensors data_local/checkpoints/
```

**Otherwise** download it from the public Hugging Face repository (no account
needed; 30.3 GB, 10 minutes to over an hour depending on the network):

```bash
uv run --no-sync python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('nvidia/Cosmos3-H-Dreams',
      'checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors', local_dir='data_local/hf'))"
mkdir -p data_local/checkpoints
mv data_local/hf/checkpoints/exp15/c3-h-d_2-step_model_bf16.safetensors data_local/checkpoints/
rm -rf data_local/hf
```

**Check:**

```bash
sha256sum data_local/checkpoints/c3-h-d_2-step_model_bf16.safetensors
```

must print `3e6fce0046a6fe46c65307245df0db041c4a1828e0b3fc76708fb30e7851c97b`.
A different hash means a broken or wrong file: delete it and download again.

## 8. SageAttention (recommended on RTX 50; optional)

SageAttention2 makes each video chunk 13–15 % faster on RTX 50 GPUs. It is
built from source with the CUDA 13 compiler. The server tests it at start-up
and falls back to the standard kernel if anything is wrong, so a failed build
here is not fatal: report it and continue with step 9.

**8a. CUDA 13 toolkit, compiler only** (sudo; about 3 GB). Use NVIDIA's
**WSL-Ubuntu** repository and the `cuda-toolkit-13-0` package only; it
contains no driver:

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
cd ~/CosmosSurgeryRobot
```

**Check:** `ls -d /usr/local/cuda-13.*` lists a directory and
`/usr/local/cuda-13.0/bin/nvcc --version` prints `release 13.0` (any 13.x is
fine).

**8b. Build** (20–40 minutes on a laptop). `MAX_JOBS` bounds parallel
compiler processes, each of which can use several GB of RAM: use 4 with a
32 GB laptop and 8 with 64 GB.

```bash
MAX_JOBS=4 scripts/install_sageattention.sh 2>&1 | tee data_local/sageattention-build.log | tail -5
```

**Check:** the last lines are `self-test: sm_120, self-test passed` and
`SageAttention installed; …`. If the build fails, keep
`data_local/sageattention-build.log` for the report and continue.

Note for later: `uv sync` removes SageAttention (it is not in `uv.lock`); run
the script again after any `uv sync`.

## 9. First start

```bash
cd ~/CosmosSurgeryRobot
nohup scripts/start_training_ground.sh > data_local/first-start.log 2>&1 &
echo $! > data_local/server.pid
```

The first start downloads 2.7 GB from Hugging Face, builds the text cache
(needs about 28 GB of RAM once) and compiles the model for the GPU. Allow
10–20 minutes. Poll until ready:

```bash
until curl -s http://localhost:8443/healthz | grep -q '"ready": true'; do
  tail -1 data_local/first-start.log; sleep 20
done
curl -s http://localhost:8443/healthz | python3 -m json.tool | grep -A3 '"attention"\|"text_cache"\|"mode"\|"model_status"'
```

If the server process exits instead (check with `kill -0 $(cat data_local/server.pid)`),
read the end of `data_local/first-start.log` and go to section 11.

**Check** in `/healthz`:
- `"ready": true` and `"model_status"` ending in "ready" or "compiled";
- `"attention": {"kernel": "sage", "reason": "sm_120, self-test passed"}` if
  step 8 succeeded (otherwise `"default"` with the reason; fine);
- `"text_cache": {"status": "built" …}` on this first start.

**9a. Switch to Low memory** (faster and smaller on RTX 50 GPUs):

```bash
curl -s -X POST -H 'Content-Type: application/json' -d '{"mode": "low_memory"}' http://localhost:8443/mode
until curl -s http://localhost:8443/healthz | grep -q '"mode_switching": false'; do sleep 5; done
curl -s http://localhost:8443/healthz | python3 -m json.tool | grep '"mode"\|"mode_error"'
```

**Check:** `"mode": "low_memory"` and `"mode_error": null`. (The mode resets
to the config's Balanced at the next start; tell the user it is under
⚙ Settings → Model mode.)

**9b. Browser test (human).** Ask the user to open
**http://localhost:8443/train** in Chrome or Edge on Windows, click the
video, move the mouse for about 30 seconds (Tab switches instrument, the wheel
moves up and down) and keep the page open. Then:

```bash
curl -s http://localhost:8443/healthz | python3 -c "
import json, sys; h = json.load(sys.stdin)
print('latency', h['latency']); print('input_rate', h['input_rate']); print('gpu', h['gpu'])"
```

**Check:** the user sees the endoscope video react to the mouse. Record
`latency.chunk_ms`, `latency.budget_ms`, `latency.status` and
`latency.response_ms` for the report.
- `status` `ok` or `tight`: done.
- `status` `behind` with `reason` `generation`: the GPU is too slow for
  30 Hz. Tell the user to set **⚙ Settings → Input rate** to the value the page
  suggests (or compute `floor(8000 / (chunk_ms × 1.1))` Hz), and check again
  after 30 s. Also confirm the laptop is plugged in and in Best performance
  power mode.

**9c. Second start (fast path).** Stop the server and start it again:

```bash
kill -INT $(cat data_local/server.pid); sleep 15
kill -0 $(cat data_local/server.pid) 2>/dev/null && kill $(cat data_local/server.pid)
start=$(date +%s)
nohup scripts/start_training_ground.sh > data_local/second-start.log 2>&1 &
echo $! > data_local/server.pid
until curl -s http://localhost:8443/healthz | grep -q '"ready": true'; do sleep 5; done
echo "ready after $(( $(date +%s) - start )) s"
curl -s http://localhost:8443/healthz | python3 -m json.tool | grep -A2 '"text_cache"'
kill -INT $(cat data_local/server.pid)
```

**Check:** ready within a few minutes and `"text_cache": {"status": "hit"`.
Delete `data_local/first-start.log` and `data_local/second-start.log` unless
something failed.

## 10. Measure the laptop (optional, about 30–60 minutes)

The repository's speed numbers for this laptop are estimates. The device
test kit measures each model mode and writes a small report. The server must
be stopped first (the kit needs the GPU alone):

```bash
uv run --no-sync python -m cosmos3_h_dreams.device_check \
  --out data_local/laptop-check --label "RTX 5090 Laptop, WSL2" 2>&1 | tail -5
```

It prints the path of `data_local/laptop-check.tar.gz`. Ask the user to send
that file to the repository owner (it holds timings, logs and small preview
images; no personal data). The kit's own documentation is
[rtx5090-test-kit.md](rtx5090-test-kit.md).

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvidia-smi: command not found` or "No devices were found" inside Ubuntu | Windows driver missing or older than 580, or WSL not restarted after an update. Fix on the Windows side (step 1a), then `wsl --shutdown` in PowerShell. Never install a driver inside Ubuntu |
| `torch.cuda.is_available()` is False | Same as above; also check `ls /usr/lib/wsl/lib/libcuda.so*` exists |
| `uv sync` fails with a timeout on large wheels | `export UV_HTTP_TIMEOUT=600` and run the same command again |
| `gh repo clone`: "Repository not found" | The GitHub account has no access to `ITRI-BDL-D/CosmosSurgeryRobot`; the user must ask to be added |
| SageAttention: `no CUDA 13.x nvcc found` | Step 8a missing; or set `CUDA_HOME=/usr/local/cuda-13.0` |
| SageAttention build killed, or WSL closes during the build | Out of memory: run again with `MAX_JOBS=2` |
| SageAttention: `self-test … differs from an eager call` | A build without the stream fix (for example installed with pip): run `scripts/install_sageattention.sh` again |
| Server log: `Killed`, or WSL restarts during "Loading model" | Not enough WSL memory for the one-time text-cache build: raise `memory=`/`swap=` in `.wslconfig` (step 1c), `wsl --shutdown`, start again |
| `CUDA out of memory` at start-up | Other programs use GPU memory (games, browsers with hardware acceleration, video calls): close them; start in Low memory (copy the config and set `fp8_scope: all`, see [training-ground-setup.md](training-ground-setup.md#6-choose-a-config)) |
| `ValueError: unknown session options: courses, instruments` | The command ran code from another checkout or branch. Run from `~/CosmosSurgeryRobot` on branch `main`, using `scripts/start_training_ground.sh` |
| Port 8443 already in use | Another server is running: stop it, or start with `scripts/start_training_ground.sh --port 8444` |
| The page does not load in the Windows browser, but `curl` inside WSL works | Use `http://127.0.0.1:8443/train`; check that `localhostForwarding` is not disabled in `.wslconfig` |
| `/healthz` shows `"recording_error"` about ffmpeg | `ffmpeg` missing: step 3 |

## 12. Letting other devices connect (only if the user asks)

By default WSL is reachable only from the laptop itself. For a colleague's PC
on the same network, or a Meta Quest headset:

1. Add `networkingMode=mirrored` under `[wsl2]` in `.wslconfig` (Windows 11
   22H2 or newer), then `wsl --shutdown`.
2. Allow the port in Windows Defender Firewall (administrator PowerShell):
   `New-NetFirewallRule -DisplayName "Training ground 8443" -Direction Inbound -Protocol TCP -LocalPort 8443 -Action Allow`.
   With mirrored networking the Hyper-V firewall may also need to allow it;
   follow Microsoft's "Accessing network applications with WSL" page if the
   port stays closed.
3. For the headset, or other PCs without the browser warning, use HTTPS:
   create `cert.pem`/`key.pem` for the laptop's IP address as in
   [training-ground-setup.md, section 5](training-ground-setup.md#5-https-certificate),
   and start with `scripts/start_training_ground.sh --https`.

Others then open `http://<laptop-IP>:8443/train` (or `https://…` with step 3).
Anyone who can reach the port can use the server; keep it on a trusted
network.

## 13. Final report to the user

Finish with a short report containing:

- driver version, GPU name, WSL memory and swap, free disk after setup;
- repository commit (`git log --oneline -1`);
- checkpoint hash check: passed or failed;
- SageAttention: installed and self-test passed, or the failure (attach the
  last 30 lines of `data_local/sageattention-build.log`);
- first start: time to ready, text cache built; second start: time to ready,
  text cache hit;
- from step 9b: `chunk_ms`, `budget_ms`, `status`, `response_ms`, the input
  rate in use, and the model mode;
- whether step 10 ran, and where `data_local/laptop-check.tar.gz` is;
- the daily start command: `cd ~/CosmosSurgeryRobot && scripts/start_training_ground.sh`,
  then open `http://localhost:8443/train`.
