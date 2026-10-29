#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optional: build SageAttention2 (thu-ml, Apache-2.0) into the project venv so
# `attention_kernel: auto` can use it on RTX 50 GPUs (13-15 % faster per chunk on an
# RTX 5090/5080; slower on H100, where auto keeps PyTorch SDPA).
#
# Upstream launches its kernels with a bare <<<grid, block>>> (the legacy default
# stream): CUDA graph capture misses them and the video turns to NaN once the FadeMem
# cache fills. This script moves every launch to PyTorch's current stream. The server
# self-tests the build at start-up and falls back to SDPA if it is unusable.
#
# Usage: scripts/install_sageattention.sh [VENV_PYTHON]
#   VENV_PYTHON defaults to $UV_PROJECT_ENVIRONMENT/bin/python, else .venv/bin/python.
#   ARCHS overrides the target (TORCH_CUDA_ARCH_LIST, e.g. "12.0"); default: this GPU.
#   CUDA_HOME selects the toolkit; its nvcc must match torch's CUDA major version.
#   MAX_JOBS (default 4) bounds parallel compiles; each can use several GB of RAM.
#
# Needs git, uv, a CUDA toolkit and a C++ compiler. Builds in a temporary directory
# (about 10-30 minutes). `uv sync` removes the package again (it is not in uv.lock);
# re-run this script afterwards, or sync with --inexact.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
VENV=${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}
PY=${1:-$VENV/bin/python}
REV=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 # measured on RTX 5090, RTX 5080 and H100

[ -x "$PY" ] || { echo "no python at $PY; pass the venv's python" >&2; exit 1; }
read -r TORCH_CUDA GPU_ARCH < <("$PY" -c "
import torch
assert torch.cuda.is_available(), 'no CUDA GPU visible'
major, minor = torch.cuda.get_device_capability()
print(torch.version.cuda, f'{major}.{minor}')")
ARCHS=${ARCHS:-$GPU_ARCH}
CUDA_HOME=${CUDA_HOME:-$(ls -d /usr/local/cuda-"${TORCH_CUDA%%.*}".* 2>/dev/null | sort -V | tail -1)}
NVCC="$CUDA_HOME/bin/nvcc"
[ -x "$NVCC" ] || { echo "no CUDA ${TORCH_CUDA%%.*}.x nvcc found; set CUDA_HOME" >&2; exit 1; }
NVCC_CUDA=$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
[ "$NVCC_CUDA" = "${TORCH_CUDA%%.*}" ] || {
  echo "nvcc is CUDA $NVCC_CUDA but torch is built for CUDA $TORCH_CUDA; set CUDA_HOME" >&2
  exit 1
}
echo "torch CUDA $TORCH_CUDA, nvcc $NVCC, arch $ARCHS"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
git clone -q https://github.com/thu-ml/SageAttention.git "$WORK/SageAttention"
cd "$WORK/SageAttention"
git checkout -q "$REV"
for f in $(grep -rlE "<<<[^>]*>>>" --include=*.cu csrc); do
  sed -i -E \
    -e "s/<<<\s*([^,>]+),\s*([^,>]+)\s*>>>/<<<\1, \2, 0, at::cuda::getCurrentCUDAStream()>>>/" \
    -e "s/<<<\s*([^,>]+),\s*([^,>]+),\s*([^,>]+)\s*>>>/<<<\1, \2, \3, at::cuda::getCurrentCUDAStream()>>>/" "$f"
  grep -q "ATen/cuda/CUDAContext.h" "$f" || sed -i "1i #include <ATen/cuda/CUDAContext.h>" "$f"
done
if grep -rhoE "<<<[^>]*>>>" --include=*.cu csrc | grep -v getCurrentCUDAStream; then
  echo "unpatched launch sites remain" >&2
  exit 1
fi
echo "launch sites on the current stream: $(grep -rhoE '<<<[^>]*getCurrentCUDAStream\(\)>>>' --include=*.cu csrc | wc -l)"

# nvcc 13 with gcc 13 rejects one line of torch's ATen/core/List_inl.h ("need 'typename'
# before 'decltype'"); compile against a copy with that cast spelled out.
TORCH_INCLUDE=$("$PY" -c "import torch.utils.cpp_extension as c; print(c.include_paths()[0])")
mkdir -p "$WORK/include/ATen/core"
sed 's/static_cast<typename decltype(impl_->list)::difference_type>(pos)/static_cast<std::ptrdiff_t>(pos)/' \
  "$TORCH_INCLUDE/ATen/core/List_inl.h" > "$WORK/include/ATen/core/List_inl.h"

# Build tools in a scratch directory, so the venv only gains sageattention itself.
uv pip install -q --python "$PY" --target "$WORK/tools" setuptools wheel ninja packaging
export CUDA_HOME PATH="$WORK/tools/bin:$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCHS"
export MAX_JOBS=${MAX_JOBS:-4} EXT_PARALLEL=${EXT_PARALLEL:-1} NVCC_APPEND_FLAGS=--threads=2
export NVCC_PREPEND_FLAGS="-I$WORK/include"
# --no-deps keeps the locked torch/cuBLAS untouched.
PYTHONPATH="$WORK/tools" uv pip install --python "$PY" --no-build-isolation --no-deps .

cd "$REPO"
PYTHONPATH="cosmos3_h_dreams:flashdreams${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c "
from cosmos3_h_dreams._runtime._attention import select_attention_kernel
print('self-test:', select_attention_kernel('sage', 'cuda:0').reason)"
echo "SageAttention installed; attention_kernel: auto uses it on RTX 50 GPUs."
