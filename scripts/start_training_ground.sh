#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the training ground for this machine's browser: http://localhost:8443/train.
#
# Usage: scripts/start_training_ground.sh [extra server options]
#   CONFIG selects the session config (default: the laptop RTX 5090 config).
#   Plain HTTP by default (fine for localhost); pass --https to use cert.pem/key.pem
#   from the config instead, e.g. for other PCs or a Meta Quest headset.
#   Extra options are passed to the server, e.g. --port 9001.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-cosmos3_h_dreams/configs/laptop_rtx5090.yaml}
# Keep compiled GPU code across reboots (/tmp is emptied), so later starts are fast.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}

scheme=(--http)
args=()
for arg in "$@"; do
  if [ "$arg" = "--https" ]; then scheme=(); else args+=("$arg"); fi
done

exec uv run --no-sync cosmos3-h-dreams-run headset --session-config "$CONFIG" \
  "${scheme[@]}" "${args[@]}"
