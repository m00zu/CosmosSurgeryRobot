# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load an exported Cosmos3 checkpoint into the native ``CosmosMoTNetwork``.

The meta bijection (``test_network_meta.py``) proves the parameter tree matches
the checkpoint's keys and shapes with zero memory. This module does the real
thing on hardware: read the exported safetensors weights, apply the
``checkpoint_remap`` rename, and ``load_state_dict(strict=True)`` into the
network — proving the shell loads the actual 31.5 GB of weights with no missing,
unexpected, or mis-shaped tensors.

The checkpoint source can be:

* a local directory holding ``model.safetensors.index.json`` + shards (default),
* a local ``*.safetensors.index.json`` path,
* a **Hugging Face** index URL (``https://huggingface.co/<org>/<repo>/resolve/<rev>/model.safetensors.index.json``)
  — shards are parallel-downloaded + merged into ``~/.cache`` (like the VAE),
* an **s3://** single-file ``.safetensors`` (with ``--credential-path`` for the
  bucket, e.g. the GCP checkpoint secret) — cached locally.

It may be given explicitly or via ``$COSMOS3_H_DREAMS_CHECKPOINT`` so the app
auto-fetches with no local export needed. Fits a single 48 GB GPU. Examples:

    uv run --no-sync python -m cosmos3_h_dreams._runtime.checkpoint_load \\
        /path/to/exported/causal_8b_sf_dmd_iter4000_student --device cuda
    COSMOS3_H_DREAMS_CHECKPOINT=https://huggingface.co/<org>/<repo>/resolve/main/model.safetensors.index.json \\
        uv run --no-sync python -m cosmos3_h_dreams._runtime.checkpoint_load
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open

from cosmos3_h_dreams._runtime._network import (
    CosmosMoTNetwork,
    CosmosMoTNetworkConfig,
    _off_generation_path,
)
from cosmos3_h_dreams._runtime.checkpoint_remap import remap_key

CHECKPOINT_ENV_VAR = "COSMOS3_H_DREAMS_CHECKPOINT"
"""Env var giving the default checkpoint source (dir / index.json / HF URL / s3 URI)."""

CREDENTIAL_ENV_VAR = "COSMOS_CHECKPOINT_CREDENTIAL"
"""Env var giving the S3/GCS credential file path for remote ``s3://`` sources."""

# Credential files are searched under ``credentials/`` (relative to CWD, then the
# launch directory) when none is given explicitly / via the env var.
_DEFAULT_CREDENTIAL_FILES = ("gcp_checkpoint.secret", "s3_checkpoint.secret")


def resolve_credential_path(
    explicit: str | os.PathLike | None = None,
    *,
    checkpoint_source: str | os.PathLike | None = None,
) -> str | None:
    """Resolve the S3/GCS credential file: explicit → env var → standard locations.

    Returns ``None`` if nothing is found (fine for local / HF sources, which need
    no credential). Lets the user drop the secret under ``<repo>/credentials/`` and
    never pass ``--credential-path``.
    """
    if explicit:
        return str(explicit)
    env = os.environ.get(CREDENTIAL_ENV_VAR)
    if env:
        return env
    files = list(_DEFAULT_CREDENTIAL_FILES)
    roots = [Path.cwd()]
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for fname in files:
            path = root / "credentials" / fname
            if path.is_file():
                return str(path)
    return None


def resolve_checkpoint_source(cli_source: str | os.PathLike | None) -> str:
    """CLI value, else ``$COSMOS3_H_DREAMS_CHECKPOINT``; error if neither set."""
    source = str(cli_source) if cli_source else os.environ.get(CHECKPOINT_ENV_VAR)
    if not source:
        raise SystemExit(
            "No checkpoint source. Pass it as an argument or set "
            f"${CHECKPOINT_ENV_VAR} (local dir, local model.safetensors.index.json, "
            "a Hugging Face index URL, or an s3:// single-file .safetensors)."
        )
    return source


def _is_remote(source: str) -> bool:
    return (
        source.startswith(("s3://", "http://", "https://"))
        or "huggingface.co" in source
    )


def _shard_files(exported_dir: Path) -> list[str]:
    index = json.loads((exported_dir / "model.safetensors.index.json").read_text())
    return sorted(set(index["weight_map"].values()))


def _load_raw_state_dict(
    source: str, device: str, credential_path: str | None
) -> dict[str, torch.Tensor]:
    """Read the raw (pre-remap) checkpoint tensors from a local dir or a remote source.

    A local directory is read shard-by-shard directly (no extra merge/cache).
    Everything else — a local ``*.index.json``, an HF index URL, or an ``s3://``
    single file — is delegated to FlashDreams' ``load_single_checkpoint``, which
    handles HF parallel-shard download + merge, s3 credentials, and local caching.
    """
    if not _is_remote(source) and Path(source).is_dir():
        exported_dir = Path(source)
        raw: dict[str, torch.Tensor] = {}
        for shard in _shard_files(exported_dir):
            with safe_open(
                str(exported_dir / shard), framework="pt", device=device
            ) as f:
                for key in f.keys():
                    raw[key] = f.get_tensor(key)
        return raw

    from flashdreams.core.checkpoint.load import load_single_checkpoint

    kwargs: dict[str, object] = {"map_location": device}
    if credential_path:
        kwargs["credential_path"] = credential_path
    return load_single_checkpoint(source, **kwargs)  # type: ignore[arg-type]


def _load_generation_state_dict(source: str, device: str) -> dict[str, torch.Tensor]:
    """Read only the raw tensors a generation step uses from a local checkpoint."""
    raw: dict[str, torch.Tensor] = {}
    for file in _checkpoint_files(source):
        with safe_open(str(file), framework="pt", device=device) as f:
            keys = list(f.keys())
            names = _remap_names(keys)
            for key in keys:
                if not _off_generation_path(names[key]):
                    raw[key] = f.get_tensor(key)
    return raw


def _remap_names(keys: Iterable[str]) -> dict[str, str]:
    """Map exported or native ``model.net`` checkpoint keys to network names."""
    keys = list(keys)
    exported = [key.startswith("model.net.") for key in keys]
    if any(exported) and not all(exported):
        raise RuntimeError("checkpoint mixes exported and native model keys")
    prefix = "" if all(exported) else "model.net."
    names = {key: remap_key(prefix + key) for key in keys}
    if None in names.values():
        raise RuntimeError("some source keys did not remap")
    return names  # type: ignore[return-value]


def _remap_state_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap exported or native ``model.net`` state-dict keys."""
    names = _remap_names(raw)
    return {names[key]: value for key, value in raw.items()}


def _checkpoint_files(source: str) -> list[Path]:
    """Local safetensors files of a single-file, shard-directory or index-json source."""
    if _is_remote(source):
        raise ValueError(f"text cache needs a local checkpoint, got {source}")
    path = Path(source)
    if path.is_file() and path.name.endswith(".index.json"):
        weight_map = json.loads(path.read_text())["weight_map"]
        return [path.parent / shard for shard in sorted(set(weight_map.values()))]
    if path.is_dir():
        return [path / shard for shard in _shard_files(path)]
    return [path]


def _num_embodiment_domains(remapped: dict[str, torch.Tensor]) -> int:
    """Return the action-projection row count encoded by a checkpoint."""
    action_projection = remapped.get("action2llm.fc.weight")
    if action_projection is None:
        return 32
    return int(action_projection.shape[0])


def _cast_floating_state_dict(
    remapped: dict[str, torch.Tensor], dtype: torch.dtype
) -> None:
    """Cast floating checkpoint tensors in-place, one tensor at a time."""
    for key, value in remapped.items():
        if value.is_floating_point() and value.dtype != dtype:
            remapped[key] = value.to(dtype=dtype)


def load_exported_checkpoint(
    source: str | os.PathLike,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    credential_path: str | None = None,
    offload_understanding: bool = False,
    fp8_generation: bool = False,
    fp8_scope: str = "all",
    fp8_gemm: str = "epilogue_scales",
    skip_understanding: bool = False,
) -> CosmosMoTNetwork:
    """Build the network and strict-load the remapped exported checkpoint.

    Args:
        source: local dir, local ``model.safetensors.index.json``, HF index URL,
            or ``s3://`` single-file ``.safetensors`` (see module docstring).
        device: Where to place the loaded weights (``cuda`` fits a 48 GB card).
        dtype: Parameter dtype (checkpoint is bf16).
        credential_path: S3/GCS credential file for remote ``s3://`` sources.
        offload_understanding: Read the checkpoint into host memory and place
            only the generation path on ``device``. The understanding half, text
            embedding and unused heads (about half the weights) stay in host
            memory; text prefill borrows them one layer at a time.
        fp8_generation: Quantize the generation-half projections to FP8 while
            moving them from host memory, one projection at a time.
        fp8_scope: ``"all"`` (attention and MLP) or ``"mlp"`` (MLP only, attention
            stays ``dtype``); used only with ``fp8_generation``.
        fp8_gemm: Where the FP8 projections apply their scales (``"fused_scales"``
            or ``"epilogue_scales"``); used only with ``fp8_generation``.
        skip_understanding: Never read the understanding half, text embedding or
            unused heads; they stay on ``meta``. Text K/V must then come from a
            saved text cache. Local checkpoints only.

    Returns:
        The loaded :class:`CosmosMoTNetwork`.

    Raises:
        RuntimeError: if the strict load reports any missing / unexpected key.
    """
    started = time.perf_counter()
    staged = offload_understanding or fp8_generation
    raw_device = "cpu" if staged else device
    if skip_understanding:
        raw = _load_generation_state_dict(str(source), raw_device)
    else:
        raw = _load_raw_state_dict(str(source), raw_device, credential_path)
    remapped = _remap_state_dict(raw)
    del raw
    _cast_floating_state_dict(remapped, dtype)

    # Auto-detect optional pathways from the remapped keys so the tree matches the
    # checkpoint for a strict load: the full export carries ``visual.*`` (ViT); the
    # camera checkpoint carries ``action2llm.*`` (action conditioning) while the
    # no-action checkpoint carries neither. With ``skip_understanding`` the ``visual.*``
    # keys are never read, so the ViT tower is left out on purpose: generation never
    # uses it (a cache miss loads it, then drops it to ``meta``).
    has_vision = any(k.startswith("visual.") for k in remapped)
    has_action = any(k.startswith("action2llm.") for k in remapped)
    with torch.device("meta"):
        net = CosmosMoTNetworkConfig(
            dtype=dtype,
            include_vision_tower=has_vision,
            action_gen=has_action,
            num_embodiment_domains=_num_embodiment_domains(remapped),
        ).setup()
    assert isinstance(net, CosmosMoTNetwork)
    if not has_vision and not skip_understanding:
        print("no visual.* keys -> building without the ViT tower (text2video).")
    if has_action:
        print(
            "action2llm.* keys -> building the action-conditioning pathway (camera checkpoint)."
        )

    strict = not skip_understanding
    missing, unexpected = net.load_state_dict(remapped, strict=strict, assign=True)  # type: ignore[arg-type]
    if skip_understanding:
        missing = [key for key in missing if not _off_generation_path(key)]
    if missing or unexpected:
        raise RuntimeError(
            f"non-strict load: missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    del remapped
    if fp8_generation:
        net.quantize_generation_fp8(device, fp8_scope, fp8_gemm)
    if skip_understanding:
        net.place_for_generation(device, understanding_device="meta")
    elif staged:
        net.place_for_generation(
            device, understanding_device="cpu" if offload_understanding else device
        )

    elapsed = time.perf_counter() - started
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"strict load OK: {n_params:,} params ({n_params * 2 / 1e9:.1f} GB bf16) "
        f"onto {device} in {elapsed:.1f}s"
    )
    if skip_understanding:  # never loaded, so never offloaded either
        print("understanding weights skipped; text K/V come from the text cache")
    elif offload_understanding:
        print(f"understanding weights offloaded to host memory; generation on {device}")
    if fp8_generation:
        which = "MLP projections" if fp8_scope == "mlp" else "projections"
        print(f"generation {which} quantized to FP8 (E4M3), {fp8_gemm}")
    if device.startswith("cuda") and torch.cuda.is_available():
        print(
            f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB "
            f"(load peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB)"
        )
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=f"checkpoint source; defaults to ${CHECKPOINT_ENV_VAR}",
    )
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    parser.add_argument(
        "--credential-path",
        default=None,
        help=f"S3/GCS credential file for s3:// sources "
        f"(default: ${CREDENTIAL_ENV_VAR} or <repo>/credentials/gcp_checkpoint.secret)",
    )
    args = parser.parse_args()
    source = resolve_checkpoint_source(args.source)
    credential_path = resolve_credential_path(
        args.credential_path, checkpoint_source=source
    )
    load_exported_checkpoint(
        source, device=args.device, credential_path=credential_path
    )
    print("DONE: native CosmosMoTNetwork loaded the exported checkpoint (strict).")


if __name__ == "__main__":
    main()
