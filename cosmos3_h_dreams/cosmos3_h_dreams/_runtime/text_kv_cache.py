# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved prompt K/V so a run can skip the understanding-half weights.

Text prefill turns the prompt tokens into one K/V pair per layer, and every
generation step afterwards reads only those. The pair depends on the prompt,
the resolution suffix, the tokenizer and the checkpoint, so the file records
all of them and is reused only when they all match. The checkpoint is
identified by the SHA-256 of its file(s); the file's size and modification time
on this machine are stored too, so later runs skip re-hashing an unchanged file.
The stat only decides whether to re-hash: a copied or touched checkpoint with the
same bytes is hashed once, still matches, and the stored stat is refreshed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from cosmos3_h_dreams._runtime._network import CosmosMoTNetwork

TEXT_KV_FORMAT_VERSION = 2
"""Layout and meaning of the saved file; the cache cannot detect code changes.

The file is keyed on weights, prompt, resolution, tokenizer and dtype, not on
the code that turns them into K/V. Bump this whenever text-prefill math, text
positions, text embedding or checkpoint key remapping change, so old files are
rebuilt instead of reused.
"""


@dataclass(frozen=True)
class TextKVMeta:
    """Everything the saved text K/V depend on."""

    checkpoint_id: str
    checkpoint_stat: str  # not part of validity: only whether to re-hash
    prompt: str
    height: int
    width: int
    tokenizer: str
    dtype: str


@dataclass(frozen=True)
class MetaLookup:
    """The expected metadata plus how it was obtained.

    Attributes:
        meta: What a valid cache file must carry.
        hash_s: Seconds spent hashing the checkpoint, or None when the stored id was reused.
        rehashed: True when the cache file stored a checkpoint stat that no longer
            matches, so the checkpoint was hashed again (the file can still be valid).
    """

    meta: TextKVMeta
    hash_s: float | None
    rehashed: bool


@dataclass
class TextKV:
    """Per-layer understanding K/V for one prompt, on the CPU."""

    text_ids: Tensor
    k: list[Tensor]
    v: list[Tensor]


def save_text_kv(path: Path, kv: TextKV, meta: TextKVMeta) -> None:
    """Write ``kv`` and ``meta`` to ``path`` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": TEXT_KV_FORMAT_VERSION,
        "meta": dataclasses.asdict(meta),
        "text_ids": kv.text_ids.cpu(),
        "text_k": [t.cpu() for t in kv.k],
        "text_v": [t.cpu() for t in kv.v],
    }
    # One temp file per writer: two servers saving the same cache never share one.
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_text_kv(path: Path, meta: TextKVMeta) -> TextKV | None:
    """Return the saved K/V if ``path`` holds a file made for ``meta``, else ``None``.

    ``checkpoint_stat`` is not compared: when only it differs the file is valid,
    and it is re-saved with the new stat (best effort) so the next run skips hashing.
    """
    path = Path(path)
    if not path.is_file():
        print(f"[text-cache] {path} not found; building it")
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - any unreadable file is rebuilt
        print(f"[text-cache] {path} unreadable ({error}); rebuilding it")
        return None
    if payload.get("format_version") != TEXT_KV_FORMAT_VERSION:
        print(f"[text-cache] {path} has format_version {payload.get('format_version')}; rebuilding it")
        return None
    saved = payload.get("meta", {})
    for field in dataclasses.fields(TextKVMeta):
        if field.name == "checkpoint_stat":
            continue
        if saved.get(field.name) != getattr(meta, field.name):
            print(f"[text-cache] {path} was built for a different {field.name}; rebuilding it")
            return None
    kv = TextKV(text_ids=payload["text_ids"], k=payload["text_k"], v=payload["text_v"])
    if saved.get("checkpoint_stat") != meta.checkpoint_stat:
        try:
            save_text_kv(path, kv, meta)
            print(f"[text-cache] checkpoint unchanged; stored its new size/mtime in {path}")
        except (OSError, RuntimeError) as error:  # the next start just re-hashes
            print(f"[text-cache] could not update the checkpoint stat in {path} ({error})")
    return kv


def checkpoint_id(source: str | Path) -> str:
    """SHA-256 of the checkpoint's bytes; shards are read in file-name order.

    For a single-file checkpoint this equals ``sha256sum`` of the file.
    """
    from cosmos3_h_dreams._runtime.checkpoint_load import _checkpoint_files

    digest = hashlib.sha256()
    for file in _checkpoint_files(str(source)):
        with open(file, "rb") as handle:
            while chunk := handle.read(64 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_stat(source: str) -> str:
    """Name, size and modification time of each checkpoint file on this machine."""
    from cosmos3_h_dreams._runtime.checkpoint_load import _checkpoint_files

    return ";".join(
        f"{file.name}:{stat.st_size}:{stat.st_mtime_ns}"
        for file in _checkpoint_files(source)
        for stat in [file.stat()]
    )


def _saved_meta(path: Path) -> dict:
    """The metadata stored in a text cache file, or ``{}`` if there is none."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True).get("meta", {})
    except Exception:  # noqa: BLE001 - missing or unreadable means nothing to reuse
        return {}


@torch.no_grad()
def compute_text_kv(
    net: CosmosMoTNetwork, text_ids: Tensor, *, native_gqa_attention: bool = False
) -> TextKV:
    """Run text prefill once and return its per-layer K/V on the CPU."""
    from cosmos3_h_dreams._runtime._network import _MoTStreamCache
    from cosmos3_h_dreams._runtime._rope import text_mrope_ids

    n = int(text_ids.shape[0])
    layers = len(net.layers)
    cache = _MoTStreamCache(text_k=[None] * layers, text_v=[None] * layers, gen=[])
    positions = text_mrope_ids(n, 0.0)[0].to(net.device)
    net.prefill_text_stream(
        text_ids.to(net.device),
        positions,
        cache,
        native_gqa_attention=native_gqa_attention,
    )
    return TextKV(
        text_ids=text_ids.cpu(),
        k=[t.cpu() for t in cache.text_k],  # type: ignore[union-attr]
        v=[t.cpu() for t in cache.text_v],  # type: ignore[union-attr]
    )


def drop_understanding(net: CosmosMoTNetwork) -> None:
    """Free every weight a generation step never reads; they become ``meta``."""
    import gc

    net.place_for_generation(net.device, understanding_device="meta")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def expected_meta(
    *, checkpoint: str, prompt: str, height: int, width: int, cache: Path
) -> TextKVMeta:
    """The metadata a text cache must carry to be valid for this run; see :func:`resolve_meta`."""
    return resolve_meta(
        checkpoint=checkpoint, prompt=prompt, height=height, width=width, cache=cache
    ).meta


def resolve_meta(
    *, checkpoint: str, prompt: str, height: int, width: int, cache: Path
) -> MetaLookup:
    """The metadata a text cache must carry to be valid for this run, and how long it took.

    The checkpoint is hashed only when ``cache`` was not built from files with
    the same names, sizes and modification times.
    """
    from cosmos3_h_dreams._runtime._tokenizer import DEFAULT_TOKENIZER
    from cosmos3_h_dreams._runtime.checkpoint_load import resolve_checkpoint_source

    source = resolve_checkpoint_source(checkpoint)
    stat = _checkpoint_stat(source)
    saved = _saved_meta(Path(cache))
    hash_s = None
    if saved.get("checkpoint_stat") == stat and saved.get("checkpoint_id"):
        identity = saved["checkpoint_id"]
        print(f"[text-cache] checkpoint {identity[:12]} unchanged; not re-hashing")
    else:
        started = time.perf_counter()
        identity = checkpoint_id(source)
        hash_s = time.perf_counter() - started
        print(f"[text-cache] checkpoint id {identity[:12]} in {hash_s:.1f}s")
    meta = TextKVMeta(
        checkpoint_id=identity,
        checkpoint_stat=stat,
        prompt=prompt,
        height=int(height),
        width=int(width),
        tokenizer=DEFAULT_TOKENIZER,
        dtype=str(torch.bfloat16),
    )
    rehashed = bool(saved.get("checkpoint_stat")) and saved["checkpoint_stat"] != stat
    return MetaLookup(meta=meta, hash_s=hash_s, rehashed=rehashed)


def ensure_text_kv(
    net: CosmosMoTNetwork | None,
    text_ids: Tensor,
    *,
    cached: TextKV | None,
    meta: TextKVMeta,
    path: Path,
    native_gqa_attention: bool,
) -> tuple[TextKV, str]:
    """Return valid text K/V, building, saving and freeing the weights on a miss.

    Returns:
        The K/V and the outcome: ``"hit"``, ``"built"`` (saved) or
        ``"built-unsaved"`` (the save failed; the next start builds it again).
    """
    if cached is not None:
        if not torch.equal(cached.text_ids, text_ids.cpu()):
            raise ValueError(
                f"text cache {path} holds different prompt tokens; delete it to rebuild"
            )
        return cached, "hit"
    assert net is not None
    kv = compute_text_kv(net, text_ids, native_gqa_attention=native_gqa_attention)
    outcome = "built"
    try:
        save_text_kv(path, kv, meta)
        print(f"[text-cache] saved {path}")
    except (OSError, RuntimeError) as error:  # torch.save raises RuntimeError for these
        print(f"[text-cache] could not save {path} ({error}); continuing")
        outcome = "built-unsaved"
    drop_understanding(net)
    return kv, outcome
