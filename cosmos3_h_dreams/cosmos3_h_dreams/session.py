# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action-replay and headset sessions for Cosmos3-H-Dreams generation."""

from __future__ import annotations

import gc
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from flashdreams.demo import (
    CanonicalInputWindow,
    IFlashDreamsApplicationSession,
    SessionInfo,
)
from flashdreams.infra.results import StepResult
from flashdreams.runtime import StepRequirements
from PIL import Image, ImageDraw, ImageFont

from ._runtime import text_kv_cache
from ._runtime._attention import select_attention_kernel
from .config import Cosmos3HDreamsConfig
from .inputs import prepare_scene
from .model import Cosmos3HDreamsModel
from .headset import GenerationRestartRequested, HeadsetActionSource
from .session_config import SCENE_FIELDS

ModelFactory = Callable[..., Any]
HeadsetSourceFactory = Callable[..., Any]

EXPANDABLE_SEGMENTS = "expandable_segments:True"
_ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def configure_cuda_allocator() -> bool:
    """Let PyTorch grow CUDA memory in expandable segments, unless configured.

    With fixed-size segments about 1.4 GiB stayed reserved but unused at steady
    state: blocks freed by model loading and warmup could neither merge nor go
    back to the driver. Expandable segments map memory in pages and grow one
    segment per stream, so reserved memory stays close to allocated memory.
    Only where tensors live changes, not what is computed.

    A ``PYTORCH_CUDA_ALLOC_CONF`` (or ``PYTORCH_ALLOC_CONF``) set by the user
    wins. Otherwise the setting is applied to this process's allocator, which
    also works after CUDA is initialised; segments that already exist keep
    their kind.

    Returns:
        Whether expandable segments were enabled here.
    """
    if any(os.environ.get(name) for name in _ALLOC_CONF_VARS):
        return False
    if not torch.cuda.is_available():
        return False
    torch._C._accelerator_setAllocatorSettings(EXPANDABLE_SEGMENTS)
    return True


def warmup_capture_chunks(
    *, warmup_iters: int, denoise_steps: int, refresh_graph_per_chunk: bool
) -> int:
    """Return how many steady chunks one prewarm must run to capture every graph.

    A CUDA graph wrapper runs ``warmup_iters`` eager calls and captures on the
    next one. The denoise graph is called ``denoise_steps`` times per chunk, but
    the FadeMem steady refresh graph only once, so with 3 or more denoise steps
    the refresh graph sets the count.

    Args:
        warmup_iters: Eager calls each wrapper makes before capturing.
        denoise_steps: Denoise graph calls per chunk.
        refresh_graph_per_chunk: Whether a once-per-chunk refresh graph must
            also be captured in the steady phase.

    Returns:
        Steady chunks after the fill phase, at least one.
    """
    calls = max(0, int(warmup_iters)) + 1
    chunks = math.ceil(calls / max(1, int(denoise_steps)))
    if refresh_graph_per_chunk:
        chunks = max(chunks, calls)
    return chunks


class Cosmos3HDreamsSession(IFlashDreamsApplicationSession):
    """Generate video from one initial frame and selected action source."""

    def __init__(
        self,
        *,
        config: Cosmos3HDreamsConfig,
        model_factory: ModelFactory = Cosmos3HDreamsModel,
        headset_source_factory: HeadsetSourceFactory = HeadsetActionSource,
    ) -> None:
        self.config = config
        self._model_factory = model_factory
        self._headset_source_factory = headset_source_factory
        self._model: Any | None = None
        self._scene: Any | None = None
        self._rollout: Any | None = None
        self._headset_source: Any | None = None
        self._rows_consumed = 0
        self._step_index = 0
        self._profile_steps: list[dict[str, float]] = []
        self._vram_stop = threading.Event()
        self._vram_thread: threading.Thread | None = None
        self._vram_profile: dict[str, float] = {}
        self._closed = False
        self.model_status = "Loading model…"
        self.text_cache: dict[str, Any] | None = None
        """Text cache outcome of the last init (``/healthz``); None before it finishes.

        ``status`` is ``off``, ``hit``, ``built`` (a miss, saved) or ``built-unsaved``
        (a miss, the save failed). ``rehashed`` is True when the checkpoint's size
        or mtime changed so it was hashed again (a hit stays a hit if the bytes
        match). ``hash_s`` is the checkpoint hash time (None when the stored id
        was reused), ``load_s`` the model load time. Set once warm-up succeeds.
        """
        self.attention: dict[str, str] | None = None
        """Mask-free attention kernel picked by the last init (``/healthz``):
        ``{"kernel": "default" | "sage", "reason": ...}``."""

    @property
    def actions_per_step(self) -> int:
        """Return action rows consumed by one full model step."""
        return self.config.chunk_size * 4

    def init(self) -> None:
        """Load the checkpoint and prepare the selected rollout."""
        if self._closed:
            raise RuntimeError("Cannot initialize a closed Cosmos3-H-Dreams session.")
        if self._model is not None:
            return
        print(f"[cosmos3-h-dreams] {self.model_status}", flush=True)
        if torch.device(self.config.device).type == "cuda":
            configure_cuda_allocator()
        if self.config.compile_denoise:
            os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
            os.environ.setdefault(
                "TORCHINDUCTOR_CACHE_DIR",
                str(Path.home() / ".cache" / "flashdreams" / "torchinductor"),
            )
        lookup = text_meta = text_kv = None
        if self.config.text_cache is not None:
            lookup = text_kv_cache.resolve_meta(
                checkpoint=self.config.checkpoint,
                prompt=self.config.prompt,
                height=self.config.height,
                width=self.config.width,
                cache=self.config.text_cache,
            )
            text_meta = lookup.meta
            text_kv = text_kv_cache.load_text_kv(self.config.text_cache, text_meta)
        choice = select_attention_kernel(self.config.attention_kernel, self.config.device)
        self.attention = {"kernel": choice.kernel, "reason": choice.reason}
        print(
            f"[cosmos3-h-dreams] attention: {choice.kernel} ({choice.reason})",
            flush=True,
        )
        load_started = time.perf_counter()
        model = self._model_factory(
            checkpoint=self.config.checkpoint,
            action_domain_id=self.config.domain_id,
            device=self.config.device,
            height=self.config.height,
            width=self.config.width,
            seed=self.config.seed,
            chunk_size=self.config.chunk_size,
            denoise_steps=self.config.denoise_steps,
            decoder=self.config.decoder,
            credential_path=self.config.credential_path,
            window_frames=self.config.window_frames,
            cache_mode=self.config.cache_mode,
            fademem_slots=self.config.fademem_slots,
            fademem_beta=self.config.fademem_beta,
            fademem_merge_selection=self.config.fademem_merge_selection,
            overlap_cache_update=self.config.overlap_cache_update,
            compile_denoise=self.config.compile_denoise,
            compile_denoise_mode=self.config.compile_denoise_mode,
            cuda_graph_denoise=self.config.cuda_graph_denoise,
            cuda_graph_warmup_iters=self.config.cuda_graph_warmup_iters,
            steady_refresh_sdpa=self.config.steady_refresh_sdpa,
            profile=self.config.profile,
            offload_understanding=self.config.offload_understanding,
            fp8_generation=self.config.fp8_generation,
            fp8_scope=self.config.fp8_scope,
            fp8_gemm=self.config.fp8_gemm,
            skip_understanding=text_kv is not None,
        )
        load_s = time.perf_counter() - load_started
        scene = prepare_scene(self.config, model)
        status = "off"
        if lookup is not None:
            model.text_kv, status = text_kv_cache.ensure_text_kv(
                model.net,
                scene.text_ids,
                cached=text_kv,
                meta=lookup.meta,
                path=self.config.text_cache,
                native_gqa_attention=model.native_gqa_attention,
            )
        text_cache = {
            "status": status,
            "rehashed": lookup is not None and lookup.rehashed,
            "path": None if lookup is None else str(self.config.text_cache),
            "hash_s": None if lookup is None else lookup.hash_s,
            "load_s": load_s,
        }
        rollout = model.new_session(scene)
        if (
            self.config.actions is not None
            and scene.replay_actions.shape[0] % self.actions_per_step
            and not rollout.accepts_partial_chunks
        ):
            raise ValueError(
                "block cache requires the action trajectory to contain only full chunks"
            )
        self._model = model
        self._scene = scene
        self._rollout = rollout
        self._warmup_rollout(rollout)
        self.text_cache = text_cache
        if self.config.actions is None:
            headset_source = self._headset_source_factory(
                fps=self.config.fps,
                normalizer=self.config.normalizer,
                transport=self.config.headset_transport,
                relay_url=self.config.headset_relay_url,
                port=self.config.headset_port,
                startup_timeout_s=self.config.headset_startup_timeout_s,
                manual_xr=self.config.headset_manual_xr,
                jaw_open=self.config.headset_jaw_open,
                jaw_closed=self.config.headset_jaw_closed,
            )
            headset_source.start()
            self._headset_source = headset_source

    def _warmup_rollout(self, rollout: Any) -> None:
        """Prepare compiled graphs before exposing a new rollout to live input."""
        if self.config.compile_denoise:
            self.model_status = "Compiling model…"
            print(f"[cosmos3-h-dreams] {self.model_status}", flush=True)
            capture_steps = warmup_capture_chunks(
                warmup_iters=self.config.cuda_graph_warmup_iters,
                denoise_steps=self.config.denoise_steps,
                refresh_graph_per_chunk=(
                    getattr(rollout, "_refresh_steady_graph", None) is not None
                ),
            )
            warmup_started = time.perf_counter()
            for _ in range(capture_steps):
                timings = rollout.prewarm_rollout(
                    rollout._compile_start_step + capture_steps, profile=True
                )
                if not self.config.cuda_graph_denoise:
                    break
                if (
                    timings[-1]["cuda_graph_captured"]
                    and timings[-1].get("cuda_graph_refresh_steady_captured", 1.0)
                    and (
                        self.config.cache_mode != "fademem"
                        or rollout._compile_start_step == 0
                        or timings[-1]["cuda_graph_fill_captured"]
                    )
                ):
                    break
            else:
                raise RuntimeError("Model warmup did not capture the denoising graphs")
            if torch.device(self.config.device).type == "cuda":
                torch.cuda.synchronize(self.config.device)
            print(
                f"[cosmos3-h-dreams] warmup completed in "
                f"{time.perf_counter() - warmup_started:.1f}s",
                flush=True,
            )
            self.model_status = "Model compiled"
        else:
            self.model_status = "Model ready"
        print(f"[cosmos3-h-dreams] {self.model_status}", flush=True)

    def session_info(self) -> SessionInfo:
        """Return the output contract for the selected action source."""
        scene = self._require_scene()
        return SessionInfo(
            output_layout="tchw",
            steady_output_frame_count=self.actions_per_step,
            frames_per_second=scene.fps,
            video_width=scene.content_width,
            video_height=scene.content_height,
            metadata={
                "model_id": "cosmos3-h-dreams",
                "action_rows": int(scene.replay_actions.shape[0]),
                "action_source": (
                    "replay" if self.config.actions is not None else "headset"
                ),
            },
        )

    def next_step_requirements(self) -> StepRequirements | None:
        """Return the next action window."""
        if self._closed:
            return None
        if self.config.actions is None:
            calibrated, _status, _fraction = self._calibration_state()
            if not calibrated:
                return StepRequirements(
                    step_index=self._step_index,
                    input_frame_count=1,
                    steady_output_frame_count=1,
                    metadata={"calibration": True},
                )
            return StepRequirements(
                step_index=self._step_index,
                input_frame_count=self.actions_per_step,
                steady_output_frame_count=self.actions_per_step,
            )
        scene = self._require_scene()
        remaining = int(scene.replay_actions.shape[0]) - self._rows_consumed
        rows = min(self.actions_per_step, remaining)
        rows -= rows % 4
        if rows <= 0:
            return None
        return StepRequirements(
            step_index=self._step_index,
            input_frame_count=rows,
            steady_output_frame_count=rows,
        )

    def step(self, inputs: CanonicalInputWindow) -> StepResult:
        """Generate the next chunk from the selected action source."""
        if inputs.values:
            raise ValueError("Cosmos3-H-Dreams consumes no host-provided inputs.")
        if self._headset_source is not None and self._headset_source.take_restart_request():
            self._restart_from_initial_frame(reset_step_index=False)
            _ready, status, fraction = self._calibration_state()
            return self._calibration_result(status, fraction)
        requirements = self.next_step_requirements()
        if requirements is None:
            raise RuntimeError("Cosmos3-H-Dreams has no action rows remaining.")
        scene = self._require_scene()
        _calibrated, status, fraction = self._calibration_state()
        if self._headset_source is not None and bool(
            requirements.metadata.get("calibration")
        ):
            return self._calibration_result(status, fraction)
        model, rollout = self._require_runtime()
        rows = requirements.input_frame_count
        input_started = time.perf_counter()
        if self._headset_source is None:
            actions = scene.replay_actions[
                self._rows_consumed : self._rows_consumed + rows
            ]
        else:
            try:
                actions = self._headset_source.next_actions(rows)
            except GenerationRestartRequested:
                self._restart_from_initial_frame(reset_step_index=False)
                _ready, status, fraction = self._calibration_state()
                return self._calibration_result(status, fraction)

        self._start_vram_profile()
        started = time.perf_counter()
        latents = rollout.step(actions, profile=self.config.profile)
        model_finished = time.perf_counter()
        video = model.decode_tensor(torch.cat(latents, dim=1))
        if self.config.profile and video.is_cuda:
            torch.cuda.current_stream(video.device).synchronize()
        finished = time.perf_counter()
        metrics = {
            **rollout.last_timing,
            "input_s": started - input_started,
            "model_wall_s": model_finished - started,
            "decode_s": finished - model_finished,
            "total_s": finished - started,
        }
        if self.config.profile:
            self._profile_steps.append(
                {name: float(value) for name, value in metrics.items()}
            )
        result = StepResult.from_video_chunk(
            step_index=self._step_index,
            video_chunk=video,
            layout="tchw",
            metadata={
                "action_start": self._rows_consumed,
                "action_rows": rows,
                "action_capture_ns": getattr(
                    self._headset_source, "last_action_capture_ns", None
                ),
                "action_first_capture_ns": getattr(
                    self._headset_source, "last_action_first_capture_ns", None
                ),
            },
            metrics=metrics,
        )
        self._rows_consumed += rows
        self._step_index += 1
        return result

    def _calibration_state(self) -> tuple[bool, str, float]:
        source = self._headset_source
        if source is None:
            return True, "", 1.0
        return source.calibration_state()

    def _calibration_result(self, status: str, fraction: float) -> StepResult:
        scene = self._require_scene()
        result = StepResult.from_video_chunk(
            step_index=self._step_index,
            video_chunk=_calibration_screen(scene.init_rgb, status, fraction),
            layout="tchw",
            metadata={
                "action_rows": 0,
                "calibration": status,
                "calibration_fraction": fraction,
            },
        )
        self._step_index += 1
        return result

    def switch_scene(self, config: Cosmos3HDreamsConfig) -> None:
        """Replace conditioning and restart the rollout with the loaded model."""
        model, rollout = self._require_runtime()
        if self._closed:
            raise RuntimeError("Cannot switch a closed Cosmos3-H-Dreams session.")
        if any(
            getattr(config, field.name) != getattr(self.config, field.name)
            for field in fields(Cosmos3HDreamsConfig)
            if field.name not in SCENE_FIELDS
        ):
            raise ValueError("Scene switches require shared runtime settings")
        domains = int(model.net.config.num_embodiment_domains)
        if not 0 <= config.domain_id < domains:
            raise ValueError(f"domain_id must be in [0, {domains})")
        scene = prepare_scene(config, model)
        saved = getattr(model, "text_kv", None)
        if saved is not None and not torch.equal(saved.text_ids, scene.text_ids.cpu()):
            raise ValueError(
                "scene prompt differs from the text cache; restart without "
                "--text-cache or rebuild it"
            )
        if (
            config.actions is not None
            and scene.replay_actions.shape[0] % self.actions_per_step
            and not rollout.accepts_partial_chunks
        ):
            raise ValueError("block cache requires only full action chunks")
        self._stop_vram_profile()
        rollout.wait_for_pending_cache_update(block=True)
        previous_domain = model.action_domain_id
        previous_status = self.model_status
        self.model_status = "Preparing scene…"
        model.action_domain_id = config.domain_id
        try:
            new_rollout = model.new_session(scene)
            self._warmup_rollout(new_rollout)
        except Exception:
            model.action_domain_id = previous_domain
            self._restart_from_initial_frame(reset_step_index=False)
            self.model_status = previous_status
            raise
        self.config = config
        self._scene = scene
        self._rollout = new_rollout
        self._rows_consumed = 0
        self._profile_steps.clear()
        self._vram_profile.clear()
        if self._headset_source is not None:
            self._headset_source.set_normalizer(config.normalizer)
        # A rollout holds reference cycles (its compiled denoise closures and
        # FadeMem keep bound methods of it), so dropping the last reference frees
        # nothing: each switch kept about 1 GiB of caches and CUDA graphs alive
        # until a full collection, and their segments reserved until empty_cache.
        # cuBLAS workspaces are not cleared here (unlike release_cuda_memory):
        # the new rollout's captured graphs may use them.
        del rollout
        gc.collect()
        if torch.device(self.config.device).type == "cuda":
            torch.cuda.empty_cache()

    def reset(self) -> None:
        """Restart generation from the original image without reloading weights."""
        self._restart_from_initial_frame(reset_step_index=True)

    def _restart_from_initial_frame(self, *, reset_step_index: bool) -> None:
        if self._closed:
            raise RuntimeError("Cannot reset a closed Cosmos3-H-Dreams session.")
        model, rollout = self._require_runtime()
        scene = self._require_scene()
        self._stop_vram_profile()
        rollout.wait_for_pending_cache_update(block=True)
        if self.config.compile_denoise:
            model.release_session(scene, rollout)
        self._rollout = model.new_session(scene)
        self._rows_consumed = 0
        self._profile_steps.clear()
        self._vram_profile.clear()
        if reset_step_index:
            self._step_index = 0
        if self._headset_source is not None:
            self._headset_source.reset_calibration()

    def close(self) -> None:
        """Finish pending cache work and release the loaded runtime."""
        if self._closed:
            return
        self._closed = True
        if self._rollout is not None:
            self._rollout.wait_for_pending_cache_update(block=True)
        self._stop_vram_profile()
        if self._headset_source is not None:
            self._headset_source.close()
        if self.config.profile:
            self._print_profile_summary()
        self._rollout = None
        self._headset_source = None
        self._scene = None
        self._model = None

    def _require_scene(self) -> Any:
        if self._scene is None:
            raise RuntimeError("Cosmos3-H-Dreams session must be initialized first.")
        return self._scene

    def _require_runtime(self) -> tuple[Any, Any]:
        if self._model is None or self._rollout is None:
            raise RuntimeError("Cosmos3-H-Dreams session must be initialized first.")
        return self._model, self._rollout

    def _start_vram_profile(self) -> None:
        """Sample allocator memory from the first inference chunk until close/reset."""
        device = torch.device(self.config.device)
        if not self.config.profile or device.type != "cuda" or self._vram_thread:
            return
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.synchronize(device)
        # Peaks since the previous reset: checkpoint load, text prefill and warmup
        # on the first run, which the per-chunk sampling below never sees.
        startup = {
            "startup_allocated_peak": torch.cuda.max_memory_allocated(device),
            "startup_reserved_peak": torch.cuda.max_memory_reserved(device),
        }
        torch.cuda.reset_peak_memory_stats(device)
        self._vram_profile.clear()
        self._vram_profile.update(startup)
        self._vram_stop.clear()
        started = time.perf_counter()
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)

        def sample() -> None:
            previous = started
            last_allocated, last_reserved = allocated, reserved
            allocated_integral = reserved_integral = 0.0
            while True:
                stopped = self._vram_stop.wait(0.01)
                now = time.perf_counter()
                elapsed = now - previous
                allocated_integral += last_allocated * elapsed
                reserved_integral += last_reserved * elapsed
                previous = now
                last_allocated = torch.cuda.memory_allocated(device)
                last_reserved = torch.cuda.memory_reserved(device)
                if stopped:
                    duration = now - started
                    self._vram_profile.update(
                        allocated_average=allocated_integral / duration
                        if duration
                        else allocated,
                        reserved_average=reserved_integral / duration
                        if duration
                        else reserved,
                        allocated_peak=torch.cuda.max_memory_allocated(device),
                        reserved_peak=torch.cuda.max_memory_reserved(device),
                    )
                    return

        # ponytail: 10 ms polling can miss short allocations in the average;
        # use allocator history for finer attribution. Peaks use allocator counters.
        self._vram_thread = threading.Thread(target=sample, daemon=True)
        self._vram_thread.start()

    def _stop_vram_profile(self) -> None:
        """Finish sampling before releasing or replacing the rollout."""
        if self._vram_thread is not None:
            self._vram_stop.set()
            self._vram_thread.join()
            self._vram_thread = None

    def _print_profile_summary(self) -> None:
        """Print first-chunk and steady-state timings collected for this rollout."""
        if not self._profile_steps:
            return
        first = self._profile_steps[0]
        skip = 1
        if self.config.cache_mode == "fademem":
            target = max(6 * self.config.chunk_size, self.config.fademem_slots)
            skip = -(-target // self.config.chunk_size)
        steady = self._profile_steps[skip:]
        print("\n=== Cosmos3-H-Dreams profile (model-side; excludes MP4 encoding) ===")
        print(
            f"steps: {len(self._profile_steps)} | actions: {self._rows_consumed} | "
            f"chunk size: {self.config.chunk_size} latent frames"
        )
        print(
            f"cache: {self.config.cache_mode} | "
            f"overlap: {self.config.overlap_cache_update} | "
            f"compile: {self.config.compile_denoise} "
            f"({self.config.compile_denoise_mode}) | "
            f"CUDA graph: {self.config.cuda_graph_denoise}"
        )
        print(f"first total: {_seconds(first.get('total_s'))}")
        if self._vram_profile:
            print(
                "VRAM before first chunk (load, text prefill, warmup): peak allocated "
                f"{self._vram_profile['startup_allocated_peak'] / 2**20:.1f} MiB | "
                f"peak reserved {self._vram_profile['startup_reserved_peak'] / 2**20:.1f} MiB"
            )
            print(
                "VRAM (PyTorch allocator; first chunk through run end; average sampled every 10 ms):"
            )
            for kind in ("allocated", "reserved"):
                print(
                    f"VRAM {kind}: peak {self._vram_profile[f'{kind}_peak'] / 2**20:.1f} MiB | "
                    f"average {self._vram_profile[f'{kind}_average'] / 2**20:.1f} MiB"
                )
        if "kv_cache_bytes" in first:
            for index, step in enumerate(self._profile_steps, start=1):
                print(
                    f"step {index}: total {_seconds(step.get('total_s'))} | "
                    f"KV cache {step['kv_cache_bytes'] / 2**20:.1f} MiB"
                )
            print(
                f"KV cache: first {first['kv_cache_bytes'] / 2**20:.1f} MiB | "
                f"last {self._profile_steps[-1]['kv_cache_bytes'] / 2**20:.1f} MiB | "
                f"peak {max(step['kv_cache_bytes'] for step in self._profile_steps) / 2**20:.1f} MiB"
            )
        if self.config.cache_mode == "fademem":
            print(
                f"FadeMem: slots={self.config.fademem_slots} beta={self.config.fademem_beta} "
                f"selection={self.config.fademem_merge_selection}; "
                f"excluded {skip} fill/remapping chunks"
            )
        if not steady:
            return
        for name in (
            "model_wall_s",
            "setup_s",
            "denoise_s",
            "cache_seed_prune_s",
            "cache_update_wait_s",
            "decode_s",
            "total_s",
        ):
            values = [step[name] for step in steady if name in step]
            if values:
                print(
                    f"steady {name}: median {_seconds(median(values))} | "
                    f"p90 {_seconds(_p90(values))} ({len(values)} steps)"
                )


def _p90(values: list[float]) -> float:
    """Return the nearest-rank 90th percentile for a non-empty sample."""
    ordered = sorted(values)
    return ordered[(9 * len(ordered) - 1) // 10]


def _seconds(value: float | None) -> str:
    """Format an optional duration for the compact terminal summary."""
    return "n/a" if value is None else f"{value * 1_000:.0f} ms"


def _calibration_screen(
    initial_rgb: np.ndarray,
    status: str,
    fraction: float,
) -> torch.Tensor:
    """Render the initial scene with concise headset calibration guidance."""
    image = Image.fromarray(np.asarray(initial_rgb, dtype=np.uint8), mode="RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    margin = max(2, min(20, min(width, height) // 16))
    panel_height = min(max(104, height // 3), height - 2 * margin)
    panel = (margin, height - panel_height - margin, width - margin, height - margin)
    draw.rounded_rectangle(panel, radius=12, fill=(8, 12, 20, 220))
    font = ImageFont.load_default()
    padding = max(2, min(18, width // 32))
    draw.text(
        (panel[0] + padding, panel[1] + padding),
        "CONTROLLER CALIBRATION",
        fill=(255, 255, 255, 255),
        font=font,
    )
    draw.text(
        (panel[0] + padding, panel[1] + padding + 24),
        status,
        fill=(220, 226, 235, 255),
        font=font,
    )
    bar_height = max(2, min(14, height // 24))
    bar = (
        panel[0] + padding,
        panel[3] - padding - bar_height,
        panel[2] - padding,
        panel[3] - padding,
    )
    draw.rounded_rectangle(bar, radius=7, fill=(50, 58, 72, 255))
    filled = bar[0] + round((bar[2] - bar[0]) * float(np.clip(fraction, 0.0, 1.0)))
    if filled > bar[0]:
        draw.rounded_rectangle(
            (bar[0], bar[1], filled, bar[3]),
            radius=7,
            fill=(74, 144, 245, 255),
        )
    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    array = np.asarray(composed, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
