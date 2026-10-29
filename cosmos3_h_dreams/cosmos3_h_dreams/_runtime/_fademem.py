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

"""Cosmos action/vision mRoPE adapter for clean FadeMem history."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from flashdreams.core.attention.fademem import FadeMemKVCache
from flashdreams.core.attention.kvcache import LayerKV
from torch import Tensor

from cosmos3_h_dreams._runtime._rope import mrope_cos_sin
from cosmos3_h_dreams._runtime._temporal_rope import rotate_half_reference

if TYPE_CHECKING:
    from cosmos3_h_dreams._runtime._spec import Cosmos3HDreamsSpec


class CosmosFadeMem:
    """Own fixed text context, phase-free temporal memory, and attention views."""

    def __init__(
        self,
        cache: Sequence[dict[str, Tensor]],
        *,
        n_text: int,
        capacity: int,
        beta: float,
        merge_selection: str = "age",
        seed_latent: Tensor | None = None,
        chunk_frames: int,
        positions: Callable[[int, int], Tensor],
        spec: Cosmos3HDreamsSpec,
    ) -> None:
        """Seed memory from text plus one clean frame in the normal Cosmos cache."""
        self.chunk_frames = chunk_frames
        self.positions: Callable[[int, int], Tensor] = positions
        self.spec = spec
        self.text = [
            (layer["k"][:, :, :n_text].clone(), layer["v"][:, :, :n_text].clone())
            for layer in cache
        ]
        self.seed = [
            (
                self.rotate_temporal(
                    layer["k"][:, :, n_text:], positions(0, 1), inverse=True
                ),
                layer["v"][:, :, n_text:].clone(),
            )
            for layer in cache
        ]
        self.seed_descriptor = (
            seed_latent.detach().reshape(1, -1).float().clone()
            if merge_selection == "cosine" and seed_latent is not None
            else None
        )
        self.memory = FadeMemKVCache(
            self.seed,
            capacity=capacity,
            beta=beta,
            merge_selection=merge_selection,
            descriptors=self.seed_descriptor,
        )
        device = self.seed[0][0].device
        if device.type == "cuda":
            # Only reset() reads the seed again: keep it in pinned host memory.
            self.seed = [
                (key.cpu().pin_memory(), value.cpu().pin_memory())
                for key, value in self.seed
            ]
        # Full-capacity views of the phase-free store, at fixed addresses.
        # Compiled passes read them and apply this chunk's temporal phase per
        # layer from ``history_cos``/``history_sin`` (see ``_temporal_rope``),
        # so no rotated copy of every layer's keys is kept.
        self.stored = self.memory.layers(full_capacity=True)
        rows = capacity * self.memory.tokens_per_frame
        self.history_cos = torch.zeros(
            (rows, self.seed[0][0].shape[-1]),
            dtype=torch.float32,
            device=device,
        )
        self.history_sin = torch.zeros_like(self.history_cos)
        self._history_cos_sin: tuple[Tensor, Tensor] | None = None
        self.mask = torch.zeros(
            n_text + capacity * self.memory.tokens_per_frame,
            dtype=torch.bool,
            device=device,
        )
        self.valid_length = torch.zeros((), dtype=torch.int32, device=self.mask.device)

    def temporal_cos_sin(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Temporal-only mRoPE tables, shared by every layer of one rotation."""
        temporal = positions.clone()
        temporal[1:] = 0
        llm = self.spec.llm
        return mrope_cos_sin(temporal, llm.head_dim, llm.rope_theta, llm.mrope_section)

    def rotate_temporal(
        self,
        key: Tensor,
        positions: Tensor | None,
        *,
        inverse: bool = False,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Remove/reapply only temporal mRoPE, including fractional action times.

        Pass ``cos_sin`` from :meth:`temporal_cos_sin` to reuse the tables
        across layers; ``positions`` is then not read.
        """
        if cos_sin is None:
            if positions is None:
                raise ValueError("rotate_temporal needs positions or cos_sin")
            cos_sin = self.temporal_cos_sin(positions)
        cos, sin = cos_sin
        return rotate_half_reference(key, cos, sin, -1 if inverse else 1)

    @torch.no_grad()
    def prepare(
        self, start: int, frames: int, *, fixed_shape: bool = False
    ) -> tuple[Tensor, list[LayerKV]]:
        """Prepare one shared positional layout for history and current Q/K.

        With ``fixed_shape`` the history is :attr:`stored`, still phase-free:
        the consumer rotates it with :attr:`history_cos`/:attr:`history_sin`,
        which are zero past the valid entries. Otherwise it is the rotated
        valid prefix.
        """
        centers, current = self.memory.attention_positions(start, self.chunk_frames)
        positions = torch.cat([self.positions(center, 1) for center in centers], dim=1)
        valid = self.memory.count * self.memory.tokens_per_frame
        cos, sin = self._history_cos_sin = self.temporal_cos_sin(positions)
        if fixed_shape:
            self.history_cos[:valid].copy_(cos)
            self.history_sin[:valid].copy_(sin)
            if valid < self.history_cos.shape[0]:
                self.history_cos[valid:].zero_()
                self.history_sin[valid:].zero_()
        self.mask.zero_()
        self.mask[: self.text[0][0].shape[2] + valid] = True
        self.valid_length.fill_(self.text[0][0].shape[2] + valid)
        history = self.stored if fixed_shape else self.history()
        return self.positions(current, frames), history

    def history(self) -> list[LayerKV]:
        """Return the rotated valid history prepared by :meth:`prepare`, eagerly."""
        if self._history_cos_sin is None:
            raise RuntimeError("FadeMem history requires prepare() first")
        return [
            (self.rotate_temporal(k, None, cos_sin=self._history_cos_sin), v)
            for k, v in self.memory.layers()
        ]

    @torch.no_grad()
    def commit(
        self,
        start: int,
        positions: Tensor,
        layers: Sequence[LayerKV],
        *,
        descriptors: Tensor | None = None,
        selection_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Commit only finalized clean keys, using their attention-time positions."""
        cos_sin = self.temporal_cos_sin(positions)
        self.memory.append(
            start,
            [
                (self.rotate_temporal(k, positions, inverse=True, cos_sin=cos_sin), v)
                for k, v in layers
            ],
            descriptors=descriptors,
            selection_stream=selection_stream,
        )

    def reset(self) -> None:
        """Restore the original seed while preserving allocated buffer addresses."""
        device = self.mask.device
        seed = [
            (key.to(device, non_blocking=True), value.to(device, non_blocking=True))
            for key, value in self.seed
        ]
        self.memory.reset(seed, descriptors=self.seed_descriptor)

    @property
    def storage_bytes(self) -> int:
        """Device bytes: memory, rotation tables, text, mask, and an on-device seed."""
        tensors = [self.history_cos, self.history_sin]
        tensors += [
            t
            for pairs in (self.seed, self.text)
            for pair in pairs
            for t in pair
            if t.device == self.mask.device
        ]
        tensors += [self.mask, self.valid_length]
        if self.seed_descriptor is not None:
            tensors.append(self.seed_descriptor)
        return self.memory.storage_bytes + sum(
            t.numel() * t.element_size() for t in tensors
        )
