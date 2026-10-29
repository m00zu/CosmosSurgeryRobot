# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video output targets for the runtime API."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner_io import DEFAULT_RUNNER_INSTALL_HINT
from flashdreams.infra.video_output import Mp4ResultWriter
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.types import StepResult

VideoWriter = Callable[..., Path]


@dataclass(slots=True)
class Mp4VideoOutputTarget:
    """Write layout-aware runtime step results to one MP4 artifact."""

    output_path: Path
    fps: int | float
    output_layout: VideoTensorLayout = "bvtchw"
    writer: VideoWriter | None = field(default=None, repr=False)
    """Whole-video writer; ``None`` streams each chunk to FFmpeg as it arrives."""
    install_hint: str = DEFAULT_RUNNER_INSTALL_HINT
    move_to_cpu: bool = True
    enabled: bool = True
    _opened: bool = field(default=False, init=False, repr=False)
    _mp4: Mp4ResultWriter | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def closed(self) -> bool:
        return not self._opened

    def open(self) -> None:
        self._mp4 = Mp4ResultWriter(
            self.output_path,
            fps=self.fps,
            output_layout=self.output_layout,
            writer=self.writer,
            install_hint=self.install_hint,
            enabled=self.enabled,
            move_to_cpu=self.move_to_cpu,
        )
        self._opened = True

    def write(self, result: StepResult) -> None:
        if not self._opened or self._mp4 is None:
            raise RuntimeError("Cannot write to a closed output target.")
        if result.layout is None:
            raise TypeError(
                "Mp4VideoOutputTarget requires a video StepResult with layout."
            )
        if result.layout != self.output_layout:
            raise ValueError(
                "Mp4VideoOutputTarget received layout "
                f"{result.layout!r}; expected {self.output_layout!r}."
            )
        self._mp4.add(result)

    def close(self) -> Sequence[OutputArtifact]:
        mp4 = self._mp4
        self._mp4 = None
        self._opened = False
        written = None if mp4 is None else mp4.close()
        if mp4 is None or written is None:
            return ()
        path, shape = written
        return (
            OutputArtifact(
                kind="video/mp4",
                uri=str(path),
                metadata={
                    "fps": self.fps,
                    "source_layout": self.output_layout,
                    "shape": shape,
                    "stats_history": tuple(mp4.collector.stats_history),
                },
            ),
        )


__all__ = ["Mp4VideoOutputTarget"]
