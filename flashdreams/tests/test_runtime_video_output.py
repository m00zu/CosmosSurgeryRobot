# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.runtime import Mp4VideoOutputTarget, StepResult, TimeWindow

pytestmark = pytest.mark.ci_cpu


def test_mp4_video_output_target_rejects_non_video_payload(tmp_path: Path) -> None:
    target = Mp4VideoOutputTarget(output_path=tmp_path / "out.mp4", fps=30)
    target.open()

    with pytest.raises(TypeError, match="video StepResult"):
        target.write(StepResult(step_index=0, output="not-video"))


def test_mp4_video_output_target_writes_artifact_on_close(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: int | float,
        layout: str,
        install_hint: str,
    ) -> Path:
        del install_hint
        calls.append(
            {
                "shape": tuple(video.shape),
                "path": path,
                "fps": fps,
                "layout": layout,
            }
        )
        return path

    target = Mp4VideoOutputTarget(
        output_path=tmp_path / "omnidreams.mp4",
        fps=24,
        writer=fake_writer,
        move_to_cpu=False,
    )
    target.open()
    target.write(
        StepResult.from_video_chunk(
            step_index=3,
            video_chunk=torch.zeros((1, 2, 4, 3, 5, 6)),
            layout="bvtchw",
            metrics={"model_step_s": 0.5},
            output_window=TimeWindow(start_s=1.0, end_s=2.0),
        )
    )

    artifacts = target.close()

    assert len(artifacts) == 1
    assert artifacts[0].kind == "video/mp4"
    assert artifacts[0].uri == str(tmp_path / "omnidreams.mp4")
    assert calls == [
        {
            "shape": (4, 5, 12, 3),
            "path": tmp_path / "omnidreams.mp4",
            "fps": 24,
            "layout": "thwc",
        }
    ]
    assert artifacts[0].metadata["stats_history"] == (
        {
            "step_index": 3,
            "frames": 4,
            "model_step_s": 0.5,
            "output_start_s": 1.0,
            "output_end_s": 2.0,
        },
    )


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffmpeg on PATH")
def test_mp4_video_output_target_streams_chunks_without_keeping_them(
    tmp_path: Path,
) -> None:
    target = Mp4VideoOutputTarget(output_path=tmp_path / "out.mp4", fps=10)
    target.open()
    for step in range(3):
        target.write(
            StepResult.from_video_chunk(
                step_index=step,
                video_chunk=torch.zeros((1, 2, 4, 3, 8, 16)),
                layout="bvtchw",
            )
        )
        # Streamed to ffmpeg, so host memory does not grow with the run.
        assert target._mp4 is not None and not target._mp4.collector.has_video

    (artifact,) = target.close()

    assert artifact.metadata["shape"] == (1, 2, 12, 3, 8, 16)
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width,height", "-of", "csv=p=0",
         artifact.uri],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert frames == "32,8,12"
