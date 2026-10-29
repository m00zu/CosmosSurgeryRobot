<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0 -->

# Cholecystectomy starting frames

Endoscope frames from CMR Surgical Versius cholecystectomy recordings in the
[Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
dataset (`Surgical/cmr_surgical/cholecystectomy`), by CMR and the
Open-H-Embodiment dataset contributors, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They were resized to
1280×720 and saved as JPEG; nothing else was changed.

`ep<episode>_<seconds>s.jpg` names the source clip and time, so
`ep000400_10s.jpg` is `videos/chunk-000/observation.images.endoscope/episode_000400.mp4`
at 10 s. Each frame shows the grasper on the left arm and the energy hook on the
right arm (the on-screen instrument icons were checked), with both instruments
in view and no third instrument.

To add one:

```bash
ffmpeg -ss 10 -i https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment/resolve/main/Surgical/cmr_surgical/cholecystectomy/videos/chunk-000/observation.images.endoscope/episode_000400.mp4 \
  -frames:v 1 -vf scale=1280:720 -q:v 2 ep000400_10s.jpg
```
