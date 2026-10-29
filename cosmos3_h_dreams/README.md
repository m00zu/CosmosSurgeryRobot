<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3-H-Dreams setup and usage

Cosmos3-H-Dreams generates surgical video from an initial frame and instrument
actions. It supports offline replay and live headset browser sessions using
the included FlashDreams framework. Start with the repository
[Quick Start](../README.md#quick-start) for a single launch command.
For a step-by-step training-ground installation and user guide, see
[docs/training-ground-setup.md](../docs/training-ground-setup.md).

All commands below run from the repository root. The distribution name is
`flashdreams-cosmos3-h-dreams`, the Python import is `cosmos3_h_dreams`, and the
application name is `cosmos3-h-dreams`.

## Installation and assets

```bash
uv sync --package flashdreams-cosmos3-h-dreams --locked
```

Install system `ffmpeg` for MP4 output. See the root
[System Requirements](../README.md#system-requirements) for the GPU environment.
The commands use `--no-sync` after installation to keep the existing environment.

Supply a checkpoint and matching scene assets. For reference, example data is available in [example_data/scenes](../assets/example_data/scenes/).
The bundled image comes from `episode_004148` of the
CMR surgical (cholecystectomy) data in
[Open-H-Embodiment](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Open-H-Embodiment)
and retains its [CC-BY-4.0 license](../LICENSES/CC-BY-4.0.txt). See the
[example-data attribution](../THIRD-PARTY-NOTICES)
for contributor credit and format information.
The bundled `trajectory.npy` is an NVIDIA-generated example action trajectory
for this project, licensed under Apache-2.0. It is
separate from the dataset-derived image.

Asset paths resolve from the launch directory, including
when a session YAML lives in another directory. Supply the checkpoint through
`--checkpoint` or `runtime.checkpoint` in the session YAML; the CLI option
overrides the YAML value.

Replay actions must be a finite `float32` NumPy array shaped `[N, 28]`. Four action
rows correspond to one latent video frame. Use actions normalized for the
checkpoint, or supply `--normalizer /path/to/normalizer.npz` with 28-element
`offset` and `scale` arrays. JSON normalizers are not accepted. Do not normalize
an already-normalized replay twice. The runtime pads actions to the model's
64-dimensional input width.

The Wan VAE, tokenizer, and standard TAEHV decoder can fetch assets on first use.
The optional `taehv2super` decoder requires an accessible checkpoint specified by
`COSMOS3_H_DREAMS_TAEHV2SUPER_CHECKPOINT`.

## Inference presets

Select a model preset with `--config`. Presets supply inference settings; asset
paths and output destinations are configured separately. Explicit CLI options
override preset values. Definitions live in
[config.py](cosmos3_h_dreams/config.py).

| Preset | Decoder | Latent frames/chunk | Denoising steps | Cache |
| --- | --- | --- | --- | --- |
| `base` | Wan | 4 | 4 | Eager `cat` |
| `wan-wan-chunk4-block-compile` | Wan | 4 | 4 | Compiled block |
| `wan-taehv-chunk4-block-compile` | TAEHV | 4 | 4 | Compiled block |
| `wan-wan-chunk3-block-compile` | Wan | 3 | 4 | Compiled block |
| `wan-taehv-chunk3-block-compile` | TAEHV | 3 | 4 | Compiled block |
| `wan-wan-chunk2-block-compile` | Wan | 2 | 4 | Compiled block |
| `wan-taehv-chunk2-block-compile` | TAEHV | 2 | 4 | Compiled block |
| `wan-taehv-chunk2-fademem-compile` | TAEHV | 2 | 4 | Compiled FadeMem |
| `wan-taehv-chunk2-2steps-fademem-compile` | TAEHV | 2 | 2 | Compiled FadeMem |

All named presets use the Wan image encoder, a 480 × 832 canvas, domain 25,
and 10 Hz model timing. Compiled presets enable CUDA graphs and overlapping
cache updates. FadeMem presets select cosine merge ranking. Choose settings
that match the checkpoint's training conditions; other resolutions and domain
IDs require separate validation. Height and width must be divisible by 16.

The bundled [real_time_2steps.yaml](configs/real_time_2steps.yaml) selects the
last model preset and the independent `headset-base` transport preset. `headset-base`
uses controller acquisition at up to 30 Hz and MJPEG playback at 20 fps. Without
that headset preset, the server defaults to MJPEG at 30 display fps and acquisition
at the model rate. Changing display FPS only changes playback pacing; changing
model FPS changes temporal conditioning.

We recommend using the 4-step checkpoint (`c3-h-d_4-step_model_bf16.safetensors`) with the `wan-taehv-chunk2-fademem-compile` preset, and the 2-step checkpoint (`c3-h-d_2-step_model_bf16.safetensors`) with the `wan-taehv-chunk2-2steps-fademem-compile` preset. Mixing them may result in suboptimal visual quality.

## Offline replay

```bash
uv run --no-sync --package flashdreams-cosmos3-h-dreams \
  flashdreams-run cosmos3-h-dreams \
  --output mp4 --output-path replay.mp4 \
  --config wan-taehv-chunk2-block-compile \
  --checkpoint checkpoints/model.safetensors \
  --initial-frame assets/example_data/scenes/example_frame.png \
  --actions assets/example_data/scenes/trajectory.npy
```

Block cache and FadeMem require full chunks, including the final chunk: the action row
count must be divisible by `4 × chunk_size` (8 for chunk2, 12 for chunk3,
16 for chunk4). Use eager `cat` for a partial final chunk, for example with
`--config base`.

For available options:

```bash
uv run --no-sync --package flashdreams-cosmos3-h-dreams \
  flashdreams-run cosmos3-h-dreams --help
```

## Headset browser sessions

Generate `cert.pem` and `key.pem` as shown in the root
[Quick Start](../README.md#quick-start). Keep the key private. The certificate
must cover the IP or hostname used by the headset; regenerate it if that changes.
`--http` serves plain HTTP with no certificate (`--cert`/`--key` are then not
needed), for a trusted closed network: `/train`, `/viewer` and `/dashboard`
work over `http://`, while `/headset` needs HTTPS for WebXR except on
localhost. Anyone who can reach the port can use the server.

```bash
uv run --no-sync cosmos3-h-dreams-run headset \
  --session-config cosmos3_h_dreams/configs/real_time_2steps.yaml
```

The launcher lists the available application names when a name is unknown.
Use `cosmos3-h-dreams-run --help` for launcher help, or
`cosmos3-h-dreams-run headset --help` for headset options. After updating
an existing checkout, run the installation command again to install the launcher
before using `uv run --no-sync`.

The server binds to `0.0.0.0:8443` by default. Open
`https://<server-LAN-IP>:8443/headset` in the headset browser and select **Enter VR**.
The initial image is displayed during startup. Compiled configurations run
warmup before accepting live input; the page reports loading, compilation, and
readiness. The first startup with a cold compiler cache can take time.

Track both controllers, release both triggers, and hold still for calibration.
A stable pose is captured after 0.3 seconds within the configured movement
thresholds. One headset controls a session. A disconnect clears calibration;
reconnecting requires calibration again.

Open `https://<server-LAN-IP>:8443/viewer` on a PC to watch without taking control.
One PC viewer is supported. Both pages provide a scene gallery when multiple
initial frames are configured. Switching scenes keeps weights loaded and resets
generation history and calibration.

### Session YAML

```yaml
runtime:
  preset: wan-taehv-chunk2-2steps-fademem-compile
  checkpoint: data_local/checkpoints/model.safetensors

headset:
  preset: headset-base
  cert: cert.pem
  key: key.pem

scenes:
  - initial_frame: data_local/scenes/*.png
```

`runtime.preset` selects model settings and `headset.preset` selects transport/input
settings. Other options use Python field names with underscores. CLI options
override YAML; `--config` and `--headset-config` replace the corresponding preset
selection. Scene CLI overrides such as `--prompt` apply to every scene.

A single-image scene requires a unique `name` and `initial_frame`; its `label`
defaults to the name. Wildcards expand in sorted path order and use each filename
as its name and label. Empty matches and duplicate names fail validation.
Optional scene fields include `prompt`, `domain_id`, `normalizer`, and replay
`actions`; the headset server uses live controller input. Omitted scene options
inherit preset/default values. Checkpoint, timing, resolution, and cache settings
are shared across scenes. All relative paths, including certificate paths,
resolve from the launch directory.

### Controls

- Controller translation and rotation drive the corresponding instrument from
  the calibrated reference pose.
- Index triggers close the jaws when squeezed and reopen them when released.
  `--headset-jaw-open` and `--headset-jaw-closed` set model-space targets (defaults
  `+1` and `-2`).
- Grip buttons drive the energy channel. Each controller’s primary button supplies the primary-button channel.
- Hold both controllers’ primary buttons for one second to restart and recalibrate. The page also provides
  **Reset simulation**.

The optional `instruments` block names each instrument and lists its tools
(`jaws`, `energy`, both or neither); the bundled training configs use:

```yaml
instruments:
  left: {name: Grasper, tools: [jaws]}
  right: {name: Hook, tools: [energy]}
```

The server zeroes the trigger (jaws) or grip (energy) of an instrument without
that tool before the model, the safety guard and the recording see it. `/train`
labels the instruments by name, dims the controls the active one lacks and says
which instrument has the tool, and exercises that use a missing tool fail
validation at start-up. Without the block both instruments have both tools.

Left-controller action channels are translation `0:3`, rotation `3:9`, trigger
`9`, grip `10`, primary button `11`, and stick `12:14`. Right-controller channels
are translation `14:17`, rotation `17:23`, trigger `23`, grip `24`, primary button
`25`, and stick `26:28`.

### Video transport and remote access

Video uses MJPEG over HTTPS. Controller poses arrive over a same-origin
WebSocket.

MJPEG can carry video and controls through a single TCP tunnel. From a PC that
can reach a remote server:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L <PC-LAN-IP>:8443:127.0.0.1:8443 <user>@<server>
```

Allow inbound TCP 8443 on that PC, use a certificate covering `<PC-LAN-IP>`, and
open `https://<PC-LAN-IP>:8443/headset` on the headset. Slow viewers skip frames
rather than accumulating an unbounded queue.

## Training ground

The training ground lets a trainee drive two virtual instruments with mouse and
keyboard instead of a headset, stops generation on unsafe motion or energy use,
and records each attempt for review. Launch it with the bundled session YAML:

```bash
uv run --no-sync cosmos3-h-dreams-run headset \
  --session-config cosmos3_h_dreams/configs/training_ground.yaml
```

On a laptop RTX 5090 (24 GB), use [laptop_rtx5090.yaml](configs/laptop_rtx5090.yaml)
instead: the same set-up, starting in the Balanced [model mode](#model-modes).

Open `https://<server-LAN-IP>:8443/train` on a PC. The page takes the place of
the headset: it sends the same controller frames over `/ws` at 30 Hz, so only
one of `/train` and `/headset` can be connected at a time. `/viewer` still works
for a second screen.

The page shows the video, a **Controls** card beside it with the active
instrument, and live **GPU memory** and **Host RAM** graphs (last 5 minutes)
under it. Model mode, sensitivity, the scene gallery, reset and the status lines
are in **⚙ Settings**. While the model loads or switches, the video area says so;
problems (lost connection, a failed switch or recording) show in a notice under
the video.

Enter a trainee name, click **Start attempt**, then click the video to capture
the pointer (Esc releases it). Hold still briefly at the start; calibration
works as for the headset.

| Input | Action |
| --- | --- |
| Mouse move | Move the active instrument left/right and forward/back |
| Scroll wheel | Move up/down |
| Shift + mouse | Tilt (yaw, pitch) |
| Q / E | Wrist roll |
| Hold left mouse button | Close the jaws (instruments with `jaws`) |
| G | Lock or unlock the active instrument's jaws closed; locked jaws stay closed after Tab, and a left click also unlocks them |
| Hold F | Energy, grip channel (instruments with `energy`) |
| Tab | Switch the active instrument |
| R | Return the active instrument to its start pose |
| Backspace | Reset the simulation |
| Space | Pause / resume |
| H | Show or hide the Controls card |

The bindings are the `KEYS` table at the top of
[train.js](cosmos3_h_dreams/web/train.js). The **Sensitivity** slider in Settings sets how far
a full video width of mouse travel moves the instrument (default 10 cm).

### Meta Quest (VR)

`/train` also runs in Quest Browser on a Meta Quest: the controllers drive the
instruments, and safety stops, recordings, courses and demos work as with the
mouse. The Quest runs the page itself over Wi-Fi; no cable or PC link app is
needed.

**First time with the Quest**

1. Charge the headset and both controllers. Put the headset on and follow its
   prompts, including drawing the play-area boundary.
2. Hold each controller with your index finger on the **trigger** (front) and
   your middle finger on the **grip** (side).
3. Start the server with HTTPS (`--cert`/`--key`, see
   [Headset browser sessions](#headset-browser-sessions)), not `--http`:
   WebXR needs HTTPS.
4. Put the Quest on the same network as the server. In Quest Browser open
   `https://<server-LAN-IP>:8443/train`.
   - "Your connection is not private": the server was reached; choose
     **Advanced → Proceed** (the certificate is self-signed).
   - "This site can't be reached": a network problem. Check both are on the
     same network, that port 8443 is allowed through the server's firewall,
     and that the Wi-Fi does not isolate devices from each other.
   - A work-managed Quest may block Quest Browser or unknown sites; that is an
     IT setting.
5. On the page, enter the trainee name, choose an exercise or scene and click
   **Start attempt**, using the controller as a laser pointer. Then click
   **Enter VR**.
6. Leave VR with the Meta button; you are back on the page, paused (the report
   card appears there after an exercise).

In VR the video is a screen 1.2 m ahead, fixed in the room. Under it a status
strip shows recording, each instrument's jaws and energy, and the current
exercise step; a controls card sits to its left.

| Quest input | Action |
| --- | --- |
| Move / turn a hand | Move / turn that side's instrument (both at once) |
| Hold trigger | Close the jaws (instruments with `jaws`) |
| Hold grip | Energy (instruments with `energy`) |
| A (right) / X (left) | Return that instrument to its start position |
| Y (left) | Pause / resume; resume after a safety stop; play / pause a demo |
| Hold B (right) 1 s | Recentre the screen in front of you |
| Trigger during a demo | Take over from the demo |
| Meta button, or taking the headset off | Pause |

The instruments never jump to your hands. On entering VR, after a pause or
safety stop, after a demo, and when a controller regains tracking, the
instrument continues from where it is and follows your hand's movement from
there. After any of these, release the trigger and grip once before they act
again. A controller that loses tracking holds its instrument still with the
jaws open and energy off. Thumbsticks are not used on `/train`.

**Troubleshooting**

- **Enter VR is missing or reads "VR needs HTTPS"**: open the page over
  `https://` in Quest Browser.
- **Black screen**: the model may still be loading (see the status strip), or
  the video connection dropped; leave VR and reload the page.
- **Instruments do not move**: hold both controllers still with the triggers
  released for a moment; the server calibrates on a still pose.
- **A safety stop right after every resume**: report it with the dashboard
  recording; the clutch should prevent this.

### Safety stops

Before each chunk is generated, the server checks the controller frames against
the `safety:` limits in the session YAML. The checks apply only while `/train`
is connected; `/headset` has no incident display, so headset motion is never
stopped. Motion is measured over spans of at
least 0.1 s; an untracked controller or a gap above 0.25 s starts a fresh
measurement.

| Option | Rule | Default |
| --- | --- | --- |
| `max_speed_m_s` | Instrument speed | 0.35 m/s |
| `max_acceleration_m_s2` | Change in speed | 3 m/s² |
| `max_rotation_rad_s` | Angular rate | 4 rad/s |
| `max_energy_s` | One continuous energy activation | 10 s |
| `max_energy_dose_s`, `energy_dose_window_s` | Energy time in a sliding window | 14 s in 15 s |
| `max_energy_sweep_m` | Distance moved during one activation | 0.10 m |

On a violation the chunk is not generated, the video holds its last frame, and
the page shows each exceeded limit, for example
`Right instrument: speed 0.98 m/s, limit 0.35 m/s`. Generation stays held until
**Resume**, which continues from the last accepted pose; Space does not clear
an incident.

The defaults are placeholders. To suggest limits from recorded un-normalized
`[N, 28]` trajectories, print their 99th-percentile values:

```bash
uv run --no-sync python -m cosmos3_h_dreams.safety calibrate trajectory.npy \
  --fps 10 --translation-scale 10
```

After each checked window the server sends `/train` a `safety` message with
each rule's highest value over its limit (worst instrument), which the page
shows as a safety meter. When a ratio first reaches 0.8 without stopping, the
recording logs a `near_miss` event; that rule logs again only after its ratio
has dropped below 0.6.

### Recordings and dashboard

**Start** begins an attempt and **Stop** ends it. A disconnect or server stop
also ends it. A reset, scene switch, or mode switch ends it and starts a new
attempt for the same trainee. Each attempt is a folder under `recording.dir`
(default `recordings/`, relative to the launch directory):

```text
recordings/<YYYYMMDD-HHMMSS>_<trainee>/
  video.mp4         generated frames at model fps (needs ffmpeg)
  actions.npy       accepted model actions, [N, 28]
  controller.jsonl  controller frames of the generated chunks
  events.jsonl      start, pause, resume, incident (with blocked frames), energy on/off, end
  session.json      trainee, scene, mode, text_cache, seed, limits, times, summary, complete flag
```

`summary.duration_s` and every event's `t_s` are active seconds (real time,
excluding pauses and safety-stop holds), so they do not depend on GPU speed;
`summary.video_duration_s` and `t_video_s` are positions in the video (frames ÷
fps), which the dashboard timeline seeks by.

The folder name uses an ASCII slug of the trainee name (`trainee` if nothing is
left, as for a Chinese name); `session.json` keeps the name as typed. Recording
failures, such as missing ffmpeg or a full disk, never stop the simulation; the
attempt is then marked `complete: false` and the page shows `REC (no video)`
when the video is affected. A failure that cannot be recovered ends the attempt
and the page shows the error.

Open `https://<server-LAN-IP>:8443/dashboard` to list attempts, sort them, filter
by trainee, and replay one with its event timeline: incidents are red markers,
energy orange spans, and pause/resume/reset grey ticks. Clicking a marker seeks
to 3 s before the event. Delete attempts by removing their folders. Trainee and
exercise selectors filter the list; with both chosen, a progress chart shows the
score per attempt. Exercise attempts show their score (`70 + ?` until an
instructor reviews them), step markers on the timeline, the rubric with
instructor checkboxes, notes and **Save review**, and a **Save as demo** box.

### Training courses

With a `courses:` block in the session YAML, `/train` gets a **Courses** button
with guided exercises and the path guide; without it the page behaves as before
(no Courses button, no path guide). The bundled `training_ground.yaml` enables it:

```yaml
courses:
  dir: cosmos3_h_dreams/courses   # one YAML per exercise: <course-folder>/<NN>-<id>.yaml
  demos_dir: data_local/demos     # demos saved from the dashboard (the default; gitignored)
  scene: example_frame.png        # optional; the default is the first scene
```

Every exercise attempt and course demo runs on one scene, since exercise
targets are fixed positions that only fit one picture: the exercise's own
`scene:` (a scene name), or else `courses.scene`. Starting one
switches `/train` to that scene first; the server refuses an exercise start on
another scene and a scene change while an exercise attempt runs. Free practice
can use any scene.

An exercise file holds `id` (lower-case letters, digits and hyphens, unique
across courses), `course`, `order`, `title`, `brief`, an optional
`time_limit_s` (shown as the exercise's rough length), an optional `demo`,
`steps` and a `rubric`; see
[01-instrument-basics.yaml](courses/starter/01-instrument-basics.yaml).
A `demo_only: true` exercise has a demo and no steps or rubric.

A scene whose frame shows an instrument already holding tissue can say so with
`start_jaws_closed: [left]` (in the session YAML's scene entry). `/train` then
starts that instrument with its jaws locked closed (as with G), so the model
is not told to let go; G or a left click releases it, and Tab does not. A demo's
leading `jaws: close` keyframe likewise sets its first frame.

- **Demo:** `scripted:` keyframes, or `recorded: <name>` for
  `demos_dir/<name>.jsonl`. A keyframe has one action, `move_to: [x, y, z]`,
  `rotate_by: [yaw, pitch, roll]`, `jaws: close|open`, `energy: "on"|"off"`,
  `home: true` or `wait_s`, plus an optional `side` (`left`, `right` or `both`;
  default the active instrument), `over_s` (motion only, default 1 s) and
  `caption`. Positions are metres in the `/train` frame (x right, y up, z
  negative forward) and start at the home poses. Motion uses smoothstep easing;
  demos are compiled to 30 Hz frames.
- **Step checks** (completed in order): `reach: {side, center, radius_m}`,
  `active: left|right`, `jaws_closed: {side, hold_s}`, `jaws_open: {side}`,
  `energy: {side, hold_s}`,
  `home: {side}` (within 1 cm of home) and `moved: {side, min_m}` (path length
  since the step began).
- **Tools:** `jaws: close` keyframes and `jaws_closed` steps need an instrument
  with `jaws`, and `energy: "on"` keyframes and `energy` steps need one with
  `energy` (see the `instruments` block under Controls). Opening jaws or
  switching energy off is always allowed.
- **Rubric:** items `{id, text, points, auto}` with `auto` one of
  `no_safety_stops: true`, `max_time_s`, `max_path_deviation_m` (needs a demo),
  `max_energy_s`, `max_near_misses` or `all_steps: true`. An item without
  `auto` is scored by an instructor. When a rubric has an `all_steps` item and
  the attempt did not complete every step, its other automatic items score 0 and
  show as "not scored: steps incomplete", so an attempt stopped early scores
  only from instructor items. The grading is illustrative, not a clinical
  standard.

The server validates every exercise at start-up and exits with an error naming
the file and field for an unknown key, a duplicate id, a missing recorded demo or
a rubric item without positive points. The starter course has Instrument basics,
Grasp and hold, Controlled energy, and the demo-only What a safety stop looks
like. Its demos stay under 60% of every default safety limit, except the last
one: its 12 cm swing in 0.15 s trips the speed limit on purpose.

An exercise attempt is a recording tagged with its exercise; each completed step
adds a `step` event to `events.jsonl`. When the attempt ends, `session.json`
gains `exercise: {id, title}` and `report: {steps: [{index, t_s}], steps_done,
near_misses, path_deviation_m, items: [{id, passed, value, points, gated}],
auto_points, auto_max, instructor_max}`. Path deviation is the mean distance from each attempt
pose to the nearest demo pose of the same instrument. A reset, scene switch or
mode switch ends an exercise attempt with its report and does not start a new
one; free practice restarts as before.

Course API:

- `GET /api/courses`: the courses with their exercises, and the recorded demos
  (`demos`).
- `GET /api/courses/{id}/demo` and `GET /api/demos/{name}`: a demo's 30 Hz
  frames and caption markers.
- `POST /api/sessions/{id}/demo` with `{name}`: copies the attempt's controller
  frames to `demos_dir/<slug>.jsonl` under a `{name, source, created}` header
  line. A name already taken is refused (409), and so is an attempt without
  frames (400).
- `POST /api/sessions/{id}/review` with `{checked, notes}`: instructor item ids
  only and at most 2000 characters of notes, written to `review.json`.
- `GET /api/sessions/{id}` adds `exercise`, `report` and `review`.

On `/train`, **Courses** opens the course menu: each exercise with its step
count, rough length and the trainee's best score, the recorded demos, and free
practice. **Start exercise** resets the simulation and starts a tagged attempt.
The right column then shows the step checklist and a safety meter (grey under
60%, amber with "near" from 60%, red at a stop), and **H** swaps it with the key
card. The path guide under the video is a top view with the demo path, the
current target zone and both instruments. The attempt ends after the last step
or with **Stop exercise**, and a report card shows the automatic score, seven
tiles, the rubric and the time per step against the demo. Its Smoothness axis
counts corrections (`report.corrections` against `demo_corrections`): per
instrument, positions are resampled to 30 Hz, low-passed (two 5-sample moving
averages) and the speed peaks above 5 mm/s with 1 cm/s prominence counted, so a
smooth reach is 1 and any input rate compares. The score is demo ÷ yours,
capped at 100; reports from before fall back to RMS jerk. **Watch demo** plays a
demo in place of the mouse; demos are never recorded (a free-practice attempt in
progress is stopped first, an exercise attempt must be ended first) and safety
stops still apply. Space pauses, Backspace restarts, 0.5×/1× sets the speed,
**Try it yourself** starts the exercise, and clicking the video takes over from
the demo's pose. However a demo ends (stop, finish, a safety stop), the jaws
open and energy turns off.

### Model modes

The **Mode** selector in `/train` **⚙ Settings** switches between three model
modes through `POST /mode` (`{"mode": "<name>"}`). `/healthz`, recordings and
the GPU chart report the same names.

| Mode (`POST /mode`) | YAML `runtime` | FP8 projections | Allocated / nvidia-smi | Quality vs Standard |
|---|---|---|---|---|
| `standard` (default) | `fp8_generation: false` | none | 14.1 GiB / 15.4 GiB | reference |
| `balanced` | `fp8_generation: true`, `fp8_scope: mlp` | MLP gate/up/down | 9.0 GiB / 10.4 GiB | 26.4 dB PSNR, LPIPS 0.13 |
| `low_memory` | `fp8_generation: true` (`fp8_scope: all`) | MLP and attention q/k/v/o | 7.6 GiB / 9.0 GiB | 25.4 dB PSNR, LPIPS 0.14 |

The memory columns are the training ground set-up (`offload_understanding` and a
text-cache hit, steady state, H100); all three generate a chunk in about the
same time (82–84 ms). Quality is the mean over 1216 frames against
deterministic Standard; for scale, two non-deterministic Standard runs differ by
30.7 dB and LPIPS 0.06. The YAML settings above select the start-up mode, and
the CLI takes the same as `--fp8-generation` and `--fp8-scope {all,mlp}`
(`--fp8-scope` alone does nothing). FP8 uses E4M3 weights with per-channel
scales, per-token activation scales and FP32 accumulation.

`runtime.steady_refresh_sdpa` (default `true`; CLI `--steady-refresh-sdpa` /
`--no-steady-refresh-sdpa`) refreshes full FadeMem memory with a second compiled
graph using plain SDPA instead of varlen FlashAttention: about 5 ms per chunk
faster on H100, but outputs are not bit-identical to the varlen refresh. `false`
skips that graph (less compile time and memory) and keeps the varlen refresh
throughout. `/healthz` reports it, and a mode switch keeps it.

`runtime.fp8_gemm` (default `epilogue_scales`; CLI `--fp8-gemm`) chooses where
the FP8 projections apply their per-token and per-channel scales.
`epilogue_scales` runs the per-tensor `_scaled_mm` with unit scales and
multiplies by the same scales afterwards; the scaling is unchanged, with one
extra bf16 rounding of each product. `fused_scales` passes them to a row-wise
`_scaled_mm` instead and reproduces earlier outputs bit for bit. Per chunk,
`epilogue_scales` measured 82 → 73 ms (Balanced) and 84 → 73 ms (Low memory) on
an H100, and 181 → 171 and 171 → 157 ms on an RTX 5090, where row-wise
`_scaled_mm` runs a generic CUTLASS kernel and per-tensor runs cuBLAS's sm_120
kernels. Its frames differ slightly from `fused_scales` (33–36 dB PSNR against
the H100 device-check reference, which passes). A mode switch keeps it. See
[generation-speed.md](../docs/generation-speed.md).

`runtime.attention_kernel` (default `auto`; CLI `--attention-kernel`) chooses the
kernel for mask-free attention (the denoise passes and the steady refresh;
masked and varlen calls keep PyTorch's). `auto` uses SageAttention2 on RTX 50
GPUs (sm_120) when it is installed and passes a start-up self-test, and PyTorch
SDPA everywhere else; `default` always uses SDPA; `sage` uses SageAttention on
any GPU and refuses to start if it is missing or fails the self-test.
SageAttention is optional and not in `uv.lock`: install it with
`scripts/install_sageattention.sh`, which builds it with every kernel launched
on PyTorch's current stream (upstream launches on the legacy default stream, so
CUDA graphs miss its kernels and the video turns to NaN). The self-test catches
such a build by comparing a captured replay with an eager call. Per chunk it
measured 170 → 153 ms (Balanced) and 156 → 138 ms (Low memory) on an RTX 5090,
253 → 216 ms (Low memory) on an RTX 5080, and 73 → 98 ms on an H100, where
cuDNN attention is about 3× faster than SageAttention, so `auto` keeps SDPA
there. Frames stay within the device check's tolerance. The start-up log,
`/healthz` (`attention`) and the device-check report show the choice and why.
See [blackwell-optimization.md](../docs/blackwell-optimization.md).

A switch ends the current attempt, unloads the model and loads it again. The
first switch into a mode compiles for 1–2 minutes; later switches take about
15–40 s. If loading fails, the previous mode is reloaded and the page shows the
error.

**CUDA allocator.** A session on CUDA switches PyTorch's caching allocator to
`expandable_segments:True` when it loads, so the server and the CLI both get
it. Fixed-size segments kept about 0.8 GiB more reserved at steady state. To
use other allocator settings, set `PYTORCH_CUDA_ALLOC_CONF` (or
`PYTORCH_ALLOC_CONF`) before launch; the session then leaves the allocator
alone. The denoise and refresh CUDA graphs of a rollout also share one memory
pool instead of one private pool each.

### Text cache

The bundled YAML sets `runtime.text_cache:
data_local/text_kv/c3-h-d_2-step_480x832.pt` (relative to the launch
directory). The file holds the prompt's text K/V, so the model can load without
its understanding weights: about 26 GB less host RAM with
`offload_understanding`, or VRAM without it.

- **When it is built.** The first start finds no file: it loads the full model,
  prefills the prompt, saves the file (creating its folder) and drops the
  understanding weights. Later starts, and every mode switch, load without them.
  The file is rebuilt only when the checkpoint's bytes, the prompt, the
  resolution or the tokenizer change. A copied, re-downloaded or touched
  checkpoint with the same bytes costs one re-hash (about 17 s) and no rebuild;
  the new size and modification time are then stored, so later starts skip the
  hash.
- **Prebuild at deploy time** by starting the server once and waiting for
  "Text cache: built and saved" on `/train` (or `[text-cache] saved` in the
  log), then stopping it. Trainees then never wait for the build. If the
  checkpoint is copied or touched afterwards, the next start spends about 17 s
  re-hashing it and still hits.
- **Scenes must share one prompt.** The file holds one prompt, height and
  width, and once the understanding weights are dropped no other prompt can be
  prefilled. The server refuses to start when `text_cache` is set and the
  scenes differ in prompt, height or width; remove `text_cache` or give every
  scene the same prompt. (Without this check, switching to such a scene fails
  with "Scene switch failed".)
- **Status.** `/train` shows a line under the model status in **⚙ Settings**, and `/gpu` the same
  line in its footer: "Text cache: hit (model loaded without understanding
  weights)", "Text cache: built and saved", or "Text cache: off", followed by
  "· checkpoint re-hashed in 16.8 s" when the checkpoint was hashed again.
  `/healthz` carries `text_cache: {status, rehashed, path, hash_s, load_s}`
  (null while the model is switching or warming up), where `status` is `off`,
  `hit`, `built` or `built-unsaved` (the save failed; it is rebuilt on the next
  start), `rehashed` is true when the checkpoint's size or modification time
  had changed so it was hashed again, `hash_s` the checkpoint hash time and
  `load_s` the model load time. Each attempt's `session.json` stores
  `text_cache: {status, rehashed}`, next to `mode`.

Remove the `text_cache` line to load the full model every time.

### GPU monitor

The server samples GPU memory once per second so you can check that VRAM stays
bounded across attempts and mode switches. It reads every GPU through NVML
(`nvidia-ml-py`): memory used and total, utilization and temperature. For the
server's own GPU it also reads PyTorch's allocated and reserved counters. The
monitor never creates a CUDA context. If NVML is unavailable it falls back to
the PyTorch counters, with reserved memory standing in for "used".

- **`/gpu`** shows a live chart of the server's GPU. The lines are used (NVML),
  PyTorch reserved and PyTorch allocated, in GiB, with dashed capacity and
  dotted baseline lines.
  - A mode switch shows as a shaded band, with a rule labelled with the new mode.
  - Attempts are bars under the time axis, and incidents are triangles.
  - Lines break where samples are missing.
  - Tiles above the chart show used now, the peak since start, the baseline (the
    first sample, taken before the model loads), PyTorch allocated,
    utilization, temperature, and host RAM (now and peak). NVML's "used" includes memory the driver
    reserves, so an idle H100 shows about 0.5 GiB.
  - **Span** picks the last 5, 10 or 30 minutes. **Scale** picks
    **Full capacity** (0 to the GPU's total) or **Fit to data**; the page
    remembers the choice.
  - **Host RAM (PSS)**, a second chart below, plots the host memory of the
    server process and its children (such as ffmpeg and torchinductor's
    compile workers) in GiB, as summed PSS (proportional set size, from
    `/proc/<pid>/smaps_rollup`). PSS splits each shared page between the
    processes that map it, so forked workers that share their parent's pages
    add only their private memory and the total is the tree's real
    footprint. The value is a coherent PSS snapshot of the server's
    process tree, refreshed every 5 s by its own thread. Every live process
    is read in the same pass, and exited and zombie processes count as 0.
    Each GPU sample carries the latest snapshot, so the host value can be
    up to 5 s old. If PSS cannot be read (not Linux, or access denied),
    the server process's own RSS from `psutil` stands in. It uses the same
    time axis, with the same mode-switch and attempt marks. This is
    where the text cache's saving shows. It is stored on the server GPU's
    sample, so the chart appears only when the server's GPU is selected.
  - The **GPU** selector shows other GPUs. **Show table** lists the last 60
    samples, including host RAM. The page polls `/api/gpu` every second.
- **`/train`** shows two small graphs under the video, GPU memory used (on a
  0-to-capacity axis) and host RAM (PSS), over the last 5 minutes, with the
  current value and peak in each title and a link to `/gpu`. `/healthz`
  carries the GPU numbers in `gpu: {used_mib, total_mib, peak_mib}`.
- **CSV log:** every sample is appended to
  `<recording.dir>/gpu-<YYYYMMDD-HHMMSS>.csv`, one file per server run. The
  columns are `time, gpu, used_mib, total_mib, util_pct, temp_c,
  torch_allocated_mib, torch_reserved_mib, host_pss_mib`, with `time` in Unix
  seconds. The PyTorch columns are empty for other GPUs and until PyTorch has
  initialised CUDA; `host_pss_mib` is filled only on the server GPU's rows.
  It grows by about 50 bytes per row, one row per GPU per period (about 340 KiB
  an hour per GPU at the default period). The server prints the file's path at
  start-up. The server keeps only the last 30 minutes of samples in memory
  (however you set the period), and the pages show only those; the CSV is the
  full record.
- **Recordings:** when an attempt ends, its `session.json` gains `summary.vram`:
  `{peak_used_mib, mean_used_mib, peak_allocated_mib, mean_allocated_mib,
  peak_host_pss_mib, mean_host_pss_mib}`. They
  cover the whole attempt when it is shorter than the monitor's 30-minute
  history; a longer attempt reports its last 30 minutes, and the CSV has the
  full record. The dashboard shows a **Peak VRAM** column (GiB) and these
  numbers, peak host RAM and the text cache status in an attempt's summary.
- **`GET /api/gpu`** returns `now`, `device`, `nvml`, `gpus`, `samples`,
  `marks` (`attempt_start`, `attempt_end`, `incident`, `mode_switch_start`,
  `mode_switch_end`), `baseline` and `peak` per GPU, `host` (`{pss_mib,
  peak_mib}`, the summed PSS of the server process tree), `csv` and `csv_error`.
  `?since=<t>` returns only newer samples and marks, and `?gpu=<index>` returns
  only one GPU's samples. Without `gpu=` it returns every GPU's 30-minute
  buffer, which is large on a multi-GPU host. It answers 404 when the monitor
  is off.

The monitor is on by default. Turn it off with `--no-gpu-monitor`, and change the
period with `--gpu-period <seconds>`. Both can also be set in the session YAML
(the values shown are the defaults); the command-line flags override the YAML:

```yaml
gpu_monitor:
  enabled: true
  period_s: 1.0
```

### Generation latency

Under the graphs, `/train` shows whether generation keeps up with your input:

```text
Generation  82 ms / chunk · 10 ms / frame        ● keeping up
Response    ~0.3 s action → frame
```

One chunk turns 8 action rows into 8 video frames, and the rows arrive at the
acquisition rate (`headset_input_fps`, else the model `fps`), so a chunk must
be ready within 8 ÷ that rate (267 ms at 30 Hz) to keep up, and its 8 frames
must play in that time too. The dot and word say **keeping up** (green),
**near the limit** (amber, generation at 80 % of that budget or more) or
**falling behind** (red: generation over the budget, or **falling behind
(playback)** when 8 frames at `display_fps` take more than 2 % longer than the
budget). The training ground configs set `display_fps: 30` to match the 30 Hz
input; the `headset-base` preset alone plays at 20 fps. Hover the readout for
the budget.

`/train` can change the acquisition rate while running (**⚙ Settings → Input
rate**, `POST /input-rate {"fps": 24}`): 5–60 Hz, or up to the configured
rate when that is higher (a headset). The page moves the instruments and sends
poses at max(30 Hz, the input rate), with smoothing and the R glide scaled by
time, so a rate above 30 Hz gets a new pose for every sample. A lower rate gives each chunk more time (8 ÷ rate) without
reloading the model; `display_fps` keeps its ratio to the input rate. The
model's own `fps` (temporal RoPE, 10 in the presets) is unchanged, so each
frame just covers more hand motion. Changes are refused (409) while an attempt
is recorded; recordings store their video at the input rate, so they play in
real time. `/healthz` reports `input_rate: {fps, min, max, chunk_frames}`.
When generation is slower than the input, the controller buffer (8 rows + 1)
drops the oldest poses, so the delay stays bounded but the video pauses
between chunks.

`/healthz` carries the medians of the last 20 chunks in `latency: {chunk_ms,
frame_ms, response_ms, budget_ms, playback_ms, status, reason}`, with `status`
`ok`, `tight` or `behind` and `reason` `generation` or `playback` when behind
(else null); it is `null` until the first chunk and again after a mode or scene
switch or a reset. Both times use the server's clock:

- `chunk_ms` runs from the start of the model step (after its action window is
  complete) to uint8 frames on the host: rollout, decode and frame conversion,
  not JPEG encoding. `frame_ms` is that per frame; `playback_ms` is 8 frames at
  `display_fps`.
- `response_ms` runs from the moment the chunk's first action row reached the
  server to the moment its first frame was handed to the video stream: waiting
  for the window to fill, generation and queueing behind the playing block.
  Browser and network delay are not included.

Playback catches up rather than letting that queue grow: while the next block
waits for the playing one, frames are shown at 3/4 of the display interval, and
once it has waited more than half a block's playback, every other frame is
skipped. With nothing waiting, playback is unchanged. So a display slower than
the input skips frames instead of falling ever further behind.

### Network access

The training ground has no login. Anyone who can reach the port can drive the
simulation, switch modes, and watch every recording, including the trainee
names, through `/dashboard` and `/api/sessions`. Run it on a trusted LAN
only, and do not expose the port to the internet.

## Performance and diagnostics

To check a new GPU (for example an RTX 5090 or 5080), run the hardware test kit in [docs/rtx5090-test-kit.md](../docs/rtx5090-test-kit.md).

Append `--profile` to record stage timings, or `--print-actions` to the headset
command to log normalized action rows and sampled controller poses. Action rows
are consumed and printed in chunks. Profiling synchronizes GPU work and can
change performance; compare steady-state runs after compilation and cache fill,
then repeat without profiling.

On CUDA, `--profile` also reports peak and time-weighted average VRAM allocated
and reserved by PyTorch, including weights, KV cache, and decoder memory. Peaks
use allocator counters; averages sample every 10 ms and can miss shorter spikes.
The measurement runs from the first inference chunk through session close,
including time between chunks, and starts fresh after a reset or scene switch.
Loading and prewarm are excluded. CUDA context memory, allocations outside
PyTorch, and other processes are not included; reserved memory includes unused
space held by PyTorch's caching allocator.

Model time, acquisition time, and display time are separate. A chunk2 step uses
eight actions. Acquiring measured poses at 30 Hz retimes those actions into the
10 Hz model timeline; it does not establish physical motion parity. Increasing
display FPS does not increase generation throughput or interpolate frames.

Playback overlap is enabled by default. `--no-overlap-playback` selects sequential
generation and playback. For MJPEG diagnostics, compare `input`, `generation`,
`queue_wait`, and `boundary_interval` after warmup. `pose_age_first_frame` runs
from server receipt of the newest pose in a block to its first JPEG publication;
it excludes network delivery and headset rendering.

### Cache options

- `cat`: eager rolling cache; accepts partial final chunks.
- `block`: fixed-buffer cache; requires full chunks. `--window-frames` controls
  the rolling attention history.
- `fademem`: bounded historical entries, configured with `--fademem-slots` and
  `--fademem-beta`. `--window-frames` does not control this mode.

The FadeMem cache is an independent implementation of the method from
[FadeMem: Distance-Aware Memory Consolidation for Autoregressive Video Diffusion](https://arxiv.org/abs/2606.10671)
by Yu Lu et al. (2026) to Cosmos3-H-Dreams.

FadeMem merges adjacent historical entries when its budget is exceeded and
protects the newest entry. Choose `--fademem-merge-selection age` or `cosine`;
cosine ranks clean latent frames and can miss small tool motion against a large
stationary background. Cache consolidation can lose scene detail. Validate long
rollouts and reversals on your checkpoint before choosing it for a workload.

Compiled presets already enable `--compile-denoise`, `--cuda-graph-denoise`, and
`--overlap-cache-update`. For comparisons, hold the checkpoint, initial frame,
actions, seed, decoder, and attention budget fixed. CPU checks cover cache logic;
they do not establish GPU speedups or visual quality.

## Tests

From an environment with the test dependencies installed and Node.js 14.18.0
or newer for the browser JavaScript checks (a currently supported LTS release
is recommended):

```bash
PYTHONPATH=cosmos3_h_dreams:flashdreams .venv/bin/python -m pytest \
  cosmos3_h_dreams/tests -m ci_cpu -q
node cosmos3_h_dreams/tests/test_headset_status.cjs
node cosmos3_h_dreams/tests/test_scene_selector.cjs
node cosmos3_h_dreams/tests/test_train_core.cjs
node cosmos3_h_dreams/tests/test_dashboard.cjs
node cosmos3_h_dreams/tests/test_gpu_chart.cjs
node cosmos3_h_dreams/tests/test_train_courses.cjs
node cosmos3_h_dreams/tests/test_xr_view.cjs
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for test dependency installation and
repository checks. CPU/browser checks use fake model sessions and do not run
GPU inference or validate live headset behavior.
