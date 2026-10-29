// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with: node cosmos3_h_dreams/tests/test_train_core.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(
  fs.readFileSync(path.join(__dirname, "../cosmos3_h_dreams/web/train_core.js"), "utf8"),
  context,
);
const { newState, applyInput, tick, frameMessage, goHome, applyRestore, HOME_STEP_M, wheelNotches,
  newVrInput, clutch, applyHands, quatMultiply, axisAngle, DEFAULT_INSTRUMENTS } = context;
const settings = { metresPerPixel: 0.001, radiansPerPixel: 0.005, metresPerWheel: 0.0002 };
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const norm = q => Math.hypot(...q);
// Arrays built inside the vm context have another realm's Array.prototype; copy them so strict deepEqual compares values.
const plain = v => Array.from(v);

// Mouse moves only the active (right) instrument's target; the pose follows over ticks.
let state = newState();
assert.equal(state.active, "right");
const start = [...state.instruments.right.pose.position];
applyInput(state, { dx: 100, dy: -50, wheel: 0, shift: false, roll: 0 }, settings);
close(state.instruments.right.target.position[0], start[0] + 0.1);
close(state.instruments.right.target.position[2], start[2] - 0.05);
close(state.instruments.left.target.position[0], -0.15);
tick(state);
const moved = state.instruments.right.pose.position[0] - start[0];
assert.ok(moved > 0 && moved < 0.1, "pose moves part of the way per tick");
for (let i = 0; i < 60; i += 1) tick(state);
close(state.instruments.right.pose.position[0], start[0] + 0.1, 1e-4);

// Wheel moves up/down; shift turns mouse motion into rotation and leaves position alone.
const y = state.instruments.right.target.position[1];
applyInput(state, { dx: 0, dy: 0, wheel: 100, shift: false, roll: 0 }, settings);
close(state.instruments.right.target.position[1], y - 0.02);
const before = [...state.instruments.right.target.position];
applyInput(state, { dx: 40, dy: 0, wheel: 0, shift: true, roll: 0.1 }, settings);
assert.deepEqual(plain(state.instruments.right.target.position), before);
close(norm(state.instruments.right.target.orientation), 1);
assert.notDeepEqual(plain(state.instruments.right.target.orientation), [0, 0, 0, 1]);

// Held state ignores input.
state.held = true;
applyInput(state, { dx: 500, dy: 0, wheel: 0, shift: false, roll: 0 }, settings);
assert.deepEqual(plain(state.instruments.right.target.position), before);
state.held = false;

// Frame message matches the headset wire schema.
state.instruments.right.jaw = true;
state.instruments.left.energy = true;
const message = frameMessage(state, 123.5);
assert.equal(message.type, "xr_frame");
assert.equal(message.t_client, 123.5);
for (const side of ["left", "right"]) {
  const c = message.controllers[side];
  assert.equal(c.position.length, 3);
  assert.equal(c.orientation.length, 4);
  close(norm(c.orientation), 1);
  assert.equal(c.tracked, true);
}
assert.equal(message.controllers.right.trigger, 1);
assert.equal(message.controllers.right.grip, 0);
assert.equal(message.controllers.left.grip, 1);
message.controllers.right.position[0] = 99;  // a copy, not live state
assert.notEqual(state.instruments.right.pose.position[0], 99);

// Going home glides no faster than HOME_STEP_M per tick.
goHome(state, "right");
let last = [...state.instruments.right.pose.position];
for (let i = 0; i < 400; i += 1) {
  tick(state);
  const p = state.instruments.right.pose.position;
  const step = Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]);
  assert.ok(step <= HOME_STEP_M + 1e-9, `home step ${step}`);
  last = [...p];
}
close(state.instruments.right.pose.position[0], 0.15, 1e-6);

// Per-frame zero input must not cancel a glide: displace and rotate, then home with zero input each tick.
applyInput(state, { dx: 200, dy: 100, wheel: 0, shift: false, roll: 0.3 }, settings);
for (let i = 0; i < 100; i += 1) tick(state);
goHome(state, "right");
last = [...state.instruments.right.pose.position];
for (let i = 0; i < 400; i += 1) {
  applyInput(state, { dx: 0, dy: 0, wheel: 0, shift: false, roll: 0 }, settings);
  tick(state);
  const p = state.instruments.right.pose.position;
  const step = Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]);
  assert.ok(step <= HOME_STEP_M + 1e-9, `zero-input home step ${step}`);
  last = [...p];
}
close(state.instruments.right.pose.position[0], 0.15, 1e-6);
close(state.instruments.right.pose.position[1], 1.0, 1e-6);
close(state.instruments.right.pose.position[2], -0.3, 1e-6);
const q = plain(state.instruments.right.pose.orientation);
[0, 0, 0].forEach((v, i) => close(q[i], v, 1e-6));
close(q[3], 1, 1e-6);

// Restore snaps pose and target, releases energy and clears the hold.
state.held = true;
state.instruments.left.energy = true;
applyRestore(state, { left: { position: [0.01, 1.1, -0.2], orientation: [0, 0, 0, 1] } });
assert.deepEqual(plain(state.instruments.left.pose.position), [0.01, 1.1, -0.2]);
assert.deepEqual(plain(state.instruments.left.target.position), [0.01, 1.1, -0.2]);
assert.equal(state.instruments.left.energy, false);
assert.equal(state.held, false);

// While held, the pose stays put: it must not keep settling toward an over-shot target.
state = newState();
applyInput(state, { dx: 300, dy: 0, wheel: 0, shift: false, roll: 0 }, settings);
tick(state);
state.held = true;
const frozen = plain(state.instruments.right.pose.position);
for (let i = 0; i < 10; i += 1) tick(state);
assert.deepEqual(plain(state.instruments.right.pose.position), frozen);
state.held = false;
tick(state);
assert.ok(state.instruments.right.pose.position[0] > frozen[0], "settling continues after the hold");

// Attempt time excludes the time held (paused or stopped by a safety stop), as the server's report does.
// `held` is sampled on each call, so a hold is counted from the first call that sees it.
const attempt = context.newActiveClock(1000);
close(context.activeSeconds(attempt, 1000, false), 0);
close(context.activeSeconds(attempt, 3000, false), 2);                 // 2 s running
close(context.activeSeconds(attempt, 4000, true), 3);                  // paused at 4000: time stands still...
close(context.activeSeconds(attempt, 9000, true), 3);                  // ...for the whole pause
close(context.activeSeconds(attempt, 10000, false), 3);                // resumed at 10000 (6 s held)
close(context.activeSeconds(attempt, 11500, false), 4.5);
close(context.activeSeconds(attempt, 12000, true), 5);                 // a safety stop
close(context.activeSeconds(attempt, 15000, true), 5);
close(context.activeSeconds(attempt, 15500, false), 5);                // resumed after 3.5 s
close(context.activeSeconds(attempt, 17500, false), 7);
// Repeated held samples keep one hold, and the clock never goes negative or backwards.
const second = context.newActiveClock(0);
context.activeSeconds(second, 1000, true);
context.activeSeconds(second, 2000, true);
context.activeSeconds(second, 3000, true);
close(context.activeSeconds(second, 3000, false), 1);
close(context.activeSeconds(context.newActiveClock(5000), 4000, false), 0);

// Latency readout: integers for ms, seconds with one decimal below 10 s, a word for each status.
{
  const { latencyView, secondsText } = context;
  assert.deepEqual({ ...latencyView(null) }, { generation: "—", response: "—", status: null, word: "—", title: "" });
  const view = latencyView({ chunk_ms: 82.4, frame_ms: 10.3, response_ms: 312.0, budget_ms: 266.7, status: "ok" });
  assert.equal(view.generation, "82 ms / chunk · 10 ms / frame");
  assert.equal(view.response, "~0.3 s action → frame");
  assert.equal(view.status, "ok");
  assert.equal(view.word, "keeping up");
  assert.equal(view.title, "8 frames per chunk; must finish within 267 ms to keep up with your input");
  assert.equal(latencyView({ chunk_ms: 230, frame_ms: 28.75, response_ms: null, budget_ms: 266.7, status: "tight" }).word,
    "near the limit");
  const behindInput = { chunk_ms: 300, frame_ms: 37.5, response_ms: 12400, budget_ms: 266.7, status: "behind" };
  const behind = latencyView(behindInput);
  assert.equal(behind.word, "falling behind");
  assert.equal(latencyView({ ...behindInput, reason: "generation" }).word, "falling behind");
  const playback = latencyView({ chunk_ms: 72, frame_ms: 9, response_ms: 690, budget_ms: 266.7, playback_ms: 400,
    status: "behind", reason: "playback" });
  assert.equal(playback.word, "falling behind (playback)");
  assert.equal(playback.status, "behind");
  assert.equal(latencyView({ chunk_ms: 72, frame_ms: 9, response_ms: 300, budget_ms: 266.7, status: "ok", reason: "playback" }).word,
    "keeping up");
  assert.equal(behind.response, "~12 s action → frame");
  assert.equal(latencyView({ chunk_ms: 50, frame_ms: 6, response_ms: null, budget_ms: null, status: null }).response, "—");
  assert.equal(latencyView({ chunk_ms: 50, frame_ms: 6, response_ms: null, budget_ms: null, status: "weird" }).status, null);
  assert.equal(secondsText(9949), "9.9 s");
  assert.equal(secondsText(9950), "10 s");
}

console.log("train_core ok");

// One wheel notch counts as one notch whichever unit the browser reports it in.
close(wheelNotches(100, 0), 1);     // Chrome: pixels
close(wheelNotches(-3, 1), -1);     // Firefox: lines
close(wheelNotches(12.5, 0), 0.125);   // high-resolution wheel / touchpad: a fraction
close(wheelNotches(800, 2), 1);     // pages

// Instruments: each side lists only its own tools; a missing tool's hint names the other instrument.
{
  const { hasTool, instrumentLabel, missingToolHint, instrumentStatus, DEFAULT_INSTRUMENTS } = context;
  const tools = { left: { name: "Grasper", tools: ["jaws"] }, right: { name: "Hook", tools: ["energy"] } };
  assert.ok(hasTool(tools, "left", "jaws") && !hasTool(tools, "right", "jaws"));
  assert.ok(hasTool(DEFAULT_INSTRUMENTS, "right", "jaws"), "both tools on both sides by default");
  assert.ok(!hasTool(undefined, "left", "jaws"));
  assert.equal(instrumentLabel(tools, "left"), "Left · Grasper");
  assert.equal(instrumentLabel(DEFAULT_INSTRUMENTS, "right"), "Right");
  assert.equal(missingToolHint(tools, "right", "jaws"), "The Hook has no jaws · Tab switches to the Grasper");
  const neither = { left: { name: "Probe", tools: [] }, right: { name: "Hook", tools: ["energy"] } };
  assert.equal(missingToolHint(neither, "right", "jaws"), "The Hook has no jaws");
  assert.equal(missingToolHint(tools, "right", "jaws", true), "The Hook has no jaws", "no Tab in VR");
  const instrument = { jaw: true, energy: true };
  assert.equal(instrumentStatus(tools, "left", instrument, 0), "Left · Grasper: jaws closed");
  assert.equal(instrumentStatus(tools, "right", instrument, 1.24), "Right · Hook: energy ON 1.2s");
  assert.equal(instrumentStatus(neither, "left", instrument, 0), "Left · Probe: no tools");
  assert.equal(instrumentStatus(DEFAULT_INSTRUMENTS, "right", { jaw: false, energy: false }, 0), "Right: jaws open, energy off");
  console.log("train_core instruments ok");
}

// Step guidance: distance and height to a reach target, progress of a "moved" step.
{
  const { stepGuidance, newStepRun } = context;
  const view = (active, y) => ({ active, left: { position: [-0.12, y, -0.40] }, right: { position: [0.15, 1.0, -0.3] } });
  const reach = { reach: { side: "left", center: [-0.12, 0.98, -0.40], radius_m: 0.015 } };
  assert.equal(stepGuidance(reach, view("left", 1.0), null), "0.5 cm to the target · 2.0 cm too high · scroll down");
  assert.equal(stepGuidance(reach, view("left", 0.95), null), "1.5 cm to the target · 3.0 cm too low · scroll up");
  assert.equal(stepGuidance(reach, view("left", 0.983), null), "0.0 cm to the target · height ✓");
  assert.equal(stepGuidance(reach, view("right", 0.98), null), "Tab to the left instrument · 0.0 cm to the target · height ✓");
  assert.equal(stepGuidance(reach, view("right", 1.0), null, true), "0.5 cm to the target · 2.0 cm too high · lower it");
  assert.equal(stepGuidance(reach, view("left", 0.95), null, true), "1.5 cm to the target · 3.0 cm too low · raise it");
  const run = newStepRun([]);
  run.pathM.left = 0.021;
  assert.equal(stepGuidance({ moved: { side: "left", min_m: 0.05 } }, view("left", 1), run), "2.1 cm / 5.0 cm");
  run.pathM.left = 0.07;
  assert.equal(stepGuidance({ moved: { side: "left", min_m: 0.05 } }, view("left", 1), run), "5.0 cm / 5.0 cm");
  assert.equal(stepGuidance({ active: "left" }, view("left", 1), run), "");
  console.log("train_core guidance ok");
}

// A scene that starts gripping: jaws closed from the first frame, kept through Tab-style releases until a click.
{
  const { newState, applyStartGrip, frameMessage, instrumentStatus } = context;
  const s = applyStartGrip(newState(), ["left"]);
  assert.ok(s.instruments.left.jaw && s.instruments.left.gripping && !s.instruments.right.jaw);
  assert.equal(frameMessage(s, 0).controllers.left.trigger, 1);
  const tools = { left: { name: "Grasper", tools: ["jaws"] }, right: { name: "Hook", tools: ["energy"] } };
  assert.equal(instrumentStatus(tools, "left", s.instruments.left, 0), "Left · Grasper: jaws locked closed (G or click releases)");
  assert.equal(instrumentStatus(tools, "left", s.instruments.left, 0, true),
    "Left · Grasper: jaws locked closed (press and release trigger to let go)");
  applyStartGrip(s, []);
  assert.ok(!s.instruments.left.gripping && !s.instruments.left.jaw, "a scene without a grip opens the previous grip");
  s.instruments.right.jaw = true;   // e.g. a mouse press in progress
  applyStartGrip(s, []);
  assert.ok(s.instruments.right.jaw, "and leaves jaws it did not close to the mouse");
  console.log("train_core start grip ok");
}

// G locks the jaws closed and unlocks them; a locked grip survives the release that Tab applies.
{
  const { newState, toggleGripLock, frameMessage } = context;
  const s = newState();
  s.active = "left";
  assert.equal(toggleGripLock(s, "left"), true);
  assert.ok(s.instruments.left.jaw && s.instruments.left.gripping);
  s.active = "right";   // Tab: train.js releaseInstrument keeps a gripping instrument's jaws
  assert.equal(frameMessage(s, 0).controllers.left.trigger, 1, "still closed while the hook is active");
  assert.equal(toggleGripLock(s, "left"), false);
  assert.ok(!s.instruments.left.jaw && !s.instruments.left.gripping);
  console.log("train_core grip lock ok");
}

// Restart progress: the server must see the reset (epoch), recalibrate, then show new video.
{
  const { prepView } = context;
  assert.equal(prepView(null, 3).done, false);
  assert.equal(prepView({ epoch: 3, calibrated: true, calibration_fraction: 1, chunks_shown: 9 }, 3).text, "Resetting the simulation…");
  const calibrating = prepView({ epoch: 4, calibrated: false, calibration_fraction: 0.5, chunks_shown: 0 }, 3);
  assert.equal(calibrating.percent, 35);
  assert.equal(calibrating.text, "Calibrating · hold still");
  assert.equal(prepView({ epoch: 4, calibrated: true, calibration_fraction: 1, chunks_shown: 0 }, 3).percent, 80);
  assert.ok(prepView({ epoch: 4, calibrated: true, calibration_fraction: 1, chunks_shown: 1 }, 3).done);
  console.log("train_core prep ok");
}

// Moving the mouse during R's glide home stops the glide where the instrument is: it must not lunge toward home
// (the old target) at the normal smoothing rate, which a safety guard sees as a speed spike.
{
  const { newState, applyInput, tick, goHome, HOME_STEP_M } = context;
  const s = newState();
  const far = [0.25, 1.0, -0.3];   // 10 cm right of the right instrument's home
  s.instruments.right.pose.position = [...far];
  s.instruments.right.target.position = [...far];
  goHome(s, "right");
  for (let i = 0; i < 3; i += 1) tick(s);
  const before = [...s.instruments.right.pose.position];
  applyInput(s, { dx: 1, dy: 0, wheel: 0, shift: false, roll: 0 }, settings);
  tick(s);
  const moved = Math.hypot(...s.instruments.right.pose.position.map((v, i) => v - before[i]));
  assert.ok(moved < 2 * HOME_STEP_M, `moved ${moved} m in one tick after the mouse took over`);
  assert.equal(s.instruments.right.homing, false);
  console.log("train_core glide takeover ok");
}

// Input rate: the budget for a typed rate, advice only when generation falls behind.
{
  const { inputRateBudgetText, inputRateAdvice } = context;
  assert.equal(inputRateBudgetText(30, 8), "267 ms per chunk");
  assert.equal(inputRateBudgetText(22, 8), "364 ms per chunk");
  assert.equal(inputRateBudgetText(0, 8), "");
  assert.equal(inputRateBudgetText(NaN, 8), "");
  assert.equal(inputRateBudgetText(24, null), "");
  const rate = { fps: 30, min: 5, max: 30, chunk_frames: 8 };
  const behind = { status: "behind", reason: "generation", chunk_ms: 300 };
  assert.equal(inputRateAdvice(behind, rate), "Chunks take 300 ms: try 24 Hz or lower (333 ms per chunk).");
  assert.equal(inputRateAdvice({ ...behind, chunk_ms: 1700 }, rate), "Chunks take 1700 ms; even 5 Hz cannot keep up.");
  assert.equal(inputRateAdvice({ ...behind, reason: "playback" }, rate), "");
  assert.equal(inputRateAdvice({ status: "ok", chunk_ms: 100 }, rate), "");
  assert.equal(inputRateAdvice(null, rate), "");
  assert.equal(inputRateAdvice(behind, { ...rate, chunk_frames: null }), "");
  console.log("train_core input rate ok");
}

// tick(state, ticks): two half ticks (a 60 Hz page) move as far as one 30 Hz tick.
{
  // newState, tick, HOME_STEP_M and close come from the top of this file.
  const once = newState(), twice = newState();
  for (const s of [once, twice]) s.instruments.right.target.position = s.instruments.right.pose.position.map(v => v + 0.01);
  tick(once, 1);
  tick(twice, 0.5); tick(twice, 0.5);
  once.instruments.right.pose.position.forEach((v, i) => close(v, twice.instruments.right.pose.position[i], 1e-12));
  const home = newState();
  home.instruments.right.homing = true;
  home.instruments.right.target.position = home.instruments.right.pose.position.map((v, i) => v + (i === 0 ? 0.05 : 0));
  const start = home.instruments.right.pose.position[0];
  tick(home, 0.5);
  close(home.instruments.right.pose.position[0] - start, HOME_STEP_M / 2, 1e-12);
  console.log("train_core tick rate ok");
}

// ---- VR hands: clutch offsets, tracking loss, release-before-act, starting grip ----
{
  const hand = (position, orientation = [0, 0, 0, 1], trigger = 0, grip = 0) => ({ position, orientation, trigger, grip });
  const closeVec = (a, b, eps = 1e-6) => b.forEach((v, i) => close(a[i], v, eps));
  const sameRotation = (a, b) => close(Math.abs(a.reduce((s, v, i) => s + v * b[i], 0)), 1, 1e-6);

  // The first tracked pose anchors: the instrument stays put, then follows hand motion one-to-one.
  let vrState = newState();
  const input = newVrInput();
  const home = plain(vrState.instruments.right.pose.position);
  const twist = axisAngle([0, 0, 1], 0.4);
  applyHands(vrState, input, { left: null, right: hand([0.5, 1.2, 0.1], twist) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(vrState.instruments.right.pose.position), home);
  sameRotation(plain(vrState.instruments.right.pose.orientation), [0, 0, 0, 1]);
  applyHands(vrState, input, { left: null, right: hand([0.53, 1.18, 0.1], twist) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(vrState.instruments.right.pose.position), [home[0] + 0.03, home[1] - 0.02, home[2]]);
  closeVec(plain(vrState.instruments.right.target.position), plain(vrState.instruments.right.pose.position));

  // A world rotation R of the hand turns the instrument by the same R.
  const turn = axisAngle([0, 1, 0], 0.3);
  applyHands(vrState, input, { left: null, right: hand([0.53, 1.18, 0.1], quatMultiply(turn, twist)) }, DEFAULT_INSTRUMENTS);
  sameRotation(plain(vrState.instruments.right.pose.orientation), turn);

  // Tracking lost: the pose holds and the buttons let go; on return the hand re-anchors without a jump.
  vrState.instruments.right.energy = true;
  vrState.instruments.right.jaw = true;
  const held = plain(vrState.instruments.right.pose.position);
  applyHands(vrState, input, { left: null, right: null }, DEFAULT_INSTRUMENTS);
  closeVec(plain(vrState.instruments.right.pose.position), held);
  assert.equal(vrState.instruments.right.energy, false);
  assert.equal(vrState.instruments.right.jaw, false);
  applyHands(vrState, input, { left: null, right: hand([-2, 0.3, 4]) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(vrState.instruments.right.pose.position), held);

  // Held (pause or safety stop): hands are ignored; resuming continues from the held pose,
  // and from the restore pose after an incident.
  vrState.held = true;
  applyHands(vrState, input, { left: null, right: hand([-1, 0.3, 4]) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(vrState.instruments.right.pose.position), held);
  applyRestore(vrState, { right: { position: [0.2, 1.1, -0.4], orientation: [0, 0, 0, 1] } });
  applyHands(vrState, input, { left: null, right: hand([-1, 0.3, 4]) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(vrState.instruments.right.pose.position), [0.2, 1.1, -0.4]);
  applyHands(vrState, input, { left: null, right: hand([-1, 0.4, 4]) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(vrState.instruments.right.pose.position), [0.2, 1.2, -0.4]);

  // After a clutch, a trigger or grip already held does nothing until released.
  clutch(input);
  applyHands(vrState, input, { left: null, right: hand([-1, 0.4, 4], [0, 0, 0, 1], 1, 1) }, DEFAULT_INSTRUMENTS);
  assert.equal(vrState.instruments.right.jaw, false);
  assert.equal(vrState.instruments.right.energy, false);
  applyHands(vrState, input, { left: null, right: hand([-1, 0.4, 4]) }, DEFAULT_INSTRUMENTS);
  applyHands(vrState, input, { left: null, right: hand([-1, 0.4, 4], [0, 0, 0, 1], 1, 1) }, DEFAULT_INSTRUMENTS);
  assert.equal(vrState.instruments.right.jaw, true);
  assert.equal(vrState.instruments.right.energy, true);

  // Tools the instrument lacks stay off.
  const hookOnly = { left: { name: "Grasper", tools: ["jaws"] }, right: { name: "Hook", tools: ["energy"] } };
  applyHands(vrState, input, { left: null, right: hand([-1, 0.4, 4], [0, 0, 0, 1], 1, 0) }, hookOnly);
  assert.equal(vrState.instruments.right.jaw, false);

  // A scene's starting grip stays closed until the trigger is pressed and released.
  const gripState = newState();
  const gripInput = newVrInput();
  gripState.instruments.left.gripping = true;
  gripState.instruments.left.jaw = true;
  const open = { left: hand([0, 1, 0]), right: null };
  const pull = { left: hand([0, 1, 0], [0, 0, 0, 1], 1), right: null };
  applyHands(gripState, gripInput, open, DEFAULT_INSTRUMENTS);
  assert.equal(gripState.instruments.left.jaw, true);
  applyHands(gripState, gripInput, { left: null, right: null }, DEFAULT_INSTRUMENTS);   // a starting grip survives a loss
  assert.equal(gripState.instruments.left.jaw, true);
  applyHands(gripState, gripInput, open, DEFAULT_INSTRUMENTS);
  applyHands(gripState, gripInput, pull, DEFAULT_INSTRUMENTS);
  assert.equal(gripState.instruments.left.jaw, true);
  applyHands(gripState, gripInput, open, DEFAULT_INSTRUMENTS);
  assert.equal(gripState.instruments.left.gripping, false);
  assert.equal(gripState.instruments.left.jaw, false);

  // A trigger or grip press makes that side active (the "active" exercise steps), but not one held through a clutch.
  const activeState = newState();
  const activeInput = newVrInput();
  const left = (trigger, grip = 0) => ({ left: hand([0, 1, 0], [0, 0, 0, 1], trigger, grip), right: null });
  applyHands(activeState, activeInput, left(1), DEFAULT_INSTRUMENTS);
  assert.equal(activeState.active, "right");
  applyHands(activeState, activeInput, left(0), DEFAULT_INSTRUMENTS);
  assert.equal(activeState.active, "right");
  applyHands(activeState, activeInput, left(0, 1), DEFAULT_INSTRUMENTS);
  assert.equal(activeState.active, "left");

  // A/X (goHome): the homing side ignores the hand and stays clutched while tick glides it home; the next hand frame
  // after the glide anchors at home, so the instrument does not jump back to the hand.
  const homeState = newState();
  const homeInput = newVrInput();
  const startRight = plain(homeState.instruments.right.pose.position);
  applyHands(homeState, homeInput, { left: null, right: hand([0.5, 1.2, 0.1]) }, DEFAULT_INSTRUMENTS);
  applyHands(homeState, homeInput, { left: null, right: hand([0.55, 1.2, 0.1]) }, DEFAULT_INSTRUMENTS);
  const away = plain(homeState.instruments.right.pose.position);
  goHome(homeState, "right");
  applyHands(homeState, homeInput, { left: null, right: hand([0.6, 1.2, 0.1]) }, DEFAULT_INSTRUMENTS);
  assert.equal(homeState.instruments.right.homing, true);
  closeVec(plain(homeState.instruments.right.pose.position), away);
  closeVec(plain(homeState.instruments.right.target.position), startRight);
  assert.equal(homeInput.offsets.right, null);
  for (let i = 0; i < 100 && homeState.instruments.right.homing; i++) {
    applyHands(homeState, homeInput, { left: null, right: hand([0.6, 1.2, 0.1]) }, DEFAULT_INSTRUMENTS);
    tick(homeState);
  }
  assert.equal(homeState.instruments.right.homing, false);
  applyHands(homeState, homeInput, { left: null, right: hand([0.7, 1.3, 0.1]) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(homeState.instruments.right.pose.position), startRight);
  applyHands(homeState, homeInput, { left: null, right: hand([0.71, 1.3, 0.1]) }, DEFAULT_INSTRUMENTS);
  closeVec(plain(homeState.instruments.right.pose.position), [startRight[0] + 0.01, startRight[1], startRight[2]]);

  // The frame sent while driving by hand matches the wire schema.
  const vrMessage = frameMessage(vrState, 1);
  assert.equal(vrMessage.controllers.right.tracked, true);
  assert.equal(vrMessage.controllers.right.position.length, 3);
}
console.log("VR hands clutch without jumps, hold through tracking loss, and act only after a release.");
