// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pure virtual-instrument state for the training page. No DOM access, so node tests can run it.
// Coordinates follow WebXR: metres, +x right, +y up, -z away from the trainee.
// SIDES and HOME_STEP_M are `var` so node vm test contexts can read them (top-level const is not a context property).
var SIDES = ["left", "right"];
const HOME_POSITIONS = { left: [-0.15, 1.0, -0.3], right: [0.15, 1.0, -0.3] };
const SMOOTHING = 0.35;      // fraction of the gap to the target closed each tick
var HOME_STEP_M = 0.003;     // per-tick cap while returning home (~0.09 m/s at 30 Hz)

function quatMultiply(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

function normalizeQuat(q) {
  const n = Math.hypot(...q);
  return q.map(v => v / n);
}

function axisAngle(axis, angle) {
  const s = Math.sin(angle / 2);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

function nlerp(a, b, t) {
  const sign = a.reduce((sum, v, i) => sum + v * b[i], 0) < 0 ? -1 : 1;
  return normalizeQuat(a.map((v, i) => v + (sign * b[i] - v) * t));
}

function newInstrument(side) {
  const position = HOME_POSITIONS[side];
  return {
    target: { position: [...position], orientation: [0, 0, 0, 1] },
    pose: { position: [...position], orientation: [0, 0, 0, 1] },
    jaw: false,
    energy: false,
    gripping: false,   // jaws locked closed (G, or a scene that starts holding tissue); G or a click releases them
    homing: false,
  };
}

// G locks the active instrument's jaws closed so they stay closed while the trainee works the other arm; G again
// unlocks them. Returns whether the jaws are now locked.
function toggleGripLock(state, side) {
  const instrument = state.instruments[side];
  instrument.gripping = !instrument.gripping;
  instrument.jaw = instrument.gripping;
  return instrument.gripping;
}

// A scene whose frame shows jaws already closed on tissue starts those instruments gripping
// (/healthz scenes[].start_jaws_closed), so the first frames do not tell the model to let go.
function applyStartGrip(state, sides) {
  for (const side of SIDES) {
    const instrument = state.instruments[side];
    const was = instrument.gripping;
    instrument.gripping = (sides ?? []).includes(side);
    if (instrument.gripping) instrument.jaw = true;
    else if (was) instrument.jaw = false;   // the previous scene's grip ends with it
  }
  return state;
}

function newState() {
  return { active: "right", held: false, instruments: { left: newInstrument("left"), right: newInstrument("right") } };
}

function applyInput(state, input, settings) {
  if (state.held) return state;
  // Zero input is a no-op, so a homing glide survives the per-frame call with no motion.
  if (!input.dx && !input.dy && !input.wheel && !input.roll) return state;
  const instrument = state.instruments[state.active];
  if (instrument.homing) {
    // The mouse takes over an R glide where the instrument is; keeping the home target would make it
    // lunge there at the normal smoothing rate (a speed spike the safety guard stops).
    instrument.target = { position: [...instrument.pose.position], orientation: [...instrument.pose.orientation] };
    instrument.homing = false;
  }
  const target = instrument.target;
  if (input.shift) {
    const yaw = axisAngle([0, 1, 0], -input.dx * settings.radiansPerPixel);
    const pitch = axisAngle([1, 0, 0], -input.dy * settings.radiansPerPixel);
    target.orientation = normalizeQuat(quatMultiply(yaw, quatMultiply(target.orientation, pitch)));
  } else {
    target.position[0] += input.dx * settings.metresPerPixel;
    target.position[2] += input.dy * settings.metresPerPixel;
  }
  target.position[1] -= input.wheel * settings.metresPerWheel;
  if (input.roll) {
    target.orientation = normalizeQuat(quatMultiply(target.orientation, axisAngle([0, 0, 1], input.roll)));
  }
  return state;
}

// `ticks` is the time since the last call in 30 Hz ticks (0.5 when the page runs at 60 Hz),
// so smoothing and the glide home move at the same speed at any page rate.
function tick(state, ticks = 1) {
  if (state.held) return state;   // a held pose must not drift toward an over-shot target
  const smoothing = 1 - Math.pow(1 - SMOOTHING, ticks);
  const homeStep = HOME_STEP_M * ticks;
  for (const side of SIDES) {
    const { pose, target, homing } = state.instruments[side];
    const gap = target.position.map((v, i) => v - pose.position[i]);
    const distance = Math.hypot(...gap);
    const fraction = homing && distance > 0 ? Math.min(1, homeStep / distance) : smoothing;
    pose.position = pose.position.map((v, i) => v + gap[i] * fraction);
    pose.orientation = nlerp(pose.orientation, target.orientation, homing ? fraction : smoothing);
    if (homing && distance <= homeStep) state.instruments[side].homing = false;
  }
  return state;
}

function frameMessage(state, tClient) {
  const controllers = {};
  for (const side of SIDES) {
    const { pose, jaw, energy } = state.instruments[side];
    controllers[side] = {
      position: [...pose.position],
      orientation: [...pose.orientation],
      trigger: jaw ? 1 : 0,
      grip: energy ? 1 : 0,
      tracked: true,
      buttons: [],
    };
  }
  return { type: "xr_frame", t_client: tClient, controllers };
}

function goHome(state, side) {
  const instrument = state.instruments[side];
  instrument.target = { position: [...HOME_POSITIONS[side]], orientation: [0, 0, 0, 1] };
  instrument.homing = true;
  return state;
}

function applyRestore(state, restore) {
  for (const side of SIDES) {
    const pose = restore?.[side];
    const instrument = state.instruments[side];
    instrument.energy = false;
    instrument.homing = false;
    if (!pose) continue;
    instrument.pose = { position: [...pose.position], orientation: [...pose.orientation] };
    instrument.target = { position: [...pose.position], orientation: [...pose.orientation] };
  }
  state.held = false;
  return state;
}

function quatConjugate(q) { return [-q[0], -q[1], -q[2], q[3]]; }

// ---- VR hands (train_vr.js) ----
// Real hands cannot be moved to the instrument's pose, so each side sends hand pose + an offset fixed at a clutch.
// `offsets[side]` null means "re-anchor on the next tracked pose". After a clutch the trigger and grip act only once
// released, so a finger already on the grip cannot fire energy on resume.
function newVrInput() {
  return { offsets: { left: null, right: null }, waitRelease: { left: true, right: true }, releasing: { left: false, right: false } };
}

function clutch(input, sides = SIDES) {
  for (const side of sides) {
    input.offsets[side] = null;
    input.waitRelease[side] = true;
    input.releasing[side] = false;
  }
  return input;
}

// ponytail: the position offset is in the room frame, so after a large clutch rotation "hand right" is "instrument
// right in the room", not along the instrument's own axes (as on a da Vinci). Upgrade: rotate the hand's position
// delta by the clutch-time orientation offset.
function applyHands(state, input, hands, instruments) {
  if (state.held) {   // a held pose ignores the hands; whatever ends the hold continues from the pose it leaves
    clutch(input);
    return state;
  }
  for (const side of SIDES) {
    const instrument = state.instruments[side];
    const hand = hands[side];
    if (!hand) {   // tracking lost: hold the pose and let go of the buttons; a scene's starting grip stays
      clutch(input, [side]);
      if (!instrument.gripping) instrument.jaw = false;
      instrument.energy = false;
      continue;
    }
    if (instrument.homing) {   // A/X glide (goHome): tick moves it; the hand re-anchors where the glide ends
      clutch(input, [side]);
      continue;
    }
    if (!input.offsets[side]) {
      const { position, orientation } = instrument.pose;
      input.offsets[side] = {
        position: position.map((v, i) => v - hand.position[i]),
        orientation: normalizeQuat(quatMultiply(quatConjugate(hand.orientation), orientation)),
      };
    }
    const offset = input.offsets[side];
    const position = hand.position.map((v, i) => v + offset.position[i]);
    const orientation = normalizeQuat(quatMultiply(hand.orientation, offset.orientation));
    instrument.pose = { position, orientation };
    instrument.target = { position: [...position], orientation: [...orientation] };
    const trigger = hand.trigger >= PRESSED;
    const grip = hand.grip >= PRESSED;
    if (input.waitRelease[side]) {
      if (trigger || grip) {
        instrument.jaw = instrument.gripping;
        instrument.energy = false;
        continue;
      }
      input.waitRelease[side] = false;
    }
    if (trigger || grip) state.active = side;   // the hand that acts is the active one (Tab with the mouse)
    if (instrument.gripping) {   // a scene's starting grip: a trigger press and release lets go
      if (trigger) input.releasing[side] = true;
      else if (input.releasing[side]) {
        instrument.gripping = false;
        input.releasing[side] = false;
      }
      instrument.jaw = instrument.gripping;
    } else {
      instrument.jaw = trigger && hasTool(instruments, side, "jaws");
    }
    instrument.energy = grip && hasTool(instruments, side, "energy");
  }
  return state;
}

// ---- Courses (pure): guided steps, demo playback, scores ---------------------------------
// Exercise steps arrive from /api/courses as {text, check: {kind: argument}}; demos as
// {duration_s, frames: [{t_s, active, left: {position, orientation, trigger, grip}, right}], markers}.
var DEMO_FPS = 30;              // courses.FPS
var HOME_TOLERANCE_M = 0.01;    // a "home" step: within 1 cm of the home position
var PRESSED = 0.5;              // trigger/grip value that counts as jaws closed / energy on (safety.ENERGY_ON)

function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
function checkSides(side) { return side === "both" ? SIDES : [side]; }

// What the step checks read: the active side and each instrument's position, trigger and grip.
// Demo frames already have this shape.
function viewOf(state) {
  const view = { active: state.active };
  for (const side of SIDES) {
    const { pose, jaw, energy } = state.instruments[side];
    view[side] = { position: [...pose.position], trigger: jaw ? 1 : 0, grip: energy ? 1 : 0 };
  }
  return view;
}

function newStepRun(steps) {
  return { steps, index: 0, stepStartS: null, holdSinceS: null, pathM: { left: 0, right: 0 }, last: null, done: [] };
}

function holdMet(run, on, holdS, tS) {
  if (!on) { run.holdSinceS = null; return false; }
  if (run.holdSinceS === null) run.holdSinceS = tS;
  return tS - run.holdSinceS >= holdS - 1e-9;
}

function checkMet(check, view, run, tS) {
  const [kind, arg] = Object.entries(check)[0];
  switch (kind) {
    case "reach": return distance(view[arg.side].position, arg.center) <= arg.radius_m;
    case "active": return view.active === arg;
    case "jaws_closed": return holdMet(run, view[arg.side].trigger >= PRESSED, arg.hold_s, tS);
    case "jaws_open": return view[arg.side].trigger < PRESSED;
    case "energy": return holdMet(run, view[arg.side].grip >= PRESSED, arg.hold_s, tS);
    case "home":
      return checkSides(arg.side).every(side => distance(view[side].position, HOME_POSITIONS[side]) <= HOME_TOLERANCE_M);
    case "moved": return checkSides(arg.side).reduce((sum, side) => sum + run.pathM[side], 0) >= arg.min_m;
    default: return false;
  }
}

// Feed one view at time tS (seconds); steps complete strictly in order. Returns the indices completed now.
function updateSteps(run, view, tS) {
  if (run.last) for (const side of SIDES) run.pathM[side] += distance(run.last[side], view[side].position);
  run.last = { left: [...view.left.position], right: [...view.right.position] };
  const completed = [];
  while (run.index < run.steps.length) {
    if (run.stepStartS === null) {   // a step begins: its hold and path start from here
      run.stepStartS = tS;
      run.holdSinceS = null;
      run.pathM = { left: 0, right: 0 };
    }
    if (!checkMet(run.steps[run.index].check, view, run, tS)) break;
    run.done.push({ index: run.index, t_s: tS });
    completed.push(run.index);
    run.index += 1;
    run.stepStartS = null;
  }
  return completed;
}

// Seconds the current hold step (jaws_closed or energy) has been held, or null.
function holdSeconds(run, tS) {
  const step = run.steps[run.index];
  if (!step || run.holdSinceS === null || !("jaws_closed" in step.check || "energy" in step.check)) return null;
  return tS - run.holdSinceS;
}

// Live guidance for the current step, shown beside its caption: how far a reach target is and whether to go up or
// down (the scroll wheel; scrolling down lowers the instrument), or how much of a "moved" distance is done.
// Returns "" for steps without a distance (jaws, energy, active, home). `vr` words it for the controllers (no Tab or
// wheel: the hand moves whichever instrument it holds).
var HEIGHT_OK_M = 0.005;        // within 5 mm of the target height counts as "height ✓"
function cm(metres) { return `${(metres * 100).toFixed(1)} cm`; }
function stepGuidance(check, view, run, vr = false) {
  const [kind, arg] = Object.entries(check)[0];
  if (kind === "reach") {
    const position = view[arg.side].position;
    const away = Math.max(0, distance(position, arg.center) - arg.radius_m);
    const dy = position[1] - arg.center[1];
    const height = Math.abs(dy) <= HEIGHT_OK_M ? "height ✓"
      : dy > 0 ? `${cm(dy)} too high · ${vr ? "lower it" : "scroll down"}`
        : `${cm(-dy)} too low · ${vr ? "raise it" : "scroll up"}`;
    const tab = vr || view.active === arg.side ? "" : `Tab to the ${arg.side} instrument · `;
    return `${tab}${cm(away)} to the target · ${height}`;
  }
  if (kind === "moved") {
    const done = checkSides(arg.side).reduce((sum, side) => sum + run.pathM[side], 0);
    return `${cm(Math.min(done, arg.min_m))} / ${cm(arg.min_m)}`;
  }
  return "";
}

// Progress while the simulation restarts (/healthz `simulation`, read before the reset gave `epoch`): the server
// must see the reset, recalibrate on a still pose, then show a chunk of new video.
function prepView(sim, epoch) {
  if (!sim || sim.epoch <= epoch) return { percent: 5, text: "Resetting the simulation…", done: false };
  if (sim.chunks_shown < 1 && !sim.calibrated) {
    return { percent: 10 + Math.round(50 * (sim.calibration_fraction || 0)), text: "Calibrating · hold still", done: false };
  }
  if (sim.chunks_shown < 1) return { percent: 80, text: "Generating the first frames…", done: false };
  return { percent: 100, text: "Ready", done: true };
}

// The demo pose at tS: positions interpolated between 30 Hz frames, trigger/grip/active from the earlier one.
function demoPose(demo, tS) {
  const frames = demo.frames;
  const i = Math.max(0, Math.min(frames.length - 1, Math.floor(tS * DEMO_FPS + 1e-9)));
  const j = Math.min(frames.length - 1, i + 1);
  const u = j === i ? 0 : Math.max(0, Math.min(1, (tS - frames[i].t_s) / (frames[j].t_s - frames[i].t_s)));
  const pose = { active: frames[i].active || "right" };
  for (const side of SIDES) {
    const a = frames[i][side];
    const b = frames[j][side];
    pose[side] = {
      position: a.position.map((v, k) => v + (b.position[k] - v) * u),
      orientation: nlerp(a.orientation, b.orientation, u),
      trigger: a.trigger,
      grip: a.grip,
    };
  }
  return pose;
}

// Playback clock: position advances with wall time × speed while playing, and stops at the end.
function newPlayback(demo, nowMs) { return { demo, positionS: 0, speed: 1, playing: true, done: false, lastMs: nowMs }; }
function advancePlayback(pb, nowMs) {
  if (pb.playing) pb.positionS += ((nowMs - pb.lastMs) / 1000) * pb.speed;
  pb.lastMs = nowMs;
  if (pb.positionS >= pb.demo.duration_s) {
    pb.positionS = pb.demo.duration_s;
    pb.playing = false;
    pb.done = true;
  }
  return pb;
}
function setPlaying(pb, playing, nowMs) { advancePlayback(pb, nowMs); pb.playing = playing && !pb.done; return pb; }
function setSpeed(pb, speed, nowMs) { advancePlayback(pb, nowMs); pb.speed = speed; return pb; }
function restartPlayback(pb, nowMs) { Object.assign(pb, { positionS: 0, playing: true, done: false, lastMs: nowMs }); return pb; }
// Index of the last marker at or before tS, or -1.
function markerIndex(markers, tS) {
  let index = -1;
  markers.forEach((marker, i) => { if (marker.t_s <= tS + 1e-9) index = i; });
  return index;
}

// Drive the page's instruments from a demo pose (no smoothing: the demo is already smooth).
function applyDemoPose(state, pose) {
  state.active = pose.active;
  for (const side of SIDES) {
    const instrument = state.instruments[side];
    const { position, orientation, trigger, grip } = pose[side];
    instrument.pose = { position: [...position], orientation: [...orientation] };
    instrument.target = { position: [...position], orientation: [...orientation] };
    instrument.jaw = trigger >= PRESSED;
    instrument.energy = grip >= PRESSED;
    instrument.gripping = false;
    instrument.homing = false;
  }
  return state;
}

// The mouse takes over from the current pose: jaws open and energy off until the trainee acts.
function takeOver(state) {
  for (const side of SIDES) {
    const instrument = state.instruments[side];
    instrument.jaw = false;
    instrument.energy = false;
    instrument.gripping = false;
    instrument.homing = false;
    instrument.target = { position: [...instrument.pose.position], orientation: [...instrument.pose.orientation] };
  }
  state.held = false;
  return state;
}

// One tick of a playing demo: advance the clock and drive the instruments from the demo's pose. While the demo
// is paused the pose and jaws hold but energy is released (a held press would keep sending fresh frames and
// trip the energy-duration safety stop; held jaws are harmless and keep a grasped tissue grasped); resuming
// re-applies the demo's own energy. Once the demo has finished nothing stays pressed (jaws open, energy off),
// whatever its last frame held.
function driveDemo(state, pb, nowMs) {
  advancePlayback(pb, nowMs);
  applyDemoPose(state, demoPose(pb.demo, pb.positionS));
  if (pb.done) {
    takeOver(state);
  } else if (!pb.playing) {
    for (const side of SIDES) state.instruments[side].energy = false;
  }
  return state;
}

// Completion time (s from the demo start) of each step when the demo is run through the checks; null if never.
function stepTimes(steps, demo) {
  const run = newStepRun(steps);
  for (const frame of demo.frames) {
    updateSteps(run, frame, frame.t_s);
    if (run.index >= steps.length) break;
  }
  return steps.map((_, i) => { const done = run.done.find(d => d.index === i); return done ? done.t_s : null; });
}

// Active time of an exercise attempt: seconds since it started minus the time held (paused, or stopped by a safety
// stop), as the server's recorder measures it for the report. `held` is sampled on every call, so call it each tick;
// a hold counts from the first call that sees it.
function newActiveClock(startMs) {
  return { startMs, heldMs: 0, heldSinceMs: null };
}

function activeSeconds(clock, nowMs, held) {
  if (held && clock.heldSinceMs === null) {
    clock.heldSinceMs = nowMs;
  } else if (!held && clock.heldSinceMs !== null) {
    clock.heldMs += nowMs - clock.heldSinceMs;
    clock.heldSinceMs = null;
  }
  const open = clock.heldSinceMs === null ? 0 : nowMs - clock.heldSinceMs;
  return Math.max(0, (nowMs - clock.startMs - clock.heldMs - open) / 1000);
}

// Report step times (active seconds: real time without pauses and safety stops; video time in old reports) as an
// array by index (null = not completed).
function reportStepTimes(report, count) {
  return Array.from({ length: count }, (_, i) => {
    const step = (report.steps || []).find(s => s.index === i);
    return step ? step.t_s : null;
  });
}

// Seconds spent on each step from completion times; null once a step was not reached.
function stepDurations(times) {
  let previous = 0;
  return times.map(t => {
    if (t === null || t === undefined) return null;
    const duration = t - previous;
    previous = t;
    return duration;
  });
}

// Report-card comparison numbers for a demo: duration, path length (both sides) and energy time.
function demoStats(demo) {
  let path = 0;
  let energy = 0;
  for (let i = 1; i < demo.frames.length; i += 1) {
    const [a, b] = [demo.frames[i - 1], demo.frames[i]];
    for (const side of SIDES) {
      path += distance(a[side].position, b[side].position);
      if (b[side].grip >= PRESSED) energy += b.t_s - a.t_s;
    }
  }
  return { duration_s: demo.duration_s, path_m: path, energy_s: energy };
}

function elapsedText(seconds) {
  const t = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
}

function roughLength(exercise) {
  const seconds = exercise.time_limit_s ?? exercise.demo_duration_s;
  if (!seconds) return "";
  return seconds < 60 ? "under 1 min" : `about ${Math.round(seconds / 60)} min`;
}

// A rubric auto item's measured value as text.
function autoValueText(kind, value) {
  if (value === null || value === undefined) return "no demo to compare";
  switch (kind) {
    case "no_safety_stops": return `${value} safety stop${value === 1 ? "" : "s"}`;
    case "max_time_s": return elapsedText(value);
    case "max_path_deviation_m": return `${(value * 100).toFixed(1)} cm`;
    case "max_energy_s": return `${Number(value).toFixed(1)} s`;
    case "max_near_misses": return `${value} near miss${value === 1 ? "" : "es"}`;
    case "all_steps": return `${value} step${value === 1 ? "" : "s"} done`;
    default: return String(value);
  }
}

// Score of an attempt from session.json (report) and review.json (review), or null for free practice.
function attemptScore(session) {
  const report = session && session.report;
  if (!report) return null;
  const review = session.review || null;
  const instructor = review ? Number(review.points) || 0 : null;
  const pending = report.instructor_max > 0 && review === null;
  const max = report.auto_max + report.instructor_max;
  const total = report.auto_points + (instructor || 0);
  return {
    auto: report.auto_points, instructor, total, max, pending,
    percent: max > 0 ? Math.round((100 * total) / max) : null,
    text: pending ? `${report.auto_points} + ?` : `${total} / ${max}`,
  };
}

// Normalise a typed trainee name as the recorder stores it (whitespace collapsed).
function traineeKey(name) { return String(name || "").trim().split(/\s+/).join(" "); }

// Scored attempts at one exercise, oldest first; trainee null means everyone.
function exerciseAttempts(sessions, trainee, exerciseId) {
  return sessions
    .filter(s => s.report && s.exercise && s.exercise.id === exerciseId && (trainee === null || s.trainee === trainee))
    .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

function bestScore(sessions, trainee, exerciseId) {
  const percents = exerciseAttempts(sessions, trainee, exerciseId).map(s => attemptScore(s).percent).filter(p => p !== null);
  return percents.length ? Math.max(...percents) : null;
}

// Points of the dashboard progress chart: one per scored attempt, oldest first.
function progressSeries(sessions, trainee, exerciseId) {
  return exerciseAttempts(sessions, trainee, exerciseId).map((session, i) => {
    const score = attemptScore(session);
    return { n: i + 1, id: session.id, percent: score.percent, pending: score.pending, text: score.text };
  });
}

// ---- Report-card radar (pure): six skills scored 0-100, where 100 is the demo's level -------
// Illustrative, like the rubric: each axis maps one report measure onto 0-100 against the demo.
// An axis without the data it needs (no demo, no deviation, no corrections or jerk) is left out.
function skillScores(report, summary, exercise, stats) {
  const pct = v => Math.round(clampTo(v, 0, 1) * 100);
  const sum = values => (values ? (values.left || 0) + (values.right || 0) : 0);
  const rubricLimit = kind => {
    const item = (exercise.rubric || []).find(r => r.auto && kind in r.auto);
    return item ? item.auto[kind] : null;
  };
  const you = Number(summary.duration_s) || 0;
  const stops = summary.incidents || 0;
  const near = report.near_misses || 0;
  const skills = [];
  if (stats) {
    // Scripted demos are far faster than a person, so time runs from the demo (100) to the limit (0).
    const limit = exercise.time_limit_s || rubricLimit("max_time_s") || 2 * Math.max(stats.duration_s, 1);
    skills.push({ key: "time", name: "Time", score: pct((limit - you) / Math.max(limit - stats.duration_s, 1e-6)),
      measured: `${elapsedText(you)} (demo ${elapsedText(stats.duration_s)}, limit ${elapsedText(limit)})`,
      short: `${elapsedText(you)} vs ${elapsedText(stats.duration_s)}` });
  }
  if (report.path_deviation_m !== null && report.path_deviation_m !== undefined) {
    const allowed = rubricLimit("max_path_deviation_m") || 0.03;   // 0 at twice the rubric's allowance
    const dev = report.path_deviation_m;
    skills.push({ key: "accuracy", name: "Path accuracy", score: pct(1 - dev / (2 * allowed)),
      measured: `${cm(dev)} from the demo path`, short: `${cm(dev)} off` });
  }
  if (stats) {
    const path = sum(summary.path_length_m);
    skills.push({ key: "economy", name: "Economy", score: path > 0 ? pct(stats.path_m / path) : 100,
      measured: `${path.toFixed(2)} m (demo ${stats.path_m.toFixed(2)} m)`, short: `${path.toFixed(2)} vs ${stats.path_m.toFixed(2)} m` });
    const energy = sum(summary.energy_s);   // one second of slack so a demo with no energy is reachable
    skills.push({ key: "energy", name: "Energy", score: pct((stats.energy_s + 1) / (energy + 1)),
      measured: `${energy.toFixed(1)} s (demo ${stats.energy_s.toFixed(1)} s)`, short: `${energy.toFixed(1)} vs ${stats.energy_s.toFixed(1)} s` });
  }
  skills.push({ key: "safety", name: "Safety", score: pct(1 - 0.25 * stops - 0.1 * near),
    measured: `${stops} safety stop${stops === 1 ? "" : "s"}, ${near} near miss${near === 1 ? "" : "es"}`,
    short: `${stops} stop · ${near} near` });
  if (report.corrections > 0 && report.demo_corrections > 0) {
    // Speed peaks: a smooth reach has one, hesitant stop-and-go motion one per correction.
    const you = report.corrections, demo = report.demo_corrections;
    skills.push({ key: "smoothness", name: "Smoothness", score: pct(demo / you),
      measured: `${you} correction${you === 1 ? "" : "s"} (demo ${demo})`, short: `${you} vs ${demo} moves` });
  } else if (report.jerk_m_s3 > 0 && report.demo_jerk_m_s3 > 0) {   // reports from before corrections
    const ratio = report.jerk_m_s3 / report.demo_jerk_m_s3;   // 100 at or below the demo, 0 at ten times it
    skills.push({ key: "smoothness", name: "Smoothness", score: pct(1 - Math.log10(Math.max(ratio, 1))),
      measured: `jerk ${ratio.toFixed(1)}× the demo`, short: `jerk ${ratio.toFixed(1)}× demo` });
  }
  return skills;
}

// Change per skill since the previous run; null for a skill the previous run did not have.
function skillDeltas(skills, previous) {
  return skills.map(skill => {
    const before = (previous || []).find(p => p.key === skill.key);
    return before ? skill.score - before.score : null;
  });
}

// The scored attempt before `id` by the same trainee at the same exercise, or null.
function previousAttempt(sessions, trainee, exerciseId, id) {
  const earlier = exerciseAttempts(sessions, trainee, exerciseId).filter(s => s.id < id);
  return earlier.length ? earlier[earlier.length - 1] : null;
}

// Radar point for axis i of n at score v (0-100), clockwise from the top.
function radarPoint(cx, cy, radius, i, n, v) {
  const angle = -Math.PI / 2 + (i * 2 * Math.PI) / n;
  return [cx + Math.cos(angle) * radius * v / 100, cy + Math.sin(angle) * radius * v / 100];
}

// ---- Path guide (pure): a top view (x right, forward up) and a height bar ------------------
var GUIDE_BOUNDS = { x: [-0.3, 0.3], z: [-0.6, -0.1], y: [0.85, 1.15] };
var GUIDE_PAD = 12;
var GUIDE_TRAIL = 90;           // 3 s of trail at 30 Hz

function clampTo(value, lo, hi) { return Math.min(hi, Math.max(lo, value)); }
function guideScale(width, height) {
  const [x0, x1] = GUIDE_BOUNDS.x;
  const [z0, z1] = GUIDE_BOUNDS.z;
  return Math.min((width - 2 * GUIDE_PAD) / (x1 - x0), (height - 2 * GUIDE_PAD) / (z1 - z0));
}
// Pixel [x, y] of a position in a width × height top view; points outside the bounds sit on the edge.
function projectTop(position, width, height) {
  const scale = guideScale(width, height);
  const [x0, x1] = GUIDE_BOUNDS.x;
  const [z0, z1] = GUIDE_BOUNDS.z;
  return [
    width / 2 + (clampTo(position[0], x0, x1) - (x0 + x1) / 2) * scale,
    height / 2 + (clampTo(position[2], z0, z1) - (z0 + z1) / 2) * scale,
  ];
}
// Pixel y of a height on the bar (higher is up).
function projectHeight(y, height) {
  const [y0, y1] = GUIDE_BOUNDS.y;
  return GUIDE_PAD + (1 - (clampTo(y, y0, y1) - y0) / (y1 - y0)) * (height - 2 * GUIDE_PAD);
}
function pushTrail(trail, position, max) {
  trail.push([...position]);
  if (trail.length > max) trail.splice(0, trail.length - max);
  return trail;
}

// ---- Safety meter (pure) ---------------------------------------------------------------
var SAFETY_RULES = [
  { rule: "speed", label: "Speed", hint: "Move more slowly" },
  { rule: "acceleration", label: "Acceleration", hint: "Start and stop more gently" },
  { rule: "rotation", label: "Rotation", hint: "Turn the wrist more slowly" },
  { rule: "energy_duration", label: "Energy held", hint: "Release F sooner" },
  { rule: "energy_dose", label: "Energy in window", hint: "Wait longer between energy activations" },
  { rule: "energy_sweep", label: "Energy sweep", hint: "Keep the instrument still while energy is on" },
];
// One row per rule: grey ("ok") below 60 %, amber ("near") from 60 % to 99 %, red ("stop") at 100 % or more.
function meterRows(ratios) {
  return SAFETY_RULES.map(({ rule, label }) => {
    const ratio = Number(ratios && ratios[rule]) || 0;
    const percent = Math.floor(ratio * 100);
    const level = ratio >= 1 ? "stop" : ratio >= 0.6 ? "near" : "ok";
    const text = level === "near" ? `${percent}% near` : level === "stop" ? `${percent}% stop` : "";
    return { rule, label, ratio, percent, level, text };
  });
}
function meterHint(ratios) {
  const worst = meterRows(ratios).reduce((a, b) => (b.ratio > a.ratio ? b : a));
  if (worst.ratio < 0.6) return "All movements within limits";
  return `${worst.label}: ${SAFETY_RULES.find(r => r.rule === worst.rule).hint}`;
}

// ---- Latency readout (pure) ------------------------------------------------------------
// /healthz `latency.status` → the word shown beside the status dot (never colour alone);
// "behind" because playback is slower than the input adds "(playback)".
var LATENCY_WORDS = { ok: "keeping up", tight: "near the limit", behind: "falling behind" };
function secondsText(ms) {
  const tenths = Math.round(ms / 100);
  return tenths < 100 ? `${(tenths / 10).toFixed(1)} s` : `${Math.round(ms / 1000)} s`;
}
// Texts for the readout under the video from /healthz `latency` (null before the first chunk: dashes).
function latencyView(latency) {
  if (!latency) return { generation: "—", response: "—", status: null, word: "—", title: "" };
  const frames = latency.frame_ms > 0 ? Math.round(latency.chunk_ms / latency.frame_ms) : null;
  return {
    generation: `${Math.round(latency.chunk_ms)} ms / chunk · ${Math.round(latency.frame_ms)} ms / frame`,
    response: latency.response_ms == null ? "—" : `~${secondsText(latency.response_ms)} action → frame`,
    status: LATENCY_WORDS[latency.status] ? latency.status : null,
    word: !LATENCY_WORDS[latency.status] ? "—"
      : latency.status === "behind" && latency.reason === "playback" ? `${LATENCY_WORDS.behind} (playback)`
      : LATENCY_WORDS[latency.status],
    title: latency.budget_ms == null || frames === null ? ""
      : `${frames} frames per chunk; must finish within ${Math.round(latency.budget_ms)} ms to keep up with your input`,
  };
}

// ---- Input rate (pure) -------------------------------------------------------------------
// The time one chunk may take at `fps` (/healthz `input_rate.chunk_frames` frames per chunk).
function inputRateBudgetText(fps, chunkFrames) {
  if (!(fps > 0) || !chunkFrames) return "";
  return `${Math.round(chunkFrames * 1000 / fps)} ms per chunk`;
}
// Advice while generation falls behind: the highest whole rate whose budget fits the measured
// chunk time with 10 % to spare; "" otherwise.
function inputRateAdvice(latency, rate) {
  if (!latency || latency.status !== "behind" || latency.reason === "playback" || !rate.chunk_frames) return "";
  const chunk = Math.round(latency.chunk_ms);
  const fits = Math.floor(rate.chunk_frames * 1000 / (latency.chunk_ms * 1.1));
  if (fits < rate.min) return `Chunks take ${chunk} ms; even ${rate.min} Hz cannot keep up.`;
  return `Chunks take ${chunk} ms: try ${fits} Hz or lower (${inputRateBudgetText(fits, rate.chunk_frames)}).`;
}

// Mouse wheels report a notch as ~100 px (Chrome), 3 lines (Firefox) or less (high-resolution
// wheels, touchpads); count notches so one notch moves the same distance in every browser.
const WHEEL_PIXELS_PER_NOTCH = 100;
const WHEEL_LINES_PER_NOTCH = 3;
function wheelNotches(deltaY, deltaMode) {
  if (deltaMode === 1) return deltaY / WHEEL_LINES_PER_NOTCH;   // DOM_DELTA_LINE
  if (deltaMode === 2) return Math.sign(deltaY);                // DOM_DELTA_PAGE: one notch per page
  return deltaY / WHEEL_PIXELS_PER_NOTCH;                       // DOM_DELTA_PIXEL
}

// ---- Instruments: which tools each side has, from /healthz `instruments` ----------------
// Both tools on both sides until /healthz answers (instruments.DEFAULT_INSTRUMENTS).
var DEFAULT_INSTRUMENTS = { left: { name: "Left", tools: ["jaws", "energy"] }, right: { name: "Right", tools: ["jaws", "energy"] } };
var TOOL_WORDS = { jaws: "jaws", energy: "energy" };
function hasTool(instruments, side, tool) { return (instruments?.[side]?.tools ?? []).includes(tool); }
// "Left · Grasper", or just "Left" when the config names the instrument after its side.
function instrumentLabel(instruments, side) {
  const sideWord = side === "left" ? "Left" : "Right";
  const name = instruments?.[side]?.name;
  return name && name.toLowerCase() !== side ? `${sideWord} · ${name}` : sideWord;
}
// Why a tool key did nothing on the active instrument, naming where the tool is instead (with the mouse; in `vr` each
// hand already holds its own instrument).
function missingToolHint(instruments, side, tool, vr = false) {
  const name = instruments?.[side]?.name ?? side;
  const other = SIDES.find(s => s !== side);
  const elsewhere = !vr && hasTool(instruments, other, tool) ? ` · Tab switches to the ${instruments[other].name}` : "";
  return `The ${name} has no ${TOOL_WORDS[tool]}${elsewhere}`;
}
// One HUD line per instrument, listing only the tools it has; `vr` says how a controller releases locked jaws.
function instrumentStatus(instruments, side, instrument, energySeconds, vr = false) {
  const parts = [];
  if (hasTool(instruments, side, "jaws")) {
    const release = vr ? "press and release trigger to let go" : "G or click releases";
    parts.push(instrument.gripping ? `jaws locked closed (${release})` : `jaws ${instrument.jaw ? "closed" : "open"}`);
  }
  if (hasTool(instruments, side, "energy")) {
    parts.push(instrument.energy ? `energy ON ${energySeconds.toFixed(1)}s` : "energy off");
  }
  return `${instrumentLabel(instruments, side)}: ${parts.join(", ") || "no tools"}`;
}
