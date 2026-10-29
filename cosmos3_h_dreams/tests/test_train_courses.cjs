// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with: node cosmos3_h_dreams/tests/test_train_courses.cjs
// Course helpers in train_core.js: step checks, demo playback, path guide, safety meter and scores.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cosmos3_h_dreams/web/train_core.js"), "utf8"), context);
const c = context;
// Arrays built inside the vm context have another realm's Array.prototype; copy them so strict deepEqual compares values.
const plain = v => JSON.parse(JSON.stringify(v));
const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const HOME = { left: [-0.15, 1.0, -0.3], right: [0.15, 1.0, -0.3] };
const view = (over = {}) => ({
  active: "right",
  left: { position: [...HOME.left], trigger: 0, grip: 0 },
  right: { position: [...HOME.right], trigger: 0, grip: 0 },
  ...over,
});

// ---- Step checks: every kind, strictly in order -------------------------------------------
const steps = [
  { text: "reach", check: { reach: { side: "right", center: [0.2, 1.0, -0.36], radius_m: 0.015 } } },
  { text: "active", check: { active: "left" } },
  { text: "jaws", check: { jaws_closed: { side: "left", hold_s: 2 } } },
  { text: "energy", check: { energy: { side: "left", hold_s: 1 } } },
  { text: "moved", check: { moved: { side: "both", min_m: 0.05 } } },
  { text: "home", check: { home: { side: "both" } } },
];
const run = c.newStepRun(steps);
// Switching to left first does not complete step 2 early: steps complete in order.
assert.deepEqual(plain(c.updateSteps(run, view({ active: "left" }), 0)), []);
const atTarget = view({ right: { position: [0.2, 1.0, -0.35], trigger: 0, grip: 0 } });   // 1 cm from the centre
assert.deepEqual(plain(c.updateSteps(run, atTarget, 1)), [0]);
assert.deepEqual(plain(c.updateSteps(run, { ...atTarget, active: "left" }, 2)), [1]);
const jaws = { ...atTarget, active: "left", left: { position: [...HOME.left], trigger: 1, grip: 0 } };
assert.deepEqual(plain(c.updateSteps(run, jaws, 3)), []);
close(c.holdSeconds(run, 4), 1);
assert.deepEqual(plain(c.updateSteps(run, { ...jaws, left: { ...jaws.left, trigger: 0 } }, 4)), []);   // released: restart
assert.equal(c.holdSeconds(run, 4), null);
c.updateSteps(run, jaws, 5);
assert.deepEqual(plain(c.updateSteps(run, jaws, 6.9)), []);
assert.deepEqual(plain(c.updateSteps(run, jaws, 7)), [2]);
const energy = { ...jaws, left: { position: [...HOME.left], trigger: 0, grip: 1 } };
c.updateSteps(run, energy, 8);
assert.deepEqual(plain(c.updateSteps(run, energy, 9)), [3]);
// moved: path since the step began, summed over both sides.
const away = { ...energy, left: { position: [-0.15, 1.0, -0.33], trigger: 0, grip: 0 } };
assert.deepEqual(plain(c.updateSteps(run, away, 10)), []);   // 3 cm so far
const back = { ...away, left: { position: [...HOME.left], trigger: 0, grip: 0 } };
// 6 cm moved (step 4 done); right is still 7.8 cm from home, so step 5 waits.
assert.deepEqual(plain(c.updateSteps(run, back, 11)), [4]);
const home = view({ right: { position: [0.155, 1.0, -0.3], trigger: 0, grip: 0 } });   // 5 mm away: within 1 cm
assert.deepEqual(plain(c.updateSteps(run, home, 12)), [5]);
assert.deepEqual(plain(run.done.map(d => d.t_s)), [1, 2, 7, 9, 11, 12]);
assert.deepEqual(plain(c.updateSteps(run, home, 13)), []);   // nothing after the last step
// Several steps can complete in one update when all are already met.
const quick = c.newStepRun([steps[1], { text: "home", check: { home: { side: "left" } } }]);
assert.deepEqual(plain(c.updateSteps(quick, view({ active: "left" }), 0)), [0, 1]);
// viewOf reads the live page state.
const live = c.newState();
live.instruments.left.jaw = true;
assert.deepEqual(plain(c.viewOf(live).left), { position: HOME.left, trigger: 1, grip: 0 });

// ---- Demo playback: pose interpolation, pause, speed, restart, end, takeover --------------
const frame = (t, x, grip = 0, active = "right") => ({
  t_s: t, active,
  left: { position: [...HOME.left], orientation: [0, 0, 0, 1], trigger: 0, grip: 0 },
  right: { position: [x, 1.0, -0.3], orientation: [0, 0, 0, 1], trigger: 0, grip },
});
const demo = {
  duration_s: 2 / 30,
  frames: [frame(0, 0.15), frame(1 / 30, 0.16, 1), frame(2 / 30, 0.17, 1, "left")],
  markers: [{ t_s: 0, caption: "Start" }, { t_s: 1 / 30, caption: "Go" }],
};
close(c.demoPose(demo, 0.5 / 30).right.position[0], 0.155);
assert.equal(c.demoPose(demo, 0.5 / 30).right.grip, 0);   // switches hold the earlier frame's value
assert.equal(c.demoPose(demo, 1 / 30).right.grip, 1);
assert.equal(c.demoPose(demo, 99).active, "left");
assert.equal(c.markerIndex(demo.markers, 0.01), 0);
assert.equal(c.markerIndex(demo.markers, 1 / 30), 1);
assert.equal(c.markerIndex([], 1), -1);
const pb = c.newPlayback(demo, 1000);
c.advancePlayback(pb, 1010);
close(pb.positionS, 0.01);
c.setPlaying(pb, false, 1020);
c.advancePlayback(pb, 5000);
close(pb.positionS, 0.02);                                  // paused: the clock stands still
c.setPlaying(pb, true, 5000);
c.setSpeed(pb, 0.5, 5000);
c.advancePlayback(pb, 5020);
close(pb.positionS, 0.03);                                  // half speed
c.advancePlayback(pb, 9000);
assert.ok(pb.done && !pb.playing);
close(pb.positionS, demo.duration_s);
c.setPlaying(pb, true, 9001);
assert.equal(pb.playing, false);                            // finished: play does nothing until restart
c.restartPlayback(pb, 9001);
assert.ok(pb.playing && !pb.done && pb.positionS === 0 && pb.speed === 0.5);
// Takeover: the instruments continue from the demo's pose, jaws open and energy off.
const driven = c.newState();
c.applyDemoPose(driven, c.demoPose(demo, 2 / 30));
assert.equal(driven.active, "left");
assert.equal(driven.instruments.right.energy, true);
close(driven.instruments.right.pose.position[0], 0.17);
driven.held = true;
c.takeOver(driven);
assert.equal(driven.instruments.right.energy, false);
assert.equal(driven.held, false);
assert.deepEqual(plain(driven.instruments.right.target.position), plain(driven.instruments.right.pose.position));
c.tick(driven);
close(driven.instruments.right.pose.position[0], 0.17);    // no jump after the takeover
// Ending a demo (Stop demo, an incident, the video click) releases whatever it had pressed, and keeps the pose.
const pressedFrame = (t, x) => ({
  t_s: t, active: "right",
  left: { position: [...HOME.left], orientation: [0, 0, 0, 1], trigger: 1, grip: 0 },
  right: { position: [x, 1.0, -0.3], orientation: [0, 0, 0, 1], trigger: 0, grip: 1 },
});
const pressed = { duration_s: 2 / 30, frames: [pressedFrame(0, 0.15), pressedFrame(1 / 30, 0.16), pressedFrame(2 / 30, 0.17)], markers: [] };
const mid = c.newState();
c.applyDemoPose(mid, c.demoPose(pressed, 1 / 30));
assert.ok(mid.instruments.left.jaw && mid.instruments.right.energy, "the demo's jaws and energy are applied");
c.takeOver(mid);
assert.ok(!mid.instruments.left.jaw && !mid.instruments.right.jaw && !mid.instruments.left.energy && !mid.instruments.right.energy);
close(mid.instruments.right.pose.position[0], 0.16);
c.tick(mid);
assert.ok(!mid.instruments.right.energy && !mid.instruments.left.jaw, "still released after the next tick");
close(mid.instruments.right.pose.position[0], 0.16);
// driveDemo: pressed while playing; released once the demo has finished, even if its last frame held them.
const drive = c.newState();
const run2 = c.newPlayback(pressed, 0);
c.driveDemo(drive, run2, 10);
assert.ok(drive.instruments.left.jaw && drive.instruments.right.energy && !run2.done);
c.driveDemo(drive, run2, 5000);
assert.ok(run2.done);
assert.ok(!drive.instruments.left.jaw && !drive.instruments.right.energy, "released at the natural end");
close(drive.instruments.right.pose.position[0], 0.17);
c.driveDemo(drive, run2, 5033);
assert.ok(!drive.instruments.left.jaw && !drive.instruments.right.energy, "and stays released while the finished demo waits");
// A paused demo releases energy (kept on it would trip the energy-duration safety stop) but keeps the pose and the
// jaws, so a grasped tissue stays grasped; resuming switches energy on again from the demo's frames.
const longPressed = { duration_s: 4, frames: Array.from({ length: 121 }, (_, i) => pressedFrame(i / 30, 0.15 + 0.001 * i)), markers: [] };
const paused = c.newState();
const run3 = c.newPlayback(longPressed, 0);
c.driveDemo(paused, run3, 500);
assert.ok(paused.instruments.left.jaw && paused.instruments.right.energy, "pressed while playing");
let sent = c.frameMessage(paused, 0.5).controllers;
assert.equal(sent.left.trigger, 1);
assert.equal(sent.right.grip, 1);
c.setPlaying(run3, false, 500);
c.driveDemo(paused, run3, 520);
const heldX = paused.instruments.right.pose.position[0];
sent = c.frameMessage(paused, 0.52).controllers;
assert.ok(paused.instruments.left.jaw && !paused.instruments.left.energy && !paused.instruments.right.energy);
assert.equal(sent.left.trigger, 1);
assert.equal(sent.right.grip, 0);
for (const ms of [1000, 2500, 5000]) {                      // a long pause: still released, pose unchanged
  c.driveDemo(paused, run3, ms);
  assert.ok(!paused.instruments.right.energy && paused.instruments.left.jaw && !run3.done);
  assert.equal(c.frameMessage(paused, ms / 1000).controllers.right.grip, 0);
  close(paused.instruments.right.pose.position[0], heldX);
  close(run3.positionS, 0.5);
}
c.setPlaying(run3, true, 5000);
c.driveDemo(paused, run3, 5100);
assert.ok(paused.instruments.left.jaw && paused.instruments.right.energy, "pressed again after the resume");
assert.equal(c.frameMessage(paused, 5.1).controllers.right.grip, 1);
assert.ok(paused.instruments.right.pose.position[0] > heldX, "and the demo moves on");

// ---- Demo comparison numbers ------------------------------------------------------------
const times = c.stepTimes([{ text: "go", check: { reach: { side: "right", center: [0.16, 1, -0.3], radius_m: 0.001 } } },
  { text: "never", check: { active: "nobody" } }], demo);
assert.deepEqual(plain(times), [1 / 30, null]);
assert.deepEqual(plain(c.stepDurations([2, 5, null, null])), [2, 3, null, null]);
assert.deepEqual(plain(c.reportStepTimes({ steps: [{ index: 1, t_s: 4 }] }, 3)), [null, 4, null]);
const stats = c.demoStats(demo);
close(stats.path_m, 0.02);
close(stats.energy_s, 2 / 30);
assert.equal(c.elapsedText(41.4), "0:41");
assert.equal(c.elapsedText(125), "2:05");
assert.equal(c.roughLength({ time_limit_s: 90 }), "about 2 min");
assert.equal(c.roughLength({ time_limit_s: null, demo_duration_s: 6 }), "under 1 min");
assert.equal(c.roughLength({}), "");
assert.equal(c.autoValueText("max_path_deviation_m", 0.0123), "1.2 cm");
assert.equal(c.autoValueText("no_safety_stops", 1), "1 safety stop");
assert.equal(c.autoValueText("max_time_s", 75), "1:15");
assert.equal(c.autoValueText("max_path_deviation_m", null), "no demo to compare");

// ---- Path guide projection ----------------------------------------------------------------
assert.deepEqual(plain(c.projectTop([0, 1, -0.35], 200, 124)), [100, 62]);   // the centre of the bounds
const [rx, ry] = c.projectTop(HOME.right, 200, 124);
const [lx] = c.projectTop(HOME.left, 200, 124);
assert.ok(rx > 100 && lx < 100 && Math.abs(rx - 100 - (100 - lx)) < 1e-9, "x to the right, symmetric");
assert.ok(c.projectTop([0, 1, -0.55], 200, 124)[1] < ry, "forward (-z) is up");
assert.deepEqual(plain(c.projectTop([5, 1, -5], 200, 124)), plain(c.projectTop([0.3, 1, -0.6], 200, 124)));   // clamped
assert.ok(c.projectHeight(1.1, 124) < c.projectHeight(1.0, 124), "higher is up");
assert.equal(c.projectHeight(9, 124), 12);
const trail = [];
for (let i = 0; i < 5; i += 1) c.pushTrail(trail, [i, 0, 0], 3);
assert.deepEqual(plain(trail.map(p => p[0])), [2, 3, 4]);

// ---- Safety meter -----------------------------------------------------------------------
const rows = c.meterRows({ speed: 0.59, acceleration: 0.6, rotation: 0.995, energy_duration: 1.2 });
assert.deepEqual(plain(rows.map(r => [r.rule, r.level, r.text])), [
  ["speed", "ok", ""], ["acceleration", "near", "60% near"], ["rotation", "near", "99% near"],
  ["energy_duration", "stop", "120% stop"], ["energy_dose", "ok", ""], ["energy_sweep", "ok", ""],
]);
assert.equal(c.meterHint({ speed: 0.7, rotation: 0.65 }), "Speed: Move more slowly");
assert.equal(c.meterHint({ speed: 0.5 }), "All movements within limits");
assert.equal(c.meterHint(null), "All movements within limits");

// ---- Scores -----------------------------------------------------------------------------
const report = { auto_points: 50, auto_max: 70, instructor_max: 30 };
assert.equal(c.attemptScore({}), null);
assert.deepEqual(plain(c.attemptScore({ report, review: null })),
  { auto: 50, instructor: null, total: 50, max: 100, pending: true, percent: 50, text: "50 + ?" });
assert.equal(c.attemptScore({ report, review: { points: 30 } }).text, "80 / 100");
assert.equal(c.attemptScore({ report: { ...report, instructor_max: 0 }, review: null }).text, "50 / 70");
const sessions = [
  { id: "20261027-100000_A", trainee: "Dr Lin", exercise: { id: "x" }, report, review: { points: 30 } },
  { id: "20261027-090000_A", trainee: "Dr Lin", exercise: { id: "x" }, report, review: null },
  { id: "20261027-110000_B", trainee: "Dr Wu", exercise: { id: "x" }, report: { ...report, auto_points: 70 }, review: null },
  { id: "20261027-120000_A", trainee: "Dr Lin", exercise: null, report: null },
];
assert.equal(c.bestScore(sessions, "Dr Lin", "x"), 80);
assert.equal(c.bestScore(sessions, "Dr Lin", "y"), null);
assert.equal(c.bestScore(sessions, null, "x"), 80);
assert.deepEqual(plain(c.progressSeries(sessions, "Dr Lin", "x")), [
  { n: 1, id: "20261027-090000_A", percent: 50, pending: true, text: "50 + ?" },
  { n: 2, id: "20261027-100000_A", percent: 80, pending: false, text: "80 / 100" },
]);
assert.equal(c.traineeKey("  Dr   Lin "), "Dr Lin");
assert.equal(c.previousAttempt(sessions, "Dr Lin", "x", "20261027-100000_A").id, "20261027-090000_A");
assert.equal(c.previousAttempt(sessions, "Dr Lin", "x", "20261027-090000_A"), null);

// ---- Report-card radar ---------------------------------------------------------------------
const radarExercise = { time_limit_s: 90, rubric: [{ id: "path", auto: { max_path_deviation_m: 0.03 } }] };
const radarStats = { duration_s: 10, path_m: 0.8, energy_s: 0 };
const radarSummary = { duration_s: 50, path_length_m: { left: 0.4, right: 0.6 }, energy_s: { left: 0, right: 1 }, incidents: 1 };
const scored = c.skillScores({ path_deviation_m: 0.015, near_misses: 1, jerk_m_s3: 20, demo_jerk_m_s3: 2 }, radarSummary, radarExercise, radarStats);
assert.deepEqual(plain(scored.map(s => [s.key, s.score])), [
  ["time", 50],        // (90 - 50) / (90 - 10)
  ["accuracy", 75],    // 1 - 1.5 cm / 6 cm
  ["economy", 80],     // 0.8 m / 1.0 m
  ["energy", 50],      // (0 + 1) / (1 + 1)
  ["safety", 65],      // 1 - 0.25 - 0.1
  ["smoothness", 0],   // ten times the demo's jerk
]);
// Smoothness from corrections when the report has them: demo / yours, capped at 100.
const moves = c.skillScores({ path_deviation_m: null, near_misses: 0, corrections: 8, demo_corrections: 2,
  jerk_m_s3: 20, demo_jerk_m_s3: 2 }, radarSummary, radarExercise, null).find(s => s.key === "smoothness");
assert.equal(moves.score, 25);
assert.equal(moves.measured, "8 corrections (demo 2)");
assert.equal(c.skillScores({ near_misses: 0, corrections: 1, demo_corrections: 3 }, radarSummary, radarExercise, null)
  .find(s => s.key === "smoothness").score, 100);
// Without a demo only safety is left; better than the demo still caps at 100.
assert.deepEqual(plain(c.skillScores({ path_deviation_m: null, near_misses: 0 }, { duration_s: 5 }, radarExercise, null).map(s => [s.key, s.score])),
  [["safety", 100]]);
assert.equal(c.skillScores({ near_misses: 0 }, { ...radarSummary, duration_s: 5 }, radarExercise, radarStats)[0].score, 100);
assert.deepEqual(plain(c.skillDeltas(scored, [{ key: "time", score: 40 }])), [10, null, null, null, null, null]);
const [x, y] = c.radarPoint(100, 100, 50, 1, 4, 50);   // a quarter turn clockwise from the top, half way out
close(x, 125); close(y, 100);
console.log("train_courses ok");
