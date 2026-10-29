// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with: node cosmos3_h_dreams/tests/test_dashboard.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({ document: { getElementById: () => null } });
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cosmos3_h_dreams/web/dashboard.js"), "utf8"), context);
// Arrays built inside the vm context have another realm's Array.prototype; copy them so strict deepEqual compares values.
const plain = v => Array.from(v);
// An old recording: no video_duration_s or t_s, duration_s and the event times are all video time.
const session = { summary: { duration_s: 20 } };
const events = [
  { kind: "start", t_video_s: 0 },
  { kind: "energy_on", side: "right", t_video_s: 2 },
  { kind: "energy_off", side: "right", t_video_s: 5 },
  { kind: "incident", t_video_s: 10, violations: [{ message: "Right instrument: speed 0.90 m/s, limit 0.25 m/s" }] },
  { kind: "resume", t_video_s: 10 },
  { kind: "energy_on", side: "left", t_video_s: 18 },
  { kind: "end", t_video_s: 20 },
];
const markers = context.timelineMarkers(session, events);
assert.deepEqual(plain(markers.map(m => m.kind)), ["energy", "incident", "tick", "energy"]);
const [energy, incident, tick, open] = markers;
assert.equal(energy.start, 0.1);
assert.equal(energy.end, 0.25);
assert.equal(energy.label, "right energy 3.0s");
assert.equal(incident.start, 0.5);
assert.equal(incident.seek_s, 7);
assert.match(incident.label, /speed 0.90/);
assert.equal(tick.label, "resume");
assert.equal(open.end, 1);
assert.deepEqual(plain(context.timelineMarkers({ summary: { duration_s: 0 } }, events).map(m => m.start)), [0, 0, 0, 0]);

// Step events become step markers.
const stepped = context.timelineMarkers(session, [{ kind: "step", index: 0, t_video_s: 4 }]);
assert.deepEqual(plain(stepped.map(m => [m.kind, m.start, m.label, m.seek_s])), [["step", 0.2, "step 1 done", 1]]);

// A recording with active time: duration_s and t_s are real seconds (here 8 s of the 20 s of video), but markers
// sit on the video's timeline, so they follow video_duration_s and t_video_s and still seek the video.
const timed = { summary: { duration_s: 8, video_duration_s: 20 } };
const timedEvents = [
  { kind: "start", t_s: 0, t_video_s: 0 },
  { kind: "step", index: 0, t_s: 2, t_video_s: 4 },
  { kind: "incident", t_s: 4, t_video_s: 10, violations: [{ message: "Right instrument: speed 0.90 m/s, limit 0.25 m/s" }] },
  { kind: "end", t_s: 8, t_video_s: 20 },
];
const timedMarkers = context.timelineMarkers(timed, timedEvents);
assert.deepEqual(plain(timedMarkers.map(m => [m.kind, m.start, m.end, m.seek_s])), [["step", 0.2, 0.2, 1], ["incident", 0.5, 0.5, 7]]);
// The time a marker is listed with is active time; without t_s (an old recording) it falls back to video time.
assert.deepEqual(plain(timedMarkers.map(m => m.t_s)), [2, 4]);
assert.deepEqual(plain(markers.map(m => m.t_s)), [2, 10, 10, 18]);
const slowEnergy = context.timelineMarkers(timed, [
  { kind: "energy_on", side: "left", t_s: 1, t_video_s: 2 },
  { kind: "energy_off", side: "left", t_s: 2, t_video_s: 5 },
]);
assert.deepEqual(plain(slowEnergy.map(m => [m.start, m.end, m.label])), [[0.1, 0.25, "left energy 3.0s"]]);

// The video length: video_duration_s, or duration_s in recordings made before duration_s became active time.
assert.equal(context.videoDurationOf({ duration_s: 8, video_duration_s: 20 }), 20);
assert.equal(context.videoDurationOf({ duration_s: 20 }), 20);
assert.equal(context.videoDurationOf({ duration_s: 8, video_duration_s: 0 }), 0);
assert.equal(context.videoDurationOf({}), 0);
assert.equal(context.videoDurationOf(undefined), 0);
// An event's active time: t_s, or t_video_s in old recordings.
assert.equal(context.eventActiveS({ t_s: 2.5, t_video_s: 4 }), 2.5);
assert.equal(context.eventActiveS({ t_s: 0, t_video_s: 4 }), 0);
assert.equal(context.eventActiveS({ t_video_s: 4 }), 4);
assert.equal(context.eventActiveS({}), 0);

// Trainee and exercise filters ("" = all, "free" = free practice).
const sessions = [
  { id: "a", trainee: "Dr Lin", exercise: { id: "x" } },
  { id: "b", trainee: "Dr Lin", exercise: null },
  { id: "c", trainee: "Dr Wu", exercise: { id: "x" } },
];
const ids = list => plain(list.map(s => s.id));
assert.deepEqual(ids(context.filterSessions(sessions, "", "")), ["a", "b", "c"]);
assert.deepEqual(ids(context.filterSessions(sessions, "Dr Lin", "")), ["a", "b"]);
assert.deepEqual(ids(context.filterSessions(sessions, "", "free")), ["b"]);
assert.deepEqual(ids(context.filterSessions(sessions, "", "x")), ["a", "c"]);
assert.deepEqual(ids(context.filterSessions(sessions, "Dr Wu", "free")), []);

// Rubric rows and the review total: auto points plus ticked instructor items.
const rubric = [
  { id: "no_stops", text: "No stops", points: 30, auto: { no_safety_stops: true } },
  { id: "path", text: "Path", points: 20, auto: { max_path_deviation_m: 0.03 } },
  { id: "smooth", text: "Smooth", points: 30, auto: null },
  { id: "tidy", text: "Tidy", points: 20, auto: null },
];
const report = {
  auto_points: 30, auto_max: 50, instructor_max: 50,
  items: [{ id: "no_stops", passed: true, value: 0, points: 30 }, { id: "path", passed: false, value: 0.05, points: 0, gated: true },
    { id: "smooth", passed: null, value: null, points: 0 }, { id: "tidy", passed: null, value: null, points: 0 }],
};
const rows = context.rubricRows(rubric, report, { checked: ["tidy"] });
assert.deepEqual(plain(rows.map(r => [r.id, r.auto, r.passed, r.value, r.gated, r.checked])), [
  ["no_stops", "no_safety_stops", true, 0, false, false],
  ["path", "max_path_deviation_m", false, 0.05, true, false],
  ["smooth", null, null, null, false, false],
  ["tidy", null, null, null, false, true],
]);
assert.equal(context.reviewTotal(rows, report, ["tidy"]), 50);
assert.equal(context.reviewTotal(rows, report, ["smooth", "tidy", "no_stops"]), 80);   // auto ids never add
assert.equal(context.reviewTotal(rows, null, []), 0);
assert.deepEqual(plain(context.rubricRows(rubric, null, null).map(r => r.passed)), [null, null, null, null]);
console.log("dashboard ok");
