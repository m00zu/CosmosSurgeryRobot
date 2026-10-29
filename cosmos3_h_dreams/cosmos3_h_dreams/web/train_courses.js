// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Training courses on /train: the course menu, guided exercises, demo playback, the path guide,
// the safety meter and the report card. Pure helpers live in train_core.js; this file is DOM wiring.
// It is loaded before train.js and uses train.js globals (state, send, resetSimulation, setAttempt,
// notify, show, legend, startButton, stopButton, traineeInput) only inside functions that run after
// train.js has loaded. Server text goes into the page with textContent only.

const courseEl = id => document.getElementById(id);
const GUIDE_COLORS = { left: "#3987e5", right: "#199e70" };
const KEY_HINTS = {
  reach: "Mouse moves the instrument; scroll down to lower it, up to raise it",
  active: "Tab switches the instrument",
  jaws_closed: "Hold the left mouse button",
  jaws_open: "Click (or let go of) the left mouse button",
  energy: "Hold F",
  home: "R returns the active instrument home",
  moved: "Move with the mouse or the wheel",
};
const course = {
  groups: [],               // /api/courses "courses": [{name, exercises}]
  exercises: [],            // exercise JSON from /api/courses, in menu order
  demos: [],                // recorded demos ("Your demos")
  sessions: [],             // /api/sessions, for best scores in the menu
  selected: null,           // {kind: "exercise", exercise} | {kind: "demo", demo}
  run: null,                // {exercise, steps: newStepRun(...), clock: newActiveClock(...), recordingId, ending}
  playback: null,           // newPlayback(...) while a demo plays
  playing: null,            // {title, exercise (null for a demo-only or recorded demo), demo}
  guideDemo: null,          // demo whose path the guide draws
  ratios: null,             // the latest safety ratios from the server
  recording: null,          // the latest /healthz.recording, from onRecordingHealth
  trails: { left: [], right: [] },
  panelShown: false,        // an exercise or demo owns the right column
  frame: 0,
  enabled: false,           // /api/courses answered with exercises; otherwise /train is as it was before courses
  scene: null,              // /api/courses "scene": exercises and course demos run on it
};

function clear(id) { courseEl(id).replaceChildren(); }
function add(parent, tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  parent.append(node);
  return node;
}
function attemptRunning() { return Boolean(course.run) || !stopButton.disabled; }

// A free-practice attempt is running when Stop is enabled or /healthz reports a recording, and no exercise run owns it.
function freePracticeRunning() { return !course.run && (!stopButton.disabled || Boolean(course.recording)); }

// Demos are never recorded, and the demo's reset would start a new free-practice attempt: end the attempt first,
// the way the Stop button does. Returns true when an attempt was stopped.
function stopFreePractice() {
  if (!freePracticeRunning()) return false;
  send({ type: "stop" });
  setAttempt(false);
  course.recording = null;
  return true;
}

function loadDemo(item) {
  const url = item.kind === "exercise"
    ? `/api/courses/${encodeURIComponent(item.exercise.id)}/demo`
    : `/api/demos/${encodeURIComponent(item.demo.slug)}`;
  return fetch(url, { cache: "no-store" }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  });
}

async function fetchCourses() {
  const response = await fetch("/api/courses", { cache: "no-store" });
  if (!response.ok) return false;   // 404: courses are off on this server
  const body = await response.json();
  course.exercises = body.courses.flatMap(group => group.exercises);
  course.groups = body.courses;
  course.demos = body.demos;
  course.scene = body.scene ?? null;
  return true;
}

// After a reset the server recalibrates and generates its first chunk before new video shows. Reset, hold the
// instruments still and show a progress bar until then, so a demo's or exercise's clock starts with the video.
// Resolves false when it gives up waiting (30 s); the caller starts anyway.
async function prepareSimulation(title) {
  const read = async () => (await (await fetch("/healthz", { cache: "no-store" })).json()).simulation;
  let before = null;
  try { before = await read(); } catch (_) { /* no status: start at once */ }
  resetSimulation();
  if (!before) return true;
  courseEl("prep-title").textContent = title;
  courseEl("prep").hidden = false;
  state.held = true;   // still instruments let the calibration finish
  try {
    const deadline = performance.now() + 30000;
    let shown = 0;   // the bar only moves forward, though a reset can restart the calibration
    while (performance.now() < deadline) {
      let sim = null;
      try { sim = await read(); } catch (_) { /* retry */ }
      const view = prepView(sim, before.epoch);
      shown = Math.max(shown, view.percent);
      courseEl("prep-text").textContent = view.text;
      courseEl("prep-bar").style.width = `${shown}%`;
      if (view.done) return true;
      await new Promise(resolve => setTimeout(resolve, 150));
    }
    notify("The simulation is slow to restart; starting anyway.");
    return false;
  } finally {
    courseEl("prep").hidden = true;
    state.held = false;
  }
}

// An exercise and its demo run on the exercise's own scene, or else the course scene (/api/courses "scene");
// switch to it first. False if the switch failed.
async function useCourseScene(exercise) {
  const scene = exercise?.scene ?? course.scene;
  if (!scene || sceneHealth.active_scene === scene) return true;
  show("Switching to the exercise scene…");
  if (await selectScene(scene)) return true;
  notify("Could not switch to the exercise scene; try again in a moment.");
  return false;
}

// ---- Course menu ------------------------------------------------------------------------
async function openCourseMenu(selectId) {
  try {
    await fetchCourses();
    const response = await fetch("/api/sessions", { cache: "no-store" });
    course.sessions = response.ok ? await response.json() : [];
  } catch (_) {
    course.sessions = [];
  }
  const exercise = course.exercises.find(e => e.id === selectId) || course.selected?.exercise || course.exercises[0];
  course.selected = exercise ? { kind: "exercise", exercise } : null;
  renderCourseList();
  renderDetail();
  courseEl("course-menu").showModal();
}

function renderCourseList() {
  const list = courseEl("course-list");
  list.replaceChildren();
  const trainee = traineeKey(traineeInput.value) || "anonymous";
  const item = (label, meta, selected, onClick, best) => {
    const button = add(list, "button", undefined);
    button.type = "button";
    add(button, "span", label);
    add(button, "span", meta, "meta");
    if (best) add(button, "span", best, "best");
    button.setAttribute("aria-pressed", String(selected));
    button.addEventListener("click", onClick);
  };
  for (const group of course.groups) {
    add(list, "h3", group.name);
    for (const exercise of group.exercises) {
      const best = bestScore(course.sessions, trainee, exercise.id);
      const meta = exercise.demo_only
        ? ["Demo only", roughLength(exercise)]
        : [`${exercise.step_count} steps`, roughLength(exercise)];
      item(`${exercise.order} · ${exercise.title}`, meta.filter(Boolean).join(" · "),
        course.selected?.exercise === exercise, () => select({ kind: "exercise", exercise }),
        best === null ? "" : `best ${best}%`);
    }
  }
  add(list, "h3", "Your demos");
  if (!course.demos.length) add(list, "p", "None yet: save an attempt as a demo on the dashboard.", "meta");
  for (const demo of course.demos) {
    item(demo.name, `recorded · ${elapsedText(demo.duration_s)}`, course.selected?.demo?.slug === demo.slug,
      () => select({ kind: "demo", demo }));
  }
  add(list, "h3", "Free practice");
  item("Free practice", "No steps or score; Start attempt records it", false, () => courseEl("course-menu").close());
}

function select(item) {
  course.selected = item;
  renderCourseList();
  renderDetail();
}

function renderDetail() {
  const item = course.selected;
  for (const id of ["detail-steps", "detail-rubric"]) clear(id);
  // Watching a demo ends a free-practice attempt (see stopFreePractice); an exercise attempt must be ended first.
  const exerciseBusy = Boolean(course.run);
  const busy = attemptRunning();
  courseEl("detail-note").textContent = exerciseBusy ? "Stop the current attempt first."
    : busy ? "Stop the current attempt to start an exercise. Watching a demo stops it." : "";
  courseEl("watch-demo").disabled = exerciseBusy;
  courseEl("start-exercise").disabled = busy;
  if (!item) {
    courseEl("detail-title").textContent = "No exercises";
    courseEl("watch-demo").hidden = courseEl("start-exercise").hidden = true;
    return;
  }
  renderPreview(item);
  if (item.kind === "demo") {
    courseEl("detail-title").textContent = item.demo.name;
    courseEl("detail-brief").textContent = `Recorded from attempt ${item.demo.source} on ${item.demo.created}.`;
    courseEl("detail-demo").textContent = `Demo: recorded, ${elapsedText(item.demo.duration_s)}`;
    courseEl("detail-steps-title").hidden = courseEl("detail-rubric-title").hidden = true;
    courseEl("watch-demo").hidden = false;
    courseEl("start-exercise").hidden = true;
    return;
  }
  const exercise = item.exercise;
  courseEl("detail-title").textContent = exercise.title;
  courseEl("detail-brief").textContent = exercise.brief;
  courseEl("detail-steps-title").hidden = courseEl("detail-rubric-title").hidden = exercise.demo_only;
  for (const step of exercise.steps) add(courseEl("detail-steps"), "li", step.text);
  for (const rubric of exercise.rubric) {
    const li = add(courseEl("detail-rubric"), "li");
    add(li, "span", `${rubric.points} pts`, "pts");
    add(li, "span", rubric.text);
    add(li, "span", rubric.auto ? "auto" : "instructor", rubric.auto ? "kind" : "kind instructor");
  }
  courseEl("detail-demo").textContent = !exercise.has_demo
    ? "No demo"
    : exercise.demo_source === "recorded"
      ? `Demo: recorded "${exercise.demo_name}", ${elapsedText(exercise.demo_duration_s)}`
      : `Demo: scripted, ${elapsedText(exercise.demo_duration_s)}`;
  courseEl("watch-demo").hidden = !exercise.has_demo;
  courseEl("start-exercise").hidden = exercise.demo_only;
}

// The menu's preview: the exercise's starting frame, with the demo's top-view path once it loads.
function renderPreview(item) {
  const scene = item.kind === "exercise" ? item.exercise.scene ?? course.scene : course.scene;
  const frame = courseEl("detail-frame");
  courseEl("detail-preview").hidden = !scene;
  frame.hidden = false;
  frame.onerror = () => { frame.hidden = true; };   // no frame: the dark box still carries the demo path
  if (scene) frame.src = `/scene/frame?name=${encodeURIComponent(scene)}`;
  courseEl("detail-path-box").hidden = true;
  const hasDemo = item.kind === "demo" || item.exercise.has_demo;
  if (!hasDemo) return;
  loadDemo(item).then(demo => {
    if (course.selected !== item) return;   // the trainee picked another item meanwhile
    drawPreviewPath(demo, item.kind === "exercise" ? item.exercise : null);
  }).catch(() => {});
}

function drawPreviewPath(demo, exercise) {
  const svg = courseEl("detail-path");
  const [width, height, pad] = [220, 120, 14];
  const reaches = (exercise ? exercise.steps : []).map(step => step.check.reach).filter(Boolean);
  // Zoomed to the demo's motion and targets (plus 2 cm), top view: x right, forward (-z) up.
  const points = SIDES.flatMap(side => demo.frames.map(f => f[side].position)).concat(reaches.map(r => r.center));
  const xs = points.map(p => p[0]);
  const zs = points.map(p => p[2]);
  const [x0, x1, z0, z1] = [Math.min(...xs) - 0.02, Math.max(...xs) + 0.02, Math.min(...zs) - 0.02, Math.max(...zs) + 0.02];
  const scale = Math.min((width - 2 * pad) / (x1 - x0), (height - 2 * pad) / (z1 - z0));
  const [ox, oy] = [(width - scale * (x1 - x0)) / 2, (height - scale * (z1 - z0)) / 2];
  const project = p => [ox + (p[0] - x0) * scale, oy + (z1 - p[2]) * scale];
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  for (const reach of reaches) {
    const [cx, cy] = project(reach.center);
    svgEl(svg, "circle", { cx, cy, r: Math.max(4, reach.radius_m * scale), fill: "none", stroke: "#f5b83d", "stroke-width": 1.5 });
  }
  for (const side of SIDES) {
    const path = demo.frames.map(f => f[side].position);
    svgEl(svg, "path", { d: guidePath(path, project), fill: "none", stroke: GUIDE_COLORS[side], "stroke-width": 2, "stroke-dasharray": "5 4" });
    const [x, y] = project(path[0]);
    svgEl(svg, "circle", { cx: x, cy: y, r: 4, fill: GUIDE_COLORS[side] });
  }
  courseEl("detail-path-box").hidden = false;
}

// ---- Course panel (right column) and chips -------------------------------------------------
function showCoursePanel(title) {
  course.panelShown = true;
  courseEl("course-panel-title").textContent = title;
  courseEl("course-panel").hidden = false;
  legend.hidden = true;
  startButton.hidden = stopButton.hidden = true;   // Stop exercise / Stop demo replace them
  courseEl("stop-exercise").hidden = false;
  courseEl("stop-exercise").textContent = course.playback ? "Stop demo" : "Stop exercise";
  courseEl("courses-button").disabled = true;
  courseEl("caption").hidden = false;
  renderMeter();
}

function hideCoursePanel() {
  if (course.run || course.playback) return;
  course.panelShown = false;
  courseEl("course-panel").hidden = true;
  legend.hidden = false;
  startButton.hidden = stopButton.hidden = false;
  courseEl("stop-exercise").hidden = true;
  courseEl("exercise-chip").hidden = true;
  courseEl("courses-button").disabled = false;
  courseEl("caption").hidden = true;
}

// H: with an exercise or demo running, swap between the course panel and the key card.
function toggleLegend() {
  if (!course.panelShown) {
    legend.hidden = !legend.hidden;
    return;
  }
  const showKeys = !courseEl("course-panel").hidden;
  courseEl("course-panel").hidden = showKeys;
  legend.hidden = !showKeys;
}

function renderSteps() {
  const list = courseEl("step-list");
  list.replaceChildren();
  courseEl("step-list-title").textContent = course.playback ? "Demo steps" : "Steps";
  if (course.playback) {
    const current = markerIndex(course.playback.demo.markers, course.playback.positionS);
    course.playback.demo.markers.forEach((marker, i) => {
      const li = add(list, "li", `${marker.caption} · ${elapsedText(marker.t_s)}`, i < current ? "done" : i === current ? "current" : "");
      if (i === current) li.setAttribute("aria-current", "step");
    });
    return;
  }
  const run = course.run;
  if (!run) return;
  run.exercise.steps.forEach((step, i) => {
    const done = run.steps.done.find(d => d.index === i);
    const status = done ? "done" : i === run.steps.index ? "current" : "pending";
    const li = add(list, "li", undefined, status);
    if (done) add(li, "span", "Done: ", "sr-only");
    add(li, "span", step.text);
    if (done) add(li, "span", elapsedText(done.t_s), "time");
    if (status === "current") {
      li.setAttribute("aria-current", "step");
      add(li, "span", KEY_HINTS[Object.keys(step.check)[0]], "hint");
    }
  });
}

function renderMeter() {
  const meter = courseEl("safety-meter");
  meter.replaceChildren();
  for (const row of meterRows(course.ratios)) {
    const line = add(meter, "div", undefined, "meter-row");
    line.dataset.level = row.level;
    add(line, "span", row.label);
    const bar = add(line, "div", undefined, "meter-bar");
    add(bar, "div", undefined, "meter-fill").style.width = `${Math.min(100, row.percent)}%`;
    add(line, "span", row.text, "meter-text");
  }
  courseEl("safety-hint").textContent = meterHint(course.ratios);
}

function onSafety(ratios) {
  course.ratios = ratios;
  if (course.panelShown) renderMeter();
}

// ---- Guided exercise --------------------------------------------------------------------
async function startExercise(exercise) {
  if (attemptRunning()) {
    notify("Stop the current attempt first.");
    return;
  }
  courseEl("course-menu").close();
  courseEl("report").close();
  if (!(await useCourseScene(exercise))) return;
  if (attemptRunning()) return;   // started elsewhere while the scene switched
  endDemo();
  await prepareSimulation(`Starting ${exercise.title}`);   // a fresh scene with both instruments at home
  if (attemptRunning()) return;
  send({ type: "start", trainee: traineeInput.value.trim() || "anonymous", exercise: exercise.id });
  setAttempt(true);
  course.run = { exercise, steps: newStepRun(exercise.steps), clock: newActiveClock(performance.now()), recordingId: null, ending: false };
  course.trails = { left: [], right: [] };
  course.guideDemo = null;
  if (exercise.has_demo) {
    loadDemo({ kind: "exercise", exercise })
      .then(demo => { if (course.run?.exercise === exercise) course.guideDemo = demo; })
      .catch(() => {});   // the guide just has no demo path
  }
  courseEl("exercise-chip").hidden = false;
  showCoursePanel(exercise.title);
  renderSteps();
}

function finishExercise() {
  if (!course.run || course.run.ending) return;
  course.run.ending = true;   // the report card opens once /healthz shows the recording ended
  send({ type: "stop" });
  if (document.pointerLockElement) document.exitPointerLock();   // free the cursor for the report card
  setAttempt(false);
}

function stopExercise() {
  if (course.playback) endDemo();
  else finishExercise();
}

// The line under the current step's caption: hold progress, else live guidance (`vr`: worded for the controllers).
function exerciseGuidance(run, tS, vr = false) {
  const step = run.exercise.steps[run.steps.index];
  const held = holdSeconds(run.steps, tS);
  const target = step && (step.check.jaws_closed || step.check.energy);
  return held !== null && target ? `${held.toFixed(1)} / ${target.hold_s} s`
    : step ? stepGuidance(step.check, viewOf(state), run.steps, vr) : "";
}

// Called every tick by train.js after the frame is sent.
function exerciseTick(nowMs) {
  if (!course.enabled) return;
  for (const side of SIDES) pushTrail(course.trails[side], state.instruments[side].pose.position, GUIDE_TRAIL);
  const run = course.run;
  if (run && !run.ending) {
    const tS = activeSeconds(run.clock, nowMs, state.held);   // without pauses and safety stops, as the report
    const completed = state.held ? [] : updateSteps(run.steps, viewOf(state), tS);
    for (const index of completed) send({ type: "step", index, t_client: nowMs });
    if (completed.length) renderSteps();
    const n = run.exercise.steps.length;
    courseEl("exercise-chip").textContent =
      `${run.exercise.title} · step ${Math.min(run.steps.index + 1, n)} of ${n} · ${elapsedText(tS)}`;
    const step = run.exercise.steps[run.steps.index];
    courseEl("caption-text").textContent = step ? step.text : "Done";
    courseEl("caption-hold").textContent = exerciseGuidance(run, tS);
    if (run.steps.index >= n) finishExercise();
  }
  course.frame += 1;
  if (course.frame % 3 === 0) renderGuide();   // 10 Hz is plenty for the guide
}

// train.js passes /healthz.recording each second; an exercise attempt that ended opens its report.
function onRecordingHealth(recording) {
  course.recording = recording;
  if (course.playback && recording) send({ type: "stop" });   // an answer that predates our stop: demos are never recorded
  const run = course.run;
  if (!run) return;
  if (recording && recording.exercise === run.exercise.id) {
    run.recordingId = recording.id;
    return;
  }
  if (!recording && (run.recordingId || run.ending)) {   // finished, stopped, reset, switched or disconnected
    course.run = null;
    courseEl("caption-hold").textContent = "";
    hideCoursePanel();
    if (run.recordingId) void showReport(run.recordingId, run.exercise);
    else notify("No report: recording is off on this server.");
  }
}

// ---- Demo playback ------------------------------------------------------------------------
async function watchDemo(item) {
  if (course.run) {   // an exercise attempt ends with its own Stop exercise, which opens the report
    notify("Stop the current attempt first.");
    return;
  }
  let demo;
  try {
    demo = await loadDemo(item);
  } catch (error) {
    notify(`Could not load the demo: ${error.message}`);
    return;
  }
  if (course.run) {   // an exercise started while the demo loaded
    notify("Stop the current attempt first.");
    return;
  }
  courseEl("course-menu").close();
  courseEl("report").close();
  if (!(await useCourseScene(item.kind === "exercise" ? item.exercise : null)) || course.run) return;
  if (stopFreePractice()) show("Stopped the current attempt: demos are never recorded.");
  const exercise = item.kind === "exercise" ? item.exercise : null;
  course.playing = {
    title: exercise ? exercise.title : item.demo.name,
    exercise: exercise && !exercise.demo_only ? exercise : null,
    demo,
  };
  // Held at its first pose (and grip) while the simulation restarts; the clock starts with the new video.
  course.playback = newPlayback(demo, performance.now());
  setPlaying(course.playback, false, performance.now());
  course.guideDemo = demo;
  course.trails = { left: [], right: [] };
  courseEl("stage").classList.add("demo");
  courseEl("demo-chip").hidden = false;
  courseEl("playback").hidden = false;
  courseEl("try-it").hidden = !course.playing.exercise;
  setDemoSpeed(1);
  renderTimeline(demo);
  showCoursePanel(`Demo: ${course.playing.title}`);
  renderSteps();
  const playback = course.playback;
  await prepareSimulation(`Preparing the demo: ${course.playing.title}`);
  if (course.playback === playback) setPlaying(playback, true, performance.now());
}

// Called every tick by train.js; true while a demo drives the instruments.
function demoTick(nowMs) {
  const pb = course.playback;
  if (!pb) return false;
  const before = markerIndex(pb.demo.markers, pb.positionS);
  driveDemo(state, pb, nowMs);
  const current = markerIndex(pb.demo.markers, pb.positionS);
  if (current !== before) renderSteps();
  courseEl("caption-text").textContent = current >= 0 ? pb.demo.markers[current].caption : "";
  courseEl("demo-time").textContent = `${elapsedText(pb.positionS)} / ${elapsedText(pb.demo.duration_s)}`;
  courseEl("demo-cursor").style.left = `${pb.demo.duration_s ? (100 * pb.positionS) / pb.demo.duration_s : 0}%`;
  courseEl("play-pause").textContent = pb.playing ? "Pause" : pb.done ? "Finished" : "Play";
  courseEl("play-pause").disabled = pb.done;
  return true;
}

function renderTimeline(demo) {
  const timeline = courseEl("demo-timeline");
  timeline.replaceChildren(courseEl("demo-cursor"));
  const width = timeline.clientWidth;
  demo.markers.forEach((marker, i) => {
    const tick = add(timeline, "div", undefined, "demo-marker");
    tick.style.left = `${demo.duration_s ? (100 * marker.t_s) / demo.duration_s : 0}%`;
    tick.title = marker.caption;
    const label = add(tick, "span", marker.caption);
    // A label stops before the next marker so neighbours never overlap; the tooltip has the whole caption.
    const next = demo.markers[i + 1];
    const room = width && demo.duration_s ? (width * ((next ? next.t_s : demo.duration_s) - marker.t_s)) / demo.duration_s : 0;
    if (room) label.style.maxWidth = `${Math.max(0, room - 8)}px`;
  });
}

function toggleDemoPlay() {
  const pb = course.playback;
  if (pb) setPlaying(pb, !pb.playing, performance.now());
}

function setDemoSpeed(speed) {
  if (course.playback) setSpeed(course.playback, speed, performance.now());
  courseEl("speed-half").setAttribute("aria-pressed", String(speed === 0.5));
  courseEl("speed-one").setAttribute("aria-pressed", String(speed === 1));
}

function restartDemo() {
  stopFreePractice();   // nothing records during a demo, whatever slipped in
  resetSimulation();   // a jump back to the start would otherwise trip the speed limit
  restartPlayback(course.playback, performance.now());
  renderSteps();
}

// Space and Backspace control a playing demo; returns true when the key was used.
function demoKey(code) {
  if (!course.playback) return false;
  if (code === KEYS.pause) toggleDemoPlay();
  else if (code === KEYS.reset) restartDemo();
  else return false;
  return true;
}

function takeOverDemo() { endDemo(); }

// However a demo ends (Stop demo, the video click, an incident, Try it yourself) the mouse takes over from its pose:
// jaws open and energy off, so a demo that was mid-grasp or mid-energy cannot leave them on.
function endDemo() {
  if (!course.playback) return;
  takeOver(state);
  course.playback = null;
  course.playing = null;
  courseEl("stage").classList.remove("demo");
  courseEl("demo-chip").hidden = true;
  courseEl("playback").hidden = true;
  courseEl("caption-text").textContent = "";
  hideCoursePanel();
}

// ---- Path guide -------------------------------------------------------------------------
function guidePath(points, project) {
  return points.map((p, i) => `${i ? "L" : "M"}${project(p).map(v => v.toFixed(1)).join(",")}`).join("");
}

function renderGuide() {
  if (!course.enabled) return;
  const svg = courseEl("path-guide");
  const width = Math.max(200, Math.round(svg.clientWidth || 300));
  const height = 170;
  const topWidth = width - 44;   // the height bar takes the right 44 px
  const project = p => projectTop(p, topWidth, height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  svgEl(svg, "rect", { x: 0.5, y: 0.5, width: topWidth - 1, height: height - 1, fill: "none", stroke: GPU_INK.grid });
  haloText(svg, { x: 6, y: 14, "font-size": 10 }, "forward ↑");
  if (course.guideDemo) {
    for (const side of SIDES) {
      svgEl(svg, "path", {
        d: guidePath(course.guideDemo.frames.map(f => f[side].position), project),
        fill: "none", stroke: GUIDE_COLORS[side], "stroke-width": 1.5, "stroke-dasharray": "4 3", opacity: 0.8,
      });
    }
  }
  const step = course.run && course.run.exercise.steps[course.run.steps.index];
  const reach = step && step.check.reach;
  if (reach) {
    const [cx, cy] = project(reach.center);
    svgEl(svg, "circle", {
      cx, cy, r: Math.max(3, reach.radius_m * guideScale(topWidth, height)), fill: "none", stroke: "#f0b429", "stroke-width": 2,
    });
  }
  const barX = width - 22;
  svgEl(svg, "line", { x1: barX, x2: barX, y1: GUIDE_PAD, y2: height - GUIDE_PAD, stroke: GPU_INK.faint, "stroke-width": 2 });
  haloText(svg, { x: barX, y: height - 1, "text-anchor": "middle", "font-size": 10 }, "height");
  if (reach) {   // the target height as a band of ± its radius, like the circle in the top view
    const top = projectHeight(reach.center[1] + reach.radius_m, height);
    const bottom = projectHeight(reach.center[1] - reach.radius_m, height);
    svgEl(svg, "rect", {
      x: barX - 13, y: top, width: 26, height: Math.max(2, bottom - top), fill: "none", stroke: "#f0b429", "stroke-width": 2,
    });
  }
  for (const side of SIDES) {
    const trail = course.trails[side];
    if (trail.length > 1) {
      svgEl(svg, "path", { d: guidePath(trail, project), fill: "none", stroke: GUIDE_COLORS[side], "stroke-width": 2, opacity: 0.6 });
    }
    const position = state.instruments[side].pose.position;
    const [x, y] = project(position);
    svgEl(svg, "circle", { cx: x, cy: y, r: 5, fill: GUIDE_COLORS[side] });
    if (side === state.active) haloText(svg, { x: x + 8, y: y + 4, fill: GPU_INK.text }, `${side} (active)`);
    svgEl(svg, "circle", { cx: barX + (side === "left" ? -6 : 6), cy: projectHeight(position[1], height), r: 4, fill: GUIDE_COLORS[side] });
  }
}

// ---- Report card ------------------------------------------------------------------------
async function showReport(id, exercise) {
  let data = null;
  for (let i = 0; i < 30 && !(data && data.report); i += 1) {   // the recorder may still be finishing the video
    try {
      const response = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { cache: "no-store" });
      if (response.ok) data = await response.json();
    } catch (_) { /* retry */ }
    if (!(data && data.report)) await new Promise(resolve => setTimeout(resolve, 1000));
  }
  if (!data || !data.report) {
    notify("The report is not ready yet; open the attempt on the dashboard.");
    return;
  }
  const demo = exercise.has_demo ? await loadDemo({ kind: "exercise", exercise }).catch(() => null) : null;
  const sessions = await fetch("/api/sessions", { cache: "no-store" })
    .then(response => (response.ok ? response.json() : [])).catch(() => []);
  renderReport(data, exercise, demo, sessions);
  if (document.pointerLockElement) document.exitPointerLock();   // a click while the report loaded may have locked it again
  courseEl("report").showModal();
}

function renderReport(data, exercise, demo, sessions) {
  const { session, report } = data;
  const summary = session.summary || {};
  const sum = values => (values ? (values.left || 0) + (values.right || 0) : 0);
  const stats = demo ? demoStats(demo) : null;
  const stops = summary.incidents || 0;
  courseEl("report-title").textContent = exercise.title;
  courseEl("report-summary").textContent = [session.trainee, `${report.steps_done} of ${exercise.steps.length} steps`,
    `${stops} safety stop${stops === 1 ? "" : "s"}`, elapsedText(summary.duration_s)].join(" · ");
  courseEl("report-pending").textContent = report.instructor_max
    ? `+ ${report.instructor_max} instructor points pending review on the dashboard`
    : "";
  renderRing(report);
  const previous = previousAttempt(sessions, session.trainee, exercise.id, session.id);
  const change = previous ? report.auto_points - previous.report.auto_points : null;
  const delta = courseEl("report-delta");
  delta.textContent = change === null ? "First attempt" : `${change > 0 ? "+" : change < 0 ? "−" : "±"}${Math.abs(change)} since last run`;
  delta.className = change > 0 ? "up" : change < 0 ? "down" : "";
  const skills = skillScores(report, summary, exercise, stats);
  const before = previous ? skillScores(previous.report, previous.summary || {}, exercise, stats) : null;
  renderRadar(skills, skillDeltas(skills, before));

  clear("report-rubric");
  for (const item of exercise.rubric) {
    const scored = report.items.find(i => i.id === item.id) || {};
    const kind = item.auto && Object.keys(item.auto)[0];
    const status = !kind ? "pending" : scored.passed ? "pass" : "fail";
    const li = add(courseEl("report-rubric"), "li", undefined, status);
    add(li, "span", { pass: "✓", fail: "✕", pending: "…" }[status], "box").setAttribute("aria-hidden", "true");
    add(li, "span", `${{ pass: "Passed", fail: "Not passed", pending: "Pending" }[status]}: `, "sr-only");
    add(li, "span", item.text);
    add(li, "span", !kind ? "instructor item" : scored.gated ? "not scored: steps incomplete" : autoValueText(kind, scored.value), "value");
    add(li, "span", `${kind ? scored.points : "—"} / ${item.points}`, "item-pts");
  }

  const yours = stepDurations(reportStepTimes(report, exercise.steps.length));
  const theirs = demo ? stepDurations(stepTimes(exercise.steps, demo)) : exercise.steps.map(() => null);
  const longest = Math.max(1e-6, ...yours.filter(v => v !== null), ...theirs.filter(v => v !== null));
  clear("report-steps");
  exercise.steps.forEach((step, i) => {
    const row = add(courseEl("report-steps"), "div", undefined, "step-row");
    add(row, "span", `${i + 1}. ${step.text}`);
    const bars = add(row, "div", undefined, "step-bars");
    for (const [who, value] of [["you", yours[i]], ["demo", theirs[i]]]) {
      if (who === "demo" && !demo) continue;
      const line = add(bars, "div");
      add(line, "span", undefined, `bar ${who}`).style.width = `${(80 * (value || 0)) / longest}%`;
      add(line, "span", value === null ? (who === "you" ? "not done" : "—") : `${value.toFixed(1)} s`);
    }
  });

  const tiles = [
    ["Time", elapsedText(summary.duration_s), stats && elapsedText(stats.duration_s)],
    ["Safety stops", String(stops), stats && "0"],
    ["Near misses", String(report.near_misses), null],
    ["Path deviation", report.path_deviation_m === null ? "—" : cm(report.path_deviation_m), null],
    ["Corrections", report.corrections == null ? "—" : String(report.corrections),
      report.demo_corrections == null ? null : String(report.demo_corrections)],
    ["Path length", `${sum(summary.path_length_m).toFixed(2)} m`, stats && `${stats.path_m.toFixed(2)} m`],
    ["Energy", `${sum(summary.energy_s).toFixed(1)} s`, stats && `${stats.energy_s.toFixed(1)} s`],
  ];
  clear("report-tiles");
  for (const [label, value, demoValue] of tiles) {
    const tile = add(courseEl("report-tiles"), "div", undefined, "tile");
    add(tile, "span", label);
    add(tile, "strong", value);
    if (demoValue) add(tile, "span", `demo ${demoValue}`);
  }
  const next = course.exercises[course.exercises.indexOf(course.exercises.find(e => e.id === exercise.id)) + 1];
  courseEl("report-demo").hidden = !exercise.has_demo;
  courseEl("report-next").hidden = !next;
  courseEl("report-demo").onclick = () => watchDemo({ kind: "exercise", exercise });
  courseEl("report-retry").onclick = () => startExercise(exercise);
  courseEl("report-next").onclick = () => { courseEl("report").close(); void openCourseMenu(next.id); };
  courseEl("report-dashboard").href = `/dashboard#${encodeURIComponent(session.id)}`;
}

// The score ring: automatic points in teal and the pending instructor share in purple, out of the rubric total.
function renderRing(report) {
  const svg = courseEl("report-ring");
  const r = 50;
  const around = 2 * Math.PI * r;
  const max = Math.max(1, report.auto_max + report.instructor_max);
  svg.replaceChildren();
  svg.setAttribute("aria-label", `Automatic score ${report.auto_points} of ${report.auto_max}`
    + (report.instructor_max ? `, ${report.instructor_max} instructor points pending` : ""));
  const arc = (share, stroke, startTurn) => svgEl(svg, "circle", {
    cx: 60, cy: 60, r, fill: "none", stroke, "stroke-width": 10, "stroke-linecap": "butt",
    "stroke-dasharray": `${around * share} ${around}`, transform: `rotate(${-90 + 360 * startTurn} 60 60)`,
  });
  arc(1, "#18212c", 0);
  if (report.instructor_max) arc(report.instructor_max / max, "#3a3160", 1 - report.instructor_max / max);
  if (report.auto_points) arc(report.auto_points / max, "#3dd6c3", 0);
  svgEl(svg, "text", { x: 60, y: 62, "text-anchor": "middle", fill: "#e6edf3", "font-size": 32, "font-weight": 600, "font-family": "ui-monospace, monospace" },
    String(report.auto_points));
  svgEl(svg, "text", { x: 60, y: 80, "text-anchor": "middle", fill: "#93a1b3", "font-size": 11, "font-family": "ui-monospace, monospace" },
    `/ ${report.auto_max} auto`);
}

// The radar: your score per skill against the demo level (the dashed 100 ring), with the change since the last run.
function renderRadar(skills, deltas) {
  const svg = courseEl("report-radar");
  const [cx, cy, radius, n] = [300, 220, 125, skills.length];
  const at = (i, v) => radarPoint(cx, cy, radius, i, n, v);
  const ring = v => skills.map((_, i) => at(i, v).map(c => c.toFixed(1)).join(",")).join(" ");
  const mono = "ui-monospace, monospace";
  svg.replaceChildren();
  const enough = n >= 3;
  svg.hidden = !enough;
  courseEl("report-radar-empty").hidden = enough;
  if (enough) {
    for (const v of [100 / 3, 200 / 3]) svgEl(svg, "polygon", { points: ring(v), fill: "none", stroke: "#1c2733" });
    skills.forEach((_, i) => {
      const [x, y] = at(i, 100);
      svgEl(svg, "line", { x1: cx, y1: cy, x2: x, y2: y, stroke: "#1c2733" });
    });
    svgEl(svg, "polygon", { points: ring(100), fill: "none", stroke: "#8a6ae0", "stroke-width": 2, "stroke-dasharray": "6 4", "stroke-linejoin": "round" });
    svgEl(svg, "text", { x: cx + 8, y: cy - radius - 4, fill: "#8a6ae0", "font-size": 10, "font-family": mono }, "100 demo");
    svgEl(svg, "polygon", {
      points: skills.map((s, i) => at(i, s.score).map(c => c.toFixed(1)).join(",")).join(" "),
      fill: "#1f9e8f", "fill-opacity": 0.22, stroke: "#1f9e8f", "stroke-width": 2, "stroke-linejoin": "round",
    });
    skills.forEach((skill, i) => {
      const [px, py] = at(i, skill.score);
      const dot = svgEl(svg, "circle", { cx: px, cy: py, r: 6, fill: "#1f9e8f", stroke: "#0d131a", "stroke-width": 2 });
      svgEl(dot, "title", {}, `${skill.name}: ${skill.score} · ${skill.measured}`);
      // The score tag sits just outside its point, inside the axis label, so tags never meet in the middle.
      const [tx, ty] = at(i, Math.min(skill.score + 16, 112));
      const change = deltas[i];
      const changeText = change === null ? "" : ` ${change > 0 ? "+" : change < 0 ? "−" : "±"}${Math.abs(change)}`;
      const width = 14 + 7.5 * (String(skill.score).length + changeText.length);
      svgEl(svg, "rect", { x: tx - width / 2, y: ty - 10, width, height: 20, rx: 10, fill: "#0d131a", stroke: "#1f9e8f" });
      const tag = svgEl(svg, "text", { x: tx, y: ty + 4, "text-anchor": "middle", fill: "#e6edf3", "font-size": 12, "font-weight": 600, "font-family": mono }, String(skill.score));
      if (changeText) svgEl(tag, "tspan", { fill: change > 0 ? "#3dd6c3" : change < 0 ? "#ff8a8a" : "#93a1b3", "font-weight": 500 }, changeText);
      const [lx, ly] = at(i, 136);
      const anchor = Math.abs(lx - cx) < 8 ? "middle" : lx > cx ? "start" : "end";
      const top = ly < cy - 20;
      svgEl(svg, "text", { x: lx, y: top ? ly - 14 : ly + 2, "text-anchor": anchor, fill: "#e6edf3", "font-size": 13, "font-weight": 600 }, skill.name);
      svgEl(svg, "text", { x: lx, y: top ? ly + 2 : ly + 18, "text-anchor": anchor, fill: "#93a1b3", "font-size": 12, "font-family": mono }, skill.short);
    });
  }
  clear("report-radar-table");
  for (const skill of skills) {
    const tr = add(courseEl("report-radar-table"), "tr");
    add(tr, "td", skill.name);
    add(tr, "td", skill.measured);
    add(tr, "td", String(skill.score));
  }
}

// ---- Start-up -----------------------------------------------------------------------------
async function initCourses() {
  courseEl("courses-button").addEventListener("click", () => openCourseMenu());
  courseEl("watch-demo").addEventListener("click", () => { if (course.selected) watchDemo(course.selected); });
  courseEl("start-exercise").addEventListener("click", () => {
    if (course.selected?.exercise) startExercise(course.selected.exercise);
  });
  courseEl("stop-exercise").addEventListener("click", stopExercise);
  courseEl("show-keys").addEventListener("click", toggleLegend);
  courseEl("play-pause").addEventListener("click", toggleDemoPlay);
  courseEl("speed-half").addEventListener("click", () => setDemoSpeed(0.5));
  courseEl("speed-one").addEventListener("click", () => setDemoSpeed(1));
  courseEl("try-it").addEventListener("click", () => {
    const exercise = course.playing?.exercise;
    endDemo();
    if (exercise) startExercise(exercise);
  });
  window.addEventListener("resize", () => {
    renderGuide();
    if (course.playback) renderTimeline(course.playback.demo);
  });
  // Without courses (404, or nothing to show) the Courses button and the path guide stay hidden.
  try {
    course.enabled = (await fetchCourses()) && course.exercises.length > 0;
  } catch (_) { /* courses unavailable */ }
  courseEl("courses-button").hidden = !course.enabled;
  courseEl("path-guide-panel").hidden = !course.enabled;
  renderGuide();
}
