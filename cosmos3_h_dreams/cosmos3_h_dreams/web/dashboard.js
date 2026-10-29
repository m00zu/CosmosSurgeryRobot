// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Dashboard for recorded training attempts: a sortable session list and a detail view with video and event timeline.
// videoDurationOf, eventActiveS, timelineMarkers, filterSessions, rubricRows and reviewTotal are top-level functions so
// node tests can call them; the DOM code below only runs on dashboard.html, which also loads gpu_chart.js and
// train_core.js (score helpers).
// Two clocks: summary.duration_s and an event's t_s are active seconds (real time without pauses and safety stops);
// summary.video_duration_s and t_video_s are positions in the video. Times shown use the first; markers sit on the
// video's timeline, so they and their seeks use the second. Recordings made before active time was measured have only
// video times, in duration_s and t_video_s, and show those.
// Trainee names are user input, so all data goes into the page with textContent.

const SEEK_LEAD_S = 3;                       // clicking a marker starts the video this long before the event
const TICK_KINDS = ["reset", "pause", "resume"];

// Length of the video in seconds: video_duration_s, or duration_s in a recording that predates active time.
function videoDurationOf(summary) {
  const s = summary || {};
  return Number(s.video_duration_s ?? s.duration_s) || 0;
}

// An event's active seconds: t_s, or t_video_s in a recording that predates active time.
function eventActiveS(event) {
  return Number(event.t_s ?? event.t_video_s) || 0;
}

// Timeline markers as fractions in [0, 1] of the video length, ordered by start (ties keep event order).
// Each marker also carries t_s, its event's active seconds, for the time it is listed with.
function timelineMarkers(session, events) {
  const duration = videoDurationOf(session.summary);
  const at = t => (duration > 0 ? Math.min(1, Math.max(0, t / duration)) : 0);
  const seek = t => Math.max(0, t - SEEK_LEAD_S);
  const finish = (running, t_end) => {
    running.marker.end = at(t_end);
    running.marker.label = `${running.side} energy ${Math.max(0, t_end - running.t).toFixed(1)}s`;
  };
  const markers = [];
  const running = new Map();                 // side -> energy span still open
  for (const event of events) {
    const t = Number(event.t_video_s) || 0;
    const t_s = eventActiveS(event);
    if (event.kind === "incident") {
      const label = (event.violations || []).map(v => v.message).join("; ");
      markers.push({ kind: "incident", start: at(t), end: at(t), label, seek_s: seek(t), t_s });
    } else if (event.kind === "energy_on" && !running.has(event.side)) {
      const marker = { kind: "energy", start: at(t), end: at(duration), label: "", seek_s: seek(t), t_s };
      markers.push(marker);
      running.set(event.side, { marker, side: event.side, t });
    } else if (event.kind === "energy_off" && running.has(event.side)) {
      finish(running.get(event.side), t);
      running.delete(event.side);
    } else if (TICK_KINDS.includes(event.kind)) {
      markers.push({ kind: "tick", start: at(t), end: at(t), label: event.kind, seek_s: seek(t), t_s });
    } else if (event.kind === "step") {
      markers.push({ kind: "step", start: at(t), end: at(t), label: `step ${Number(event.index) + 1} done`, seek_s: seek(t), t_s });
    }
  }
  for (const open of running.values()) finish(open, duration);
  return markers.sort((a, b) => a.start - b.start);
}

// Sessions matching the trainee ("" = all) and exercise ("" = all, "free" = free practice) filters.
function filterSessions(sessions, trainee, exercise) {
  return sessions.filter(s => (!trainee || s.trainee === trainee)
    && (!exercise || (exercise === "free" ? !s.exercise : Boolean(s.exercise) && s.exercise.id === exercise)));
}

// Detail rubric rows: automatic items with the report's result, instructor items with the review's ticks.
function rubricRows(rubric, report, review) {
  const scored = new Map(((report && report.items) || []).map(item => [item.id, item]));
  const checked = new Set((review && review.checked) || []);
  return rubric.map(item => {
    const auto = item.auto ? Object.keys(item.auto)[0] : null;
    const result = scored.get(item.id) || {};
    return {
      id: item.id, text: item.text, points: item.points, auto,
      passed: auto ? result.passed ?? null : null,
      value: auto ? result.value ?? null : null,
      gated: Boolean(auto && result.gated),   // steps incomplete: not scored
      checked: !auto && checked.has(item.id),
    };
  });
}

// auto_points plus the points of the ticked instructor items.
function reviewTotal(rows, report, checkedIds) {
  const checked = new Set(checkedIds);
  const instructor = rows.filter(r => !r.auto && checked.has(r.id)).reduce((sum, r) => sum + r.points, 0);
  return (report ? report.auto_points : 0) + instructor;
}

if (document.getElementById("sessions")) {
  const $ = id => document.getElementById(id);
  const MODE_LABELS = { standard: "Standard", balanced: "Balanced (FP8 MLP)", low_memory: "Low memory (FP8)" };
  const player = $("player");
  const summaryOf = s => s.summary || {};
  const sideTotal = values => (values && values.left || 0) + (values && values.right || 0);
  const modeLabel = mode => MODE_LABELS[mode] || String(mode);
  const clock = seconds => {
    const t = Math.round(Number(seconds) || 0);
    return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`;
  };
  const vramOf = s => summaryOf(s).vram || {};
  const gib = mib => (mib === null || mib === undefined ? "—" : `${(Number(mib) / 1024).toFixed(1)} GiB`);
  const dateOf = id => id.replace(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2}).*$/, "$1-$2-$3 $4:$5");

  // Sortable columns: `value` is the sort key, `text` what the cell shows.
  const exerciseTitle = s => (s.exercise ? s.exercise.title : "Free practice");
  const scoreOf = s => attemptScore(s);
  const COLUMNS = [
    { title: "Date", value: s => s.id.slice(0, 15), text: s => dateOf(s.id) },
    { title: "Trainee", value: s => (s.trainee || "").toLowerCase(), text: s => s.trainee || "" },
    { title: "Exercise", value: s => exerciseTitle(s), text: s => exerciseTitle(s) },
    { title: "Scene", value: s => String(s.scene), text: s => String(s.scene) },
    { title: "Mode", value: s => modeLabel(s.mode), text: s => modeLabel(s.mode) },
    { title: "Duration", value: s => summaryOf(s).duration_s || 0, text: s => clock(summaryOf(s).duration_s) },
    { title: "Safety stops", value: s => summaryOf(s).incidents || 0, text: s => String(summaryOf(s).incidents || 0) },
    { title: "Score", value: s => (scoreOf(s) ? scoreOf(s).percent ?? -1 : -1), text: s => (scoreOf(s) ? scoreOf(s).text : "—") },
    { title: "Energy", value: s => sideTotal(summaryOf(s).energy_s), text: s => `${sideTotal(summaryOf(s).energy_s).toFixed(1)} s` },
    { title: "Peak VRAM", value: s => vramOf(s).peak_used_mib ?? -1, text: s => gib(vramOf(s).peak_used_mib) },
    { title: "Status", value: s => (s.complete ? s.end_reason || "" : "incomplete"), text: s => (s.complete ? s.end_reason || "" : "incomplete") },
  ];
  const compare = (a, b) => (typeof a === "number" ? a - b : a.localeCompare(b));
  let sessions = [];
  let exercises = null;                      // exercise JSON by id from /api/courses; null when courses are off
  let sortColumn = 0;
  let sortDirection = -1;                    // newest first, as the server returns them

  const headers = COLUMNS.map((column, index) => {
    const th = document.createElement("th");
    th.textContent = column.title;
    th.addEventListener("click", () => {
      sortDirection = index === sortColumn ? -sortDirection : 1;
      sortColumn = index;
      renderList();
    });
    return th;
  });
  $("columns").replaceChildren(...headers);

  function renderRow(session) {
    const tr = document.createElement("tr");
    tr.addEventListener("click", () => { location.hash = session.id; });
    COLUMNS.forEach((column, index) => {
      const td = document.createElement("td");
      if (index === 0) {                     // a real link, so rows can be opened from the keyboard
        const link = document.createElement("a");
        link.href = `#${session.id}`;
        link.textContent = column.text(session);
        td.append(link);
      } else {
        td.textContent = column.text(session);
      }
      if (index === COLUMNS.length - 1 && !session.complete) {
        td.className = "incomplete";
        td.title = [session.write_error, session.video_error].filter(Boolean).join("; ");
      }
      tr.append(td);
    });
    return tr;
  }

  // Replace the options after the first `fixed` ones ("All …", "Free practice"), keeping the selection if it still exists.
  function fillFilter(select, fixed, options) {
    const keep = select.value;
    select.replaceChildren(...Array.from(select.options).slice(0, fixed), ...options.map(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    }));
    select.value = Array.from(select.options).some(option => option.value === keep) ? keep : "";
  }

  function fillFilters() {
    const trainees = [...new Set(sessions.map(s => s.trainee).filter(Boolean))].sort();
    fillFilter($("trainee-filter"), 1, trainees.map(name => [name, name]));
    const titles = new Map();
    for (const exercise of exercises ? exercises.values() : []) if (!exercise.demo_only) titles.set(exercise.id, exercise.title);
    for (const s of sessions) if (s.exercise && !titles.has(s.exercise.id)) titles.set(s.exercise.id, s.exercise.title);
    fillFilter($("exercise-filter"), 2, [...titles]);
  }

  function renderProgress() {
    const trainee = $("trainee-filter").value;
    const exercise = $("exercise-filter").value;
    const points = trainee && exercise && exercise !== "free" ? progressSeries(sessions, trainee, exercise) : [];
    $("progress").hidden = !points.length;
    if (!points.length) return;
    const title = $("exercise-filter").selectedOptions[0].textContent;
    $("progress-title").textContent = `Progress: ${trainee} · ${title}`;
    const svg = $("progress-chart");
    const width = Math.max(320, Math.round(svg.clientWidth || 600));
    const height = 180;
    const m = { top: 16, right: 56, bottom: 28, left: 44 };
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.replaceChildren();
    const xs = scaleLinear(0.5, points.length + 0.5, m.left, width - m.right);   // half a step of room each side
    const ys = scaleLinear(0, 100, height - m.bottom, m.top);
    for (const tick of [0, 50, 100]) {
      svgEl(svg, "line", { x1: m.left, x2: width - m.right, y1: ys(tick), y2: ys(tick), stroke: GPU_INK.grid });
      haloText(svg, { x: m.left - 6, y: ys(tick) + 4, "text-anchor": "end" }, `${tick}%`);
    }
    haloText(svg, { x: (m.left + width - m.right) / 2, y: height - 6, "text-anchor": "middle" }, "attempt");
    const color = GPU_SERIES[0].color;
    svgEl(svg, "path", { d: linePath(points.map(p => ({ x: p.n, y: p.percent })), xs, ys), fill: "none", stroke: color, "stroke-width": 2 });
    for (const p of points) {
      const dot = svgEl(svg, "circle", {
        cx: xs(p.n), cy: ys(p.percent), r: 4, fill: p.pending ? GPU_INK.surface : color, stroke: color, "stroke-width": 2,
      });
      svgEl(dot, "title", {}, `Attempt ${p.n}: ${p.text}${p.pending ? " (instructor review pending)" : ""}`);
    }
    const last = points[points.length - 1];
    haloText(svg, { x: xs(last.n) + 8, y: ys(last.percent) + 4, fill: GPU_INK.text }, `${last.percent}%`);
  }

  function renderList() {
    const column = COLUMNS[sortColumn];
    const rows = filterSessions(sessions, $("trainee-filter").value, $("exercise-filter").value)
      .sort((a, b) => sortDirection * compare(column.value(a), column.value(b)));
    renderProgress();
    $("sessions").replaceChildren(...rows.map(renderRow));
    headers.forEach((th, index) => {
      th.setAttribute("aria-sort", index !== sortColumn ? "none" : sortDirection > 0 ? "ascending" : "descending");
    });
    $("list-status").textContent = !sessions.length
      ? "No sessions recorded yet."
      : rows.length ? "" : "No sessions match the filter.";
  }

  async function loadExercises() {
    if (exercises !== null) return;
    try {
      const response = await fetch("/api/courses", { cache: "no-store" });
      if (!response.ok) return;              // courses are off: no rubric checkboxes or Save as demo
      const body = await response.json();
      exercises = new Map(body.courses.flatMap(group => group.exercises).map(e => [e.id, e]));
    } catch (_) { /* keep null */ }
  }

  async function loadSessions() {
    try {
      await loadExercises();
      const response = await fetch("/api/sessions");
      if (!response.ok) throw new Error(response.status === 404 ? "recording is off on this server" : `HTTP ${response.status}`);
      sessions = await response.json();
      fillFilters();
      renderList();
    } catch (error) {
      sessions = [];
      renderList();
      $("list-status").textContent = `Could not load sessions: ${error.message}`;
    }
  }

  function seekTo(seconds) {
    player.currentTime = seconds;
    player.play().catch(() => {});           // autoplay can be refused; the seek still happened
  }

  function renderTimeline(markers) {
    $("timeline").replaceChildren(...markers.map(marker => {
      const el = document.createElement("div");
      el.className = `marker ${marker.kind}`;
      el.style.left = `${marker.start * 100}%`;
      if (marker.kind === "energy") el.style.width = `${(marker.end - marker.start) * 100}%`;
      el.title = marker.label || marker.kind;
      el.addEventListener("click", () => seekTo(marker.seek_s));
      return el;
    }));
  }

  function renderIncidents(markers) {
    const items = markers.filter(m => m.kind === "incident").map(marker => {
      const li = document.createElement("li");
      const time = document.createElement("button");
      time.textContent = clock(marker.t_s);      // active time; the click seeks the video by video time
      time.addEventListener("click", () => seekTo(marker.seek_s));
      const message = document.createElement("span");
      message.textContent = marker.label || "Safety stop";
      li.append(time, message);
      return li;
    });
    if (!items.length) {
      const li = document.createElement("li");
      li.textContent = "No incidents";
      items.push(li);
    }
    $("incidents").replaceChildren(...items);
  }

  function renderSummary(session) {
    const stats = summaryOf(session);
    const sides = (values, unit, digits) =>
      `left ${(values && values.left || 0).toFixed(digits)} ${unit}, right ${(values && values.right || 0).toFixed(digits)} ${unit}`;
    const rows = [
      ["Trainee", session.trainee],
      ["Started", dateOf(session.id)],
      ["Scene", String(session.scene)],
      ["Mode", modeLabel(session.mode)],
      ["Duration", clock(stats.duration_s)],
      ["Path length", sides(stats.path_length_m, "m", 2)],
      ["Energy time", sides(stats.energy_s, "s", 1)],
      ["Incidents", String(stats.incidents || 0)],
      ["Peak VRAM used", gib(vramOf(session).peak_used_mib)],
      ["Mean VRAM used", gib(vramOf(session).mean_used_mib)],
      ["Peak PyTorch allocated", gib(vramOf(session).peak_allocated_mib)],
      ["Mean PyTorch allocated", gib(vramOf(session).mean_allocated_mib)],
      ["Peak host RAM (PSS)", gib(vramOf(session).peak_host_pss_mib)],
      ["Text cache", session.text_cache ? `${session.text_cache.status}${session.text_cache.rehashed ? " (re-hashed)" : ""}` : "—"],
      ["End reason", session.end_reason || "none (recording did not finish)"],
    ];
    if (session.write_error) rows.push(["Write error", session.write_error]);
    $("summary").replaceChildren(...rows.flatMap(([name, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = name;
      dd.textContent = value;
      return [dt, dd];
    }));
  }

  function renderRubric(id, data) {
    const exercise = data.exercise && exercises && exercises.get(data.exercise.id);
    $("rubric-section").hidden = !data.exercise;
    if (!data.exercise) return;
    $("rubric-title").textContent = `Rubric · ${data.exercise.title}`;
    const report = data.report;
    $("rubric-note").textContent = !exercise
      ? "The rubric is unavailable: this exercise is not loaded on the server."
      : report ? "" : "No report: the attempt did not finish.";
    $("save-review").disabled = !exercise || !report;
    if (!exercise) {
      $("rubric").replaceChildren();
      return;
    }
    const rows = rubricRows(exercise.rubric, report, data.review);
    const boxes = [];
    const total = () => {
      const checked = boxes.filter(box => box.checked).map(box => box.dataset.id);
      const max = report ? report.auto_max + report.instructor_max : 0;
      $("review-total").textContent = report ? `Total ${reviewTotal(rows, report, checked)} / ${max}` : "";
      return checked;
    };
    $("rubric").replaceChildren(...rows.map(row => {
      const li = document.createElement("li");
      if (row.auto) {
        const mark = row.passed === null ? "–" : row.passed ? "✓" : "✗";
        li.textContent = `${mark} ${row.text} · ${row.passed ? row.points : 0}/${row.points}`;
        const value = document.createElement("span");
        value.className = "value";
        value.textContent = row.gated ? "not scored: steps incomplete" : autoValueText(row.auto, row.value);
        li.append(value);
      } else {
        const label = document.createElement("label");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.dataset.id = row.id;
        box.checked = row.checked;
        box.addEventListener("change", total);
        boxes.push(box);
        label.append(box, ` ${row.text} · ${row.points} pts (instructor)`);
        li.append(label);
      }
      return li;
    }));
    $("review-notes").value = (data.review && data.review.notes) || "";
    $("review-status").textContent = data.review ? `Reviewed ${data.review.saved}` : "Not reviewed yet";
    total();
    $("save-review").onclick = async () => {
      $("review-status").textContent = "Saving…";
      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(id)}/review`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ checked: total(), notes: $("review-notes").value }),
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
        $("review-status").textContent = `Saved · ${body.points} of ${body.instructor_max} instructor points`;
      } catch (error) {
        $("review-status").textContent = `Could not save: ${error.message}`;
      }
    };
  }

  function renderSaveDemo(id) {
    $("save-demo-section").hidden = exercises === null;
    $("demo-status").textContent = "";
    $("demo-name").value = "";
    $("save-demo").onclick = async () => {
      $("demo-status").textContent = "Saving…";
      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(id)}/demo`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: $("demo-name").value }),
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
        $("demo-status").textContent = `Saved as "${body.name}". Play it from /train: Courses, Your demos.`;
      } catch (error) {
        $("demo-status").textContent = `Could not save: ${error.message}`;
      }
    };
  }

  function clearDetail() {
    player.pause();
    player.removeAttribute("src");
    player.load();
    player.hidden = true;
    $("video-message").textContent = "";
    $("rubric-section").hidden = true;
    $("save-demo-section").hidden = true;
    for (const id of ["timeline", "incidents", "summary", "rubric"]) $(id).replaceChildren();
    $("detail-status").textContent = "Loading…";
  }

  let detailRequest = 0;                     // lets a newer navigation discard a slower, older response
  async function showDetail(id) {
    const request = ++detailRequest;
    $("list").hidden = true;
    $("detail").hidden = false;
    clearDetail();
    let data;
    try {
      const response = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error(response.status === 404 ? "unknown session" : `HTTP ${response.status}`);
      data = await response.json();
    } catch (error) {
      if (request === detailRequest) $("detail-status").textContent = `Could not load session: ${error.message}`;
      return;
    }
    if (request !== detailRequest) return;
    const { session, events } = data;
    $("detail-status").textContent = "";
    if (session.video_ok) {
      player.hidden = false;
      player.src = `/recordings/${encodeURIComponent(id)}/video.mp4`;
    } else {
      $("video-message").textContent = `No video recorded (${session.video_error})`;
    }
    const markers = timelineMarkers(session, events);
    renderTimeline(markers);
    renderIncidents(markers);
    renderSummary(session);
    await loadExercises();
    if (request !== detailRequest) return;
    renderRubric(id, data);
    renderSaveDemo(id);
  }

  function showList() {
    detailRequest++;
    clearDetail();
    $("detail").hidden = true;
    $("list").hidden = false;
    loadSessions();
  }

  function route() {
    let id = location.hash.slice(1);
    try { id = decodeURIComponent(id); } catch (error) { /* malformed escape: the server answers 404 for it */ }
    if (id) showDetail(id); else showList();
  }

  player.addEventListener("error", () => {
    if (!player.getAttribute("src")) return;
    player.hidden = true;
    $("video-message").textContent = "The video could not be loaded.";
  });
  // Read-only: the current model mode is changed on /train.
  async function refreshMode() {
    try {
      const response = await fetch("/healthz", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const health = await response.json();
      $("current-mode").textContent = health.mode_switching
        ? "Model mode: switching…"
        : `Model mode: ${modeLabel(health.mode)}`;
    } catch (error) {
      $("current-mode").textContent = "Model mode: unavailable";
    } finally {
      setTimeout(refreshMode, 5000);
    }
  }

  $("trainee-filter").addEventListener("change", renderList);
  $("exercise-filter").addEventListener("change", renderList);
  window.addEventListener("resize", renderProgress);
  $("back").addEventListener("click", () => { location.hash = ""; });
  window.addEventListener("hashchange", route);
  route();
  refreshMode();
}
