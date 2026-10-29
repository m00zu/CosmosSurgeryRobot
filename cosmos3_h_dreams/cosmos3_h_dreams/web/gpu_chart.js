// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// GPU memory charts for /gpu and the /train resource graphs: pure helpers plus an SVG renderer.
// Helpers are top-level function declarations so node tests can call them in a vm context.
// Every label goes into the page with textContent; text never uses a series colour.

var GPU_SERIES = [
  { key: "used_mib", name: "used", color: "#3987e5" },
  { key: "torch_reserved_mib", name: "reserved", color: "#d95926" },
  { key: "torch_allocated_mib", name: "allocated", color: "#199e70" },
];
// Host RAM chart (/gpu, below the GPU chart): one series, its own GiB axis.
var GPU_HOST_SERIES = [{ key: "host_pss_mib", name: "host PSS", color: "#3987e5" }];
var GPU_INK = { text: "#e6e9ef", muted: "#aeb8c8", faint: "#68758b", grid: "#232b3a", surface: "#10131a" };
var GPU_MIB_PER_GIB = 1024;
var GPU_MODE_NAMES = { standard: "Standard", balanced: "Balanced", low_memory: "Low memory" };

// The text K/V cache line for /healthz.text_cache ({status, rehashed, hash_s, ...}); "" when there is none.
function textCacheText(cache) {
  if (!cache) return "";
  const lines = {
    off: "off",
    hit: "hit (model loaded without understanding weights)",
    built: "built and saved",
    "built-unsaved": "built but not saved (check server logs); rebuilt on the next start",
  };
  const hashed = cache.hash_s === null || cache.hash_s === undefined ? "" : ` in ${Number(cache.hash_s).toFixed(1)} s`;
  return `Text cache: ${lines[cache.status] || cache.status}${cache.rehashed ? ` · checkpoint re-hashed${hashed}` : ""}`;
}

// Highest finite value (MiB) of `keys` over the samples, or null when there is none.
function seriesMaxMib(samples, keys) {
  let max = null;
  for (const sample of samples) {
    for (const key of keys) {
      const v = sample[key];
      if (v !== null && v !== undefined && Number.isFinite(v) && (max === null || v > max)) max = v;
    }
  }
  return max;
}

// "low_memory" -> "Low memory"; "failed: standard" -> "failed: Standard".
function modeName(label) {
  const text = String(label || "");
  const failed = text.startsWith("failed: ");
  const mode = failed ? text.slice(8) : text;
  return (failed ? "failed: " : "") + (GPU_MODE_NAMES[mode] || mode);
}

// That GPU's samples with t >= nowS - spanS, sorted by t.
function windowSamples(samples, gpu, nowS, spanS) {
  const from = nowS - spanS;
  return samples.filter(s => s.gpu === gpu && s.t >= from).sort((a, b) => a.t - b.t);
}

function scaleLinear(d0, d1, r0, r1) {
  const span = d1 - d0;
  return value => (span === 0 ? r0 : r0 + ((value - d0) / span) * (r1 - r0));
}

// SVG path through {x, y} points; a null or undefined y ends the segment and the next point starts a new M.
function linePath(points, xs, ys) {
  let d = "";
  let pen = false;
  for (const point of points) {
    if (point.y === null || point.y === undefined || !Number.isFinite(point.y)) {
      pen = false;
      continue;
    }
    const x = +xs(point.x).toFixed(1);
    const y = +ys(point.y).toFixed(1);
    d += `${pen ? "L" : "M"}${x},${y}`;
    pen = true;
  }
  return d;
}

// Points {x: t, y} with a {x, y: null} break inserted wherever consecutive x differ by more than
// 3 × the median interval (3 s when there are fewer than two intervals), so linePath never bridges
// a server restart, sampler stall or outage with a straight line.
function withGaps(points) {
  const steps = [];
  for (let i = 1; i < points.length; i++) steps.push(points[i].x - points[i - 1].x);
  const sorted = steps.filter(d => d > 0).sort((a, b) => a - b);
  const median = sorted.length >= 2 ? sorted[sorted.length >> 1] : null;
  const limit = median === null ? 3 : 3 * median;
  const out = [];
  points.forEach((point, i) => {
    if (i > 0 && point.x - points[i - 1].x > limit) out.push({ x: (point.x + points[i - 1].x) / 2, y: null });
    out.push(point);
  });
  return out;
}

// Runs of consecutive points with a numeric y (what linePath draws as separate M segments).
function runs(points) {
  const out = [];
  let run = [];
  for (const point of points) {
    if (point.y === null || point.y === undefined || !Number.isFinite(point.y)) {
      if (run.length) out.push(run);
      run = [];
    } else {
      run.push(point);
    }
  }
  if (run.length) out.push(run);
  return out;
}

// The smallest 1, 2, 2.5 or 5 × 10^n step that is >= raw.
function niceStep(raw) {
  const power = Math.pow(10, Math.floor(Math.log10(raw > 0 ? raw : 1)));
  return [1, 2, 2.5, 5, 10].map(m => m * power).find(s => s >= raw);
}

// Zoomed y domain for "Fit to data": from max(0, min − pad) to max + pad, widened to round
// ticks. Returns { lo, hi, ticks } (same unit as values), or null when there are no values.
function fitDomain(values, pad) {
  const finite = values.filter(v => v !== null && v !== undefined && Number.isFinite(v));
  if (!finite.length) return null;
  const low = Math.max(0, Math.min(...finite) - pad);
  const high = Math.max(...finite) + pad;
  const step = niceStep(Math.max(high - low, 1e-6) / 4);
  const lo = Math.floor(low / step + 1e-9) * step;
  const hi = Math.ceil(high / step - 1e-9) * step;
  const ticks = [];
  for (let i = 0; lo + i * step <= hi + step * 1e-6; i++) ticks.push(+(lo + i * step).toFixed(6));
  return { lo: ticks[0], hi: ticks[ticks.length - 1], ticks };
}

// GiB values of `keys` over the samples for the fit domain; null or missing counters (e.g. PyTorch
// numbers before CUDA is initialised) are skipped, never counted as 0.
function fitValues(samples, keys) {
  const out = [];
  for (const sample of samples) {
    for (const key of keys) {
      const mib = sample[key];
      if (mib !== null && mib !== undefined && Number.isFinite(mib)) out.push(mib / GPU_MIB_PER_GIB);
    }
  }
  return out;
}

var GPU_MARK_PAIRS = { attempt_start: "attempt_end", mode_switch_start: "mode_switch_end" };

// Marks still worth keeping at `cutoff`: anything at or after it, plus the start of a pair
// (attempt, mode switch) that is still open or whose end is at or after the cutoff.
function pruneMarks(marks, cutoff) {
  const sorted = marks.slice().sort((a, b) => a.t - b.t);
  const keep = new Set();
  const open = {};                           // start kind -> index of its unmatched start
  sorted.forEach((mark, i) => {
    if (mark.t >= cutoff) keep.add(i);
    if (mark.kind in GPU_MARK_PAIRS) {
      open[mark.kind] = i;
      return;
    }
    for (const [start, end] of Object.entries(GPU_MARK_PAIRS)) {
      if (mark.kind === end && open[start] !== undefined) {
        if (mark.t >= cutoff) keep.add(open[start]);
        delete open[start];
      }
    }
  });
  for (const index of Object.values(open)) keep.add(index);
  return sorted.filter((mark, i) => keep.has(i));
}

// Round ticks from 0 to at least max, about `count` of them (1, 2, 2.5 or 5 × 10^n steps).
function niceTicks(max, count) {
  const top = max > 0 ? max : 1;
  const step = niceStep(top / Math.max(1, count - 1));
  const ticks = [];
  for (let i = 0; ; i++) {
    const tick = +(i * step).toFixed(6);
    ticks.push(tick);
    if (tick >= top) break;
  }
  return ticks;
}

// Index of the sample whose t is closest to `t` (samples sorted by t); -1 when empty.
function nearestIndex(samples, t) {
  if (!samples.length) return -1;
  let lo = 0;
  let hi = samples.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (samples[mid].t < t) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && t - samples[lo - 1].t <= samples[lo].t - t) return lo - 1;
  return lo;
}

function formatGiB(mib) {
  if (mib === null || mib === undefined || !Number.isFinite(Number(mib))) return "—";
  return `${(Number(mib) / GPU_MIB_PER_GIB).toFixed(1)} GiB`;
}

// Spread label centres (any order) at least `gap` apart inside [lo, hi], keeping each as close to its target as it can.
function spreadLabels(ys, gap, lo, hi) {
  const order = ys.map((y, i) => i).sort((a, b) => ys[a] - ys[b]);
  const out = ys.slice();
  let prev = -Infinity;
  for (const i of order) {
    out[i] = Math.max(ys[i], prev + gap, lo);
    prev = out[i];
  }
  let next = Infinity;
  for (const i of order.slice().reverse()) {
    out[i] = Math.min(out[i], next - gap, hi);
    next = out[i];
  }
  return out;
}

function clockText(t) {
  const d = new Date(t * 1000);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map(n => String(n).padStart(2, "0")).join(":");
}

// ---- DOM -------------------------------------------------------------------------------
var GPU_SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(parent, tag, attrs, text) {
  const node = document.createElementNS(GPU_SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs || {})) node.setAttribute(name, String(value));
  if (text !== undefined) node.textContent = text;
  if (parent) parent.append(node);
  return node;
}

function haloText(parent, attrs, text) {
  // A surface-coloured halo keeps text legible where it crosses a line.
  return svgEl(parent, "text", {
    fill: GPU_INK.muted, "font-size": 12, "paint-order": "stroke", stroke: GPU_INK.surface,
    "stroke-width": 4, "stroke-linejoin": "round", ...attrs,
  }, text);
}

// data: { samples (one GPU, sorted), t0, t1, totalMib, baselineMib, marks, fit (zoom y to the data),
//         series (default GPU_SERIES), details (util/temp row in the tooltip; default true) }
// options: { height, tooltip (absolutely positioned element), live (aria-live element) }
function renderChart(svg, data, options) {
  const state = svg.gpuChart || (svg.gpuChart = { cursorT: null, follow: false });
  state.data = data;
  state.options = options || {};
  const width = Math.max(320, Math.round(svg.clientWidth || svg.getBoundingClientRect().width || 900));
  const height = state.options.height || 360;
  const narrow = width < 600;
  const m = { top: 56, right: narrow ? 12 : 168, bottom: 44, left: 52 };
  const plotW = width - m.left - m.right;
  const plotH = height - m.top - m.bottom;
  const samples = data.samples;
  const series = (data.series || GPU_SERIES).filter(s => samples.some(sample => sample[s.key] !== null && sample[s.key] !== undefined));
  // Fit: pad by 10% of the range, at least 0.05 GiB, so a 30 MB creep is visible.
  const finite = data.fit ? fitValues(samples, series.map(s => s.key)) : [];
  const zoom = data.fit && finite.length
    ? fitDomain(finite, Math.max(0.05, 0.1 * (Math.max(...finite) - Math.min(...finite))))
    : null;
  const dataMax = seriesMaxMib(samples, (data.series || GPU_SERIES).map(s => s.key));
  const ticks = zoom ? zoom.ticks : niceTicks((data.totalMib || Math.max(1, dataMax || 0)) / GPU_MIB_PER_GIB, 5);
  const loGiB = ticks[0];
  const maxGiB = ticks[ticks.length - 1];
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
  const decimals = Math.min(3, (String(+step.toFixed(6)).split(".")[1] || "").length);
  const xs = scaleLinear(data.t0, data.t1, m.left, m.left + plotW);
  const ys = scaleLinear(loGiB * GPU_MIB_PER_GIB, maxGiB * GPU_MIB_PER_GIB, m.top + plotH, m.top);
  Object.assign(state, { xs, ys, m, plotW, plotH, width, height, series });

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("height", height);
  svg.replaceChildren();

  // Grid and y-axis (GiB): solid hairlines, recessive.
  for (const tick of ticks) {
    const y = ys(tick * GPU_MIB_PER_GIB);
    svgEl(svg, "line", { x1: m.left, x2: m.left + plotW, y1: y, y2: y, stroke: GPU_INK.grid });
    svgEl(svg, "text", { x: m.left - 8, y: y + 4, "text-anchor": "end", fill: GPU_INK.faint, "font-size": 12 }, tick.toFixed(decimals));
  }
  haloText(svg, { x: zoom ? 4 : m.left - 8, y: m.top - 26 + (zoom ? -16 : 0), "text-anchor": zoom ? "start" : "end", fill: zoom ? GPU_INK.text : GPU_INK.faint }, zoom ? "GiB (zoomed)" : "GiB");
  if (zoom && loGiB > 0) {   // axis-break hint: the y-axis does not start at zero
    const by = m.top + plotH - 6;
    svgEl(svg, "path", { d: `M${m.left - 5},${by + 3}l10,-4M${m.left - 5},${by + 7}l10,-4`, stroke: GPU_INK.muted, "stroke-width": 1.5 });
  }

  // X axis: local clock ticks every 1, 2 or 5 minutes.
  const spanS = data.t1 - data.t0;
  const perTick = spanS / Math.max(1, Math.floor(plotW / 70));   // keep tick labels at least ~70px apart
  const every = [60, 120, 300, 600, 900, 1800].find(e => e >= perTick) || 3600;
  const axisY = m.top + plotH;
  svgEl(svg, "line", { x1: m.left, x2: m.left + plotW, y1: axisY, y2: axisY, stroke: GPU_INK.faint });
  for (let t = Math.ceil(data.t0 / every) * every; t <= data.t1; t += every) {
    const x = xs(t);
    svgEl(svg, "line", { x1: x, x2: x, y1: axisY, y2: axisY + 4, stroke: GPU_INK.faint });
    svgEl(svg, "text", { x, y: axisY + 34, "text-anchor": "middle", fill: GPU_INK.faint, "font-size": 12 }, clockText(t).slice(0, 5));
  }

  const inRange = t => t >= data.t0 && t <= data.t1;
  const marks = (data.marks || []).slice().sort((a, b) => a.t - b.t);

  // Mode switches: a faint band from start to end, a vertical rule at the end labelled with the new mode.
  let open = null;
  const modeLabels = [];
  for (const mark of marks) {
    if (mark.kind === "mode_switch_start") open = mark;
    if (mark.kind !== "mode_switch_end") continue;
    const start = Math.max(data.t0, open ? open.t : mark.t);
    if (open && mark.t >= data.t0) {
      svgEl(svg, "rect", { x: xs(start), y: m.top, width: Math.max(0, xs(Math.min(mark.t, data.t1)) - xs(start)), height: plotH, fill: GPU_INK.muted, opacity: 0.08 });
    }
    open = null;
    if (!inRange(mark.t)) continue;
    const x = xs(mark.t);
    svgEl(svg, "line", { x1: x, x2: x, y1: m.top - 4, y2: axisY, stroke: GPU_INK.muted, "stroke-width": 1 });
    modeLabels.push({ x, text: `→ ${modeName(mark.label)}` });
  }
  if (open && open.t <= data.t1) {
    const x = xs(Math.max(data.t0, open.t));
    svgEl(svg, "rect", { x, y: m.top, width: m.left + plotW - x, height: plotH, fill: GPU_INK.muted, opacity: 0.08 });
    modeLabels.push({ x, text: "switching…" });
  }
  // Up to three label rows; a label that fits in none is dropped and its rule stays unlabelled.
  const rowRight = [-Infinity, -Infinity, -Infinity];
  for (const label of modeLabels) {
    const w = label.text.length * 7;
    const anchorEnd = label.x + 4 + w > width - 4;
    const left = anchorEnd ? label.x - 4 - w : label.x + 4;
    const row = rowRight.findIndex(right => left >= right + 6);
    if (row < 0) continue;
    haloText(svg, { x: anchorEnd ? label.x - 4 : label.x + 4, y: m.top - 8 - row * 16, "text-anchor": anchorEnd ? "end" : "start" }, label.text);
    rowRight[row] = left + w;
  }

  // Reference lines: dashed capacity, dotted baseline; their labels sit at the left inside the plot.
  // When zoomed, a reference outside the domain is not drawn; a note at the plot edge says where it is.
  const loMib = loGiB * GPU_MIB_PER_GIB;
  const hiMib = maxGiB * GPU_MIB_PER_GIB;
  const notes = { above: [], below: [] };
  const reference = (mib, dash, name, labelDy) => {
    if (mib === null || mib === undefined) return;
    if (mib > hiMib + 1e-6) { notes.above.push(`${name} ${formatGiB(mib)} above`); return; }
    if (mib < loMib - 1e-6) { notes.below.push(`${name} ${formatGiB(mib)} below`); return; }
    const y = ys(mib);
    svgEl(svg, "line", { x1: m.left, x2: m.left + plotW, y1: y, y2: y, stroke: GPU_INK.muted, "stroke-dasharray": dash });
    haloText(svg, { x: m.left + 6, y: y + labelDy }, `${name} ${formatGiB(mib)}`);
  };
  reference(data.totalMib, "6 4", "capacity", -5);
  reference(data.baselineMib, "1 3", "baseline", 15);
  notes.above.forEach((note, i) => haloText(svg, { x: m.left + 6, y: m.top + 14 + i * 15 }, `↑ ${note}`));
  notes.below.forEach((note, i) => haloText(svg, { x: m.left + 6, y: m.top + plotH - 6 - i * 15 }, `↓ ${note}`));

  // Series: 2px lines, drawn allocated → used so "used" sits on top.
  for (const s of series.slice().reverse()) {
    svgEl(svg, "path", {
      d: linePath(withGaps(samples.map(sample => ({ x: sample.t, y: sample[s.key] }))), xs, ys),
      fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
    });
  }

  // Direct end labels (line key + name + value, in text colour) with leader lines when nudged apart.
  const last = samples[samples.length - 1];
  if (!narrow && last) {
    const ends = series.map(s => {
      const lastWith = [...samples].reverse().find(sample => sample[s.key] !== null && sample[s.key] !== undefined);
      return { s, x: xs(lastWith.t), y: ys(lastWith[s.key]), value: lastWith[s.key] };
    });
    const placed = spreadLabels(ends.map(e => e.y), 18, m.top, axisY);
    ends.forEach((end, i) => {
      const lx = m.left + plotW + 10;
      svgEl(svg, "circle", { cx: end.x, cy: end.y, r: 4, fill: end.s.color, stroke: GPU_INK.surface, "stroke-width": 2 });
      svgEl(svg, "path", { d: `M${end.x + 5},${end.y}L${lx - 2},${placed[i]}`, stroke: GPU_INK.faint, fill: "none" });
      svgEl(svg, "line", { x1: lx, x2: lx + 14, y1: placed[i], y2: placed[i], stroke: end.s.color, "stroke-width": 2, "stroke-linecap": "round" });
      const text = svgEl(svg, "text", { x: lx + 20, y: placed[i] + 4, fill: GPU_INK.muted, "font-size": 13 });
      svgEl(text, "tspan", {}, `${end.s.name} `);
      svgEl(text, "tspan", { fill: GPU_INK.text, "font-weight": 600 }, formatGiB(end.value));
    });
  }

  // Attempts as a bar under the axis, incidents as triangles on top of it.
  const attempts = [];
  let attempt = null;
  for (const mark of marks) {
    if (mark.kind === "attempt_start") attempt = { t0: mark.t, t1: data.t1 };
    if (mark.kind === "attempt_end" && attempt) { attempt.t1 = mark.t; attempts.push(attempt); attempt = null; }
  }
  if (attempt) attempts.push(attempt);
  for (const a of attempts) {
    if (a.t1 < data.t0 || a.t0 > data.t1) continue;
    const x0 = xs(Math.max(a.t0, data.t0));
    const x1 = xs(Math.min(a.t1, data.t1));
    svgEl(svg, "rect", { x: x0, y: axisY + 8, width: Math.max(2, x1 - x0), height: 4, rx: 2, fill: GPU_INK.faint });
  }
  for (const mark of marks) {
    if (mark.kind !== "incident" || !inRange(mark.t)) continue;
    const x = xs(mark.t);
    const tri = svgEl(svg, "path", { d: `M${x},${axisY + 6}l5,9h-10z`, fill: GPU_INK.text, stroke: GPU_INK.surface, "stroke-width": 1.5 });
    svgEl(tri, "title", {}, `Incident ${clockText(mark.t)}: ${mark.label}`);
  }

  // Crosshair layer (drawn last) and the interaction handlers, attached once per svg.
  state.cross = svgEl(svg, "g", { "pointer-events": "none" });
  svgEl(svg, "rect", { class: "hit", x: m.left, y: m.top, width: plotW, height: plotH + m.bottom, fill: "transparent" });
  if (!state.bound) bindChart(svg, state);
  if (state.follow && samples.length) state.cursorT = last.t;
  drawCursor(svg, state);
}

function drawCursor(svg, state) {
  const { data, xs, ys, m, plotH, series } = state;
  const tooltip = state.options.tooltip;
  state.cross.replaceChildren();
  const index = state.cursorT === null ? -1 : nearestIndex(data.samples, state.cursorT);
  if (index < 0) {
    if (tooltip) tooltip.hidden = true;
    return;
  }
  const sample = data.samples[index];
  const x = xs(sample.t);
  svgEl(state.cross, "line", { x1: x, x2: x, y1: m.top, y2: m.top + plotH, stroke: GPU_INK.muted });
  for (const s of series) {
    if (sample[s.key] === null || sample[s.key] === undefined) continue;
    svgEl(state.cross, "circle", { cx: x, cy: ys(sample[s.key]), r: 4, fill: s.color, stroke: GPU_INK.surface, "stroke-width": 2 });
  }
  if (!tooltip) return;
  const rows = [];
  const head = document.createElement("div");
  head.className = "tip-time";
  head.textContent = clockText(sample.t);
  rows.push(head);
  for (const s of series) {
    const row = document.createElement("div");
    const key = document.createElement("span");
    key.className = "tip-key";
    key.style.background = s.color;
    const value = document.createElement("strong");
    value.textContent = formatGiB(sample[s.key]);
    const name = document.createElement("span");
    name.className = "tip-name";
    name.textContent = ` ${s.name}`;
    row.append(key, value, name);
    rows.push(row);
  }
  const extra = document.createElement("div");
  extra.className = "tip-name";
  extra.textContent = data.details === false ? "" : [
    sample.util_pct === null || sample.util_pct === undefined ? null : `util ${Math.round(sample.util_pct)}%`,
    sample.temp_c === null || sample.temp_c === undefined ? null : `${Math.round(sample.temp_c)} °C`,
  ].filter(Boolean).join(" · ");
  if (extra.textContent) rows.push(extra);
  tooltip.replaceChildren(...rows);
  tooltip.hidden = false;
  // Keep the tooltip beside the crosshair, flipping left near the right edge (scaled from viewBox to CSS pixels).
  const scale = (svg.clientWidth || state.width) / state.width;
  const tipW = tooltip.offsetWidth || 160;
  const px = x * scale;
  tooltip.style.left = `${px + 12 + tipW > svg.clientWidth ? px - 12 - tipW : px + 12}px`;
  tooltip.style.top = `${m.top * scale}px`;
}

function bindChart(svg, state) {
  state.bound = true;
  svg.setAttribute("tabindex", "0");
  const toT = event => {
    const box = svg.getBoundingClientRect();
    const vx = ((event.clientX - box.left) / box.width) * state.width;
    return state.data.t0 + ((vx - state.m.left) / state.plotW) * (state.data.t1 - state.data.t0);
  };
  svg.addEventListener("pointermove", event => {
    state.follow = false;
    state.cursorT = toT(event);
    drawCursor(svg, state);
  });
  svg.addEventListener("pointerleave", () => {
    if (document.activeElement === svg) return;   // a keyboard cursor stays put
    state.cursorT = null;
    drawCursor(svg, state);
  });
  svg.addEventListener("blur", () => {
    state.cursorT = null;
    state.follow = false;
    drawCursor(svg, state);
  });
  svg.addEventListener("keydown", event => {
    const samples = state.data.samples;
    if (!samples.length) return;
    const lastIndex = samples.length - 1;
    let index = state.cursorT === null ? lastIndex : nearestIndex(samples, state.cursorT);
    const stride = event.shiftKey ? 10 : 1;
    if (event.key === "ArrowLeft") index -= state.cursorT === null ? 0 : stride;
    else if (event.key === "ArrowRight") index += stride;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = lastIndex;
    else if (event.key === "Escape") { state.cursorT = null; state.follow = false; drawCursor(svg, state); return; }
    else return;
    event.preventDefault();
    index = Math.max(0, Math.min(lastIndex, index));
    state.cursorT = samples[index].t;
    state.follow = index === lastIndex;   // at the newest sample the cursor follows live data
    drawCursor(svg, state);
    const live = state.options.live;
    if (live) {
      const sample = samples[index];
      live.textContent = `${clockText(sample.t)}: ` + state.series.map(s => `${s.name} ${formatGiB(sample[s.key])}`).join(", ");
    }
  });
}
