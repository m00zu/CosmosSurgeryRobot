// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with: node cosmos3_h_dreams/tests/test_gpu_chart.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../cosmos3_h_dreams/web/gpu_chart.js"), "utf8"), context);
// Arrays built inside the vm context have another realm's Array.prototype; copy them so strict deepEqual compares values.
const plain = v => Array.from(v);
const c = context;

// windowSamples: filter by GPU and window, sort by t.
const samples = [
  { t: 105, gpu: 0, used_mib: 5 },
  { t: 90, gpu: 0, used_mib: 1 },
  { t: 100, gpu: 0, used_mib: 3 },
  { t: 101, gpu: 1, used_mib: 9 },
  { t: 95, gpu: 0, used_mib: 2 },
];
assert.deepEqual(plain(c.windowSamples(samples, 0, 105, 10).map(s => s.t)), [95, 100, 105]);
assert.deepEqual(plain(c.windowSamples(samples, 1, 105, 10).map(s => s.t)), [101]);
assert.deepEqual(plain(c.windowSamples(samples, 2, 105, 10)), []);

// scaleLinear maps the endpoints (and works inverted, as a y-axis does).
const x = c.scaleLinear(10, 20, 0, 200);
assert.equal(x(10), 0);
assert.equal(x(20), 200);
assert.equal(x(15), 100);
const y = c.scaleLinear(0, 100, 300, 0);
assert.equal(y(0), 300);
assert.equal(y(100), 0);
assert.equal(c.scaleLinear(5, 5, 0, 10)(5), 0);   // a zero-width domain does not divide by zero

// linePath: M…L…, a null y starts a new M segment.
const ident = v => v;
assert.equal(c.linePath([{ x: 0, y: 1 }, { x: 1, y: 2 }], ident, ident), "M0,1L1,2");
const gap = c.linePath(
  [{ x: 0, y: 1 }, { x: 1, y: 2 }, { x: 2, y: null }, { x: 3, y: 4 }, { x: 4, y: 5 }],
  ident, ident,
);
assert.equal(gap, "M0,1L1,2M3,4L4,5");
assert.equal(c.linePath([], ident, ident), "");
assert.equal(c.linePath([{ x: 0, y: null }], ident, ident), "");

// niceTicks: starts at 0, a round step, last tick >= max.
const max = 81559 / 1024;
const ticks = plain(c.niceTicks(max, 5));
assert.equal(ticks[0], 0);
const step = ticks[1] - ticks[0];
assert.ok([1, 2, 2.5, 5, 10, 20, 25, 50].includes(step), `round step, got ${step}`);
ticks.forEach((t, i) => assert.equal(t, i * step));
assert.ok(ticks[ticks.length - 1] >= max);
assert.ok(ticks.length >= 4 && ticks.length <= 7, `about 5 ticks, got ${ticks.length}`);
assert.deepEqual(plain(c.niceTicks(0, 5)).slice(0, 1), [0]);
assert.ok(c.niceTicks(0, 5).length >= 2);

// nearestIndex: binary search over t.
const ts = [{ t: 0 }, { t: 10 }, { t: 20 }, { t: 30 }];
assert.equal(c.nearestIndex(ts, -5), 0);
assert.equal(c.nearestIndex(ts, 0), 0);
assert.equal(c.nearestIndex(ts, 99), 3);
assert.equal(c.nearestIndex(ts, 14), 1);
assert.equal(c.nearestIndex(ts, 16), 2);
assert.equal(c.nearestIndex([], 5), -1);

// formatGiB: one decimal.
assert.equal(c.formatGiB(15052.8), "14.7 GiB");
assert.equal(c.formatGiB(81559), "79.6 GiB");
assert.equal(c.formatGiB(null), "—");

// spreadLabels: keeps order, enforces the gap, stays inside the bounds.
assert.deepEqual(plain(c.spreadLabels([50, 10, 90], 18, 0, 100)), [50, 10, 90]);
const spread = plain(c.spreadLabels([50, 52, 51], 18, 0, 100));
const sorted = spread.slice().sort((a, b) => a - b);
assert.ok(sorted[1] - sorted[0] >= 18 && sorted[2] - sorted[1] >= 18);
assert.ok(spread[0] < spread[2] && spread[2] < spread[1]);   // order follows the targets
assert.ok(Math.max(...plain(c.spreadLabels([99, 100, 100], 18, 0, 100))) <= 100);

// modeName: friendly mode names, failures keep their prefix.
assert.equal(c.modeName("low_memory"), "Low memory");
assert.equal(c.modeName("balanced"), "Balanced");
assert.equal(c.modeName("failed: standard"), "failed: Standard");
assert.equal(c.modeName("other"), "other");

// withGaps: a break where consecutive t differ by more than 3 × the median interval.
const pts = [0, 1, 2, 3, 4, 20, 21, 22].map(t => ({ x: t, y: 5 }));
const gapped = plain(c.withGaps(pts));
assert.equal(gapped.length, pts.length + 1);
assert.equal(gapped[5].y, null);
assert.ok(gapped[5].x > 4 && gapped[5].x < 20);
assert.equal(c.linePath(gapped, ident, ident), "M0,5L1,5L2,5L3,5L4,5M20,5L21,5L22,5");
assert.equal(c.withGaps([0, 1, 2, 3.5, 5].map(t => ({ x: t, y: 1 }))).length, 5);   // jitter under 3× is not a gap
assert.equal(c.withGaps([{ x: 0, y: 1 }, { x: 2, y: 1 }]).length, 2);                // one interval: 3 s fallback
assert.equal(c.withGaps([{ x: 0, y: 1 }, { x: 10, y: 1 }]).length, 3);
assert.deepEqual(plain(c.withGaps([])), []);
assert.deepEqual(plain(c.runs(gapped).map(r => r.length)), [5, 3]);

// fitDomain: padded, round, never below zero, ticks cover the data.
const fit = c.fitDomain([14.70, 14.73, 14.76], 0.05);
assert.ok(fit.lo <= 14.65 && fit.lo > 14.0, `lo ${fit.lo}`);
assert.ok(fit.hi >= 14.81 && fit.hi < 15.5, `hi ${fit.hi}`);
const fticks = plain(fit.ticks);
assert.equal(fticks[0], fit.lo);
assert.equal(fticks[fticks.length - 1], fit.hi);
const fstep = +(fticks[1] - fticks[0]).toFixed(6);
fticks.forEach((t, i) => assert.ok(Math.abs(t - (fit.lo + i * fstep)) < 1e-6));
assert.ok(fticks.length >= 3 && fticks.length <= 7, `ticks ${fticks}`);
assert.equal(c.fitDomain([0.01, 0.02], 0.05).lo, 0);   // clamped at zero
assert.equal(c.fitDomain([], 0.05), null);
assert.equal(c.fitDomain([null, undefined], 0.05), null);
const flat = c.fitDomain([8.2, 8.2], 0.05);
assert.ok(flat.lo < 8.2 && flat.hi > 8.2);

// fitValues: GiB values of the plotted series, skipping null counters (samples before CUDA init),
// so the zoomed axis is not dragged down to 0.
const early = [];
for (let i = 0; i < 10; i++) early.push({ t: i, used_mib: 15052.8 + i, torch_reserved_mib: null, torch_allocated_mib: null });
for (let i = 10; i < 20; i++) early.push({ t: i, used_mib: 15052.8 + i, torch_reserved_mib: 14500 + i, torch_allocated_mib: 12500 + i });
const fv = plain(c.fitValues(early, ["used_mib", "torch_reserved_mib", "torch_allocated_mib"]));
assert.equal(fv.length, 40);
assert.ok(fv.every(v => v > 12), `no zeros from null counters: min ${Math.min(...fv)}`);
const zoomed = c.fitDomain(fv, 0.05);
assert.ok(zoomed.lo > 11, `zoomed lower bound stays near the data, got ${zoomed.lo}`);
assert.equal(plain(c.fitValues([{ used_mib: undefined }], ["used_mib"])).length, 0);

// pruneMarks: drop what ended before the cutoff, keep open or straddling pairs.
const pm = [
  { t: 10, kind: "attempt_start", label: "old" },
  { t: 20, kind: "attempt_end", label: "old" },
  { t: 30, kind: "incident", label: "old incident" },
  { t: 40, kind: "attempt_start", label: "long" },            // ends inside the window
  { t: 50, kind: "mode_switch_start", label: "low_memory" },  // still open
  { t: 150, kind: "attempt_end", label: "long" },
  { t: 160, kind: "incident", label: "new incident" },
  { t: 170, kind: "attempt_start", label: "open" },
];
const kept = plain(c.pruneMarks(pm, 100)).map(m => `${m.t}:${m.kind}`);
assert.deepEqual(kept, ["40:attempt_start", "50:mode_switch_start", "150:attempt_end", "160:incident", "170:attempt_start"]);
assert.deepEqual(plain(c.pruneMarks([], 100)), []);

// seriesMaxMib: the highest finite value of the plotted keys (MiB), null when there is none;
// the host RAM chart scales its own y-axis from it (no capacity line).
assert.equal(c.seriesMaxMib([{ host_pss_mib: 2048 }, { host_pss_mib: null }, { host_pss_mib: 4096 }], ["host_pss_mib"]), 4096);
assert.equal(c.seriesMaxMib([{ used_mib: 10, host_pss_mib: 5 }], ["host_pss_mib"]), 5);
assert.equal(c.seriesMaxMib([{ host_pss_mib: null }, {}], ["host_pss_mib"]), null);
assert.equal(c.seriesMaxMib([], ["host_pss_mib"]), null);

// The host RAM chart has one series in the series-1 colour.
assert.equal(c.GPU_HOST_SERIES.length, 1);
assert.equal(c.GPU_HOST_SERIES[0].key, "host_pss_mib");
assert.equal(c.GPU_HOST_SERIES[0].name, "host PSS");
assert.equal(c.GPU_HOST_SERIES[0].color, "#3987e5");

// textCacheText: the status line under the model status (/train) and in the /gpu footer.
assert.equal(c.textCacheText({ status: "hit", rehashed: false, path: "/c/t.pt", hash_s: null }), "Text cache: hit (model loaded without understanding weights)");
assert.equal(c.textCacheText({ status: "built", rehashed: false, path: "/c/t.pt", hash_s: 16.8 }), "Text cache: built and saved");
assert.equal(c.textCacheText({ status: "off", rehashed: false, path: null, hash_s: null }), "Text cache: off");
assert.equal(c.textCacheText({ status: "built-unsaved", rehashed: false, path: "/c/t.pt", hash_s: 16.8 }), "Text cache: built but not saved (check server logs); rebuilt on the next start");
assert.equal(c.textCacheText({ status: "hit", rehashed: true, path: "/c/t.pt", hash_s: 16.84 }), "Text cache: hit (model loaded without understanding weights) · checkpoint re-hashed in 16.8 s");
assert.equal(c.textCacheText({ status: "built", rehashed: true, path: "/c/t.pt", hash_s: null }), "Text cache: built and saved · checkpoint re-hashed");
assert.equal(c.textCacheText({ status: "new-thing" }), "Text cache: new-thing");
assert.equal(c.textCacheText(null), "");
assert.equal(c.textCacheText(undefined), "");

console.log("gpu_chart ok");
