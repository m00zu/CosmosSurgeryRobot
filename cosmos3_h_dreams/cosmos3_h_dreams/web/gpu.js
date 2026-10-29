// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Live GPU memory page. Polls /api/gpu?gpu=<index>&since=<last t> every second, merges the
// samples and marks, and redraws the chart, tiles and table. Chart helpers live in gpu_chart.js.

(() => {
  const $ = id => document.getElementById(id);
  const POLL_MS = 1000;
  const KEEP_S = 1800 + 60;                  // the longest span plus a little
  const svg = $("chart");
  const tooltip = $("tooltip");
  const hostSvg = $("host-chart");
  const HEALTH_EVERY = 5;                    // polls between /healthz reads (text cache line)
  let polls = 0;
  const select = $("gpu-select");
  let gpu = null;                            // selected GPU index; null until the first answer names the device
  let samples = [];
  let marks = [];
  let markKeys = new Set();
  let last = null;                           // the latest /api/gpu body
  let since = null;                          // t of the newest sample held
  let spanS = 1800;
  let fit = false;                           // y-scale: false = 0..capacity, true = zoomed to the visible data
  let serverId = null;                       // csv name + device baseline: changes when the server restarts
  try { fit = localStorage.getItem("gpu.scale") === "fit"; } catch (_) { /* storage blocked: default scale */ }
  let generation = 0;                        // bumps on a GPU change so a stale answer is dropped
  let timer = null;

  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(poll, ms);
  }

  function reset() {
    samples = [];
    since = null;
    generation++;
  }

  // A restarted server has a new CSV log and baseline (samples use wall time, so t keeps rising): when that
  // identity changes, or the server's clock is behind our newest sample, drop the old process's data.
  function restarted(body) {
    const baseline = body.baseline[String(body.device)];
    const id = baseline === undefined ? null : `${body.csv}|${baseline}`;
    const changed = (since !== null && body.now < since) || (id !== null && serverId !== null && id !== serverId);
    if (id !== null) serverId = id;
    if (!changed) return false;
    reset();
    marks = [];
    markKeys = new Set();
    serverId = id;
    return true;
  }

  function merge(body) {
    for (const mark of body.marks) {
      const key = `${mark.t}|${mark.kind}|${mark.label}`;
      if (!markKeys.has(key)) { markKeys.add(key); marks.push(mark); }
    }
    const fresh = body.samples.filter(s => s.gpu === gpu && (since === null || s.t > since));
    samples = samples.concat(fresh).filter(s => s.t >= body.now - KEEP_S);
    marks = pruneMarks(marks, body.now - KEEP_S);   // keeps the start of an attempt or switch still in view
    markKeys = new Set(marks.map(mark => `${mark.t}|${mark.kind}|${mark.label}`));
    if (samples.length) since = samples[samples.length - 1].t;
  }

  function fillSelect(body) {
    const wanted = body.gpus.map(g => String(g.index)).join(",");
    if (select.dataset.gpus === wanted) return;
    select.dataset.gpus = wanted;
    select.replaceChildren(...body.gpus.map(g => {
      const option = document.createElement("option");
      option.value = String(g.index);
      option.textContent = `${g.index}: ${g.name}${g.index === body.device ? " (server)" : ""}`;
      return option;
    }));
    select.value = String(gpu);
  }

  function tile(label, value, sub) {
    const box = document.createElement("div");
    box.className = "tile";
    const name = document.createElement("div");
    name.className = "tile-label";
    name.textContent = label;
    const number = document.createElement("div");
    number.className = "tile-value";
    number.textContent = value;
    const note = document.createElement("div");
    note.className = "tile-sub";
    note.textContent = sub || "";
    box.append(name, number, note);
    return box;
  }

  function renderTiles() {
    const now = samples[samples.length - 1] || {};
    const total = now.total_mib ?? (last.gpus.find(g => g.index === gpu) || {}).total_mib;
    const baseline = last.baseline[String(gpu)];
    const peak = last.peak[String(gpu)];
    const delta = now.used_mib !== undefined && baseline !== undefined ? now.used_mib - baseline : null;
    const signed = mib => `${mib >= 0 ? "+" : "−"}${formatGiB(Math.abs(mib))}`;
    const host = last.host || {};
    $("tiles").replaceChildren(
      tile("Used now", formatGiB(now.used_mib), total ? `of ${formatGiB(total)}` : ""),
      tile("Peak since start", formatGiB(peak), peak !== undefined && baseline !== undefined ? `${signed(peak - baseline)} vs baseline` : ""),
      tile("Baseline", formatGiB(baseline), delta === null ? "first sample" : `now ${signed(delta)}`),
      tile("PyTorch allocated", formatGiB(now.torch_allocated_mib), now.torch_reserved_mib == null ? "server GPU only" : `reserved ${formatGiB(now.torch_reserved_mib)}`),
      tile("Utilization", now.util_pct == null ? "—" : `${Math.round(now.util_pct)} %`, ""),
      tile("Temperature", now.temp_c == null ? "—" : `${Math.round(now.temp_c)} °C`, ""),
      tile("Host RAM (PSS)", formatGiB(host.pss_mib), host.peak_mib == null ? "server process tree" : `peak ${formatGiB(host.peak_mib)}`),
    );
  }

  function renderTable() {
    if ($("table-wrap").hidden) return;
    const gib = mib => (mib == null ? "—" : (mib / 1024).toFixed(2));
    $("table-body").replaceChildren(...samples.slice(-60).reverse().map(s => {
      const tr = document.createElement("tr");
      for (const text of [clockText(s.t), gib(s.used_mib), gib(s.torch_reserved_mib), gib(s.torch_allocated_mib), s.util_pct == null ? "—" : String(Math.round(s.util_pct)), gib(s.host_pss_mib)]) {
        const td = document.createElement("td");
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }));
  }

  function render() {
    if (!last) return;
    const info = last.gpus.find(g => g.index === gpu) || {};
    renderTiles();
    renderChart(svg, {
      samples: windowSamples(samples, gpu, last.now, spanS),
      t0: last.now - spanS,
      t1: last.now,
      totalMib: info.total_mib ?? (samples[samples.length - 1] || {}).total_mib,
      baselineMib: last.baseline[String(gpu)],
      marks,
      fit,
    }, { tooltip, live: $("chart-live"), height: 360 });
    // Host PSS rides on the server device's samples: other GPUs have none to draw.
    const onDevice = gpu === last.device;
    $("host-wrap").hidden = !onDevice;
    $("host-note").hidden = onDevice;
    $("host-note").textContent = onDevice ? "" : `Host RAM is sampled with the server GPU; select GPU ${last.device} to see it.`;
    if (onDevice) {
      renderChart(hostSvg, {
        samples: windowSamples(samples, gpu, last.now, spanS),
        t0: last.now - spanS,
        t1: last.now,
        marks,
        series: GPU_HOST_SERIES,
        details: false,
      }, { tooltip: $("host-tooltip"), live: $("host-live"), height: 220 });
    }
    renderTable();
    $("legend-used").textContent = last.nvml ? "used (NVML)" : "used (PyTorch reserved)";
    const source = last.nvml ? "NVML" : "PyTorch counters only (NVML unavailable)";
    $("csv").textContent = last.csv_error
      ? `CSV log stopped: ${last.csv_error}`
      : last.csv ? `CSV log: ${last.csv} · ${source}` : `No CSV log · ${source}`;
  }

  async function pollHealth() {
    try {
      const response = await fetch("/healthz", { cache: "no-store" });
      if (response.ok) $("text-cache").textContent = textCacheText((await response.json()).text_cache);
    } catch (_) { /* the next poll retries */ }
  }

  async function poll() {
    const asked = generation;
    if (polls++ % HEALTH_EVERY === 0) pollHealth();
    try {
      // Before the first answer the server's device is unknown: ask for GPU -1 (no samples) to learn it.
      const params = new URLSearchParams({ gpu: String(gpu ?? -1) });
      if (since !== null) params.set("since", String(since));
      const response = await fetch(`/api/gpu?${params}`, { cache: "no-store" });
      if (response.status === 404) {
        $("status").textContent = "GPU monitor is off (start the server with --gpu-monitor).";
        if (asked === generation) schedule(10000);   // the server may be restarted with the monitor on
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const body = await response.json();
      if (asked !== generation) return;
      if (restarted(body)) {
        $("status").textContent = "Server restarted · reloading its history…";
        schedule(0);
        return;
      }
      last = body;
      if (gpu === null) {
        gpu = body.device;
        fillSelect(body);
        generation++;
        schedule(0);                         // fetch the device's history right away
        return;
      }
      fillSelect(body);
      merge(body);
      $("status").textContent = samples.length ? "" : "Waiting for the first sample…";
      render();
    } catch (error) {
      $("status").textContent = "Connection lost · reconnecting…";
    }
    if (asked === generation) schedule(POLL_MS);
  }

  for (const button of document.querySelectorAll("[data-span]")) {
    button.addEventListener("click", () => {
      spanS = Number(button.dataset.span);
      for (const other of document.querySelectorAll("[data-span]")) other.setAttribute("aria-pressed", String(other === button));
      render();
    });
  }
  function showScale() {
    for (const button of document.querySelectorAll("[data-scale]")) {
      button.setAttribute("aria-pressed", String((button.dataset.scale === "fit") === fit));
    }
  }
  for (const button of document.querySelectorAll("[data-scale]")) {
    button.addEventListener("click", () => {
      fit = button.dataset.scale === "fit";
      try { localStorage.setItem("gpu.scale", fit ? "fit" : "full"); } catch (_) { /* not remembered */ }
      showScale();
      render();
    });
  }
  showScale();
  select.addEventListener("change", () => {
    gpu = Number(select.value);
    reset();
    render();
    schedule(0);
  });
  $("table-toggle").addEventListener("click", () => {
    const wrap = $("table-wrap");
    wrap.hidden = !wrap.hidden;
    $("table-toggle").textContent = wrap.hidden ? "Show table" : "Hide table";
    $("table-toggle").setAttribute("aria-expanded", String(!wrap.hidden));
    renderTable();
  });
  window.addEventListener("resize", render);
  poll();
})();
