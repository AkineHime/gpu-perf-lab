"use strict";

// ---------------------------------------------------------------- setup
const $ = (s) => document.querySelector(s);
const KERNELS = [
  { id: "cublas", short: "cuBLAS", color: "#e9eef8" },
  { id: "v0_naive", short: "v0 naive", color: "#ff4d6d" },
  { id: "v1_coalesced", short: "v1 coalesced", color: "#ff8a3d" },
  { id: "v2_smem", short: "v2 smem", color: "#ffd23f" },
  { id: "v3_tile1d", short: "v3 tile 1D", color: "#22d3ee" },
  { id: "v4_tile2d", short: "v4 tile 2D", color: "#a78bfa" },
  { id: "v5_vectorized", short: "v5 float4", color: "#76b900" },
];
const K = Object.fromEntries(KERNELS.map((k) => [k.id, k]));
const ALL_SIZES = [256, 512, 1024, 2048, 4096];
const TELE = [
  { key: "gpu_util", label: "GPU load", unit: "%", max: () => 100, fmt: (v) => v.toFixed(0) },
  { key: "sm_clock", label: "SM clock", unit: "MHz", max: () => state.gpu.max_sm_clock || 2000, fmt: (v) => v.toFixed(0) },
  { key: "mem_clock", label: "Mem clock", unit: "MHz", max: () => state.gpu.max_mem_clock || 8000, fmt: (v) => v.toFixed(0) },
  { key: "power_w", label: "Power", unit: "W", max: () => state.gpu.power_limit_w || 100, fmt: (v) => v.toFixed(1) },
  { key: "temp_c", label: "Temp", unit: "°C", max: () => 100, fmt: (v) => v.toFixed(0) },
  { key: "mem_used_gb", label: "VRAM", unit: "GB", max: () => state.gpu.mem_total_gb || 8, fmt: (v) => v.toFixed(2) },
];
const HISTORY = 160;

const state = {
  gpu: {}, device: null, probes: {}, results: {}, // results[kernel][n] = record
  history: Object.fromEntries(TELE.map((t) => [t.key, []])),
  shown: Object.fromEntries(TELE.map((t) => [t.key, 0])),
  sizes: loadSizes(), selectedN: null, userPickedN: false, running: false,
  points: {},       // animated roofline positions per kernel
  sweepReveal: 0,   // 0..1 line-drawing progress
  hover: null,
};

function loadSizes() {
  try { const s = JSON.parse(localStorage.getItem("sizes")); if (Array.isArray(s) && s.length) return s; } catch {}
  return [...ALL_SIZES];
}
function saveSizes() { try { localStorage.setItem("sizes", JSON.stringify(state.sizes)); } catch {} }

const fmtNum = (v, d = 0) => v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const fmtFlops = (g) => (g >= 1000 ? `${fmtNum(g / 1000, 2)} TFLOPS` : `${fmtNum(g, 0)} GFLOPS`);

function fit(canvas) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

// ---------------------------------------------------------------- telemetry cards
function buildTelemetry() {
  $("#telemetry").innerHTML = TELE.map((t) => `
    <div class="tele" id="t-${t.key}">
      <div class="label">${t.label}</div>
      <div class="max"></div>
      <div class="value"><span class="v">–</span><span class="unit">${t.unit}</span></div>
      <canvas></canvas>
    </div>`).join("");
}

function pushTelemetry(s) {
  for (const t of TELE) {
    const v = s[t.key];
    if (v == null) continue;
    const h = state.history[t.key];
    h.push(v);
    if (h.length > HISTORY) h.shift();
  }
}

function drawTelemetry(dt) {
  for (const t of TELE) {
    const el = document.getElementById(`t-${t.key}`);
    const h = state.history[t.key];
    if (!h.length) continue;
    const target = h[h.length - 1];
    state.shown[t.key] += (target - state.shown[t.key]) * Math.min(1, dt * 8);
    el.querySelector(".v").textContent = t.fmt(state.shown[t.key]);
    const max = Math.max(t.max(), ...h);
    el.querySelector(".max").textContent = `max ${t.fmt(max)}`;

    const { ctx, w, h: H } = fit(el.querySelector("canvas"));
    const hot = target / max > 0.85;
    const color = t.key === "temp_c" && target > 80 ? "#ff4d6d" : hot ? "#b8f236" : "#76b900";
    const step = w / (HISTORY - 1);
    const x0 = w - (h.length - 1) * step;
    const y = (v) => H - 3 - (v / max) * (H - 8);
    ctx.beginPath();
    h.forEach((v, i) => (i ? ctx.lineTo(x0 + i * step, y(v)) : ctx.moveTo(x0, y(v))));
    const line = new Path2D();
    h.forEach((v, i) => (i ? line.lineTo(x0 + i * step, y(v)) : line.moveTo(x0, y(v))));
    ctx.lineTo(w, H); ctx.lineTo(x0, H); ctx.closePath();
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, hexA(color, 0.35)); g.addColorStop(1, hexA(color, 0));
    ctx.fillStyle = g; ctx.fill();
    ctx.shadowColor = color; ctx.shadowBlur = 10;
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke(line);
    ctx.shadowBlur = 0;
  }
}

// ---------------------------------------------------------------- data
function resultsAt(n) {
  return KERNELS.map((k) => state.results[k.id]?.[n]).filter(Boolean);
}
function availableNs() {
  const s = new Set();
  for (const byN of Object.values(state.results)) for (const n of Object.keys(byN)) s.add(Number(n));
  return [...s].sort((a, b) => a - b);
}
function ceilings() {
  return { bw: state.probes.dram_bandwidth?.gbps || 150, peak: state.probes.fp32_peak?.gflops || 10000 };
}

function ingest(ev) {
  if (ev.type === "device") state.device = ev;
  else if (ev.type === "probe") state.probes[ev.name] = ev;
  else if (ev.type === "result") {
    (state.results[ev.kernel] ||= {})[ev.n] = { ...ev, arrived: performance.now() };
    if (!state.userPickedN) state.selectedN = ev.n;
  }
}

function resetResults() {
  state.results = {}; state.probes = {}; state.points = {}; state.sweepReveal = 0;
  state.userPickedN = false; state.selectedN = null;
}

// ---------------------------------------------------------------- roofline
const RL = { xmin: -1, xmax: 2.2, ymin: 1 };  // log10 ranges (x: FLOP/B, y: GFLOPS)

function roofGeometry(w, h) {
  const { peak } = ceilings();
  const pad = { l: 58, r: 18, t: 18, b: 42 };
  const ymax = Math.log10(peak * 1.8);
  const X = (fi) => pad.l + ((Math.log10(fi) - RL.xmin) / (RL.xmax - RL.xmin)) * (w - pad.l - pad.r);
  const Y = (gf) => h - pad.b - ((Math.log10(gf) - RL.ymin) / (ymax - RL.ymin)) * (h - pad.t - pad.b);
  return { pad, ymax, X, Y };
}

function drawRoofline(dt, now) {
  const canvas = $("#roofline");
  const { ctx, w, h } = fit(canvas);
  const { pad, ymax, X, Y } = roofGeometry(w, h);
  const { bw, peak } = ceilings();

  // grid
  ctx.font = "11px JetBrains Mono, monospace";
  ctx.fillStyle = "#7d8aa5";
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let e = Math.ceil(RL.xmin); e <= Math.floor(RL.xmax); e++) {
    for (let m = 1; m < 10; m++) {
      const v = m * 10 ** e;
      if (Math.log10(v) > RL.xmax) break;
      ctx.globalAlpha = m === 1 ? 1 : 0.4;
      ctx.beginPath(); ctx.moveTo(X(v), pad.t); ctx.lineTo(X(v), h - pad.b); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    ctx.textAlign = "center";
    ctx.fillText(e < 0 ? (10 ** e).toFixed(-e) : String(10 ** e), X(10 ** e), h - pad.b + 16);
  }
  for (let e = RL.ymin; e <= Math.floor(ymax); e++) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(10 ** e)); ctx.lineTo(w - pad.r, Y(10 ** e)); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(10 ** e >= 1000 ? `${10 ** e / 1000}k` : String(10 ** e), pad.l - 8, Y(10 ** e) + 4);
  }
  ctx.textAlign = "center";
  ctx.fillText("global-load intensity (FLOP / byte)", (pad.l + w - pad.r) / 2, h - 6);
  ctx.save(); ctx.translate(14, (pad.t + h - pad.b) / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("GFLOPS", 0, 0); ctx.restore();

  // roof
  const ridge = peak / bw;
  const xs = [10 ** RL.xmin, ridge, 10 ** RL.xmax];
  const roofPts = xs.map((x) => [X(x), Y(Math.min(peak, bw * x))]);
  const area = ctx.createLinearGradient(0, Y(peak), 0, h - pad.b);
  area.addColorStop(0, "rgba(118,185,0,0.16)"); area.addColorStop(1, "rgba(118,185,0,0)");
  ctx.beginPath();
  roofPts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.lineTo(X(10 ** RL.xmax), h - pad.b); ctx.lineTo(X(10 ** RL.xmin), h - pad.b); ctx.closePath();
  ctx.fillStyle = area; ctx.fill();
  ctx.beginPath();
  roofPts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.shadowColor = "#76b900"; ctx.shadowBlur = 14;
  ctx.strokeStyle = "#76b900"; ctx.lineWidth = 2.2; ctx.stroke();
  ctx.shadowBlur = 0;

  // roof labels
  ctx.fillStyle = "#b8f236"; ctx.font = "600 11px JetBrains Mono, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`FP32 peak ${fmtFlops(peak)} (measured)`, w - pad.r - 4, Y(peak) - 8);
  const sx = X(1.2), sy = Y(bw * 1.2);  // clear of the v0/v1 points at 0.25 FLOP/B
  const angle = Math.atan2(Y(bw * 12) - sy, X(12) - sx);
  ctx.save(); ctx.translate(sx, sy - 8); ctx.rotate(angle); ctx.textAlign = "left";
  ctx.fillText(`DRAM ${fmtNum(bw, 0)} GB/s`, 0, 0); ctx.restore();
  ctx.setLineDash([3, 4]); ctx.strokeStyle = "rgba(184,242,54,0.35)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(ridge), Y(peak)); ctx.lineTo(X(ridge), h - pad.b); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#7d8aa5"; ctx.font = "10px JetBrains Mono, monospace"; ctx.textAlign = "right";
  ctx.fillText(`ridge ${ridge.toFixed(0)} FLOP/B`, X(ridge) - 5, h - pad.b - 6);

  const n = state.selectedN;
  if (n == null) {
    ctx.fillStyle = "#7d8aa5"; ctx.font = "13px Inter, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Run the benchmark to launch the kernels onto the roofline", (pad.l + w - pad.r) / 2, (pad.t + h - pad.b) / 2);
    return;
  }

  // cuBLAS line
  const ref = state.results.cublas?.[n];
  if (ref) {
    ctx.setLineDash([6, 6]); ctx.strokeStyle = "rgba(233,238,248,0.55)"; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(ref.gflops)); ctx.lineTo(w - pad.r, Y(ref.gflops)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#e9eef8"; ctx.font = "600 11px JetBrains Mono, monospace"; ctx.textAlign = "left";
    ctx.fillText(`cuBLAS ${fmtFlops(ref.gflops)}`, pad.l + 8, Y(ref.gflops) - 7);
  }

  // kernels: glide to their measured spot
  const placed = [];
  for (const r of resultsAt(n)) {
    if (r.kernel === "cublas" || r.load_intensity == null) continue;
    const k = K[r.kernel];
    const tx = X(r.load_intensity), ty = Y(Math.max(r.gflops, 10 ** RL.ymin));
    const p = (state.points[r.kernel] ||= { x: X(10 ** RL.xmin), y: h - pad.b, trail: [] });
    p.x += (tx - p.x) * Math.min(1, dt * 4);
    p.y += (ty - p.y) * Math.min(1, dt * 4);
    p.trail.push([p.x, p.y]);
    if (p.trail.length > 18) p.trail.shift();

    for (let i = 1; i < p.trail.length; i++) {
      ctx.strokeStyle = hexA(k.color, (i / p.trail.length) * 0.5);
      ctx.lineWidth = (i / p.trail.length) * 3;
      ctx.beginPath(); ctx.moveTo(...p.trail[i - 1]); ctx.lineTo(...p.trail[i]); ctx.stroke();
    }
    const age = (now - r.arrived) / 1000;
    if (age < 1.6) {  // arrival shockwave
      ctx.strokeStyle = hexA(k.color, 1 - age / 1.6);
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.x, p.y, 6 + age * 26, 0, Math.PI * 2); ctx.stroke();
    }
    const pulse = 1 + 0.15 * Math.sin(now / 300 + r.load_intensity);
    ctx.shadowColor = k.color; ctx.shadowBlur = 18;
    ctx.fillStyle = r.correct === false ? "#ff4d6d" : k.color;
    ctx.beginPath(); ctx.arc(p.x, p.y, 6 * pulse, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#05070d";
    ctx.beginPath(); ctx.arc(p.x, p.y, 2.2, 0, Math.PI * 2); ctx.fill();

    // label, nudged so labels at the same intensity don't collide
    let ly = p.y + 4;
    for (const q of placed) if (Math.abs(q.x - p.x) < 60 && Math.abs(q.y - ly) < 14) ly = q.y + (ly > q.y ? 14 : -14);
    placed.push({ x: p.x, y: ly });
    ctx.fillStyle = k.color; ctx.font = "600 11px JetBrains Mono, monospace"; ctx.textAlign = "left";
    ctx.fillText(k.short, p.x + 12, ly);
    r._screen = [p.x, p.y];
  }
}

function roofTooltip(e) {
  const rect = $("#roofline").getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bestD = 16;
  for (const r of resultsAt(state.selectedN)) {
    if (!r._screen) continue;
    const d = Math.hypot(r._screen[0] - mx, r._screen[1] - my);
    if (d < bestD) { best = r; bestD = d; }
  }
  const tip = $("#tip");
  if (!best) { tip.classList.add("hidden"); return; }
  const { bw } = ceilings();
  const aboveDram = best.gflops > bw * best.load_intensity;
  tip.innerHTML = `<b style="color:${K[best.kernel].color}">${best.name}</b><br>
    ${best.idea}<br>
    <b>${fmtFlops(best.gflops)}</b> · ${best.pct_cublas}% of cuBLAS · ${best.ms.toFixed(3)} ms<br>
    intensity ${best.load_intensity} FLOP/B · err ${best.max_rel_err}
    ${aboveDram ? "<br><span style='color:#ffd23f'>Above the DRAM roof: most of its loads are served by L1/L2 cache.</span>" : ""}`;
  tip.classList.remove("hidden");
  const tx = Math.min(mx + 14, rect.width - tip.offsetWidth - 6);
  tip.style.left = `${tx}px`;
  tip.style.top = `${Math.max(6, my - tip.offsetHeight - 10)}px`;
}

// ---------------------------------------------------------------- size sweep
function drawSweep(dt) {
  const { ctx, w, h } = fit($("#sweep"));
  const ns = availableNs();
  if (!ns.length) return;
  state.sweepReveal = Math.min(1, state.sweepReveal + dt * 0.8);
  const pad = { l: 52, r: 14, t: 12, b: 30 };
  const maxG = Math.max(...Object.values(state.results).flatMap((byN) => Object.values(byN).map((r) => r.gflops))) * 1.1;
  const lo = Math.log2(ns[0]), hi = Math.log2(ns[ns.length - 1]);
  const X = (n) => pad.l + (hi === lo ? 0.5 : (Math.log2(n) - lo) / (hi - lo)) * (w - pad.l - pad.r);
  const Y = (g) => h - pad.b - (g / maxG) * (h - pad.t - pad.b);

  ctx.font = "11px JetBrains Mono, monospace"; ctx.fillStyle = "#7d8aa5";
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  const step = maxG > 4000 ? 1000 : maxG > 1500 ? 500 : 200;
  for (let g = 0; g <= maxG; g += step) {
    ctx.beginPath(); ctx.moveTo(pad.l, Y(g)); ctx.lineTo(w - pad.r, Y(g)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(g >= 1000 ? `${g / 1000}k` : g, pad.l - 8, Y(g) + 4);
  }
  ctx.textAlign = "center";
  for (const n of ns) ctx.fillText(n, X(n), h - pad.b + 16);

  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, pad.l + (w - pad.l) * state.sweepReveal, h); ctx.clip();
  for (const k of KERNELS) {
    const pts = ns.map((n) => state.results[k.id]?.[n]).filter(Boolean);
    if (!pts.length) continue;
    ctx.strokeStyle = k.color; ctx.lineWidth = k.id === "cublas" ? 1.5 : 2.2;
    ctx.setLineDash(k.id === "cublas" ? [6, 5] : []);
    ctx.shadowColor = k.color; ctx.shadowBlur = k.id === "cublas" ? 0 : 10;
    ctx.beginPath();
    pts.forEach((r, i) => (i ? ctx.lineTo(X(r.n), Y(r.gflops)) : ctx.moveTo(X(r.n), Y(r.gflops))));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = k.color;
    for (const r of pts) { ctx.beginPath(); ctx.arc(X(r.n), Y(r.gflops), 3, 0, Math.PI * 2); ctx.fill(); }
    ctx.shadowBlur = 0;
  }
  ctx.restore();
}

// ---------------------------------------------------------------- DOM panels
function renderNChips() {
  const ns = availableNs();
  $("#n-chips").innerHTML = ns.map((n) =>
    `<button class="chip ${n === state.selectedN ? "on" : ""}" data-n="${n}">${n}³</button>`).join("");
}

function renderRace() {
  const n = state.selectedN;
  const rows = resultsAt(n);
  $("#race-hint").textContent = n ? `GFLOPS at ${n}×${n}×${n}, vs. cuBLAS` : "GFLOPS at the selected size, vs. cuBLAS";
  if (!rows.length) { $("#race").innerHTML = '<div class="empty">No results yet</div>'; return; }
  const max = Math.max(...rows.map((r) => r.gflops));
  const existing = new Map([...$("#race").querySelectorAll(".lane")].map((el) => [el.dataset.k, el]));
  if (existing.size !== rows.length || rows.some((r) => !existing.has(r.kernel))) {
    $("#race").innerHTML = rows.map((r) => `
      <div class="lane" data-k="${r.kernel}">
        <div class="row"><span class="name" style="color:${K[r.kernel].color}">${r.name}</span><span class="num"></span></div>
        <div class="track"><div class="bar" style="background:linear-gradient(90deg, ${hexA(K[r.kernel].color, 0.35)}, ${K[r.kernel].color}); box-shadow: 0 0 14px ${hexA(K[r.kernel].color, 0.6)}"></div></div>
      </div>`).join("");
  }
  requestAnimationFrame(() => {
    for (const r of rows) {
      const lane = $("#race").querySelector(`.lane[data-k="${r.kernel}"]`);
      lane.querySelector(".bar").style.width = `${(r.gflops / max) * 100}%`;
      lane.querySelector(".num").innerHTML = r.correct === false
        ? `<span class="bad">✗ wrong result</span>`
        : `<b>${fmtNum(r.gflops, 0)}</b> GFLOPS · ${r.pct_cublas}%`;
    }
  });
}

function renderLegend() {
  $("#legend").innerHTML = KERNELS.filter((k) => state.results[k.id])
    .map((k) => `<span style="--c:${k.color}">${k.short}</span>`).join("");
}

function renderCeilings() {
  const { bw, peak } = ceilings();
  const have = state.probes.fp32_peak && state.probes.dram_bandwidth;
  const best = Math.max(0, ...Object.entries(state.results).filter(([k]) => k !== "cublas")
    .flatMap(([, byN]) => Object.values(byN).map((r) => r.gflops)));
  const d = state.device;
  $("#ceilings").innerHTML = `
    <div class="ceil"><div class="l">FP32 peak</div><div class="v">${have ? fmtFlops(peak) : "–"}</div><div class="d">independent FMA chains</div></div>
    <div class="ceil"><div class="l">DRAM bandwidth</div><div class="v">${have ? fmtNum(bw, 0) + " GB/s" : "–"}</div><div class="d">float4 copy, 2 × 256 MB</div></div>
    <div class="ceil"><div class="l">Ridge point</div><div class="v">${have ? (peak / bw).toFixed(0) : "–"}</div><div class="d">FLOP/byte to become compute-bound</div></div>
    <div class="ceil"><div class="l">Best kernel</div><div class="v">${best && have ? ((100 * best) / peak).toFixed(0) + "%" : "–"}</div><div class="d">of measured FP32 peak</div></div>
    ${d ? `<div class="d" style="grid-column:1/-1;color:var(--muted);font-family:var(--mono);font-size:11px">${d.name} · sm_${d.cc.replace(".", "")} · ${d.sms} SMs · ${d.l2_kb} KB L2 · CUDA ${Math.floor(d.cuda_runtime / 1000)}.${(d.cuda_runtime % 1000) / 10}</div>` : ""}`;
}

function renderAll() { renderNChips(); renderRace(); renderLegend(); renderCeilings(); }

function log(line) {
  const el = $("#log");
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

function setRunning(on) {
  state.running = on;
  $("#run").classList.toggle("busy", on);
  $("#run").textContent = on ? "● Running…" : "▶ Run benchmark";
  $("#stop").classList.toggle("hidden", !on);
}

function renderSizeChips() {
  $("#size-chips").innerHTML = ALL_SIZES.map((n) =>
    `<button class="chip ${state.sizes.includes(n) ? "on" : ""}" data-size="${n}">${n}</button>`).join("");
}

// ---------------------------------------------------------------- events
$("#size-chips").addEventListener("click", (e) => {
  const n = Number(e.target.dataset.size);
  if (!n) return;
  state.sizes = state.sizes.includes(n) ? state.sizes.filter((s) => s !== n) : [...state.sizes, n].sort((a, b) => a - b);
  if (!state.sizes.length) state.sizes = [n];
  saveSizes(); renderSizeChips();
});
$("#n-chips").addEventListener("click", (e) => {
  const n = Number(e.target.dataset.n);
  if (!n) return;
  state.selectedN = n; state.userPickedN = true; state.points = {};
  renderAll();
});
$("#run").addEventListener("click", async () => {
  if (state.running) return;
  const res = await fetch("/api/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sizes: state.sizes }),
  });
  if (!res.ok) log(`✗ ${(await res.json()).detail}`);
});
$("#stop").addEventListener("click", () => fetch("/api/stop", { method: "POST" }));
$("#roofline").addEventListener("mousemove", roofTooltip);
$("#roofline").addEventListener("mouseleave", () => $("#tip").classList.add("hidden"));

function connect() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    switch (ev.type) {
      case "telemetry": pushTelemetry(ev); break;
      case "run_started": resetResults(); setRunning(true); $("#log").textContent = ""; log(`$ gemm_bench ${ev.cmd}`); renderAll(); break;
      case "run_finished":
        setRunning(false);
        if (ev.code === 0) log("✓ done");
        else { log(`✗ exited with code ${ev.code}${ev.hint ? ": " + ev.hint : ""}`); loadSaved(); }
        break;
      case "log": log(ev.line); break;
      case "bench_error": log(`✗ ${ev.message}`); break;
      default:
        if (ev.type.startsWith("bench_")) { ingest({ ...ev, type: ev.type.slice(6) }); renderAll(); }
    }
  };
}

// ---------------------------------------------------------------- main loop
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  drawTelemetry(dt);
  drawRoofline(dt, now);
  drawSweep(dt);
  requestAnimationFrame(frame);
}

async function loadSaved() {
  const prev = await (await fetch("/api/results")).json();
  resetResults();
  const t = performance.now() - 5000;  // no arrival shockwaves for old results
  prev.forEach((ev) => ingest(ev));
  for (const byN of Object.values(state.results)) for (const r of Object.values(byN)) r.arrived = t;
  if (prev.length) log(`loaded ${prev.filter((e) => e.type === "result").length} results from the last successful run`);
  renderAll();
}

(async () => {
  buildTelemetry();
  renderSizeChips();
  const status = await (await fetch("/api/status")).json();
  state.gpu = status.gpu || {};
  $("#gpu-name").textContent = status.gpu?.name
    ? `${status.gpu.name} · driver ${status.gpu.driver}` : status.telemetry_error || "no GPU telemetry";
  setRunning(status.running);
  if (!status.bench_built) log("gemm_bench is not built yet: run build.bat");
  await loadSaved();
  connect();
  requestAnimationFrame(frame);
})();
