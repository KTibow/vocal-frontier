import * as E from "./src/engine.js";
import { drawBoard, setScene, EXAMPLE, BESIDE, LEGEND, color } from "./chart.js";
import { REPO, DEADLINE } from "./config.js";

const $ = (id) => document.getElementById(id);
const Q = new URLSearchParams(location.search);
const OFFICIAL = Q.get("device");                 // ?device=adreno turns on the official protocol: ABBA rounds, and the file names the device
const fmt = (x, d = 2) => (x == null ? "–" : x.toFixed(d));
const songTime = (xrt) => { const s = 210 / xrt; return s < 90 ? `${s | 0} s` : s < 5400 ? `${(s / 60).toFixed(s < 600 ? 1 : 0)} min` : `${(s / 3600).toFixed(1)} h`; };
const store = { get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} }, del(k) { try { localStorage.removeItem(k); } catch {} } };

let MODELS = {}, REF = "htdemucs", backend = "webgpu", adapter = null, clip = null, EXPECT = {};
const clipOnce = async () => (clip ??= await E.loadClip("clip"));

// ---- board ---------------------------------------------------------------------------------------------------------
async function board() {
  let b;
  try { b = await (await fetch("data/board.json")).json(); } catch { $("board-status").textContent = "No results yet."; return; }
  if (!b.points.length) {
    $("board-status").textContent = "Starting frontier is being measured on the six devices.";
    $("legend").hidden = true;
    $("chart").innerHTML = `<p class="muted" style="padding:48px 16px;text-align:center">No official timings yet. You can already test models on this device below.</p>`;
    document.querySelector(".table-wrap").hidden = true;
    return;
  }
  await addPreview(b);
  const prov = b.points.some((p) => p.provisional);
  const measured = b.devices.filter((d) => b.points.some((p) => p.devices[d]?.xrt));
  $("board-status").textContent = `${prov ? `Provisional: measured on ${measured.length} of ${b.devices.length} devices so far (${measured.map((d) => DEVICE_NAMES[d]).join(", ")}). Scores will move as the rest land. ` : ""}Updated ${b.built}.`;
  $("legend").innerHTML = LEGEND.filter(([n]) => n !== "Contestants" || b.points.some((p) => !["baseline", "maintainers"].includes(p.owner))).map(([n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("") + (prov ? `<span><i class="hollow"></i>Provisional</span>` : "");
  const draw = () => drawBoard($("chart"), $("tip"), b, (p) => { const r = document.querySelector(`[data-row="${p.name}"]`); r?.scrollIntoView({ block: "center", behavior: "smooth" }); });
  draw();
  story(b);
  const a = b.area, pay = b.payouts, others = a.baseline ?? 0, ours = a.maintainers ?? 0, theirs = Object.entries(a).filter(([k]) => !["baseline", "maintainers"].includes(k));
  const added = theirs.reduce((s, [, v]) => s + v, 0);
  $("pot").innerHTML = [[`$${b.pot}`, "at most in the pot, USDC on Base"], [fmt(others, 1), "units covered by others' models"], [fmt(ours, 1), "units added by the maintainers' models"], [fmt(added, 1), "units added by contestants"], [`$${fmt(b.pot / (ours + added + 1), 0)}`, "for the next full unit added"]]
    .map(([v, k]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("") + `<p class="muted pot-note">One unit is one doubling of speed times one dB. Amounts are provisional until every device is measured.</p>`;
  const devs = b.devices.filter((d) => b.points.some((p) => p.devices[d]));   // a device nobody has run yet would be a column of dashes
  const contestants = b.points.some((p) => !["baseline", "maintainers"].includes(p.owner));
  const head = `<thead><tr><th>Entry</th><th class="num">Speed ×</th><th class="num">Quality, dB</th>${contestants ? `<th class="num">Units added</th>` : ""}${devs.map((d) => `<th class="num">${DEVICE_NAMES[d]}</th>`).join("")}${contestants ? `<th class="num">If it ended now</th>` : ""}<th>Weights license</th></tr></thead>`;
  // Units only mean money for contestants. Among the starting entries they depend on an arbitrary order, so those rows show a dash; the totals are in the stats above.
  const startsBoard = (p) => ["baseline", "maintainers"].includes(p.owner);
  const rows = [...b.points].sort((x, y) => startsBoard(x) - startsBoard(y) || (startsBoard(x) ? 0 : y.gain - x.gain) || y.q - x.q).map((p) => {
    const paid = pay.shares[p.owner] != null && p.owner !== "maintainers" ? `$${fmt(pay.shares[p.owner] * (a[p.owner] ? p.gain / a[p.owner] : 0), 2)}` : "–";
    return `<tr data-row="${p.name}"><td class="name"><span class="dot" style="background:${color(p.owner)}"></span>${esc(p.title)}<small>${esc(p.owner === "baseline" ? (p.kind === "tuned" ? "existing, tuned" : "existing model") : p.owner === "maintainers" ? "maintainers" : "@" + p.owner)}</small></td><td class="num">${fmt(p.s)}×</td><td class="num">${fmt(p.q)}</td>${contestants ? `<td class="num">${startsBoard(p) ? "–" : p.copy ? "copy" : fmt(p.gain, 1)}</td>` : ""}${devs.map((d) => { const v = p.devices[d]; return `<td class="num ${v && v.status !== "ok" ? "fail" : ""}" title="${esc(v?.status ?? "not measured")}">${v ? (v.status === "ok" ? fmt(v.ratio) : /^no .*recipe$/.test(v.status) ? "no recipe" : "failed") : "–"}</td>`; }).join("")}${contestants ? `<td class="num">${paid}</td>` : ""}<td>${esc(p.license?.weights ?? "–")}</td></tr>`;
  }).join("");
  $("board-table").innerHTML = head + `<tbody>${rows}</tbody>`;
  EXPECT = Object.fromEntries(b.points.map((p) => [p.name, p.clip_sdr]));
}
const DEVICE_NAMES = { native: "Native", browser: "Browser", vega: "Vega 8", adreno: "Adreno X1", valhall: "Mali Valhall", gen12lp: "Intel Gen12LP", "your-gpu": "Your GPU" };
// `vf preview` writes data/preview.json into a local build: your entry joins the board as it would look, with what it would earn. The live site has no such file.
const geo = (a) => Math.exp(a.reduce((s, v) => s + Math.log(v), 0) / a.length);
function unionArea(pts, x0, y0) {
  const p = pts.map((o) => [Math.log2(o.s), o.q]).filter(([x, y]) => x > x0 && y > y0).sort((m, n) => n[0] - m[0]);
  let best = 0, area = 0;
  p.forEach(([x, y], i) => { best = Math.max(best, y); area += (x - (i + 1 < p.length ? p[i + 1][0] : x0)) * (best - y0); });
  return area;
}
let PREVIEW = false;
// On a wide window the chart stays put and each part of the story changes what it shows. On a narrow one it's an ordinary page.
function story(b) {
  const others = b.points.filter((p) => p.q >= b.floor_q && p.owner !== "you"), x0 = Math.log2(b.origin_s);
  const units = unionArea([...others, EXAMPLE], x0, b.floor_q) - unionArea(others, x0, b.floor_q);
  $("example").textContent = `Say you enter something ${EXAMPLE.s}× as fast as HTDemucs, at ${EXAMPLE.q} dB. That covers ${fmt(units)} new units, worth about $${Math.round(b.pot * units / (newArea(b) + units))} on today's board.`;
  const wide = matchMedia(BESIDE), steps = [...document.querySelectorAll(".step")];
  const seen = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting && wide.matches) {
      steps.forEach((s) => s.classList.toggle("now", s === e.target));
      setScene($("chart"), e.target.dataset.scene);
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  steps.forEach((s) => seen.observe(s));
  wide.addEventListener("change", () => { if (!wide.matches) { setScene($("chart"), ""); steps.forEach((s) => s.classList.remove("now")); } });
}
// New area already on the board: the maintainers' and every contestant's. A new entry's share is split with all of it, as vf board does.
function newArea(b) { return Object.entries(b.area).reduce((t, [k, v]) => t + (k === "baseline" || k === "you" ? 0 : v), 0); }

async function addPreview(b) {
  const r = await fetch("data/preview.json").catch(() => null);
  if (!r?.ok) return;
  PREVIEW = true;
  const pv = await r.json(), devices = { native: { xrt: pv.native.xrt, ratio: pv.native.ratio, status: "ok" } };
  const here = { browser: store.get("vf-preview-browser"), "your-gpu": store.get("vf-preview-webgpu") };
  for (const d in here) if (here[d]) devices[d] = { ...here[d], status: "ok" };
  const cpu = ["native", "browser"].filter((d) => devices[d]).map((d) => devices[d].ratio), gpu = devices["your-gpu"] ? [devices["your-gpu"].ratio] : [];
  const s = geo([geo(cpu), ...(gpu.length ? [geo(gpu)] : [])]);
  const others = b.points.filter((p) => p.q >= b.floor_q), copy = others.some((e) => e.s * 1.05 >= s && e.q + .1 >= pv.q);
  const me = { name: pv.name, title: pv.title, kind: "contestant", owner: "you", s, q: pv.q, devices, provisional: true, license: pv.license, description: pv.description, copy, sets: {} };
  const x0 = Math.log2(b.origin_s), gain = copy || pv.q < b.floor_q ? 0 : unionArea([...others, me], x0, b.floor_q) - unionArea(others, x0, b.floor_q);
  me.gain = gain;
  const paid = b.pot * gain / (newArea(b) + gain || 1);
  b.area.you = gain; b.payouts.shares.you = paid; b.points.push(me);
  if (!others.some((e) => e.s >= s && e.q >= pv.q)) b.frontier.push(pv.name);
  if (devices["your-gpu"]) b.devices.push("your-gpu");
  const box = $("preview"); box.hidden = false;
  box.innerHTML = `<b>Preview of ${esc(pv.title)}, on this machine.</b> ${fmt(s)}× at ${fmt(pv.q)} dB${pv.q_estimated ? ` (rough: ${esc(pv.q_estimated)})` : ""}. ` +
    (others.some((e) => e.s >= s && e.q >= pv.q) ? "It adds no area: an earlier entry is both faster and cleaner." : copy ? "It's within 5% and 0.1 dB of an earlier entry, so it counts as a copy and earns nothing." : gain ? `It would add ${fmt(gain, 2)} units, worth $${fmt(paid, 0)} if nobody else entered.` : "It adds no area: earlier entries already cover this point.") +
    ` ${gpu.length && here.browser ? "" : `Run it below${gpu.length ? " on the CPU tab" : here.browser ? " on the GPU tab" : " on both tabs"} to fill in the rest. `}Official numbers come from the maintainers' devices.`;
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function weights() {
  const cell = (cls, label, f) => `<div class="${cls}" style="flex:${f}">${label}</div>`;
  $("weights").innerHTML = cell("c", "native", 25) + cell("c", "browser", 25) + ["Vega", "Adreno", "Valhall", "Gen12LP"].map((d) => cell("g", d, 12.5)).join("");
}

// ---- running. A big model can crash the tab; so each model runs from a queue kept in storage, with a reload between them.
// That also returns GPU memory between models. A reload that finds a model still marked as running knows it crashed the tab.
const QKEY = "vf-queue";
const STAGES = { download: "downloading", load: "loading", warmup: "warming up", run: "separating", stitch: "stitching" };
const newQueue = (names, be, opts = {}) => {
  const others = names.filter((n) => n !== REF);
  let order;
  if (opts.official) order = [REF, ...others, ...others.slice().reverse(), REF];      // ABBA, so drift hits everyone about equally
  else order = [REF, ...others];
  return { backend: be, order, i: 0, results: [], running: null, official: !!opts.official };
};

// ---- the screen stays on while anything runs: a phone that sleeps suspends the tab and ruins the timing.
let wakeWanted = false, wakeLock = null, wakeAsking = false;
async function holdScreen() {
  wakeWanted = true;
  if (wakeLock || wakeAsking || document.visibilityState !== "visible") return;
  wakeAsking = true;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    wakeLock.addEventListener("release", () => { wakeLock = null; });
    $("run-note").textContent = "The screen stays on until this finishes.";
  } catch { $("run-note").textContent = "This browser can't keep the screen on, so please keep it awake yourself."; }
  wakeAsking = false;
  if (!wakeWanted) releaseScreen();
}
function releaseScreen() {
  if (wakeWanted) $("run-note").textContent = "";
  wakeWanted = false; wakeLock?.release(); wakeLock = null;
}
document.addEventListener("visibilitychange", () => { if (wakeWanted) holdScreen(); });   // the browser drops the lock whenever the tab is hidden

// ---- the page reloads between models, so keep the progress bar at the same height on screen, however the page above it lays out.
const SCROLL_KEY = "vf-scroll";
const barTop = () => $("progress").getBoundingClientRect().top;
const saveScroll = () => { if (!$("progress").hidden) store.set(SCROLL_KEY, barTop()); };
function restoreScroll() {
  const top = store.get(SCROLL_KEY);
  if (top != null && !$("progress").hidden) window.scrollTo({ top: scrollY + barTop() - top, behavior: "instant" });
}

// A custom engine gets the clip at a random level and rotated by a random amount, so it can't recognize it and play back a stored answer. SDR doesn't care about either.
function disguised(c) {
  const g = .7 + .3 * Math.random(), n = c.mix[0].length, at = Math.floor(n * (.1 + .8 * Math.random()));
  const change = ([L, R]) => [L, R].map((x) => { const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = g * x[(i + at) % n]; return o; });
  return { ...c, mix: change(c.mix), vocals: change(c.vocals) };
}

// ---- battery: a device on battery can throttle, which moves the timing, and a long run can drain it. The API only exists in some browsers, so it's a help and never a requirement.
const LOW_BATTERY = .2;
async function minding(q) {
  const b = await navigator.getBattery?.().catch(() => null);
  if (!b) return;
  const note = $("power-note");
  if (!b.charging) q.onBattery = true;
  store.set(QKEY, q);
  note.hidden = b.charging;
  note.textContent = `On battery (${Math.round(b.level * 100)}%). It can slow down to save power, and running every model can drain it. Plug in for numbers you can trust.`;
  while (!b.charging && b.level < LOW_BATTERY) {
    setProgress(`Paused at ${Math.round(b.level * 100)}% battery. Plug in and it carries on.`, q.i / q.order.length);
    await new Promise((r) => { b.addEventListener("chargingchange", r, { once: true }); setTimeout(r, 30000); });
  }
  note.hidden = b.charging;
}

async function runNext(q) {
  holdScreen();
  await minding(q);
  if (q.running) {                                    // we crashed last time
    q.results.push({ model: q.running, error: "the tab crashed or ran out of memory" });
    q.running = null; q.i++;
  }
  showResults(q);
  if (q.i >= q.order.length) { store.del(QKEY); store.set("vf-last", q); finish(q); return; }
  const name = q.order[q.i], m = MODELS[name], r = m[q.backend];
  q.running = name; store.set(QKEY, q);
  setProgress(`${m.title}: starting`, 0);
  const refRuns = q.results.filter((x) => x.model === REF && x.xrt);
  const stop = refRuns.length && name !== REF ? geo(refRuns.map((x) => x.xrt)) * E.FLOOR : 0;
  let res;
  try {
    if (!r) throw new Error(`no ${q.backend} recipe`);
    const c = r.engine ? disguised(await clipOnce()) : await clipOnce();
    const out = await E.separate({ ...r, overlap: m.overlap, layout: m.layout }, c.mix, { stopBelowXrt: stop,
      onProgress: ({ stage, f }) => setProgress(`${m.title} (${q.i + 1} of ${q.order.length}): ${STAGES[stage]}`, f) });
    if (out.belowFloor) res = { model: name, xrt: out.xrt, error: "slower than 0.05× HTDemucs" };
    else {
      setProgress(`${m.title} (${q.i + 1} of ${q.order.length}): checking the output`, .98);
      const s = E.sdr(c.vocals[0], c.vocals[1], out.voc[0], out.voc[1]), want = EXPECT[name];
      res = { model: name, xrt: out.xrt, sdr: +s.toFixed(3), loadMs: out.loadMs | 0, warmMs: out.warmMs | 0, downloadMs: out.downloadMs | 0, ...(out.tiled ? { tiled: true } : {}) };
      if (want != null && Math.abs(s - want) > 0.5) { res.error = `wrong output: ${s.toFixed(2)} dB here, ${want.toFixed(2)} dB natively`; }
      setProgress(`${m.title} (${q.i + 1} of ${q.order.length}): saving for Listen`, .99);
      res.stemSaved = await saveStem(name, out.voc).then(() => true).catch((e) => String(e?.message ?? e).slice(0, 80));
    }
  } catch (e) { res = { model: name, error: String(e?.message ?? e).slice(0, 240) }; }
  q.results.push(res); q.running = null; q.i++; store.set(QKEY, q);
  if (q.i < q.order.length) { saveScroll(); location.reload(); return; }
  store.del(QKEY); store.set("vf-last", q); showResults(q); finish(q);
}

function summarize(q) {
  const out = {};
  for (const n of [...new Set(q.order)]) {
    const rs = q.results.filter((x) => x.model === n), ok = rs.filter((x) => x.xrt && !x.error);
    const bad = rs.find((x) => x.error);
    out[n] = { xrt: bad || !ok.length ? null : +geo(ok.map((x) => x.xrt)).toFixed(4), runs: rs.map((x) => (x.xrt ? +x.xrt.toFixed(3) : null)), sdr: ok[0]?.sdr ?? null, loadMs: ok[0]?.loadMs, warmMs: ok[0]?.warmMs, error: bad?.error, pending: !rs.length, ...(ok[0]?.tiled ? { tiled: true } : {}) };
  }
  return out;
}

function showResults(q) {
  if (!q.results.length) return;
  const s = summarize(q), ref = s[REF]?.xrt, t = $("results");
  t.hidden = false;
  const max = Math.max(...Object.values(s).map((r) => r.xrt ?? 0), 1e-3);
  t.innerHTML = `<thead><tr><th>Model</th><th class="num">× realtime</th><th>Speed</th><th class="num">vs HTDemucs here</th><th class="num">A 3.5 min song</th><th class="num">Output</th><th></th></tr></thead><tbody>` +
    Object.entries(s).map(([n, r]) => {
      const m = MODELS[n], ok = r.xrt != null;
      const ratio = ok && ref ? E.FLOOR > r.xrt / ref ? E.FLOOR : r.xrt / ref : null;
      return `<tr><td class="name">${esc(m.title)}</td><td class="num">${ok ? fmt(r.xrt) : r.pending ? "…" : "–"}</td><td><span class="speedbar" style="width:${ok ? Math.max(3, 140 * r.xrt / max) : 0}px"></span></td><td class="num">${ratio != null ? fmt(ratio) + "×" : "–"}</td><td class="num">${ok ? songTime(r.xrt) : "–"}</td><td class="num ${r.error ? "fail" : "ok"}" title="${esc(r.error ?? (EXPECT[n] != null ? "within 0.5 dB of the native result on this clip. Not the board's quality score" : "SDR on this clip. Not the board's quality score"))}">${r.error ? esc(r.error.length > 44 ? r.error.slice(0, 42) + "…" : r.error) : r.sdr != null ? `✓ ${fmt(r.sdr)} dB` : "…"}</td><td>${ok ? `<button data-listen="${n}">Listen</button>` : ""}</td></tr>`;
    }).join("") + "</tbody>";
  t.querySelectorAll("[data-listen]").forEach((b) => b.addEventListener("click", () => listen(b.dataset.listen)));
  $("result-actions").hidden = false;
}

function report(q) {
  return { device: OFFICIAL || "unofficial", backend: q.backend, official: q.official, onBattery: !!q.onBattery, date: new Date().toISOString(), ua: navigator.userAgent, adapter, cores: navigator.hardwareConcurrency, memoryGB: navigator.deviceMemory ?? null, coi: self.crossOriginIsolated, models: summarize(q) };
}

function finish(q) {
  releaseScreen(); store.del(SCROLL_KEY);
  const res = summarize(q), mine = Object.keys(res).find((n) => n !== REF && res[n].xrt && res[REF]?.xrt);
  if (PREVIEW && mine) { store.set(`vf-preview-${q.backend}`, { xrt: res[mine].xrt, ratio: Math.max(E.FLOOR, res[mine].xrt / res[REF].xrt) }); board(); }
  setProgress(null);
  $("run").disabled = false;
  const rep = report(q);
  $("copy").onclick = async () => { await navigator.clipboard.writeText(JSON.stringify(rep, null, 1)); $("copy").textContent = "Copied"; setTimeout(() => ($("copy").textContent = "Copy results as JSON"), 1500); };
  $("download").onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(rep, null, 1)], { type: "application/json" })); a.download = `${rep.device}-${rep.date.replace(/[:.]/g, "")}.json`; a.click(); };
  $("official-note").textContent = OFFICIAL ? (q.onBattery ? `This ran on battery at some point, so it won't count as official. Plug in and run it again.` : `Official run on “${OFFICIAL}”: put this file in results/devices/${OFFICIAL}/.`) : (q.onBattery ? "This ran on battery at some point, so treat the numbers as rough." : "");
  if (REPO) { const i = $("issue"); i.hidden = false; i.href = `${REPO}/issues/new?title=${encodeURIComponent(`Device results: ${adapter?.description ?? "unknown"}`)}&body=${encodeURIComponent("```json\n" + JSON.stringify(rep) .slice(0, 6000) + "\n```")}`; }
  window.__vfDone = rep;
}

function setProgress(text, f = 0) {
  const p = $("progress");
  if (text == null) { p.hidden = true; return; }
  p.hidden = false; $("progress-text").textContent = text; p.querySelector("i").style.width = `${Math.min(99, Math.round(f * 100))}%`;     // full means done, and then the bar is gone
}

// ---- stems for listening: kept in IndexedDB, since the page reloads between models. Only a convenience; failure is fine.
function db() { return new Promise((res, rej) => { const r = indexedDB.open("vf", 1); r.onupgradeneeded = () => r.result.createObjectStore("stems"); r.onsuccess = () => res(r.result); r.onerror = r.onblocked = () => rej(r.error ?? new Error("database blocked")); }); }
// Listening is a convenience, so saving may fail or stall (quota, a blocked database) without ever holding up the run.
async function saveStem(name, voc) {
  const save = (async () => {
    const d = await db();
    try { await new Promise((res, rej) => { const t = d.transaction("stems", "readwrite"); t.objectStore("stems").put({ L: voc[0], R: voc[1] }, name); t.oncomplete = res; t.onerror = t.onabort = () => rej(t.error ?? new Error("save aborted")); }); }
    finally { d.close(); }
  })();
  return Promise.race([save, new Promise((_, rej) => setTimeout(() => rej(new Error("save timed out")), 8000))]);
}
async function loadStem(name) { const d = await db(); return new Promise((res, rej) => { const q = d.transaction("stems").objectStore("stems").get(name); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }); }

let actx, node, playing = null, buffers = {};
async function listen(name) {
  const c = await clipOnce(), st = await loadStem(name).catch(() => null);
  if (!st) { $("run-note").textContent = "This browser didn't keep the audio. Run it again."; return; }
  actx ??= new AudioContext({ sampleRate: 44100 });
  const mk = (L, R) => { const b = actx.createBuffer(2, L.length, 44100); b.copyToChannel(L, 0); b.copyToChannel(R, 1); return b; };
  const instL = c.mix[0].map((v, i) => v - st.L[i]), instR = c.mix[1].map((v, i) => v - st.R[i]);
  buffers = { mix: mk(c.mix[0], c.mix[1]), vocals: mk(st.L, st.R), inst: mk(instL, instR), truth: mk(c.vocals[0], c.vocals[1]) };
  $("listen").hidden = false; $("listen-title").textContent = MODELS[name].title; drawWave(st.L, c.vocals[0]);
  $("credits").innerHTML = "Test clip: " + c.meta.credits.map((x) => `<a href="${x.url}" target="_blank" rel="noopener">${esc(x.title)}</a> (${x.role}, ${esc(x.license)})`).join(", ") + ". " + esc(c.meta.note);
  $("listen").scrollIntoView({ block: "nearest" });
}
function drawWave(est, truth) {
  const cv = $("wave"), dpr = devicePixelRatio || 1, w = cv.clientWidth, h = 90;
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext("2d"), css = getComputedStyle(document.documentElement), step = est.length / w | 0;
  g.scale(dpr, dpr);
  const band = (a, col, y0, y1) => { g.fillStyle = col; for (let x = 0; x < w; x++) { let m = 0; for (let i = 0; i < step; i += 16) m = Math.max(m, Math.abs(a[x * step + i] || 0)); const v = Math.min(1, m) * (y1 - y0) / 2; g.fillRect(x, (y0 + y1) / 2 - v, 1, Math.max(1, v * 2)); } };
  band(truth, css.getPropertyValue("--ink3"), 0, 44); band(est, css.getPropertyValue("--s-ours"), 46, 90);
}
document.querySelector(".ab").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  node?.stop(); node = null;
  document.querySelectorAll(".ab button").forEach((x) => x.setAttribute("aria-pressed", "false"));
  if (b.dataset.track && buffers[b.dataset.track]) {
    actx.resume(); node = actx.createBufferSource(); node.buffer = buffers[b.dataset.track]; node.connect(actx.destination); node.start();
    b.setAttribute("aria-pressed", "true");
  }
});

// ---- picking what to run ---------------------------------------------------------------------------------------------
const fileBytes = (f) => f.bytes + (f.data?.bytes ?? 0);
function mb(n) { return n >= 1e9 ? `${(n / 1e9).toFixed(1)} GB` : `${Math.round(n / 1e6)} MB`; }
function drawModels() {
  const list = Object.values(MODELS).filter((m) => m[backend]);
  $("model-list").innerHTML = list.map((m) => {
    const ref = m.name === REF, c = m[backend];
    return `<label class="${ref ? "ref" : ""}"><input type="checkbox" value="${m.name}" ${ref ? "checked disabled" : store.get("vf-sel-" + m.name) ? "checked" : ""}>${esc(m.title)}${ref ? " · 1× here" : ""}<small>${mb(fileBytes(c.file))}</small></label>`;
  }).join("");
  $("model-list").onchange = (e) => { store.set("vf-sel-" + e.target.value, e.target.checked); note(); };
  note();
  if (OFFICIAL) $("model-list").querySelectorAll("input").forEach((i) => (i.checked = true));
}
function note() {
  const picked = [...$("model-list").querySelectorAll("input:checked")].map((i) => MODELS[i.value][backend].file);
  const uniq = new Map(picked.map((f) => [f.sha256, fileBytes(f)]));
  const total = [...uniq.values()].reduce((a, b) => a + b, 0);
  $("run-note").textContent = `Downloads up to ${mb(total)} once, then stays cached. Plug in if you're on battery.`;
}
function setBackend(b) {
  backend = b;
  $("tab-webgpu").setAttribute("aria-selected", b === "webgpu"); $("tab-browser").setAttribute("aria-selected", b === "browser");
  drawModels();
  const blocked = b === "webgpu" && !adapter?.ok;
  $("run").disabled = blocked;
  if (blocked) $("run-note").textContent = `Can't run on the GPU here: ${adapter?.reason}. ${/Linux/.test(navigator.userAgent) ? "On Linux, Chrome may need chrome://flags/#enable-vulkan and #enable-unsafe-webgpu." : ""}`;
}

// ---- start -----------------------------------------------------------------------------------------------------------
async function main() {
  weights();
  if (REPO) { const l = $("repo-link"); l.hidden = false; l.href = REPO; }
  $("deadline").textContent = DEADLINE ? `and it ends ${new Date(DEADLINE).toUTCString().slice(5, 22)} UTC` : "and the dates are still to come";
  $("foot").textContent = "Runs on ONNX Runtime Web. Voc_FT is by the Ultimate Vocal Remover developers; Kim Vocal 2 and MelBand Kim are by Kimberley Jensen; HTDemucs is by Meta. Models keep their own licenses, listed in the table.";
  const j = await (await fetch("models.json")).json(); MODELS = j.models; REF = j.reference;
  await board();
  adapter = await E.adapterInfo().catch((e) => ({ ok: false, reason: String(e) }));
  $("device").innerHTML = `<b>${adapter.ok ? esc(adapter.description || adapter.architecture || adapter.vendor || "GPU") : "No usable GPU"}</b>${adapter.ok ? ` · ${esc(adapter.vendor ?? "")} ${esc(adapter.architecture ?? "")} · fp16 ${adapter.f16 ? "yes" : "no"}` : ` · ${esc(adapter.reason)}`} · ${navigator.hardwareConcurrency} threads${navigator.deviceMemory ? ` · ${navigator.deviceMemory}+ GB` : ""}${self.crossOriginIsolated ? "" : " · not cross-origin isolated"}`;
  $("tab-webgpu").onclick = () => setBackend("webgpu"); $("tab-browser").onclick = () => setBackend("browser");
  setBackend(Q.get("backend") || (adapter.ok ? "webgpu" : "browser"));
  $("run").onclick = () => {
    const names = [...$("model-list").querySelectorAll("input:checked")].map((i) => i.value);
    const q = newQueue(names, backend, { official: !!OFFICIAL });
    $("run").disabled = true; $("results").hidden = true;
    runNext(q);
  };
  const pending = store.get(QKEY), last = store.get("vf-last");
  const html = document.documentElement;
  if (pending) {
    $("run").disabled = true; setBackend(pending.backend);
    runNext(pending);                                  // puts the bar and the results back before its first wait
    restoreScroll();
    addEventListener("scroll", saveScroll, { passive: true });   // so a crash, which can't save anything, still comes back here
    await Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 1500))]);
    restoreScroll();                                   // fonts can change what's above the bar
    html.classList.remove("resuming");
  } else {
    if (last) { showResults(last); finish(last); }
    addEventListener("scroll", saveScroll, { passive: true });
    // A link to #bench scrolled before the board existed and pushed it down, so go there again. A reload keeps wherever you were.
    const nav = performance.getEntriesByType("navigation")[0]?.type;
    if (location.hash && nav === "navigate") document.querySelector(location.hash)?.scrollIntoView({ behavior: "instant" });
  }
  requestAnimationFrame(() => html.classList.add("ready"));
  window.VF = { store, MODELS, summarize, start: (names, be, official) => { store.del("vf-last"); return runNext(newQueue(names, be, { official })); } };
  window.__vfReady = true;
}
main().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p style="padding:16px">${esc(e.message)}</p>`); });
