// Runs a model on this device: fetch, one warm-up chunk (not timed), then the whole clip, end to end. Same stitching
// as vf/demix.py. Chunks and fades are what quality is scored with, so speed and quality can't disagree.

export const FLOOR = 0.05;

const reflect = (x, i, L) => { while (i < 0 || i >= L) i = i < 0 ? -i : 2 * L - 2 - i; return x[i]; };

function sha256hex(buf) {
  return crypto.subtle.digest("SHA-256", buf).then((d) => [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join(""));
}

// A file is {sha256, bytes, parts: [urls]}. Parts keep each asset under Cloudflare's 25 MiB limit. Cached whole by hash.
export async function fetchFile(file, onProgress = () => {}) {
  const cache = "caches" in self ? await caches.open("vf-models-1").catch(() => null) : null;
  const key = `/__vf_cache/${file.sha256}`;
  const hit = await cache?.match(key).catch(() => null);
  if (hit) {                                      // checked every time: an engine shares this origin's storage, so a cached model can't be taken on trust
    const b = await hit.arrayBuffer();
    if (await sha256hex(b) === file.sha256) return b;
    await cache.delete(key);
  }
  const out = new Uint8Array(file.bytes);
  let got = 0;
  for (const url of file.parts) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`download failed: ${r.status} ${url}`);
    const rd = r.body.getReader();
    for (;;) {
      const { done, value } = await rd.read();
      if (done) break;
      out.set(value, got); got += value.length; onProgress(got / file.bytes);
    }
  }
  const h = await sha256hex(out);
  if (h !== file.sha256) throw new Error(`hash mismatch: got ${h.slice(0, 12)}, expected ${file.sha256.slice(0, 12)}`);
  await cache?.put(key, new Response(out)).catch(() => {});
  return out.buffer;
}

class Pool {
  constructor(n, engine) {
    const url = engine ? new URL(engine, location.href) : new URL("./worker.js", import.meta.url);
    this.ws = Array.from({ length: n }, () => new Worker(url, { type: "module" }));
    this.seq = 0; this.wait = new Map(); this.dead = null;
    for (const w of this.ws) {
      w.onmessage = ({ data }) => { if (data.deviceLost) return this.fail(new Error("the GPU device was lost: " + data.deviceLost)); const p = this.wait.get(data.id); if (!p) return; this.wait.delete(data.id); data.error ? p.reject(new Error(data.error)) : p.resolve(data); };
      w.onerror = (e) => this.fail(new Error(e.message || "worker crashed"));
    }
  }
  fail(err) { this.dead = err; for (const p of this.wait.values()) p.reject(err); this.wait.clear(); }
  // limitMs: a GPU can hang without ever raising an error, so a call that never returns has to count as a failure
  call(w, msg, transfer = [], limitMs = 0) {
    if (this.dead) return Promise.reject(this.dead);
    return new Promise((resolve, reject) => {
      const id = ++this.seq;
      const timer = limitMs ? setTimeout(() => { this.wait.delete(id); this.fail(new Error(`no answer from the ${msg.type === "init" ? "runtime" : "GPU or CPU"} for ${Math.round(limitMs / 1000)} s: it hung`)); reject(this.dead); }, limitMs) : 0;
      this.wait.set(id, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject: (e) => { clearTimeout(timer); reject(e); } });
      w.postMessage({ ...msg, id }, transfer);
    });
  }
  close() { for (const w of this.ws) w.terminate(); }
}

export function adapterInfo() {
  return (async () => {
    if (!navigator.gpu) return { ok: false, reason: "this browser doesn't expose WebGPU" };
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!a) return { ok: false, reason: "no WebGPU adapter" };
    const i = a.info ?? {};
    const soft = a.isFallbackAdapter || i.isFallbackAdapter || i.architecture === "swiftshader" || /swiftshader|llvmpipe/i.test(i.description ?? "");
    return { ok: !soft, reason: soft ? "only a software adapter (SwiftShader), which isn't a GPU" : "", vendor: i.vendor, architecture: i.architecture, description: i.description,
             f16: a.features.has("shader-f16"), maxBuffer: a.limits.maxBufferSize, maxStorageBuffers: a.limits.maxStorageBuffersPerShaderStage, maxBinding: a.limits.maxStorageBufferBindingSize };
  })();
}

// A GPU that allows less in one buffer than a graph needs gets the same model cut into time pieces, which `vf build` makes (vf/tile.py).
let binding;
const gpuBinding = () => (binding ??= adapterInfo().then((i) => i.maxBinding ?? Infinity, () => Infinity));

export const DEFAULT_THREADS = () => Math.min(navigator.hardwareConcurrency || 4, 4);

// plan = {runtime, workers, threads, file, overlap, layout}. Returns vocals and timing, or throws. stopBelowXrt ends the run
// as soon as finishing can't beat it, which is what makes a floor run take minutes rather than hours.
export async function separate(plan, [L0, R0], { onProgress = () => {}, stopBelowXrt = 0, bytes = null } = {}) {
  if (plan.tiled && !bytes && (await gpuBinding()) < plan.tiled.below) plan = { ...plan, file: plan.tiled.file, usedTiled: true };
  const t00 = performance.now();
  const buf = bytes ?? await fetchFile(plan.file, (f) => onProgress({ stage: "download", f }));
  const side = plan.file.data && await fetchFile(plan.file.data, (f) => onProgress({ stage: "download", f }));   // weights kept outside the graph: they reach the GPU without a copy in the WASM heap
  const downloadMs = performance.now() - t00;
  const n = plan.workers || 1, pool = new Pool(n, plan.engine);
  try {
    onProgress({ stage: "load", f: 0 });
    const t0 = performance.now();
    // Moved, not copied: a clone of the model file would sit next to ONNX Runtime's own, which a big model can't afford on a shared-memory iGPU.
    const infos = await Promise.all(pool.ws.map((w, i) => {
      const last = i === n - 1, own = bytes || !last ? buf.slice(0) : buf, weights = side && (last ? side : side.slice(0));
      return pool.call(w, { type: "init", runtime: plan.runtime, threads: plan.threads || 1, bytes: own, data: side ? { name: plan.file.data.name, bytes: weights } : null, layoutSpec: plan.layout }, side ? [own, weights] : [own], 600000);
    }));
    const C = infos[0].chunk, loadMs = performance.now() - t0;
    if (infos[0].whole) {                         // an engine that takes the whole song: it does its own chunking and stitching, and all of it is timed
      const w = pool.ws[0], clipS = L0.length / 44100, ten = 44100 * 10;
      onProgress({ stage: "warmup", f: 0 });
      const t1 = performance.now();
      await pool.call(w, { type: "whole", L: L0.slice(0, ten), R: R0.slice(0, ten), overlap: plan.overlap }, [], 600000);
      const warmMs = performance.now() - t1;
      onProgress({ stage: "run", f: .5 });
      const limit = stopBelowXrt ? Math.max(120000, 2 * clipS / stopBelowXrt * 1000) : 3600000;
      const start = performance.now();
      const r = await pool.call(w, { type: "whole", L: L0.slice(), R: R0.slice(), overlap: plan.overlap }, [], limit);
      const wallMs = performance.now() - start;
      return { voc: [r.L, r.R], xrt: clipS / (wallMs / 1000), wallMs, coreMs: r.coreMs ?? wallMs, loadMs, warmMs, downloadMs, chunks: 1, belowFloor: stopBelowXrt > 0 && clipS / (wallMs / 1000) < stopBelowXrt };
    }
    const T0 = L0.length, step = Math.max(1, Math.floor(C / plan.overlap)), fade = Math.floor(C / 10), border = C - step;
    const pad = border > 0 && T0 > 2 * border ? border : 0, T = T0 + 2 * pad;
    const padded = (c) => { if (!pad) return c; const o = new Float32Array(T); for (let i = 0; i < T; i++) o[i] = reflect(c, i - pad, T0); return o; };
    const X = [padded(L0), padded(R0)];
    const part = (i) => {
      const len = Math.min(C, T - i);
      return [X[0], X[1]].map((c) => {
        const o = new Float32Array(C);
        if (len < C && len > C / 2 + 1) for (let k = 0; k < C; k++) o[k] = reflect(c.subarray(i, i + len), k, len);
        else o.set(c.subarray(i, i + len));
        return o;
      }).concat([len]);
    };
    await Promise.all(pool.ws.map((w) => pool.call(w, { type: "song", L: L0, R: R0 })));
    // warm-up: first run compiles shaders and sizes arenas; once per worker, not timed
    onProgress({ stage: "warmup", f: 0 });
    const t1 = performance.now();
    await Promise.all(pool.ws.map((w) => { const [a, b] = part(0); return pool.call(w, { type: "run", L: a, R: b }, [a.buffer, b.buffer], 600000); }));
    const warmMs = performance.now() - t1;

    const starts = []; for (let i = 0; i < T; i += step) { starts.push(i); if (i + C >= T) break; }
    const out = [new Float32Array(T), new Float32Array(T)], cnt = new Float32Array(T);
    const win = new Float32Array(C).fill(1);
    for (let i = 0; i < fade; i++) { const v = i / (fade - 1); win[i] = v; win[C - 1 - i] = v; }
    let next = 0, done = 0, coreMs = 0, stopped = false;
    const clipS = T0 / 44100, start = performance.now();
    // the slowest a chunk may take: well beyond what the warm-up chunk took, and beyond anything that could still beat the floor
    const chunkLimit = Math.max(120000, 6 * warmMs, stopBelowXrt ? 2 * (C / 44100 / plan.overlap) / stopBelowXrt * 1000 : 0);
    await Promise.all(pool.ws.map(async (w) => {
      while (next < starts.length && !stopped) {
        const i = starts[next++], [a, b, len] = part(i);
        const r = await pool.call(w, { type: "run", L: a, R: b }, [a.buffer, b.buffer], chunkLimit);
        coreMs += r.coreMs;
        for (let k = 0; k < len; k++) {
          let g = win[k];
          if (i === 0 && k < fade) g = 1;
          if (i + C >= T && k >= C - fade) g = 1;
          out[0][i + k] += r.L[k] * g; out[1][i + k] += r.R[k] * g; cnt[i + k] += g;
        }
        done++;
        onProgress({ stage: "run", f: .97 * done / starts.length });
        if (stopBelowXrt) {
          const proj = (performance.now() - start) / done * starts.length;
          if (clipS / (proj / 1000) < stopBelowXrt) stopped = true;
        }
      }
    }));
    onProgress({ stage: "stitch", f: .97 });
    const wallMs = performance.now() - start;
    if (stopped) { const proj = wallMs / done * starts.length; return { xrt: clipS / (proj / 1000), belowFloor: true, loadMs, warmMs, downloadMs, chunks: done }; }
    for (let k = 0; k < T; k++) { const c = Math.max(cnt[k], 1e-8); out[0][k] /= c; out[1][k] /= c; }
    const voc = pad ? out.map((c) => c.slice(pad, pad + T0)) : out;
    return { voc, xrt: clipS / (wallMs / 1000), wallMs, coreMs, loadMs, warmMs, downloadMs, chunks: starts.length, tiled: !!plan.usedTiled };
  } finally {
    pool.close();
  }
}

export function sdr(refL, refR, estL, estR) {
  let s = 0, e = 0;
  for (let i = 0; i < refL.length; i++) {
    s += refL[i] * refL[i] + refR[i] * refR[i];
    const a = refL[i] - estL[i], b = refR[i] - estR[i]; e += a * a + b * b;
  }
  return 10 * Math.log10((s + 1e-10) / (e + 1e-10));
}

export async function loadClip(base = "clip") {
  const dec = async (name) => {
    const b = new Int16Array(await (await fetch(`${base}/${name}.i16`)).arrayBuffer()), n = b.length / 2;
    const L = new Float32Array(n), R = new Float32Array(n);
    for (let i = 0; i < n; i++) { L[i] = b[2 * i] / 32767; R[i] = b[2 * i + 1] / 32767; }
    return [L, R];
  };
  return { mix: await dec("mix"), vocals: await dec("vocals"), meta: await (await fetch(`${base}/clip.json`)).json() };
}
