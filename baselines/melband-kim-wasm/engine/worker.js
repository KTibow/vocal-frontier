// MelBand Kim in WASM. The whole graph can't fit ONNX Runtime's 4 GB heap: each time layer's attention scores alone come to
// 1.23 GB, and with tensors of every size in one graph the heap fragments past the limit. So this engine holds one session per
// stage, from starter/slice_melband.py, and feeds each transformer a few bands or frames at a time. Between calls ONNX Runtime
// holds nothing but weights. The idea of separate graphs per stage comes from adaliea's PR on Better Lyrics Composer, which does it on WebGPU;
// no code is shared. Same messages as starter/engine/worker.js. Bundle: u32 header length, JSON header, then each stage's graph and weights.
import { makeLayout } from "/src/layouts.js";

const BAND_BATCH = 10, FRAME_BATCH = 134;
let ort, layout, head, S = {};

async function session(bundle, base, e) {
  const graph = new Uint8Array(bundle, base + e.graph[0], e.graph[1]), weights = new Uint8Array(bundle, base + e.data[0], e.data[1]);
  return ort.InferenceSession.create(graph, { executionProviders: ["wasm"], graphOptimizationLevel: "all", enableCpuMemArena: false, enableMemPattern: false,
    externalData: e.data[1] ? [{ path: "w", data: weights }] : [] });
}

async function init({ runtime, bytes, layoutSpec, threads }) {
  const version = runtime.split("@")[1], base = new URL(`/ort/${version}/`, import.meta.url).href;
  ort = await import(`${base}ort.wasm.min.mjs`);
  ort.env.wasm.wasmPaths = base; ort.env.wasm.numThreads = threads; ort.env.wasm.proxy = false;
  const n = new DataView(bytes).getUint32(0, true), offset = 4 + n;
  head = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 4, n)));
  for (const e of head.stages) S[e.name] = await session(bytes, offset, e);
  layout = makeLayout(layoutSpec);
  return { chunk: layout.chunk };
}

// [A, B, D] <-> [B, A, D]
function swap(x, A, B, D) {
  const o = new Float32Array(x.length);
  for (let a = 0; a < A; a++) for (let b = 0; b < B; b++) o.set(x.subarray((a * B + b) * D, (a * B + b + 1) * D), (b * A + a) * D);
  return o;
}

// Runs a [A, S, D] tensor through a layer, batch rows at a time.
async function batched(sess, x, A, Sq, D, batch) {
  const o = new Float32Array(x.length), row = Sq * D;
  for (let a = 0; a < A; a += batch) {
    const k = Math.min(batch, A - a);
    const r = await sess.run({ x: new ort.Tensor("float32", x.subarray(a * row, (a + k) * row), [k, Sq, D]) });
    o.set(r.y.data, a * row);
    r.y.dispose?.();
  }
  return o;
}

// Neighbouring bands from `first` up averaged in pairs, on [T, F, D]. See merge_bands in PooledMelBandRoformer.
function merge(x, T, F, D, first) {
  const tokens = first + (F - first) / 2, o = new Float32Array(T * tokens * D);
  for (let t = 0; t < T; t++) {
    const s = t * F * D, d = t * tokens * D;
    o.set(x.subarray(s, s + first * D), d);
    for (let j = 0; j < (F - first) / 2; j++) for (let k = 0; k < D; k++)
      o[d + (first + j) * D + k] = (x[s + (first + 2 * j) * D + k] + x[s + (first + 2 * j + 1) * D + k]) / 2;
  }
  return o;
}

// x += change, each merged band's change going to both of its bands
function addChange(x, z, lo, T, F, D, first) {
  const tokens = first + (F - first) / 2;
  for (let t = 0; t < T; t++) for (let f = 0; f < F; f++) {
    const token = f < first ? f : first + ((f - first) >> 1), s = (t * F + f) * D, m = (t * tokens + token) * D;
    for (let k = 0; k < D; k++) x[s + k] += z[m + k] - lo[m + k];
  }
}

async function run(L, R) {
  const { frames: T, bands: F, tokens, dim: D, layers, merged_from: first } = head;
  const raw = layout.feeds(L, R, ["stft_repr"]);
  const t0 = performance.now();
  const h = await S.head.run({ stft_repr: new ort.Tensor("float32", raw.stft_repr.data, raw.stft_repr.dims) });
  let x = h.x.data.slice(); h.x.dispose?.();
  for (let i = 0; i < layers; i++) {
    const lo = first == null ? x : merge(x, T, F, D, first);
    let z = swap(lo, T, tokens, D);
    z = await batched(S[`tt${i}`], z, tokens, T, D, BAND_BATCH);
    z = swap(z, tokens, T, D);
    z = await batched(S[`ft${i}`], z, T, tokens, D, FRAME_BATCH);
    if (first == null) x = z; else addChange(x, z, lo, T, F, D, first);
  }
  const m = await S.tail.run({ x: new ort.Tensor("float32", x, [1, T, F, D]) });
  const coreMs = performance.now() - t0;
  const v = layout.wave({ masks: m.masks }, ["masks"]);
  m.masks.dispose?.();
  return { v, coreMs };
}

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === "init") return self.postMessage({ id: m.id, ...(await init(m)) });
    if (m.type === "song") return self.postMessage({ id: m.id });
    if (m.type === "run") {
      const { v, coreMs } = await run(m.L, m.R);
      return self.postMessage({ id: m.id, L: v[0], R: v[1], coreMs }, [v[0].buffer, v[1].buffer]);
    }
  } catch (e) {
    self.postMessage({ id: m.id, error: String(e?.message ?? e) });
  }
};
