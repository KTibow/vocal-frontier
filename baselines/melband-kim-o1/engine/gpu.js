// MelBand Kim on WebGPU as Better Lyrics Composer runs it since PR #461: one ONNX Runtime session per stage
// (band split, 12 transformers, mask estimator), each output kept on the GPU for the next, and a wait for the GPU after each stage so
// its released buffers can be reused. Bundle from starter/slice_melband_gpu.py. Same messages as starter/engine/worker.js. No code of theirs.
import { makeLayout } from "/src/layouts.js";

let ort, layout, sessions = [], device;

async function init({ runtime, bytes, layoutSpec }) {
  const version = runtime.split("@")[1], base = new URL(`/ort/${version}/`, import.meta.url).href;
  ort = await import(`${base}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = base; ort.env.wasm.proxy = false;
  const r = await fetch(`${base}webgpu-wasm-parts.json`);
  if (r.ok && (r.headers.get("content-type") || "").includes("json")) {
    const { parts } = await r.json();
    const bufs = await Promise.all(Array.from({ length: parts }, (_, i) => fetch(`${base}ort-wasm-simd-threaded.asyncify.wasm.${i}`).then((x) => x.arrayBuffer())));
    const whole = new Uint8Array(bufs.reduce((n, b) => n + b.byteLength, 0));
    let o = 0; for (const b of bufs) { whole.set(new Uint8Array(b), o); o += b.byteLength; }
    ort.env.wasm.wasmBinary = whole.buffer;
  }
  const n = new DataView(bytes).getUint32(0, true), offset = 4 + n, head = JSON.parse(new TextDecoder().decode(new Uint8Array(bytes, 4, n)));
  for (const [i, e] of head.stages.entries()) {
    sessions.push(await ort.InferenceSession.create(new Uint8Array(bytes, offset + e.graph[0], e.graph[1]), {
      executionProviders: [{ name: "webgpu", storageBufferCacheMode: "bucket" }], graphOptimizationLevel: "all",
      preferredOutputLocation: i < head.stages.length - 1 ? "gpu-buffer" : "cpu",
      externalData: e.data[1] ? [{ path: "w", data: new Uint8Array(bytes, offset + e.data[0], e.data[1]) }] : [] }));
  }
  device = await ort.env.webgpu.device;
  device.lost.then((i) => { if (i.reason !== "destroyed") self.postMessage({ deviceLost: i.message || i.reason || "GPU device lost" }); }).catch(() => {});
  layout = makeLayout(layoutSpec);
  return { chunk: layout.chunk };
}

async function run(L, R) {
  const raw = layout.feeds(L, R, ["stft_repr"]);
  const t0 = performance.now();
  let input = new ort.Tensor("float32", raw.stft_repr.data, raw.stft_repr.dims), first = true, masks;
  for (const s of sessions) {
    const o = await s.run({ [s.inputNames[0]]: input });
    if (!first) input.dispose?.();
    first = false;
    await device.queue.onSubmittedWorkDone();
    input = masks = o[s.outputNames[0]];
  }
  const coreMs = performance.now() - t0;
  const v = layout.wave({ masks }, ["masks"]);
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
