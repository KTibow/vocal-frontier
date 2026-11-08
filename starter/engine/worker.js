// Starter engine: the stock worker, so it runs exactly like the site does until you change it.
// The page sends these messages, and everything you do after `init` is in the timing.
//   {type: "init", runtime, threads, bytes, data, layoutSpec} -> {chunk} (samples per chunk). Your file's bytes, in any format you can read.
//     An .onnx file arrives split: bytes is the graph and data is {name, bytes}, its weights, or null if it has none worth splitting. Any other extension arrives as is.
//   {type: "song", L, R}                                -> {}      the whole song, before any chunk, if you want it
//   {type: "run", L, R}                                 -> {L, R, coreMs}  one chunk in, its vocals out, same length; the page stitches
// Or take over stitching too: reply to init with {whole: true}, and the page sends instead
//   {type: "whole", L, R, overlap}                      -> {L, R}  the whole song in, its vocals out
// Every reply repeats the message's `id`. On failure, send {id, error}. /src/ and /ort/ are there to import, and the network isn't.
import { makeLayout } from "/src/layouts.js";

let ort, sess, layout, inNames, outNames;

async function init({ runtime, bytes, data, layoutSpec, threads }) {
  const [ep, version] = (runtime || "wasm@1.30.0").split("@");          // "webgpu@1.23.0" or "wasm@1.23.0"
  const base = new URL(`/ort/${version}/`, import.meta.url).href;
  ort = await import(`${base}${ep === "webgpu" ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"}`);
  ort.env.wasm.wasmPaths = base;
  if (ep === "webgpu") {       // some builds are shipped in parts, because of the host's per-file limit
    const r = await fetch(`${base}webgpu-wasm-parts.json`);
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) {
      const { parts } = await r.json();
      const bufs = await Promise.all(Array.from({ length: parts }, (_, i) => fetch(`${base}ort-wasm-simd-threaded.asyncify.wasm.${i}`).then((x) => x.arrayBuffer())));
      const whole = new Uint8Array(bufs.reduce((n, b) => n + b.byteLength, 0));
      let o = 0; for (const b of bufs) { whole.set(new Uint8Array(b), o); o += b.byteLength; }
      ort.env.wasm.wasmBinary = whole.buffer;
    }
  }
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = false;
  const so = { executionProviders: [ep === "webgpu" ? { name: "webgpu", preferredLayout: "NCHW" } : "wasm"],
               graphOptimizationLevel: "all", freeDimensionOverrides: { batch_size: 1 } };
  if (data) so.externalData = [{ path: data.name, data: new Uint8Array(data.bytes) }];
  sess = await ort.InferenceSession.create(new Uint8Array(bytes), so);
  inNames = sess.inputNames; outNames = sess.outputNames;
  layout = makeLayout(layoutSpec);
  if (ep === "webgpu") Promise.resolve(ort.env.webgpu?.device).then((d) => d?.lost?.then((i) => { if (i.reason !== "destroyed") self.postMessage({ deviceLost: i.message || i.reason || "GPU device lost" }); })).catch(() => {});
  return { chunk: layout.chunk, inputs: inNames, outputs: outNames };
}

async function run(L, R) {
  const raw = layout.feeds(L, R, inNames), feeds = {};
  for (const k in raw) feeds[k] = new ort.Tensor("float32", raw[k].data, raw[k].dims);
  const t = performance.now();
  const o = await sess.run(feeds);
  const coreMs = performance.now() - t;
  const v = layout.wave(o, outNames);
  for (const k in o) o[k].dispose?.();
  return { v, coreMs };
}

self.onmessage = async ({ data: m }) => {
  try {
    if (m.type === "init") return self.postMessage({ id: m.id, ...(await init(m)) });
    if (m.type === "song") { layout.song?.(m.L, m.R); return self.postMessage({ id: m.id }); }
    if (m.type === "run") {
      const { v, coreMs } = await run(m.L, m.R);
      return self.postMessage({ id: m.id, L: v[0], R: v[1], coreMs }, [v[0].buffer, v[1].buffer]);
    }
  } catch (e) {
    self.postMessage({ id: m.id, error: String(e?.message ?? e) });
  }
};
