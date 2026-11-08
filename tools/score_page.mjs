// Scores clips through the site's real code path (its engine or stock worker), with the same memory watchdog as an official run.
//   node tools/score_page.mjs <url> <model> <browser|webgpu> <sets dir> <out.json> <clips json: {set: [clip...]}>
// `vf transfer` and `vf quality --backend` call this. Writes {"set/clip": sdr} after every clip and skips clips already in <out.json>, so a
// stopped run picks up where it left off. A full run takes hours on a machine that does other things: when the watchdog fires, this waits
// for memory to come back and tries that clip again. A clip that hits it three times in a row stops the run.
import puppeteer from "puppeteer-core";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
const [url, name, be, dir, out, clips] = process.argv.slice(2);
const avail = () => +/MemAvailable:\s+(\d+)/.exec(fs.readFileSync("/proc/meminfo", "utf8"))[1] / 1024;
const pcm = (f) => Buffer.from(execFileSync("ffmpeg", ["-v", "error", "-i", f, "-ar", "44100", "-ac", "2", "-f", "f32le", "-"], { maxBuffer: 1 << 30 })).toString("base64");
const profile = fs.mkdtempSync(path.join(os.homedir(), ".cache", "vf-score-"));
process.on("exit", () => fs.rmSync(profile, { recursive: true, force: true }));
// A fresh browser per clip: memory built up across clips in one browser until the watchdog stopped a WebGPU run on its fourth clip.
let b;
const start = async () => {
  b = await puppeteer.launch({ executablePath: process.env.VF_CHROME || "/usr/bin/chromium", headless: "new", protocolTimeout: 7200000, userDataDir: profile,
  args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--use-vulkan=native", "--disable-dev-shm-usage", ...(be === "browser" ? ["--disable-gpu"] : [])] });
  const page = await b.newPage();
  await page.goto(`${url}?backend=${be}`, { waitUntil: "load" });
  await page.waitForFunction("window.__vfReady === true");
  if (be === "webgpu") {
    // After a logout the render node can belong to the login screen, and Chromium falls back to a software adapter. Don't score on that.
    const gpu = await page.evaluate(async () => (await import("/src/engine.js")).adapterInfo());
    if (!gpu.ok) { console.error(`no usable GPU: ${gpu.reason}. Is the desktop session logged in? Run it again to continue.`); process.exit(7); }
  }
  return page;
};
// One clip used 2.7 GB on WebGPU, so 4 GB free clears the floor. Asking for more stalled a run for hours on a box that sat near 4.3 GB free.
const FLOOR_MB = 1200, START_MB = 4000, TRIES = 3, WAIT_MS = 3600000;
let scoring = false, overMemory = false;
const wd = setInterval(() => { if (scoring && avail() < FLOOR_MB) { if (!overMemory) console.error(`memory watchdog: MemAvailable ${avail() | 0} MB, killing the browser`); overMemory = true; b?.process()?.kill("SIGKILL"); } }, 250);
const enoughMemory = async () => {
  const until = Date.now() + WAIT_MS;
  while (avail() < START_MB) {
    if (Date.now() > until) { console.error(`gave up after an hour waiting for ${START_MB} MB free, have ${avail() | 0} MB. Run it again to continue.`); process.exit(6); }
    await new Promise((r) => setTimeout(r, 10000));
  }
};
const res = fs.existsSync(out) ? JSON.parse(fs.readFileSync(out, "utf8")) : {};
const save = () => { fs.writeFileSync(`${out}.tmp`, JSON.stringify(res)); fs.renameSync(`${out}.tmp`, out); };
const score = (p, n, s, c) => p.evaluate(async (n, be, mix64, voc64) => {
  const E = await import("/src/engine.js");
  const dec = (x) => { const u = Uint8Array.from(atob(x), (ch) => ch.charCodeAt(0)), f = new Float32Array(u.buffer), L = new Float32Array(f.length / 2), R = new Float32Array(f.length / 2); for (let i = 0; i < L.length; i++) { L[i] = f[2 * i]; R[i] = f[2 * i + 1]; } return [L, R]; };
  const m = window.VF.MODELS[n], mix = dec(mix64), voc = dec(voc64);
  const o = await E.separate({ ...m[be], overlap: m.overlap, layout: m.layout }, mix);
  return E.sdr(voc[0], voc[1], o.voc[0], o.voc[1]);
}, n, be, pcm(path.join(dir, s, c, "mixture.flac")), pcm(path.join(dir, s, c, "vocals.flac")));
const done = Object.keys(res).length;
if (done) console.log(`continuing: ${done} clips already scored`);
for (const [s, list] of Object.entries(JSON.parse(clips))) {
  for (const c of list) {
    if (`${s}/${c}` in res) continue;
    for (let tries = 1; ; tries++) {
      await enoughMemory();
      overMemory = false;
      scoring = true;
      try {
        res[`${s}/${c}`] = await score(await start(), name, s, c);
        scoring = false;
        await b.close();
        break;
      } catch (e) {
        scoring = false;
        if (!overMemory) throw e;
        if (tries >= TRIES) { console.error(`stopped: clip ${Object.keys(res).length + 1} ran out of memory ${TRIES} times in a row. Run it again to continue.`); process.exit(6); }
      }
    }
    save();
    console.log(`${s} ${Object.keys(res).length}: done`);
  }
}
clearInterval(wd);
