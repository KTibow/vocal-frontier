// An official run of the page on this machine's Chromium: the same code path a person gets, with ABBA rounds and a reload per model.
//   node tools/official.mjs <device: browser|vega> <url> <model...>     writes results/devices/<device>/<date>.json
// Run it through `vf official`, which takes the machine's lock, waits for memory and pins one thread per physical core.
import puppeteer from "puppeteer-core";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
const [device, url, ...names] = process.argv.slice(2);
const backend = device === "browser" ? "browser" : "webgpu";
const avail = () => +/MemAvailable:\s+(\d+)/.exec(fs.readFileSync("/proc/meminfo", "utf8"))[1] / 1024;
// The browser reference alone peaks near 4 GB over the watchdog's floor. Starting short of that voids the whole run, so wait for other work to finish first.
const needMb = backend === "browser" ? 4800 : 2500;
for (let waited = 0; avail() < needMb; waited++) {
  if (waited >= 180) { console.error(`refusing to start: MemAvailable ${avail() | 0} MB, need ${needMb}`); process.exit(5); }
  await new Promise((r) => setTimeout(r, 10000));
}
// Kept across restarts, since the queue lives in its storage. On disk, never in os.tmpdir(): /tmp is often RAM, and the model cache alone
// would use a gigabyte of the memory being measured, until the watchdog below killed every model.
const profileRoot = path.join(os.homedir(), ".cache", "vf-official");
fs.mkdirSync(profileRoot, { recursive: true });
const profile = fs.mkdtempSync(path.join(profileRoot, `${device}-`));
process.on("exit", () => fs.rmSync(profile, { recursive: true, force: true }));
const launch = () => puppeteer.launch({ executablePath: process.env.VF_CHROME || "/usr/bin/chromium", headless: "new", protocolTimeout: 7200000, userDataDir: profile,
  args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--use-vulkan=native", "--disable-dev-shm-usage", ...(backend === "browser" ? ["--disable-gpu"] : [])] });
let rep = null, first = true, restarts = 0;
while (!rep) {
  const browser = await launch();
  // Over budget means that model failed. Kill the browser, and the next launch records the crash and moves on to the next model.
  let killed = false;
  const wd = setInterval(() => { if (avail() < 1200) { killed = true; console.error(`memory watchdog: MemAvailable ${avail() | 0} MB, killing the browser`); browser.process()?.kill("SIGKILL"); } }, 250);
  browser.on("disconnected", () => { killed = true; });
  try {
    const page = await browser.newPage();
    page.setDefaultTimeout(0);
    page.on("console", (m) => { if (m.type() === "error") console.error("[page]", m.text().slice(0, 200)); });
    await page.goto(`${url}?device=${device}&backend=${backend}#bench`, { waitUntil: "load" });
    await page.waitForFunction("window.__vfReady === true");
    if (first) { first = false; page.evaluate((n, b) => { localStorage.removeItem("vf-last"); return window.VF.start(n, b, true); }, names, backend).catch(() => {}); }
    while (!rep && !killed) {
      await new Promise((r) => setTimeout(r, 2000));
      try { rep = await page.evaluate(() => window.__vfDone || null); } catch { /* reloading */ }
    }
  } catch (e) { if (!killed) console.error("restart:", String(e.message).slice(0, 200)); }
  clearInterval(wd);
  await browser.close().catch(() => {});
  if (!rep && ++restarts > names.length * 2 + 2) { console.error("too many restarts"); process.exit(6); }   // every model can crash once per round
  if (!rep) await new Promise((r) => setTimeout(r, 15000));     // let memory come back
}
const dir = `results/devices/${device}`; fs.mkdirSync(dir, { recursive: true });
const file = `${dir}/${rep.date.replace(/[:.]/g, "")}.json`;
fs.writeFileSync(file, JSON.stringify(rep, null, 1));
console.log(file); for (const [n, r] of Object.entries(rep.models)) console.log(n.padEnd(24), r.xrt ?? `FAILED ${r.error}`, r.sdr ?? "");
