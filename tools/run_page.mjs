// Runs code in the real page under headless Chromium, on the machine's real GPU, with a memory watchdog.
// node tools/run_page.mjs <url> '<js expression returning a promise>' [timeoutS]
import puppeteer from "puppeteer-core";
import fs from "node:fs";
const [url, expr, timeoutS = "900"] = process.argv.slice(2);
const avail = () => +/MemAvailable:\s+(\d+)/.exec(fs.readFileSync("/proc/meminfo", "utf8"))[1] / 1024;
if (avail() < 2500) { console.error(`refusing to start: MemAvailable ${avail() | 0} MB < 2500 MB`); process.exit(5); }
const browser = await puppeteer.launch({ executablePath: process.env.VF_CHROME || "/usr/bin/chromium", headless: "new",
  args: ["--no-sandbox", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=vulkan", "--use-vulkan=native", "--enable-webgpu-developer-features", "--disable-dev-shm-usage"] });
const kill = (why) => { console.error("killed: " + why); browser.process()?.kill("SIGKILL"); process.exit(6); };
const wd = setInterval(() => { if (avail() < 1200) kill(`MemAvailable ${avail() | 0} MB`); }, 250);
const hard = setTimeout(() => kill(`timeout ${timeoutS}s`), +timeoutS * 1000);
const page = await browser.newPage();
page.on("console", (m) => { const t = m.text(); if (!/powerPreference/.test(t)) console.error("[page]", t.slice(0, 400)); });
page.on("pageerror", (e) => console.error("[pageerror]", e.message));
await page.goto(url, { waitUntil: "load" });
const out = await page.evaluate(expr);
console.log(JSON.stringify(out));
clearInterval(wd); clearTimeout(hard);
await browser.close();
