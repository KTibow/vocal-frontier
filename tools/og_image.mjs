// Draws site/og.png, the picture a shared link shows, from the current board. `vf board` runs it.
import puppeteer from "puppeteer-core";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), ".."), port = 8797;
const srv = spawn("node", [path.join(root, "tools/serve.mjs"), path.join(root, "site"), String(port)], { stdio: "ignore" });
await new Promise((r) => setTimeout(r, 800));
const b = await puppeteer.launch({ executablePath: process.env.VF_CHROME || "/usr/bin/chromium", headless: "new", args: ["--no-sandbox", "--disable-gpu"] });
try {
  const p = await b.newPage();
  await p.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }, { name: "prefers-reduced-motion", value: "reduce" }]);
  await p.setViewport({ width: 1200, height: 630 });
  await p.goto(`http://127.0.0.1:${port}/style.css`);
  await p.setContent(`<!doctype html><html><head><base href="http://127.0.0.1:${port}/"><link rel="stylesheet" href="style.css"><style>
    body { width: 1200px; height: 630px; overflow: hidden; display: grid; grid-template-columns: 410px 1fr; align-items: center; gap: 8px; padding: 0 28px 0 52px; box-sizing: border-box; }
    h1 { font-size: 88px; margin: 0 0 24px; }
    .say { font-size: 30px; line-height: 1.18; font-weight: 600; letter-spacing: -.01em; margin: 0; }
    .chart text.lab, .chart .leader, .chart text.axis-title { display: none; }
    .chart text { font-size: 15px; }
  </style></head><body><div><h1>Vocal Frontier</h1><p class="say">Up to $1000 for faster, cleaner vocal separation</p></div><div id="chart" class="chart"></div></body></html>`, { waitUntil: "load" });
  await p.evaluate(async () => {
    const { drawBoard } = await import("/chart.js");
    const board = await (await fetch("/data/board.json")).json();
    await document.fonts.load('800 88px "Bricolage"');
    await document.fonts.ready;
    drawBoard(document.getElementById("chart"), document.createElement("div"), board);
  });
  await new Promise((r) => setTimeout(r, 500));
  await p.screenshot({ path: path.join(root, "site/og.png") });
  console.log("wrote site/og.png");
} finally {
  await b.close();
  srv.kill();
}
