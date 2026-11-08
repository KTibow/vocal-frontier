// Local server for dist/. Sets the same headers as the Cloudflare site. Serves /local-models/ from $VF_MODELS for a --local build.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
const [dir = "dist", port = "8787"] = process.argv.slice(2);
const models = process.env.VF_MODELS || "models";
const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml", ".woff2": "font/woff2" };
http.createServer((req, res) => {
  const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp", "Cross-Origin-Resource-Policy": "same-origin" };
  let u = decodeURIComponent(req.url.split("?")[0]);
  if (u.startsWith("/e/")) headers["Content-Security-Policy"] = "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self'";
  if (u.endsWith("/")) u += "index.html";
  const file = u.startsWith("/local-models/") ? path.join(models, path.normalize(u.slice(14))) : path.join(dir, path.normalize(u));
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404, headers); return res.end("not found"); }
  res.writeHead(200, { ...headers, "Content-Type": types[path.extname(file)] || "application/octet-stream", "Content-Length": fs.statSync(file).size });
  fs.createReadStream(file).pipe(res);
}).listen(+port, "127.0.0.1", () => console.log(`http://127.0.0.1:${port}/ (${dir})`));
