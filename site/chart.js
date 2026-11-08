// The frontier chart: log2(S) across, Q up, area under the staircase by who added it. Plain SVG, no library.
const M = { l: 54, r: 18, t: 14, b: 50 };
const color = (o) => (o === "baseline" ? "var(--s-others)" : o === "maintainers" ? "var(--s-ours)" : "var(--s-you)");
const NS = "http://www.w3.org/2000/svg";
export const BESIDE = "(min-width: 1100px)";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const el = (n, a = {}, parent) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); parent?.appendChild(e); return e; };
const rank = (p) => (p.owner === "baseline" ? 0 : p.owner === "maintainers" ? 1 : 2);

function stairPath(points, X, Y, x0, y0) {
  const pts = points.filter((p) => p.q > y0 && p.s > x0).sort((a, b) => a.s - b.s);
  const front = pts.filter((p, i) => !pts.some((o, j) => j > i && o.q >= p.q));
  if (!front.length) return null;
  let d = `M${X(x0)},${Y(y0)}`, top = null;
  for (const p of front) { d += `V${Y(p.q)}H${X(p.s)}`; top = p; }
  return d + `V${Y(y0)}Z`;
}

// Label placement. Every label tries positions all around its point at growing distances, and a score picks the set with the fewest collisions,
// checked against all markers (labelled or not), every other label, the leader lines, the frontier line and the chart's own text.
const hit = (a, b, pad = 0) => a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad;
const cross = (a, b, c, d) => {
  const o = (p, q, r) => (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
  return o(a, b, c) * o(a, b, d) < 0 && o(c, d, a) * o(c, d, b) < 0;
};
const segHitsBox = (p, q, r) => {
  if (Math.max(p.x, q.x) < r.x || Math.min(p.x, q.x) > r.x + r.w || Math.max(p.y, q.y) < r.y || Math.min(p.y, q.y) > r.y + r.h) return false;
  if ((p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) || (q.x >= r.x && q.x <= r.x + r.w && q.y >= r.y && q.y <= r.y + r.h)) return true;
  const c = [{ x: r.x, y: r.y }, { x: r.x + r.w, y: r.y }, { x: r.x + r.w, y: r.y + r.h }, { x: r.x, y: r.y + r.h }];
  return c.some((v, i) => cross(p, q, v, c[(i + 1) % 4]));
};
const nearestOnBox = (p, r) => ({ x: Math.max(r.x, Math.min(p.x, r.x + r.w)), y: Math.max(r.y, Math.min(p.y, r.y + r.h)) });

function placeLabels(labels, markers, obstacles, lines, plot) {
  const RADII = [10, 15, 22, 32, 46, 64, 88, 116], ANGLES = 16;
  const cands = labels.map((l) => {
    const out = [];
    for (const r of RADII) for (let k = 0; k < ANGLES; k++) {
      const t = k / ANGLES * 2 * Math.PI, c = Math.cos(t), s = Math.sin(t);
      // slide the box out along this direction until its nearest edge is exactly r away, so r really is the distance to the point
      const at = (t) => ({ x: l.px + c * t - l.w / 2, y: l.py + s * t - l.h / 2, w: l.w, h: l.h });
      const apart = (b) => { const q = nearestOnBox({ x: l.px, y: l.py }, b); return Math.hypot(q.x - l.px, q.y - l.py); };
      let lo = 0, hi = r + Math.abs(c) * l.w / 2 + Math.abs(s) * l.h / 2;
      for (let n = 0; n < 12; n++) { const m = (lo + hi) / 2; if (apart(at(m)) < r) lo = m; else hi = m; }
      const box = at(hi);
      if (box.x < plot.x0 || box.x + box.w > plot.x1 || box.y < plot.y0 || box.y + box.h > plot.y1) continue;
      const end = nearestOnBox({ x: l.px, y: l.py }, box), from = { x: l.px, y: l.py };
      let cost = r * 2 + (1 - c) * 3 + (s > 0 ? 2 : 0);                     // close first, then a slight preference for up and to the right
      // A label must read as its own point's: if another point is nearer to it than its own, someone will take it for that one.
      const gap = (m) => { const q = nearestOnBox(m, box); return Math.hypot(q.x - m.x, q.y - m.y); };
      const own = gap({ x: l.px, y: l.py });
      for (const m of markers) if (m !== l.marker && gap(m) < own + 4) cost += 2500;
      for (const m of markers) {
        if (m !== l.marker && hit(box, { x: m.x - 8, y: m.y - 8, w: 16, h: 16 })) cost += 20000;
        if (m !== l.marker && segHitsBox(from, end, { x: m.x - 7, y: m.y - 7, w: 14, h: 14 })) cost += 6000;
      }
      for (const o of obstacles) if (hit(box, o, 2)) cost += 20000;
      for (const ln of lines) if (segHitsBox(ln.a, ln.b, box)) cost += ln.cost;
      out.push({ box, from, end, cost, long: r > 15 });
    }
    return out;
  });
  const order = (a, b) => {             // neighbours keep their left-to-right order when their labels share a side, which is how people read them
    const same = (a.box.y + a.box.h / 2 < a.from.y) === (b.box.y + b.box.h / 2 < b.from.y);
    return same && (a.from.x - b.from.x) * (a.box.x + a.box.w / 2 - b.box.x - b.box.w / 2) < 0 ? 1200 : 0;
  };
  const pair = (a, b) => order(a, b) + (hit(a.box, b.box, 3) ? 30000 : 0) + (a.long && segHitsBox(a.from, a.end, b.box) ? 6000 : 0) + (b.long && segHitsBox(b.from, b.end, a.box) ? 6000 : 0) + (a.long && b.long && cross(a.from, a.end, b.from, b.end) ? 3000 : 0);
  let seed = 7; const rnd = () => (seed = (seed * 48271) % 2147483647) / 2147483647;     // fixed, so the same data always draws the same way
  let best = null, bestCost = Infinity;
  for (let restart = 0; restart < 14; restart++) {
    const pick = cands.map((c) => (restart ? c[Math.floor(rnd() * c.length)] : c.reduce((x, y) => (y.cost < x.cost ? y : x))));
    for (let pass = 0; pass < 8; pass++) {
      let moved = false;
      for (let i = 0; i < labels.length; i++) {
        let top = pick[i], topCost = Infinity;
        for (const c of cands[i]) {
          let v = c.cost;
          for (let j = 0; j < labels.length && v < topCost; j++) if (j !== i) v += pair(c, pick[j]);
          if (v < topCost) { topCost = v; top = c; }
        }
        if (top !== pick[i]) { pick[i] = top; moved = true; }
      }
      if (!moved) break;
    }
    let total = pick.reduce((s, c) => s + c.cost, 0);
    for (let i = 0; i < pick.length; i++) for (let j = i + 1; j < pick.length; j++) total += pair(pick[i], pick[j]);
    if (total < bestCost) { bestCost = total; best = pick.slice(); }
  }
  return best;
}

// On a phone there's no room for full names, and the table below has them.
const shortTitle = (t) => t.replace(/^UVR /, "").replace(/^Musetric duality/, "Duality").replace(/^Voc_FT at /, "").replace(/^HTDemucs-lite-/, "Lite-").replace(", no overlap", " o1").replace(", mixed", " mix").replace(", tuned", " tuned");

export function drawBoard(host, tip, b, onPick) {
  // Drawn at the host's real width, so text stays readable on a phone instead of shrinking with a fixed viewBox.
  const W = Math.max(320, Math.min(960, host.clientWidth || 960)), narrow = W < 600;
  // Beside the story it stays on screen, so it takes the window's height instead of a fixed one.
  const beside = matchMedia(BESIDE).matches, H = beside ? Math.max(420, Math.min(720, innerHeight - 104)) : narrow ? 600 : 560;
  host._ro?.disconnect();
  host._ro = new ResizeObserver(() => { if (Math.abs(Math.min(960, host.clientWidth) - W) > 8) drawBoard(host, tip, b, onPick); });
  host._ro.observe(host);
  if (!host._resize) { host._resize = true; addEventListener("resize", () => drawBoard(host, tip, b, onPick)); }   // the height follows the window too
  if (!host._fonts) { host._fonts = true; document.fonts?.ready.then(() => drawBoard(host, tip, b, onPick)); }   // labels are measured, so they need the real font
  host.textContent = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, host);
  const x0 = b.origin_s, pts = b.points.filter((p) => p.s >= x0);     // an entry slower than the origin stays in the table below
  const y0 = b.floor_q ?? Math.min(...pts.map((p) => p.q)) - .2;
  const smax = Math.max(...pts.map((p) => p.s)) * 1.25, qmax = Math.max(...pts.map((p) => p.q), y0 + 3) + .35;
  const X = (s) => M.l + (Math.log2(s) - Math.log2(x0)) / (Math.log2(smax) - Math.log2(x0)) * (W - M.l - M.r);
  const Y = (q) => H - M.b - (q - y0) / (qmax - y0) * (H - M.t - M.b);
  for (let q = Math.ceil(y0); q <= qmax; q++) {
    el("line", { x1: M.l, x2: W - M.r, y1: Y(q), y2: Y(q), stroke: "var(--grid)" }, svg);
    el("text", { x: M.l - 8, y: Y(q) + 4, "text-anchor": "end" }, svg).textContent = q;
  }
  for (const s of new Set((narrow ? [x0, 0.25, 1, 4, 16, 64] : [x0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]).filter((s) => s >= x0 && s <= smax))) {
    el("line", { x1: X(s), x2: X(s), y1: M.t, y2: H - M.b, stroke: "var(--grid)" }, svg);
    el("text", { x: X(s), y: H - M.b + 18, "text-anchor": "middle" }, svg).textContent = s + "×";
  }
  el("text", { class: "axis-title", x: (M.l + W - M.r) / 2, y: H - 8, "text-anchor": "middle" }, svg).textContent = "Speed index, times faster than HTDemucs →";
  el("text", { class: "axis-title", transform: `translate(14,${(M.t + H - M.b) / 2}) rotate(-90)`, "text-anchor": "middle" }, svg).textContent = "Quality, mean SDR in dB →";
  // area: draw the biggest coalition first, smaller ones on top, so each band shows what its owner added
  const under = el("g", {}, svg);        // a scene's extra area goes under the washes, so only what's new shows
  const layers = [[(p) => true, "var(--s-you)"], [(p) => rank(p) <= 1, "var(--s-ours)"], [(p) => rank(p) === 0, "var(--s-others)"]];
  for (const [f, c] of layers) {
    const d = stairPath(pts.filter(f), X, Y, x0, y0);
    if (d) {
      el("path", { d, fill: "var(--surface)" }, svg);       // solid under each wash, so washes don't mix into a new color
      el("path", { d, fill: c, "fill-opacity": .14 }, svg);
    }
  }
  const front = stairPath(pts, X, Y, x0, y0);
  if (front) el("path", { d: front, fill: "none", stroke: "var(--ink3)", "stroke-width": 1 }, svg);
  el("line", { x1: X(1), x2: X(1), y1: M.t, y2: H - M.b, stroke: "var(--ink3)" }, svg);
  const fixed = [];
  const note = (a, text) => { const t = el("text", { ...a, class: "axis-title" }, svg); t.textContent = text; fixed.push(t); };
  note({ x: X(1) + 6, y: M.t + 12 }, "HTDemucs speed");
  if (b.floor_q != null) note({ x: M.l + 8, y: Y(y0) - 8 }, narrow ? "Quality floor: 10.0 dB" : "Quality floor: 10.0 dB, 1 dB under HTDemucs");
  const frontier = new Set(b.frontier), toLabel = [];
  for (const p of [...pts].sort((a, b) => frontier.has(a.name) - frontier.has(b.name))) {
    const g = el("g", { class: "pt", "data-name": p.name, tabindex: 0, "aria-label": `${p.title}: ${p.s.toFixed(2)} times, ${p.q.toFixed(2)} dB` }, svg);
    const x = X(p.s), y = Y(p.q), c = color(p.owner);
    el("circle", { cx: x, cy: y, r: 14, fill: "transparent" }, g);
    el("circle", { class: "m", cx: x, cy: y, r: 6, fill: p.provisional ? "var(--surface)" : c, stroke: p.provisional ? c : "var(--surface)", "stroke-width": 2 }, g);
    const show = (e) => {
      tip.hidden = false;
      const dev = Object.entries(p.devices).map(([d, v]) => `${d} ${v.ratio.toFixed(2)}×`).join(" · ");
      tip.innerHTML = `<b>${esc(p.title)}</b>${p.description ? `<span class="tip-desc">${esc(p.description)}</span>` : ""}<code>${p.s.toFixed(2)}× · ${p.q.toFixed(2)} dB${p.owner === "baseline" || p.owner === "maintainers" ? "" : ` · ${p.gain.toFixed(1)} units added`}</code><br>${dev}`;
      const r = host.getBoundingClientRect(), pr = host.parentElement.getBoundingClientRect(), k = r.width / W;
      const left = r.left - pr.left + x * k + 12, top = r.top - pr.top + y * k + 12;
      tip.style.left = Math.min(left, pr.width - 290) + "px"; tip.style.top = top + "px";
    };
    const hide = () => { tip.hidden = true; };
    g.addEventListener("pointerenter", show); g.addEventListener("pointerleave", hide); g.addEventListener("focus", show); g.addEventListener("blur", hide);
    g.addEventListener("click", () => onPick?.(p));
    if (frontier.has(p.name) || p.name === "htdemucs" || (p.owner !== "baseline" && p.owner !== "maintainers")) toLabel.push(p);   // the frontier, the 1× reference, contestants. The rest are in the tooltip and table
  }
  const markers = pts.map((p) => ({ x: X(p.s), y: Y(p.q), name: p.name }));
  const labels = toLabel.map((p) => {
    const t = el("text", { class: "lab", "data-name": p.name, x: 0, y: 0 }, svg);
    t.textContent = narrow ? shortTitle(p.title) : p.title;
    const bb = t.getBBox();
    return { t, name: p.name, w: bb.width, h: bb.height, dy: -bb.y, px: X(p.s), py: Y(p.q), marker: markers.find((m) => m.name === p.name) };
  });
  const sorted = pts.filter((p) => p.q > y0 && p.s > x0).sort((a, c) => a.s - c.s);
  const stair = sorted.filter((p, i) => !sorted.some((o, j) => j > i && o.q >= p.q));
  const lines = [{ a: { x: X(1), y: M.t }, b: { x: X(1), y: H - M.b }, cost: 150 }];
  let cur = { x: X(x0), y: Y(y0) };
  for (const p of stair) {
    const up = { x: cur.x, y: Y(p.q) }, over = { x: X(p.s), y: Y(p.q) };
    lines.push({ a: cur, b: up, cost: 400 }, { a: up, b: over, cost: 400 });
    cur = over;
  }
  lines.push({ a: cur, b: { x: cur.x, y: Y(y0) }, cost: 400 });
  const plot = { x0: M.l + 2, x1: W - M.r - 2, y0: M.t + 2, y1: H - M.b - 2 };
  const obstacles = fixed.map((t) => { const r = t.getBBox(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  const chosen = placeLabels(labels, markers, obstacles, lines, plot);
  labels.forEach((l, i) => {
    const c = chosen[i];
    l.t.setAttribute("x", c.box.x); l.t.setAttribute("y", c.box.y + l.dy);
    if (c.long) {       // far from its point, so draw a leader. It starts at the marker's edge
      const dx = c.end.x - l.px, dy = c.end.y - l.py, n = Math.hypot(dx, dy) || 1;
      svg.insertBefore(el("line", { class: "leader", "data-name": l.name, x1: l.px + dx / n * 7, y1: l.py + dy / n * 7, x2: c.end.x, y2: c.end.y }), l.t);
    }
  });
  host._stage = { svg, under, over: el("g", {}, svg), X, Y, x0, y0, smax, W, H, pts, b, refq: b.reference_q };
  applyScene(host);
  host._labelCheck = { labels: chosen.map((c, i) => ({ text: labels[i].t.textContent, box: c.box, long: c.long })), markers, obstacles };
}

// What the chart shows beside each part of the story. Only ever adds to the board: every scene's numbers come from b.
export const EXAMPLE = { s: 4, q: 12 };
export function setScene(host, name) { host._scene = name || ""; applyScene(host); }
function applyScene(host) {
  const st = host._stage, name = host._scene || "";
  if (!st) return;
  const { svg, under, over, X, Y, x0, y0, smax, W, pts, b, refq } = st, top = M.t, bottom = st.H - M.b;
  under.textContent = over.textContent = "";
  svg.querySelectorAll(".hi").forEach((e) => e.classList.remove("hi"));
  if (name) svg.dataset.scene = name; else delete svg.dataset.scene;
  const hi = (names) => names.forEach((n) => svg.querySelectorAll(`[data-name="${n}"]`).forEach((e) => e.classList.add("hi")));
  const text = (a, t, parent = over) => { el("text", { class: "lab scene-note", ...a }, parent).textContent = t; };
  const rule = (a) => el("line", { class: "scene-rule", ...a }, over);
  if (name === "speed") {
    hi(["htdemucs"]);
    rule({ x1: X(1), x2: X(1), y1: top, y2: bottom });
    const y = (top + bottom) / 2;
    text({ x: X(1) - 10, y, "text-anchor": "end" }, "← slower than HTDemucs");
    text({ x: X(1) + 10, y }, "faster →");
  } else if (name === "quality") {
    hi(["htdemucs"]);
    svg.querySelector('text.lab[data-name="htdemucs"]')?.classList.remove("hi");     // the rule's own label says it, and the rule runs through this one
    rule({ x1: M.l, x2: W - M.r, y1: Y(refq), y2: Y(refq) });
    text({ x: W - M.r - 8, y: Y(refq) - 8, "text-anchor": "end" }, `HTDemucs: ${refq.toFixed(1)} dB`);
    rule({ x1: M.l, x2: W - M.r, y1: Y(y0), y2: Y(y0) });
  } else if (name === "money") {
    const me = { ...EXAMPLE };
    const d = stairPath([...pts, me], X, Y, x0, y0);
    if (d) el("path", { d, fill: "var(--s-you)", "fill-opacity": .55 }, under);
    el("circle", { cx: X(me.s), cy: Y(me.q), r: 7, fill: "var(--s-you)", stroke: "var(--surface)", "stroke-width": 2 }, over);
    text({ x: X(me.s) + 4, y: Y(me.q) - 13, "text-anchor": "end" }, `Example: ${me.s}×, ${me.q} dB`);
  } else if (name === "frontier") {
    hi(b.frontier);
  }
}

export const LEGEND = [["Others' models, stock and tuned", "var(--s-others)"], ["Maintainers'", "var(--s-ours)"], ["Contestants", "var(--s-you)"]];
export { color };
