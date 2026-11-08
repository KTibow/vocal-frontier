"""Builds the static site: pages, ONNX Runtime, the test clip, and the models, each ONNX file split into a graph and its weights,
and both cut into parts under Cloudflare's 25 MiB per-asset limit."""
import hashlib
import json
import os
import shutil

from . import manifest as M

PART = 24 * 1024 * 1024
HEADERS = """/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Resource-Policy: same-origin
/m/*
  Cache-Control: public, max-age=31536000, immutable
/ort/*
  Cache-Control: public, max-age=31536000, immutable
/e/*
  Content-Security-Policy: default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; worker-src 'self'
"""
WRANGLER = """name = "vocal-frontier"
account_id = "dbeb50c60a3a8939ca01c69a271f6159"
routes = [{ pattern = "vocal-frontier.kendell.dev", custom_domain = true }]
compatibility_date = "2026-10-01"

[assets]
directory = "."
"""


def _ort(out, versions):
    for v in versions:
        src = os.path.join(M.ROOT, "node_modules", f"onnxruntime-web-{'-'.join(v.split('.')[:2])}", "dist")
        d = os.path.join(out, "ort", v)
        os.makedirs(d, exist_ok=True)
        want = ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm",
                "ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]
        for f in want:
            shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
        # Newer WebGPU builds outgrow the per-asset limit. The worker reassembles the parts and hands the binary to ONNX Runtime.
        big = os.path.join(d, "ort-wasm-simd-threaded.asyncify.wasm")
        if os.path.getsize(big) > PART:
            parts = 0
            with open(big, "rb") as f:
                while b := f.read(PART):
                    with open(f"{big}.{parts}", "wb") as o:
                        o.write(b)
                    parts += 1
            os.remove(big)
            with open(os.path.join(d, "webgpu-wasm-parts.json"), "w") as f:
                json.dump({"parts": parts}, f)


def _file(m, key, out, chunked, done):
    """What the browser downloads for one manifest file. ONNX is always split: weights embedded in the graph pass through
    ONNX Runtime's WASM heap, which grows to fit and never shrinks. Passed beside it, they don't. Other formats go as they are."""
    f = m["files"][key]
    p = os.path.join(M.MODELS, f["path"])
    pinned = M.sha256(p)
    if f.get("sha256") and f["sha256"] != pinned:      # the manifest pins what was uploaded; what the site serves is derived from it, and pinned below
        raise SystemExit(f"{m['name']}: files.{key} is {pinned}, and the manifest says {f['sha256']}")
    if p not in done:
        done[p] = _split(p, out) if p.endswith(".onnx") else _whole(p, f["path"], out, chunked)
    return done[p]


def _split(path, out, model=None):
    """Same graph and same weight bytes, rewritten as graph + one weights file. Tensors under 1 KiB stay in the graph (shapes, axes,
    scalars: ONNX Runtime wants those at hand), so a tiny model just gets a small second file or none. Deterministic, so a rebuild
    gives the same hashes. Also takes a source that already keeps its weights in a file beside it, which onnx.load reads in."""
    import onnx
    from onnx.external_data_helper import set_external_data
    m = model if model is not None else onnx.load(path)
    name = os.path.basename(path) + ".data"
    tmp = os.path.join(out, ".split")
    h, size = hashlib.sha256(), 0
    with open(tmp, "wb") as w:
        for t in m.graph.initializer:
            if t.HasField("raw_data") and len(t.raw_data) >= 1024:
                b = t.raw_data
                set_external_data(t, location=name, offset=size, length=len(b))
                t.ClearField("raw_data")
                t.data_location = onnx.TensorProto.EXTERNAL
                w.write(b); h.update(b); size += len(b)
    graph = m.SerializeToString(deterministic=True)
    del m
    e = {**_chunks(graph, hashlib.sha256(graph).hexdigest(), out)}
    if size:
        e["data"] = {"sha256": h.hexdigest(), "bytes": size, "name": name, "parts": _parts(tmp, h.hexdigest(), out)}
    os.remove(tmp)
    return e


def _tiled(m, key, out, done):
    """The same GPU file cut into time pieces, and the binding limit under which the page uses it instead (vf/tile.py)."""
    from . import tile
    p = os.path.join(M.MODELS, m["files"][key]["path"])
    k = p + "#tiled"
    if k not in done:
        t = tile.for_gpu(p) if p.endswith(".onnx") else None
        done[k] = t and {"file": _split(p, out, t[0]), "below": int(t[1] * tile.HEADROOM)}
        if t:
            print(f"{m['name']}: webgpu file's largest tensor is {t[1] / 2**20:.0f} MiB, so GPUs under {t[1] * tile.HEADROOM / 2**20:.0f} MiB per binding get a copy in time pieces ({t[2] / 2**20:.0f} MiB)")
    return done[k]


def _chunks(b, h, out):
    d = os.path.join(out, "m", h[:16])
    os.makedirs(d, exist_ok=True)
    parts = []
    for i in range(0, max(len(b), 1), PART):
        with open(os.path.join(d, f"{i // PART:03d}.bin"), "wb") as o:
            o.write(b[i:i + PART])
        parts.append(f"m/{h[:16]}/{i // PART:03d}.bin")
    return {"sha256": h, "bytes": len(b), "parts": parts}


def _parts(path, h, out):
    d = os.path.join(out, "m", h[:16])
    os.makedirs(d, exist_ok=True)
    parts = []
    with open(path, "rb") as f:
        while b := f.read(PART):
            with open(os.path.join(d, f"{len(parts):03d}.bin"), "wb") as o:
                o.write(b)
            parts.append(f"m/{h[:16]}/{len(parts):03d}.bin")
    return parts


def _whole(p, rel, out, chunked):
    size, h = os.path.getsize(p), M.sha256(p)
    return {"sha256": h, "bytes": size, "parts": _parts(p, h, out) if chunked else [f"local-models/{rel}"]}


def build(out, chunked=True, only=None):
    out = os.path.join(M.ROOT, out)
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(os.path.join(M.ROOT, "site"), out)
    models, versions, done = {}, set(), {}
    for n in M.names():
        if only and n not in only:
            continue
        m = M.load(n)
        e = {"name": n, "title": m.get("title", n), "kind": m.get("kind"), "role": m.get("role"), "layout": m["layout"], "overlap": m["overlap"]}
        for backend in ("browser", "webgpu"):
            r = m.get(backend)
            if not r:
                continue
            path = M.path_of(m, r["file"])
            if not os.path.exists(path):
                print(f"skipping {n} {backend}: {path} isn't here")
                continue
            e[backend] = {"runtime": r.get("runtime"), "workers": r.get("workers", 1), "threads": r.get("threads", 1), "file": _file(m, r["file"], out, chunked, done)}
            if backend == "webgpu" and (t := _tiled(m, r["file"], out, done)):
                e[backend]["tiled"] = t
            if r.get("runtime"):
                versions.add(r["runtime"].split("@")[1])
            if r.get("engine"):                   # served under /e/, where the headers above keep it off the network
                src, dst = m["dir"], os.path.join(out, "e", n)
                top = r["engine"].split("/")[0]
                if not os.path.exists(os.path.join(dst, top)):
                    shutil.copytree(os.path.join(src, top), os.path.join(dst, top))
                e[backend]["engine"] = f"e/{n}/{r['engine']}"
        models[n] = e
    _ort(out, sorted(versions))
    with open(os.path.join(out, "models.json"), "w") as f:
        json.dump({"models": models, "reference": M.REFERENCE}, f, indent=1)
    with open(os.path.join(out, "_headers"), "w") as f:
        f.write(HEADERS)
    with open(os.path.join(out, "wrangler.toml"), "w") as f:
        f.write(WRANGLER)
    with open(os.path.join(out, ".assetsignore"), "w") as f:
        f.write("wrangler.toml\n.assetsignore\n")
    total = sum(os.path.getsize(os.path.join(dp, x)) for dp, _, fs in os.walk(out) for x in fs)
    files = sum(len(fs) for _, _, fs in os.walk(out))
    print(f"{out}: {files} files, {total / 1e6:.0f} MB")
