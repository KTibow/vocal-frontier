"""Checks a submission against the rules before anyone runs it. Only YAML and ONNX go through here. External-data paths are
checked, because an ONNX file can otherwise point at any file on the machine that loads it. Submissions can't use them at all:
a manifest has one url per file, so upload plain ONNX and vf/build.py splits it for the browser."""
import os
import re

from . import manifest as M

PLACEHOLDER_WALLET = "0x" + "0" * 40
ENGINE_FILES = {".js", ".mjs", ".wasm", ".wgsl", ".json", ".md", ".txt"}
MAX_ASSET = 24 * 1024 * 1024

LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0", "CC0-1.0", "CC-BY-4.0", "Unlicense", "Zlib"}
DOMAINS = {"", "ai.onnx", "ai.onnx.ml", "com.microsoft"}
RUNTIMES = {"onnxruntime", "openvino"}
WEB = {"wasm@1.23.0", "wasm@1.30.0", "webgpu@1.23.0", "webgpu@1.30.0"}
LAYOUTS = {"mdx": 1, "htdemucs": 2, "mask": 1}


def validate(m):
    bad, warn = [], []
    contest = m["group"] == "submissions"
    if contest:
        if m.get("kind") != "contestant":
            bad.append("kind: must be contestant")
        if not re.fullmatch(r"[A-Za-z0-9-]{1,39}", str(m.get("github", ""))) or m.get("github") == "your-github-login":
            bad.append("github: your GitHub login")
        if not re.fullmatch(r"0x[0-9a-fA-F]{40}", str(m.get("wallet", ""))) or m.get("wallet") == PLACEHOLDER_WALLET:
            bad.append("wallet: your own 0x address on Base, for USDC. Money sent to the placeholder is gone")
        for key, f in (m.get("files") or {}).items():
            if not str(f.get("path", "")).startswith(m["name"] + "/"):
                bad.append(f"files.{key}: path has to start with {m['name']}/, so it can't land on another entry's file")
            if not f.get("url") or not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))):
                bad.append(f"files.{key}: needs a url and a sha256 (`sha256sum` on the file gives it), so the file can't change after you take your place in line")
            elif "huggingface.co/you/" in str(f["url"]):
                bad.append(f"files.{key}: url is still the template's placeholder. Upload the file and put its real url here")
            elif os.path.exists(os.path.join(M.MODELS, str(f.get("path", "")))) and M.sha256(os.path.join(M.MODELS, f["path"])) != f["sha256"]:
                bad.append(f"files.{key}: sha256 doesn't match {f['path']} here. Run `sha256sum` on the file you uploaded")
        lic = m.get("license") or {}
        for k in ("weights", "code"):
            if lic.get(k) not in LICENSES:
                bad.append(f"license.{k}: {lic.get(k)!r} isn't on the list ({', '.join(sorted(LICENSES))}). If it's derived from weights with no stated license, write that under license.upstream.")
        tr = os.path.join(m["dir"], "training.txt")
        lines = [l for l in open(tr).read().splitlines() if l.strip() and not l.lstrip().startswith("#")] if os.path.exists(tr) else []
        if not lines:
            bad.append("training.txt: list every dataset, track or take you trained on, and any teacher model")
    lay = (m.get("layout") or {}).get("type")
    if lay not in LAYOUTS:
        bad.append(f"layout.type: {lay!r}. Have {sorted(LAYOUTS)}. A new layout goes in vf/hosts.py, site/src/layouts.js and vf/validate.py, and maintainers review it")
    if not (1 <= float(m.get("overlap", 0)) <= 8):
        bad.append("overlap: between 1 and 8")
    for name, r in (("native", m.get("native")), ("browser", m.get("browser")), ("webgpu", m.get("webgpu"))):
        if not r:
            warn.append(f"no {name} recipe: {'that device' if name != 'webgpu' else 'all four GPUs'} will score 0.05×")
            continue
        if r.get("file") not in (m.get("files") or {}):
            bad.append(f"{name}.file: {r.get('file')!r} isn't in files")
        if name == "native":
            if r.get("runtime") not in RUNTIMES:
                bad.append(f"native.runtime: one of {sorted(RUNTIMES)}")
            if r.get("processes", 1) * r.get("threads", 4) > 4:
                bad.append("native: processes x threads can't exceed the 4 cores it's measured on")
        else:
            if r.get("engine"):
                bad += _check_engine(m, name, r["engine"])
            elif r.get("runtime") not in WEB:
                bad.append(f"{name}.runtime: one of {sorted(WEB)}, or bring your own engine")
            if name == "browser" and r.get("workers", 1) * r.get("threads", 1) > 4:
                bad.append("browser: workers x threads can't exceed 4")
    for key, f in (m.get("files") or {}).items():
        p = os.path.join(M.MODELS, f["path"])
        if os.path.relpath(os.path.realpath(p), os.path.realpath(M.MODELS)).startswith(".."):
            bad.append(f"files.{key}: path leaves the models directory")
            continue
        if not os.path.exists(p):
            bad.append(f"files.{key}: {p} doesn't exist")
            continue
        if not p.endswith(".onnx"):                # an engine can load its own format; native can't
            if (m.get("native") or {}).get("file") == key:
                bad.append(f"files.{key}: native runs ONNX only")
            continue
        bad += _check_onnx(p, key, lay, m)
    if contest and not bad:
        warn += _same_model_as(m)
    for b in bad:
        print(f"  ✗ {b}")
    for w in warn:
        print(f"  ! {w}")
    print(f"{m['name']}: {'FAILED' if bad else 'ok'}")
    return not bad


def _same_model_as(m):
    """A heads-up, not a failure: building on an entry is fine, but an unchanged model only earns if it's clearly faster."""
    from .transfer import file_id
    try:
        mine = file_id(m, m["native"]["file"])
    except Exception:
        return []
    out = []
    for n in M.names():
        if n == m["name"]:
            continue
        o = M.load(n)
        if "native" not in o:
            continue
        try:
            if file_id(o, o["native"]["file"]) == mine:
                out.append(f"your native file is the same model as {n}'s. That's allowed, but it only earns if it's more than 5% faster "
                           f"or 0.1 dB cleaner than {n} (README: copies earn nothing)")
        except Exception:
            pass
    return out


def _check_engine(m, name, engine):
    """An engine runs in a browser worker with no network (see vf/build.py). It has to be readable, and rebuildable if it's compiled."""
    bad = []
    top = str(engine).split("/")[0]
    entry = os.path.realpath(os.path.join(m["dir"], str(engine)))
    if not entry.startswith(os.path.realpath(os.path.join(m["dir"], top)) + os.sep) or not os.path.exists(entry):
        return [f"{name}.engine: {engine!r} has to be a file in a folder next to the manifest, like engine/worker.js"]
    wasm = False
    for dp, _, fs in os.walk(os.path.join(m["dir"], top)):
        for f in fs:
            ext = os.path.splitext(f)[1]
            wasm = wasm or ext == ".wasm"
            if ext not in ENGINE_FILES:
                bad.append(f"{name}.engine: {f}: only {', '.join(sorted(ENGINE_FILES))} can go in an engine")
            if os.path.getsize(os.path.join(dp, f)) > MAX_ASSET:
                bad.append(f"{name}.engine: {f} is over 24 MiB, the host's limit. Split it")
    if wasm and not os.path.exists(os.path.join(m["dir"], top, "SOURCE.md")):
        bad.append(f"{name}.engine: compiled code needs a SOURCE.md with its source and the commands that rebuild it")
    return bad


def _check_onnx(path, key, lay, m):
    import onnx
    bad = []
    mp = onnx.load(path, load_external_data=False)
    base = os.path.realpath(os.path.dirname(path))
    for t in onnx.external_data_helper._get_all_tensors(mp):
        if t.data_location == onnx.TensorProto.EXTERNAL:
            loc = next((e.value for e in t.external_data if e.key == "location"), "")
            if m["group"] == "submissions":
                bad.append(f"files.{key}: keep the weights in the ONNX file. A manifest has one url per file, and the site splits it for the browser anyway")
            elif os.path.isabs(loc) or not os.path.realpath(os.path.join(base, loc)).startswith(base + os.sep):
                bad.append(f"files.{key}: external data points outside the model's folder: {loc!r}")
    ops = {n.domain for n in mp.graph.node} | {o.domain for o in mp.opset_import}
    if ops - DOMAINS:
        bad.append(f"files.{key}: custom op domains aren't allowed: {sorted(ops - DOMAINS)}")
    inits = {i.name for i in mp.graph.initializer}
    inputs = [i.name for i in mp.graph.input if i.name not in inits]
    if lay and lay in LAYOUTS and len(inputs) != LAYOUTS[lay]:
        bad.append(f"files.{key}: layout {lay} takes {LAYOUTS[lay]} input(s); this graph has {inputs}")
    if lay == "mdx" and inputs:
        dims = [d.dim_value or None for d in next(i for i in mp.graph.input if i.name == inputs[0]).type.tensor_type.shape.dim]
        want = {2: ("dim_f", m["layout"].get("dim_f", 3072)), 3: ("dim_t", m["layout"].get("dim_t", 256))}
        for ax, (k, v) in want.items():
            if len(dims) == 4 and dims[ax] and dims[ax] != v:
                bad.append(f"files.{key}: layout.{k} is {v}, but this graph takes {dims[ax]}")
    if lay == "htdemucs" and set(inputs) != {"input", "x"}:
        bad.append(f'files.{key}: the htdemucs layout needs inputs "input" and "x"; found {inputs}')
    return bad
