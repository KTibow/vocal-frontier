"""The score transfer rule from the README. A file with the same weights as a scored native file inherits its score if, through the file's own
pipeline, every clip of a fixed set is within CLIP_DB of the native result and the set's mean within MEAN_DB. Contestants have no private30, so
they get a verdict on the public clips, and maintainers finish it. Per-clip scores of private clips are printed but never written to the repo."""
import hashlib
import json
import os
import re
import subprocess
import time

from . import guard
from . import manifest as M
from .audio import load
from .demix import Separator
from .score import sdr

CLIP_DB, MEAN_DB = 0.05, 0.02
CLIPS = {"musdb30": ["PR_-_Happy_Daze", "Motor_Tapes_-_Shore"], "cc30": ["c40", "c17"], "private30": ["s07", "s27", "s38", "s41"]}
COMPLETE = sum(len(c) for c in CLIPS.values())


def available_clips():
    return {s: c for s, c in CLIPS.items() if all(os.path.exists(os.path.join(M.DATA, "sets", s, x, "mixture.flac")) for x in c)}


def judge(parent, variant):
    """{clip: dB} twice. 'pass' needs all the clips; passing only those it has is 'public-pass', which no board row can inherit from."""
    delta = {k: variant[k] - parent[k] for k in parent}
    worst, mean = max(abs(d) for d in delta.values()), sum(delta.values()) / len(delta)
    ok = worst <= CLIP_DB and abs(mean) <= MEAN_DB
    return {"verdict": "fail" if not ok else "pass" if len(delta) >= COMPLETE else "public-pass", "max_abs_delta": round(worst, 4), "mean_delta": round(mean, 4)}, delta


def file_id(m, key):
    """What a file is, for matching. On this machine: the model it holds (graph and weights), so a file rejoined from the site's split
    copy matches the original. Otherwise the manifest's sha256 when it has one, else None."""
    f = m["files"][key]
    p = M.path_of(m, key)
    if not os.path.exists(p):
        return f.get("sha256")
    try:
        return M.model_id(p)
    except Exception:          # not a readable ONNX file: fall back to its bytes
        pass
    d, base = os.path.split(p)
    side = sorted(x for x in os.listdir(d) if x.startswith(base) and x != base)
    return M.sha256(p) if not side else hashlib.sha256("".join(M.sha256(os.path.join(d, x)) for x in [base] + side).encode()).hexdigest()


def engine_id(m, recipe):
    """A hash of the engine's entry file and every file in its folder that it names in quotes, followed through what those name, or None for the
    stock worker. Two entries running the same code from folders with different neighbours get the same hash."""
    if not recipe.get("engine"):
        return None
    top = os.path.join(m["dir"], str(recipe["engine"]).split("/")[0])
    files = sorted(os.path.relpath(os.path.join(r, f), top) for r, _, fs in os.walk(top) for f in fs)
    entry = os.path.relpath(os.path.join(m["dir"], str(recipe["engine"])), top)
    used, todo = set(), [entry]
    while todo:
        f = todo.pop()
        if f in used:
            continue
        used.add(f)
        with open(os.path.join(top, f), "rb") as fh:
            text = fh.read()
        todo += [g for g in files if g not in used and re.search(rb"[\"'`/]" + re.escape(os.path.basename(g).encode()) + rb"[\"'`?]", text)]   # a quoted name, not a comment
    h = hashlib.sha256()
    for f in sorted(used):
        h.update((f if f != entry else "").encode() + b"\0")        # the entry's own name doesn't matter; the names it loads do
        with open(os.path.join(top, f), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def reference_key(m, backend, parent):
    """The parent's file a variant is compared with: its native file, or the lower-quality file it's scored at when the variant runs that one."""
    return parent["scored_at"] if parent.get("scored_at") and m[backend].get("file") == parent["scored_at"] else parent["native"]["file"]


def identity(m, backend, parent):
    """What a verdict is about. A recorded pass only stands while all three are unchanged."""
    return {"file": file_id(m, m[backend]["file"]), "engine": engine_id(m, m[backend]), "against": file_id(parent, reference_key(m, backend, parent))}


def still_matches(rec, m, backend, parent):
    """A record from before a file, engine or parent file changed doesn't count. Parts not on this machine can't be compared, and the maintainers' machine has them all."""
    if not rec.get("files"):
        return False
    now = identity(m, backend, parent)
    return all(rec["files"].get(k) == v for k, v in now.items() if v is not None or k == "engine")


def record(name, backend, parent_name, runtime, overlap, parent, variant, files=None):
    res, delta = judge(parent, variant)
    public = {k: {"parent": parent[k], "variant": variant[k], "delta": round(delta[k], 4)} for k in delta if not k.startswith("private30")}
    private = [abs(d) for k, d in delta.items() if k.startswith("private30")]
    out = {"variant": name, "backend": backend, "parent": parent_name, "parent_runtime": runtime, "overlap": overlap, **res, "date": time.strftime("%Y-%m-%d"), "files": files, "clips": public,
           "private30_clips": {"count": len(private), "max_abs_delta": round(max(private), 4)} if private else {"count": 0}}
    d = os.path.join(M.ROOT, "results", "transfer")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.{backend}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def native_scores(m, clips, file=None):
    n = m["native"]
    sep = Separator(M.path_of(m, file or n["file"]), m["layout"], m["overlap"], n.get("runtime", "onnxruntime"), n.get("processes", 1), n.get("threads", 4), guard.physical_cores())
    res = {}
    for s, cl in clips.items():
        for c in cl:
            d = os.path.join(M.DATA, "sets", s, c)
            res[f"{s}/{c}"] = round(float(sdr(load(os.path.join(d, "vocals.flac")), sep(load(os.path.join(d, "mixture.flac"))))), 4)
    sep.close()
    return res


def _progress_file(name, backend):
    """Where a page run keeps finished clips, so a stopped run can continue. Kept outside the repo because it holds private30's per-clip scores.
    Named after the built entry and site code: change either and the run starts over, so one result never mixes two versions."""
    h = hashlib.sha256()
    preview = os.path.join(M.ROOT, ".preview")
    code = [os.path.join(preview, "models.json")]
    for d in ("src", "e"):
        code += sorted(os.path.join(r, f) for r, _, fs in os.walk(os.path.join(preview, d)) for f in fs)
    for p in code:
        with open(p, "rb") as f:
            h.update(f.read())
    d = os.path.join(os.path.expanduser("~"), ".cache", "vf-score-progress")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{name}.{backend}.{h.hexdigest()[:16]}.json")


def page_scores(name, backend, clips, port=8795):
    """Runs clips through the real site code on a local build of this entry: its own engine, or the stock worker."""
    from .build import build
    build(".preview", chunked=False, only=[M.REFERENCE, name])
    srv = subprocess.Popen(["node", os.path.join(M.ROOT, "tools", "serve.mjs"), os.path.join(M.ROOT, ".preview"), str(port)], stdout=subprocess.DEVNULL)
    try:
        out = _progress_file(name, backend)
        cpus = ",".join(map(str, guard.physical_cores()))
        subprocess.run(["taskset", "-c", cpus, "node", os.path.join(M.ROOT, "tools", "score_page.mjs"), f"http://127.0.0.1:{port}/", name, backend, os.path.join(M.DATA, "sets"), out, json.dumps(clips)], check=True)
        with open(out) as f:
            got = json.load(f)
        os.remove(out)
        return {f"{s}/{c}": got[f"{s}/{c}"] for s, cl in clips.items() for c in cl}
    finally:
        srv.terminate()


def run(name, parent_name=None, backends=None):
    guard.lock_and_wait(f"vf transfer {name}")
    m, parent = M.load(name), M.load(parent_name or name)
    if "native" not in parent:
        raise SystemExit(f"{parent['name']} has no native recipe to compare with")
    clips = available_clips()
    if not clips:
        raise SystemExit(f"None of the check's clips are in {os.path.join(M.DATA, 'sets')}. Run: vf fetch-data")
    missing = [s for s in CLIPS if s not in clips]
    if missing:
        print(f"No {', '.join(missing)} here: this gives a verdict on the public clips, and the maintainers finish it on the rest.")
    print(f"Scoring {parent['name']} natively at overlap {parent['overlap']} on {sum(map(len, clips.values()))} clips", flush=True)
    refs = {}
    verdicts = []
    for be in backends or [b for b in ("browser", "webgpu") if b in m]:
        print(f"{name} on {be}", flush=True)
        # An entry scored at one of its lower-quality files (`scored_at`) is checked against that file run natively, because that's the score it claims.
        target = reference_key(m, be, parent)
        if target not in refs:
            refs[target] = native_scores(parent, clips, target)
        ref = refs[target]
        var = page_scores(name, be, clips)
        out = record(name, be, parent["name"], parent["native"].get("runtime", "onnxruntime"), parent["overlap"], ref, var, identity(m, be, parent))
        print(f"  {'clip':32} {'native':>8} {'this':>8} {'diff':>7}")
        for k in ref:
            print(f"  {k:32} {ref[k]:8.3f} {var[k]:8.3f} {var[k] - ref[k]:+7.3f}")
        print(f"{name} on {be}: {out['verdict'].upper()}, worst clip {out['max_abs_delta']:.3f} dB, mean {out['mean_delta']:+.3f} dB (limits {CLIP_DB}, {MEAN_DB})", flush=True)
        verdicts.append(out["verdict"])
    return verdicts
