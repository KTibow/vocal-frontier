"""vf: everything a contestant runs. `vf --help` lists it; the README walks through it."""
import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from . import guard
from . import index as I
from . import manifest as M

SETS = ("musdb30", "cc30", "private30")
PUBLIC_SETS = ("musdb30", "cc30")
CLIP_TOLERANCE_DB = 0.5


def native_sep(m, cpus=None):
    from .demix import Separator
    n = m["native"]
    return Separator(M.path_of(m, n["file"]), m["layout"], m["overlap"], n.get("runtime", "onnxruntime"),
                     n.get("processes", 1), n.get("threads", 4), cpus)


def cmd_list(a):
    b = board_data()
    for p in sorted(b["points"], key=lambda p: -p["q"]):
        print(f"{p['name']:24} Q {p['q']:6.2f} dB   S {p['s']:6.2f}x   {'provisional' if p['provisional'] else 'official'}   {p['owner']}")


def cmd_validate(a):
    from .validate import validate
    sys.exit(0 if validate(M.load(a.name)) else 1)


def cmd_quality(a):
    """Separate and score sets natively. Writes results/quality/<name>.json"""
    from .audio import load, save
    from .score import sdr
    guard.lock_and_wait(f"vf quality {a.name}")
    m = M.load(a.name)
    if getattr(a, "backend", None):
        return _quality_on_page(a, m)
    if "native" not in m:
        raise SystemExit(f"{a.name} has no native recipe. `vf quality` scores the native file; maintainers score the rest.")
    # An entry that claims the score of a lower-quality file (`scored_at`) is scored at that file, on its native runtime.
    sep = native_sep(dict(m, native=dict(m["native"], file=m["scored_at"])) if m.get("scored_at") else m)
    path = os.path.join(M.ROOT, "results", "quality", f"{a.name}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"sets": {}}
    for s in a.sets or PUBLIC_SETS:
        d = os.path.join(M.DATA, "sets", s)
        if not os.path.isdir(d):
            raise SystemExit(f"{d} missing. " + ("Only the maintainers can run private30." if s == "private30" else "Run: vf fetch-data"))
        clips = sorted(c for c in os.listdir(d) if os.path.exists(os.path.join(d, c, "mixture.flac")))[: a.limit]
        scores = {}
        for i, c in enumerate(clips):
            x, ref = load(os.path.join(d, c, "mixture.flac")), load(os.path.join(d, c, "vocals.flac"))
            y = sep(x)
            scores[c] = round(sdr(ref, y), 3)
            print(f"[{s} {i + 1}/{len(clips)}] {c}: {scores[c]:.2f} dB", flush=True)
            if a.save:
                os.makedirs(os.path.join(a.save, s), exist_ok=True)
                save(os.path.join(a.save, s, f"{c}.flac"), np.clip(y, -1, 1))
        mean = float(np.mean(list(scores.values())))
        res["sets"][s] = {"mean": round(mean, 3), "clips": len(scores)}
        if s != "private30":
            res["sets"][s]["per_clip"] = scores          # the private set's per-clip scores are never written out
        print(f"{s}: {mean:.3f} dB over {len(scores)} clips")
    sep.close()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path if a.limit is None else path.replace(".json", ".quick.json"), "w"), indent=1)      # the board never reads a quick look


def _quality_on_page(a, m):
    """Full scoring of a non-native file through its own pipeline. Writes results/quality/<name>.<backend>.json, with no per-clip scores for private30."""
    from .transfer import page_scores
    sets = {}
    for s in a.sets or PUBLIC_SETS:
        d = os.path.join(M.DATA, "sets", s)
        if not os.path.isdir(d):
            raise SystemExit(f"{d} missing")
        sets[s] = sorted(c for c in os.listdir(d) if os.path.exists(os.path.join(d, c, "mixture.flac")))[: a.limit]
    got = page_scores(a.name, a.backend, sets)
    res = {"sets": {}, "backend": a.backend}
    for s, cl in sets.items():
        scores = {c: round(got[f"{s}/{c}"], 3) for c in cl}
        res["sets"][s] = {"mean": round(float(np.mean(list(scores.values()))), 3), "clips": len(cl)}
        if s != "private30":
            res["sets"][s]["per_clip"] = scores
        print(f"{s}: {res['sets'][s]['mean']:.3f} dB over {len(cl)} clips")
    suffix = "" if a.limit is None else ".quick"
    json.dump(res, open(os.path.join(M.ROOT, "results", "quality", f"{a.name}.{a.backend}{suffix}.json"), "w"), indent=1)


def _rounds(names, rounds):
    """ABBA order, so drift (thermal throttling, a neighbour starting up) hits everyone about equally."""
    seq = []
    for r in range(rounds):
        seq += names if r % 2 == 0 else names[::-1]
    return seq


def cmd_speed(a):
    """Native timing on this machine, interleaved with the reference. --official writes results/devices/native/."""
    from .manifest import REFERENCE
    from .score import sdr
    guard.lock_and_wait("vf speed")
    voc = np.fromfile(os.path.join(M.ROOT, "site/clip/vocals.i16"), dtype="<i2").reshape(-1, 2).T.astype(np.float32) / 32767
    clip_sdr = {}
    raw = np.fromfile(os.path.join(M.ROOT, "site/clip/mix.i16"), dtype="<i2").reshape(-1, 2).T.astype(np.float32) / 32767
    cpus = [int(c) for c in a.cpus.split(",")] if a.cpus else guard.physical_cores()
    names = [REFERENCE] + [n for n in (a.names or []) if n != REFERENCE]
    times = {n: [] for n in names}
    fails = {}
    for n in _rounds(names, a.rounds):
        if n in fails:
            continue
        m = M.load(n)
        if "native" not in m:
            fails[n] = "no native recipe"
            continue
        try:
            t0 = time.perf_counter()
            sep = native_sep(m, cpus)
            load_s = time.perf_counter() - t0
            sep(raw[:, : 44100 * 10])                         # warm-up, not timed
            y, wall = sep.timed(raw)
            sep.close()
            clip_sdr[n] = round(sdr(voc, y), 3)
        except Exception as e:                                # a crash is a result, not a reason to stop the session
            fails[n] = str(e)[:200]
            print(f"{n}: FAILED {fails[n]}", flush=True)
            continue
        times[n].append(wall)
        print(f"{n}: {raw.shape[1] / 44100 / wall:.3f}x realtime (load {load_s:.1f} s, system load {os.getloadavg()[0]:.1f})", flush=True)
    run = {"device": "native", "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpu": _cpu_name(), "cpus": a.cpus,
           "models": {n: ({"xrt": round(30 / float(np.exp(np.mean(np.log(t)))), 4), "runs": [round(v, 2) for v in t], "sdr": clip_sdr.get(n)} if t else {"xrt": None, "error": fails.get(n)})
                      for n, t in times.items()}}
    ref = run["models"][REFERENCE]["xrt"]
    if not ref:
        raise SystemExit(f"HTDemucs didn't run: {fails.get(REFERENCE)}. Without it there's nothing to compare against. Try: vf fetch-reference")
    print(f"\n{'model':24}  xrt     vs {REFERENCE}")
    for n, r in run["models"].items():
        print(f"{n:24}  {r['xrt'] if r['xrt'] else 'failed':<7} {I.device_ratio(r['xrt'], ref):.3f}x")
    if getattr(a, "out", None):
        json.dump(run, open(a.out, "w"), indent=1)
    if a.official:
        d = os.path.join(M.ROOT, "results", "devices", "native")
        os.makedirs(d, exist_ok=True)
        json.dump(run, open(os.path.join(d, run["date"].replace(":", "") + ".json"), "w"), indent=1)


def _cpu_name():
    try:
        return next(l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return "unknown"


def _latest_runs():
    """{device: {model: run}}: each model's newest session on each device, with that session's reference. A failure right after a
    success only counts once a second session fails too (README: Speed index), so until then the success stands, marked as such."""
    out = {}
    for d in I.ALL_DEVICES:
        out[d] = {}
        for f in sorted(glob.glob(os.path.join(M.ROOT, "results", "devices", d, "*.json"))):
            run = json.load(open(f))
            ref = (run["models"].get(M.REFERENCE) or {}).get("xrt")
            if not ref or run.get("onBattery"):      # a device on battery can throttle, so that timing doesn't count
                continue
            for n, r in run["models"].items():
                new = {"xrt": r.get("xrt"), "ref": ref, "error": r.get("error"), "sdr": r.get("sdr"), "date": run["date"]}
                old = out[d].get(n)
                if not new["xrt"] and old and old["xrt"] and not old.get("failed_since"):
                    new = {**old, "failed_since": run["date"]}       # one failure after a success: keep the success until it repeats
                out[d][n] = new
    return out


def _transfer(name, parent):
    """The score transfer rule's recorded verdict: pass only if every file checked for this entry passed against that parent,
    or failed and has since been scored in full on its own backend, in which case _full_scores counts that score."""
    from .transfer import still_matches
    m, pm = M.load(name), M.load(parent)
    fs = glob.glob(os.path.join(M.ROOT, "results", "transfer", f"{name}.*.json"))
    def ok(f, r):
        be = os.path.basename(f).split(".")[-2]
        return r["parent"] == parent and (_full_score(name, be) is not None or (r["verdict"] == "pass" and be in m and still_matches(r, m, be, pm)))
    return "pass" if fs and all(ok(f, json.load(open(f))) for f in fs) else "none"


def _same_weights(name, parent):
    """An entry that inherits a score has to run the parent's own file natively. None when either file isn't on this machine."""
    from .transfer import file_id
    m, pm = M.load(name), M.load(parent)
    a, b = file_id(m, m["native"]["file"]), file_id(pm, pm["native"]["file"])
    return None if a is None or b is None else a == b


def _full_score(name, backend):
    p = os.path.join(M.ROOT, "results", "quality", f"{name}.{backend}.json")
    return json.load(open(p))["sets"] if os.path.exists(p) else None


def _cleared_elsewhere(name, backend):
    """Another entry scored this same file, through the same engine, in full, and it held up against that entry's native score. Only the overlap can differ,
    and that's stitching, already covered by the native score. If it came out low, this entry needs its own full score, so that it can come out low too."""
    from .transfer import MEAN_DB, engine_id, file_id
    m = M.load(name)
    mine = file_id(m, m[backend]["file"]), engine_id(m, m[backend])
    if mine[0] is None:
        return False
    for other in M.names():
        o = M.load(other)
        full = _full_score(other, backend) if other != name and backend in o else None
        if full is None or (file_id(o, o[backend]["file"]), engine_id(o, o[backend])) != mine:
            continue
        p = os.path.join(M.ROOT, "results", "quality", f"{other}.json")
        native = json.load(open(p))["sets"] if os.path.exists(p) else {}
        shared = [k for k in full if k in native]
        if shared and np.mean([full[k]["mean"] for k in shared]) >= np.mean([native[k]["mean"] for k in shared]) - MEAN_DB:
            return True
    return False


def _unchecked(name, parent):
    """The backends where this entry runs a file other than its native one, with neither a passed transfer check nor a full score. Baselines too: they set the bar."""
    from .transfer import still_matches
    m = M.load(name)
    native = m.get("native", {}).get("file")
    out = []
    for backend in ("browser", "webgpu"):
        if backend not in m or m[backend].get("file") == native or _full_score(name, backend) is not None or _cleared_elsewhere(name, backend):
            continue
        p = os.path.join(M.ROOT, "results", "transfer", f"{name}.{backend}.json")
        rec = json.load(open(p)) if os.path.exists(p) else {}
        if rec.get("verdict") != "pass" or rec.get("parent") != parent or not still_matches(rec, m, backend, M.load(parent)):
            out.append(backend)
    return out


def _lowest(name, sets):
    """Q is the lowest across an entry's files: its native sets, or any file scored in full on its own backend, whichever averages lower."""
    best = sets
    for backend in ("browser", "webgpu"):
        full = _full_score(name, backend)
        if full and set(full) >= set(best) and np.mean([full[k]["mean"] for k in best]) < np.mean([best[k]["mean"] for k in best]):
            best = full
    return best


def board_data():
    runs = _latest_runs()
    quals, inherited = {}, {}
    for n in M.names():
        p = os.path.join(M.ROOT, "results", "quality", f"{n}.json")
        if os.path.exists(p):
            j = json.load(open(p))
            s = j["sets"]
            if j.get("inherits") and _transfer(n, j["inherits"]) != "pass":      # no row borrows a score without a recorded pass
                print(f"{n}: inherits from {j['inherits']} with no recorded pass in results/transfer for its current files, so it's left off the board")
                continue
            if j.get("inherits") and _same_weights(n, j["inherits"]) is False:
                print(f"{n}: inherits from {j['inherits']} but runs a different file natively, so it needs its own vf quality")
                continue
            unchecked = _unchecked(n, j.get("inherits") or n)
            if unchecked:       # README: a file that differs from the native one passes the check or is scored in full
                missing = [k for k in M.load(n)["files"] if not os.path.exists(M.path_of(M.load(n), k))]
                if missing:
                    print(f"{n}: its files aren't in {M.MODELS} ({', '.join(missing)}), so its checks can't be matched and it's left off the board. Run `vf fetch {n}` or set VF_MODELS")
                    continue
                print(f"{n}: its {' and '.join(unchecked)} file has no passed `vf transfer` and no full score, so it's left off the board")
                continue
            inherited[n] = j.get("inherits")
            s = _lowest(n, s)
            quals[n] = {k: s[k]["mean"] for k in SETS if k in s}
    refq = np.mean(list(quals[M.REFERENCE].values())) if M.REFERENCE in quals and len(quals[M.REFERENCE]) == 3 else None
    points = []
    for n in M.names():
        m = M.load(n)
        ratios, devices = {}, {}
        for d in I.ALL_DEVICES:
            r = runs[d].get(n)
            # a model with no recipe for this device can't run there, however long we wait. Scored as failed, if the device has been measured at all,
            # and an old timing from before its recipe was removed doesn't count
            recipe = {"native": "native", "browser": "browser"}.get(d, "webgpu")
            if recipe not in m:
                if runs[d]:
                    ratios[d] = I.FLOOR
                    devices[d] = {"status": "no recipe", "ratio": I.FLOOR}
                continue
            if r is None:
                continue
            want = (runs["native"].get(n) or {}).get("sdr")
            wrong = d != "native" and r.get("xrt") and want is not None and r.get("sdr") is not None and abs(r["sdr"] - want) > CLIP_TOLERANCE_DB
            ratios[d] = I.FLOOR if wrong else I.device_ratio(r["xrt"], r["ref"])
            status = f"wrong output: {r['sdr']:.2f} dB, native {want:.2f} dB" if wrong else (("ok" if not r.get("failed_since") else f"ok, failed once on {r['failed_since']}: counts if it fails again") if r["xrt"] else (r.get("error") or "failed"))
            devices[d] = {"xrt": r["xrt"], "ratio": round(ratios[d], 4), "status": status, "sdr": r.get("sdr")}
        q = quals.get(n)
        if not ratios or not q or len(q) < 3:
            continue
        missing = [d for d in I.ALL_DEVICES if d not in ratios]
        s = I.speed_index(ratios, provisional=bool(missing))
        points.append({"name": n, "title": m.get("title", n), "kind": m.get("kind"), "owner": M.owner(m), "t": M.submitted_at(m), "s": round(s, 4), "q": round(float(np.mean(list(q.values()))), 3),
                       "sets": q, "devices": devices, "clip_sdr": (runs["native"].get(n) or {}).get("sdr"), "provisional": bool(missing), "license": m.get("license"), "description": m.get("description"), "overlap": m.get("overlap"), "inherits": inherited.get(n)})
    attr = I.attribute(points, refq) if refq else {"area": {}, "total": 0, "steps": []}
    gains = {st["name"]: round(st["gain"], 4) for st in attr["steps"]}
    copies = {st["name"] for st in attr["steps"] if st.get("copy")}
    for p in points:
        p["gain"] = gains.get(p["name"], 0)
        p["copy"] = p["name"] in copies
    return {"points": points, "reference_q": refq, "floor_q": None if refq is None else refq - I.QUALITY_MARGIN_DB, "floor_s": I.FLOOR, "origin_s": I.AREA_ORIGIN, "area": {k: round(v, 3) for k, v in attr["area"].items()},
            "payouts": I.payouts(attr["area"]), "frontier": [p["name"] for p in I.frontier(points)], "devices": list(I.ALL_DEVICES), "pot": I.POT_USD, "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}


TEMPLATE = """# Everything about your entry. `vf validate {name}` checks it. README.md explains each field.
title: {name}
kind: contestant
github: your-github-login
wallet: "0x0000000000000000000000000000000000000000"     # Base, for USDC
license: {{ weights: MIT, code: MIT }}
description: One or two sentences on what this is.
# layout.type is mdx, htdemucs or mask. README.md has a table, and every entry in baselines/ shows one filled in. This one is mdx:
layout: {{ type: mdx, n_fft: 7680, hop: 1024, dim_f: 3072, dim_t: 256, compensate: 1.021 }}
overlap: 1                                                 # passes over each sample: 1 = none, 1.33 = 25%, 2 = 50%
files:
  fp32: {{ path: {name}/core.onnx, url: "https://huggingface.co/you/{name}/resolve/main/core.onnx", sha256: "" }}
  fp16: {{ path: {name}/core_f16.onnx, url: "https://huggingface.co/you/{name}/resolve/main/core_f16.onnx", sha256: "" }}
native: {{ file: fp32, runtime: onnxruntime, processes: 1, threads: 4 }}
browser: {{ file: fp32, runtime: wasm@1.30.0, workers: 1, threads: 4 }}
webgpu: {{ file: fp16, runtime: webgpu@1.23.0 }}
"""


def cmd_new(a):
    d = os.path.join(M.ROOT, "submissions", a.name)
    if os.path.exists(d):
        raise SystemExit(f"{d} already exists")
    os.makedirs(d)
    open(os.path.join(d, "manifest.yaml"), "w").write(TEMPLATE.format(name=a.name))
    if a.engine:
        shutil.copytree(os.path.join(M.ROOT, "starter", "engine"), os.path.join(d, "engine"))
        with open(os.path.join(d, "manifest.yaml"), "a") as f:
            f.write("# Your engine is in engine/. Point browser and webgpu at it: add  engine: engine/worker.js  to each recipe.\n")
    open(os.path.join(d, "training.txt"), "w").write("# Every dataset, track or take you trained on, one per line, with its source. Also any teacher model.\n")
    os.makedirs(os.path.join(M.MODELS, a.name), exist_ok=True)
    print(f"Wrote {d}. Put your ONNX in {os.path.join(M.MODELS, a.name)}, edit the manifest, then: vf validate {a.name}")


def cmd_fetch_data(a):
    py = sys.executable
    d = M.DATA
    os.makedirs(d, exist_ok=True)
    musdb = os.path.join(d, "musdb")
    subprocess.run([py, os.path.join(M.ROOT, "tools", "fetch_musdb.py"), musdb], check=True)
    subprocess.run([py, os.path.join(M.ROOT, "tools", "make_sets.py"), "musdb30", musdb, os.path.join(d, "sets", "musdb30")], check=True)
    subprocess.run([py, os.path.join(M.ROOT, "tools", "make_sets.py"), "cc", musdb, os.path.join(d, "sets", "cc30")], check=True)


def _fetch_entry(name):
    """An entry's files from the site's copy, into the models folder at the paths its manifest names. A file only native uses isn't on the site."""
    import hashlib
    import urllib.request
    def get(path):
        return urllib.request.urlopen(urllib.request.Request(f"{M.SITE}/{path}", headers={"User-Agent": "vocal-frontier-cli"})).read()
    models = json.loads(get("models.json"))["models"]
    ref, m = models[name], M.load(name)
    for key in m["files"]:
        recipe = next((r for r in (m.get("browser"), m.get("webgpu")) if r and r["file"] == key), None)
        if recipe is None:
            print(f"{key}: native only, so the site doesn't have it")
            continue
        site = ref["browser" if recipe is m.get("browser") else "webgpu"]["file"]
        out = M.path_of(m, key)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        # The site serves each ONNX file as a graph and its weights. Both land in the model's folder, which every runtime here reads as one model.
        for what, sub, dest in (("graph", site, out), ("weights", site.get("data"), os.path.join(os.path.dirname(out), (site.get("data") or {}).get("name", "")))):
            if not sub:
                continue
            h = hashlib.sha256()
            with open(dest + ".part", "wb") as f:
                for i, part in enumerate(sub["parts"]):
                    print(f"{key} {what}: part {i + 1} of {len(sub['parts'])}", flush=True)
                    b = get(part)
                    h.update(b); f.write(b)
            if h.hexdigest() != sub["sha256"]:
                raise SystemExit(f"{key}: checksum doesn't match. Try again.")
            os.replace(dest + ".part", dest)
        print(f"{key}: {out}")


def cmd_fetch_reference(a):
    _fetch_entry(M.REFERENCE)


def cmd_fetch(a):
    if a.name not in M.names():
        raise SystemExit(f"No entry called {a.name}. They're the folders in baselines/.")
    _fetch_entry(a.name)


def cmd_preview(a):
    """Your fork of the site, built and served here, with your entry on the real board and what it would earn."""
    m = M.load(a.name)
    if "native" not in m:
        raise SystemExit("Preview times your native file first, so it needs a native recipe.")
    ref = M.load(M.REFERENCE)
    if not all(os.path.exists(M.path_of(ref, k)) for k in ref["files"]):
        _fetch_entry(M.REFERENCE)
    qdir = os.path.join(M.ROOT, "results", "quality")
    full, quick = os.path.join(qdir, f"{a.name}.json"), os.path.join(qdir, f"{a.name}.quick.json")
    if not os.path.exists(full) and not os.path.exists(quick):
        if not all(os.path.exists(os.path.join(M.DATA, "sets", s)) for s in PUBLIC_SETS):
            raise SystemExit("No score yet, and scoring needs the public sets. Run: vf fetch-data (about 90 minutes)")
        print("No score yet, so taking a quick look: 5 clips per public set.")
        cmd_quality(argparse.Namespace(name=a.name, sets=None, limit=5, save=None))
    q = json.load(open(full if os.path.exists(full) else quick))["sets"]
    if not os.path.exists(os.path.join(M.DATA, "sets", "cc30")) and not q:
        raise SystemExit("Run: vf fetch-data")
    board = board_data()
    public = [s for s in PUBLIC_SETS if s in q]
    if len(public) == 2 and "private30" not in q:
        # no private30 here, so carry over how the three-set score relates to the public mean across the board's entries
        shift = float(np.mean([p["q"] - np.mean([p["sets"][s] for s in public]) for p in board["points"]]))
        score, note = float(np.mean([q[s]["mean"] for s in public])) + shift, "private30 is estimated"
    else:
        score, note = float(np.mean([v["mean"] for v in q.values()])), None
    if not os.path.exists(full):
        note = "5 clips per set" + (f", {note}" if note else "")
    timing = os.path.join(M.ROOT, ".preview-native.json")
    cmd_speed(argparse.Namespace(names=[a.name], rounds=2, cpus=None, official=False, out=timing))
    run = json.load(open(timing))["models"]
    from .build import build
    build(".preview", chunked=False, only=[M.REFERENCE, a.name])
    json.dump({"name": a.name, "title": m.get("title", a.name), "q": round(score, 3), "q_estimated": note, "license": m.get("license"), "description": m.get("description"),
               "native": {"xrt": run[a.name]["xrt"], "ratio": I.device_ratio(run[a.name]["xrt"], run[M.REFERENCE]["xrt"])}},
              open(os.path.join(M.ROOT, ".preview", "data", "preview.json"), "w"), indent=1)
    guard.release()
    if a.no_serve:
        return
    if not os.path.exists(full):
        print("This score is from 5 clips per set and can be half a dB off, so the payout it shows can be wrong. Run vf quality before trusting it.")
    print(f"Your preview: http://127.0.0.1:{a.port}/  Run your model on the page to add this browser and GPU. Ctrl-C to stop.")
    subprocess.run(["node", os.path.join(M.ROOT, "tools", "serve.mjs"), os.path.join(M.ROOT, ".preview"), str(a.port)], check=True)


def _write_board():
    b = board_data()
    os.makedirs(os.path.join(M.ROOT, "site", "data"), exist_ok=True)
    json.dump(b, open(os.path.join(M.ROOT, "site", "data", "board.json"), "w"), indent=1)
    return b


PAYOUT_TABLE = re.compile(r"(<!-- vf board writes this table -->\n).*?(<!-- end of table -->)", re.S)


def payout_rows(b):
    """What one new entry would earn on this board. Fixed gaps, plus one step past the entries a contestant is likeliest to start from."""
    by_name = {p["name"]: p for p in b["points"]}
    rows = [(s, q, None) for s, q in ((16, 10.5), (8, 11), (4, 12), (2, 13))]
    if "htdemucs-lite-8k" in by_name:
        p = by_name["htdemucs-lite-8k"]
        rows.insert(0, (p["s"] * 2, p["q"], "lite-8k, twice as fast"))
    for name, label in (("voc-ft-o1", "tuned Voc_FT"), ("melband-kim-wasm", "MelBand Kim in WASM")):
        if name in by_name:
            rows.append((by_name[name]["s"], by_name[name]["q"] + 1, f"{label}, 1 dB cleaner"))
    # Shared with the maintainers' area and every contestant's already on the board, as the real payout is.
    x0, y0, m = math.log2(I.AREA_ORIGIN), b["floor_q"], sum(v for k, v in b["area"].items() if k != "baseline")
    base = [(math.log2(p["s"]), p["q"]) for p in b["points"] if p["q"] >= y0]
    out = ["| new entry | units | paid |", "|---|---|---|"]
    for s, q, label in rows:
        u = I._union_area(base + [(math.log2(s), q)], x0, y0) - I._union_area(base, x0, y0)
        out.append(f"| {s:.3g}×, {q:.4g} dB{f' ({label})' if label else ''} | {u:.2f} | ${b['pot'] * u / (m + u):.0f} |")
    return "\n".join(out)


def _write_readme_table(b):
    path = os.path.join(M.ROOT, "README.md")
    with open(path) as f:
        text = f.read()
    new, n = PAYOUT_TABLE.subn(lambda m: m.group(1) + payout_rows(b) + "\n" + m.group(2), text)
    if n != 1:
        raise SystemExit("README.md has lost its payout table markers")
    with open(path, "w") as f:
        f.write(new)


def cmd_board(a):
    """The one step after results change: the board, the README's payout table and the picture a shared link shows."""
    b = _write_board()
    _write_readme_table(b)
    subprocess.run(["node", os.path.join(M.ROOT, "tools", "og_image.mjs")], check=True)
    print(json.dumps({k: b[k] for k in ("reference_q", "floor_q", "area", "payouts", "frontier")}, indent=1))


def cmd_build(a):
    from .build import build
    _write_board()      # a build always carries the board its results and code give today
    build(a.out, chunked=not a.local, only=a.only)


def cmd_transfer(a):
    from .transfer import run
    sys.exit(0 if all(v != "fail" for v in run(a.name, a.parent, a.backend)) else 1)


def cmd_official(a):
    """An official session against the live site, on this machine: native, then browser CPU, then its GPU. Writes results/devices/."""
    guard.lock_and_wait("vf official")
    cpus = ",".join(map(str, guard.physical_cores()))
    if a.device in ("native", "all"):
        cmd_speed(argparse.Namespace(names=a.names, rounds=2, cpus=cpus, official=True, out=None))
    for d in [d for d in ("browser", a.gpu) if a.device in (d, "all")]:
        guard.lock_and_wait(f"vf official {d}")
        subprocess.run(["taskset", "-c", cpus, "node", os.path.join(M.ROOT, "tools", "official.mjs"), d, a.url.rstrip("/") + "/", *a.names], check=True, cwd=M.ROOT)


def cmd_serve(a):
    subprocess.run(["node", os.path.join(M.ROOT, "tools", "serve.mjs"), a.dir, str(a.port)], check=True)


def main():
    ap = argparse.ArgumentParser(prog="vf", description="Vocal Frontier: everything a contestant runs")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list", help="the board, as this checkout sees it").set_defaults(fn=cmd_list)
    p = sp.add_parser("validate", help="check a manifest and its ONNX files against the rules"); p.add_argument("name"); p.set_defaults(fn=cmd_validate)
    p = sp.add_parser("quality", help="score a model on the public sets"); p.add_argument("name"); p.add_argument("--sets", nargs="*"); p.add_argument("--limit", type=int); p.add_argument("--save"); p.add_argument("--backend", choices=["browser", "webgpu"], help="score through the page's own pipeline instead of natively"); p.set_defaults(fn=cmd_quality)
    p = sp.add_parser("speed", help="native speed here, interleaved with HTDemucs"); p.add_argument("names", nargs="*"); p.add_argument("--rounds", type=int, default=2); p.add_argument("--cpus"); p.add_argument("--official", action="store_true"); p.set_defaults(fn=cmd_speed)
    p = sp.add_parser("new", help="start a submission"); p.add_argument("name"); p.add_argument("--engine", action="store_true", help="also copy the starter engine, to change how it runs in the browser"); p.set_defaults(fn=cmd_new)
    p = sp.add_parser("transfer", help="check a non-native file against the native score: the README's transfer rule"); p.add_argument("name"); p.add_argument("--parent", help="the entry whose native file and overlap to compare with, default its own"); p.add_argument("--backend", nargs="*", choices=["browser", "webgpu"]); p.set_defaults(fn=cmd_transfer)
    p = sp.add_parser("official", help="maintainers: official timing against the live site on this machine"); p.add_argument("device", choices=["native", "browser", "vega", "all"]); p.add_argument("url"); p.add_argument("names", nargs="+"); p.add_argument("--gpu", default="vega", help="the name this machine's GPU goes under in results/devices/"); p.set_defaults(fn=cmd_official)
    sp.add_parser("fetch-data", help="download MUSDB18-HQ test (about 3.4 GB) and ccMixter takes, build the public sets").set_defaults(fn=cmd_fetch_data)
    p = sp.add_parser("preview", help="build and serve your fork here, with your entry on the real board and what it would earn"); p.add_argument("name"); p.add_argument("--port", type=int, default=8787); p.add_argument("--no-serve", action="store_true"); p.set_defaults(fn=cmd_preview)
    sp.add_parser("fetch-reference", help="download HTDemucs, the 1x reference that local comparisons need").set_defaults(fn=cmd_fetch_reference)
    p = sp.add_parser("fetch", help="download an existing entry's files, to build on them"); p.add_argument("name"); p.set_defaults(fn=cmd_fetch)
    sp.add_parser("board", help="recompute index and payouts into site/data/board.json").set_defaults(fn=cmd_board)
    p = sp.add_parser("build", help="build the static site"); p.add_argument("--out", default="dist"); p.add_argument("--local", action="store_true", help="for `vf serve` here: models stay in place, unchunked"); p.add_argument("--only", nargs="*"); p.set_defaults(fn=cmd_build)
    p = sp.add_parser("serve", help="serve the site locally with the right headers"); p.add_argument("--dir", default="dist"); p.add_argument("--port", type=int, default=8787); p.set_defaults(fn=cmd_serve)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
