"""Reads submissions/<name>/manifest.yaml and baselines/<name>/manifest.yaml. No code runs from a submission: only YAML and
ONNX files go through here. vf/validate.py rejects anything else."""
import hashlib
import json
import os

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.environ.get("VF_MODELS", os.path.join(ROOT, "models"))
DATA = os.environ.get("VF_DATA", os.path.join(ROOT, "data"))
CACHE = os.path.join(ROOT, ".sha256-cache.json")
REFERENCE = "htdemucs"
UNQUEUED = 9e15                                  # not in results/queue.json yet: behind everyone
SITE = "https://vocal-frontier.kendell.dev"


def names():
    out = []
    for base in ("baselines", "submissions"):
        d = os.path.join(ROOT, base)
        if os.path.isdir(d):
            out += [n for n in sorted(os.listdir(d)) if os.path.exists(os.path.join(d, n, "manifest.yaml"))]
    return out


def load(name):
    for base in ("baselines", "submissions"):
        p = os.path.join(ROOT, base, name, "manifest.yaml")
        if os.path.exists(p):
            with open(p) as f:
                m = yaml.safe_load(f)
            m["name"], m["dir"], m["group"] = name, os.path.dirname(p), base
            return m
    raise SystemExit(f"no manifest for {name!r}. Have: {', '.join(names())}")


def owner(m):
    """Who the area belongs to: baseline (others' models), maintainers (our own models), or a contestant's GitHub login."""
    if m["group"] == "submissions":              # never from the manifest's own kind, which its author controls
        return str(m.get("github") or "unknown")
    return "maintainers" if m.get("kind") == "ours" else "baseline"


def _read(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def submitted_at(m):
    """A place in line. The starting board is 0. Others come from results/queue.json, which maintainers keep from PR history, never from a manifest.
    Once results/starting.json lists the starting board (written at the start), a baseline that isn't on it queues like anyone else."""
    queue = _read(os.path.join(ROOT, "results", "queue.json"), {})
    starting = _read(os.path.join(ROOT, "results", "starting.json"), None)
    first = m["group"] == "baselines" and (starting is None or m["name"] in starting)
    return queue.get(m["name"], 0 if first else UNQUEUED)


def path_of(m, key):
    return os.path.join(MODELS, m["files"][key]["path"])


def sha256(path):
    try:
        with open(CACHE) as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}
    st = os.stat(path)
    k = f"{path}:{st.st_size}:{int(st.st_mtime)}"
    if k in cache:
        return cache[k]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    cache[k] = h.hexdigest()
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=0)
    return cache[k]


def model_id(path):
    """What an ONNX file computes, however it's stored: one file, or weights split beside it (core.onnx.data), as the site serves them
    and `vf fetch` saves them. The graph without its weights, then each weight's name, type, shape and values."""
    d, base = os.path.split(path)
    parts = [path] + sorted(os.path.join(d, x) for x in os.listdir(d) if x.startswith(base) and x != base)
    try:
        with open(CACHE) as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}
    k = "model:" + "|".join(f"{p}:{os.stat(p).st_size}:{int(os.stat(p).st_mtime)}" for p in parts)
    if k in cache:
        return cache[k]
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    h = hashlib.sha256()
    inits = list(m.graph.initializer)
    del m.graph.initializer[:]
    h.update(m.SerializeToString(deterministic=True))
    for t in sorted(inits, key=lambda t: t.name):
        a = numpy_helper.to_array(t)
        h.update(f"{t.name}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    del m, inits
    cache[k] = h.hexdigest()
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=0)
    return cache[k]
