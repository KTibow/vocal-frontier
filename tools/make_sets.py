"""Build the evaluation sets (directories of <track>/mixture.flac + vocals.flac, 44.1 kHz stereo).

    uv run tools/make_sets.py musdb30 data/musdb data/sets/musdb30
        One 30 s excerpt per MUSDB18-HQ test track: the window with the most 1 s steps of vocal activity
        (> -30 dB re the track's loudest vocal second), the earliest such window on ties.

    uv run tools/make_sets.py cc data/musdb data/sets/cc30 [n=50]
        Real a cappellas from ccMixter (CC BY /
        BY-NC / CC0, one take per artist), listed with URL, sha1, license, attribution and usable span in
        tools/data/ccmixter_takes.json and downloaded into data/ccmixter/raw if missing. Each placement is a 6-20 s
        excerpt of the next take, cut at quiet points. Also writes <dst>.ATTRIBUTION.txt for the takes used.
"""
import base64
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import urllib.request

import numpy as np

from vf.audio import SR, load, save

LEN = 30 * SR
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CCMAN = os.path.join(ROOT, "tools", "data", "ccmixter_takes.json")
CCDIR = os.path.join(ROOT, "data", "ccmixter", "raw")


def best_window(v, length=LEN):
    e = (v[:, :len(v[0]) // SR * SR] ** 2).reshape(2, -1, SR).sum((0, 2))
    act = (e > e.max() * 1e-3).astype(int)
    w = length // SR
    if len(act) <= w:
        return 0
    c = np.convolve(act, np.ones(w, int), "valid")
    return int(np.argmax(c)) * SR


def musdb30(src, dst):
    for t in sorted(os.listdir(src)):
        if not os.path.exists(os.path.join(src, t, "vocals.flac")):
            continue
        v = load(os.path.join(src, t, "vocals.flac"))
        m = load(os.path.join(src, t, "mixture.flac"))
        s = best_window(v)
        os.makedirs(os.path.join(dst, t), exist_ok=True)
        save(os.path.join(dst, t, "vocals.flac"), v[:, s:s + LEN])
        save(os.path.join(dst, t, "mixture.flac"), m[:, s:s + LEN])
        print(t, s / SR)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def overlay(src, dst, n, rnd, take, prefix):
    """n clips: a 30 s MUSDB test accompaniment (mixture - vocals, one track per clip in turn) with vocal takes laid
    in sequence after 0.5-3 s gaps, at a vocal-to-accompaniment ratio drawn from -6..+3 dB, peak-limited to 0.99.
    take(room) -> (vocals [2, <= room], label). Writes <dst>/<prefix>NN/{mixture,vocals}.flac and <dst>.tsv."""
    tracks = sorted(t for t in os.listdir(src) if os.path.exists(os.path.join(src, t, "vocals.flac")))
    os.makedirs(dst, exist_ok=True)
    log = open(os.path.normpath(dst) + ".tsv", "w")
    clips = {}
    for i in range(n):
        t = tracks[i % len(tracks)]
        m, v = load(os.path.join(src, t, "mixture.flac")), load(os.path.join(src, t, "vocals.flac"))
        acc = m - v
        s = rnd.randrange(SR * 10, max(SR * 11, acc.shape[1] - LEN - SR * 10))
        acc = acc[:, s:s + LEN]
        voc = np.zeros_like(acc)
        pos = int(rnd.uniform(0.5, 3) * SR)
        used = []
        while pos < LEN - 3 * SR:
            x, label = take(LEN - pos)
            voc[:, pos:pos + x.shape[1]] += x
            used.append(label)
            pos += x.shape[1] + int(rnd.uniform(0.5, 3) * SR)
        act = np.abs(voc).max(0) > 1e-4
        vr = rnd.uniform(-6, 3)
        g = rms(acc[:, act]) / max(rms(voc[:, act]), 1e-6) * 10 ** (vr / 20)
        voc *= g
        mix = voc + acc
        peak = np.abs(mix).max()
        if peak > 0.99:
            mix, voc = mix * 0.99 / peak, voc * 0.99 / peak
        name = f"{prefix}{i:02d}"
        os.makedirs(os.path.join(dst, name), exist_ok=True)
        save(os.path.join(dst, name, "mixture.flac"), mix)
        save(os.path.join(dst, name, "vocals.flac"), voc)
        log.write(f"{name}\t{t}\t{s / SR:.1f}\t{vr:.1f}\t{' | '.join(used)}\n")
        print(name, t, f"vr {vr:.1f} dB", len(used), "takes", flush=True)
        clips[name] = used
    log.close()
    return clips



def cc_fetch(t, d=CCDIR):
    """Download one manifest take into d (if missing) and check its sha1 (ccMixter's base32 file hash)."""
    path = os.path.join(d, t["file"])
    if not os.path.exists(path):
        os.makedirs(d, exist_ok=True)
        # ccmixter.org/content refuses requests without a Referer from the site (hotlink protection)
        req = urllib.request.Request(t["url"], headers={"Referer": "https://ccmixter.org/", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(path + ".part", path)
        print("fetched", t["file"], flush=True)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    if base64.b32encode(h.digest()).decode() != t["sha1"]:
        raise SystemExit(f"{path}: sha1 differs from the manifest (file changed upstream?); delete it to refetch")
    return path


def cc_decode(path, start, end):
    """[start, end) s of a downloaded take -> float32 [2, n] at 44.1 kHz. Downloads are untrusted: ffmpeg only."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                          "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.ascontiguousarray(np.frombuffer(raw, np.float32).reshape(-1, 2).T)


def quietest(x, a, b, w=SR // 50):
    """Start of the quietest 20 ms frame in x[:, a:b]."""
    n = (b - a) // w
    if n < 1:
        return a
    e = (x[:, a:a + n * w] ** 2).reshape(2, n, w).sum((0, 2))
    return a + int(np.argmin(e)) * w


def cc(src, dst, n=50, seed=0):
    man = json.load(open(CCMAN))
    takes = sorted(man["takes"], key=lambda t: t["id"])
    paths = {t["id"]: cc_fetch(t) for t in takes}
    rnd = random.Random(seed)
    rnd.shuffle(takes)
    k = 0
    fade = np.linspace(0, 1, SR // 100, dtype=np.float32)       # 10 ms

    def take(room):
        """A 6-20 s excerpt of the next take (fewer if the room or the take is shorter), its cut points moved to
        the quietest 20 ms within 1 s inside the excerpt, with 10 ms fades."""
        nonlocal k
        t = takes[k % len(takes)]
        k += 1
        x = cc_decode(paths[t["id"]], t["start"], t["end"])
        L = min(int(rnd.uniform(6, 20) * SR), room, x.shape[1])
        a = rnd.randrange(0, x.shape[1] - L + 1)
        b = a + L
        a = quietest(x, a, a + min(SR, L // 4))
        b = quietest(x, b - min(SR, L // 4), b) + SR // 50
        y = x[:, a:b].copy()
        y[:, :len(fade)] *= fade
        y[:, -len(fade):] *= fade[::-1]
        return y, f"ccmixter:{t['id']}@{t['start'] + a / SR:.2f}+{y.shape[1] / SR:.2f}"

    clips = overlay(src, dst, n, rnd, take, "c")
    used = {}
    for name, labels in clips.items():
        for lab in labels:
            used.setdefault(int(lab.split(":")[1].split("@")[0]), []).append(name)
    with open(os.path.normpath(dst) + ".ATTRIBUTION.txt", "w") as f:
        f.write(f"{os.path.basename(os.path.normpath(dst))}: vocal takes from ccMixter (ccmixter.org), excerpted, "
                "gain-adjusted and mixed over MUSDB18-HQ test\naccompaniments (MUSDB18-HQ is under its own terms: "
                "https://sigsep.github.io/datasets/musdb.html). Each take keeps its\nCreative Commons license, listed "
                "below; NC-licensed takes may not be used commercially. Built by bench/make_sets.py cc\nfrom "
                "tools/data/ccmixter_takes.json.\n\n")
        lic = {}
        for t in sorted(takes, key=lambda t: t["id"]):
            if t["id"] in used:
                lic[t["license"]] = lic.get(t["license"], 0) + 1
                f.write(f"{t['attribution']}\n    clips: {' '.join(sorted(set(used[t['id']])))}\n")
        f.write("\n" + ", ".join(f"{v} x {k}" for k, v in sorted(lic.items())) + f" ({len(used)} takes)\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "musdb30":
        musdb30(sys.argv[2], sys.argv[3])
    elif cmd == "cc":
        cc(sys.argv[2], sys.argv[3], *(int(a) for a in sys.argv[4:]))
