"""Real a cappellas for training (FINDINGS.md "Pooled MelBand Kim student (agent)"): ccMixter uploads tagged acappella,
screened like the cc30 eval set (bench/make_sets.py cc, bench/data/ccmixter_takes.json) but disjoint from it.

    python train/acap_fetch.py <out dir> [--max 450] [--per_user 3] [--workers 8] [--lic by,zero] [--maxlen 420]

Commercial-OK set (results/acappella_sources.md): --lic by,zero --maxlen 420 (CC BY + CC0 only, whole takes, cap
counted over those licenses only).

Kept: CC BY / BY-NC / CC0 only; no AI-generated, spoken-word/poetry or choir tags; 44.1/48 kHz files of 40 s-7 min;
at most --per_user takes per uploader (newest first); **no take and no uploader of the cc30 manifest**. Each take:
download (Referer header: ccMixter's hotlink protection), check ccMixter's base32 SHA-1, decode with ffmpeg only
(downloads are untrusted), keep the <= --maxlen (60) s span with the most audio above -40 dB re its loudest 100 ms and cut
silences > 1 s out of it, then screen: < 0.12 of the energy below 100 Hz, and Kim Vocal 2 (one 5.9 s chunk) keeps
>= 0.95 of the energy of both the span's loudest and its quietest mostly-active 5.9 s (otherwise instruments).
Writes <out>/<id>.flac (16-bit, 44.1 kHz stereo, trimmed) for the kept takes, <out>/manifest.json (every candidate,
with url, sha1, uploader, license, attribution, screen values and why it was dropped) and <out>/READY when done.
The mp3s go to <out>_raw/ and are deleted after decoding.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SR = 44100
API = "http://ccmixter.org/api/query?tags=acappella&f=json&sort=date&ord=desc&limit={n}&offset={o}"
CC30 = os.path.join(ROOT, "bench", "data", "ccmixter_takes.json")
BAD_TAGS = ("ai_generated", "ai_assisted", "ia_generated", "spoken", "poetry", "poem", "choir")
UA = {"Referer": "https://ccmixter.org/", "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"}


def get(url, timeout=120, tries=4):
    # the API repeats the whole JSON body in an X-JSON response header: a 200-upload page's header line is past
    # http.client's 64 kB limit, so raise it. Be polite: back off 5/20/60 s on 429, 5xx and network errors
    import http.client
    import urllib.error
    http.client._MAXLINE = 1 << 24
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or k == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or (5, 20, 60)[min(k, 2)]
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
            if k == tries - 1:
                raise
            wait = (5, 20, 60)[min(k, 2)]
        time.sleep(wait)


LIC_URL = {"by": "/licenses/by/", "by-nc": "/licenses/by-nc/", "zero": "/publicdomain/zero/"}


def license_ok(url, lics=None):
    u = url.lower()
    return any(LIC_URL[k] in u for k in (lics or LIC_URL)) and "-sa" not in u and "-nd" not in u


def seconds(ps):
    try:
        p = [float(v) for v in ps.split(":")]
        return p[0] * 60 + p[1] if len(p) == 2 else p[0] * 3600 + p[1] * 60 + p[2]
    except Exception:
        return 0


def scan(ex_ids, ex_users, lic=None, pause=0.0):
    """Yield the eligible uploads tagged acappella, newest first (API order), optionally only one license."""
    o, n, fails = 0, 10, 0
    while True:
        # the X-JSON header repeats the body: past ~100 kB of it (~10 uploads, fewer with long descriptions) the
        # reply comes back empty or the connection is reset, so halve the page on failure, down to 1 upload
        try:
            page = json.loads(get(API.format(n=n, o=o) + (f"&lic={lic}" if lic else ""), tries=2))
            n, fails = min(10, n * 2), 0
        except Exception as e:
            if n > 1:
                n //= 2
                continue
            fails += 1
            print("query failed at", o, type(e).__name__, flush=True)
            if fails > 20:
                return
            o += 1                     # one upload too big to list: skip it
            time.sleep(2)
            continue
        if not page:
            return
        o += len(page)
        time.sleep(pause)
        for u in page:
            user = u["user_name"]
            tags = u.get("upload_tags", "").lower()
            if u["upload_id"] in ex_ids or user.lower() in ex_users:
                continue
            if any(b in tags for b in BAD_TAGS) or not license_ok(u.get("license_url", ""), [lic] if lic else None):
                continue
            mp3 = [f for f in u["files"] if f.get("file_format_info", {}).get("default-ext") == "mp3"
                   and f["file_format_info"].get("sr") in ("44k", "48k")
                   and 40 <= seconds(f["file_format_info"].get("ps", "0")) <= 420 and f.get("file_extra", {}).get("sha1")]
            if not mp3:
                continue
            f = mp3[0]
            yield {"id": u["upload_id"], "url": f["download_url"], "sha1": f["file_extra"]["sha1"],
                   "page": u["file_page_url"], "title": u["upload_name"], "artist": u.get("user_real_name") or user,
                   "user": user, "uploaded": u.get("upload_date_format", ""), "license": u["license_name"],
                   "license_url": u["license_url"], "seconds": seconds(f["file_format_info"]["ps"])}
        print(f"scanned {o} uploads{' (' + lic + ')' if lic else ''}", flush=True)


def candidates(maxn, per_user, lics=None):
    """At most per_user takes per uploader, newest first. Without lics: one stream over all licenses, stopping at
    maxn (the original behaviour). With lics: one API query per license (&lic=), merged newest first, so the cap
    counts only takes of the selected licenses."""
    man = json.load(open(CC30))["takes"]
    ex_ids, ex_users = {t["id"] for t in man}, {t["user"].lower() for t in man}
    if lics:
        pool = sorted((t for lic in lics for t in scan(ex_ids, ex_users, lic, pause=1.0)), key=lambda t: -t["id"])
    else:
        pool = scan(ex_ids, ex_users)
    out, per = [], {}
    for t in pool:
        if per.get(t["user"], 0) >= per_user:
            continue
        per[t["user"]] = per.get(t["user"], 0) + 1
        out.append(t)
        if len(out) >= maxn:
            break
    print(f"{len(out)} candidates, {sum(t['seconds'] for t in out) / 3600:.1f} h, {len(per)} uploaders", flush=True)
    return out


def active_span(x, maxlen=60.0):
    """The <= maxlen s span with the most 100 ms frames above -40 dB re the loudest, silences > 1 s cut out."""
    w = SR // 10
    n = x.shape[1] // w
    e = (x[:, :n * w] ** 2).reshape(2, n, w).sum((0, 2))
    act = e > e.max() * 1e-4
    L = min(n, int(maxlen * 10))
    c = np.concatenate([[0], np.cumsum(act)])
    s = int(np.argmax(c[L:] - c[:n - L + 1]))
    keep = act[s:s + L].copy()
    out, i = np.zeros(L, bool), 0
    while i < L:                                   # keep silences up to 1 s (and 0.2 s margins of longer ones)
        j = i
        while j < L and not keep[j]:
            j += 1
        if j > i:
            out[i:j] = j - i <= 10
            if j - i > 10:
                out[i:i + 2] = out[max(i, j - 2):j] = True
            i = j
        else:
            out[i] = True
            i += 1
    frames = np.nonzero(out)[0] + s
    y = x[:, (frames[:, None] * w + np.arange(w)).ravel()]
    return y, s / 10, float(act[s:s + L].mean())


def chunks(y, n):
    """The loudest and the quietest mostly-active n-sample windows (1 s steps)."""
    w = SR // 10
    starts = list(range(0, max(1, y.shape[1] - n + 1), SR))
    stats = []
    for a in starts:
        seg = y[:, a:a + n]
        k = seg.shape[1] // w
        e = (seg[:, :k * w] ** 2).reshape(2, k, w).sum((0, 2))
        stats.append((float(e.sum()), float((e > e.max() * 1e-3).mean())))
    loud = starts[int(np.argmax([s[0] for s in stats]))]
    ok = [i for i, s in enumerate(stats) if s[1] >= 0.7] or list(range(len(stats)))
    quiet = starts[min(ok, key=lambda i: stats[i][0])]
    return loud, quiet


def process(t, d, sep, prior=None, maxlen=60.0):
    import soundfile as sf
    out = os.path.join(d, f"{t['id']}.flac")
    if prior and prior.get(t["id"], "kept") != "kept":           # resumed run: dropped before, do not refetch
        return dict(t, drop=prior[t["id"]])
    if os.path.exists(out):                                      # resumed run: kept before, re-screen the file
        try:
            y = sf.read(out, dtype="float32")[0].T.copy()
            return screen(t, y, None, 1.0, sep, out) if maxlen <= 60 else screen_windows(t, y, None, 1.0, sep, out)
        except Exception:                                        # cut off by the interruption: fetch again
            os.remove(out)
    mp3 = os.path.join(os.path.normpath(d) + "_raw", f"{t['id']}.mp3")
    try:
        if not os.path.exists(mp3):
            data = get(t["url"], timeout=300)
            open(mp3, "wb").write(data)
        h = hashlib.sha1(open(mp3, "rb").read()).digest()
        if base64.b32encode(h).decode() != t["sha1"]:
            return dict(t, drop="sha1 mismatch")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", mp3, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                             capture_output=True, check=True, timeout=300).stdout
        x = np.frombuffer(raw, np.float32).reshape(-1, 2).T
    except Exception as e:
        return dict(t, drop=f"fetch/decode: {type(e).__name__}")
    finally:
        if os.path.exists(mp3):
            os.remove(mp3)
    if x.shape[1] < 10 * SR or np.abs(x).max() < 1e-3:
        return dict(t, drop="too short or silent")
    del raw
    y, start, act = active_span(x, maxlen)
    del x
    return screen(t, y, start, act, sep, out) if maxlen <= 60 else screen_windows(t, y, start, act, sep, out)


def screen_windows(t, y, start, act, sep, out, win=60.0):
    """Whole-take mode (--maxlen > 60): split the silence-cut take into equal windows of <= win s and screen each
    exactly like a 60 s span (screen()); keep the passing windows, joined with 10 ms fades. A take whose windows
    all pass is kept whole. Most whole takes fail somewhere (count-in beats, an instrumental intro or break), and
    dropping the take, or keeping it whole, would lose clean singing or keep instruments."""
    import soundfile as sf
    L = y.shape[1]
    k = max(1, int(np.ceil(L / (win * SR) - 1e-9)))
    edges = np.linspace(0, L, k + 1).astype(int)
    wins, keep = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        r = screen({}, y[:, a:b], None, 1.0, sep, None)
        wins.append({"at_s": round(a / SR, 2), "s": round((b - a) / SR, 2), "below_100hz": r.get("below_100hz"),
                     "kimv2_loud": r.get("kimv2_vocal_frac_loud"), "kimv2_quiet": r.get("kimv2_vocal_frac_quiet"),
                     "drop": r.get("drop")})
        if r.get("kept"):
            keep.append((a, b))
    t = dict(t, total_s=round(L / SR, 2), windows=wins, **({} if start is None else {"start": start, "active": round(act, 3)}))
    if not keep:
        drops = [w["drop"] for w in wins]
        return dict(t, drop=max(set(drops), key=drops.count))
    f = int(0.01 * SR)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    parts = []
    for i, (a, b) in enumerate(keep):
        seg = y[:, a:b].copy()
        if i > 0 and a != keep[i - 1][1]:
            seg[:, :f] *= ramp
        if i < len(keep) - 1 and b != keep[i + 1][0]:
            seg[:, -f:] *= ramp[::-1]
        parts.append(seg)
    z = np.concatenate(parts, 1)
    if not os.path.exists(out):
        sf.write(out, (z / max(np.abs(z).max(), 1.0)).T, SR, subtype="PCM_16")
    return dict(t, kept=True, kept_s=round(z.shape[1] / SR, 2),
                span="full" if len(keep) == k else f"{len(keep)}/{k} windows")


def screen(t, y, start, act, sep, out):
    import soundfile as sf
    spec = np.abs(np.fft.rfft(y.mean(0))) ** 2
    low = float(spec[: int(100 * len(spec) / (SR / 2))].sum() / spec.sum())
    del spec
    t = dict(t, kept_s=round(y.shape[1] / SR, 2), below_100hz=round(low, 4),
             **({} if start is None else {"start": start, "active": round(act, 3)}))
    if low >= 0.12:
        return dict(t, drop="energy below 100 Hz")
    n = sep.chunk
    if y.shape[1] < n:
        return dict(t, drop="span shorter than a separator chunk")
    fr = []
    for a in chunks(y, n):
        seg = np.ascontiguousarray(y[:, a:a + n], np.float32)
        v = sep(seg)
        fr.append(float((v ** 2).sum() / max((seg ** 2).sum(), 1e-12)))
    t = dict(t, kimv2_vocal_frac_loud=round(fr[0], 4), kimv2_vocal_frac_quiet=round(fr[1], 4))
    if min(fr) < 0.95:
        return dict(t, drop="kimv2 keeps < 0.95 (instruments)")
    if out and not os.path.exists(out):
        sf.write(out, (y / max(np.abs(y).max(), 1.0)).T, SR, subtype="PCM_16")
    return dict(t, kept=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=450)
    ap.add_argument("--per_user", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", help="log of an interrupted run: its '<id> <outcome>' lines are reused")
    ap.add_argument("--lic", help="comma list of by,by-nc,zero: query each license separately (&lic=) and keep only "
                    "those; the per-uploader cap then counts only them (default: all three in one stream)")
    ap.add_argument("--maxlen", type=float, default=60.0, help="longest span kept per take, s; above 60 (420 = "
                    "whole takes) the span is screened in <= 60 s windows and the passing windows are kept")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.normpath(a.out) + "_raw", exist_ok=True)
    prior = {}
    if a.resume:
        for line in open(a.resume):
            p = line.strip().split(" ", 1)
            if len(p) == 2 and p[0].isdigit():
                prior[int(p[0])] = "kept" if p[1].startswith("kept") else p[1]     # "kept full" / "kept 60 s fallback"
    os.environ.setdefault("ORT_ARENA", "0")      # ten Kim Vocal 2 sessions with arenas grew past 25 GB
    lics = a.lic.split(",") if a.lic else None
    assert not lics or all(k in LIC_URL for k in lics), lics
    cache = os.path.join(a.out, f"candidates_{a.lic or 'all'}_{a.per_user}_{a.max}.json")   # reruns do not re-scan the API
    if os.path.exists(cache):
        cands = json.load(open(cache))
    else:
        cands = candidates(a.max, a.per_user, lics)
        json.dump(cands, open(cache, "w"))
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    from models import load_model
    seps = [load_model("kimv2", 2) for _ in range(a.workers)]   # one ORT session per worker thread
    import threading
    tl = threading.local()
    pool_ix = iter(range(a.workers))
    lock = threading.Lock()

    jl = os.path.join(a.out, "manifest.jsonl")
    done = {}
    if os.path.exists(jl):
        for line in open(jl):
            r = json.loads(line)
            done[r["id"]] = r
    logged = set(done)

    def job(t):
        if t["id"] in done:                                      # finished in an earlier run into this dir: reuse
            return done[t["id"]]
        if not hasattr(tl, "sep"):
            with lock:
                tl.sep = seps[next(pool_ix)]
        r = process(t, a.out, tl.sep, prior, a.maxlen)
        print(t["id"], r.get("drop", "kept"), r.get("span", ""), flush=True)
        with lock:                                               # partial results usable before the run ends
            if t["id"] not in logged:                            # (a resumed run does not repeat earlier lines)
                logged.add(t["id"])
                with open(jl, "a") as f:
                    f.write(json.dumps(r) + "\n")
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(job, cands))
    for r in res:
        r["attribution"] = (f"\"{r['title']}\" by {r['artist']} (ccMixter: {r['user']}), {r['page']}, "
                            f"license {r['license']} ({r['license_url']})")
    kept = [r for r in res if r.get("kept")]
    json.dump({"about": __doc__.strip(), "kept": len(kept), "kept_hours": round(sum(r["kept_s"] for r in kept) / 3600, 2),
               "takes": res}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    drops = {}
    for r in res:
        drops[r.get("drop", "kept")] = drops.get(r.get("drop", "kept"), 0) + 1
    print(json.dumps({"candidates": len(res), "kept": len(kept), "kept_h": round(sum(r["kept_s"] for r in kept) / 3600, 2),
                      "users": len({r["user"] for r in kept}), "outcomes": drops}))
    open(os.path.join(a.out, "READY"), "w").write(f"{len(kept)} takes\n")


if __name__ == "__main__":
    main()
