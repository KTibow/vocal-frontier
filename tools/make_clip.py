"""Builds the public test clip used for the on-device correctness check: a CC BY a cappella over a CC BY instrumental.
Neither track is in any scored set. Run once; the result is committed in site/clip."""
import http.client, json, os, subprocess, sys, tempfile
import numpy as np

SR, LEN = 44100, 30 * 44100
VOCAL = ("/content/jlbrock44/jlbrock44_-_Reasons_Vocals_1.mp3", "\"Reasons Vocals\" by spinningmerkaba", "https://ccmixter.org/files/jlbrock44/68790", "CC BY 4.0")
INST = ("/content/AlexBeroza/AlexBeroza_-_Drive.mp3", '"Drive" by Alex Beroza', "https://ccmixter.org/files/AlexBeroza/43098", "CC BY 3.0")


def fetch(path):
    http.client._MAXLINE = 10 ** 7
    c = http.client.HTTPSConnection("ccmixter.org", timeout=120)
    c.request("GET", path, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/150 Safari/537.36", "Referer": "https://ccmixter.org/", "Accept": "*/*"})
    r = c.getresponse()
    if r.status != 200:
        raise SystemExit(f"{path}: {r.status}")
    return r.read()


def decode(data):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(data); f.flush()
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f.name, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


def best_window(x):
    env = np.convolve((x ** 2).sum(0)[::441], np.ones(3000), "same")
    return max(0, min(int(np.argmax(env)) * 441 - LEN // 2, x.shape[1] - LEN))


out = sys.argv[1]
v, i = decode(fetch(VOCAL[0])), decode(fetch(INST[0]))
vs = best_window(v)
v = v[:, vs:vs + LEN]
i = i[:, 44100 * 30:44100 * 30 + LEN]
v *= 0.3 / np.sqrt((v ** 2).mean()) * 0.5
i *= 0.3 / np.sqrt((i ** 2).mean()) * 0.5
peak = max(np.abs(v + i).max(), 1e-6)
g = min(1.0, 0.98 / peak)
v, i = v * g, i * g
mix = v + i
os.makedirs(out, exist_ok=True)
to16 = lambda a: np.round(np.clip(a, -1, 1) * 32767).astype("<i2").T.tobytes()
# the check compares against the quantized truth; the mix is exactly the sum of what ships, up to rounding
open(f"{out}/mix.i16", "wb").write(to16(mix))
open(f"{out}/vocals.i16", "wb").write(to16(v))
json.dump({"sr": SR, "channels": 2, "samples": LEN, "vocal_start_s": vs / SR,
           "credits": [{"role": "vocals", "title": VOCAL[1], "url": VOCAL[2], "license": VOCAL[3]},
                       {"role": "instrumental", "title": INST[1], "url": INST[2], "license": INST[3]}],
           "note": "Mixed and level-matched for this benchmark."}, open(f"{out}/clip.json", "w"), indent=1)
print("rms vocals/inst dB", 10 * np.log10((v ** 2).mean() / (i ** 2).mean()), "peak", np.abs(mix).max(), "vocal start", vs / SR)
