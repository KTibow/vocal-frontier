"""Teacher labels for distillation: MelBand Kim (fp16, GPU) separates one excerpt per training song.

    python train/label.py <song list> <audio dir> <out dir> [--shard i/n] [--secs 40] [--overlap 2]

Each song gets one excerpt starting uniformly in 15-85% of its length (seeded by song id), decoded from the
original full-band file. Writes <out>/<id>/mixture.flac and vocals.flac (the teacher's), 44.1 kHz stereo.
"""
import argparse
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "data", "ext", "MSST"))
SR = 44100


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy()


@torch.inference_mode()
def demix_batched(model, x, C, overlap, dev):
    """Same chunking and fades as bench/separate.py demix, all chunks of x in one fp16 batch."""
    step, fade, border = int(C / overlap), C // 10, C - int(C / overlap)
    T0 = x.shape[-1]
    if T0 > 2 * border:
        x = np.pad(x, ((0, 0), (border, border)), mode="reflect")
    T = x.shape[-1]
    starts = []
    i = 0
    while True:
        starts.append(i)
        if i + C >= T:
            break
        i += step
    parts = np.stack([np.pad(x[:, s:s + C], ((0, 0), (0, max(0, C - (T - s))))) for s in starts])
    ys = []
    for b in range(0, len(parts), 8):
        with torch.autocast("cuda", dtype=torch.float16):
            ys.append(model(torch.from_numpy(parts[b:b + 8]).to(dev)).float().cpu().numpy())
    ys = np.concatenate(ys)
    win = np.ones(C, np.float32)
    win[:fade], win[-fade:] = np.linspace(0, 1, fade), np.linspace(1, 0, fade)
    out, cnt = np.zeros_like(x), np.zeros(T, np.float32)
    for k, s in enumerate(starts):
        L = min(C, T - s)
        w = win.copy()
        if k == 0:
            w[:fade] = 1
        if k == len(starts) - 1:
            w[-fade:] = 1
        out[:, s:s + L] += ys[k][:, :L] * w[:L]
        cnt[s:s + L] += w[:L]
    out /= np.maximum(cnt, 1e-8)
    return out[:, border:border + T0] if T0 > 2 * border else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("songs")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--secs", type=float, default=40)
    ap.add_argument("--overlap", type=float, default=2)
    ap.add_argument("--cfg", default=os.path.join(ROOT, "data/models/melkim/config.yaml"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "data/models/melkim/MelBandRoformer.ckpt"))
    a = ap.parse_args()
    from utils.settings import get_model_from_config
    model, cfg = get_model_from_config("mel_band_roformer", a.cfg)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd))
    dev = "cuda"
    model = model.to(dev).eval()
    C = int(cfg.audio.chunk_size)
    k, n = map(int, a.shard.split("/"))
    songs = [l.strip() for l in open(a.songs) if l.strip()][k::n]
    for j, f in enumerate(songs):
        sid = f.rsplit(".", 1)[0]
        d = os.path.join(a.out, sid)
        if os.path.exists(os.path.join(d, "vocals.flac")):
            continue
        try:
            x = decode(os.path.join(a.audio, f))
        except subprocess.CalledProcessError:
            print("decode failed", f, flush=True)
            continue
        L = int(a.secs * SR)
        if x.shape[1] < L + SR:
            continue
        rnd = random.Random(sid)
        s = int(rnd.uniform(0.15, 0.85) * (x.shape[1] - L))
        mix = x[:, s:s + L]
        voc = demix_batched(model, mix, C, a.overlap, dev)
        os.makedirs(d, exist_ok=True)
        sf.write(os.path.join(d, "mixture.flac"), mix.T, SR, subtype="PCM_16")
        sf.write(os.path.join(d, "vocals.tmp.flac"), np.clip(voc, -1, 1).T, SR, subtype="PCM_16")
        os.replace(os.path.join(d, "vocals.tmp.flac"), os.path.join(d, "vocals.flac"))
        if j % 50 == 0:
            print(f"[{k}] {j}/{len(songs)}", flush=True)


if __name__ == "__main__":
    main()
