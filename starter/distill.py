"""Distil a small MSST Mel-band RoFormer from teacher-labelled excerpts (train/label.py output).

    python train/distill.py <student config.yaml> [--arch cpunet|msst|pooled] <label dir> <out dir> [--steps N] [--bs B] [--lr LR]
                            [--musdb data/musdb_train] [--init ckpt]

Batches mix three kinds of example, each a random crop of the config's chunk length:
  - song: an excerpt's mixture with the teacher's vocals as target;
  - remix (p=0.5): teacher vocals of excerpt A over excerpt B's accompaniment (B's mixture minus B's teacher
    vocals), each with a random gain, so the student sees more vocal/backing pairings than there are songs;
  - true stems (p=--p_true, if --musdb is given): MUSDB18-HQ *train* mixtures with their real vocals;
  - a cappella (p=--p_acap, if --acap is given): a true a cappella crop (gain -6..+3 dB) over an excerpt's teacher
    accompaniment (mixture minus teacher vocals), target the a cappella.
Loss is MSST's own (waveform L1 + multi-resolution STFT L1). AdamW, warmup + cosine, bf16 autocast, EMA 0.999.
Every --eval_every steps: SDR of the EMA model against the teacher on 32 held-out label crops, logged to
<out>/log.jsonl; EMA weights saved to <out>/ema.ckpt. --arch pooled: musetric's PooledMelBandRoformer from a
train/pooled_mbr.py config (trained like msst; with --init the val is also logged at step 0).
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "data", "ext", "MSST"))
sys.path.insert(0, HERE)


def load_pairs(d, limit=None):
    from concurrent.futures import ThreadPoolExecutor

    def one(sid):
        p = os.path.join(d, sid)
        if not os.path.exists(os.path.join(p, "vocals.flac")):
            return None
        m, _ = sf.read(os.path.join(p, "mixture.flac"), dtype="int16")
        v, _ = sf.read(os.path.join(p, "vocals.flac"), dtype="int16")
        n = min(len(m), len(v))
        return m[:n].T.copy(), v[:n].T.copy()
    with ThreadPoolExecutor(16) as ex:                     # libsndfile releases the GIL
        return [r for r in ex.map(one, sorted(os.listdir(d))[:limit]) if r is not None]


def load_cached(d):
    """Equal-length label excerpts as one int16 memmap [N, 2 (mix, voc), 2, L] at <d>.npy, built on first use, so
    several trainers share one page-cache copy instead of ~30 GB of private arrays each."""
    path = os.path.normpath(d) + ".npy"
    if not os.path.exists(path):
        pairs = load_pairs(d)
        L = min(m.shape[1] for m, _ in pairs)
        arr = np.lib.format.open_memmap(path + ".tmp.npy", mode="w+", dtype=np.int16, shape=(len(pairs), 2, 2, L))
        for i, (m, v) in enumerate(pairs):
            arr[i, 0], arr[i, 1] = m[:, :L], v[:, :L]
        arr.flush()
        del arr, pairs
        os.replace(path + ".tmp.npy", path)
    arr = np.load(path, mmap_mode="r")
    return [(arr[i, 0], arr[i, 1]) for i in range(len(arr))]


def crop(pair, C, rnd):
    m, v = pair
    s = rnd.randrange(0, max(1, m.shape[1] - C))
    f = lambda x: np.pad(x[:, s:s + C], ((0, 0), (0, max(0, C - (x.shape[1] - s))))).astype(np.float32) / 32768
    return f(m), f(v)


def batch(pairs, true, C, bs, rnd, p_remix, p_true, acap=(), p_acap=0.0):
    X, Y = [], []
    p_acap = p_acap if acap else 0.0
    for _ in range(bs):
        r = rnd.random()
        if true and r < p_true:
            m, v = crop(rnd.choice(true), C, rnd)
        elif r < p_true + p_acap:                               # a true a cappella over a teacher accompaniment
            a = rnd.choice(acap)
            v, _ = crop((a, a), C, rnd)
            mb, vb = crop(rnd.choice(pairs), C, rnd)
            gv, ga = 10 ** (rnd.uniform(-6, 3) / 20), 10 ** (rnd.uniform(-3, 3) / 20)
            v = v * gv
            m = v + (mb - vb) * ga
        elif r < p_true + p_acap + p_remix:
            ma, va = crop(rnd.choice(pairs), C, rnd)
            mb, vb = crop(rnd.choice(pairs), C, rnd)
            gv, ga = 10 ** (rnd.uniform(-6, 3) / 20), 10 ** (rnd.uniform(-3, 3) / 20)
            v = va * gv
            m = v + (mb - vb) * ga
        else:
            m, v = crop(rnd.choice(pairs), C, rnd)
        if rnd.random() < 0.5:                                  # channel swap
            m, v = m[::-1], v[::-1]
        g = 10 ** (rnd.uniform(-6, 3) / 20)
        peak = max(np.abs(m).max() * g, 1e-6)
        if peak > 1:
            g /= peak
        X.append(m * g)
        Y.append(v * g)
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


def pooled_sdr(ref, est):
    """SDR pooled over crops whose reference is voiced (a near-silent target makes a per-crop SDR meaningless)."""
    e = ref.pow(2).sum((1, 2))
    keep = e > e.max() * 1e-3
    return float(10 * torch.log10(e[keep].sum() / (ref - est)[keep].pow(2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--musdb")
    ap.add_argument("--p_true", type=float, default=0.15)
    ap.add_argument("--p_remix", type=float, default=0.5)
    ap.add_argument("--acap", help="dir of true a cappella .flac files (train/acap_fetch.py)")
    ap.add_argument("--p_acap", type=float, default=0.3)
    ap.add_argument("--acap_limit", type=int, help="use a seeded random subset of this many a cappella files (RAM)")
    ap.add_argument("--mem_frac", type=float, help="torch.cuda.set_per_process_memory_fraction, for a shared GPU")
    ap.add_argument("--init")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--hours", type=float, default=99)
    ap.add_argument("--save_every", type=int, default=0, help="also keep <out>/ema_s<step>.ckpt every N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="cpunet", choices=["cpunet", "msst", "pooled"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.arch == "cpunet":
        import yaml
        from cpunet import CpuNet
        cfg = yaml.safe_load(open(a.cfg))
        model, C = CpuNet(**cfg["model"]), int(cfg["audio"]["chunk_size"])
    elif a.arch == "msst":
        from utils.settings import get_model_from_config
        model, cfg = get_model_from_config("mel_band_roformer", a.cfg)
        model.use_torch_checkpoint = True          # MelBand Kim-width students don't fit 24 GB at batch 4 otherwise
        C = int(cfg.audio.chunk_size)
    if a.arch == "pooled":
        from pooled_mbr import build, load_cfg
        model, C = build(a.cfg), int(load_cfg(a.cfg)["audio"]["chunk_size"])
        model.use_torch_checkpoint = True
    msst = a.arch in ("msst", "pooled")
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    dev = "cuda"
    if a.mem_frac:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    model = model.to(dev).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M chunk {C / 44100:.2f}s", flush=True)
    pairs = load_cached(a.labels)
    rnd = random.Random(a.seed)
    rnd.shuffle(pairs)
    val, pairs = pairs[:32], pairs[32:]
    true = load_pairs(a.musdb) if a.musdb else []
    acap = []
    if a.acap:
        import glob
        files = sorted(glob.glob(os.path.join(a.acap, "*.flac")))
        if a.acap_limit:
            files = sorted(random.Random(a.seed).sample(files, min(a.acap_limit, len(files))))
        for f in files:
            try:                                                # skip a file still being written by acap_fetch.py
                acap.append(sf.read(f, dtype="int16")[0].T.copy())
            except Exception:
                print("skipped", f, flush=True)
        print(f"{len(acap)} a cappellas, {sum(x.shape[1] for x in acap) / 44100 / 3600:.2f} h", flush=True)
    print(f"{len(pairs)} train excerpts, {len(val)} val, {len(true)} true-stem tracks", flush=True)
    vr = random.Random(1)
    VX, VY = zip(*[crop(p, C, vr) for p in val])
    VX, VY = torch.from_numpy(np.stack(VX)), torch.from_numpy(np.stack(VY))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    # MSST's forward calls view_as_complex, which rejects bf16: those models train in fp16 with a grad scaler, as MSST does
    amp = torch.float16 if msst else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    sched = lambda s: min(1, (s + 1) / a.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    t0, losses = time.time(), []
    if a.init:                                                  # starting point of a teacher-initialised student
        with torch.no_grad(), torch.autocast("cuda", dtype=amp):
            est = torch.cat([ema(VX[i:i + 8].to(dev)).float().cpu() for i in range(0, len(VX), 8)])
        rec = {"step": 0, "val_sdr_teacher": pooled_sdr(VY, est)}
        print(json.dumps(rec), flush=True)
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        x, y = batch(pairs, true, C, a.bs, rnd, a.p_remix, a.p_true, acap, a.p_acap)
        with torch.autocast("cuda", dtype=amp):
            yt = y.to(dev)
            loss = model(x.to(dev), target=yt[:, None] if msst else yt)   # MSST: [b, stems, s, t]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        losses.append(loss.item())
        last = step + 1 == a.steps or (time.time() - t0) / 3600 > a.hours
        if (step + 1) % a.eval_every == 0 or last:
            with torch.no_grad(), torch.autocast("cuda", dtype=amp):
                est = torch.cat([ema(VX[i:i + 8].to(dev)).float().cpu() for i in range(0, len(VX), 8)])
            rec = {"step": step + 1, "loss": float(np.mean(losses)), "val_sdr_teacher": pooled_sdr(VY, est),
                   "lr": opt.param_groups[0]["lr"], "h": round((time.time() - t0) / 3600, 3)}
            losses = []
            print(json.dumps(rec), flush=True)
            with open(os.path.join(a.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            torch.save(ema.state_dict(), os.path.join(a.out, "ema.ckpt"))
            if a.save_every and (step + 1) % a.save_every == 0:
                torch.save(ema.state_dict(), os.path.join(a.out, f"ema_s{step + 1}.ckpt"))
        if last:
            break


if __name__ == "__main__":
    main()
