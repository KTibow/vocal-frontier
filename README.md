# Vocal Frontier

**An open challenge with up to $1000 in prizes: push out the speed-against-quality frontier of vocal separation.**

Separating a vocal lets you sync lyrics, transcribe a song, match samples and make karaoke. Today you trade one for the
other. On a low-power mini PC's CPU, HTDemucs-lite-8k separates a 3.5-minute song in about 30 seconds and scores 10.1 dB.
MelBand Kim scores 14.3 dB but takes over an hour on the same CPU, and as one graph it doesn't fit in a browser's memory. This
challenge pays for closing that gap, and every paid entry publishes its own work under a permissive license.

- **Board and in-browser benchmark:** https://vocal-frontier.kendell.dev/
- **Runs:** November 7, 2026 to 8 pm Pacific on November 21, 2026 (04:00 UTC, November 22). Two weeks.
- **Paid in:** USDC on Base

## How you win

Every entry is a point on a chart. Across is speed, up is quality. A point covers everything to its lower left, and you get
paid for the area you cover that nobody before you did. You don't have to be the fastest or the cleanest.

```
 quality
 (dB)  ▲
       │░░░░░░●                         ● existing models
       │░░░░░░│                         ◆ your entry
       │░░░░░░│███████████████◆         ░ already covered
       │░░░░░░░░░░░░░░░░●█████│         █ new: what you're paid for
       │░░░░░░░░░░░░░░░░│█████│
       │░░░░░░░░░░░░░░░░│█████│         One unit = one doubling of speed × one dB.
       │░░░░░░░░░░░░░░░░░░░░░░░░░░●
       │░░░░░░░░░░░░░░░░░░░░░░░░░░│
       │░░░░░░░░░░░░░░░░░░░░░░░░░░│
 floor ┼────────────────────────────▶ speed (log scale, from 1/16×)
```

**The worked example.** Suppose others' existing models cover 40 units, the maintainers' own models add 10, and three
contestants add 10 each without overlapping. Others' 40 units earn nothing. New area is 10 + 30 = 40 units for $1000, or $25
a unit: each contestant gets $250. The maintainers' $250 is never paid to anyone, and that share shrinks as you add more.

**What pays is area, so gaps pay best.** What one new entry would earn on today's provisional board:

<!-- vf board writes this table -->
| new entry | units | paid |
|---|---|---|
| 11.7×, 10.07 dB (lite-8k, twice as fast) | 0.07 | $28 |
| 16×, 10.5 dB | 0.91 | $289 |
| 8×, 11 dB | 1.44 | $390 |
| 4×, 12 dB | 1.90 | $459 |
| 2×, 13 dB | 1.04 | $317 |
| 0.961×, 14.26 dB (tuned Voc_FT, 1 dB cleaner) | 1.67 | $427 |
| 0.284×, 15.25 dB (MelBand Kim in WASM, 1 dB cleaner) | 2.17 | $492 |
<!-- end of table -->

**The chart starts at 1/16× HTDemucs,** so speed under that earns nothing, however clean. The right side has no limit: the chart
widens to fit the fastest entry.

**Earlier entries keep the area they cover.** Your place in line is the moment your PR first contains a complete entry: a manifest that passes
`vf validate`, with a URL and sha256 for every file. Changing a file, hash or recipe moves you to the time of that change. Adding or
updating your results files doesn't. An
entry earns only the area no earlier entry covers, your own included.

**Copies earn nothing.** Timing and SDR wobble, so an entry that isn't more than 5% faster or 0.1 dB better than an earlier
one counts as a copy: it earns nothing and covers nothing. You can build on someone's public weights if you name them in
`training.txt`, and you earn what clears that margin. Multiple entries are fine: yours add up.

## Quickstart

You need Linux (the only system this has been run on), [uv](https://docs.astral.sh/uv/), Node with pnpm, ffmpeg, Chromium and about 5 GB of disk.

```sh
git clone https://github.com/KTibow/vocal-frontier && cd vocal-frontier
uv sync && pnpm install

uv run vf fetch-data                   # MUSDB18-HQ test + ccMixter takes, then builds the two public sets. Zenodo is slow: allow 90 minutes
uv run vf fetch-reference              # HTDemucs, the 1× reference, which every comparison below needs
uv run vf new my-model                 # creates submissions/my-model/ (manifest.yaml, training.txt) and models/my-model/
# put your ONNX in models/my-model/ and edit the manifest. With one file, delete the fp16 line and point webgpu at fp32

uv run vf preview my-model             # your fork of the site at http://127.0.0.1:8787: your entry on the real board, and what it would earn. Needs fetch-data done
uv run vf quality my-model             # the public score of your native file, which preview uses once it exists
uv run vf transfer my-model            # if your browser or GPU file differs from your native one: does it still score the same?
# upload your files, then fill in each url and sha256 (`sha256sum models/my-model/core.onnx`)
uv run vf validate my-model            # the checks the maintainers run on your PR. It doesn't download your URLs; they do
```

`vf preview` scores your model on 5 clips per set if you haven't run `vf quality`, times it natively, and serves the site with your entry on
the chart. Run your model on the page, on the CPU and GPU tabs, to add this browser and GPU. It's your hardware and a rough score, so it's
a preview: official numbers come from the maintainers' devices. Five clips can be half a dB off, so run `vf quality` before trusting what it says you'd earn.
You can't run `private30`, so preview estimates it from how the other entries' public and private scores compare. Your real Q can move once the maintainers run it.

**Building on an entry that's already here?** `uv run vf fetch <entry>` downloads the files the site serves into your models folder, at the paths in
`baselines/<entry>/manifest.yaml`, for example `models/vocft_t512/core.onnx` with its weights beside it in `core.onnx.data`. Your files belong in your
own folder as one file each, so put the weights back inside as you copy it over:
`python -c 'import onnx; onnx.save(onnx.load("models/vocft_t512/core.onnx"), "models/my-model/core.onnx")'`.
If your weights are unchanged, you don't need `vf quality`: run `vf transfer my-model --parent <entry>`, copy `results/quality/<entry>.json` to
`results/quality/my-model.json`, and add `"inherits": "<entry>"` as a top-level key in it, next to `"sets"`. Your native file has to hold the same
model as that entry's native file: the same graph and weights, however it's stored, so a file rejoined as above counts. Remember that copies earn
nothing, so this only pays if your entry is faster than that one by more than the margin.
Some entries run a native file the site doesn't serve: the HTDemucs-lite, HTDemucs-tuned, duality, MelBand Kim and `-mix` entries. `vf transfer` can't
score their native side for you, so say in your PR which entry you started from and the maintainers run the check.

**Settings** are environment variables. `VF_MODELS` is the folder manifest paths start from (`models/` by default), `VF_DATA` holds the sets (`data/`),
and `VF_CHROME` is your Chromium if it isn't `/usr/bin/chromium`. Commands that time or score take a lock so they don't disturb each other, and wait for
`VF_MIN_MEM_MB` of free memory, 4800 MB by default, since a browser run of a big model peaks near 4 GB. On a smaller machine, set it lower
for small models. A page run saves its progress in `~/.cache/vf-score-progress`, so a stopped run continues.

## The two axes

### Speed index

Times faster than HTDemucs, end to end. That includes the STFT, the stitching and the iSTFT; it excludes download,
session creation and a short warm-up run. Everywhere on the board, **×** means times as fast as HTDemucs. When we mean against the
song's own length, we write **× realtime**: 2× realtime separates a 4-minute song in 2 minutes.

```
r_device = your × realtime on that device / HTDemucs × realtime on that device, same session

CPU = geomean(native, browser)                          both on a Ryzen 5 3501U mini PC (4 cores, 7 GB). geomean(a, b) = √(a·b)
GPU = geomean(Vega 8, Adreno X1, Mali Valhall, Intel Gen12LP)
S   = geomean(CPU, GPU)
```

Vega 8 is that mini PC's GPU, Adreno X1 a Snapdragon X Elite laptop, Mali Valhall a Nothing phone, and Gen12LP an Intel
ChromeOS laptop. Doubling your speed on a CPU path multiplies S by 1.19, and on a GPU by 1.09, however fast you already were.

**Failing costs a lot, but it isn't disqualifying.** A crash, an out-of-memory, a wrong output or anything slower than
**0.05×** HTDemucs scores 0.05× on that device. A run slower than the floor gets the same score as failing, so there's
never a reason to fail on purpose. A failure right after a successful run doesn't count until the next session on that
device fails too; until then the successful run stands, and the board marks it. For a model
that's 4× everywhere:

| fails on | S |
|---|---|
| nothing | 4.00× |
| one GPU | 2.31× |
| all four GPUs | 0.45× |
| browser CPU | 1.34× |

So a model that runs everywhere usually beats a faster one that doesn't. GPUs also cap single buffers: `voc-ft-qat8` failed
on Adreno with a 160 MiB binding against a 128 MiB limit, so check your largest tensor.

**1×** is HTDemucs as the Better Lyrics composer runs it, because it's a separator people already ship (this challenge isn't affiliated with Better Lyrics): 25% overlap (`overlap: 1.3333`) on stock ONNX Runtime. That means fp32
on 4 threads natively, fp32 on **1 thread** in the browser, and fp16 with fp32 ConvTranspose on GPUs, because the fp32 file runs
out of memory on Vega. They score the same. It's in [`baselines/htdemucs`](baselines/htdemucs/manifest.yaml).

### Quality index

Mean SDR in dB over three sets of 50 clips of 30 s, each 44.1 kHz stereo, scored as global SDR per clip, with the chunking
your manifest declares. SDR compares your vocal with the real one: 10·log10 of the real vocal's energy over the error's energy,
over the whole clip. Each extra dB is about a fifth less error energy.

| set | what it is | can you run it? |
|---|---|---|
| `musdb30` | MUSDB18-HQ test, the most vocal 30 s per track | yes |
| `cc30` | ccMixter a cappellas under Creative Commons licenses (most of them non-commercial) over MUSDB test accompaniments. Vocals no MUSDB-trained model has seen | yes |
| `private30` | Produced a cappellas from commercial sample packs over MUSDB test accompaniments | no, the maintainers run it |

**The floor** is 1 dB under HTDemucs (10.0 dB). Under it, a point earns nothing, however fast.

**Quality must be honest about numerics.** If your manifest uses different files per backend (fp32 natively and fp16 on
WebGPU, say), Q is the lowest of them. You score your native file with `vf quality`. For each other file:

- **Same weights** (differing only in runtime, precision, export or stitching): it inherits the native score if it passes the
  **transfer check**. Run through its own pipeline on 8 fixed clips, every clip's SDR has to be within ±0.05 dB of the native file's,
  and their mean within ±0.02 dB. The clips are `PR_-_Happy_Daze` and `Motor_Tapes_-_Shore` from `musdb30`, `c40` and `c17` from `cc30`,
  and four from `private30`. `vf transfer my-model` runs the ones you have and prints each clip. Passing those is `public-pass`; the
  maintainers run the private four and record the verdict in `results/transfer/`, along with the hashes of the files and engine it checked.
- **Fails the check, or different weights:** it's scored in full on the backend it runs on, and Q is the lower of that and native. The
  check errs on the safe side: an fp16 file can fail on one low-SDR clip and still match in full.
- **Already scored in full in another entry:** if that full score's mean came out no more than 0.02 dB under its native score, the same
  file through the same engine needs nothing more, at any overlap.
- **Scored at a lower-quality file** (`scored_at: int8`): that file is checked against itself run natively.

No entry goes on the board without one of these for each such file, baselines included. Changing a file or engine undoes its pass.

**Correct output on every device.** Each device also separates a public test clip and compares SDR with the native
result. More than 0.5 dB off counts as a failure there. This catches kernels that are fast and silently wrong, which
happens: HTDemucs in full fp16 loses its time branch on WebGPU.

## What you submit

A PR adding `submissions/<name>/manifest.yaml`, a `training.txt`, and links to your files. The ONNX files themselves
stay out of the repo: host them yourself (Hugging Face is easiest) with the URL in the manifest.

```yaml
title: My Model
kind: contestant
github: your-login
wallet: "0x…"                              # yours on Base, for USDC: see Prizes
license: { weights: MIT, code: MIT }
description: One or two sentences.
layout: { type: mdx, n_fft: 7680, hop: 1024, dim_f: 3072, dim_t: 256, compensate: 1.021 }
overlap: 1                                 # passes over each sample, from 1 to 8: 1 = none, 1.33 = 25%, 2 = 50%. Same on every device and for quality
files:
  fp32: { path: my-model/core.onnx, url: "https://…", sha256: "…" }
  fp16: { path: my-model/core_f16.onnx, url: "https://…", sha256: "…" }
native:  { file: fp32, runtime: onnxruntime, processes: 1, threads: 4 }   # or openvino; processes × threads ≤ 4
browser: { file: fp32, runtime: wasm@1.30.0, workers: 1, threads: 4 }     # workers × threads ≤ 4
webgpu:  { file: fp16, runtime: webgpu@1.23.0 }
```

`compensate` multiplies the output, because MDX-Nets come out slightly quiet (1.021 for Voc_FT, 1.009 for Kim Vocal 2). `wasm` and
`webgpu` can each be 1.23.0 or 1.30.0; ask in your PR to add a version. Leave out a backend and that backend scores 0.05×.

**Layouts** are the stock host code around your core, for entries without their own engine. There are three, and new architectures usually fit one:

| `layout.type` | core inputs | core outputs | examples |
|---|---|---|---|
| `mdx` | `[1, 4, dim_f, dim_t]` spectrum | vocal spectrum, same shape | Voc_FT, Kim Vocal 2 |
| `htdemucs` | `input` waveform, `x` spectrogram | `output`, `add_67` branches | HTDemucs, HTDemucs-lite |
| `mask` | `[1, 2F, frames, 2]` real STFT | complex masks, same shape | RoFormers |

If yours doesn't fit, propose a layout in `vf/hosts.py`, `site/src/layouts.js` and `vf/validate.py`, and the maintainers will review
it. Otherwise a submission is only YAML and ONNX, with no custom ops. Upload one plain ONNX file with its weights inside, and `sha256` pins that file. For the browser the site splits every ONNX file into a graph and a weights file, so the weights stay out of ONNX Runtime's WASM heap. You don't have to do anything for that, and the output is the same.

### Bring your own engine

Speed work outside the model counts too. A browser or WebGPU recipe can run your own code: a different runtime, a faster STFT,
your own shaders or WASM kernels, or your own chunking and stitching.

```sh
uv run vf new my-model --engine        # copies the starter engine to submissions/my-model/engine/
```

```yaml
browser: { engine: engine/worker.js, file: fp32, runtime: wasm@1.30.0, threads: 4 }   # runtime is passed to your engine as is; file can be any format it reads (.onnx gets split, see the starter)
webgpu:  { engine: engine/worker.js, file: fp16, runtime: webgpu@1.23.0 }
```

[`starter/engine/worker.js`](starter/engine/worker.js) is the stock worker and documents the messages it gets. It's open source under your
entry's code license, and compiled code needs a `SOURCE.md` that rebuilds it. `vf preview` runs your engine exactly as the site will.

- **It runs in a browser worker with no network**, so it can reach this site and nothing else, and the maintainers read it before running it.
- **The test clip is disguised**, at a random level and rotated by a random amount, so an engine can't recognize it.
- **Your engine is public under your entry's code license**, so others can build on it as that license allows. Your entry's place in line stays yours, so being first still pays.
- **Any non-native file gets checked through your engine** by the maintainers: the transfer check above if it has the same weights, otherwise in full. Q is still the lowest across your files.
- **Native stays ONNX Runtime or OpenVINO.** That's code on the maintainers' own machine, and it's already fast.
- **The 1× reference and the starting board stay on stock runtimes**, so 1× doesn't move and old entries don't gain by themselves. The exceptions are MelBand Kim's WebGPU recipes (`melband-kim` and `melband-kim-o1`), which mirror how Better Lyrics Composer runs it, as a chain of 14 sessions. Engines by the maintainers are their own entries, such as `melband-kim-wasm`.

## How entries get scored

1. Your PR passes `vf validate` and commits `results/quality/<name>.json`, which `vf quality` writes, and `results/transfer/<name>.*.json` if you ran `vf transfer`.
2. The maintainers run `private30` and spot-check 10 public clips. If their mean differs from yours by over 0.05 dB in either
   direction, your score gets recomputed in full. The board uses the maintainers' numbers whenever they've computed them.
3. Device runs happen in batches on the six setups, plugged in, about every 3 days and in a last batch after the deadline. Your own speed
   numbers don't count: speed is measured on the same hardware for everyone, interleaved with HTDemucs.
4. The board updates, including what you'd be paid if it ended now.

Up to **5 live entries** per person, because every entry takes real time on real devices. One person gets one set of entries,
and the maintainers can merge accounts. You can replace an entry at any time: the replacement queues at its own time, and its old
area is freed for later entries. Entries must be complete by the deadline: 8 pm Pacific, November 21, 2026.

## Rules

- **Don't train or tune on** MUSDB18-HQ test, the cc30 takes or anything by their uploaders (listed in
  [`tools/data/ccmixter_takes.json`](tools/data/ccmixter_takes.json)), or vocals from commercial sample packs. Publish your
  training list in `training.txt`. Other data and any teacher model are fine.
- **A model that's better on the public sets than `private30` can explain** gets a closer look. Maintainers may ask how it
  was trained and can exclude an entry they believe broke the data rule.
- **License:** what you publish (your weights and code) must be permissive: `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `MPL-2.0`, `CC0-1.0`, `CC-BY-4.0`, `Unlicense` or `Zlib`, written exactly so.
  Beyond the list above we don't police what your data was licensed under, since that's how nearly every model in this field is made. If you start from
  weights without a clear license, say so under `license.upstream`. Two common ones: UVR's README asks developers using its models to "honor the MIT license by
  providing credit to UVR and its developers", and hasn't answered questions about whether that covers Voc_FT's weights. Kim Vocal 2 has no license of its own that we could find.
  Weights under an explicit non-commercial or strong copyleft license (GPL, AGPL) aren't eligible. Your license covers your own work; whatever
  you built on keeps its own terms, so a fine-tune of Voc_FT doesn't make Voc_FT's weights MIT.
- **Your entry stays public.** That's the point of the prize. Keep the URL working for at least a year after the end.
- **Memory:** the Ryzen mini PC has 7 GB. A browser run is killed if free memory drops under 1.2 GB, so plan for about 4 GB there, and about 6 GB natively.

## Prizes

- The pot is up to $1000 in USDC on Base. Contestants split it by area as above when the challenge ends; the share for area the
  maintainers' own models add isn't paid out, and it shrinks as contestants add more.
- Amounts under $1 aren't sent. Payment goes to the wallet in your manifest within two weeks of the final board.
- The wallet is a `0x` address on Base, a network that runs on Ethereum. Any self-custody wallet that supports Base gives you one.
  Use one you control: an exchange deposit address may not accept USDC on Base, and a payment sent there can't be undone.
- If nobody adds any area, nobody gets paid.
- You're responsible for any taxes where you live, and for the prize being legal to accept there. No entry fee, no purchase needed.
  We can't pay anyone we're legally barred from paying, such as people on sanctions lists.
- The maintainer, @KTibow, decides scoring and eligibility, and explains each decision in the PR. Bugs in the harness get
  fixed and everything is rescored, up to the final board; nobody loses a place in line for that.
- Maintainers can enter new models after the start. They queue by time like everyone else's, are never paid, and are marked on the board. Only the starting board comes first.

## What's already been tried

[`TIPS.md`](TIPS.md) has what we learned getting here, including what didn't work. [`starter/`](starter/) has the training code behind
the maintainers' entries: teacher labels, distillation, QAT, MDX fine-tuning. It's reference code; expect to edit paths.

## License

The framework is MIT. Models keep their own licenses, listed on the board. The test clip in `site/clip/` is CC BY, credited in `clip.json`,
and the sets keep their sources' licenses.
