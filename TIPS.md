# What's already been tried

Notes from building the starting frontier. Speed ratios are on a Ryzen 5 3501U with a Radeon Vega 8 unless noted.

## Where the frontier is thin

| | why it looks open |
|---|---|
| Fast and over 12 dB on every device | MDX-Nets are 13 dB but slow in WASM. HTDemucs-lite is fast but 10.1-10.4 dB. |
| 2× to 5× HTDemucs | Nothing sits between HTDemucs-tuned (about 2×, 11.0 dB) and HTDemucs-lite-11k (about 5×, 10.4 dB). More overlap on lite-11k barely helps: overlap 1.5 gained about 0.2 dB on a 6-clip-per-set probe, at about 1.45× the cost. A lite model with a higher cutoff, or a smaller MDX-Net, is the likelier route. |
| 0.32× to 0.68× | Between MelBand Kim in WASM (14.25 dB) and Voc_FT at overlap 1.5 (13.36 dB), nothing. A faster MelBand Kim, or a model between the two in size, would cover a lot of it. |
| RoFormers on WebGPU | The good ones exceed memory or storage-buffer limits on iGPUs. Musetric's kara2 student fails with 17 storage buffers against a limit of 16. |
| MelBand Kim in a browser | It fits, but slowly: `melband-kim-wasm` runs at 0.16× realtime. Making it faster, and getting it onto more GPUs, is open. See below. |
| Phones and Arm laptops | Little has been measured there. Valhall and Adreno carry a quarter of the speed index's exponent between them. |
| Native int8 on AVX2 | Our pruned int8 files ran 9-16% slower than fp32 natively on this CPU, which has no VNNI, and 11-15% faster in the browser. Nobody has made it win natively here. |

## MelBand Kim in a browser

- **WASM, as one graph, is out of reach:** 3.9-5.4 GB in Chrome, and fp16 weights still end in `bad_alloc`.
- **Cut into stages, it fits.** `melband-kim-wasm` uses `starter/slice_melband.py` to make one graph per stage and feeds each transformer 10 bands or 134 frames at a time. It peaks at about 3.3 GB, runs at 0.16× realtime without overlap, and scores within 0.002 dB of the native file on all 8 transfer clips.
- **One graph on a GPU took four fixes:**
  1. Keep the weights in an external-data file passed to `InferenceSession.create`. ONNX Runtime's WASM heap stays near 60 MB, where an embedded file grows it to about 1 GB that never shrinks, and the lowest free memory rises by about 800 MB.
  2. Move the file to the worker instead of cloning it.
  3. Rebuild `Split` and `Concat` nodes with more than 15 inputs or outputs as trees, to fit the 16 storage buffer limit.
  4. Keep the RMS norms after the band split in fp32. Their 1e-24 epsilon flushes to zero in fp16, so a silent token divides 0 by 0.
- **Check any fp16 RoFormer against its fp32 parent.** With the transformer RMSNorms in fp16, x² is zero for |x| under about 2e-4 and the epsilon is zero too, so quiet passages lose their norm, with no NaN and no error. One test clip lost 7.8 dB. `starter/fp32_norms.py` moves those norms to fp32 and cut the worst clip to 0.8 dB.
- **One graph or a chain:** the chain got within 0.15 dB, at 0.45× against 0.58× realtime on Vega, so neither is exact on our quietest test clip.
- **Another route:** Better Lyrics Composer's [PR](https://github.com/better-lyrics/composer/pull/461) by adaliea, merged on 2026-11-03, runs 14 graphs as separate sessions with a wait on the GPU after each. It reports 4.5 GB peak on a UHD 630. That setup is the `melband-kim` baseline here, and it may suit GPUs with less room. The board has its speed.
- **OpenVINO** returned near-silence for one fused export (RMS 1.8e-6 against ONNX Runtime's 0.088 on the same chunk), with no error, though it scores correctly on `melband-kim-o1`'s native file. Check its output before trusting a score.
- Other GPUs are open.

## Runtimes

- **Native:** OpenVINO beat ONNX Runtime by about 1.6× on HTDemucs, but ONNX Runtime won for pruned Voc_FT on this CPU. Measure both, more than once: with `dim_t` 512 Voc_FT, OpenVINO at 1 process × 4 threads ran at 0.36× realtime in most runs and 0.73× in a few, on the same file and clip. ONNX Runtime 1 × 4 gave 0.72-0.74× every time, with identical output, but used about 5.1 GB against OpenVINO's 2.9 GB.
- **Process layout:** 4 processes × 1 thread can beat 1 × 4 by up to 30%, but RoFormers don't gain and memory multiplies: fp32 MDX at 4 × 1 needed 9.5 GB, the HTDemucs family 4-5 GB and int8 3.2 GB. Half our recipes stayed on 1 × 4.
- **Browser CPU:** nearly all MDX time is GEMM already at the SIMD128 ceiling, so graph rewrites gain little. You need fewer FLOPs. onnxruntime-web speed moved ±60% between versions on the same file, which is why runtimes are pinned.
- **Browser files:** fp16 compute is about 1.15× *slower* than fp32 in WASM, so use fp16 storage with fp32 compute. GELU fusion (`com.microsoft` Gelu) is free speed. Dynamic int8 MatMul works in WASM, but dynamic int8 Conv is 2.5× slower. Winograd, static batching and Transpose hoisting gained nothing. Unfused attention exports hit `bad_alloc`.
- **Weights beside the graph:** the site now splits every ONNX file for you, so you get this without doing anything. Output is byte-identical. It cuts ONNX Runtime's WASM heap after session create: HTDemucs 513 to 287 MB, Voc_FT 182 to 99 MB, duality student 900 to 612 MB. On Vega HTDemucs drops from 520 to 369 MB and Voc_FT p55 from 54 to 16 MB. The heap during a run barely moves in WASM: HTDemucs stays near 2.4 GB and Voc_FT near 1.15 GB, since activations dominate there.
- **Browser memory:** each worker builds its own ONNX Runtime memory arena, about 700 MB even for a 10 MB model. On the Ryzen, four workers went over the 4 GB limit, and two gained 8-23% but ran right up against it, and one model crashed. One worker with four threads is the safe start.
- **WebGPU:** ORT-web 1.23 was fastest for MDX. int8 is 2-3× *slower* than fp32 on WebGPU, so ship a different file there. Batching chunks and 2 workers didn't help: throughput stayed about the same and memory grew. Mixed fp16/fp32 graphs use much more GPU memory than either pure version, and Adreno caps a single buffer at 128 MiB. The JS host took 16-27% of end-to-end time and padding 5-18%, so a shorter `dim_t` may be worth trying; we haven't.
- **GPUs with a 128 MiB binding limit:** WebGPU only promises 128 MiB in one storage binding, and Adreno gives exactly that. An MDX-Net at full resolution holds [1, 48, T, 3072] tensors: 144 MiB at `dim_t` 512 in fp16, or at 256 computing in fp32, and ONNX Runtime fails with "Binding size ... is larger than the maximum storage buffer binding size". `vf build` handles this for you: when a GPU graph's largest tensor is that big, it adds a copy whose full-resolution layers run on two overlapping time pieces (`vf/tile.py`), checks it matches the original, and the page uses it only on a GPU whose limit the original doesn't fit. Graphs with layers that read across time, like a transformer, can't be cut this way, so keep their tensors under about 85 MiB.
- **fp16 on WebGPU:** usually about 1.8× over fp32 with the same SDR. Keep statistics and shape nodes in fp32; HTDemucs's variance overflows to inf in fp16. Its ConvTranspose also has to stay in fp32 on ORT-web, or the time branch breaks.
- **Linux Chromium** silently falls back to SwiftShader without `--enable-features=Vulkan --use-angle=vulkan`. The site refuses to score software adapters.

## Overlap

Overlap buys little quality for a lot of speed. Overlap 2 costs about 2× (1.89-2.29× here). From 1 to 2, Voc_FT gains 0.21 dB and MelBand Kim 0.06 dB. On a log-speed axis that's rarely worth it. Quality is measured at the overlap you declare.

Without overlap, longer chunks are the cheaper win. MDX files ship with `dim_t` 256 (about 6 s), but the input's time axis can be set to 512 by editing the graph, with no retraining. On all 150 clips it gained 0.13-0.22 dB on every model we tried:

| model, no overlap | `dim_t` 256 | `dim_t` 512 |
|---|---|---|
| Voc_FT | 13.08 | 13.27 |
| Kim Vocal 2 | 13.03 | 13.16 |
| Voc_FT p85 | 12.52 | 12.71 |
| Voc_FT p70 | 12.46 | 12.59 |
| Voc_FT p70, int8 | 12.11 | 12.24 |
| Voc_FT p55 | 11.78 | 12.00 |
| Voc_FT p55, int8 | 11.30 | 11.48 |

That makes Voc_FT without overlap about as good as it was at overlap 1.5 or 1.75 with the shorter chunks, at much less time. Overlap 1.11 and `dim_t` 1024 added nothing over 512 on a 6-clip-per-set probe, and the output gain (`compensate`) is already within 0.005 dB of its best.

## Training

- **Distilling from scratch is slow.** Starting from existing weights and cutting them worked far better than training a small model from zero.
- **Pruning and self-distilling MDX-Nets:** Voc_FT at 55% of its compute kept 11.8 dB. See `starter/kv2_ft.py` and `starter/distill.py`.
- **Cutting HTDemucs:** a spectral branch cut at 11 kHz, a half-rate transformer and no waveform decoder: about 2.5× faster than tuned HTDemucs at −0.56 dB, self-distilled toward the original. Training can't recover what the cut loses: the index caps at about 10.07 for 8 kHz, so don't aim lower. See `starter/arch_htd.py` and `starter/htd_sd.py`.
- **Students can learn to gate whole chunks** as non-vocal, which validation loss misses and a hard clip's SDR catches. Judge candidates on real stems.
- **Pruning Voc_FT:** recalibrate BatchNorm and then freeze it, because train-mode stats explode in eval. Keeping 50% of channels was a dead end at about 8 dB. Distilling HTDemucs toward MelBand Kim didn't carry over to true stems.
- **Quantization-aware int8:** keep the few full-resolution layers in fp32. All-int8 lost another 0.15 dB or more. See `starter/qat_mdx.py`.
- **Post-training int8 without QAT** broke HTDemucs outright: its output on a real chunk matched its own fp32 output to only 13.7 dB, where a faithful conversion matches almost exactly.
- **Other dead ends:** a small CPU-first net from scratch reached only 9.3 dB after 134k steps. SCNet as published lost 4.4 dB on `private30`. Musetric's students trained on `cc30` takes, so fine-tuning them breaks the data rule.
- **Real a cappellas in training** help on `cc30`-style vocals, but the wrong takes hurt: many ccMixter uploads tagged "acappella" contain instruments or heavy effects. `starter/acap_fetch.py` screens them.

## Scoring traps

- **`musdb30` isn't enough.** Models trained on MUSDB can look better than they are; `cc30` and `private30` use vocals they haven't seen. A gain that comes from one set usually doesn't hold.
- **Same weights, different numerics, different output.** Check the on-device SDR on the page, not just the speed.
- **Don't trust a timing made next to another job.** `vf speed` interleaves models with HTDemucs for this reason, and even so a busy machine moves ratios.
