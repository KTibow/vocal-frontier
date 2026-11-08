"""Vendored from musetric/lab @ 64ab58494b760c0bf724e873c16d32a2f60ee950, musetric_toolkit/separate_audio/roformer/
(https://github.com/musetric/lab), MIT License, Copyright (c) 2025 Vladlen Popelenkov: see LICENSE in this
directory. mel_band_roformer.py and attend.py are in turn based on lucidrains' BS-RoFormer (MIT).

Local changes, each marked "vsb:" in the code:
  - imports made package-relative;
  - mel_band_roformer.py: forward() zeroes the DC bin before the iSTFT when zero_dc (as MSST's MelBandRoformer does,
    so the training loss and outputs match MelBand Kim's);
  - mel_band_roformer.py: a [b, 1, s, t] target (MSST's layout) is squeezed to the [b, s, t] recon before the L1 term;
  - pooled_mel_band_roformer.py: every per-pair setting is optional (defaults = the teacher's shape), mask_rank=None
    keeps the teacher's full mask estimator (mask_estimator_depth as given), and use_torch_checkpoint
    checkpoints each pair and the mask estimator while training; repeat_interleave written as expand + flatten
    (same values; the TorchScript ONNX exporter loses ranks after repeat_interleave).
The duality student's weights (huggingface.co/musetric/mel-band-roformer-duality-student) are Apache-2.0.
"""
