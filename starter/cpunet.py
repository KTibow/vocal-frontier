"""CpuNet: a band-split axial transformer shaped for ONNX Runtime on CPU (and WASM).

Same idea as Mel-/BS-RoFormer (band tokens, attention across time per band and across bands per frame,
per-band mask heads), with the parts that profiled as overhead in ORT removed:
  - contiguous, non-overlapping bands: the mask is a concat, no gather/scatter-add;
  - band-split and mask-head layers batched by band width: one batched MatMul per width group instead of
    one small Gemm per band;
  - no rotary embeddings (they cost Slice/Split/Mul/Concat per layer): a learned band embedding, and a
    depthwise conv over time before each time-attention block for relative position;
  - hop 882 by default: half the frames of MelBand's 441.

Core I/O matches musetric's cores (bench/models.MaskCore): real STFT [b, (f s), t, 2] -> complex masks, same shape.
forward(raw_audio [b, 2, T], target=None) -> vocals, or MSST's loss (waveform L1 + multi-res STFT L1) with target.
"""
import torch
import torch.nn.functional as F
from torch import nn

# (bands, bins per band): 47 bands over 1025 bins (n_fft 2048): ~86 Hz bands up to 1.4 kHz, coarser above
DEFAULT_BANDS = ((16, 4), (12, 8), (8, 16), (6, 32), (4, 64), (1, 289))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class GroupLinear(nn.Module):
    """n independent Linear(k -> d) applied to x [..., n, k] as one batched MatMul."""

    def __init__(self, n, k, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, k, d) * k ** -0.5)
        self.b = nn.Parameter(torch.zeros(n, d))

    def forward(self, x):                                   # [B, n, k] -> [B, n, d]
        return torch.matmul(x.transpose(0, 1), self.w).transpose(0, 1) + self.b


class Block(nn.Module):
    def __init__(self, dim, heads, ff_mult, conv=0):
        super().__init__()
        self.heads = heads
        self.conv = nn.Conv1d(dim, dim, conv, padding=conv // 2, groups=dim) if conv else None
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))

    def forward(self, x):                                   # [B, L, d]
        if self.conv is not None:
            x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(a.transpose(1, 2).reshape(B, L, D))
        return x + self.ff(self.n2(x))


class CpuNet(nn.Module):
    def __init__(self, dim=128, depth=6, heads=4, ff_mult=2, mask_hidden=256, n_fft=2048, hop=882,
                 bands=DEFAULT_BANDS, conv=5, multi_stft_windows=(4096, 2048, 1024, 512, 256), multi_stft_hop=147):
        super().__init__()
        assert sum(n * w for n, w in bands) == n_fft // 2 + 1
        self.n_fft, self.hop, self.bands = n_fft, hop, tuple(tuple(b) for b in bands)
        self.nb = sum(n for n, _ in self.bands)
        self.split_norm = nn.ModuleList([RMSNorm(w * 4) for _, w in self.bands])
        self.split = nn.ModuleList([GroupLinear(n, w * 4, dim) for n, w in self.bands])
        self.band_emb = nn.Parameter(torch.zeros(self.nb, dim))
        self.layers = nn.ModuleList([nn.ModuleList([Block(dim, heads, ff_mult, conv), Block(dim, heads, ff_mult)])
                                     for _ in range(depth)])
        self.final_norm = RMSNorm(dim)
        self.mask1 = nn.ModuleList([GroupLinear(n, dim, mask_hidden) for n, _ in self.bands])
        self.mask2 = nn.ModuleList([GroupLinear(n, mask_hidden, w * 4) for n, w in self.bands])
        self.msw, self.msh = multi_stft_windows, multi_stft_hop

    def core(self, stft_repr):                              # [B, (f s), t, 2] -> masks, same shape
        B, _, T, _ = stft_repr.shape
        x = stft_repr.reshape(B, -1, 2, T, 2).permute(0, 3, 1, 2, 4)       # [B, t, f, s, c]
        x = x.reshape(B * T, -1, 4)                                        # [(B t), f, (s c)]
        toks, f0 = [], 0
        for (n, w), norm, lin in zip(self.bands, self.split_norm, self.split):
            g = x[:, f0:f0 + n * w].reshape(B * T, n, w * 4)
            toks.append(lin(norm(g)))
            f0 += n * w
        h = torch.cat(toks, 1) + self.band_emb                             # [(B t), N, d]
        N, D = h.shape[1], h.shape[2]
        for tb, bb in self.layers:
            h = h.view(B, T, N, D).transpose(1, 2).reshape(B * N, T, D)
            h = tb(h)
            h = h.view(B, N, T, D).transpose(1, 2).reshape(B * T, N, D)
            h = bb(h)
        h = self.final_norm(h)
        outs, b0 = [], 0
        for (n, w), l1, l2 in zip(self.bands, self.mask1, self.mask2):
            outs.append(l2(F.gelu(l1(h[:, b0:b0 + n]))).reshape(B * T, n * w, 4))
            b0 += n
        m = torch.cat(outs, 1).view(B, T, -1, 2, 2).permute(0, 2, 3, 1, 4)  # [B, f, s, t, c]
        return m.reshape(B, -1, T, 2)

    def stft(self, audio):                                  # [B, 2, L] -> [B, (f s), t, 2]
        B, S, L = audio.shape
        z = torch.stft(audio.reshape(B * S, L), self.n_fft, self.hop, window=torch.hann_window(self.n_fft, device=audio.device),
                       return_complex=True)
        z = torch.view_as_real(z).view(B, S, z.shape[1], z.shape[2], 2)    # [B, s, f, t, c]
        return z.permute(0, 2, 1, 3, 4).reshape(B, -1, z.shape[3], 2)

    def forward(self, raw_audio, target=None):
        B, S, L = raw_audio.shape
        rep = self.stft(raw_audio)
        m = self.core(rep).float()
        y = torch.view_as_complex(rep.float().contiguous()) * torch.view_as_complex(m.contiguous())
        y = y.view(B, -1, 2, y.shape[-1]).permute(0, 2, 1, 3).reshape(B * S, -1, y.shape[-1])
        y = torch.istft(y, self.n_fft, self.hop, window=torch.hann_window(self.n_fft, device=y.device), length=L).view(B, S, L)
        if target is None:
            return y
        loss = F.l1_loss(y, target)
        yt, tt = y.reshape(B * S, L), target.reshape(B * S, L)
        for w in self.msw:
            kw = dict(n_fft=max(w, 2048), hop_length=self.msh, win_length=w, window=torch.hann_window(w, device=y.device),
                      return_complex=True)
            loss = loss + F.l1_loss(torch.view_as_real(torch.stft(yt, **kw)), torch.view_as_real(torch.stft(tt, **kw)))
        return loss
