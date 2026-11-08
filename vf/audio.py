"""Audio I/O and an STFT/iSTFT that match torch.stft/istft (center=True, reflect pad, periodic Hann)."""
import subprocess

import numpy as np
import soundfile as sf

SR = 44100


def load(path, sr=SR):
    """-> float32 [2, samples] at sr. Anything ffmpeg reads; wav/flac at sr are read directly."""
    try:
        x, fsr = sf.read(path, dtype="float32", always_2d=True)
        if fsr == sr:
            x = x.T
            return np.ascontiguousarray(np.repeat(x, 2, 0) if x.shape[0] == 1 else x[:2])
    except sf.LibsndfileError:
        pass
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"],
                         capture_output=True, check=True).stdout
    return np.ascontiguousarray(np.frombuffer(raw, np.float32).reshape(-1, 2).T)


def save(path, x, sr=SR):
    sf.write(path, np.asarray(x).T, sr, subtype="PCM_16" if path.endswith(".flac") else "FLOAT")


def hann(n):
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def stft(x, n_fft, hop):
    """x [..., T] -> complex64 [..., n_fft//2+1, frames]."""
    w = hann(n_fft)
    p = n_fft // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(p, p)], mode="reflect")
    n = 1 + (xp.shape[-1] - n_fft) // hop
    fr = np.lib.stride_tricks.as_strided(xp, xp.shape[:-1] + (n, n_fft),
                                         xp.strides[:-1] + (xp.strides[-1] * hop, xp.strides[-1]))
    return np.fft.rfft(fr * w, axis=-1).astype(np.complex64).swapaxes(-1, -2)


def istft(z, n_fft, hop, length):
    """z [..., F, frames] -> float32 [..., length]."""
    w = hann(n_fft)
    fr = np.fft.irfft(z.swapaxes(-1, -2), n=n_fft, axis=-1).astype(np.float32) * w
    n = fr.shape[-2]
    total = n_fft + hop * (n - 1)
    out = np.zeros(z.shape[:-2] + (total,), np.float32)
    env = np.zeros(total, np.float32)
    for i in range(n):
        out[..., i * hop:i * hop + n_fft] += fr[..., i, :]
        env[i * hop:i * hop + n_fft] += w * w
    p = n_fft // 2
    out, env = out[..., p:p + length], env[p:p + length]
    return out / np.where(env > 1e-8, env, 1.0)
