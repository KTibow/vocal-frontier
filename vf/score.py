"""SDR over a whole clip, both channels (the MDX challenge's "uSDR"). The index uses this and nothing else."""
import numpy as np

EPS = 1e-10


def sdr(ref, est):
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[:, :n].astype(np.float64), est[:, :n].astype(np.float64)
    return float(10 * np.log10((np.sum(ref ** 2) + EPS) / (np.sum((ref - est) ** 2) + EPS)))
