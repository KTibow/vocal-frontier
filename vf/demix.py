"""Chunked overlap-add over a whole song. One definition for quality and speed, so the two can't disagree.
Chunks every chunk/overlap samples, linear fades of chunk/10 (none at the song's ends), normalized by the summed window.
site/src/stitch.js is the same in the browser."""
import multiprocessing as mp
import os
import queue
import time

import numpy as np

from .hosts import make_core


def plan(n_samples, chunk, overlap):
    step = max(1, int(chunk / overlap))
    border = chunk - step
    pad = border if border > 0 and n_samples > 2 * border else 0
    total = n_samples + 2 * pad
    starts = []
    i = 0
    while i < total:
        starts.append(i)
        if i + chunk >= total:
            break
        i += step
    return starts, pad, total


def _part(x, i, chunk):
    part = x[:, i:i + chunk]
    n = part.shape[-1]
    if n < chunk:
        part = np.pad(part, ((0, 0), (0, chunk - n)), mode="reflect" if n > chunk // 2 + 1 else "constant")
    return np.ascontiguousarray(part), n


def demix(x, chunk, overlap, run_all):
    """run_all(parts) -> vocals for each part, in order. Returns vocals [2, T]."""
    T0 = x.shape[-1]
    starts, pad, total = plan(T0, chunk, overlap)
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    fade = chunk // 10
    win = np.ones(chunk, np.float32)
    if fade:
        win[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
        win[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
    parts = [_part(x, i, chunk) for i in starts]
    out = np.zeros_like(x)
    cnt = np.zeros(total, np.float32)
    for k, (i, y) in enumerate(zip(starts, run_all([p for p, _ in parts]))):
        n = parts[k][1]
        w = win.copy()
        if i == 0:
            w[:fade] = 1
        if i + chunk >= total:
            w[-fade:] = 1
        out[:, i:i + n] += y[:, :n] * w[:n]
        cnt[i:i + n] += w[:n]
    out /= np.maximum(cnt, 1e-8)
    return out[:, pad:pad + T0] if pad else out


def _die_with_parent():
    """Without this, killing a run leaves its workers holding gigabytes. Linux only; elsewhere the daemon flag has to do."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except OSError:
        pass


def _worker(path, layout, threads, runtime, cpus, tasks, results):
    _die_with_parent()
    if cpus:
        os.sched_setaffinity(0, cpus)
    core = make_core(path, layout, threads, runtime)
    results.put(("ready", core.chunk))
    while True:
        job = tasks.get()
        if job is None:
            return
        if job[0] == "song":
            core.song(job[1])
            continue
        k, part = job
        results.put((k, core(part)))


class Separator:
    """processes x threads on a CPU list. Workers keep the model loaded, so loading isn't timed."""

    def __init__(self, path, layout, overlap, runtime="onnxruntime", processes=1, threads=4, cpus=None):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} doesn't exist. For HTDemucs, run: vf fetch-reference. For another entry: vf fetch <entry>.")
        self.overlap, self.processes = overlap, processes
        ctx = mp.get_context("spawn")
        self.tasks = [ctx.Queue() for _ in range(processes)]
        self.results = ctx.Queue()
        groups = [None] * processes
        if cpus:
            n = len(cpus) // processes
            groups = [set(cpus[i * n:(i + 1) * n]) for i in range(processes)]
        self.procs = [ctx.Process(target=_worker, args=(path, layout, threads, runtime, groups[i], self.tasks[i], self.results),
                                  daemon=True) for i in range(processes)]
        for p in self.procs:
            p.start()
        self.chunk = self._ready()

    def _ready(self):
        """Waits for every worker to load. A worker that dies says so now, instead of after a long silent timeout."""
        chunks, deadline = [], time.monotonic() + 600
        while len(chunks) < len(self.procs):
            try:
                chunks.append(self.results.get(timeout=2))
            except queue.Empty:
                if any(not p.is_alive() for p in self.procs):
                    raise RuntimeError("a separator worker died while loading. See the error above. "
                                       "A script using Separator needs its code under `if __name__ == \"__main__\":`.") from None
                if time.monotonic() > deadline:
                    raise TimeoutError("separator workers took over 10 minutes to load") from None
        return chunks[0][1]

    def _run_all(self, parts):
        next_part, busy, out = 0, 0, [None] * len(parts)
        free = list(range(self.processes))
        owner = {}
        while next_part < len(parts) or busy:
            while free and next_part < len(parts):
                w = free.pop()
                owner[next_part] = w
                self.tasks[w].put((next_part, parts[next_part]))
                next_part += 1
                busy += 1
            k, y = self.results.get(timeout=3600)
            out[k] = y
            free.append(owner.pop(k))
            busy -= 1
        return out

    def __call__(self, x):
        for q in self.tasks:
            q.put(("song", x))
        return demix(x, self.chunk, self.overlap, self._run_all)

    def timed(self, x):
        t = time.perf_counter()
        y = self(x)
        return y, time.perf_counter() - t

    def close(self):
        for q in self.tasks:
            q.put(None)
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
