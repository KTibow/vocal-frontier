"""Keeps heavy work honest on a shared machine: one heavy job at a time, enough free memory before it starts, pinned to one thread per
physical core. Every command that times or scores goes through here, so nobody has to remember a wrapper. VF_LOCK and VF_MIN_MEM_MB override the defaults."""
import fcntl
import os
import time

MIN_MEM_MB = int(os.environ.get("VF_MIN_MEM_MB", 4800))     # the browser reference alone peaks near 4 GB over official.mjs's 1200 MB floor
WAIT_S = int(os.environ.get("VF_MEM_WAIT_S", 1800))
_held = None


def available_mb():
    with open("/proc/meminfo") as f:
        return int(next(l for l in f if l.startswith("MemAvailable")).split()[1]) // 1024


def physical_cores(n=4):
    """One logical CPU per physical core, up to n: timings on two hyperthreads of one core mean nothing."""
    seen, out = set(), []
    for c in sorted(int(d[3:]) for d in os.listdir("/sys/devices/system/cpu") if d[:3] == "cpu" and d[3:].isdigit()):
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                sib = f.read().strip()
        except OSError:
            continue
        if sib not in seen:
            seen.add(sib)
            out.append(c)
    return out[:n]


def lock_and_wait(why):
    """Takes the machine-wide lock, then waits for memory. Held until the process exits, so it covers everything the command starts."""
    global _held
    if _held is None:
        path = os.environ.get("VF_LOCK", os.path.expanduser("~/.cache/vf-timing.lock"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _held = open(path, "w")
        try:
            fcntl.flock(_held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"{why}: another heavy job holds {path}, waiting", flush=True)
            fcntl.flock(_held, fcntl.LOCK_EX)
    t0 = time.time()
    if available_mb() < MIN_MEM_MB:
        print(f"{why}: waiting for {MIN_MEM_MB} MB free, {available_mb()} MB now. VF_MIN_MEM_MB changes the limit.", flush=True)
    while available_mb() < MIN_MEM_MB:
        if time.time() - t0 > WAIT_S:
            raise SystemExit(f"{why}: only {available_mb()} MB free after {WAIT_S} s, and it needs {MIN_MEM_MB}. Close something, or set VF_MIN_MEM_MB if this machine can't reach it")
        time.sleep(10)


def release():
    """For a command that goes on to wait on a person, such as a preview server."""
    global _held
    if _held is not None:
        _held.close()
        _held = None
