"""The two axes and the payout. Pure functions, no I/O; tests/test_index.py pins every rule in the README.

Speed index S (x axis): how many times faster than the reference HTDemucs, on a mix of devices.

    r_d = xrt on device d / reference xrt on device d, measured in one session, clamped to at least FLOOR
    CPU = geomean(r_native, r_browser)
    GPU = geomean(r_vega, r_adreno, r_valhall, r_gen12lp)
    S   = geomean(CPU, GPU)

A failure on a device (crash, out of memory, wrong output, slower than the floor) scores FLOOR there.
Running slower than FLOOR scores the same as failing, so nobody gains by making a model fail on purpose.

Quality index Q (y axis): mean SDR in dB over three 50-clip sets. Points under REFERENCE_Q - 1 dB don't count.

Area: x is log2(S) from log2(AREA_ORIGIN), y is Q from the floor, so one unit is one doubling of speed times one dB.
"""
import math

FLOOR = 0.05            # the least a device can score, which is also what failing on it scores
AREA_ORIGIN = 1 / 16    # where area starts on the x axis. Separate from FLOOR: moving it changes the chart and nobody's device score
QUALITY_MARGIN_DB = 1.0
POT_USD = 1000.0
DUST_USD = 1.0
COPY_SPEED_RATIO = 1.05   # timing wobbles by about this much, so anything closer than this to an earlier entry counts as a copy
COPY_Q_DB = 0.1

CPU_DEVICES = ("native", "browser")
GPU_DEVICES = ("vega", "adreno", "valhall", "gen12lp")
ALL_DEVICES = CPU_DEVICES + GPU_DEVICES


def geomean(values):
    values = list(values)
    if not values:
        raise ValueError("geomean of nothing")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def device_ratio(xrt, ref_xrt):
    """Speed on one device relative to the reference on the same device. None means it failed there."""
    if xrt is None or not (xrt > 0):
        return FLOOR
    if not (ref_xrt and ref_xrt > 0):
        raise ValueError("a device without a reference timing can't be scored")
    return max(FLOOR, xrt / ref_xrt)


def speed_index(ratios, provisional=False):
    """ratios: {device: r_d}, already clamped. A missing device raises, unless provisional, where the
    missing devices are left out of their group. A provisional index is only for display; it never pays."""
    missing = [d for d in ALL_DEVICES if d not in ratios]
    if missing and not provisional:
        raise ValueError(f"unmeasured devices: {missing}")
    groups = []
    for group in (CPU_DEVICES, GPU_DEVICES):
        have = [ratios[d] for d in group if d in ratios]
        if have:
            groups.append(geomean(have))
    if not groups:
        raise ValueError("no devices measured")
    return geomean(groups)


def quality_floor(reference_q):
    return reference_q - QUALITY_MARGIN_DB


def eligible(q, reference_q):
    return q >= quality_floor(reference_q)


def _union_area(points, x0, y0):
    """Area of the union of the rectangles [x0, x] x [y0, y]. points: (x, y)."""
    best = 0.0
    area = 0.0
    # sweep from the fastest point down: each point adds its width at the best height seen so far
    pts = sorted(((x, y) for x, y in points if x > x0 and y > y0), key=lambda p: -p[0])
    for i, (x, y) in enumerate(pts):
        best = max(best, y)
        next_x = pts[i + 1][0] if i + 1 < len(pts) else x0
        area += (x - next_x) * (best - y0)
    return area


def to_plane(s):
    return math.log2(s)


def frontier(points):
    """Points not dominated by another. A point is dominated if another is at least as fast and at least as good
    and strictly better on one. points: dicts with s and q. Returned fastest first."""
    out = []
    for p in points:
        if not any(o is not p and o["s"] >= p["s"] and o["q"] >= p["q"] and (o["s"] > p["s"] or o["q"] > p["q"])
                   for o in points):
            out.append(p)
    return sorted(out, key=lambda p: -p["s"])


def attribute(points, reference_q):
    """Who gets credit for which area, chronologically.

    points: dicts with s, q, owner, t. The starting board has t 0: owner "baseline" (other people's existing models, earns
    nothing) first, then "maintainers" (our own models, keeps its share of the pot). After that everyone queues by t,
    maintainers included. Each point earns only the area nobody before it already covers. A point after the starting board
    that isn't more than COPY_SPEED_RATIO faster or COPY_Q_DB better than some earlier point is a copy: it earns nothing and
    covers nothing. Ties in t are broken by the name of the owner, so the result never depends on the order of the input.
    """
    x0, y0 = to_plane(AREA_ORIGIN), quality_floor(reference_q)

    def rank(p):
        t = p.get("t", 0)
        order = {"baseline": 0, "maintainers": 1}.get(p["owner"], 2) if not t else 2
        return (order, t, str(p["owner"]), -p["s"], -p["q"])

    covered, seen, area, steps = [], [], {}, []
    total = 0.0
    for p in sorted((p for p in points if eligible(p["q"], reference_q)), key=rank):
        if rank(p)[0] == 2 and any(e["s"] * COPY_SPEED_RATIO >= p["s"] and e["q"] + COPY_Q_DB >= p["q"] for e in seen):
            area.setdefault(p["owner"], 0.0)
            steps.append({**p, "gain": 0.0, "copy": True})
            continue
        seen.append(p)
        covered.append((to_plane(p["s"]), p["q"]))
        now = _union_area(covered, x0, y0)
        gain = max(0.0, now - total)
        total = now
        area[p["owner"]] = area.get(p["owner"], 0.0) + gain
        steps.append({**p, "gain": gain})
    return {"area": area, "total": total, "steps": steps}


def payouts(area, pot=POT_USD, dust=DUST_USD):
    """Split the pot in proportion to area, over everything but the baseline. Maintainers keep their part (it is not
    paid to anyone). Amounts under dust aren't sent, and their share stays with the maintainers."""
    payable = {k: v for k, v in area.items() if k != "baseline" and v > 0}
    denom = sum(payable.values())
    if denom <= 0:
        return {"per_unit": 0.0, "shares": {}, "kept": pot}
    per_unit = pot / denom
    shares = {k: v * per_unit for k, v in payable.items() if k != "maintainers"}
    shares = {k: round(v, 2) for k, v in shares.items() if v >= dust}
    return {"per_unit": per_unit, "shares": shares, "kept": round(pot - sum(shares.values()), 2)}
