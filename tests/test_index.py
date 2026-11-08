import math

import pytest

from vf import index as I

REF_Q = 11.0


def test_reference_scores_one():
    r = {d: 1.0 for d in I.ALL_DEVICES}
    assert I.speed_index(r) == pytest.approx(1.0)


def test_weights_are_quarter_cpu_eighth_gpu():
    r = {d: 1.0 for d in I.ALL_DEVICES}
    assert I.speed_index({**r, "native": 16.0}) == pytest.approx(2.0)     # 16^(1/4)
    assert I.speed_index({**r, "vega": 256.0}) == pytest.approx(2.0)      # 256^(1/8)


def test_failure_is_floor_and_slower_than_floor_is_no_better_than_failing():
    assert I.device_ratio(None, 2.0) == I.FLOOR
    assert I.device_ratio(0.0, 2.0) == I.FLOOR
    assert I.device_ratio(0.01, 2.0) == I.FLOOR   # 0.005x would be worse than failing without the clamp
    assert I.device_ratio(1.0, 2.0) == 0.5


def test_a_device_failure_costs_a_lot_but_is_not_fatal():
    good = {d: 4.0 for d in I.ALL_DEVICES}
    bad = {**good, "valhall": I.device_ratio(None, 2.0)}
    assert I.speed_index(bad) == pytest.approx(4.0 ** (7 / 8) * I.FLOOR ** (1 / 8))
    assert I.speed_index(bad) == pytest.approx(0.578 * I.speed_index(good), abs=0.01)


def test_missing_device_only_allowed_when_provisional():
    r = {"native": 2.0, "browser": 2.0, "vega": 8.0}
    with pytest.raises(ValueError):
        I.speed_index(r)
    assert I.speed_index(r, provisional=True) == pytest.approx(4.0)


def rect(s, q):
    return (I.to_plane(s), q)


def test_area_of_one_point():
    x = I.attribute([{"s": 1.0, "q": 11.0, "owner": "baseline"}], REF_Q)
    assert x["total"] == pytest.approx((0 - math.log2(I.AREA_ORIGIN)) * 1.0)


def test_floor_excludes_points():
    x = I.attribute([{"s": 100.0, "q": 9.99, "owner": "a", "t": 1}], REF_Q)
    assert x["total"] == 0 and x["area"] == {}


def test_brief_example_40_10_30_pays_250():
    # Start at 40 units, we add 10, three contestants add 10 each without overlapping: $250 each and we keep $250.
    # Unit squares in a plane with y from the floor: build them as adjacent columns.
    x0 = I.to_plane(I.AREA_ORIGIN)
    def s_at(dx):
        return 2 ** (x0 + dx)
    pts = [{"s": s_at(40), "q": REF_Q, "owner": "baseline"},                  # 40 wide, 1 dB tall: 40 units
           {"s": s_at(50), "q": REF_Q, "owner": "maintainers"},               # +10
           {"s": s_at(60), "q": REF_Q, "owner": "alice", "t": 1},             # +10
           {"s": s_at(70), "q": REF_Q, "owner": "bob", "t": 2},               # +10
           {"s": s_at(80), "q": REF_Q, "owner": "carol", "t": 3}]             # +10
    a = I.attribute(pts, REF_Q)["area"]
    assert a["baseline"] == pytest.approx(40) and a["maintainers"] == pytest.approx(10)
    p = I.payouts(a)
    assert p["shares"] == {"alice": 250.0, "bob": 250.0, "carol": 250.0}
    assert p["kept"] == 250.0 and p["per_unit"] == pytest.approx(25)


def test_copying_earns_nothing_and_first_wins():
    pts = [{"s": 1.0, "q": REF_Q, "owner": "baseline"},
           {"s": 4.0, "q": 12.0, "owner": "alice", "t": 1},
           {"s": 4.0, "q": 12.0, "owner": "copycat", "t": 2}]
    a = I.attribute(pts, REF_Q)["area"]
    assert a["copycat"] == 0 and a["alice"] > 0
    assert I.payouts(a)["shares"] == {"alice": 1000.0}


def test_near_copy_is_a_copy_and_covers_nothing():
    pts = [{"s": 1.0, "q": REF_Q, "owner": "baseline"},
           {"s": 4.0, "q": 12.0, "owner": "alice", "t": 1},
           {"s": 4.1, "q": 12.05, "owner": "noisy", "t": 2},       # 2.5% faster and 0.05 dB better: noise
           {"s": 4.1, "q": 12.05, "owner": "third", "t": 3}]       # measured against alice too, not against the copy
    x = I.attribute(pts, REF_Q)
    assert x["area"]["noisy"] == 0 and x["area"]["third"] == 0
    assert I.attribute(pts[:2] + [{"s": 4.3, "q": 12.05, "owner": "real", "t": 2}], REF_Q)["area"]["real"] > 0


def test_a_late_maintainer_entry_queues_behind_earlier_contestants():
    pts = [{"s": 1.0, "q": REF_Q, "owner": "baseline"},
           {"s": 4.0, "q": 12.0, "owner": "alice", "t": 5},
           {"s": 4.0, "q": 12.5, "owner": "maintainers", "t": 9}]
    x = I.attribute(pts, REF_Q)["area"]
    assert x["alice"] == pytest.approx(I.attribute(pts[:2], REF_Q)["area"]["alice"])


def test_overlap_pays_only_for_the_new_part():
    pts = [{"s": 1.0, "q": REF_Q, "owner": "baseline"},
           {"s": 2.0, "q": 12.0, "owner": "a", "t": 1},
           {"s": 4.0, "q": 12.0, "owner": "b", "t": 2}]
    a = I.attribute(pts, REF_Q)["area"]
    w = I.to_plane(1.0) - I.to_plane(I.AREA_ORIGIN)
    assert a["a"] == pytest.approx((w + 1) * 2 - w * 1)
    assert a["b"] == pytest.approx(1 * 2)


def test_being_dominated_shrinks_our_share_towards_zero():
    pts = [{"s": 1.0, "q": REF_Q, "owner": "baseline"},
           {"s": 1.2, "q": 11.1, "owner": "maintainers"},
           {"s": 64.0, "q": 16.0, "owner": "a", "t": 1}]
    p = I.payouts(I.attribute(pts, REF_Q)["area"])
    assert p["shares"]["a"] > 980


def test_input_order_does_not_matter():
    pts = [{"s": 1.0, "q": REF_Q, "owner": "baseline"},
           {"s": 2.0, "q": 12.0, "owner": "a", "t": 5},
           {"s": 3.0, "q": 11.5, "owner": "b", "t": 5},
           {"s": 1.5, "q": 13.0, "owner": "maintainers"}]
    assert I.attribute(pts, REF_Q)["area"] == pytest.approx(I.attribute(pts[::-1], REF_Q)["area"])


def test_nobody_scores_means_we_keep_it_and_dust_is_not_sent():
    assert I.payouts({"baseline": 40.0})["kept"] == 1000.0
    p = I.payouts({"baseline": 40.0, "maintainers": 1000.0, "a": 0.5})
    assert p["shares"] == {} and p["kept"] == 1000.0


def test_frontier():
    a, b, c = {"s": 1, "q": 3}, {"s": 2, "q": 2}, {"s": 1, "q": 2}
    assert I.frontier([a, b, c]) == [b, a]
