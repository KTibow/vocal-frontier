"""One-time import: scores already measured with the same scorer in the private bench, for models whose numerics match.
Writes results/quality/<name>.json. Rerun `vf quality` to replace any of them."""
import json, os, sys
src = json.load(open(sys.argv[1]))
want = {"htdemucs": ("htdemucs", 1.3333), "htdemucs-tuned": ("htdemucs", 1.0), "voc-ft": ("vocft", 2.0), "voc-ft-o1": ("vocft", 1.0), "kim-vocal-2": ("kimv2", 2.0),
        "duality-student": ("duality_student", 1.0), "melband-kim": ("melkim", 2.0), "htdemucs-lite-11k": ("htdemucs_sdBby5000", 1.0), "voc-ft-p55": ("vocft_p55", 1.0), "voc-ft-qat8": ("vocft_qat8mix", 1.0)}
os.makedirs("results/quality", exist_ok=True)
for n, (f, ov) in want.items():
    r = next(r for r in src if r.get("file") == f and abs((r.get("overlap") or 0) - ov) < 0.01 and r.get("index"))
    json.dump({"sets": {"musdb30": {"mean": r["musdb30"], "clips": 50}, "cc30": {"mean": r["cc30"], "clips": 50, **({"estimated": True} if r.get("cc30_raw") not in (None, r["cc30"]) else {})},
                        "private30": {"mean": r["splice30"], "clips": 50}}, "source": "imported from the maintainers' bench"}, open(f"results/quality/{n}.json", "w"), indent=1)
    print(n, r["musdb30"], r["cc30"], r["splice30"], r["index"])
