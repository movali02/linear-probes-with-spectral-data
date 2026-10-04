#!/usr/bin/env python3
"""Are the QM9S frequencies scaled? (roadmap 1.4)

Prints the peaks of a few QM9S molecules next to their experimental Raman
bands. Benzene's ring-breathing band is at 992 cm-1 in experiment; unscaled
harmonic B3LYP puts it near 1010-1015. A carbonyl stretch sitting just outside
the 1750 cm-1 window (see T3) depends on this.

  python scripts/qm9s_benzene_check.py --records data/processed/qm9s_records.jsonl"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit.io import load_config  # noqa: E402
from probe2circuit.labels import canonical  # noqa: E402
from probe2circuit.strings import parse_string  # noqa: E402

REF = {  # experimental Raman bands inside 500-1750 cm-1 (liquid / gas, +/- a few cm-1)
    "c1ccccc1": ("benzene", [606, 992, 1178, 1585, 1606]),
    "Cc1ccccc1": ("toluene", [521, 786, 1004, 1030, 1210, 1605]),
    "c1ccncc1": ("pyridine", [605, 652, 991, 1030, 1218, 1483, 1581]),
    "c1cc[nH]c1": ("pyrrole", [1144, 1384, 1470]),
    "CC(C)=O": ("acetone", [530, 787, 1066, 1222, 1430, 1710]),
    "CC(=O)OC": ("methyl acetate", [640, 844, 1048, 1438, 1740]),
    "CC=O": ("acetaldehyde", [509, 1114, 1352, 1395, 1436, 1743]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="data/processed/qm9s_records.jsonl")
    ap.add_argument("--config", default="configs/string_v1.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    want = {canonical(k): v for k, v in REF.items()}
    found, ratios = set(), []
    for line in open(a.records):
        r = json.loads(line)
        k = canonical(r["smiles"]) if r.get("smiles") else None
        if k in want and k not in found:
            found.add(k)
            name, exp = want[k]
            pk = [(p["position"], p["intensity"]) for p in parse_string(r["text"], cfg)] if r.get("text") else []
            print(f"\n{name} ({r['id']})")
            print("  QM9S peaks : " + "  ".join(f"{p:.0f}({i:.2f})" for p, i in pk))
            print("  experiment : " + "  ".join(str(e) for e in exp))
            for e in exp:
                near = [p for p, _ in pk if abs(p - e) <= 0.04 * e]
                if near:
                    q = min(near, key=lambda p: abs(p - e))
                    ratios.append(e / q)
                    print(f"    {e:5d} <- {q:6.0f}   experiment / QM9S = {e / q:.4f}")
        if len(found) == len(want):
            break
    if ratios:
        import statistics
        m = statistics.median(ratios)
        print(f"\nmedian experiment / QM9S = {m:.4f} over {len(ratios)} bands "
              f"({'already scaled (~1.00)' if m > 0.985 else 'UNSCALED: apply ~%.3f to QM9S' % m})")
    missing = [want[k][0] for k in want if k not in found]
    if missing:
        print(f"not found in the records: {missing}")


if __name__ == "__main__":
    main()
