#!/usr/bin/env python3
"""T3: where does the string lose information? (edge-of-window test)

For every QM9S label it prints the best AUROC on the string re-drawn as a
spectrum (F2) and on the processed spectrum (F3), their gap, and the same gap
 - with both cropped below 1700 cm-1 (native run), and
 - with the window extended to 1800 cm-1 (the `wide` variant).
If the carbonyl gap (ester, lactam, ketone, amide) shrinks in both, the loss is
the C=O stretch at the 1750 cm-1 edge, not the string format.

  python scripts/compare_string_loss.py --native reports/baselines/native/T3.csv \
      --wide reports/baselines/wide/T3.csv"""
import argparse
import csv


def best(path, scheme):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["metric"] != "auroc" or r["scheme"] != scheme or r["model"] == "majority" or r["value"] == "":
            continue
        k = (r["label"], r["feature"])
        out[k] = max(out.get(k, 0.0), float(r["value"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", default="reports/baselines/native/T3.csv")
    ap.add_argument("--wide", default=None)
    ap.add_argument("--scheme", default="random")
    a = ap.parse_args()
    n = best(a.native, a.scheme)
    w = best(a.wide, a.scheme) if a.wide else {}
    labels = sorted({k[0] for k in n}, key=lambda l: -(n.get((l, "F3"), 0) - n.get((l, "F2"), 0)))
    print(f"{'label':22s} {'F2':>6s} {'F3':>6s} {'gap':>6s} | {'gap <1700':>9s} | {'wide F2':>7s} {'wide F3':>7s} {'wide gap':>8s}")
    for l in labels:
        g = n.get((l, "F3"), float("nan")) - n.get((l, "F2"), float("nan"))
        gc = n.get((l, "F3.lt1700"), float("nan")) - n.get((l, "F2.lt1700"), float("nan"))
        line = (f"{l.replace('fg_', ''):22s} {n.get((l, 'F2'), float('nan')):6.3f} {n.get((l, 'F3'), float('nan')):6.3f} "
                f"{g:+6.3f} | {gc:+9.3f} |")
        if w:
            gw = w.get((l, "F3"), float("nan")) - w.get((l, "F2"), float("nan"))
            line += f" {w.get((l, 'F2'), float('nan')):7.3f} {w.get((l, 'F3'), float('nan')):7.3f} {gw:+8.3f}"
        print(line)


if __name__ == "__main__":
    main()
