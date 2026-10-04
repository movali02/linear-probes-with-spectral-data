#!/usr/bin/env python3
"""
calibrate_vs_qm9s.py
====================
First rung of the ladder: the same molecules computed two ways. For the 9
calibration molecules (in QM9S AND run with your method in the gas phase), find

  1. the frequency scale factor s that best maps your DFT band positions onto
     the QM9S ones (also run against experiment later if you like), and
  2. which intensity convention reproduces QM9S's relative intensities:
     raw activity, or Raman intensity at 785 nm (int785).

Both answers go into configs/string_v1.yaml (dft.scale, and which stick
variant to use), so every simulated source is on one footing before the ladder
starts. Needs QM9S records built with the CORRECTED SMILES join
(scripts/qm9s_alignment_check.py).

  python dft/calibrate_vs_qm9s.py \
      --qm9s_records data/processed/qm9s_records.jsonl \
      --sticks data/dft_sticks/calibration_gas --molecules dft/molecules.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr

WINDOW = (500.0, 1750.0)


def canon(smi):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)


def qm9s_peaks(records_path, wanted):
    """{canonical smiles: [(pos, int)]} from qm9s_records.jsonl (text = 'pos int fwhm | ...')."""
    out = {}
    for line in open(records_path):
        r = json.loads(line)
        c = canon(r["smiles"])
        if c in wanted and r.get("text"):
            out[c] = [(float(p.split()[0]), float(p.split()[1])) for p in r["text"].split(" | ")]
    return out


def dft_peaks(freq, height, scale, fwhm=10.0, min_height=0.05):
    grid = np.arange(WINDOW[0], WINDOW[1] + 1.0)
    hw = fwhm / 2.0
    y = (height[None, :] * hw ** 2 / ((grid[:, None] - freq[None, :] * scale) ** 2 + hw ** 2)).sum(1)
    if y.max() <= 0:
        return []
    y = y / y.max()
    idx, _ = find_peaks(y, height=min_height, prominence=0.03)
    return [(float(grid[i]), float(y[i])) for i in idx]


def match(a, b, tol=20.0):
    """Greedy one-to-one matching of peak lists by position; strongest first."""
    pairs, used = [], set()
    for pa in sorted(a, key=lambda p: -p[1]):
        cands = [(abs(pa[0] - pb[0]), j) for j, pb in enumerate(b) if j not in used and abs(pa[0] - pb[0]) <= tol]
        if cands:
            _, j = min(cands)
            used.add(j)
            pairs.append((pa, b[j]))
    return pairs


def load_sticks(path):
    arr = np.loadtxt(path, comments="#")
    return arr[:, 0], arr[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qm9s_records", required=True)
    ap.add_argument("--sticks", required=True, help="dft_to_sticks.py output for calibration_gas")
    ap.add_argument("--molecules", default=str(Path(__file__).resolve().parent / "molecules.csv"))
    ap.add_argument("--scales", default="0.940:1.010:0.001")
    a = ap.parse_args()

    cal = {r["name"]: r["smiles"] for r in csv.DictReader(open(a.molecules)) if r["set"] == "calibration_gas"}
    wanted = {canon(s): n for n, s in cal.items()}
    ref = qm9s_peaks(a.qm9s_records, wanted)
    missing = [n for c, n in wanted.items() if c not in ref]
    if missing:
        print(f"not found in QM9S records: {missing}")
    lo, hi, st = (float(x) for x in a.scales.split(":"))
    scales = np.arange(lo, hi + st / 2, st)
    sticks = {}
    for c, n in wanted.items():
        p = Path(a.sticks) / "activity" / f"{n}.txt"
        if c in ref and p.exists():
            sticks[n] = (c, *load_sticks(p), load_sticks(Path(a.sticks) / "int785" / f"{n}.txt")[1])
    if not sticks:
        raise SystemExit("no molecule has both QM9S peaks and DFT sticks")

    # 1. scale factor: minimise median |position error| over matched strong bands
    best = None
    for s in scales:
        errs = []
        for n, (c, f, act, i785) in sticks.items():
            for pd_, pq in match(dft_peaks(f, act, s), ref[c]):
                if pd_[1] >= 0.2 and pq[1] >= 0.2:
                    errs.append(abs(pd_[0] - pq[0]))
        if errs:
            score = (np.median(errs), -len(errs))
            if best is None or score < best[0]:
                best = (score, s, len(errs))
    (med, _), s_best, n_pairs = best
    print(f"frequency scale (your DFT -> QM9S): {s_best:.3f}  "
          f"median |error| {med:.1f} cm-1 over {n_pairs} matched strong bands")

    # 2. intensity convention at that scale: rank correlation of matched intensities
    for label, col in (("activity", 2), ("int785", 3)):
        xs, ys = [], []
        for n, t in sticks.items():
            c, f = t[0], t[1]
            for pd_, pq in match(dft_peaks(f, t[col], s_best), ref[c]):
                xs.append(pd_[1])
                ys.append(pq[1])
        rho = spearmanr(xs, ys).correlation if len(xs) > 3 else float("nan")
        print(f"intensity convention {label:9s}: Spearman rho with QM9S = {rho:.3f} over {len(xs)} bands")
    print("\nPer molecule at the chosen scale (DFT activity peaks vs QM9S peaks):")
    for n, (c, f, act, i785) in sticks.items():
        d = [round(p[0]) for p in dft_peaks(f, act, s_best) if p[1] >= 0.2]
        q = [round(p[0]) for p in ref[c] if p[1] >= 0.2]
        print(f"  {n:18s} DFT {d}\n  {'':18s} QM9S {q}")


if __name__ == "__main__":
    main()
