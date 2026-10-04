#!/usr/bin/env python3
"""Find the correct join between raman_boraden.csv rows and qm9s.pt molecules,
and write the SMILES map only if the chemistry confirms it.

Why: make_views.py joined CSV column 0 to `.number`. In the resulting file the
first spectra belong to the NEXT molecule (the "CH4" row has NH3's bands, the
"NH3" row has water's), and across all 126k rows a C#C / C#N bond does not
predict a 2050-2350 cm-1 band at all. So that join is wrong.

How: two bands that are unmistakable in a full-range (400-4000) Raman spectrum
are used as physics markers:
  * triple bond  C#C / C#N   ->  band in 2050-2350 cm-1
  * free O-H     alcohol/phenol/acid O-H  ->  band in 3610-3720 cm-1
For each candidate join, the marker strength is scored against the RDKit
substructure of the SMILES it assigns (AUROC). The right join gives AUROC close
to 1 for both markers; a wrong one sits near 0.5.

Run on Aura (needs torch + torch_geometric only to read qm9s.pt):
  python scripts/qm9s_alignment_check.py \
      --csv data/qm9s/raman_boraden.csv --pt data/qm9s/qm9s.pt \
      --out data/qm9s_number_smiles.csv

Writes --out (columns: number, smiles, list_pos, qm9_number) where `number` is
the CSV's own first-column id, which is what build_strings.py keys on. Exit
code 1 and nothing written if no join passes.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

MARKERS = {
    "triple_bond": ("[#6,#7;X2,X1]#[#6,#7]", (2050.0, 2350.0)),
    "free_OH": ("[OX2H]", (3610.0, 3720.0)),
}
PASS_AUROC = 0.90


# ----------------------------------------------------------------- inputs
def load_molecule_list(pt: str | None, list_csv: str | None) -> list[dict]:
    """[{list_pos, number, smiles}] in qm9s.pt list order."""
    if list_csv:  # test/fallback path: columns list_pos, number, smiles
        with open(list_csv, newline="") as f:
            return [{"list_pos": int(r["list_pos"]), "number": int(r["number"]), "smiles": r["smiles"]}
                    for r in csv.DictReader(f)]
    import torch
    data = torch.load(pt, weights_only=False)  # trusted Figshare pickle
    first = data[0]
    try:
        keys = list(first.keys()) if callable(getattr(first, "keys", None)) else list(first.keys)
    except Exception:  # noqa: BLE001
        keys = [k for k in dir(first) if not k.startswith("_")][:40]
    print(f"[pt] {len(data)} entries; attributes of entry 0: {keys}")
    out = []
    for i, d in enumerate(data):
        out.append({"list_pos": i, "number": int(d.number), "smiles": str(d.smile)})
    return out


def scan_csv(path: str, chunksize: int = 4000):
    """Return (ids, row_pos, marker_scores{name: array}, preview) streaming the CSV."""
    import pandas as pd
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
    if header[0].startswith("PLACEHOLDER"):
        raise SystemExit(f"{path} is still the placeholder file")
    axis = np.asarray([float(h) for h in header[1:]])
    masks = {k: (axis >= lo) & (axis <= hi) for k, (_, (lo, hi)) in MARKERS.items()}
    if not all(m.any() for m in masks.values()):
        raise SystemExit(f"CSV axis {axis.min():.0f}-{axis.max():.0f} does not reach the marker bands")
    ids, scores, preview = [], {k: [] for k in MARKERS}, []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        v = chunk.to_numpy(dtype=float)
        cid, y = v[:, 0].astype(int), v[:, 1:]
        mx = np.maximum(y.max(axis=1), 1e-30)
        for k, m in masks.items():
            scores[k].append(y[:, m].max(axis=1) / mx)
        ids.append(cid)
        if len(preview) < 10:
            for row_id, row in zip(cid[: 10 - len(preview)], y[: 10 - len(preview)]):
                top = np.argsort(row)[::-1]
                picked = []
                for t in top:  # 4 strongest well-separated points
                    if all(abs(axis[t] - axis[p]) > 40 for p in picked):
                        picked.append(t)
                    if len(picked) == 4:
                        break
                preview.append((int(row_id), sorted(round(float(axis[p])) for p in picked)))
    ids = np.concatenate(ids)
    return ids, np.arange(len(ids)), {k: np.concatenate(s) for k, s in scores.items()}, preview


# ----------------------------------------------------------------- scoring
def auroc(score, label):
    from sklearn.metrics import roc_auc_score
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    return float(roc_auc_score(label, score))


def candidate_joins(mols):
    by_num = {m["number"]: m for m in mols}
    by_pos = {m["list_pos"]: m for m in mols}
    return {
        "number == csv_id  (make_views.py)": lambda cid, r: by_num.get(cid),
        "number == csv_id + 1": lambda cid, r: by_num.get(cid + 1),
        "number == csv_id - 1": lambda cid, r: by_num.get(cid - 1),
        "list_pos == csv_id": lambda cid, r: by_pos.get(cid),
        "list_pos == csv_id - 1": lambda cid, r: by_pos.get(cid - 1),
        "list_pos == csv row index": lambda cid, r: by_pos.get(r),
    }


def evaluate(mols, ids, rows, scores, max_rows=None):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    pats = {k: Chem.MolFromSmarts(s) for k, (s, _) in MARKERS.items()}
    cache = {}

    def feats(smi):
        if smi not in cache:
            m = Chem.MolFromSmiles(smi)
            cache[smi] = None if m is None else {k: m.HasSubstructMatch(p) for k, p in pats.items()}
        return cache[smi]

    sel = np.arange(len(ids))
    if max_rows and len(sel) > max_rows:
        sel = np.sort(np.random.default_rng(0).choice(len(ids), max_rows, replace=False))
    results = []
    for name, fn in candidate_joins(mols).items():
        lab = {k: [] for k in MARKERS}
        sc = {k: [] for k in MARKERS}
        mapped = 0
        for i in sel:
            m = fn(int(ids[i]), int(rows[i]))
            if m is None:
                continue
            f = feats(m["smiles"])
            if f is None:
                continue
            mapped += 1
            for k in MARKERS:
                lab[k].append(f[k])
                sc[k].append(scores[k][i])
        res = {"join": name, "coverage": mapped / len(sel)}
        for k in MARKERS:
            res[f"auroc_{k}"] = auroc(np.array(sc[k]), np.array(lab[k])) if lab[k] else float("nan")
        res["passes"] = (res["coverage"] >= 0.99
                         and all(res[f"auroc_{k}"] >= PASS_AUROC for k in MARKERS))
        results.append(res)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--pt")
    ap.add_argument("--list_csv", help="list_pos,number,smiles instead of --pt (testing)")
    ap.add_argument("--out", default="data/qm9s_number_smiles.csv")
    ap.add_argument("--max_rows", type=int, default=40000, help="rows scored per join (speed)")
    a = ap.parse_args()
    if not (a.pt or a.list_csv):
        ap.error("give --pt or --list_csv")

    mols = load_molecule_list(a.pt, a.list_csv)
    nums = sorted(m["number"] for m in mols)
    print(f"[pt] numbers {nums[0]}..{nums[-1]}, {nums[-1] - nums[0] + 1 - len(nums)} gaps")
    ids, rows, scores, preview = scan_csv(a.csv)
    print(f"[csv] {len(ids)} rows, first-column ids {ids.min()}..{ids.max()}, "
          f"{'unique' if len(set(ids.tolist())) == len(ids) else 'NOT unique'}")

    results = evaluate(mols, ids, rows, scores, a.max_rows)
    print(f"\n{'join':38s} {'coverage':>8s} {'AUROC C#C/C#N':>14s} {'AUROC O-H':>10s}")
    for r in results:
        print(f"{r['join']:38s} {r['coverage']:8.3f} {r['auroc_triple_bond']:14.3f} "
              f"{r['auroc_free_OH']:10.3f}  {'PASS' if r['passes'] else ''}")

    passing = [r for r in results if r["passes"]]
    # Joins that assign the same molecule to every row are one join under two names
    # (e.g. when the CSV ids are exactly 0..N-1, "csv_id" and "row index" coincide).
    joins = candidate_joins(mols)

    def signature(name):
        fn = joins[name]
        return tuple((m["list_pos"] if m else -1) for m in (fn(int(c), i) for i, c in enumerate(ids)))

    classes = {}
    for r in passing:
        classes.setdefault(signature(r["join"]), []).append(r["join"])
    if len(classes) != 1:
        print(f"\n{len(classes)} different joins pass (need exactly 1). Nothing written. Send this output back.")
        sys.exit(1)
    names = next(iter(classes.values()))
    if len(names) > 1:
        print(f"\nthese passing joins are identical on every row: {names}")
    best = names[0]
    fn = joins[best]
    print(f"\nchosen join: {best}")
    print("first rows under this join (4 strongest bands, full range):")
    for cid, bands in preview:
        m = fn(cid, int(np.where(ids == cid)[0][0]))
        print(f"   csv_id {cid:>6d}  {m['smiles'] if m else '-':20s} {bands}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "smiles", "list_pos", "qm9_number"])
        for i, cid in enumerate(ids):
            m = fn(int(cid), i)
            if m is not None:
                w.writerow([int(cid), m["smiles"], m["list_pos"], m["number"]])
                n += 1
    print(f"wrote {n} rows -> {a.out}  (column `number` = CSV id, as build_strings.py expects)")


if __name__ == "__main__":
    main()
