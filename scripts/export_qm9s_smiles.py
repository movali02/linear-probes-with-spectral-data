#!/usr/bin/env python3
"""DEPRECATED: keys on `.number`, which is the WRONG join for raman_boraden.csv
(spectra end up paired with other molecules' SMILES). Use
scripts/qm9s_alignment_check.py, which tests the candidate joins against the
spectra and writes the map only when one is confirmed.

Run once on Aura, where qm9s.pt lives. Writes number,smiles CSV.

  python scripts/export_qm9s_smiles.py --pt /path/to/qm9s.pt --out data/qm9s_number_smiles.csv

qm9s.pt is a list of torch_geometric Data objects: SMILES in `.smile`, 1-based
molecule id in `.number` (the key used by raman_boraden.csv)."""
import argparse
import csv

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", default="data/qm9s_number_smiles.csv")
    a = ap.parse_args()
    data = torch.load(a.pt, weights_only=False)  # trusted Figshare pickle
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "smiles"])
        for d in data:
            w.writerow([int(d.number), d.smile])
    print(f"wrote {len(data)} rows -> {a.out}")


if __name__ == "__main__":
    main()
