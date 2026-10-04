#!/usr/bin/env python3
"""
dft_to_sticks.py
================
Turn checked ORCA Raman outputs into stick files for the string pipeline, in
three intensity conventions. Each one is a rung of the DFT -> SERS ladder, and
none needs a new calculation:

  activity     S = 45 a^2 + 7 gamma^2        (what ORCA prints; orientation-averaged,
                                               90-degree / unpolarised backscatter)
  int785       S * f(nu)                      (Raman INTENSITY at 785 nm: frequency
                                               factor + Bose population)
  gapiso785    (45 a^2 + 4 gamma^2) * f(nu)   (plasmonic gap: incident AND scattered
                                               field along the gap axis z, molecule
                                               still randomly oriented -> <alpha'_zz^2>)

  f(nu) = (nu0 - nu)^4 / nu / (1 - exp(-c2 nu / T)),  nu0 = 1e7 / laser_nm

The frequency factor matters in the 500-1750 window: at 785 nm it raises a
500 cm-1 band about 5.6x relative to a 1700 cm-1 band of equal activity. QM9S and your DFT must use
the same convention, and the calibration set is how you find out which one
QM9S used.

a^2 and gamma^2 come from ORCA's activity S and depolarisation rho (your
rank_anisotropy.py relations):  gamma^2 = S rho / (3 (1 + rho)),  a^2 = (S - 7 gamma^2) / 45.

Frequencies are written UNSCALED. The string pipeline applies one factor
(config: dft.scale), which you set from the calibration set.

  python dft/dft_to_sticks.py --root /local/scratch/mv487/dft/zwitterion_water \
      --out_dir data/dft_sticks/zwitterion_water
  python dft/dft_to_sticks.py --root /local/scratch/mv487/dft/AAs --set neutral_gas \
      --out_dir data/dft_sticks/neutral_gas
Then, per variant:
  python scripts/build_strings.py --dft_dir data/dft_sticks/zwitterion_water/int785 \
      --dft_smiles data/dft_sticks/zwitterion_water/smiles_map.json --out_dir data/processed/dft_zw_int785
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import dft_check

HERE = Path(__file__).resolve().parent
C2 = 1.4387769  # second radiation constant, cm K


def invariants(S, rho):
    S, rho = np.asarray(S, float), np.asarray(rho, float)
    g2 = S * rho / (3.0 * (1.0 + rho))
    a2 = (S - 7.0 * g2) / 45.0
    return np.clip(a2, 0, None), np.clip(g2, 0, None)


def freq_factor(nu, laser_nm=785.0, T=298.15):
    nu = np.asarray(nu, float)
    nu0 = 1e7 / laser_nm
    return (nu0 - nu) ** 4 / nu / (1.0 - np.exp(-C2 * nu / T))


def sticks_for(raman_rows, laser_nm, T):
    arr = np.asarray(raman_rows, float)
    nu, S, rho = arr[:, 0], arr[:, 1], arr[:, 2]
    keep = nu > 1.0
    nu, S, rho = nu[keep], S[keep], rho[keep]
    a2, g2 = invariants(S, rho)
    f = freq_factor(nu, laser_nm, T)
    return {
        "freq": nu, "S": S, "rho": rho, "a2": a2, "g2": g2,
        "activity": S,
        "int785": S * f,
        "gapiso785": (45.0 * a2 + 4.0 * g2) * f,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--set", help="set name in molecules.csv, for runs without smiles.txt")
    ap.add_argument("--molecules", default=str(HERE / "molecules.csv"))
    ap.add_argument("--analytes", default=str(HERE.parent / "data" / "analytes.csv"))
    ap.add_argument("--laser_nm", type=float, default=785.0)
    ap.add_argument("--T", type=float, default=298.15)
    ap.add_argument("--allow_fail", action="store_true", help="also convert jobs that fail dft_check")
    ap.add_argument("--soft_imag", type=float, default=0.0,
                    help="also convert a job whose ONLY problem is one imaginary mode weaker than this (cm-1), "
                         "e.g. 50. Such a mode is a nearly free torsion; it is dropped from the sticks and "
                         "the bands in the 500-1800 window are not affected. Flagged in soft_imag.csv.")
    a = ap.parse_args()

    mols = list(csv.DictReader(open(a.molecules)))
    by_set = {r["name"]: r for r in mols if a.set and r["set"] == a.set}
    by_name_any = {}
    for r in mols:
        by_name_any.setdefault(r["name"], r)
    parent = {r["name"]: r["smiles"] for r in csv.DictReader(open(a.analytes))}

    out = Path(a.out_dir)
    variants = ["activity", "int785", "gapiso785"]
    for v in variants + ["modes"]:
        (out / v).mkdir(parents=True, exist_ok=True)
    smiles_map, done, flagged = {}, [], []
    for wd in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
        name = wd.name
        s = wd / "smiles.txt"
        row = by_set.get(name) or by_name_any.get(name)
        if s.exists():
            smi, ch = s.read_text().split("\n")[:2]
            ch = int(ch)
        elif row:
            smi, ch = row["smiles"], int(row["charge"])
        else:
            print(f"skip {name}: unknown molecule")
            continue
        chk = dft_check.check_job(wd, name, smi, ch)
        soft = (chk["verdict"] != "PASS" and chk.get("detail") == "1 imaginary mode(s)"
                and len(chk.get("imag_freqs") or []) == 1 and abs(chk["imag_freqs"][0]) < a.soft_imag)
        if soft:
            print(f"accept {name}: one soft imaginary mode ({chk['imag_freqs'][0]:.1f} cm-1), flagged")
            flagged.append((name, chk["imag_freqs"][0]))
        elif chk["verdict"] != "PASS" and not a.allow_fail:
            print(f"skip {name}: {chk['detail']}")
            continue
        rows = dft_check.raman_block((wd / f"{name}.out").read_text(errors="ignore"))
        st = sticks_for(rows, a.laser_nm, a.T)
        for v in variants:
            np.savetxt(out / v / f"{name}.txt", np.c_[st["freq"], st[v]], fmt="%.4f %.6e",
                       header=f"freq_cm-1_unscaled {v}")
        np.savetxt(out / "modes" / f"{name}.csv",
                   np.c_[st["freq"], st["S"], st["rho"], st["a2"], st["g2"], st["int785"], st["gapiso785"]],
                   delimiter=",", fmt="%.6g", header="freq,S,rho,a2,gamma2,int785,gapiso785", comments="")
        # labels always from the NEUTRAL parent molecule, as for the SERS spectra
        analyte = (row or {}).get("analyte") or ""
        smiles_map[name] = parent.get(analyte, smi)
        done.append(name)
    (out / "smiles_map.json").write_text(json.dumps(smiles_map, indent=1))
    with open(out / "soft_imag.csv", "w") as f:
        f.write("name,imaginary_cm-1\n" + "".join(f"{n},{v:.1f}\n" for n, v in flagged))
    print(f"{len(done)} molecules -> {out}/{{{','.join(variants)}}}/  (+ modes/, smiles_map.json)")


if __name__ == "__main__":
    main()
