#!/usr/bin/env python3
"""
dft_check.py
============
Check finished ORCA Opt+NumFreq+Raman jobs before their spectra are used.

Per job:
  * ORCA terminated normally
  * geometry optimisation converged
  * no imaginary frequencies
  * RAMAN SPECTRUM block present (number of modes)
  * solvent actually used (read from the input echo), and charge
  * the FINAL geometry still has the intended constitution, including where
    every H sits. This catches a zwitterion that transferred its proton back
    (NH3+ ... -OOC  ->  NH2 ... HOOC) and a His that switched tautomer.
  * the final geometry has the intended diastereomer (catches allo-Thr/Ile).
    An overall mirror image is accepted: enantiomers have identical Raman.

New runs (dft_batch.py) leave smiles.txt in each job folder. For your existing
runs, point at the folder and name the set, and the SMILES come from
molecules.csv by folder name:

  python dft/dft_check.py --root /local/scratch/mv487/dft/zwitterion_water
  python dft/dft_check.py --root /local/scratch/mv487/dft/AAs --set neutral_gas
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------- structure
def _skeleton(mol):
    """Charge-free, bond-order-free graph with explicit H atoms."""
    rw = Chem.RWMol()
    for a in mol.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetNoImplicit(True)
        rw.AddAtom(na)
    for b in mol.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), Chem.BondType.SINGLE)
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return m


def _h_on(mol, symbol):
    return sorted(sum(1 for n in a.GetNeighbors() if n.GetAtomicNum() == 1)
                  for a in mol.GetAtoms() if a.GetSymbol() == symbol)


def compare_structure(xyz_block: str, smiles: str, charge: int = 0) -> dict:
    """Compare a 3D geometry (xyz block) with the intended SMILES."""
    want = Chem.AddHs(Chem.MolFromSmiles(smiles))
    geo = Chem.MolFromXYZBlock(xyz_block)
    if geo is None:
        return {"constitution_ok": False, "stereo": "unknown", "detail": "unreadable geometry"}
    rdDetermineBonds.DetermineConnectivity(geo, charge=charge)
    sw, sg = _skeleton(want), _skeleton(geo)
    same = Chem.MolToSmiles(sw) == Chem.MolToSmiles(sg)
    if not same:
        parts = []
        for el in ("N", "O", "S"):
            a, b = _h_on(want, el), _h_on(geo, el)
            if a != b:
                parts.append(f"H on {el}: intended {a}, found {b}")
        return {"constitution_ok": False, "stereo": "unknown",
                "detail": "; ".join(parts) or "an H moved between atoms of the same element (tautomer) "
                          "or heavy-atom connectivity changed"}
    # map intended atoms onto geometry atoms, then read stereo from 3D
    match = sg.GetSubstructMatch(sw)
    centres = Chem.FindMolChiralCenters(want, includeUnassigned=False, useLegacyImplementation=False)
    if not centres:
        return {"constitution_ok": True, "stereo": "none", "detail": "constitution and H positions as intended"}
    probe = Chem.Mol(want)
    conf = Chem.Conformer(probe.GetNumAtoms())
    gconf = geo.GetConformer()
    for i_want, i_geo in enumerate(match):
        conf.SetAtomPosition(i_want, gconf.GetAtomPosition(i_geo))
    probe.RemoveAllConformers()
    probe.AddConformer(conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(probe)
    got = dict(Chem.FindMolChiralCenters(probe, includeUnassigned=True, useLegacyImplementation=False))
    want_d = dict(centres)
    flips = [got.get(i) != c for i, c in want_d.items()]
    if not any(flips):
        st = "same"
    elif all(flips) and all(got.get(i) in ("R", "S") for i in want_d):
        st = "same"  # overall mirror image: identical Raman spectrum
    else:
        st = "diastereomer"
    return {"constitution_ok": True, "stereo": st,
            "detail": f"stereo intended {sorted(want_d.values())} found {sorted(v for v in got.values())}"}


# ----------------------------------------------------------------- ORCA output
def _last_block_after(lines, header, start=0):
    idx = [i for i in range(start, len(lines)) if lines[i].strip() == header]
    return idx[-1] if idx else None


def final_geometry(out_text: str):
    """Coordinates printed after the optimisation converged."""
    lines = out_text.splitlines()
    anchor = None
    for key in ("FINAL ENERGY EVALUATION AT THE STATIONARY POINT", "HURRAY"):
        hits = [i for i, l in enumerate(lines) if key in l]
        if hits:
            anchor = hits[-1]
            break
    if anchor is None:
        return None
    for i in range(anchor, len(lines)):
        if lines[i].strip() == "CARTESIAN COORDINATES (ANGSTROEM)":
            atoms = []
            for l in lines[i + 2:]:
                p = l.split()
                if len(p) != 4:
                    break
                atoms.append(p)
            return "\n".join([str(len(atoms)), "final"] + [" ".join(a) for a in atoms])
    return None


def frequencies(out_text: str):
    lines = out_text.splitlines()
    i = _last_block_after(lines, "VIBRATIONAL FREQUENCIES")
    if i is None:
        return None
    freqs = []
    for l in lines[i + 1:]:
        if l.strip() == "NORMAL MODES":
            break
        m = re.match(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", l)
        if m:
            freqs.append(float(m.group(1)))
    return freqs


def raman_block(out_text: str):
    """(freq, activity, depol) rows from the last RAMAN SPECTRUM block."""
    lines = out_text.splitlines()
    i = _last_block_after(lines, "RAMAN SPECTRUM")
    if i is None:
        return []
    rows = []
    for l in lines[i + 1:]:
        p = l.split()
        if p and p[0].rstrip(":").isdigit() and len(p) >= 4:
            try:
                rows.append((float(p[1]), float(p[2]), float(p[3])))
            except ValueError:
                pass
        elif rows and not any(c.isdigit() for c in l):
            break
    return rows


def input_settings(out_text: str):
    solvent = "water" if re.search(r"CPCM\s*\(\s*water\s*\)", out_text, re.I) else "gas"
    m = re.search(r"\*\s*xyz\s+(-?\d+)\s+(\d+)", out_text, re.I)
    charge = int(m.group(1)) if m else None
    return solvent, charge


def check_job(wd: Path, name: str, smiles: str, charge: int) -> dict:
    out = Path(wd) / f"{name}.out"
    res = {"name": name, "verdict": "FAIL", "detail": ""}
    if not out.exists():
        res["detail"] = "no .out file"
        return res
    txt = out.read_text(errors="ignore")
    solvent, inp_charge = input_settings(txt)
    freqs = frequencies(txt) or []
    n_imag = sum(f < 0 for f in freqs)
    raman = raman_block(txt)
    geo = final_geometry(txt)
    st = compare_structure(geo, smiles, charge) if geo else {"constitution_ok": False, "stereo": "unknown",
                                                             "detail": "no final geometry in .out"}
    res.update({
        "terminated": "ORCA TERMINATED NORMALLY" in txt,
        "opt_converged": ("HURRAY" in txt) or ("OPTIMIZATION HAS CONVERGED" in txt),
        "n_imaginary": n_imag,
        "imag_freqs": [f for f in freqs if f < 0],
        "n_raman_modes": len(raman),
        "solvent": solvent,
        "charge_in_input": inp_charge,
        "constitution_ok": st["constitution_ok"],
        "stereo": st["stereo"],
    })
    problems = []
    if not res["terminated"]:
        problems.append("did not terminate normally")
    if not res["opt_converged"]:
        problems.append("optimisation not converged")
    if n_imag:
        problems.append(f"{n_imag} imaginary mode(s)")
    if not raman:
        problems.append("no RAMAN SPECTRUM block")
    if inp_charge is not None and inp_charge != charge:
        problems.append(f"charge in input {inp_charge} != intended {charge}")
    if not st["constitution_ok"]:
        problems.append(f"structure changed: {st['detail']}")
    if st["stereo"] == "diastereomer":
        problems.append(f"wrong diastereomer: {st['detail']}")
    res["verdict"] = "PASS" if not problems else "FAIL"
    res["detail"] = "; ".join(problems) or f"{len(raman)} Raman modes, {solvent}, charge {charge:+d}"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with one sub-folder per molecule")
    ap.add_argument("--set", help="molecule set in molecules.csv (for runs without smiles.txt)")
    ap.add_argument("--molecules", default=str(HERE / "molecules.csv"))
    a = ap.parse_args()

    table = {}
    if a.set:
        table = {r["name"]: r for r in csv.DictReader(open(a.molecules)) if r["set"] == a.set}
    results = []
    for wd in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
        name = wd.name
        s = wd / "smiles.txt"
        if s.exists():
            smi, ch = s.read_text().split("\n")[:2]
            ch = int(ch)
        elif name in table:
            smi, ch = table[name]["smiles"], int(table[name]["charge"])
        else:
            print(f"skip {name}: no smiles.txt and not in set {a.set}")
            continue
        r = check_job(wd, name, smi, ch)
        results.append(r)
        print(f"{r['verdict']:4s} {name:22s} {r['detail']}")
    if results:
        keys = ["name", "verdict", "terminated", "opt_converged", "n_imaginary", "n_raman_modes",
                "solvent", "charge_in_input", "constitution_ok", "stereo", "detail"]
        with open(Path(a.root) / "check.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        n_ok = sum(r["verdict"] == "PASS" for r in results)
        print(f"\n{n_ok}/{len(results)} PASS -> {Path(a.root) / 'check.csv'}")


if __name__ == "__main__":
    main()
