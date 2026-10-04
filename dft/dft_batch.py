#!/usr/bin/env python3
"""
dft_batch.py
============
Your batch.py, generalised to molecule SETS with charges and optional solvent.
Same method (B3LYP/6-311+G(d,p), Opt + NumFreq, %elprop Polar 1 for Raman),
same rule: no geometry reaches ORCA unless it passes every check first.

What changed from batch.py, and why
-----------------------------------
* Sets come from dft/molecules.csv:
    neutral_gas       20 amino acids, neutral, gas phase
    neutral_water     20 amino acids, neutral, CPCM(Water)
    zwitterion_water  20 amino acids, pH 7 species, CPCM(Water)
                      (Asp/Glu -1, Lys/Arg +1, His neutral N-tau-H)
    pI_water          Asp, Glu, Lys, Arg net-neutral species (standards in water)
    calibration_gas   9 QM9S molecules, to measure the QM9S <-> your-DFT offset
* Charge-aware: "* xyz <charge> 1", electron parity = sum(Z) - charge, and the
  formula check compares the charged formula (e.g. C4H6NO4-).
* Isomeric SMILES: batch.py used SMILES without stereo, so Thr and Ile could
  embed as allo-Thr / allo-Ile (different diastereomers, different spectra).
  The embedded geometry is now checked against the SMILES stereo.
* His uses the N-tau-H tautomer (the major one in water); batch.py's SMILES
  gave N-pi-H.
* Water sets pick the starting conformer with a distance-dependent dielectric
  in MMFF, so zwitterions do not start in a vacuum-style salt-bridge geometry
  that invites proton transfer during the DFT optimisation.
* The pre-DFT geometry is saved as <name>_init.xyz (ORCA's Opt overwrites
  <name>.xyz with the final geometry, which dft_check.py then inspects).
* Resumable: a job whose .out already ends normally is skipped.
* Smallest molecules first, so problems surface early.
* After each job, dft_check.py runs: normal termination, optimisation
  converged, no imaginary modes, Raman block present, and the final geometry
  still has the intended protonation/tautomer and diastereomer.

Run on aura inside tmux:
    tmux new -s dft
    conda activate <env-with-rdkit>
    python dft/dft_batch.py --set zwitterion_water --dry-run   # inputs only, check them
    python dft/dft_batch.py --set zwitterion_water
Detach with Ctrl+B then D.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dft_check  # noqa: E402

# ----------------------------------------------------------------------
# Defaults (override on the command line)
# ----------------------------------------------------------------------
ORCA = "/home-local/mv487/orca/orca"          # full path is required by ORCA
WORKROOT = "/local/scratch/mv487/dft"
NPROCS = 20
MAXCORE = 8000
METHOD = "! B3LYP 6-311+G(d,p) Opt NumFreq"   # identical to your existing runs

BOND_RANGES = {
    frozenset(("C", "C")): (1.20, 1.62),
    frozenset(("C", "N")): (1.20, 1.55),
    frozenset(("C", "O")): (1.15, 1.50),
    frozenset(("N", "O")): (1.15, 1.50),
    frozenset(("C", "S")): (1.70, 1.90),
    frozenset(("S", "H")): (1.25, 1.42),
    frozenset(("C", "H")): (0.95, 1.15),
    frozenset(("N", "H")): (0.95, 1.10),
    frozenset(("O", "H")): (0.90, 1.05),
}
MIN_NONBONDED = 0.90


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_set(path, set_name, only=None):
    rows = [r for r in csv.DictReader(open(path)) if r["set"] == set_name]
    if not rows:
        raise SystemExit(f"no molecules in set {set_name!r} in {path}")
    if only:
        keep = set(only.split(","))
        rows = [r for r in rows if r["name"] in keep]
    for r in rows:
        r["charge"] = int(r["charge"])
    return rows


def solvent_of(set_name):
    return "water" if set_name.endswith("_water") else None


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def n_conformers(mol):
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return int(min(60, max(20, 10 * rot)))


def generate_geometry(smiles, solvent):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES '{smiles}'")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.enforceChirality = True
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers(mol), params=params))
    if not cids:
        raise RuntimeError("conformer embedding produced no conformers")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if solvent:
            # distance-dependent dielectric (eps = 4r): damps the NH3+ ... -OOC
            # attraction that a vacuum force field over-rewards
            props.SetMMFFDielectricModel(2)  # 2 = distance-dependent (1 = constant)
            props.SetMMFFDielectricConstant(4.0)
        energies = []
        for cid in cids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            ff.Minimize(maxIts=2000)
            energies.append(ff.CalcEnergy())
        ff_name = "MMFF94" + ("(eps=4r)" if solvent else "")
    else:
        res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        energies = [e for _c, e in res]
        ff_name = "UFF"
    k = min(range(len(cids)), key=lambda i: energies[i])
    best = cids[k]
    conf = mol.GetConformer(best)
    atoms = [(a.GetSymbol(), *(lambda p: (p.x, p.y, p.z))(conf.GetAtomPosition(a.GetIdx())))
             for a in mol.GetAtoms()]
    return mol, best, atoms, ff_name, len(cids)


def _dist(conf, i, j):
    a, b = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def validate(mol, conf_id, smiles, expected_formula, charge, atoms):
    problems = []
    formula = rdMolDescriptors.CalcMolFormula(mol)
    if formula != expected_formula:
        problems.append(f"formula mismatch: RDKit={formula}, expected={expected_formula}")
    q = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if q != charge:
        problems.append(f"formal charges sum to {q}, table says {charge}")
    n_elec = sum(a.GetAtomicNum() for a in mol.GetAtoms()) - charge
    if n_elec % 2:
        problems.append(f"odd electron count ({n_elec}) incompatible with singlet")
    if mol.GetNumAtoms() != len(atoms):
        problems.append(f"atom-count mismatch: graph={mol.GetNumAtoms()}, coords={len(atoms)}")
    conf = mol.GetConformer(conf_id)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        si, sj = mol.GetAtomWithIdx(i).GetSymbol(), mol.GetAtomWithIdx(j).GetSymbol()
        rng = BOND_RANGES.get(frozenset((si, sj)))
        if rng and not (rng[0] <= _dist(conf, i, j) <= rng[1]):
            problems.append(f"bond {si}{i}-{sj}{j} = {_dist(conf, i, j):.3f} A outside {rng}")
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(i + 1, n):
            if _dist(conf, i, j) < MIN_NONBONDED:
                problems.append(f"atoms {i} and {j} too close ({_dist(conf, i, j):.3f} A)")
    # stereo: the embedded 3D geometry must carry the SMILES configuration
    xyz = "\n".join([str(len(atoms)), "init"] + [f"{s} {x} {y} {z}" for s, x, y, z in atoms])
    st = dft_check.compare_structure(xyz, smiles, charge)
    if not st["constitution_ok"]:
        problems.append(f"embedded constitution differs: {st['detail']}")
    if st["stereo"] not in ("same", "none"):
        problems.append(f"embedded stereo {st['stereo']}: {st['detail']}")
    if problems:
        raise ValueError("validation FAILED:\n    - " + "\n    - ".join(problems))
    return formula, n_elec, mol.GetRingInfo().NumRings()


# ----------------------------------------------------------------------
# ORCA
# ----------------------------------------------------------------------
def write_xyz(path, name, atoms):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{name}\n")
        for sym, x, y, z in atoms:
            f.write(f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def write_orca_input(path, atoms, charge, solvent, nprocs, maxcore):
    lines = [METHOD]
    if solvent == "water":
        lines.append("! CPCM(Water)")
    lines += [f"%pal nprocs {nprocs} end", f"%maxcore {maxcore}",
              "%elprop", "  Polar 1", "end", "", f"* xyz {charge} 1"]
    lines += [f"  {s:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for s, x, y, z in atoms]
    lines += ["*", ""]
    Path(path).write_text("\n".join(lines))


def finished(out):
    return out.exists() and "ORCA TERMINATED NORMALLY" in out.read_text(errors="ignore")


def run_orca(orca, name, workdir):
    inp, out = workdir / f"{name}.inp", workdir / f"{name}.out"
    with open(out, "w") as fout:
        subprocess.run([orca, str(inp)], stdout=fout, stderr=subprocess.STDOUT, cwd=workdir)
    return finished(out)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True,
                    choices=["neutral_gas", "neutral_water", "zwitterion_water", "pI_water", "calibration_gas"])
    ap.add_argument("--molecules", default=str(HERE / "molecules.csv"))
    ap.add_argument("--only", help="comma-separated names, e.g. glycine,alanine")
    ap.add_argument("--workroot", default=WORKROOT)
    ap.add_argument("--orca", default=ORCA)
    ap.add_argument("--nprocs", type=int, default=NPROCS)
    ap.add_argument("--maxcore", type=int, default=MAXCORE)
    ap.add_argument("--dry-run", action="store_true", help="validate and write inputs, do not run ORCA")
    a = ap.parse_args()

    rows = load_set(a.molecules, a.set, a.only)
    solvent = solvent_of(a.set)
    root = Path(a.workroot) / a.set
    root.mkdir(parents=True, exist_ok=True)

    # geometry + validation for everything first, so a bad entry fails in seconds
    jobs = []
    for r in rows:
        name = r["name"]
        try:
            mol, best, atoms, ff, nconf = generate_geometry(r["smiles"], solvent)
            f_got, n_elec, n_rings = validate(mol, best, r["smiles"], r["formula"], r["charge"], atoms)
            log(f"{name}: validated | formula={f_got} charge={r['charge']:+d} electrons={n_elec} "
                f"rings={n_rings} atoms={len(atoms)} ff={ff} conformers={nconf}")
            jobs.append((len(atoms), r, atoms))
        except Exception as e:  # noqa: BLE001
            log(f"{name}: SKIPPED -- {e}")
    jobs.sort(key=lambda j: j[0])

    summary = []
    for n_atoms, r, atoms in jobs:
        name = r["name"]
        wd = root / name
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / f"{name}.out"
        if finished(out):
            log(f"{name}: already finished, skipping ORCA")
        else:
            write_xyz(wd / f"{name}_init.xyz", name, atoms)
            write_orca_input(wd / f"{name}.inp", atoms, r["charge"], solvent, a.nprocs, a.maxcore)
            (wd / "smiles.txt").write_text(f"{r['smiles']}\n{r['charge']}\n")
            if a.dry_run:
                log(f"{name}: wrote {name}.inp ({n_atoms} atoms) [dry run]")
                summary.append({"name": name, "status": "inputs written"})
                continue
            log(f"{name}: launching ORCA on {a.nprocs} cores ({n_atoms} atoms) ...")
            t0 = time.time()
            ok = run_orca(a.orca, name, wd)
            log(f"{name}: {'TERMINATED NORMALLY' if ok else 'ORCA ERROR -- inspect .out'} "
                f"after {(time.time() - t0) / 60:.1f} min")
        chk = dft_check.check_job(wd, name, r["smiles"], r["charge"])
        log(f"{name}: check {chk['verdict']} | {chk['detail']}")
        summary.append({"name": name, "status": chk["verdict"], "detail": chk["detail"]})

    log("================= SUMMARY =================")
    for s in summary:
        log(f"  {s['name']:22s} {s['status']:10s} {s.get('detail', '')}")
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "status", "detail"])
        w.writeheader()
        w.writerows(summary)
    log(f"summary -> {root / 'summary.csv'}")
    log("Frequencies are left UNSCALED here; dft_to_sticks.py writes unscaled sticks and the "
        "string pipeline applies one scale factor (set it from the calibration set).")


if __name__ == "__main__":
    main()
