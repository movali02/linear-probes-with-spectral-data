#!/usr/bin/env python3
"""
dft_fix_imag.py
===============
Repair ORCA jobs that finished with imaginary frequencies (a saddle point, not
a minimum), then re-check them with dft_check.py.

What an imaginary mode usually is here
--------------------------------------
With CPCM + Opt at default convergence, the optimiser often stops on a very
flat stretch of the surface: a NH3+ or CH3 rotor, a COO- or side-chain
torsion. NumFreq then finds one small imaginary mode (typically 10i-80i
cm-1). The standard fix: push the geometry downhill along that mode and
reoptimise with tighter convergence.

  --report   (default) prints each imaginary mode: its frequency and which
             atoms move (e.g. "H(N) x3" = NH3+ rotation). No ORCA.
  --fix      for every job with an imaginary mode:
               1. moves the original <name>.* files to <name>/attempt<k>_imag/
               2. displaces the stationary geometry along the imaginary mode
                  (largest atom displacement = --step A)
               3. reruns the same method with TightOpt (level 1), then, if
                  still imaginary, TightOpt + a finer grid (level 2)
               4. runs dft_check.py on the result
             The fixed job keeps the name <name>.out, so dft_to_sticks.py and
             gapmode.py need no change. The original run stays in the attempt folder.

Level 2 changes the integration grid, so its frequencies can differ from
the rest of the set by a few cm-1. That is well below the ~19 cm-1 SERS
linewidth, and the grid used is written to fix_summary.csv.

  python dft/dft_fix_imag.py --root /local/scratch/mv487/dft/zwitterion_water
  python dft/dft_fix_imag.py --root /local/scratch/mv487/dft/zwitterion_water --fix
  python dft/dft_fix_imag.py --root ... --fix --only glycine,alanine --nprocs 16   # split over two tmux windows
  python dft/dft_fix_imag.py --root /local/scratch/mv487/dft/neutral_water --set neutral_water --fix
      # --set: for runs not made by dft_batch.py (no smiles.txt in the job folders)
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dft_check  # noqa: E402

BOHR = 0.529177210903
ORCA = "/home-local/mv487/orca/orca"
BASE_METHOD = "B3LYP 6-311+G(d,p)"          # identical to dft_batch.py
LEVELS = {1: "TightOpt NumFreq", 2: "TightOpt NumFreq {grid}"}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------ parsing
def _matrix_block(lines, start, nrows, ncols):
    """ORCA column-blocked matrix: a header line of column indices, then nrows
    lines '<row> v v v ...', repeated until ncols columns are read."""
    M = np.zeros((nrows, ncols))
    i, done = start, 0
    while done < ncols and i < len(lines):
        head = lines[i].split()
        if not head or not all(t.isdigit() for t in head):
            i += 1
            continue
        cols = [int(t) for t in head]
        for r in range(nrows):
            p = lines[i + 1 + r].split()
            vals = [float(x) for x in p[1:1 + len(cols)]]
            M[int(p[0]), cols[0]:cols[0] + len(vals)] = vals
        done = cols[-1] + 1
        i += 1 + nrows
    return M


def parse_hess(path):
    """(freqs [3N], modes [3N x 3N] columns = modes, symbols, coords in A)."""
    lines = Path(path).read_text(errors="ignore").splitlines()
    idx = {l.strip(): k for k, l in enumerate(lines) if l.strip().startswith("$")}
    k = idx["$vibrational_frequencies"]
    n = int(lines[k + 1].split()[0])
    freqs = np.array([float(lines[k + 2 + j].split()[1]) for j in range(n)])
    k = idx["$normal_modes"]
    nr, nc = (int(t) for t in lines[k + 1].split()[:2])
    modes = _matrix_block(lines, k + 2, nr, nc)
    k = idx["$atoms"]
    na = int(lines[k + 1].split()[0])
    sym, xyz = [], []
    for j in range(na):
        p = lines[k + 2 + j].split()
        sym.append(p[0])
        xyz.append([float(v) * BOHR for v in p[2:5]])
    return freqs, modes, sym, np.array(xyz)


def parse_out(path):
    """Fallback when there is no .hess: frequencies, NORMAL MODES and the final geometry from the .out."""
    txt = Path(path).read_text(errors="ignore")
    freqs = np.array(dft_check.frequencies(txt) or [])
    lines = txt.splitlines()
    k = max(i for i, l in enumerate(lines) if l.strip() == "NORMAL MODES")
    modes = _matrix_block(lines, k + 1, len(freqs), len(freqs))
    geo = dft_check.final_geometry(txt).splitlines()[2:]
    sym = [g.split()[0] for g in geo]
    xyz = np.array([[float(v) for v in g.split()[1:4]] for g in geo])
    return freqs, modes, sym, xyz


def load(wd, name):
    h = wd / f"{name}.hess"
    return parse_hess(h) if h.exists() else parse_out(wd / f"{name}.out")


# ------------------------------------------------------------------ analysis
def describe_mode(vec, sym, xyz, top=4):
    """Which atoms move, labelled by what they are bonded to: 'H(N) x3, C'."""
    d = np.linalg.norm(vec.reshape(-1, 3), axis=1)
    order = np.argsort(-d)
    labels = []
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    for i in order[:top]:
        if d[i] < 0.25 * d[order[0]]:
            break
        if sym[i] == "H" and heavy:
            j = min(heavy, key=lambda h: np.linalg.norm(xyz[h] - xyz[i]))
            labels.append(f"H({sym[j]}{j})")
        else:
            labels.append(f"{sym[i]}{i}")
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    return ", ".join(f"{l} x{c}" if c > 1 else l for l, c in counts.items())


def imaginary(freqs, modes):
    return [(float(freqs[k]), modes[:, k]) for k in range(len(freqs)) if freqs[k] < -1e-3]


def displaced(xyz, vec, step):
    v = vec.reshape(-1, 3)
    return xyz + step * v / np.linalg.norm(v, axis=1).max()


# ------------------------------------------------------------------ ORCA
def job_info(wd, name, table=None):
    """(SMILES, charge, solvent). SMILES and charge come from smiles.txt (runs made by
    dft_batch.py) or, for older runs without it, from molecules.csv via --set."""
    s = wd / "smiles.txt"
    if s.exists():
        smi, ch = s.read_text().split("\n")[:2]
    elif table and name in table:
        smi, ch = table[name]["smiles"], table[name]["charge"]
    else:
        return None
    out = (wd / f"{name}.out").read_text(errors="ignore")
    solvent, _ = dft_check.input_settings(out)
    return smi, int(ch), solvent


def write_input(path, sym, xyz, charge, solvent, level, grid, nprocs, maxcore):
    kw = LEVELS[level].format(grid=grid)
    lines = [f"! {BASE_METHOD} {kw}"]
    if solvent == "water":
        lines.append("! CPCM(Water)")
    lines += [f"%pal nprocs {nprocs} end", f"%maxcore {maxcore}", "%elprop", "  Polar 1", "end", "",
              f"* xyz {charge} 1"]
    lines += [f"  {s:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for s, (x, y, z) in zip(sym, xyz)]
    lines += ["*", ""]
    Path(path).write_text("\n".join(lines))


def archive(wd, name):
    k = 1
    while (wd / f"attempt{k}_imag").exists():
        k += 1
    dest = wd / f"attempt{k}_imag"
    dest.mkdir()
    keep = {f"{name}_init.xyz", "smiles.txt"}
    for p in wd.iterdir():
        if p.is_file() and p.name.startswith(name) and p.name not in keep:
            shutil.move(str(p), dest / p.name)
    return dest


def run_orca(orca, wd, name):
    with open(wd / f"{name}.out", "w") as fout:
        subprocess.run([orca, str(wd / f"{name}.inp")], stdout=fout, stderr=subprocess.STDOUT, cwd=wd)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="set folder, e.g. /local/scratch/mv487/dft/zwitterion_water")
    ap.add_argument("--only", help="comma-separated names")
    ap.add_argument("--set", help="set name in molecules.csv, for runs without smiles.txt (e.g. neutral_water)")
    ap.add_argument("--molecules", default=str(HERE / "molecules.csv"))
    ap.add_argument("--fix", action="store_true", help="rerun ORCA (default: report only)")
    ap.add_argument("--max_level", type=int, default=2, choices=[1, 2])
    ap.add_argument("--grid", default="DefGrid3", help="level-2 grid keyword (ORCA 5/6; ORCA 4: 'Grid5 FinalGrid6')")
    ap.add_argument("--step", type=float, default=0.10, help="largest atom displacement along the mode, A")
    ap.add_argument("--orca", default=ORCA)
    ap.add_argument("--nprocs", type=int, default=20)
    ap.add_argument("--maxcore", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true", help="with --fix: write inputs, do not archive or run")
    a = ap.parse_args()

    root = Path(a.root)
    only = set(a.only.split(",")) if a.only else None
    table = {r["name"]: r for r in csv.DictReader(open(a.molecules)) if r["set"] == a.set} if a.set else {}
    rows, unknown = [], []
    for wd in sorted(p for p in root.iterdir() if p.is_dir()):
        name = wd.name
        if only and name not in only:
            continue
        if not (wd / f"{name}.out").exists():
            continue
        info = job_info(wd, name, table)
        if info is None:
            unknown.append(name)
            continue
        smi, ch, solvent = info
        chk = dft_check.check_job(wd, name, smi, ch)
        if not chk.get("n_imaginary"):
            continue
        try:
            freqs, modes, sym, xyz = load(wd, name)
        except Exception as e:  # noqa: BLE001
            log(f"{name}: cannot read modes ({e})")
            continue
        im = imaginary(freqs, modes)
        other = [p for p in chk["detail"].split("; ") if "imaginary" not in p]
        for f, v in im:
            log(f"{name:16s} {f:8.1f} cm-1   moving: {describe_mode(v, sym, xyz)}"
                + (f"   ALSO: {'; '.join(other)}" if other else ""))
        rows.append((wd, name, smi, ch, solvent, im, sym, xyz, other))

    if unknown:
        log(f"{len(unknown)} folder(s) skipped, no smiles.txt and not in --set {a.set}: {unknown[:6]}"
            + ("" if a.set else "  -> these look like older runs: add --set <set name>, e.g. --set neutral_water"))
    if not rows:
        log("no job with imaginary modes" + (" among the folders that could be read" if unknown else ""))
        return
    if not a.fix:
        log(f"{len(rows)} job(s) with imaginary modes. Rerun with --fix to repair them "
            "(add --dry-run first to see the inputs).")
        return

    summary = []
    for wd, name, smi, ch, solvent, im, sym, xyz, other in rows:
        if other:
            log(f"{name}: not only an imaginary mode ({'; '.join(other)}) -- fix that first, skipped")
            summary.append({"name": name, "status": "SKIPPED", "detail": "; ".join(other)})
            continue
        status, detail, level = "FAIL", "", 0
        for level in range(1, a.max_level + 1):
            f0, v0 = min(im, key=lambda t: t[0])                 # the most negative mode
            new = displaced(xyz, v0, a.step)
            if a.dry_run:
                write_input(wd / f"{name}_fix{level}.inp", sym, new, ch, solvent, level, a.grid, a.nprocs, a.maxcore)
                log(f"{name}: wrote {name}_fix{level}.inp (displaced along {f0:.1f} cm-1) [dry run]")
                status, detail = "inputs written", ""
                break
            dest = archive(wd, name)
            if not (wd / "smiles.txt").exists():        # older run: record what it is, as dft_batch.py does
                (wd / "smiles.txt").write_text(f"{smi}\n{ch}\n")
            write_input(wd / f"{name}.inp", sym, new, ch, solvent, level, a.grid, a.nprocs, a.maxcore)
            log(f"{name}: level {level} ({LEVELS[level].format(grid=a.grid)}), displaced {a.step} A along "
                f"{f0:.1f} cm-1; previous run -> {dest.name}/")
            t0 = time.time()
            run_orca(a.orca, wd, name)
            chk = dft_check.check_job(wd, name, smi, ch)
            log(f"{name}: level {level} {chk['verdict']} after {(time.time() - t0) / 60:.0f} min | {chk['detail']}")
            status, detail = chk["verdict"], chk["detail"]
            if chk["verdict"] == "PASS" or not chk.get("n_imaginary"):
                break
            try:                                                   # still imaginary: go again from here
                freqs, modes, sym, xyz = load(wd, name)
                im = imaginary(freqs, modes)
            except Exception as e:  # noqa: BLE001
                detail += f"; cannot read modes ({e})"
                break
        grid = a.grid if level == 2 else "default"
        summary.append({"name": name, "status": status, "level": level, "grid": grid, "detail": detail,
                        "original_imag_cm-1": "; ".join(f"{f:.1f}" for f, _ in rows[[r[1] for r in rows].index(name)][5])})

    log("================= FIX SUMMARY =================")
    for s in summary:
        log(f"  {s['name']:16s} {s['status']:14s} level {s.get('level', '-')}  {s.get('detail', '')}")
    if not a.dry_run:
        path = root / "fix_summary.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "status", "level", "grid", "original_imag_cm-1", "detail"])
            if new:
                w.writeheader()
            w.writerows(summary)
        log(f"-> {path}")
        log("Then: python dft/dft_check.py --root " + str(root))


if __name__ == "__main__":
    main()
