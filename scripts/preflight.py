#!/usr/bin/env python3
"""Check that everything step 1 needs is in place BEFORE the long run.

  python scripts/preflight.py                       # uses the default paths
  python scripts/preflight.py --sers_dir ... --qm9s_csv ... --qm9s_pt ...

Exit code 1 if anything blocking is wrong. Prints what it found."""
import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ok = True


def report(status, msg):
    global ok
    if status == "FAIL":
        ok = False
    print(f"{status:5s} {msg}")


def is_placeholder(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(11) == b"PLACEHOLDER"
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sers_dir", default=str(ROOT / "data/raw_sers"))
    ap.add_argument("--qm9s_csv", default=str(ROOT / "data/qm9s/raman_boraden.csv"))
    ap.add_argument("--qm9s_pt", default=str(ROOT / "data/qm9s/qm9s.pt"))
    ap.add_argument("--qm9s_jsonl", default=None, help="only if you're using the peak-list fallback")
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()

    # --- python packages
    for mod in ["numpy", "scipy", "pandas", "sklearn", "yaml", "rdkit"]:
        try:
            importlib.import_module(mod)
            report("OK", f"import {mod}")
        except ImportError:
            report("FAIL", f"import {mod}  -> pip install {'scikit-learn' if mod == 'sklearn' else 'pyyaml' if mod == 'yaml' else mod}")

    # --- QM9S inputs
    if a.qm9s_jsonl:
        p = Path(a.qm9s_jsonl)
        report("OK" if p.exists() else "FAIL", f"QM9S peak-list jsonl: {p}")
    else:
        for label, p in [("raman_boraden.csv", Path(a.qm9s_csv)), ("qm9s.pt", Path(a.qm9s_pt))]:
            smiles_csv = ROOT / "data/qm9s_number_smiles.csv"
            if label == "qm9s.pt" and smiles_csv_ok():
                report("OK", f"qm9s.pt not needed: {smiles_csv} made by the alignment check")
                continue
            if label == "qm9s.pt" and smiles_csv.exists():
                report("WARN", f"{smiles_csv} was made with the old .number join and will be rebuilt "
                               "by scripts/qm9s_alignment_check.py (needs qm9s.pt)")
            if not p.exists():
                report("FAIL", f"{label} missing at {p}")
            elif is_placeholder(p):
                report("FAIL", f"{label} is still the placeholder. Run: ln -sf /real/path/{label} {p}")
            else:
                report("OK", f"{label}: {p.resolve()} ({p.stat().st_size / 1e6:.0f} MB)")
        p = Path(a.qm9s_csv)
        if p.exists() and not is_placeholder(p):
            with open(p) as f:
                head = f.readline().rstrip("\n").split(",")
                first = f.readline().split(",")
            try:
                ax = [float(h) for h in head[1:]]
                report("OK" if min(ax) <= 500 and max(ax) >= 1750 else "FAIL",
                       f"CSV axis {min(ax):.0f}-{max(ax):.0f} cm-1, {len(ax)} points "
                       f"(must cover 500-1750); first molecule number {first[0]}")
            except ValueError:
                report("FAIL", "CSV header is not a numeric wavenumber axis")
        if not smiles_csv_ok():
            try:
                importlib.import_module("torch")
                importlib.import_module("torch_geometric")
                report("OK", "torch + torch_geometric available to read qm9s.pt")
            except ImportError as e:
                report("FAIL", f"reading qm9s.pt needs torch and torch_geometric ({e.name} missing)")

    # --- SERS inputs
    from probe2circuit import substrate as sub
    from probe2circuit.pipeline import analyte_from_id, kind_from_id, load_analytes, normalise_id
    analytes = load_analytes(ROOT / "data/analytes.csv")
    from probe2circuit.io import find_sers_files, subset_of
    files = find_sers_files(a.sers_dir)
    if not files:
        report("FAIL", f"no SERS .txt files in {a.sers_dir}")
    else:
        ids = [normalise_id(f.stem) for f in files]
        dup = [k for k, v in Counter(ids).items() if v > 1]
        kinds = Counter(kind_from_id(i) for i in ids)
        report("FAIL" if dup else "OK", f"{len(files)} SERS files: {dict(kinds)}"
               + (f"; duplicate ids after normalising (same name in two folders?): {dup[:5]}" if dup else ""))
        by_folder = Counter((subset_of(f, a.sers_dir), kind_from_id(normalise_id(f.stem))) for f in files)
        report("OK", "by folder: " + ", ".join(f"{d}/{k}={n}" for (d, k), n in sorted(by_folder.items())))
        no_an = [i for i in ids if analyte_from_id(i, analytes) is None]
        report("WARN" if no_an else "OK",
               f"{len(ids) - len(no_an)}/{len(ids)} ids matched to an analyte"
               + (f"; unmatched (add a row to data/analytes.csv or rename): {no_an[:10]}" if no_an else ""))
        refs = {i: None for i in ids if kind_from_id(i) == "substrate_only"}
        unpaired = lambda group: [i for i in group
                                  if sub.find_reference(i, {k: 0 for k in refs}, "condition_mean")[1] is None]
        cells = [i for i in ids if kind_from_id(i) == "cell"]
        stds = [i for i in ids if kind_from_id(i) == "standard"]
        cu = unpaired(cells)
        report("WARN" if cu else "OK",
               f"{len(cells) - len(cu)}/{len(cells)} cell spectra have their own CB file (condition mean)"
               + (f"; without, will use the generic CB: e.g. {cu[:5]}" if cu else ""))
        su = unpaired(stds)
        gpath = ROOT / "data/generic_cb.txt"
        if su:
            src = f"{gpath} (exists)" if gpath.exists() else f"built from {len(refs)} CB files this run"
            report("OK" if (gpath.exists() or refs) else "FAIL",
                   f"{len(su)} standard spectra without their own CB file will use the generic CB: {src}")

    # --- tokenizer
    if a.tokenizer:
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(a.tokenizer)
            report("OK", f"tokenizer {a.tokenizer} loads")
        except Exception as e:  # noqa: BLE001
            report("FAIL", f"tokenizer {a.tokenizer} does not load: {e!s:.120}. "
                           "On a node without internet, pass the local model directory.")
    else:
        report("WARN", "no tokenizer given: token counts will be estimates")

    print("\nREADY" if ok else "\nNOT READY: fix the FAIL lines above")
    sys.exit(0 if ok else 1)


def smiles_csv_ok():
    """True only for a map written by qm9s_alignment_check.py (it has a list_pos column)."""
    p = ROOT / "data/qm9s_number_smiles.csv"
    if not p.exists() or is_placeholder(p):
        return False
    with open(p) as f:
        return "list_pos" in f.readline()


if __name__ == "__main__":
    main()
