#!/usr/bin/env python3
"""Chemical, label and spectral coverage of the SERS analytes by QM9S.

  python scripts/coverage_report.py --qm9_smiles data/qm9s_number_smiles.csv \
      --records data/processed/sers_records.jsonl data/processed/qm9s_records.jsonl
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit.coverage import (chemical_coverage, label_support, shared_peak_audit,  # noqa: E402
                                    spectral_summary, to_markdown_table)
from probe2circuit.io import load_config, read_jsonl  # noqa: E402
from probe2circuit.pipeline import load_analytes  # noqa: E402


def write_csv(rows, path):
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--analytes", default="data/analytes.csv")
    ap.add_argument("--qm9_smiles", help="CSV with a `smiles` column")
    ap.add_argument("--qm9_jsonl", help="make_views.py peak-list jsonl (SMILES taken from responses)")
    ap.add_argument("--usable_min_peaks", type=int, default=3,
                    help="a QM9S string with fewer window peaks than this counts as unusable")
    ap.add_argument("--records", nargs="*", default=[])
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    cfg = load_config(a.config)
    analytes = load_analytes(a.analytes)
    if a.qm9_jsonl:
        from probe2circuit.io import iter_peaklist_jsonl
        qm9 = [smi for _, smi, *_ in iter_peaklist_jsonl(a.qm9_jsonl)]
        qm9_src = a.qm9_jsonl
    else:
        qm9 = [r["smiles"] for r in csv.DictReader(open(a.qm9_smiles))]
        qm9_src = a.qm9_smiles
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    recs = [r for p in a.records for r in read_jsonl(p)]
    qrecs = [r for r in recs if r["source"] == "qm9s"]
    usable = {r["smiles"] for r in qrecs if r["n_peaks"] >= a.usable_min_peaks} if qrecs else None
    chem = chemical_coverage(analytes, qm9)
    labs = label_support(analytes, qm9, usable_smiles=usable)
    spec, overlap = spectral_summary(recs, cfg) if recs else ([], {})
    shared = shared_peak_audit(recs, cfg) if recs else []

    write_csv(chem, out / "coverage_chemical.csv")
    write_csv(labs, out / "coverage_labels.csv")
    write_csv(spec, out / "coverage_spectral.csv")

    n_in = sum(r["in_qm9_domain"] for r in chem)
    n_exact = sum(r["exact_in_qm9"] for r in chem)
    md = [f"# Coverage report ({cfg['schema']})\n", a.note + "\n" if a.note else "",
          f"QM9 reference: {len(qm9)} SMILES from `{qm9_src}`.\n",
          (f"\nUsable QM9S strings (>= {a.usable_min_peaks} peaks in {cfg['window'][0]:.0f}-"
           f"{cfg['window'][1]:.0f} cm-1): {len(usable)} of {len(qm9)} molecules.\n" if usable is not None else ""),
          f"\n## 1. Chemical coverage\n\n{n_in}/{len(chem)} analytes inside QM9's domain "
          f"(<=9 heavy atoms, C/N/O/F, no isotopes); {n_exact} found verbatim in the reference set.\n\n",
          to_markdown_table(chem),
          "\n## 2. Label support in QM9\n\nA probe trained on QM9S cannot learn a label that QM9 "
          "does not contain. Status is judged on `n_qm9s_usable` when QM9S records are supplied "
          "(molecules whose window string has enough peaks to learn from). `NO QM9 SUPPORT` labels "
          "must be dropped from any QM9S->SERS transfer claim.\n\n",
          to_markdown_table(labs),
          "\n## 3. Spectral shape by source\n\nToken counts are estimates (1 token per digit); "
          "run leakage checks with `--tokenizer` on Aura for exact numbers.\n\n",
          to_markdown_table(spec),
          "\nPeak-position histogram overlap (25 cm-1 bins, 1 = identical distributions):\n\n",
          "".join(f"- {k}: {v}\n" for k, v in overlap.items()) or "_(one source only)_\n",
          "\n## 4. Peaks shared across different analytes\n\nBands present in spectra of at least half "
          "of the SERS analytes. A flag, not a verdict: shared chemistry (e.g. ring breathing near "
          "1000 cm-1 in Phe/Trp/Tyr) also shows up here. Check each against the CB-only spectra.\n\n",
          to_markdown_table(shared)]
    (out / "coverage.md").write_text("".join(md))
    print(f"wrote {out/'coverage.md'}")


if __name__ == "__main__":
    main()
