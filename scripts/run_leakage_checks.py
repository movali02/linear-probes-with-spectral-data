#!/usr/bin/env python3
"""Run every leakage check on a records file. Exit code 1 if any check FAILs.

  python scripts/run_leakage_checks.py --records data/processed/sers_records.jsonl
  # on Aura, exact token counts:
  python scripts/run_leakage_checks.py --records ... --tokenizer Qwen/Qwen2.5-7B-Instruct
  # audit the old string JSONs as well:
  python scripts/run_leakage_checks.py --records ... --legacy_json_dir "path/to/string JSONs"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit.coverage import to_markdown_table  # noqa: E402
from probe2circuit.io import load_config, read_jsonl  # noqa: E402
from probe2circuit.leakage import audit_legacy_json, run_all  # noqa: E402
from probe2circuit.pipeline import load_analytes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--analytes", default="data/analytes.csv")
    ap.add_argument("--records", required=True)
    ap.add_argument("--label_key", default="analyte")
    ap.add_argument("--identity_key", default=None,
                    help="what makes two records the same sample for the duplicate check "
                         "(default: label_key; use `smiles` for QM9S)")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--legacy_json_dir", default=None)
    ap.add_argument("--out", default="reports/leakage.md")
    a = ap.parse_args()

    cfg = load_config(a.config)
    analytes = load_analytes(a.analytes)
    recs = read_jsonl(a.records)
    results = run_all(recs, cfg, analytes, a.tokenizer, a.label_key, a.identity_key)

    lines = [f"# Leakage checks: `{Path(a.records).name}` ({len(recs)} records, schema {cfg['schema']})\n\n"]
    for r in results:
        lines.append(f"- **{r['status']}** `{r['check']}`: {r['detail']}\n")
        if r.get("examples"):
            lines.append(f"  - examples: `{r['examples']}`\n")
    if a.legacy_json_dir:
        docs = {p.stem: json.load(open(p)) for p in sorted(Path(a.legacy_json_dir).glob("*.json"))}
        cat_path = Path(a.records).parent / "substrate_catalogue.json"
        cat = json.load(open(cat_path))["catalogue"] if cat_path.exists() else None
        rows = audit_legacy_json(docs, analytes, recs, cat)
        lines += ["\n## Old strings vs new strings for the same spectra\n\n",
                  "`old_substrate_peaks` = old peaks within 4 cm-1 of a major CB[5] band. Old token "
                  "counts are for the full old string (prefix + 9 fields per peak).\n\n",
                  to_markdown_table(rows)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(lines))
    Path(a.out).with_suffix(".json").write_text(json.dumps(results, indent=1, default=str))
    for r in results:
        print(f"{r['status']:5s} {r['check']:24s} {r['detail']}")
    print(f"-> {a.out}")
    sys.exit(1 if any(r["status"] == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
