#!/usr/bin/env bash
# Step 1 end to end.
#   bash scripts/run_step1.sh                  # uses data/raw_sers and data/qm9s/{raman_boraden.csv,qm9s.pt}
#   QM9S_JSONL=.../qm9s_raman.jsonl bash scripts/run_step1.sh   # peak-list fallback instead of the CSV
# Any path can be overridden: SERS_DIR=... QM9S_CSV=... QM9S_PT=... TOKENIZER=...
set -euo pipefail
cd "$(dirname "$0")/.."
SERS_DIR=${SERS_DIR:-data/raw_sers}                  # all .txt, including CB_* substrate-only files
QM9S_JSONL=${QM9S_JSONL:-}
if [ -z "$QM9S_JSONL" ]; then
  QM9S_CSV=${QM9S_CSV:-data/qm9s/raman_boraden.csv}
  QM9S_PT=${QM9S_PT:-data/qm9s/qm9s.pt}
else
  QM9S_CSV=""; QM9S_PT=""
fi
TOKENIZER=${TOKENIZER:-}                             # e.g. Qwen/Qwen2.5-7B-Instruct for exact token counts
OUT=${OUT:-reports/run_$(date +%Y%m%d)}
TOK=(); [ -n "$TOKENIZER" ] && TOK=(--tokenizer "$TOKENIZER")

# qm9s_raman.jsonl (make_views.py) pairs spectra with the wrong SMILES; refuse it
if [ -n "$QM9S_JSONL" ] && [ -z "${ALLOW_MISALIGNED_JSONL:-}" ]; then
  echo "QM9S_JSONL: this file was built with the broken SMILES join (see scripts/qm9s_alignment_check.py)."
  echo "Use QM9S_CSV + QM9S_PT instead. Set ALLOW_MISALIGNED_JSONL=1 only to reproduce old results."
  exit 1
fi

# stop here, before anything slow, if inputs are missing or still placeholders
if [ -n "$QM9S_JSONL" ]; then
  python scripts/preflight.py --sers_dir "$SERS_DIR" --qm9s_jsonl "$QM9S_JSONL" ${TOK[@]+"${TOK[@]}"}
else
  python scripts/preflight.py --sers_dir "$SERS_DIR" --qm9s_csv "$QM9S_CSV" --qm9s_pt "$QM9S_PT" ${TOK[@]+"${TOK[@]}"}
fi

if [ -n "$QM9S_CSV" ]; then
  # The SMILES map must come from the alignment check (it has a list_pos column).
  # Maps made by export_qm9s_smiles.py use the broken make_views.py join and are replaced.
  if [ ! -s data/qm9s_number_smiles.csv ] || ! head -1 data/qm9s_number_smiles.csv | grep -q list_pos; then
    python scripts/qm9s_alignment_check.py --csv "$QM9S_CSV" --pt "$QM9S_PT" --out data/qm9s_number_smiles.csv
  fi
  QM9S_ARGS=(--qm9s_csv "$QM9S_CSV" --qm9s_smiles data/qm9s_number_smiles.csv)
  QM9_REF=(--qm9_smiles data/qm9s_number_smiles.csv)
else
  QM9S_ARGS=(--qm9s_jsonl "$QM9S_JSONL")
  QM9_REF=(--qm9_jsonl "$QM9S_JSONL")
fi

# Generic CB[5] reference for spectra without their own CB file (the amino-acid
# and indole standards), built from the cell-assay CB files. Rebuilt whenever a
# CB file is newer than it. MANIFEST (optional) limits it to the files listed there.
GENERIC_CB=${GENERIC_CB:-data/generic_cb.txt}
MANIFEST=${MANIFEST:-}
if [ ! -s "$GENERIC_CB" ] || [ -n "$(find -L "$SERS_DIR" -name 'CB_*.txt' -newer "$GENERIC_CB" -print -quit)" ]; then
  python scripts/make_generic_cb.py --sers_dir "$SERS_DIR" --out "$GENERIC_CB" ${MANIFEST:+--manifest "$MANIFEST"}
fi
mkdir -p "$OUT" && cp "${GENERIC_CB%.txt}_report.json" "$OUT/" 2>/dev/null || true

python scripts/build_strings.py --sers_dir "$SERS_DIR" "${QM9S_ARGS[@]}" --out_dir data/processed
python scripts/coverage_report.py "${QM9_REF[@]}" \
    --records data/processed/sers_records.jsonl data/processed/qm9s_records.jsonl --out_dir "$OUT"
python scripts/run_leakage_checks.py --records data/processed/sers_records.jsonl ${TOK[@]+"${TOK[@]}"} \
    --out "$OUT/leakage_sers.md"
python scripts/run_leakage_checks.py --records data/processed/qm9s_records.jsonl \
    --label_key labels.fg_aromatic_ring --identity_key smiles ${TOK[@]+"${TOK[@]}"} --out "$OUT/leakage_qm9s.md" || true
