#!/usr/bin/env bash
# Step 2b: the fixes from the step-2 review. CPU only.
#   bash scripts/run_step2b.sh
#   REBUILD_SERS=1 bash scripts/run_step2b.sh       # after adding files to data/raw_sers (e.g. the control cultures)
#   TASKS="T1 T2" bash scripts/run_step2b.sh        # a subset
#   EXTRA_ARGS="--quick" bash scripts/run_step2b.sh # smoke test (few bootstraps/shuffles, 2 % of QM9S)
#   CONTROL_REGEX='_ctrlWT_' bash scripts/run_step2b.sh   # which ids are no-analyte controls (default: ctrl|control|blank|noaa|none|neg|medium)
set -euo pipefail
cd "$(dirname "$0")/.."
SERS_DIR=${SERS_DIR:-data/raw_sers}
QM9S_CSV=${QM9S_CSV:-data/qm9s/raman_boraden.csv}
QM9S_SMILES=${QM9S_SMILES:-data/qm9s_number_smiles.csv}
QM9S_FRAC=${QM9S_FRAC:-0.2}
N_JOBS=${N_JOBS:-$(nproc)}
TASKS=${TASKS:-"T1 T2 T3 T4"}
OLD_MANIFEST=${OLD_MANIFEST:-}
OLD_PREDICTIONS=${OLD_PREDICTIONS:-}
CONTROL_REGEX=${CONTROL_REGEX:-}
PERM_BUDGET_MIN=${PERM_BUDGET_MIN:-120}
OUT=reports/baselines2b
mkdir -p "$OUT"

# 0. data checks (seconds)
python scripts/inspect_cb.py --sers_dir "$SERS_DIR" --out "$OUT/cb_inspect" | tee "$OUT/cb_inspect.txt"
if [ -s data/processed/qm9s_records.jsonl ]; then
  python scripts/qm9s_benzene_check.py --records data/processed/qm9s_records.jsonl | tee "$OUT/qm9s_frequency_check.txt"
fi

# 1. SERS records: only rebuilt on request (new files). QM9S records are left as they are.
if [ -n "${REBUILD_SERS:-}" ]; then
  for V in native matched; do
    python scripts/build_strings.py --variant "$V" --sers_dir "$SERS_DIR" --save_spectra
  done
fi
# 2. QM9S with the window extended to 1800 cm-1 (T3 edge test); built once
if [[ " $TASKS " == *" T3 "* ]] && [ ! -s data/processed_wide/qm9s_spectra.npz ]; then
  python scripts/build_strings.py --variant wide --qm9s_csv "$QM9S_CSV" --qm9s_smiles "$QM9S_SMILES" \
      --save_spectra --spectra_frac "$QM9S_FRAC"
fi

T1_ARGS=()
[ -n "$OLD_MANIFEST" ] && T1_ARGS+=(--old_manifest "$OLD_MANIFEST")
[ -n "$OLD_PREDICTIONS" ] && T1_ARGS+=(--old_predictions "$OLD_PREDICTIONS")
[ -n "$CONTROL_REGEX" ] && T1_ARGS+=(--control_regex "$CONTROL_REGEX")
COMMON=(--n_jobs "$N_JOBS" --qm9s_frac "$QM9S_FRAC" --perm_budget_min "$PERM_BUDGET_MIN" ${EXTRA_ARGS:-})

for T in $TASKS; do
  EXTRA=(); [ "$T" = T1 ] && EXTRA=(${T1_ARGS[@]+"${T1_ARGS[@]}"})
  python scripts/run_baselines.py --task "$T" --variant native --out_dir "$OUT/native" "${COMMON[@]}" \
      ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee "$OUT/log_${T}_native.txt"
  if [ "$T" = T3 ]; then
    python scripts/run_baselines.py --task T3 --variant wide --out_dir "$OUT/wide" "${COMMON[@]}" \
        2>&1 | tee "$OUT/log_T3_wide.txt"
    python scripts/compare_string_loss.py --native "$OUT/native/T3.csv" --wide "$OUT/wide/T3.csv" \
        | tee "$OUT/T3_string_loss.txt"
  fi
  if [ "$T" = T4 ]; then
    python scripts/run_baselines.py --task T4 --variant matched --out_dir "$OUT/matched" "${COMMON[@]}" \
        2>&1 | tee "$OUT/log_T4_matched.txt"
  fi
done
echo "done: $OUT/native/*.md, $OUT/matched/T4.md, $OUT/T3_string_loss.txt, $OUT/cb_inspect.txt"
