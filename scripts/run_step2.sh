#!/usr/bin/env bash
# Step 2 end to end: rebuild with processed spectra (F3), build the matched
# variant, run T1-T4. CPU only.
#   bash scripts/run_step2.sh
#   N_JOBS=32 OLD_MANIFEST=data/split_manifest.csv OLD_PREDICTIONS=data/test_predictions.csv bash scripts/run_step2.sh
#   SKIP_BUILD=1 TASKS="T1" bash scripts/run_step2.sh          # rerun one task on existing records
# Any path can be overridden: SERS_DIR=... QM9S_CSV=... QM9S_SMILES=...
set -euo pipefail
cd "$(dirname "$0")/.."
SERS_DIR=${SERS_DIR:-data/raw_sers}
QM9S_CSV=${QM9S_CSV:-data/qm9s/raman_boraden.csv}
QM9S_SMILES=${QM9S_SMILES:-data/qm9s_number_smiles.csv}
QM9S_FRAC=${QM9S_FRAC:-0.2}              # QM9S subsample for T3/T4 (and for the saved F3 spectra)
N_JOBS=${N_JOBS:-$(nproc)}
TASKS=${TASKS:-"T1 T2 T3 T4"}
OLD_MANIFEST=${OLD_MANIFEST:-}           # old fine-tune split_manifest.csv -> 58/174 comparison in T1
OLD_PREDICTIONS=${OLD_PREDICTIONS:-}     # old test_predictions csv (file, ..., Result)
META_CSV=${META_CSV:-}                   # optional id,<cols> metadata (e.g. date) for the metadata-only check
PERM_BUDGET_MIN=${PERM_BUDGET_MIN:-120}  # wall-clock minutes of permutation tests per task

if [ -z "${SKIP_BUILD:-}" ]; then
  if ! head -1 "$QM9S_SMILES" 2>/dev/null | grep -q list_pos; then
    echo "$QM9S_SMILES missing or without list_pos: run scripts/run_step1.sh first (alignment check)"; exit 1
  fi
  for V in native matched; do
    # native: same strings as step 1 (the build is deterministic), plus sers/qm9s_spectra.npz
    python scripts/build_strings.py --variant "$V" --sers_dir "$SERS_DIR" \
        --qm9s_csv "$QM9S_CSV" --qm9s_smiles "$QM9S_SMILES" --save_spectra --spectra_frac "$QM9S_FRAC"
  done
fi

T1_ARGS=()
[ -n "$OLD_MANIFEST" ] && T1_ARGS+=(--old_manifest "$OLD_MANIFEST")
[ -n "$OLD_PREDICTIONS" ] && T1_ARGS+=(--old_predictions "$OLD_PREDICTIONS")
[ -n "$META_CSV" ] && T1_ARGS+=(--meta_csv "$META_CSV")
COMMON=(--n_jobs "$N_JOBS" --qm9s_frac "$QM9S_FRAC" --perm_budget_min "$PERM_BUDGET_MIN")

mkdir -p reports/baselines
for T in $TASKS; do
  EXTRA=(); [ "$T" = T1 ] && EXTRA=(${T1_ARGS[@]+"${T1_ARGS[@]}"})
  # SERS-only tasks on the native strings; T4 (QM9S -> SERS) on both variants (tests P4)
  python scripts/run_baselines.py --task "$T" --variant native "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} \
      2>&1 | tee "reports/baselines/log_${T}_native.txt"
  if [ "$T" = T4 ]; then
    python scripts/run_baselines.py --task T4 --variant matched "${COMMON[@]}" \
        2>&1 | tee reports/baselines/log_T4_matched.txt
  fi
done
echo "done: reports/baselines/native/*.md, reports/baselines/matched/T4.md"
