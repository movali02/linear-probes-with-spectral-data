#!/usr/bin/env bash
# Kit 2c: 20 analytes, the corrected SERS pipeline, the frequency correction,
# and the test of what the probe reads (positions only).
#
#   STAGE=cpu bash scripts/run_step2c.sh    # strings, ladders, classical curves, T1 control retest   (no GPU)
#   STAGE=gpu bash scripts/run_step2c.sh    # activations + probes for v2m, v2mf, and v2mf with fewer fields
#   bash scripts/run_step2c.sh              # both
#
# Run `kinit` first: a job that outlives the Kerberos ticket loses your home folder.
# Variants (configs/string_v1.yaml): v2 = SERS fixes; v2m = v2 + matched; v2mf = v2m + frequencies x 1.0134.
# LAYER_STEP=3 keeps 13 of the 37 layers (the layer profile was flat): about 6 GB per extraction, 24 GB in all.
set -uo pipefail
cd "$(dirname "$0")/.."
STAGE=${STAGE:-all}
DFT_ROOT=${DFT_ROOT:-/local/scratch/mv487/dft}
SERS_DIR=${SERS_DIR:-data/raw_sers}
QM9S_CSV=${QM9S_CSV:-data/qm9s/raman_boraden.csv}
QM9S_SMILES=${QM9S_SMILES:-data/qm9s_number_smiles.csv}
MODEL=${MODEL:-Qwen/Qwen3-8B}
BATCH=${BATCH:-16}
LAYER_STEP=${LAYER_STEP:-3}
QM9S_FRAC=${QM9S_FRAC:-0.1}
SOFT_IMAG=${SOFT_IMAG:-50}
N_JOBS=${N_JOBS:-$(nproc)}
RUN_T1=${RUN_T1:-1}
LADDER_VARIANTS=${LADDER_VARIANTS:-"v2m v2mf"}
FIELDS_VARIANT=${FIELDS_VARIANT:-v2mf}
REP=reports/ladder
OUT2C=reports/baselines2c
mkdir -p "$REP" "$OUT2C"
if [ -z "${HF_HOME:-}" ] && [ -d /local/scratch/mv487/hf ]; then export HF_HOME=/local/scratch/mv487/hf; fi
step () { echo; echo "===== $* ====="; }

if [ "$STAGE" != gpu ]; then
  step "1. ORCA outputs -> sticks (one imaginary mode weaker than ${SOFT_IMAG} cm-1 is accepted and flagged)"
  for SET in calibration_gas neutral_gas neutral_water zwitterion_water; do
    if [ -d "$DFT_ROOT/$SET" ]; then
      python dft/dft_to_sticks.py --root "$DFT_ROOT/$SET" --set "$SET" --out_dir "data/dft_sticks/$SET" \
          --soft_imag "$SOFT_IMAG" | grep -E "accept|skip|molecules ->" || true
    else
      echo "[2c] $DFT_ROOT/$SET not found: that rung will be missing"
    fi
  done
  if [ -d data/dft_sticks/calibration_gas/activity ] && [ -s data/processed/qm9s_records.jsonl ]; then
    python dft/calibrate_vs_qm9s.py --qm9s_records data/processed/qm9s_records.jsonl \
        --sticks data/dft_sticks/calibration_gas | tee "$REP/calibration.txt"
  fi

  step "2. strings (SERS + QM9S) for the new variants; a finished one is skipped (REBUILD=1 forces)"
  for V in v2 $LADDER_VARIANTS; do
    D="data/processed_$V"
    if [ -n "${REBUILD:-}" ] || [ ! -s "$D/sers_records.jsonl" ]; then
      python scripts/build_strings.py --variant "$V" --sers_dir "$SERS_DIR" --save_spectra | tee "$OUT2C/build_sers_$V.txt"
    fi
    if [ "$V" != v2 ] && { [ -n "${REBUILD:-}" ] || [ ! -s "$D/qm9s_records.jsonl" ]; }; then
      python scripts/build_strings.py --variant "$V" --qm9s_csv "$QM9S_CSV" --qm9s_smiles "$QM9S_SMILES" \
          | tee "$OUT2C/build_qm9s_$V.txt"
    fi
  done

  step "3. ladders and classical curves"
  for V in $LADDER_VARIANTS; do
    python scripts/build_ladder.py --variant "$V" --qm9s_frac "$QM9S_FRAC" | tee "$REP/build_ladder_$V.txt"
    python scripts/run_probes.py --variant "$V" --baseline_only | tee "$REP/probes_baseline_$V.txt"
    python scripts/run_probes.py --variant "$V" --baseline_only --baseline_feature F1pos \
        | tee "$REP/probes_baseline_F1pos_$V.txt"
  done

  if [ "$RUN_T1" = 1 ]; then
    step "4. T1 again on the corrected SERS pipeline (the control-subtraction test of step 2b)"
    python scripts/run_baselines.py --task T1 --variant v2 --out_dir "$OUT2C/v2" --n_jobs "$N_JOBS" \
        --perm_budget_min "${PERM_BUDGET_MIN:-30}" ${EXTRA_ARGS:-} 2>&1 | tee "$OUT2C/log_T1_v2.txt"
  fi
fi

if [ "$STAGE" != cpu ]; then
  TAG=$(basename "$MODEL")
  probe () {   # $1 = variant, $2 = fields (all | posint | pos)
    local V=$1 F=$2 SUF="" BF=F1
    [ "$F" != all ] && SUF="-$F"
    [ "$F" = pos ] && BF=F1pos
    step "activations + probes: $V, fields $F"
    python scripts/extract_activations.py --model "$MODEL" --variant "$V" --batch_size "$BATCH" \
        --layer_step "$LAYER_STEP" --fields "$F" || return 1
    python scripts/extract_activations.py --model "$MODEL" --variant "$V" --batch_size "$BATCH" \
        --layer_step "$LAYER_STEP" --fields "$F" --random_init || return 1
    python scripts/run_probes.py --variant "$V" --acts_dir "data/activations/$V/$TAG$SUF" \
        --acts_random "data/activations/$V/$TAG-random$SUF" --baseline_feature "$BF" \
        | tee "$REP/probes_${TAG}${SUF}_$V.txt"
  }
  for V in $LADDER_VARIANTS; do probe "$V" all; done
  probe "$FIELDS_VARIANT" pos
  probe "$FIELDS_VARIANT" posint
  echo
  echo "figures: $REP/<variant>/$TAG*/degradation.png"
fi

echo
echo "To send back:"
echo "  zip -r step2c_results.zip reports/ladder reports/baselines2c \$(find data/activations -name meta.json) -x '*.npy'"
