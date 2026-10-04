#!/usr/bin/env bash
# The degradation figure: ladder strings -> activations -> probes.
#   STAGE=cpu bash scripts/run_ladder.sh      # sticks, calibration, ladder strings, classical curve (no GPU)
#   STAGE=gpu bash scripts/run_ladder.sh      # activations (Qwen3-8B + untrained control) and probes
#   bash scripts/run_ladder.sh                # both
# MODEL=/path/to/Qwen3-8B if the GPU node has no internet. DFT_ROOT is where dft_batch.py wrote its sets.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${STAGE:-all}
VARIANT=${VARIANT:-matched}
DFT_ROOT=${DFT_ROOT:-/local/scratch/mv487/dft}
MODEL=${MODEL:-Qwen/Qwen3-8B}
BATCH=${BATCH:-16}
LAYER_STEP=${LAYER_STEP:-1}
QM9S_FRAC=${QM9S_FRAC:-0.1}
REP=reports/ladder
mkdir -p "$REP"

if [ "$STAGE" != gpu ]; then
  # 1. ORCA outputs -> stick files (three intensity conventions per set). A set folder that is missing is skipped.
  for SET in calibration_gas neutral_gas neutral_water zwitterion_water; do
    if [ -d "$DFT_ROOT/$SET" ]; then
      python dft/dft_to_sticks.py --root "$DFT_ROOT/$SET" --set "$SET" --out_dir "data/dft_sticks/$SET"
    else
      echo "[ladder] $DFT_ROOT/$SET not found: that rung will be missing (set DFT_ROOT, or rename the folder)"
    fi
  done
  # 2. frequency scale and intensity convention, from the 9 molecules that are in QM9S
  if [ -d data/dft_sticks/calibration_gas/activity ] && [ -s data/processed/qm9s_records.jsonl ]; then
    python dft/calibrate_vs_qm9s.py --qm9s_records data/processed/qm9s_records.jsonl \
        --sticks data/dft_sticks/calibration_gas | tee "$REP/calibration.txt"
  fi
  # 3. matched records (built by step 2; rebuilt here only if missing)
  if [ ! -s "data/processed_$VARIANT/qm9s_records.jsonl" ] || [ ! -s "data/processed_$VARIANT/sers_records.jsonl" ]; then
    python scripts/build_strings.py --variant "$VARIANT" --sers_dir "${SERS_DIR:-data/raw_sers}" \
        --qm9s_csv "${QM9S_CSV:-data/qm9s/raman_boraden.csv}" --qm9s_smiles "${QM9S_SMILES:-data/qm9s_number_smiles.csv}" \
        --save_spectra
  fi
  # 4. ladder strings + similarity table, then the classical curve (the binned string, no model)
  python scripts/build_ladder.py --variant "$VARIANT" --qm9s_frac "$QM9S_FRAC" | tee "$REP/build_ladder.txt"
  python scripts/run_probes.py --variant "$VARIANT" --baseline_only | tee "$REP/probes_baseline.txt"
fi

if [ "$STAGE" != cpu ]; then
  TAG=$(basename "$MODEL")
  python scripts/extract_activations.py --model "$MODEL" --variant "$VARIANT" --batch_size "$BATCH" --layer_step "$LAYER_STEP"
  python scripts/extract_activations.py --model "$MODEL" --variant "$VARIANT" --batch_size "$BATCH" --layer_step "$LAYER_STEP" --random_init
  python scripts/run_probes.py --variant "$VARIANT" --acts_dir "data/activations/$VARIANT/$TAG" \
      --acts_random "data/activations/$VARIANT/$TAG-random" | tee "$REP/probes_$TAG.txt"
  echo "figure: $REP/$VARIANT/$TAG/degradation.png"
fi
