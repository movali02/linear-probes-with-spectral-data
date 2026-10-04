# Applying this kit on Aura (27 Sep 2026)

Unzip over the root of your existing `probe2circuit` checkout. Paths match the
repo layout, and nothing outside these files changes.

```bash
cd ~ && unzip -o probe2circuit_dft_kit.zip      # writes into ~/probe2circuit/
cd ~/probe2circuit && pytest -q tests/test_dft.py
```

## What is in it

| path | new / changed | purpose |
|---|---|---|
| `scripts/qm9s_alignment_check.py` | new | finds the correct `raman_boraden.csv` ↔ `qm9s.pt` join and writes the SMILES map only when the spectra confirm it |
| `scripts/run_step1.sh` | changed | builds the SMILES map with the alignment check; refuses `qm9s_raman.jsonl` |
| `scripts/preflight.py` | changed | treats a SMILES map from the old `.number` join as invalid |
| `scripts/export_qm9s_smiles.py` | changed | marked deprecated (wrong join) |
| `dft/*` | new | DFT kit; see `dft/README_DFT.md` |
| `tests/test_dft.py` | new | 11 tests for the kit |

## Three things to run, in this order

**1. Fix the QM9S join** (a few minutes):

```bash
rm -f data/qm9s_number_smiles.csv
python scripts/qm9s_alignment_check.py --csv data/qm9s/raman_boraden.csv \
    --pt data/qm9s/qm9s.pt --out data/qm9s_number_smiles.csv
```

It prints an AUROC table: one line per candidate join. Exactly one line should
say PASS, with both AUROCs close to 1. Send me that table whatever it shows.

**2. Record decision D1** (10 scans = one culture, so a culture is the split unit):

```bash
sed -i 's/^  group_by: id$/  group_by: strip_replicate   # D1: Cell_536_trp_1..10 = one culture/' configs/string_v1.yaml
grep -n group_by configs/string_v1.yaml
```

**3. Start the DFT** (see `dft/README_DFT.md` for the full order):

```bash
python dft/dft_check.py --root /local/scratch/mv487/dft/AAs --set neutral_gas   # what are the existing runs?
tmux new -s dft
python dft/dft_batch.py --set calibration_gas --dry-run && python dft/dft_batch.py --set calibration_gas
python dft/dft_batch.py --set zwitterion_water
```

Then, once the SERS files and CB files are uploaded, rerun `bash scripts/run_step1.sh`.
