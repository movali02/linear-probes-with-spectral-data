# Kit 2c: apply and run

Unzip from the folder that CONTAINS the checkout (not inside it):

    cd /local/scratch/mv487/qm9s
    unzip -o probe2circuit_step2c.zip
    cd probe2circuit
    python -m pytest -q tests/test_step2c.py        # 7 tests, about 1 minute

Then (run `kinit` first; the GPU part needs about 24 GB of disk under data/activations):

    STAGE=cpu N_JOBS=32 nohup bash scripts/run_step2c.sh > step2c_cpu.log 2>&1 &
    # when the line "3. ladders and classical curves" has finished in the log, the GPU stage can start:
    STAGE=gpu nohup bash scripts/run_step2c.sh > step2c_gpu.log 2>&1 &

The GPU stage only needs steps 1-3 of the CPU stage. Step 4 (T1) is slow and
independent; RUN_T1=0 skips it.

## What changes

| | old (`matched`) | new |
|---|---|---|
| analytes on every rung | 18 | 20: a job whose only problem is ONE imaginary mode weaker than 50 cm-1 is accepted and listed in `data/dft_sticks/<set>/soft_imag.csv` (Phe gas, Asn water) |
| window | 500-1750 | 510-1800 (amide / lactam C=O; your axis runs to 1900) |
| CB reference band | 829 +/- 4 | 829 +/- 8, and each spectrum is shifted so its band sits at 829 |
| generic CB | data/generic_cb.txt | rebuilt in-run from the aligned CB files |
| simulated frequencies | as in QM9S | variant `v2mf`: x 1.0134 (QM9S and DFT) |

Variants: `v2` (SERS fixes, native linewidths, for T1), `v2m` (v2 + matched),
`v2mf` (v2m + frequency correction). The old variants are untouched, so every
earlier result can be reproduced.

## What it runs

1. Sticks for all four DFT sets, with the soft-imaginary rule.
2. Strings: SERS for v2, v2m, v2mf; QM9S for v2m, v2mf.
3. Ladders for v2m and v2mf, with a new table of what the strings look like on
   each rung (`reports/ladder/<variant>/string_stats.csv`), and the classical
   curve twice: on the binned string (F1) and on positions only (F1pos).
4. T1 on v2: the control-subtraction test again, now that the control
   cultures' CB files are accepted and the axes are aligned.
5. GPU: activations and probes for
   - v2m, all fields
   - v2mf, all fields
   - v2mf, positions and intensities (`--fields posint`)
   - v2mf, positions only (`--fields pos`), with the F1pos baseline
   each for the pretrained and the untrained model, every third layer.

## What each run answers

| question | compare |
|---|---|
| Does the probe lean on intensity and width digits? (finding 2 of the ladder review) | `v2mf/Qwen3-8B` vs `-posint` vs `-pos` at R5 and R6 |
| Does the frequency correction help? (P14) | `v2m` vs `v2mf` at R8, baseline and probe |
| Amide and the window (P17) | amide at R0: `matched` (done) vs `v2m` |
| Aromatic ring with four positives (P16) | `v2m`, `v2mf` |
| Is the control subtraction any use? | `reports/baselines2c/v2/T1.md`, gate lines F3.ctrl / ncm |
| Why did the intensity rungs differ? | `string_stats.csv`: peaks per string, share of weak peaks, where the strongest peak sits |

New in every `ladder.md`: paired differences between methods with a CI (bold =
resolved), and single-positive labels as that analyte's rank.

## Notes

- `extract_activations.py` now stores a fingerprint of the strings. A rebuilt
  ladder with the same number of strings is no longer mistaken for a finished
  extraction.
- `LAYER_STEP=1` restores all 37 layers (three times the disk).
- `SOFT_IMAG=0` restores the strict rule (18 analytes).
- To send back:
  `zip -r step2c_results.zip reports/ladder reports/baselines2c $(find data/activations -name meta.json) -x '*.npy'`
