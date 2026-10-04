# Step 2b: fixes from the Step 2 review

Unzip from the folder that contains your checkout (paths start with
`probe2circuit/`). Nothing in `data/` is touched.

```bash
cd /local/scratch/mv487/qm9s
unzip -o ~/probe2circuit_step2b.zip
cd probe2circuit
pytest -q                          # 84 passed, ~2 min
```

## Before you run: the control cultures

Put the no-amino-acid control spectra and their CB-only files for **both
strains** in `data/raw_sers/Cell/`, with the names you already use:

```
Cell_BW_ctrl_0  … Cell_BW_ctrl_9      CB_BW_ctrl_0  … CB_BW_ctrl_4
Cell_536_ctrl_0 … Cell_536_ctrl_9     CB_536_ctrl_0 … CB_536_ctrl_4
```

- These names are recognised as they are: a `Cell_…` file with no amino-acid
  code and `ctrl` in its name is a control, and the second part of the name is
  its strain. No pattern needs passing.
- The build will print `WARNING: no analyte matched for ['CB_536_ctrl_0', …]`.
  That is expected: controls have no analyte.
- The log line `[T1] N no-analyte control spectra: ['Cell_536_ctrl', 'Cell_BW_ctrl']`
  confirms both were found.
- Controls are never trained on or tested on. They are used in two places:
  - the `.ctrl` features (`F1.ctrl`, `F3.ctrl`): each spectrum minus the mean
    of its own strain's control culture. Unlike `.bg`, this uses no spectra of
    the cultures being classified;
  - the detectability table (`r_cross_ctrl`).
- If any KO-strain controls sit in the same folder under a `BW` name, keep them
  out: your cell spectra are wild type.

## Run

```bash
REBUILD_SERS=1 N_JOBS=32 OLD_MANIFEST=data/split_manifest.csv \
  OLD_PREDICTIONS=data/test_predictions_sanitized.csv \
  nohup bash scripts/run_step2b.sh > step2b.log 2>&1 &
```

- `REBUILD_SERS=1` rebuilds the SERS records so the new control files are
  included. It takes a minute. QM9S is not rebuilt.
- Results go to `reports/baselines2b/`, so the first run in `reports/baselines/`
  stays for comparison.
- Smoke test first (roughly 10–20 minutes): add `EXTRA_ARGS="--quick"`.
- Rough runtime on 32 cores: 4–6 hours. T2 and the two T3 runs are the long ones.

## What changed

| | first run | Step 2b |
|---|---|---|
| **T1 features** | F1, F1pos, F2, F3 | + `.ctrl` (the strain's control culture subtracted), `.bg` (each strain's mean spectrum subtracted; no labels used), `F3.d1` (first derivative), `F3.snv`, `F3.bg.d1` |
| **T1/T2 models** | majority, LR, HGB, kNN | + `ncm` (nearest class mean), `plsda` (PLS-DA), `pca_lr` (PCA then LR) |
| **T1 metrics** | per scan | + per culture (`all@culture`: majority vote, n = 40) |
| **T1 detectability** | – | `T1_detectability.csv`, also printed in `T1.md` |
| **T1 old result** | manifest split | + `old_prediction_rows`: test set = the non-CB rows of the old predictions file, with the old LLM scored on the same rows |
| **T2 metric** | pooled leave-one-analyte-out AUROC (biased low) | leave-pair-out AUROC |
| **T2 design** | cells and standards pooled | `lpo:cells`, `lpo:standards`, `lpo:all` with a sample-type-only control, and `std_to_cells` |
| **T3** | – | + `F1.sqrt`; `F2.lt1700` and `F3.lt1700`; a second run on QM9S strings built to 1800 cm⁻¹; `T3_string_loss.txt` |
| **T4** | per scan | + per analyte (`sers_*@analyte`); `.bg` features; labels with one SERS analyte moved to a rank table; benzene ring recovered (min 30 QM9S positives) |
| **Checks** | – | `cb_inspect.txt/.csv/.png`, `qm9s_frequency_check.txt` |

## How to read the new outputs

**T1 gate.** `T1.md` now lists, for leave-one-strain-out:
- F3 as in the first run;
- `F3.ctrl` with nearest centroid, `F3.bg` with nearest centroid and LR on
  `F1.bg`: these are fixed in advance for this run, so they are not picked on
  the result. `F3.ctrl` is the cleanest of the three;
- the best of each family, which is picked on the result, so read it as optimistic.

**Detectability.** For each amino acid, its signature is its culture mean minus
the strain mean. No classifier is involved.
- `r_cross`: how well the BW and 536 signatures agree.
- `rank_*`: whether an analyte's own signature is the best match in the other strain (1 = yes).
- `r_std_*`: agreement with the pure standard.
- `r_cross_ctrl`: the same as `r_cross`, with each strain's control culture as the background.
- `top_bands_cm`: the three strongest bands of the signature.

Analytes with high `r_cross` and rank 1 are the SERS-visible ones. This is the
list to pre-register as the "detectable subset" for Steps 3–5.

**T2.**
- `lpo:*` AUROC is at analyte level: one positive and one negative analyte are
  held out together, and only their two scores from the same model are
  compared. The majority model is now exactly 0.5.
- The `matrix-only` column is the same AUROC from sample type alone (BW cell,
  536 cell, standard). When it is far from 0.5, ignore `lpo:all` for that label
  and read `lpo:cells` and `lpo:standards`.
- `std_to_cells` trains only on standards of analytes that have no cell spectra
  (the indole family), then tests on cells. A hit there can't come from
  recognising the analyte.
- Permutation p-values for `lpo:*` use default hyperparameters on both the real
  and the shuffled labels.

**T3.** `reports/baselines2b/T3_string_loss.txt` gives, per label, the gap
between the spectrum and the re-drawn string, three ways: as before, cropped
below 1700 cm⁻¹, and with the window extended to 1800 cm⁻¹. If the carbonyl
gap (ester, lactam, ketone, amide) shrinks in the last two, the string loses
the C=O stretch at the window edge, not information in general.

**CB check.** `cb_inspect.txt` gives a verdict for each condition whose CB
files lacked the 829 band:
- `shifted`: the band is there but outside ±4 cm⁻¹. Raise `ref_tol`.
- `weak`: the CB signal is low in that well.
- `absent`: the file doesn't look like the other CB spectra. Open it.

`cb_inspect.png` overlays those files on the mean of the good ones.

## Reproduce the first run

`python scripts/run_baselines.py --task T1 --no_preps --models majority,lr,hgb,knn`
gives the first run's T1 numbers. T2 always uses leave-pair-out now.
