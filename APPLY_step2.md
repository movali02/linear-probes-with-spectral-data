# Step 2: classical baselines

Unzip from the folder that contains your checkout. Every path in the zip starts
with `probe2circuit/`, so the files land on top of it. Your data,
`data/analytes.csv` and `data/processed/` are not touched.

```bash
cd /local/scratch/mv487/qm9s              # the parent of probe2circuit/
unzip -o ~/probe2circuit_step2.zip
cd probe2circuit
pip install -e .                  # adds matplotlib (figures); everything else is already there
pytest -q                         # 81 passed, ~40 s
```

## What's new

| file | what |
|---|---|
| `src/probe2circuit/baselines.py` | features F1/F1pos/F2/F3, models, nested grouped CV, metrics, grouped bootstrap, group permutations, scaffold split |
| `src/probe2circuit/baseline_tasks.py` | T1–T4 and the T1 confound checks |
| `scripts/run_baselines.py` | `--task T1..T4 --variant native\|matched` → `reports/baselines/<variant>/` |
| `scripts/run_step2.sh` | everything below in one go |
| `configs/string_v1.yaml` | new `variants:` block (native / matched, roadmap 1.4) |
| `io.py`, `pipeline.py`, `build_strings.py` | `--variant`, `--save_spectra` (F3), `--spectra_frac` |
| `tests/test_baselines.py`, `tests/synth_step2.py` | 14 tests; synthetic data with your real folder layout |
| `tests/test_labels_coverage.py` | one assertion changed: isatin now has a ketone, so "unused" is checked on alkyne |

## Run

Copy the old fine-tune files first if you want the 58/174 comparison:
`data/split_manifest.csv` and the old predictions CSV (`file, …, Result`).

```bash
N_JOBS=32 OLD_MANIFEST=data/split_manifest.csv OLD_PREDICTIONS=data/test_predictions.csv \
  nohup bash scripts/run_step2.sh > step2.log 2>&1 &
```

The script:

1. Rebuilds `data/processed/` with `--save_spectra`. The strings are identical
   to step 1 (the build is deterministic); what's new is `sers_spectra.npz` and
   `qm9s_spectra.npz` (the 20 % QM9S subsample) for F3.
2. Builds the matched variant into `data/processed_matched/`: QM9S broadened by
   a 13 cm⁻¹ Gaussian (≈ 19 cm⁻¹ total, the SERS width), and SERS rescaled so
   the strongest analyte peak is 1.0. Schema `v1.2-matched`.
3. Runs T1, T2, T3, T4 on native, and T4 again on matched.

Rerun one task without rebuilding: `SKIP_BUILD=1 TASKS=T1 bash scripts/run_step2.sh`.
Check the plumbing first with `python scripts/run_baselines.py --task T1 --quick`
(about a minute).

**Runtime (rough estimate).** It's CPU only. Gradient boosting on the 1,251-point features (F2,
F3) is the slow part. T1 and T2 take tens of minutes on 32 cores. T3 and T4
take about an hour each. Permutation tests are capped by `--perm_budget_min`
(default 120 min per task). A slow combination gets fewer shuffles, never fewer
than 100, and the number actually used is in the `n_perm` column.

## Tasks

| task | data | label | split | bootstrap unit |
|---|---|---|---|---|
| T1 `loso` | cells | analyte (20) | train BW → test 536, and the reverse | culture |
| T1 `step1_split` | cells | analyte | the split in `sers_records.jsonl` (the LLM's) | culture |
| T1 `cells_to_standards` | train cells, test standards | analyte | – | standard |
| T2 `loao` | cells + standards | each fg_* label | leave one analyte out | analyte |
| T3 `random` / `scaffold` | QM9S (20 %) | each fg_* label | step-1 molecule split / Bemis–Murcko | molecule / scaffold |
| T4 `qm9s_to_sers` | train QM9S, test SERS | shared fg_* labels | domain | analyte (SERS side) |

T2 merges analytes by constitution, so indole and indole-d6 are held out
together. A label is tested only when at least 2 analytes have it and at least 2
don't. `T2_labels.csv` lists every label with the reason if it was skipped. With
8-hydroxyquinoline added, phenol becomes testable (Tyr + 8-HQ).

T4 picks the threshold on QM9S only. It reports QM9S held-out AUROC next to
SERS (all, cells, standards), plus ECE. `T4_labels.csv` flags labels where one
side has only one SERS analyte. Carboxylic acid, thiol and thioether drop out
because QM9S has no support for them.

**Features.** F1 is the string binned into 10 cm⁻¹ bins. F1pos is F1 with
positions only. F2 redraws each peak as a Lorentzian with its FWHM. F3 is the
spectrum before peak picking. For native SERS, F3 is on the 829 = 1.0 scale
with the CB bands still in, exactly what the peak picker sees. For matched, it
is the spectrum minus its CB reference, rescaled to max 1 like QM9S.

**Models.** Majority; logistic regression (standardised, L2, balanced, C by
inner grouped CV); HistGradientBoosting (balanced, `min_samples_leaf=5`,
because T1 trains on 10–20 scans per class); kNN (cosine). The binary threshold
comes from inner grouped CV on the training fold.

## Outputs (`reports/baselines/<variant>/`)

- `<task>.md`: headline table and, for T1, the **gate** and the confound table. Read this first.
- `<task>.csv`: every scheme × test set × label × feature × model × metric, with
  value, 95 % CI, chance, permutation p, n and hyperparameters.
- `<task>.png`: the figure.
- `T1_confounds.csv`, `T1_per_class.csv`, `T1_confusion_*.csv`, `T2/T4_per_analyte.csv`, `*_labels.csv`.
- `<task>_predictions.csv.gz`: out-of-fold predictions of LR-F1, the best model and the best F3 model.
  Step 3 uses them for the paired LLM-vs-baseline bootstrap.
- `<task>_run.json`: arguments, SHA-256 of every input, versions, runtime.

## Things to know when reading the numbers

1. **T1 LOSO has no hyperparameter search.** Each strain has one culture per
   amino acid, so the training fold can't be split by culture with every class
   on both sides. LR uses C = 0.1 and kNN k = 5 (`hp_source = default`). To
   check sensitivity, rerun with `--lr_C 1` or `--lr_C 0.01`; don't pick the best.
2. **The step-1 split is uneven for T1.** Each amino acid has three groups:
   the BW culture, the 536 culture and the standard. The step-1 split puts one
   of the three in test. When that is the standard, both cultures of that amino
   acid are in training and none in test. The log prints how many analytes the
   test cells cover, and the `step1_split` note records it. LOSO is the primary
   T1 number. For step 3 I'd train the LLM on the LOSO folds too, so the two are
   compared like for like.
3. **Metadata-only.** Under LOSO, strain is constant within each fold, so the
   design metadata (strain, plus any `--meta_csv` columns such as date) cannot
   predict anything by construction. The informative part is the acquisition
   metadata: absolute 829 counts, 829 position and the number of CB peaks
   discarded. None of these reach the model. If they predict the analyte,
   something about how or when a spectrum was taken tracks the analyte.
4. **Substrate-only** runs two ways, cells → CB and CB → CB, with LR-F1 and the
   best T1 combination. Both must sit at chance. CB → CB on F3 is the strongest
   session test, because F3 keeps the whole CB spectrum.
5. **Permutations keep the chosen hyperparameters and thresholds fixed** at the
   real-label values instead of re-tuning for each shuffle. This is standard
   and much cheaper, and slightly anti-conservative. Labels are shuffled between
   whole groups: within strain for LOSO, within train/test for the step-1
   split, between analytes for T2.
6. **"Best"** is chosen on the test result, so treat it as optimistic. LR-F1 is
   the pre-specified comparison for the LLM.

## Gate (roadmap 2.6)

`T1.md` ends with the gate check: is the F3 LOSO CI above chance? If it isn't,
the SERS arm becomes a negative control. F3 − F1 is how much the string format
loses.

## Still open

In 9 BW conditions the scans come from two sample IDs (for example s1_AL1 and
s2_AL2). If those are separate cultures, they are currently kept as one group,
which is the conservative choice. If they are separate, BW could supply an
inner CV and a second test culture.
