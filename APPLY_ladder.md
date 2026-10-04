# Ladder kit: the degradation figure

Three steps: build the ladder strings from your finished DFT, extract
activations from Qwen3-8B, and fit probes that are scored on every rung.

It needs the Step 2b kit already unpacked. Unzip from the folder that contains
your checkout:

```bash
cd /local/scratch/mv487/qm9s
unzip -o ~/probe2circuit_ladder.zip
cd probe2circuit
pytest -q tests/test_ladder.py      # 4 passed (needs torch + transformers; skipped without them)
```

## The ladder

Each rung changes one thing. R2 to R9 are the same 20 amino acids, so a probe's
score is followed analyte by analyte from simulation to SERS.

| rung | spectra | what changes |
|---|---|---|
| R0 | QM9S, held-out molecules | where the probe is trained |
| R2 | amino acids, neutral, gas (your DFT) | the molecules leave the QM9 domain |
| R3 | + water (implicit) | solvent |
| R4 | + pH 7 species | protonation (zwitterion) |
| R5 | R4 as Raman intensity at 785 nm | activity → intensity |
| R6 | R4 with the gap field along z | plasmonic gap, random orientation |
| R8 | SERS, pure standards | substrate, surface binding |
| R9 | SERS, cell supernatant | biological matrix |

- R1 (the 9 calibration molecules, also in QM9S) sets the frequency scale and
  checks that the same molecule gives the same probe score from QM9S and from
  your DFT.
- R7 (a fitted orientation in the gap) needs a per-molecule fit and is left for
  the mechanism step.
- Labels always come from the neutral parent molecule, as for SERS.
- Everything uses the `matched` strings, so linewidth and intensity scale are
  the same in every rung.

## Run

**CPU part** (your usual environment, minutes):

```bash
STAGE=cpu bash scripts/run_ladder.sh
```

1. Converts the ORCA outputs under `/local/scratch/mv487/dft/<set>/` to stick
   files. The sets are `calibration_gas`, `neutral_gas`, `neutral_water` and
   `zwitterion_water`. A missing folder is reported and that rung is left out.
   If your neutral gas-phase runs sit under another name, symlink them:
   `ln -s /local/scratch/mv487/dft/AAs /local/scratch/mv487/dft/neutral_gas`.
2. Runs the calibration (`reports/ladder/calibration.txt`): the DFT frequency
   scale, and whether QM9S intensities match raw activity or 785 nm intensity.
   The ladder uses both automatically.
3. Builds `data/ladder/matched/` and a similarity table: how close each
   simulated rung is to the SERS standards and cells, with no model involved.
4. Fits the classical baseline (the binned string) and scores it on every rung:
   `reports/ladder/matched/baseline/degradation.png`. This is the comparison
   curve, and it is available straight away.

**GPU part** (an environment with torch and transformers; RDKit is not needed):

```bash
STAGE=gpu MODEL=Qwen/Qwen3-8B bash scripts/run_ladder.sh
```

- Use `MODEL=/path/to/Qwen3-8B` if the node has no internet.
- It extracts activations for the trained model and for the same architecture
  with untrained weights (a control), then fits the probes.
- Smoke test first: `python scripts/extract_activations.py --model $MODEL --limit 64`.
- Disk: about 9 GB per model for the default probe set, so about 18 GB with the
  control. `LAYER_STEP=2` halves it.
- Out of memory is handled by halving the batch size automatically.
- I have not timed this on your hardware. The whole pipeline has only been run
  on a tiny test model here.

## What you get (`reports/ladder/matched/<model>/`)

- `degradation.png`: the figure. One panel per chemical group plus the mean.
  Lines: logistic-regression probe (with a 95 % band over analytes), mass-mean
  probe, the classical baseline, and the untrained-model control.
- `degradation_centred.png`: the same after each rung is centred on its own
  mean, a label-free recalibration.
- `ladder.md`, `degradation.csv`: the same numbers as a table.
- `layers.png`, `layers.csv`: AUROC against layer, for QM9S and the SERS rungs.
  The most robust layer may not be the best in-domain layer.
- `per_analyte.csv`: each analyte's score and rank on each rung.
- `run.json`: settings, the chosen layers, and the controls.

## How it is kept honest

- Probes are fitted on QM9S train. The regularisation and the layer are chosen
  on a QM9S validation split. QM9S test and every rung are only scored; nothing
  about SERS or the DFT is used to choose anything.
- A label is used when QM9S has at least 30 training positives and the 20 amino
  acids contain both classes. Rare groups are topped up to 300 QM9S positives
  where they exist.
- With one positive amino acid (phenol, imidazole, pyrrole), AUROC is that
  analyte's rank among the 20. The panel title gives the count.
- The activation is taken once per prompt, with no answer appended, at two
  positions: the last prompt token, and the mean over the peak-list tokens.
  `meta.json` holds one exact prompt as it was fed to the model.
- Controls: the untrained model, shuffled labels at the chosen layer, and the
  classical baseline through the identical procedure.

## Before the probe run

Write the predictions down first (`PREDICTIONS.md`): which rung you expect each
group to break at, and why. For example: primary amine at R4 (it becomes NH3+),
aromatic and indole rings surviving to R9, carbonyl-dependent groups weak
throughout because of the 1750 cm⁻¹ window edge. The figure then scores them.
