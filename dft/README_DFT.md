# DFT kit: the ladder from QM9S to SERS

Same method as your `batch.py`: B3LYP/6-311+G(d,p), `Opt NumFreq`,
`%elprop Polar 1` for Raman. CPCM(Water) is used where the set name ends in
`_water`. Every geometry is validated before ORCA sees it, and every result is
checked before its spectrum is used.

## Files

| file | job |
|---|---|
| `molecules.csv` | all sets, with SMILES, charge and expected formula (checked with RDKit) |
| `dft_batch.py` | geometry → validation → ORCA input → run → check, per set; resumable |
| `dft_check.py` | normal termination, converged, no imaginary modes, Raman present, solvent, **final geometry still has the intended protonation, tautomer and diastereomer** |
| `dft_to_sticks.py` | stick files in 3 intensity conventions (`activity`, `int785`, `gapiso785`) + per-mode table |
| `calibrate_vs_qm9s.py` | frequency scale and intensity convention, from the 9 QM9S molecules |
| `gapmode.py` | nanogap orientation rung from the `.hess` polarisability derivatives |
| `dft_fix_imag.py` | repairs jobs that ended with imaginary modes: displace along the mode, `TightOpt`, recheck |

## Sets

| set | n | what | charge |
|---|---|---|---|
| `calibration_gas` | 9 | benzene, toluene, phenol, pyrrole, imidazole, 4-methylimidazole, indole, ethanol, acetamide (all in QM9S) | 0 |
| `neutral_gas` | 20 | amino acids, neutral, gas phase | 0 |
| `neutral_water` | 20 | amino acids, neutral, CPCM water | 0 |
| `zwitterion_water` | 20 | pH 7 species: NH3+ / COO−; Asp, Glu −1; Lys, Arg +1; His neutral (Nτ-H) | −1…+1 |
| `pI_water` | 4 | Asp, Glu, Lys, Arg net-neutral species, closer to the standards dissolved in water | 0 |

## Before running: check your existing neutral runs

`batch.py` writes `! CPCM(Water)` into every input, so the runs you think of as
"neutral gas phase" may be neutral **in water**. Its SMILES also carry no
stereo, so Thr and Ile may be the allo diastereomers, and its histidine is the
Nπ-H tautomer. One command tells you all three:

```bash
python dft/dft_check.py --root /local/scratch/mv487/dft/AAs --set neutral_gas
```

Read the `solvent` column in `check.csv`:
- **water** → these are `neutral_water`. Rename the folder to `/local/scratch/mv487/dft/neutral_water`, then run `neutral_gas`.
- **gas** → these are `neutral_gas`. Rename accordingly, then run `neutral_water`.

Any FAIL for Thr, Ile or His (wrong diastereomer or tautomer) means rerunning
that molecule with `--only threonine,isoleucine,histidine`.

## Run order

On aura, in tmux. Do a dry run first (inputs only, about a minute), then the real run.

```bash
tmux new -s dft
conda activate mv_env
cd ~/probe2circuit

python dft/dft_batch.py --set calibration_gas --dry-run
python dft/dft_batch.py --set calibration_gas          # small molecules: ~2-4 h in total
python dft/dft_batch.py --set zwitterion_water         # the main new set
python dft/dft_batch.py --set neutral_gas              # or neutral_water: whichever the check says is missing
python dft/dft_batch.py --set pI_water                 # optional, 4 jobs
```

Two sets can run side by side on the 32-core node with `--nprocs 16` each, in
two tmux windows. Your hypoxanthine run (14 atoms, 16 cores) took 10–15 min.
NumFreq cost grows steeply with size, so expect roughly 1–3 h for the largest
(Trp, Arg, Lys: 25–27 atoms) and something like a day per 20-molecule set.
Jobs run smallest first, and a rerun skips anything already finished.

Outputs land in `/local/scratch/mv487/dft/<set>/<name>/`:
- `<name>_init.xyz`: the pre-DFT geometry
- `<name>.out` and `<name>.hess`
- `smiles.txt`
- `summary.csv` for the set

## After each set

```bash
python dft/dft_check.py --root /local/scratch/mv487/dft/zwitterion_water
python dft/dft_to_sticks.py --root /local/scratch/mv487/dft/zwitterion_water \
    --out_dir data/dft_sticks/zwitterion_water
for h in /local/scratch/mv487/dft/zwitterion_water/*/*.hess; do
  n=$(basename "$h" .hess); python dft/gapmode.py --hess "$h" --out data/gapmode/zwitterion_water/$n
done
```

Watch for this failure in the check: a zwitterion whose NH3+ handed its proton
back to COO− during optimisation. It shows as `structure changed: H on N:
intended [..3..] found [..2..]`. The CPCM solvent and the solvent-damped
starting conformer make it rare. If it happens, rerun that molecule. If it
happens again, start from a conformer without the NH3+···−OOC hydrogen bond.

## Jobs that end with an imaginary mode

`dft_check.py` fails any job with an imaginary frequency. That is a saddle
point, not a minimum. With CPCM and default Opt convergence this usually means
the optimiser stopped on a flat rotor: NH3+, CH3, COO− or a side-chain torsion.

```bash
python dft/dft_fix_imag.py --root /local/scratch/mv487/dft/zwitterion_water          # report: frequency + which atoms move
python dft/dft_fix_imag.py --root /local/scratch/mv487/dft/zwitterion_water --fix --dry-run
python dft/dft_fix_imag.py --root /local/scratch/mv487/dft/zwitterion_water --fix    # in tmux
```

`--fix` does the following for each affected job:
- moves the old files to `<name>/attempt<k>_imag/`
- displaces the geometry 0.1 Å along the imaginary mode
- reruns with `TightOpt` (level 1), then with `TightOpt DefGrid3` (level 2) if the mode is still there
- rechecks the result

The fixed run keeps the name `<name>.out`, so the steps after it need no change.
Results go to `fix_summary.csv`, including which level and grid each fix needed.
To split the work, run two windows with `--only a,b,c --nprocs 16`.

## Calibration (do this once the QM9S join is fixed)

```bash
python dft/dft_to_sticks.py --root /local/scratch/mv487/dft/calibration_gas \
    --out_dir data/dft_sticks/calibration_gas
python dft/calibrate_vs_qm9s.py --qm9s_records data/processed/qm9s_records.jsonl \
    --sticks data/dft_sticks/calibration_gas
```

- Put the reported scale in `configs/string_v1.yaml` under `dft.scale`.
- Use whichever stick variant (`activity` or `int785`) matches QM9S for the neutral DFT rungs, so QM9S and your DFT differ only in the molecules.

## The ladder

Each rung changes one thing. Build strings for every rung with the same
pipeline and config, then compare each rung with the SERS spectra: direct
spectral similarity, and probe transfer (step 5). The rung where agreement
drops most is the answer to "why does simulated not transfer to real".

| rung | spectra | what changes from the rung above | new calcs? |
|---|---|---|---|
| R0 | QM9S | (starting point: QM9 molecules) | no |
| R1 | `calibration_gas` vs QM9S | nothing but method and basis: sets scale and convention | 9 |
| R2 | `neutral_gas` | the molecules themselves (amino acids, outside QM9) | 20 |
| R3 | `neutral_water` | implicit solvent | 20 |
| R4 | `zwitterion_water` | protonation state (pH 7) | 20 |
| R5 | R4 as `int785` | activity → Raman intensity at 785 nm | no |
| R6 | R4 as `gapiso785` | gap field along z, random orientation | no |
| R7 | R4 + best orientation (`gapmode.py`) | fixed orientation in the gap | no |
| R8 | SERS standards | substrate, chemical enhancement, anharmonicity | — |
| R9 | SERS cells | biological matrix and medium | — |

Notes on R7:
- The orientation has 3 degrees of freedom, so `gapmode.py` fits on one half of the window and scores the other half. It also reports where the best orientation sits among random ones.
- Treat an orientation as real only if it wins on held-out bands. Better still, fit on standards and test on cells.
- Your anisotropy ranking (`2.rank_anisotropy.py`) picks the molecules where orientation effects are large enough to detect.
