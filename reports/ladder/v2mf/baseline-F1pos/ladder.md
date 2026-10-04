# Shift ladder: baseline-F1pos (v2mf)

AUROC of a probe fitted on QM9S, scored on each rung. R2 onwards are the same 20 amino acids, one score per analyte. Layer and C were chosen on QM9S validation only.

## As trained

| label | n pos | method | layer | R0 | R2 | R3 | R4 | R5 | R6 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MEAN |  | classifier on the string (F1pos) |  | 0.88 | 0.85 [0.78, 0.96] | 0.82 [0.72, 0.87] | 0.54 [0.44, 0.67] | 0.49 [0.35, 0.64] | 0.45 [0.32, 0.60] | 0.61 [0.38, 0.79] | 0.60 [0.38, 0.81] |
| primary_amine | 19 | classifier on the string (F1pos) | - -1 | 0.91 [0.87, 0.95] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.21 [0.05, 0.42] | 0.21 [0.05, 0.42] | 0.26 [0.11, 0.47] | 0.79 [0.58, 0.95] | 0.11 [0.00, 0.26] |
| secondary_amine | 1 | classifier on the string (F1pos) | - -1 | 0.70 [0.67, 0.72] | 0.37 [0.16, 0.61] | 0.74 [0.53, 0.94] | 0.95 [0.84, 1.00] | 0.95 [0.84, 1.00] | 0.95 [0.84, 1.00] | 1.00 [1.00, 1.00] | 0.74 [0.53, 0.94] |
| amide | 2 | classifier on the string (F1pos) | - -1 | 0.82 [0.79, 0.84] | 0.97 [0.88, 1.00] | 0.78 [0.56, 0.95] | 0.81 [0.58, 1.00] | 0.83 [0.58, 1.00] | 0.72 [0.37, 1.00] | 0.57 [0.13, 1.00] | 0.64 [0.21, 1.00] |
| guanidine | 1 | classifier on the string (F1pos) | - -1 | 0.95 [0.92, 0.97] | 1.00 [1.00, 1.00] | 0.95 [0.83, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.89 [0.74, 1.00] | 0.68 [0.47, 0.89] |
| hydroxyl_aliphatic | 2 | classifier on the string (F1pos) | - -1 | 0.72 [0.70, 0.74] | 0.92 [0.77, 1.00] | 0.53 [0.29, 0.75] | 0.56 [0.18, 0.89] | 0.47 [0.05, 0.89] | 0.47 [0.05, 0.89] | 0.22 [0.00, 0.58] | 0.56 [0.11, 1.00] |
| phenol | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.88, 0.93] | 0.53 [0.32, 0.74] | 0.89 [0.74, 1.00] | 0.58 [0.36, 0.79] | 0.53 [0.32, 0.76] | 0.53 [0.28, 0.74] | 0.47 [0.26, 0.72] | 0.89 [0.74, 1.00] |
| aromatic_ring | 4 | classifier on the string (F1pos) | - -1 | 0.97 [0.96, 0.98] | 0.95 [0.83, 1.00] | 0.84 [0.55, 1.00] | 0.44 [0.14, 0.74] | 0.33 [0.06, 0.62] | 0.28 [0.05, 0.57] | 0.56 [0.23, 0.88] | 0.56 [0.05, 1.00] |
| benzene_ring | 3 | classifier on the string (F1pos) | - -1 | 0.99 [0.98, 0.99] | 0.82 [0.62, 1.00] | 0.55 [0.21, 0.94] | 0.80 [0.61, 0.97] | 0.47 [0.11, 0.95] | 0.51 [0.21, 0.89] | 0.51 [0.12, 1.00] | 0.75 [0.47, 1.00] |
| pyrrole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.87, 0.94] | 0.95 [0.83, 1.00] | 0.95 [0.83, 1.00] | 0.42 [0.21, 0.67] | 0.26 [0.11, 0.53] | 0.32 [0.15, 0.58] | 0.32 [0.11, 0.53] | 1.00 [1.00, 1.00] |
| imidazole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.88, 0.94] | 1.00 [1.00, 1.00] | 0.95 [0.82, 1.00] | 0.63 [0.41, 0.84] | 0.89 [0.74, 1.00] | 0.47 [0.26, 0.72] | 0.79 [0.58, 0.95] | 0.11 [0.00, 0.26] |

## Each rung centred on its own mean (label-free recalibration)

| label | n pos | method | layer | R0 | R2 | R3 | R4 | R5 | R6 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MEAN |  | classifier on the string (F1pos) |  | 0.88 | 0.85 [0.78, 0.96] | 0.82 [0.72, 0.87] | 0.54 [0.43, 0.67] | 0.49 [0.33, 0.66] | 0.45 [0.32, 0.59] | 0.61 [0.41, 0.79] | 0.61 [0.37, 0.81] |
| primary_amine | 19 | classifier on the string (F1pos) | - -1 | 0.91 [0.87, 0.95] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.21 [0.05, 0.42] | 0.21 [0.05, 0.42] | 0.26 [0.11, 0.47] | 0.79 [0.58, 0.95] | 0.11 [0.00, 0.26] |
| secondary_amine | 1 | classifier on the string (F1pos) | - -1 | 0.70 [0.67, 0.72] | 0.37 [0.16, 0.61] | 0.74 [0.53, 0.94] | 0.95 [0.84, 1.00] | 0.95 [0.84, 1.00] | 0.95 [0.84, 1.00] | 1.00 [1.00, 1.00] | 0.79 [0.58, 0.95] |
| amide | 2 | classifier on the string (F1pos) | - -1 | 0.82 [0.79, 0.84] | 0.97 [0.88, 1.00] | 0.78 [0.56, 0.95] | 0.81 [0.58, 1.00] | 0.83 [0.58, 1.00] | 0.72 [0.37, 1.00] | 0.57 [0.13, 1.00] | 0.64 [0.21, 1.00] |
| guanidine | 1 | classifier on the string (F1pos) | - -1 | 0.95 [0.92, 0.97] | 1.00 [1.00, 1.00] | 0.95 [0.83, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.89 [0.74, 1.00] | 0.68 [0.47, 0.89] |
| hydroxyl_aliphatic | 2 | classifier on the string (F1pos) | - -1 | 0.72 [0.70, 0.74] | 0.92 [0.77, 1.00] | 0.53 [0.29, 0.75] | 0.56 [0.18, 0.89] | 0.47 [0.05, 0.89] | 0.47 [0.05, 0.89] | 0.22 [0.00, 0.58] | 0.56 [0.11, 1.00] |
| phenol | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.88, 0.93] | 0.53 [0.32, 0.74] | 0.89 [0.74, 1.00] | 0.58 [0.36, 0.79] | 0.53 [0.32, 0.76] | 0.53 [0.28, 0.74] | 0.47 [0.26, 0.72] | 0.89 [0.74, 1.00] |
| aromatic_ring | 4 | classifier on the string (F1pos) | - -1 | 0.97 [0.96, 0.98] | 0.95 [0.83, 1.00] | 0.84 [0.55, 1.00] | 0.44 [0.14, 0.74] | 0.33 [0.06, 0.62] | 0.28 [0.05, 0.57] | 0.56 [0.23, 0.88] | 0.56 [0.05, 1.00] |
| benzene_ring | 3 | classifier on the string (F1pos) | - -1 | 0.99 [0.98, 0.99] | 0.82 [0.62, 1.00] | 0.55 [0.21, 0.94] | 0.80 [0.61, 0.97] | 0.47 [0.11, 0.95] | 0.51 [0.21, 0.89] | 0.51 [0.12, 1.00] | 0.76 [0.50, 1.00] |
| pyrrole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.87, 0.94] | 0.95 [0.83, 1.00] | 0.95 [0.83, 1.00] | 0.42 [0.21, 0.67] | 0.26 [0.11, 0.53] | 0.32 [0.15, 0.58] | 0.32 [0.11, 0.53] | 1.00 [1.00, 1.00] |
| imidazole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.88, 0.94] | 1.00 [1.00, 1.00] | 0.95 [0.82, 1.00] | 0.63 [0.41, 0.84] | 0.89 [0.74, 1.00] | 0.47 [0.26, 0.72] | 0.79 [0.58, 0.95] | 0.11 [0.00, 0.26] |

## Labels with one positive analyte, as that analyte's rank

Rank 1 = the probe scores it highest of the 20 analytes. An AUROC here is only this rank.

| label | analyte | method | R2 | R3 | R4 | R5 | R6 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|
| secondary_amine | L-proline | classifier on the string (F1pos) | 13 | 6 | 2 | 2 | 2 | 1 | 6 |
| guanidine | L-arginine | classifier on the string (F1pos) | 1 | 2 | 20 | 20 | 20 | 3 | 7 |
| phenol | L-tyrosine | classifier on the string (F1pos) | 10 | 3 | 9 | 10 | 10 | 11 | 3 |
| pyrrole_ring | L-tryptophan | classifier on the string (F1pos) | 2 | 2 | 12 | 15 | 14 | 14 | 1 |
| imidazole_ring | L-histidine | classifier on the string (F1pos) | 1 | 2 | 8 | 3 | 11 | 5 | 18 |

## Rungs

- **R0**: QM9S (held out)
- **R2**: amino acids: neutral, gas
- **R3**: + water (implicit)
- **R4**: + pH 7 protonation
- **R5**: + 785 nm intensities
- **R6**: + gap field along z
- **R8**: SERS standards
- **R9**: SERS cells

## Controls

- R1_agreement_baseline_lr: 0.897
- R1_n_molecules: 9

## Spectral similarity to SERS (no model)

| rung | vs | mean cosine | own standard best match | top 3 |
|---|---|---|---|---|
| R2 | R8 | 0.4904 | 15% | 35% |
| R2 | R9 | 0.5728 | 10% | 10% |
| R3 | R8 | 0.5275 | 20% | 40% |
| R3 | R9 | 0.5659 | 15% | 35% |
| R4 | R8 | 0.5773 | 15% | 30% |
| R4 | R9 | 0.6329 | 10% | 20% |
| R5 | R8 | 0.5221 | 10% | 20% |
| R5 | R9 | 0.5936 | 10% | 15% |
| R6 | R8 | 0.4759 | 10% | 20% |
| R6 | R9 | 0.5423 | 10% | 15% |
