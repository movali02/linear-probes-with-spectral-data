# Shift ladder: baseline-F1pos (v2m)

AUROC of a probe fitted on QM9S, scored on each rung. R2 onwards are the same 20 amino acids, one score per analyte. Layer and C were chosen on QM9S validation only.

## As trained

| label | n pos | method | layer | R0 | R2 | R3 | R4 | R5 | R6 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MEAN |  | classifier on the string (F1pos) |  | 0.88 | 0.89 [0.83, 0.99] | 0.80 [0.67, 0.88] | 0.60 [0.47, 0.73] | 0.54 [0.37, 0.71] | 0.56 [0.42, 0.67] | 0.49 [0.31, 0.64] | 0.65 [0.49, 0.74] |
| primary_amine | 19 | classifier on the string (F1pos) | - -1 | 0.93 [0.90, 0.96] | 0.95 [0.83, 1.00] | 0.84 [0.65, 1.00] | 0.42 [0.21, 0.65] | 0.47 [0.26, 0.74] | 0.58 [0.37, 0.79] | 0.89 [0.74, 1.00] | 0.79 [0.58, 0.95] |
| secondary_amine | 1 | classifier on the string (F1pos) | - -1 | 0.70 [0.67, 0.73] | 0.32 [0.11, 0.56] | 0.89 [0.74, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.79 [0.58, 0.95] | 1.00 [1.00, 1.00] | 0.26 [0.06, 0.47] |
| amide | 2 | classifier on the string (F1pos) | - -1 | 0.85 [0.82, 0.87] | 1.00 [1.00, 1.00] | 0.67 [0.31, 1.00] | 0.92 [0.76, 1.00] | 1.00 [1.00, 1.00] | 0.97 [0.88, 1.00] | 0.35 [0.15, 0.58] | 0.64 [0.21, 1.00] |
| guanidine | 1 | classifier on the string (F1pos) | - -1 | 0.95 [0.93, 0.97] | 0.95 [0.83, 1.00] | 0.95 [0.83, 1.00] | 0.16 [0.00, 0.33] | 0.00 [0.00, 0.00] | 0.11 [0.00, 0.26] | 0.63 [0.42, 0.84] | 0.74 [0.53, 0.92] |
| hydroxyl_aliphatic | 2 | classifier on the string (F1pos) | - -1 | 0.72 [0.70, 0.74] | 0.94 [0.78, 1.00] | 0.75 [0.53, 0.94] | 0.53 [0.18, 0.84] | 0.47 [0.05, 0.89] | 0.47 [0.11, 0.79] | 0.28 [0.00, 0.63] | 0.61 [0.18, 1.00] |
| phenol | 1 | classifier on the string (F1pos) | - -1 | 0.90 [0.88, 0.92] | 0.84 [0.67, 1.00] | 0.53 [0.29, 0.74] | 0.37 [0.16, 0.61] | 0.21 [0.05, 0.39] | 0.42 [0.21, 0.67] | 0.11 [0.00, 0.26] | 0.95 [0.83, 1.00] |
| aromatic_ring | 4 | classifier on the string (F1pos) | - -1 | 0.97 [0.97, 0.98] | 0.95 [0.81, 1.00] | 0.88 [0.64, 1.00] | 0.44 [0.14, 0.74] | 0.28 [0.06, 0.53] | 0.30 [0.05, 0.61] | 0.69 [0.34, 0.95] | 0.59 [0.00, 0.95] |
| benzene_ring | 3 | classifier on the string (F1pos) | - -1 | 0.98 [0.97, 0.99] | 0.92 [0.78, 1.00] | 0.53 [0.31, 0.76] | 0.76 [0.53, 0.97] | 0.51 [0.16, 0.89] | 0.55 [0.22, 0.89] | 0.33 [0.00, 0.63] | 0.25 [0.00, 0.63] |
| pyrrole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.92 [0.88, 0.94] | 1.00 [1.00, 1.00] | 0.95 [0.83, 1.00] | 0.47 [0.26, 0.71] | 0.42 [0.22, 0.67] | 0.58 [0.37, 0.79] | 0.11 [0.00, 0.24] | 1.00 [1.00, 1.00] |
| imidazole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.88, 0.93] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.89 [0.74, 1.00] | 1.00 [1.00, 1.00] | 0.79 [0.58, 0.95] | 0.53 [0.26, 0.75] | 0.68 [0.47, 0.88] |

## Each rung centred on its own mean (label-free recalibration)

| label | n pos | method | layer | R0 | R2 | R3 | R4 | R5 | R6 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MEAN |  | classifier on the string (F1pos) |  | 0.88 | 0.89 [0.82, 0.99] | 0.80 [0.66, 0.89] | 0.60 [0.47, 0.72] | 0.54 [0.36, 0.73] | 0.56 [0.42, 0.67] | 0.49 [0.30, 0.64] | 0.64 [0.48, 0.73] |
| primary_amine | 19 | classifier on the string (F1pos) | - -1 | 0.93 [0.90, 0.96] | 0.95 [0.83, 1.00] | 0.84 [0.65, 1.00] | 0.42 [0.21, 0.65] | 0.47 [0.26, 0.74] | 0.58 [0.37, 0.79] | 0.89 [0.74, 1.00] | 0.79 [0.58, 0.95] |
| secondary_amine | 1 | classifier on the string (F1pos) | - -1 | 0.70 [0.67, 0.73] | 0.32 [0.11, 0.56] | 0.89 [0.74, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.79 [0.58, 0.95] | 1.00 [1.00, 1.00] | 0.26 [0.06, 0.47] |
| amide | 2 | classifier on the string (F1pos) | - -1 | 0.85 [0.82, 0.87] | 1.00 [1.00, 1.00] | 0.67 [0.31, 1.00] | 0.92 [0.76, 1.00] | 1.00 [1.00, 1.00] | 0.97 [0.88, 1.00] | 0.35 [0.15, 0.58] | 0.64 [0.21, 1.00] |
| guanidine | 1 | classifier on the string (F1pos) | - -1 | 0.95 [0.93, 0.97] | 0.95 [0.83, 1.00] | 0.95 [0.83, 1.00] | 0.16 [0.00, 0.33] | 0.00 [0.00, 0.00] | 0.11 [0.00, 0.26] | 0.63 [0.42, 0.84] | 0.74 [0.53, 0.92] |
| hydroxyl_aliphatic | 2 | classifier on the string (F1pos) | - -1 | 0.72 [0.70, 0.74] | 0.94 [0.78, 1.00] | 0.75 [0.53, 0.94] | 0.53 [0.18, 0.84] | 0.47 [0.05, 0.89] | 0.47 [0.11, 0.79] | 0.28 [0.00, 0.63] | 0.61 [0.18, 1.00] |
| phenol | 1 | classifier on the string (F1pos) | - -1 | 0.90 [0.88, 0.92] | 0.84 [0.67, 1.00] | 0.53 [0.29, 0.74] | 0.37 [0.16, 0.61] | 0.21 [0.05, 0.39] | 0.42 [0.21, 0.67] | 0.11 [0.00, 0.26] | 0.95 [0.83, 1.00] |
| aromatic_ring | 4 | classifier on the string (F1pos) | - -1 | 0.97 [0.97, 0.98] | 0.95 [0.81, 1.00] | 0.88 [0.64, 1.00] | 0.44 [0.14, 0.74] | 0.28 [0.06, 0.53] | 0.30 [0.05, 0.61] | 0.69 [0.34, 0.95] | 0.59 [0.00, 0.95] |
| benzene_ring | 3 | classifier on the string (F1pos) | - -1 | 0.98 [0.97, 0.99] | 0.92 [0.78, 1.00] | 0.53 [0.31, 0.76] | 0.76 [0.53, 0.97] | 0.51 [0.16, 0.89] | 0.55 [0.22, 0.89] | 0.33 [0.00, 0.63] | 0.24 [0.00, 0.61] |
| pyrrole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.92 [0.88, 0.94] | 1.00 [1.00, 1.00] | 0.95 [0.83, 1.00] | 0.47 [0.26, 0.71] | 0.42 [0.22, 0.67] | 0.58 [0.37, 0.79] | 0.11 [0.00, 0.24] | 0.95 [0.84, 1.00] |
| imidazole_ring | 1 | classifier on the string (F1pos) | - -1 | 0.91 [0.88, 0.93] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.89 [0.74, 1.00] | 1.00 [1.00, 1.00] | 0.79 [0.58, 0.95] | 0.53 [0.26, 0.75] | 0.68 [0.47, 0.88] |

## Labels with one positive analyte, as that analyte's rank

Rank 1 = the probe scores it highest of the 20 analytes. An AUROC here is only this rank.

| label | analyte | method | R2 | R3 | R4 | R5 | R6 | R8 | R9 |
|---|---|---|---|---|---|---|---|---|---|
| secondary_amine | L-proline | classifier on the string (F1pos) | 14 | 3 | 1 | 1 | 5 | 1 | 15 |
| guanidine | L-arginine | classifier on the string (F1pos) | 2 | 2 | 17 | 20 | 18 | 8 | 6 |
| phenol | L-tyrosine | classifier on the string (F1pos) | 4 | 10 | 13 | 16 | 12 | 18 | 2 |
| pyrrole_ring | L-tryptophan | classifier on the string (F1pos) | 1 | 2 | 11 | 12 | 9 | 18 | 1 |
| imidazole_ring | L-histidine | classifier on the string (F1pos) | 1 | 1 | 3 | 1 | 5 | 10 | 7 |

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

- R1_agreement_baseline_lr: 0.89
- R1_n_molecules: 9

## Spectral similarity to SERS (no model)

| rung | vs | mean cosine | own standard best match | top 3 |
|---|---|---|---|---|
| R2 | R8 | 0.5118 | 10% | 20% |
| R2 | R9 | 0.5804 | 15% | 20% |
| R3 | R8 | 0.514 | 10% | 15% |
| R3 | R9 | 0.5566 | 5% | 10% |
| R4 | R8 | 0.5495 | 5% | 10% |
| R4 | R9 | 0.6009 | 0% | 10% |
| R5 | R8 | 0.4953 | 5% | 20% |
| R5 | R9 | 0.5647 | 0% | 5% |
| R6 | R8 | 0.4468 | 10% | 25% |
| R6 | R9 | 0.5147 | 0% | 5% |
