# Coverage report (v1.2)
QM9 reference: 129817 SMILES from `data/qm9s_number_smiles.csv`.

Usable QM9S strings (>= 3 peaks in 500-1750 cm-1): 129770 of 129817 molecules.

## 1. Chemical coverage

11/28 analytes inside QM9's domain (<=9 heavy atoms, C/N/O/F, no isotopes); 2 found verbatim in the reference set.

| analyte | n_heavy | elements | in_qm9_domain | exact_in_qm9 | nn_tanimoto | nn_smiles | n_qm9_tanimoto_ge_0.5 | why_out |
|---|---|---|---|---|---|---|---|---|
| L-alanine | 6 | CNO | 1 | 0 | 0.474 | COC(=O)C(C)N | 0 |  |
| L-arginine | 12 | CNO | 0 | 0 | 0.353 | NC(=N)C(=O)NCCO | 0 | 12 heavy atoms > 9 |
| L-asparagine | 9 | CNO | 1 | 0 | 0.65 | NC(CC(N)=O)C(N)=O | 1 |  |
| L-aspartic acid | 9 | CNO | 1 | 0 | 0.391 | NC(CC(N)=O)C(N)=O | 0 |  |
| L-cysteine | 7 | CNOS | 0 | 0 | 0.333 | NC(CC(N)=O)C(N)=O | 0 | contains S |
| L-glutamic acid | 10 | CNO | 0 | 0 | 0.31 | NC(=O)CCC(O)C#C | 0 | 10 heavy atoms > 9 |
| L-glutamine | 10 | CNO | 0 | 0 | 0.5 | CC(C)CCC(N)=O | 3 | 10 heavy atoms > 9 |
| L-glycine | 5 | CNO | 1 | 0 | 0.438 | NCC(=O)NC(=O)CN | 0 |  |
| L-histidine | 11 | CNO | 0 | 0 | 0.529 | NC(=O)CC1=CNC=N1 | 3 | 11 heavy atoms > 9 |
| L-isoleucine | 9 | CNO | 1 | 0 | 0.48 | CCC(C)C(C)C(N)=O | 0 |  |
| L-leucine | 9 | CNO | 1 | 0 | 0.458 | CC(C)CC(C)C(N)=O | 0 |  |
| L-lysine | 10 | CNO | 0 | 0 | 0.286 | NC(CC(N)=O)C(N)=O | 0 | 10 heavy atoms > 9 |
| L-methionine | 9 | CNOS | 0 | 0 | 0.312 | CCCC(N)C(=O)OC | 0 | contains S |
| L-phenylalanine | 12 | CNO | 0 | 0 | 0.385 | OCC1=CC=CC=C1 | 0 | 12 heavy atoms > 9 |
| L-proline | 8 | CNO | 1 | 0 | 0.556 | O=C(C#N)C1CCCN1 | 1 |  |
| L-serine | 7 | CNO | 1 | 0 | 0.478 | COC(=O)C(N)CO | 0 |  |
| L-threonine | 8 | CNO | 1 | 0 | 0.522 | COC(=O)C(N)C(C)O | 1 |  |
| L-tryptophan | 15 | CNO | 0 | 0 | 0.263 | NC(=O)C1=CC=CC=C1 | 0 | 15 heavy atoms > 9 |
| L-tyrosine | 13 | CNO | 0 | 0 | 0.393 | OCC1=CC=C(O)C=C1 | 0 | 13 heavy atoms > 9 |
| L-valine | 8 | CNO | 1 | 0 | 0.381 | CC(C)C(C)C(N)=O | 0 |  |
| indole | 9 | CN | 1 | 1 | 1.0 | N1C=CC2=C1C=CC=C2 | 1 |  |
| indole-d6 | 9 | CN | 0 | 1 | 1.0 | N1C=CC2=C1C=CC=C2 | 1 | isotopically labelled |
| oxindole | 10 | CNO | 0 | 0 | 0.433 | O=C1CC2=C(N1)C=CN2 | 0 | 10 heavy atoms > 9 |
| indole-3-acetic acid | 13 | CNO | 0 | 0 | 0.294 | OCC1=CC=CC=C1O | 0 | 13 heavy atoms > 9 |
| hypoxanthine | 10 | CNO | 0 | 0 | 0.462 | NC1=C(N)C(=O)NC=N1 | 0 | 10 heavy atoms > 9 |
| 8-hydroxyquinoline | 11 | CNO | 0 | 0 | 0.52 | OC1=CC=CN=C1O | 1 | 11 heavy atoms > 9 |
| isatin | 11 | CNO | 0 | 0 | 0.32 | O=C1NC(=O)C=CC=C1 | 0 | 11 heavy atoms > 9 |
| tryptophol | 12 | CNO | 0 | 0 | 0.438 | OCCC1=C(O)NC=C1 | 0 | 12 heavy atoms > 9 |

## 2. Label support in QM9

A probe trained on QM9S cannot learn a label that QM9 does not contain. Status is judged on `n_qm9s_usable` when QM9S records are supplied (molecules whose window string has enough peaks to learn from). `NO QM9 SUPPORT` labels must be dropped from any QM9S->SERS transfer claim.

| label | n_qm9 | frac_qm9 | n_qm9s_usable | n_analytes | analytes | status |
|---|---|---|---|---|---|---|
| carboxylic_acid | 0 | 0.0 | 0 | 21 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole-3-acetic acid | NO QM9 SUPPORT |
| primary_amine | 2538 | 0.0196 | 2537 | 19 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine | ok |
| secondary_amine | 13423 | 0.1034 | 13423 | 1 | L-proline | ok |
| amide | 8862 | 0.0683 | 8861 | 4 | L-asparagine, L-glutamine, oxindole, isatin | ok |
| guanidine | 246 | 0.0019 | 246 | 1 | L-arginine | ok |
| hydroxyl_aliphatic | 37722 | 0.2906 | 37721 | 3 | L-serine, L-threonine, tryptophol | ok |
| phenol | 3762 | 0.029 | 3760 | 2 | L-tyrosine, 8-hydroxyquinoline | ok |
| thiol | 0 | 0.0 | 0 | 1 | L-cysteine | NO QM9 SUPPORT |
| thioether | 0 | 0.0 | 0 | 1 | L-methionine | NO QM9 SUPPORT |
| ketone | 14706 | 0.1133 | 14701 | 1 | isatin | ok |
| aldehyde | 13488 | 0.1039 | 13479 | 0 |  | unused |
| ester | 4561 | 0.0351 | 4561 | 0 |  | unused |
| ether | 51428 | 0.3962 | 51424 | 0 |  | unused |
| nitrile | 16372 | 0.1261 | 16356 | 0 |  | unused |
| alkyne | 16985 | 0.1308 | 16962 | 0 |  | unused |
| alkene | 16461 | 0.1268 | 16455 | 0 |  | unused |
| fluoro | 2080 | 0.016 | 2080 | 0 |  | unused |
| aromatic_ring | 22186 | 0.1709 | 22167 | 12 | L-histidine, L-phenylalanine, L-tryptophan, L-tyrosine, indole, indole-d6, oxindole, indole-3-acetic acid, hypoxanthine, 8-hydroxyquinoline, isatin, tryptophol | ok |
| benzene_ring | 273 | 0.0021 | 273 | 10 | L-phenylalanine, L-tryptophan, L-tyrosine, indole, indole-d6, oxindole, indole-3-acetic acid, 8-hydroxyquinoline, isatin, tryptophol | ok |
| pyrrole_ring | 2343 | 0.018 | 2343 | 5 | L-tryptophan, indole, indole-d6, indole-3-acetic acid, tryptophol | ok |
| imidazole_ring | 1378 | 0.0106 | 1378 | 2 | L-histidine, hypoxanthine | ok |
| indole | 1 | 0.0 | 1 | 5 | L-tryptophan, indole, indole-d6, indole-3-acetic acid, tryptophol | thin |
| lactam | 5860 | 0.0451 | 5860 | 2 | oxindole, isatin | ok |

## 3. Spectral shape by source

Token counts are estimates (1 token per digit); run leakage checks with `--tokenizer` on Aura for exact numbers.

| source | kind | n_spectra | peaks_median | peaks_range | fwhm_median | fwhm_iqr | tokens_median_est | tokens_max_est |
|---|---|---|---|---|---|---|---|---|
| qm9s | sim | 129817 | 19.0 | 1-32 | 10.0 | 8-10 | 268 | 440 |
| sers | cell | 590 | 15.0 | 8-22 | 18.0 | 12-26 | 216 | 314 |
| sers | standard | 29 | 11.0 | 7-24 | 17.0 | 12-27 | 156 | 338 |
| sers | substrate_only | 250 | 11.0 | 7-16 | 24.0 | 16-31 | 158 | 230 |

Peak-position histogram overlap (25 cm-1 bins, 1 = identical distributions):

- qm9s/sim vs sers/cell: 0.592
- qm9s/sim vs sers/standard: 0.677
- qm9s/sim vs sers/substrate_only: 0.401
- sers/cell vs sers/standard: 0.729
- sers/cell vs sers/substrate_only: 0.665
- sers/standard vs sers/substrate_only: 0.604

## 4. Peaks shared across different analytes

Bands present in spectra of at least half of the SERS analytes. A flag, not a verdict: shared chemistry (e.g. ring breathing near 1000 cm-1 in Phe/Trp/Tyr) also shows up here. Check each against the CB-only spectra.

| position | n_analytes | frac_analytes | analytes |
|---|---|---|---|
| 616 | 20 | 0.74 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tyrosine, L-valine, indole |
| 669 | 23 | 0.85 | 8-hydroxyquinoline, L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole, indole-3-acetic acid |
| 755 | 14 | 0.52 | L-alanine, L-arginine, L-asparagine, L-glutamic acid, L-glycine, L-isoleucine, L-leucine, L-serine, L-tryptophan, L-tyrosine, L-valine, indole, isatin, oxindole |
| 786 | 22 | 0.81 | 8-hydroxyquinoline, L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tyrosine, L-valine, indole, indole-3-acetic acid |
| 885 | 17 | 0.63 | L-alanine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-valine |
| 923 | 15 | 0.56 | L-arginine, L-aspartic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-threonine, L-tryptophan, L-valine, indole |
| 963 | 22 | 0.81 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole, isatin |
| 1000 | 18 | 0.67 | L-alanine, L-arginine, L-asparagine, L-glutamic acid, L-glutamine, L-glycine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole, tryptophol |
| 1116 | 21 | 0.78 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole |
| 1202 | 18 | 0.67 | L-alanine, L-arginine, L-asparagine, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-isoleucine, L-leucine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole, indole-d6, oxindole |
| 1329 | 21 | 0.78 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tyrosine, L-valine, indole, indole-d6 |
| 1375 | 22 | 0.81 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tyrosine, L-valine, indole, indole-3-acetic acid, isatin |
| 1411 | 23 | 0.85 | 8-hydroxyquinoline, L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole, indole-3-acetic acid |
| 1476 | 17 | 0.63 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-glutamic acid, L-glutamine, L-glycine, L-isoleucine, L-leucine, L-lysine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, isatin |
| 1598 | 23 | 0.85 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole, indole-3-acetic acid, isatin, tryptophol |
| 1699 | 21 | 0.78 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole-d6 |
