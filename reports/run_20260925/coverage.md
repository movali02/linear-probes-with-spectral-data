# Coverage report (v1)
Run on 25 Sep 2026: the 7 SERS files you sent, and your qm9s_raman.jsonl (125,996 molecules) as the QM9 reference and QM9S spectra. QM9S strings come from peak lists that were normalised over 400-4000 cm-1 and cut at 0.1, so they are missing weak fingerprint peaks (see effective_floor).
QM9 reference: 125996 SMILES from `/mnt/user-data/uploads/qm9s_raman.jsonl`.

Usable QM9S strings (>= 3 peaks in 500-1750 cm-1): 15473 of 125996 molecules.

## 1. Chemical coverage

11/25 analytes inside QM9's domain (<=9 heavy atoms, C/N/O/F, no isotopes); 2 found verbatim in the reference set.

| analyte | n_heavy | elements | in_qm9_domain | exact_in_qm9 | nn_tanimoto | nn_smiles | n_qm9_tanimoto_ge_0.5 | why_out |
|---|---|---|---|---|---|---|---|---|
| L-alanine | 6 | CNO | 1 | 0 | 0.474 | COC(=O)C(C)N | 0 |  |
| L-arginine | 12 | CNO | 0 | 0 | 0.353 | N=C(N)C(=O)NCCO | 0 | 12 heavy atoms > 9 |
| L-asparagine | 9 | CNO | 1 | 0 | 0.65 | NC(=O)CC(N)C(N)=O | 1 |  |
| L-aspartic acid | 9 | CNO | 1 | 0 | 0.391 | NC(=O)CC(N)C(N)=O | 0 |  |
| L-cysteine | 7 | CNOS | 0 | 0 | 0.333 | NC(=O)CC(N)C(N)=O | 0 | contains S |
| L-glutamic acid | 10 | CNO | 0 | 0 | 0.31 | C#CC(O)CCC(N)=O | 0 | 10 heavy atoms > 9 |
| L-glutamine | 10 | CNO | 0 | 0 | 0.5 | CC(C)CCC(N)=O | 3 | 10 heavy atoms > 9 |
| L-glycine | 5 | CNO | 1 | 0 | 0.438 | NCC(=O)NC(=O)CN | 0 |  |
| L-histidine | 11 | CNO | 0 | 0 | 0.529 | NC(=O)Cc1c[nH]cn1 | 3 | 11 heavy atoms > 9 |
| L-isoleucine | 9 | CNO | 1 | 0 | 0.48 | CCC(C)C(C)C(N)=O | 0 |  |
| L-leucine | 9 | CNO | 1 | 0 | 0.458 | CC(C)CC(C)C(N)=O | 0 |  |
| L-lysine | 10 | CNO | 0 | 0 | 0.286 | NC(=O)CC(N)C(N)=O | 0 | 10 heavy atoms > 9 |
| L-methionine | 9 | CNOS | 0 | 0 | 0.312 | CCCC(N)C(=O)OC | 0 | contains S |
| L-phenylalanine | 12 | CNO | 0 | 0 | 0.385 | OCc1ccccc1 | 0 | 12 heavy atoms > 9 |
| L-proline | 8 | CNO | 1 | 0 | 0.556 | N#CC(=O)C1CCCN1 | 1 |  |
| L-serine | 7 | CNO | 1 | 0 | 0.478 | COC(=O)C(N)CO | 0 |  |
| L-threonine | 8 | CNO | 1 | 0 | 0.522 | COC(=O)C(N)C(C)O | 1 |  |
| L-tryptophan | 15 | CNO | 0 | 0 | 0.263 | NC(=O)c1ccccc1 | 0 | 15 heavy atoms > 9 |
| L-tyrosine | 13 | CNO | 0 | 0 | 0.393 | OCc1ccc(O)cc1 | 0 | 13 heavy atoms > 9 |
| L-valine | 8 | CNO | 1 | 0 | 0.381 | CC(C)C(C)C(N)=O | 0 |  |
| indole | 9 | CN | 1 | 1 | 1.0 | c1ccc2[nH]ccc2c1 | 1 |  |
| indole-d6 | 9 | CN | 0 | 1 | 1.0 | c1ccc2[nH]ccc2c1 | 1 | isotopically labelled |
| oxindole | 10 | CNO | 0 | 0 | 0.433 | O=C1Cc2[nH]ccc2N1 | 0 | 10 heavy atoms > 9 |
| indole-3-acetic acid | 13 | CNO | 0 | 0 | 0.294 | OCc1ccccc1O | 0 | 13 heavy atoms > 9 |
| hypoxanthine | 10 | CNO | 0 | 0 | 0.462 | Nc1nc[nH]c(=O)c1N | 0 | 10 heavy atoms > 9 |

## 2. Label support in QM9

A probe trained on QM9S cannot learn a label that QM9 does not contain. Status is judged on `n_qm9s_usable` when QM9S records are supplied (molecules whose window string has enough peaks to learn from). `NO QM9 SUPPORT` labels must be dropped from any QM9S->SERS transfer claim.

| label | n_qm9 | frac_qm9 | n_qm9s_usable | n_analytes | analytes | status |
|---|---|---|---|---|---|---|
| carboxylic_acid | 0 | 0.0 | 0 | 21 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-proline, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine, indole-3-acetic acid | NO QM9 SUPPORT |
| primary_amine | 2482 | 0.0197 | 142 | 19 | L-alanine, L-arginine, L-asparagine, L-aspartic acid, L-cysteine, L-glutamic acid, L-glutamine, L-glycine, L-histidine, L-isoleucine, L-leucine, L-lysine, L-methionine, L-phenylalanine, L-serine, L-threonine, L-tryptophan, L-tyrosine, L-valine | ok |
| secondary_amine | 13346 | 0.1059 | 1022 | 1 | L-proline | ok |
| amide | 8836 | 0.0701 | 620 | 3 | L-asparagine, L-glutamine, oxindole | ok |
| guanidine | 246 | 0.002 | 17 | 1 | L-arginine | thin |
| hydroxyl_aliphatic | 37121 | 0.2946 | 2619 | 2 | L-serine, L-threonine | ok |
| phenol | 3132 | 0.0249 | 1376 | 1 | L-tyrosine | ok |
| thiol | 0 | 0.0 | 0 | 1 | L-cysteine | NO QM9 SUPPORT |
| thioether | 0 | 0.0 | 0 | 1 | L-methionine | NO QM9 SUPPORT |
| ketone | 14606 | 0.1159 | 1046 | 0 |  | unused |
| aldehyde | 13156 | 0.1044 | 1278 | 0 |  | unused |
| ester | 4545 | 0.0361 | 348 | 0 |  | unused |
| ether | 51176 | 0.4062 | 4434 | 0 |  | unused |
| nitrile | 16161 | 0.1283 | 1356 | 0 |  | unused |
| alkyne | 16720 | 0.1327 | 1515 | 0 |  | unused |
| alkene | 16379 | 0.13 | 1288 | 0 |  | unused |
| fluoro | 337 | 0.0027 | 211 | 0 |  | unused |
| aromatic_ring | 18742 | 0.1488 | 8644 | 9 | L-histidine, L-phenylalanine, L-tryptophan, L-tyrosine, indole, indole-d6, oxindole, indole-3-acetic acid, hypoxanthine | ok |
| benzene_ring | 184 | 0.0015 | 89 | 7 | L-phenylalanine, L-tryptophan, L-tyrosine, indole, indole-d6, oxindole, indole-3-acetic acid | thin |
| pyrrole_ring | 2296 | 0.0182 | 997 | 4 | L-tryptophan, indole, indole-d6, indole-3-acetic acid | ok |
| imidazole_ring | 1332 | 0.0106 | 595 | 2 | L-histidine, hypoxanthine | ok |
| indole | 1 | 0.0 | 1 | 4 | L-tryptophan, indole, indole-d6, indole-3-acetic acid | thin |
| lactam | 5840 | 0.0464 | 364 | 1 | oxindole | ok |

## 3. Spectral shape by source

Token counts are estimates (1 token per digit); run leakage checks with `--tokenizer` on Aura for exact numbers.

| source | kind | n_spectra | peaks_median | peaks_range | fwhm_median | fwhm_iqr | tokens_median_est | tokens_max_est |
|---|---|---|---|---|---|---|---|---|
| qm9s | sim | 59010 | 1.0 | 1-12 | 10.0 | 10-11 | 12 | 170 |
| sers | cell | 1 | 15.0 | 15-15 | 19.0 | 16-24 | 216 | 216 |
| sers | standard | 5 | 10.0 | 8-15 | 19.0 | 15-29 | 143 | 217 |
| sers | substrate_only | 1 | 13.0 | 13-13 | 20.0 | 17-32 | 186 | 186 |

Peak-position histogram overlap (25 cm-1 bins, 1 = identical distributions):

- qm9s/sim vs sers/cell: 0.305
- qm9s/sim vs sers/standard: 0.47
- qm9s/sim vs sers/substrate_only: 0.226
- sers/cell vs sers/standard: 0.471
- sers/cell vs sers/substrate_only: 0.533
- sers/standard vs sers/substrate_only: 0.5

## 4. Peaks shared across different analytes

Bands present in spectra of at least half of the SERS analytes. A flag, not a verdict: shared chemistry (e.g. ring breathing near 1000 cm-1 in Phe/Trp/Tyr) also shows up here. Check each against the CB-only spectra.

| position | n_analytes | frac_analytes | analytes |
|---|---|---|---|
| 671 | 5 | 1.0 | L-cysteine, L-phenylalanine, L-threonine, L-tryptophan, L-tyrosine |
| 962 | 3 | 0.6 | L-threonine, L-tryptophan, L-tyrosine |
| 998 | 3 | 0.6 | L-phenylalanine, L-tryptophan, L-tyrosine |
| 1118 | 3 | 0.6 | L-phenylalanine, L-tryptophan, L-tyrosine |
| 1190 | 3 | 0.6 | L-cysteine, L-phenylalanine, L-threonine |
| 1286 | 4 | 0.8 | L-cysteine, L-phenylalanine, L-threonine, L-tyrosine |
| 1327 | 3 | 0.6 | L-phenylalanine, L-threonine, L-tyrosine |
| 1377 | 4 | 0.8 | L-cysteine, L-phenylalanine, L-threonine, L-tyrosine |
| 1601 | 5 | 1.0 | L-cysteine, L-phenylalanine, L-threonine, L-tryptophan, L-tyrosine |
