# Leakage checks: `sers_records.jsonl` (7 records, schema v1)

- **PASS** `grammar_whitelist`: all 7 inputs are pure peak lists and round-trip
- **PASS** `forbidden_terms`: no analyte names, codes, SMILES, formulas or ids in inputs
- **PASS** `prompt_template`: fixed instruction, no label terms
- **PASS** `split_disjoint`: 7 groups, none spans train and test
- **PASS** `duplicates`: exact cross-split duplicates: 0 same-label, 0 different-label; near-duplicate pairs (cos>=0.98): 0 same-label, 0 different-label
- **PASS** `token_budget`: prompt tokens estimated (1 token per digit): median 215, max 273, limit 2016
- **SKIP** `shuffled_label_control`: need >=2 samples in every class and >=2 classes
- **SKIP** `substrate_only_control`: 1 substrate-only spectra with a trained label (need >=5)

## Old strings vs new strings for the same spectra

`old_substrate_peaks` = old peaks within 4 cm-1 of a major CB[5] band. Old token counts are for the full old string (prefix + 9 fields per peak).

| file | old_peaks | old_substrate_peaks | old_tokens_est | new_peaks | new_tokens_est | leaks_in_old |
|---|---|---|---|---|---|---|
| Cell_BW_tyr_1 | 20 | 4 | 1096 | 15 | 216 | Spectrum.medium=M9_tyrosine; meta name/path contains label (safe only if never serialised) |
| L-cysteine | 15 | 3 | 985 | 8 | 115 | Molecule.chemical_formula=C3H7NO2S; Molecule.SMILES=N[C@@H](CS)C(=O)O; Molecule block filled (SMILES/formula/scaffold/functional_groups); 6 peaks carry mode_assignment looked up by analyte name; meta name/path contains label (safe only if never serialised) |
| L-phenylalanine | 18 | 5 | 1203 | 11 | 159 | Molecule.chemical_formula=C9H11NO2; Molecule.SMILES=N[C@@H](Cc1ccccc1)C(=O)O; Molecule block filled (SMILES/formula/scaffold/functional_groups); 8 peaks carry mode_assignment looked up by analyte name; meta name/path contains label (safe only if never serialised) |
| L-threonine | 21 | 5 | 1312 | 8 | 113 | Molecule.chemical_formula=C4H9NO3; Molecule.SMILES=C[C@@H](O)[C@H](N)C(=O)O; Molecule block filled (SMILES/formula/scaffold/functional_groups); 11 peaks carry mode_assignment looked up by analyte name; meta name/path contains label (safe only if never serialised) |
| L-tryptophan | 17 | 3 | 1198 | 15 | 217 | Molecule.chemical_formula=C11H12N2O2; Molecule.SMILES=N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O; Molecule.scaffold=indole; Molecule.functional_groups=primary_amine_carboxylic_acid_indole; Molecule block filled (SMILES/formula/scaffold/functional_groups); 10 peaks carry mode_assignment looked up by analyte name; meta name/path contains label (safe only if never serialised) |
| L-tyrosine | 15 | 3 | 1011 | 10 | 143 | Molecule.chemical_formula=C9H11NO3; Molecule.SMILES=N[C@@H](Cc1ccc(O)cc1)C(=O)O; Molecule block filled (SMILES/formula/scaffold/functional_groups); 5 peaks carry mode_assignment looked up by analyte name; meta name/path contains label (safe only if never serialised) |
