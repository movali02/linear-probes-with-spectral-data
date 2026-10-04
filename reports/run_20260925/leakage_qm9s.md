# Leakage checks: `qm9s_records.jsonl` (59010 records, schema v1)

- **PASS** `grammar_whitelist`: all 59010 inputs are pure peak lists and round-trip
- **PASS** `forbidden_terms`: no analyte names, codes, SMILES, formulas or ids in inputs
- **PASS** `prompt_template`: fixed instruction, no label terms
- **PASS** `split_disjoint`: 59010 groups, none spans train and test
- **WARN** `duplicates`: exact cross-split duplicates: 0 same-label, 1448 different-label; near-duplicate pairs (cos>=0.98): 0 same-label, 1029535 different-label (near-dup scan on a random 3000 test items)
  - examples: `[['qm9s_1', 'qm9s_401', 'qm9s_2711', 'qm9s_4145'], ['qm9s_6', 'qm9s_22284', 'qm9s_24125', 'qm9s_24767'], ['qm9s_8', 'qm9s_25552', 'qm9s_32127', 'qm9s_50160'], ('qm9s_113567', 'qm9s_284', 1.0), ('qm9s_113567', 'qm9s_823', 1.0)]`
- **PASS** `token_budget`: prompt tokens estimated (1 token per digit): median 68, max 226, limit 2016
- **PASS** `shuffled_label_control`: [labels.fg_aromatic_ring, n=4000] LR baseline acc 0.800 (majority class 0.719); shuffled-label acc 0.717 +/- 0.001; permutation p = 0.032
- **SKIP** `substrate_only_control`: 0 substrate-only spectra with a trained label (need >=5)
