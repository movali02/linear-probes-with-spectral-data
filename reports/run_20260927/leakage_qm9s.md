# Leakage checks: `qm9s_records.jsonl` (129817 records, schema v1.2)

- **PASS** `grammar_whitelist`: all 129817 inputs are pure peak lists and round-trip
- **PASS** `forbidden_terms`: no analyte names, codes, SMILES, formulas or ids in inputs
- **PASS** `prompt_template`: fixed instruction, no label terms
- **PASS** `split_disjoint`: 129817 groups, none spans train and test
- **WARN** `duplicates`: exact cross-split duplicates: 0 same-label, 26 different-label; near-duplicate pairs (cos>=0.98): 0 same-label, 33 different-label (near-dup scan on a random 3000 test items)
  - examples: `[['qm9s_894', 'qm9s_897'], ['qm9s_4050', 'qm9s_4054'], ['qm9s_4386', 'qm9s_4391'], ('qm9s_2777', 'qm9s_47761', 0.983), ('qm9s_28067', 'qm9s_14090', 0.982)]`
- **PASS** `token_budget`: prompt tokens estimated (1 token per digit): median 324, max 496, limit 2016
- **PASS** `shuffled_label_control`: [labels.fg_aromatic_ring, n=4000] LR baseline acc 0.928 (majority class 0.827); shuffled-label acc 0.826 +/- 0.000; permutation p = 0.032
- **SKIP** `substrate_only_control`: 0 substrate-only spectra with a trained label (need >=5)
