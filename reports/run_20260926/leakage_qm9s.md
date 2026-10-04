# Leakage checks: `qm9s_records.jsonl` (125996 records, schema v1.1)

- **PASS** `grammar_whitelist`: all 125996 inputs are pure peak lists and round-trip
- **PASS** `forbidden_terms`: no analyte names, codes, SMILES, formulas or ids in inputs
- **PASS** `prompt_template`: fixed instruction, no label terms
- **PASS** `split_disjoint`: 125996 groups, none spans train and test
- **WARN** `duplicates`: exact cross-split duplicates: 0 same-label, 16 different-label; near-duplicate pairs (cos>=0.98): 0 same-label, 72 different-label (near-dup scan on a random 3000 test items)
  - examples: `[['qm9s_894', 'qm9s_897'], ['qm9s_4050', 'qm9s_4054'], ['qm9s_20866', 'qm9s_20904'], ('qm9s_96655', 'qm9s_1145', 0.984), ('qm9s_96655', 'qm9s_8577', 0.985)]`
- **PASS** `token_budget`: prompt tokens exact (/auto/homes/mv487/tokenisation/qwen_lora): median 278, max 436, limit 2016
- **PASS** `shuffled_label_control`: [labels.fg_aromatic_ring, n=4000] LR baseline acc 0.899 (majority class 0.853); shuffled-label acc 0.852 +/- 0.000; permutation p = 0.032
- **SKIP** `substrate_only_control`: 0 substrate-only spectra with a trained label (need >=5)
