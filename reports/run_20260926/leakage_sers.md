# Leakage checks: `sers_records.jsonl` (7 records, schema v1.1)

- **PASS** `grammar_whitelist`: all 7 inputs are pure peak lists and round-trip
- **PASS** `forbidden_terms`: no analyte names, codes, SMILES, formulas or ids in inputs
- **PASS** `prompt_template`: fixed instruction, no label terms
- **PASS** `split_disjoint`: 7 groups, none spans train and test
- **PASS** `duplicates`: exact cross-split duplicates: 0 same-label, 0 different-label; near-duplicate pairs (cos>=0.98): 0 same-label, 0 different-label
- **PASS** `token_budget`: prompt tokens exact (/auto/homes/mv487/tokenisation/qwen_lora): median 162, max 243, limit 2016
- **SKIP** `shuffled_label_control`: need >=2 samples in every class and >=2 classes
- **SKIP** `substrate_only_control`: 1 substrate-only spectra with a trained label (need >=5)
