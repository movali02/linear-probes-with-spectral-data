# Leakage checks: `sers_records.jsonl` (869 records, schema v1.2)

- **PASS** `grammar_whitelist`: all 869 inputs are pure peak lists and round-trip
- **PASS** `forbidden_terms`: no analyte names, codes, SMILES, formulas or ids in inputs
- **PASS** `prompt_template`: fixed instruction, no label terms
- **PASS** `split_disjoint`: 109 groups, none spans train and test
- **WARN** `duplicates`: exact cross-split duplicates: 0 same-label, 0 different-label; near-duplicate pairs (cos>=0.98): 0 same-label, 7 different-label
  - examples: `[('Cell_BW_asn_18', 'Cell_BW_arg_12', 0.984), ('Cell_BW_asn_7', 'Cell_BW_ala_7', 0.981), ('Cell_BW_asn_8', 'Cell_536_tyr_5', 0.98), ('Cell_BW_gly_6', 'Cell_BW_tyr_13', 0.981), ('Cell_BW_ile_12', 'Cell_BW_glu_16', 0.983)]`
- **PASS** `token_budget`: prompt tokens estimated (1 token per digit): median 256, max 394, limit 2016
- **SKIP** `shuffled_label_control`: need >=2 samples in every class and >=2 classes
- **PASS** `substrate_only_control`: accuracy on substrate-only spectra 0.068 vs chance 0.060
