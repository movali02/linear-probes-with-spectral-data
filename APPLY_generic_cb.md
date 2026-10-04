# Generic CB reference (27 Sep 2026)

Unzip over your probe2circuit checkout on Aura (`unzip -o probe2circuit_generic_cb.zip` from ~).
Then `pytest -q` (66 tests).

Changed: src/probe2circuit/{pipeline,substrate}.py, configs/string_v1.yaml (schema v1.2,
generic_cb block, group_by: strip_replicate), scripts/{build_strings,run_step1,preflight}.py/.sh,
tests/{test_reference_norm,test_strings_and_pipeline}.py. New: scripts/make_generic_cb.py.

Run:
    cp /path/to/split_manifest.csv data/split_manifest.csv      # optional: limits the generic CB to these files
    MANIFEST=data/split_manifest.csv bash scripts/run_step1.sh

run_step1.sh builds data/generic_cb.txt first (and rebuilds it whenever a CB file is newer),
prints the CB band table, and copies generic_cb_report.json into the report folder.
