"""Each leakage check must PASS on clean data and FAIL on data with that leak
injected. A check that can't fail is not a check."""
import copy
import json

from conftest import synth_records

from probe2circuit import leakage as L


def status(res):
    return res["status"]


def test_clean_synthetic_passes_everything(cfg, analytes):
    recs = synth_records(cfg)
    for r in L.run_all(recs, cfg, analytes):
        assert r["status"] in ("PASS", "SKIP", "WARN"), r


def test_grammar_catches_label_in_text(cfg):
    recs = synth_records(cfg)
    recs[3]["text"] += " | tyr"
    assert status(L.check_grammar(recs, cfg)) == "FAIL"


def test_forbidden_catches_names_smiles_ids(cfg, analytes):
    for inj in ["L-tyrosine", "tryptophan", "N[C@@H](CS)C(=O)O", "C9H11NO3", "medium:M9_tyr", "s0_1"]:
        recs = synth_records(cfg)
        recs[0]["text"] = recs[0]["text"] + " " + inj
        assert status(L.check_forbidden_terms(recs, analytes)) == "FAIL", inj


def test_forbidden_no_false_positive_on_numbers(cfg, analytes):
    assert status(L.check_forbidden_terms(synth_records(cfg), analytes)) == "PASS"


def test_prompt_template(cfg, analytes):
    assert status(L.check_prompt_template(cfg["prompt_template"], analytes)) == "PASS"
    assert status(L.check_prompt_template("Is this tryptophan? {text}", analytes)) == "FAIL"
    assert status(L.check_prompt_template("no slot", analytes)) == "FAIL"


def test_split_checks(cfg):
    recs = synth_records(cfg)
    recs[0]["group"] = recs[5]["group"]         # test and train items share a group
    assert status(L.check_split_disjoint(recs)) == "FAIL"


def test_exact_duplicate_across_splits(cfg):
    recs = synth_records(cfg)
    te = next(r for r in recs if r["split"] == "test")
    tr = next(r for r in recs if r["split"] == "train")
    tr["text"] = te["text"]
    assert status(L.check_duplicates(recs, cfg)) == "FAIL"


def test_near_duplicate_flagged(cfg):
    recs = synth_records(cfg)
    te = next(r for r in recs if r["split"] == "test")
    clone = copy.deepcopy(te)
    clone.update(id="clone", group="clone", split="train")
    from probe2circuit.strings import parse_string, to_string
    peaks = parse_string(clone["text"], cfg)
    for p in peaks:
        p["width"] += 1                                     # widths change, peaks identical
    clone["text"] = to_string(peaks, cfg)
    recs.append(clone)
    assert status(L.check_duplicates(recs, cfg)) == "WARN"


def test_token_budget(cfg):
    recs = synth_records(cfg)
    long = " | ".join(["1003 1.00 10"] * 300)
    recs[0]["text"] = long
    assert status(L.check_token_budget(recs, cfg)) == "FAIL"


def test_shuffled_control_passes_and_reports_baseline(cfg):
    res = L.shuffled_label_control(synth_records(cfg, signal=True), cfg, n_perm=10)
    assert res["status"] == "PASS" and res["lr_accuracy"] > 0.8 and res["p_value"] < 0.2


def test_shuffled_control_no_signal_is_chance(cfg):
    res = L.shuffled_label_control(synth_records(cfg, signal=False), cfg, n_perm=10)
    assert res["lr_accuracy"] < 0.6


def test_substrate_only_control_catches_session_leak(cfg):
    """Simulate a session effect: each class measured on a different day with a
    day-specific band that ALSO shows up in that day's CB-only spectrum."""
    recs = synth_records(cfg, n_classes=4, per_class=8, signal=False)
    day_band = {f"class{c}": 600 + 250 * c for c in range(4)}
    for r in recs:
        r["text"] += f" | {day_band[r['analyte']]} 0.90 12"
        from probe2circuit.strings import parse_string, to_string
        r["text"] = to_string(parse_string(r["text"], cfg), cfg)
    blanks = []
    for c in range(4):
        for k in range(3):
            blanks.append({"id": f"CB_{c}_{k}", "group": f"CB_{c}_{k}", "kind": "substrate_only",
                           "analyte": f"class{c}", "split": "control",
                           "text": f"829 1.00 15 | {day_band[f'class{c}']} 0.90 12"})
    res = L.substrate_only_control(recs + blanks, cfg)
    assert res["status"] == "FAIL", res


def test_legacy_audit_finds_known_leaks(root, analytes):
    docs = {p.stem: json.load(open(p)) for p in (root / "data/legacy_string_json").glob("*.json")}
    rows = {r["file"]: r["leaks_in_old"] for r in L.audit_legacy_json(docs, analytes)}
    assert "medium=M9_tyrosine" in rows["Cell_BW_tyr_1"]
    assert "Molecule block filled" in rows["L-tryptophan"]
    assert "mode_assignment" in rows["L-tryptophan"]
