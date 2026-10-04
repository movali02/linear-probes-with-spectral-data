import json

import numpy as np

from probe2circuit import substrate as sub
from probe2circuit.leakage import check_duplicates
from probe2circuit.pipeline import build_qm9s_records_from_peaklists, lorentz_width_to_fwhm
from probe2circuit.strings import parse_string


def _row(pos, ints, widths, smi):
    j = {"Wavenumbers": ",".join(map(str, pos)), "Intensities": ",".join(map(str, ints)),
         "Widths": ",".join(map(str, widths))}
    return json.dumps({"prompt": "Raman spectroscopy " + json.dumps(j), "response": "##SMILES: " + smi})


def test_lorentzian_width_conversion():
    # a 10 cm-1 FWHM Lorentzian measured at rel_height 0.7 is 15.28 cm-1 wide
    assert abs(lorentz_width_to_fwhm(15.275, 0.7) - 10.0) < 0.01


def test_peaklist_window_renormalisation(cfg, tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join([
        _row([1000.0, 1450.0, 2950.0], [0.2, 0.1, 1.0], [15.3, 15.3, 15.3], "CCO"),   # C-H dominates
        _row([2950.0, 3400.0], [1.0, 0.5], [15.3, 15.3], "C"),                         # nothing in window
    ]) + "\n")
    recs, st = build_qm9s_records_from_peaklists(p, cfg)
    assert st["rows"] == 2 and st["no_peak_in_window"] == 1 and len(recs) == 1
    pk = parse_string(recs[0]["text"], cfg)
    assert [q["position"] for q in pk] == [1000, 1450]
    assert [q["intensity"] for q in pk] == [1.0, 0.5]
    assert all(q["width"] == 10 for q in pk)
    assert recs[0]["effective_floor"] == 0.5     # 0.1 cut / 0.2 window max


def test_condition_mean_pairing():
    refs = {"CB_536_trp_1": np.ones(3), "CB_536_trp_2": 3 * np.ones(3), "CB_BW_trp_1": 10 * np.ones(3)}
    ref, lab = sub.find_reference("Cell_536_trp_7", refs, "condition_mean")
    assert np.allclose(ref, 2.0) and "n=2" in lab
    ref, lab = sub.find_reference("Cell_536_trp_1", refs, "same_index")
    assert lab == "CB_536_trp_1"
    assert sub.find_reference("Cell_536_his_1", refs, "condition_mean") == (None, None)


def test_identical_strings_different_labels_warn_not_fail(cfg):
    recs = [{"id": "a", "group": "a", "split": "train", "analyte": "X", "text": "1000 1.00 10"},
            {"id": "b", "group": "b", "split": "test", "analyte": "Y", "text": "1000 1.00 10"}]
    assert check_duplicates(recs, cfg)["status"] == "WARN"
    recs[1]["analyte"] = "X"
    assert check_duplicates(recs, cfg)["status"] == "FAIL"
