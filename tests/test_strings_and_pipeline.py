import numpy as np
import pytest

from probe2circuit import preprocess as pp
from probe2circuit import substrate as sub
from probe2circuit.io import load_sers_txt
from probe2circuit.pipeline import analyte_from_id, build_qm9s_records, build_sers_records, kind_from_id
from probe2circuit.io import load_number_smiles
from probe2circuit.peaks import detect_peaks
from probe2circuit.strings import estimate_tokens, grammar, parse_string, to_string


def lorentz(grid, pos, amp, fwhm):
    hw = fwhm / 2
    return sum(a * hw**2 / ((grid - p) ** 2 + hw**2) for p, a in zip(pos, amp))


def test_roundtrip(cfg):
    peaks = [{"position": 1003.4, "intensity": 1.0, "width": 9.6},
             {"position": 620.6, "intensity": 0.254, "width": 14.2}]
    s = to_string(peaks, cfg)
    assert s == "621 0.25 14 | 1003 1.00 10"
    assert to_string(parse_string(s, cfg), cfg) == s


@pytest.mark.parametrize("bad", ["621 0.25 14 | tyr", "medium:M9 621 0.25 14", "621 0.25", "",
                                 "621 0.25 14 | 1003 1.00 10 | N[C@@H](CS)C(=O)O"])
def test_grammar_rejects_anything_but_peaks(cfg, bad):
    assert not grammar(cfg).match(bad)


def test_detector_recovers_known_peaks_and_widths(cfg):
    grid = pp.make_grid(cfg)
    y = lorentz(grid, [700, 1003, 1600], [0.5, 1.0, 0.3], 12.0)
    peaks = detect_peaks(grid, y / y.max(), cfg)
    assert [round(p["position"]) for p in peaks] == [700, 1003, 1600]
    assert all(abs(p["width"] - 12.0) < 1.0 for p in peaks)


def test_detector_ignores_noise(cfg):
    rng = np.random.default_rng(1)
    grid = pp.make_grid(cfg)
    y = lorentz(grid, [1003], [1.0], 12.0) + rng.normal(0, 0.02, grid.size)
    from probe2circuit.pipeline import _finish
    peaks, _ = _finish(grid, y, cfg)
    assert [round(p["position"]) for p in peaks] == [1003]


def test_qm9s_renormalises_inside_window(cfg, root):
    """C-H stretches (~3000) dominate the raw fixture; inside the window the
    strongest fingerprint peak must be 1.00."""
    n2s = load_number_smiles(root / "tests/fixtures/qm9s_mini/number_smiles.csv")
    recs, stats = build_qm9s_records(root / "tests/fixtures/qm9s_mini/raman_boraden.csv", n2s, cfg)
    assert stats["no_smiles"] == 1 and len(recs) == 13
    for r in recs:
        ints = [p["intensity"] for p in parse_string(r["text"], cfg)]
        assert max(ints) == 1.0
        assert all(500 <= p["position"] <= 1750 for p in parse_string(r["text"], cfg))


def test_window_must_be_covered(cfg):
    x = np.arange(600, 1800.0)
    with pytest.raises(ValueError):
        pp.prepare_simulated(x, np.ones_like(x), cfg)


def test_substrate_subtraction_recovers_scale_and_shift(cfg):
    grid = pp.make_grid(cfg)
    ref = lorentz(grid, [617, 676, 755, 829, 881], [0.1, 0.1, 0.15, 1.0, 0.3], 14)
    analyte = lorentz(grid, [1003, 1550], [0.6, 0.4], 12)
    shifted_ref = np.interp(grid, grid + 1.5, ref)          # 1.5 cm-1 calibration drift
    sample = analyte + 0.7 * shifted_ref
    resid, s, shift = sub.subtract(sample, ref, grid, cfg)
    assert abs(s - 0.7) < 0.02 and abs(shift - 1.5) < 0.3
    assert np.abs(resid - analyte).max() < 0.03


def test_mask_catalogue(cfg):
    peaks = [{"position": 827, "intensity": 1, "width": 10}, {"position": 1003, "intensity": 1, "width": 10}]
    kept, n = sub.mask_peaks(peaks, [829.0], cfg["substrate"]["tol_cm"])
    assert n == 1 and kept[0]["position"] == 1003


def test_normalise_id():
    from probe2circuit.pipeline import normalise_id
    assert normalise_id("Cell_Cell_BW_pro_18") == "Cell_BW_pro_18"
    assert normalise_id("Cell_CB_536_trp_4") == "CB_536_trp_4"
    assert normalise_id("AAs_L-threonine") == "L-threonine"
    assert normalise_id("CB_536_trp_1") == "CB_536_trp_1"
    assert kind_from_id(normalise_id("Cell_CB_536_trp_4")) == "substrate_only"


def test_standard_cb_pairing_convention():
    import numpy as np
    ref, lab = sub.find_reference("L-tyrosine", {"CB_L-tyrosine_1": np.ones(2), "CB_L-tyrosine_2": np.ones(2)},
                                  "condition_mean")
    assert ref is not None and "n=2" in lab


def test_id_parsing(analytes):
    assert analyte_from_id("Cell_BW_tyr_1", analytes)["name"] == "L-tyrosine"
    assert analyte_from_id("CB_536_trp_1", analytes)["name"] == "L-tryptophan"
    assert analyte_from_id("L-phenylalanine", analytes)["name"] == "L-phenylalanine"
    assert analyte_from_id("CB_L-tyrosine_3", analytes)["name"] == "L-tyrosine"
    assert kind_from_id("CB_536_trp_1") == "substrate_only"
    assert sub.pair_name("Cell_536_trp_1") == "CB_536_trp_1"


def test_sample_sers_files_build(cfg, analytes, root):
    paths = sorted((root / "data/raw_sers_sample").glob("*.txt"))
    recs, info = build_sers_records(paths, cfg, analytes)
    assert len(recs) == 7 and info["n_refs"] == 1
    assert all(grammar(cfg).match(r["text"]) for r in recs)
    # reference_norm catalogue = CB bands inside cb_regions, read off the smoothed
    # CB-only mean; the 829 band sits at 826-828 in this session
    assert any(825.0 <= p <= 830.0 for p in info["catalogue"])
    # the CB[5] reference band itself never reaches a string
    for r in recs:
        if r["substrate"]["mode_used"].startswith("reference_norm"):
            assert not any(abs(p["position"] - r["substrate"]["ref_pos_cm"]) <= 2
                           for p in parse_string(r["text"], cfg))


def test_loader_handles_header_and_descending(root):
    x, y = load_sers_txt(root / "data/raw_sers_sample/CB_536_trp_1.txt")
    assert x[0] < x[-1] and len(x) == 4060


def test_token_estimate_counts_digits():
    assert estimate_tokens("1003 1.00 10") == 4 + 1 + 4 + 1 + 2


def test_sers_files_found_in_subfolders(tmp_path, root, cfg, analytes):
    """raw_sers/AAs/, raw_sers/indoles/, raw_sers/Cell/ (one of them a symlink)."""
    import os
    import shutil
    from probe2circuit.io import find_sers_files, subset_of
    src = root / "data/raw_sers_sample"
    sers = tmp_path / "raw_sers"
    (sers / "AAs").mkdir(parents=True)
    real_cell = tmp_path / "elsewhere_cell"
    real_cell.mkdir()
    os.symlink(real_cell, sers / "Cell")
    (sers / "indoles").mkdir()
    for f in src.glob("L-*.txt"):
        shutil.copy(f, sers / "AAs" / f.name)
    shutil.copy(src / "L-tryptophan.txt", sers / "indoles" / "indole.txt")
    for f in ("Cell_BW_tyr_1.txt", "CB_536_trp_1.txt"):
        shutil.copy(src / f, real_cell / f)
    (sers / "AAs" / "README.txt").write_text("notes")
    files = find_sers_files(sers)
    assert len(files) == 8
    assert {subset_of(f, sers) for f in files} == {"AAs", "indoles", "Cell"}
    recs, info = build_sers_records(files, cfg, analytes)
    kinds = {r["id"]: r["kind"] for r in recs}
    assert kinds["Cell_BW_tyr_1"] == "cell" and kinds["CB_536_trp_1"] == "substrate_only"
    assert kinds["indole"] == "standard" and kinds["L-tyrosine"] == "standard"
    assert analyte_from_id("indole", analytes)["name"] == "indole"
