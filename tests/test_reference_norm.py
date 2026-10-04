"""SERS substrate.mode = reference_norm: CB[5] ~829 band = 1.0, then discard the
peaks the substrate explains, without discarding Trp/indole 758 or Trp 877.

All spectra here are synthetic Lorentzian sums written as two-column .txt files
(descending axis, like the WiRE exports) so they go through the real loader,
ALS baseline and resampling."""
import numpy as np
import pytest

from probe2circuit import substrate as sub
from probe2circuit.pipeline import _finish, build_sers_records
from probe2circuit import preprocess as pp
from probe2circuit.strings import grammar, parse_string, to_string

# CB[5] on MLagg: bands near 617, 676, 755, 827 (reference), 880
CB_POS = [617, 676, 755, 827, 880]
CB_AMP = [0.10, 0.12, 0.30, 1.00, 0.35]
TRP = ([758, 877, 1010, 1550], [0.90, 0.60, 0.50, 0.40])   # strong Trp 758 and 877
PHE = ([1003, 1032, 1605], [1.50, 0.30, 0.40])            # no bands near CB 755/880
REGIONS = [(745, 765), (812, 845), (870, 895)]


def lorentz(x, pos, amp, fwhm=18.0):
    hw = fwhm / 2
    return sum(a * hw**2 / ((x - p) ** 2 + hw**2) for p, a in zip(pos, amp))


def write_spectrum(path, cb_scale, analyte=None, analyte_scale=1.0, seed=0, overall=1000.0,
                   cb_ratio_jitter=0.05):
    rng = np.random.default_rng(seed)
    x = np.arange(3200.0, 99.0, -1.0)                         # descending, like WiRE
    amp = np.array(CB_AMP) * (1 + rng.normal(0, cb_ratio_jitter, len(CB_AMP)))
    amp[3] = CB_AMP[3]                                        # reference band amplitude fixed
    y = cb_scale * lorentz(x, CB_POS, amp)
    if analyte:
        y = y + analyte_scale * lorentz(x, analyte[0], analyte[1])
    y = overall * y + 200 + 0.05 * x + rng.normal(0, 2.0, x.size)   # offset + slope + noise
    np.savetxt(path, np.c_[x, y], fmt="%.4f")
    return path


@pytest.fixture
def cfg_nofile(cfg):
    """Config whose generic CB is built in-run (no generic_cb.txt on disk)."""
    return {**cfg, "substrate": {**cfg["substrate"],
                                 "generic_cb": {**cfg["substrate"]["generic_cb"], "path": None}}}


@pytest.fixture
def session(tmp_path):
    """3 CB-only repeats + 3 cell spectra each for a Trp and a Phe condition."""
    paths = []
    for med, ana in [("trp", TRP), ("phe", PHE)]:
        for k in range(1, 4):
            paths.append(write_spectrum(tmp_path / f"CB_536_{med}_{k}.txt", 1.0, seed=10 * k + len(med)))
            paths.append(write_spectrum(tmp_path / f"Cell_536_{med}_{k}.txt", 0.6 + 0.3 * k, ana,
                                        seed=100 * k + len(med), overall=500.0 * k))
    return sorted(paths)


def _peaks(rec, cfg):
    return parse_string(rec["text"], cfg)


def _in(pk, lo, hi):
    return [p for p in pk if lo <= p["position"] <= hi]


def test_config_defaults_to_reference_norm(cfg):
    assert cfg["substrate"]["mode"] == "reference_norm"
    assert cfg["substrate"]["reference_norm"]["cb_regions"] == [list(r) for r in REGIONS]


def test_cb_only_records_have_no_cb_region_peaks(cfg_nofile, session):
    cfg = cfg_nofile
    recs, info = build_sers_records(session, cfg, [])
    assert info["n_refs"] == 6
    for r in (r for r in recs if r["kind"] == "substrate_only"):
        pk = _peaks(r, cfg)
        for lo, hi in REGIONS:
            assert not _in(pk, lo, hi), (r["id"], r["text"])
        assert r["substrate"]["cb_ref"].startswith("mean(536_")


def test_phe_cells_lose_cb_bands_and_keep_analyte_above_one(cfg_nofile, session):
    cfg = cfg_nofile
    recs, _ = build_sers_records(session, cfg, [])
    for r in (r for r in recs if r["id"].startswith("Cell_536_phe")):
        pk = _peaks(r, cfg)
        for lo, hi in REGIONS:
            assert not _in(pk, lo, hi), (r["id"], r["text"])
        phe = _in(pk, 998, 1008)
        assert phe, r["text"]
        # Phe 1003 amplitude is 1.5 against CB 827 amplitude 1.0 x cb_scale, so on
        # the 829 = 1.0 scale it must come out > 1 (it would be clipped before)
        assert phe[0]["intensity"] > 1.0
        assert r["substrate"]["n_masked"] >= 3                 # 755, 827, 880


def test_trp_758_and_877_survive_cb_755_and_880(cfg_nofile, session):
    cfg = cfg_nofile
    recs, _ = build_sers_records(session, cfg, [])
    for r in (r for r in recs if r["id"].startswith("Cell_536_trp")):
        pk = _peaks(r, cfg)
        assert _in(pk, 754, 762), f"Trp 758 discarded: {r['text']} / {r['substrate']['discarded']}"
        assert _in(pk, 872, 884), f"Trp 877 discarded: {r['text']} / {r['substrate']['discarded']}"
        assert not _in(pk, 820, 834), r["text"]              # reference band always removed
        assert _in(pk, 1005, 1015) and _in(pk, 1545, 1555)


def test_strings_are_invariant_to_overall_intensity(cfg_nofile, tmp_path):
    """Normalising to 829 means a spectrum recorded 5x brighter gives the same string."""
    cfg = cfg_nofile
    a = [write_spectrum(tmp_path / f"CB_536_phe_{k}.txt", 1.0, seed=k) for k in (1, 2)]
    b1 = write_spectrum(tmp_path / "Cell_536_phe_1.txt", 0.8, PHE, seed=7, overall=400.0)
    r1, _ = build_sers_records(a + [b1], cfg, [])
    b1 = write_spectrum(tmp_path / "Cell_536_phe_1.txt", 0.8, PHE, seed=7, overall=2000.0)
    r2, _ = build_sers_records(a + [b1], cfg, [])
    t1 = next(r["text"] for r in r1 if r["kind"] == "cell")
    t2 = next(r["text"] for r in r2 if r["kind"] == "cell")
    p1, p2 = parse_string(t1, cfg), parse_string(t2, cfg)
    assert [p["position"] for p in p1] == [p["position"] for p in p2]
    assert np.allclose([p["intensity"] for p in p1], [p["intensity"] for p in p2], atol=0.02)


def test_all_strings_valid_and_non_negative(cfg_nofile, session):
    cfg = cfg_nofile
    recs, _ = build_sers_records(session, cfg, [])
    for r in recs:
        assert grammar(cfg).match(r["text"])
        assert to_string(parse_string(r["text"], cfg), cfg) == r["text"]
        assert all(p["intensity"] >= 0 for p in _peaks(r, cfg))


def test_no_cb_band_falls_back_and_is_flagged(cfg_nofile, tmp_path):
    cfg = cfg_nofile
    p = write_spectrum(tmp_path / "L-phenylalanine.txt", 0.0, PHE, seed=3)
    recs, _ = build_sers_records([p], cfg, [])
    assert recs[0]["substrate"]["mode_used"] == "max_in_window(ref_missing)"
    assert max(q["intensity"] for q in _peaks(recs[0], cfg)) == 1.0


def test_unpaired_standard_uses_generic_cb_and_is_flagged(cfg_nofile, tmp_path):
    cfg = cfg_nofile
    refs = [write_spectrum(tmp_path / f"CB_536_trp_{k}.txt", 1.0, seed=k) for k in (1, 2, 3)]
    std = write_spectrum(tmp_path / "L-phenylalanine.txt", 1.0, PHE, seed=9)
    recs, info = build_sers_records(refs + [std], cfg, [])
    r = next(r for r in recs if r["id"] == "L-phenylalanine")
    assert r["substrate"]["cb_ref"].startswith("generic(in-run, n=3")
    assert info["generic_cb"]["n"] == 3 and info["n_cells_on_generic"] == 0
    for lo, hi in REGIONS:
        assert not _in(_peaks(r, cfg), lo, hi), r["text"]


def test_rescale_option_restores_max_one(cfg_nofile, session):
    cfg = cfg_nofile
    c = {**cfg, "substrate": {**cfg["substrate"],
                              "reference_norm": {**cfg["substrate"]["reference_norm"],
                                                 "rescale_to_max_after_discard": True}}}
    recs, _ = build_sers_records(session, c, [])
    for r in recs:
        pk = _peaks(r, c)
        if pk:
            assert max(p["intensity"] for p in pk) == 1.0


# ---------------------------------------------------------------- generic CB
def test_generic_cb_file_is_used_for_standards_not_for_paired_cells(cfg, tmp_path):
    """Build the generic CB from cell-assay CB files with the script's code path,
    save it, and check a standard processed in a SEPARATE run uses it."""
    from probe2circuit.io import load_sers_txt
    from probe2circuit.pipeline import normalised_cb_references
    cb = [write_spectrum(tmp_path / f"CB_{s}_{aa}_{k}.txt", 1.0, seed=31 * k + i)
          for i, (s, aa) in enumerate([("BW", "ala"), ("536", "gly")]) for k in (1, 2, 3, 4)]
    loaded, grid = {}, None
    for p in cb:
        x, y = load_sers_txt(p)
        grid, loaded[p.stem] = pp.prepare_experimental(x, y, cfg)
    _, _, refs_norm = normalised_cb_references(loaded, grid, cfg)
    g, rep = sub.build_generic_cb(refs_norm, 0.9)
    gpath = tmp_path / "generic_cb.txt"
    sub.save_generic_cb(gpath, grid, g, {"n_files": rep["n_files"], "n_used": rep["n_used"]})
    cfg2 = {**cfg, "substrate": {**cfg["substrate"],
                                 "generic_cb": {**cfg["substrate"]["generic_cb"], "path": str(gpath)}}}
    std_dir = tmp_path / "standards"
    std_dir.mkdir()
    std = write_spectrum(std_dir / "L-phenylalanine.txt", 1.0, PHE, seed=5)
    cell = write_spectrum(std_dir / "Cell_BW_ala_1.txt", 1.0, PHE, seed=6)
    cbown = write_spectrum(std_dir / "CB_BW_ala_1.txt", 1.0, seed=8)
    recs, info = build_sers_records([std, cell, cbown], cfg2, [])
    by = {r["id"]: r for r in recs}
    assert by["L-phenylalanine"]["substrate"]["cb_ref"] == "generic(file:generic_cb.txt, n=8, p90)"
    assert by["Cell_BW_ala_1"]["substrate"]["cb_ref"].startswith("mean(BW_ala")   # own CB wins
    for lo, hi in REGIONS:
        assert not _in(_peaks(by["L-phenylalanine"], cfg2), lo, hi)
    assert _in(_peaks(by["L-phenylalanine"], cfg2), 998, 1008)


def test_generic_cb_rejects_an_odd_cb_file(cfg, tmp_path):
    from probe2circuit.io import load_sers_txt
    from probe2circuit.pipeline import normalised_cb_references
    paths = [write_spectrum(tmp_path / f"CB_BW_ala_{k}.txt", 1.0, seed=k) for k in range(1, 7)]
    paths.append(write_spectrum(tmp_path / "CB_BW_ala_7.txt", 1.0, ([1003, 1600], [3.0, 2.0]), seed=99))
    loaded, grid = {}, None
    for p in paths:
        x, y = load_sers_txt(p)
        grid, loaded[p.stem] = pp.prepare_experimental(x, y, cfg)
    _, _, refs_norm = normalised_cb_references(loaded, grid, cfg)
    g, rep = sub.build_generic_cb(refs_norm, 0.9)
    assert rep["n_used"] == 6 and rep["rejected"][0][0] == "CB_BW_ala_7"


def test_p90_heights_are_at_least_the_mean(cfg, tmp_path):
    grid = pp.make_grid(cfg)
    rng = np.random.default_rng(0)
    refs = {f"CB_BW_x_{k}": lorentz(grid, CB_POS, np.array(CB_AMP) * rng.uniform(0.7, 1.3, 5)) for k in range(20)}
    refs = {k: v / v[np.abs(grid - 827).argmin()] for k, v in refs.items()}
    g, _ = sub.build_generic_cb(refs, 0.5)
    args = (grid, cfg["substrate"]["reference_norm"]["cb_regions"], 0.1, cfg["grid_step"])
    mean_b = dict(sub.generic_bands(g, *args, height="mean"))
    p90_b = dict(sub.generic_bands(g, *args, height="p90"))
    assert set(mean_b) == set(p90_b) and all(p90_b[p] >= mean_b[p] - 1e-9 for p in mean_b)


# ---------------------------------------------------------------- unit level
def test_discard_rule():
    bands = [(755.0, 0.30), (827.0, 1.0), (880.0, 0.35)]
    peaks = [{"position": 756, "intensity": 0.32, "width": 18},   # CB 755 alone -> drop
             {"position": 758, "intensity": 1.10, "width": 18},   # Trp on top of CB -> keep
             {"position": 827, "intensity": 1.00, "width": 18},   # reference -> drop
             {"position": 879, "intensity": 0.42, "width": 18},   # ratio 1.2 < 1.5 -> drop
             {"position": 1003, "intensity": 1.4, "width": 12}]   # far from CB -> keep
    kept, dropped = sub.discard_substrate_peaks(peaks, bands, tol=5.0, min_excess=0.10, min_ratio=1.5)
    assert [p["position"] for p in kept] == [758, 1003]
    assert [p["position"] for p in dropped] == [756, 827, 879]


def test_reference_height_tracks_band_drift(cfg):
    grid = pp.make_grid(cfg)
    for centre in (826.0, 829.0, 831.0):
        y = lorentz(grid, [centre], [2.5])
        h, pos = sub.reference_height(grid, y, 829.0, 4.0)
        assert abs(h - 2.5) < 1e-6 and pos == centre


def test_to_string_keeps_intensities_above_one(cfg):
    s = to_string([{"position": 1003, "intensity": 1.374, "width": 12}], cfg)
    assert s == "1003 1.37 12"


def test_qm9s_dft_path_still_max_normalised(cfg):
    grid = pp.make_grid(cfg)
    y = lorentz(grid, [700, 1003], [0.4, 3.0], 12.0)
    peaks, _ = _finish(grid, y, cfg)
    assert max(p["intensity"] for p in peaks) == pytest.approx(1.0)
