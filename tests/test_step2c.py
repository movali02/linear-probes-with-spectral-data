"""Kit 2c: axis alignment, frequency correction, soft imaginary modes, reduced-field prompts."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from probe2circuit.io import find_sers_files, load_config, load_number_smiles
from probe2circuit.pipeline import build_qm9s_records, build_sers_records, load_analytes
from probe2circuit.strings import parse_string

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "string_v1.yaml"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "dft"))


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2500:] + r.stderr[-2500:]
    return r.stdout


def test_variants():
    for v, fs in (("v2", None), ("v2m", None), ("v2mf", 1.0134)):
        c = load_config(CFG, v)
        assert c["window"] == [510.0, 1800.0] and c["schema"].endswith("-" + v)
        rn = c["substrate"]["reference_norm"]
        assert rn["ref_tol"] == 8.0 and rn["align_axis"] is True and c["substrate"]["generic_cb"]["path"] is None
        assert c.get("sim_freq_scale") == fs
    assert load_config(CFG, "matched")["window"] == [500.0, 1750.0]          # old variants untouched
    assert "align_axis" not in load_config(CFG, "native")["substrate"]["reference_norm"]


def _shifted_sers(root, shift):
    """One culture whose whole axis sits `shift` cm-1 off: CB band at 829 + shift, analyte band at 1200 + shift."""
    from synth_step2 import CB, _lor, _write
    rng = np.random.default_rng(0)
    x = np.linspace(300, 1900, 1601)
    base = 400 + 0.3 * (x - 300)
    cb = [(p + shift, h, w) for p, h, w in CB]
    d = Path(root) / "raw"
    for n in (1, 2, 3):
        _write(d / f"Cell_BW_ala_{n}.txt", x, 3000 * (_lor(x, cb) + _lor(x, [(1200 + shift, 0.8, 16.0)])) + base
               + rng.normal(0, 20, len(x)))
        _write(d / f"CB_BW_ala_{n}.txt", x, 3000 * _lor(x, cb) + base + rng.normal(0, 20, len(x)))
    return d


def test_axis_alignment_recovers_a_shifted_session(tmp_path):
    analytes = load_analytes(ROOT / "data" / "analytes.csv")
    paths = find_sers_files(_shifted_sers(tmp_path, -6.0))
    # old pipeline: the band at 823 is outside 829 +/- 4, so the spectrum cannot be put on the 829 scale
    old = load_config(CFG, "native")
    old["substrate"]["generic_cb"]["path"] = None
    recs, _ = build_sers_records(paths, old, analytes)
    assert all(r["substrate"]["mode_used"].startswith("max_in_window") for r in recs)
    # v2: found, shifted by +6, and the analyte band lands at 1200 (not 1194); its own CB reference is used
    recs, info = build_sers_records(paths, load_config(CFG, "v2"), analytes)
    cell = [r for r in recs if r["kind"] == "cell"]
    assert len(cell) == 3
    for r in cell:
        s = r["substrate"]
        assert s["mode_used"] == "reference_norm" and abs(s["axis_shift_cm"] - 6.0) <= 1.0
        assert abs(s["ref_pos_cm"] - 829.0) <= 1.0 and s["cb_ref"].startswith("mean(")
        pos = [p["position"] for p in parse_string(r["text"], load_config(CFG, "v2"))]
        assert any(abs(p - 1200) <= 2 for p in pos) and not any(abs(p - 829) <= 4 for p in pos)
    assert info["axis_shift_cm"]["n"] == 6 and abs(info["axis_shift_cm"]["median"] - 6.0) <= 1.0


def test_frequency_correction_moves_qm9s_and_dft(tmp_path):
    from synth_step2 import make_qm9s
    from probe2circuit import preprocess as pp
    csv_p, map_p = make_qm9s(tmp_path, n=6)
    n2s = load_number_smiles(map_p)
    a, _ = build_qm9s_records(csv_p, n2s, load_config(CFG, "v2m"))
    b, _ = build_qm9s_records(csv_p, n2s, load_config(CFG, "v2mf"))
    cfg = load_config(CFG, "v2m")
    for ra, rb in zip(a, b):
        pa = max(parse_string(ra["text"], cfg), key=lambda p: p["intensity"])["position"]
        pb = max(parse_string(rb["text"], cfg), key=lambda p: p["intensity"])["position"]
        assert abs(pb / pa - 1.0134) < 0.003, (pa, pb)
    # DFT sticks: dft.scale x sim_freq_scale
    c = load_config(CFG, "v2mf")
    c["dft"]["scale"] = 0.967
    grid, y = pp.prepare_dft_sticks(np.array([1000.0]), np.array([1.0]), c)
    assert abs(grid[np.argmax(y)] - 1000 * 0.967 * 1.0134) <= 1.0


def test_soft_imaginary_mode_is_accepted_and_flagged(tmp_path):
    pytest.importorskip("rdkit")
    import dft_check
    from test_dft import write_job
    gly = "NCC(=O)O"
    wd = write_job(tmp_path, gly, f6="-32.60", imag=" ***imaginary mode***")
    (wd / "smiles.txt").write_text(f"{gly}\n0\n")
    r = dft_check.check_job(wd, "job", gly, 0)
    assert r["verdict"] == "FAIL" and r["detail"] == "1 imaginary mode(s)" and r["imag_freqs"] == [-32.6]
    args = [ROOT / "dft" / "dft_to_sticks.py", "--root", tmp_path, "--out_dir", tmp_path / "sticks"]
    assert "skip job" in _run(*args)                                         # default: still refused
    assert not (tmp_path / "sticks" / "activity" / "job.txt").exists()
    assert "skip job" in _run(*args, "--soft_imag", "20")                    # 32.6 is not below 20
    out = _run(*args, "--soft_imag", "50")
    assert "accept job" in out
    st = np.loadtxt(tmp_path / "sticks" / "activity" / "job.txt")
    assert st.shape[0] == 3 and (st[:, 0] > 0).all()
    assert "job,-32.6" in (tmp_path / "sticks" / "soft_imag.csv").read_text()


def test_reduced_fields_prompt():
    from probe2circuit.activations import build_prompt, reduce_fields, template_for
    cfg = load_config(CFG, "v2mf")
    text = "532 0.11 9 | 746 1.00 20 | 1610 0.47 46"
    assert reduce_fields(text, cfg, "all") == text
    assert reduce_fields(text, cfg, "posint") == "532 0.11 | 746 1.00 | 1610 0.47"
    assert reduce_fields(text, cfg, "pos") == "532 | 746 | 1610"
    t = template_for(cfg, "pos")
    assert "(position cm-1)" in t and "intensity" not in t and "FWHM" not in t
    assert "relative intensity" in template_for(cfg, "posint") and "FWHM" not in template_for(cfg, "posint")
    s, (b0, b1) = build_prompt(text, cfg, None, chat=False, fields="pos")
    assert s[b0:b1] == "532 | 746 | 1610" and "0.11" not in s


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from synth_step2 import make_dft_sticks, make_qm9s, make_sers_dir, make_tiny_lm
    from probe2circuit.io import write_jsonl
    from probe2circuit.labels import add_labels
    from probe2circuit.pipeline import assign_splits
    root = tmp_path_factory.mktemp("step2c")
    cfg = load_config(CFG, "v2mf")
    analytes = load_analytes(ROOT / "data" / "analytes.csv")
    proc = root / "processed_v2mf"
    recs, _ = build_sers_records(find_sers_files(make_sers_dir(root, n_scans=2)), cfg, analytes)
    add_labels(recs)
    assign_splits([r for r in recs if r["kind"] != "substrate_only"], cfg)
    write_jsonl(recs, proc / "sers_records.jsonl")
    csv_p, map_p = make_qm9s(root, n=600)
    q, _ = build_qm9s_records(csv_p, load_number_smiles(map_p), cfg)
    add_labels(q)
    assign_splits(q, {**cfg, "split": {**cfg["split"], "group_by": "id"}}, label_key="source")
    write_jsonl(q, proc / "qm9s_records.jsonl")
    sticks = make_dft_sticks(root, ROOT / "dft" / "molecules.csv", ROOT / "data" / "analytes.csv")
    return root, proc, sticks, make_tiny_lm(root / "tiny_lm")


def test_step2c_ladder_end_to_end(world):
    root, proc, sticks, lm = world
    lad, rep, acts = root / "ladder", root / "rep", root / "acts"
    out = _run(ROOT / "scripts" / "build_ladder.py", "--variant", "v2mf", "--data_dir", proc, "--sticks_dir", sticks,
               "--out_dir", lad, "--report_dir", rep, "--qm9s_frac", "1.0", "--enrich", "0",
               "--calibration", root / "none.txt", "--dft_scale", "0.98")
    assert "simulated frequencies x 1.0134" in out and "what the strings look like" in out
    st = {r["rung"]: r for r in csv.DictReader(open(rep / "string_stats.csv"))}
    assert {"R0 (QM9S)", "R2", "R4", "R5", "R6", "R8", "R9"} <= set(st) and float(st["R4"]["peaks_median"]) >= 3

    # classical curve on positions only
    _run(ROOT / "scripts" / "run_probes.py", "--variant", "v2mf", "--ladder_dir", lad, "--baseline_only",
         "--baseline_feature", "F1pos", "--out_dir", rep / "baseline-F1pos", "--min_pos", "10", "--n_boot", "30")
    deg = list(csv.DictReader(open(rep / "baseline-F1pos" / "degradation.csv")))
    assert {r["method"] for r in deg} == {"baseline_lr"}

    # positions-only prompts: own tag, the model never sees an intensity
    for extra in ((), ("--random_init",)):
        _run(ROOT / "scripts" / "extract_activations.py", "--model", lm, "--variant", "v2mf", "--ladder_dir", lad,
             "--out_dir", acts / ("tiny" + ("-random" if extra else "") + "-pos"), "--batch_size", "32",
             "--fields", "pos", *extra)
    meta = json.loads((acts / "tiny-pos" / "ladder" / "meta.json").read_text())
    assert meta["fields"] == "pos" and " | " in meta["example_span"] and "." not in meta["example_span"]
    assert "(position cm-1)" in meta["example_prompt"] and meta["text_hash"]
    # a rerun is skipped; the same folder asked for with other fields is NOT taken as finished
    again = _run(ROOT / "scripts" / "extract_activations.py", "--model", lm, "--variant", "v2mf", "--ladder_dir", lad,
                 "--out_dir", acts / "tiny-pos", "--batch_size", "32", "--fields", "pos")
    assert again.count("already done") == 2
    other = _run(ROOT / "scripts" / "extract_activations.py", "--model", lm, "--variant", "v2mf", "--ladder_dir", lad,
                 "--out_dir", acts / "tiny-pos", "--batch_size", "32", "--fields", "posint", "--sets", "ladder")
    assert "already done" not in other
    _run(ROOT / "scripts" / "extract_activations.py", "--model", lm, "--variant", "v2mf", "--ladder_dir", lad,
         "--out_dir", acts / "tiny-pos", "--batch_size", "32", "--fields", "pos", "--sets", "ladder")

    out = _run(ROOT / "scripts" / "run_probes.py", "--variant", "v2mf", "--ladder_dir", lad, "--acts_dir",
               acts / "tiny-pos", "--acts_random", acts / "tiny-random-pos", "--baseline_feature", "F1pos",
               "--out_dir", rep / "tiny-pos", "--min_pos", "10", "--n_boot", "40")
    d = list(csv.DictReader(open(rep / "tiny-pos" / "paired_differences.csv")))
    assert {(r["a"], r["b"]) for r in d} == {("probe_lr", "baseline_lr"), ("probe_lr", "random_lr"),
                                             ("probe_lr", "probe_mm")}
    assert {r["rung"] for r in d} == {"R2", "R3", "R4", "R5", "R6", "R8", "R9"}
    md = (rep / "tiny-pos" / "ladder.md").read_text()
    assert "Paired differences" in md and "one positive analyte" in md and "classifier on the string (F1pos)" in md
    assert (rep / "tiny-pos" / "degradation.png").exists()


def test_run_script_parses():
    r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "run_step2c.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
