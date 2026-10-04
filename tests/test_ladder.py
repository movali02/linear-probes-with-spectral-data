"""Ladder strings, activation extraction and probes (the degradation figure)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from probe2circuit import ladder as L
from probe2circuit import probes as P
from probe2circuit.io import load_config

ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_read_calibration(tmp_path):
    p = tmp_path / "cal.txt"
    p.write_text("frequency scale (your DFT -> QM9S): 0.967  median |error| 3.0 cm-1 over 41 matched strong bands\n"
                 "intensity convention activity : Spearman rho with QM9S = 0.812 over 60 bands\n"
                 "intensity convention int785   : Spearman rho with QM9S = 0.655 over 60 bands\n")
    assert L.read_calibration(p) == (0.967, "activity")
    assert L.read_calibration(tmp_path / "missing.txt") == (None, None)


def test_logreg_and_mass_mean_find_a_planted_direction():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 40)).astype(np.float32)
    Y = np.stack([(X[:, 3] + 0.3 * rng.normal(size=600) > 0), (X[:, 7] > 1.2)], 1).astype(np.float32)  # 2nd is rare
    (Xs,), _ = P.standardise(X)
    W, b = P.fit_logreg(Xs, Y, C=0.1)
    assert np.all(P.auroc_cols(Y, Xs @ W + b) > 0.95)
    assert np.argmax(np.abs(W[:, 0])) == 3 and np.argmax(np.abs(W[:, 1])) == 7
    Wm, bm = P.mass_mean(Xs, Y)
    assert np.all(P.auroc_cols(Y, Xs @ Wm + bm) > 0.9)
    S, Ya, names = P.by_analyte(Xs @ W + b, Y, np.repeat([f"a{i}" for i in range(60)], 10))
    assert S.shape == (60, 2) and len(names) == 60


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "tests"))
    from synth_step2 import make_dft_sticks, make_qm9s, make_sers_dir, make_tiny_lm
    from probe2circuit.io import find_sers_files, load_number_smiles, write_jsonl
    from probe2circuit.labels import add_labels
    from probe2circuit.pipeline import assign_splits, build_qm9s_records, build_sers_records, load_analytes
    root = tmp_path_factory.mktemp("ladder")
    cfg = load_config(ROOT / "configs" / "string_v1.yaml", "matched")
    cfg["substrate"]["generic_cb"]["path"] = str(root / "none.txt")
    analytes = load_analytes(ROOT / "data" / "analytes.csv")
    proc = root / "processed_matched"
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
    lm = make_tiny_lm(root / "tiny_lm")
    return root, proc, sticks, lm


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_ladder_extract_probe_end_to_end(world):
    root, proc, sticks, lm = world
    lad, rep, acts = root / "ladder", root / "rep", root / "acts"
    out = _run(ROOT / "scripts" / "build_ladder.py", "--variant", "matched", "--data_dir", proc, "--sticks_dir", sticks,
               "--out_dir", lad, "--report_dir", rep, "--qm9s_frac", "1.0", "--enrich", "0",
               "--calibration", root / "none.txt", "--dft_scale", "0.98")
    recs = [json.loads(l) for l in open(lad / "ladder_records.jsonl")]
    rungs = {r["rung"] for r in recs}
    assert {"R1", "R1q", "R2", "R3", "R4", "R5", "R6", "R8", "R9"} <= rungs
    r4 = [r for r in recs if r["rung"] == "R4"]
    assert len(r4) == 20 and {r["analyte"] for r in r4} >= {"L-alanine", "L-tryptophan"}
    trp = next(r for r in r4 if r["analyte"] == "L-tryptophan")
    assert trp["labels"]["fg_indole"] == 1 and trp["labels"]["fg_carboxylic_acid"] == 1      # neutral parent's labels
    assert not any(r["analyte"] is None for r in recs)                                       # controls / CB left out
    assert (rep / "similarity_summary.csv").exists() and "DFT frequency scale 0.980" in out

    _run(ROOT / "scripts" / "extract_activations.py", "--model", lm, "--variant", "matched", "--ladder_dir", lad,
         "--out_dir", acts / "tiny", "--batch_size", "32")
    from probe2circuit.activations import Acts, build_prompt
    from transformers import AutoModelForCausalLM, AutoTokenizer
    A = Acts(acts / "tiny" / "ladder")
    assert A.layers == [0, 1, 2] and A.meta["chat_template"] and len(A.ids) == len([r for r in recs if r["text"]])
    # one prompt recomputed alone must give the same vectors (batch of 1: no padding at all)
    tok, model = AutoTokenizer.from_pretrained(lm), AutoModelForCausalLM.from_pretrained(lm).eval()
    cfg = load_config(ROOT / "configs" / "string_v1.yaml", "matched")
    rec = max(recs, key=lambda r: len(r["text"]))             # the longest prompt is never padded in its batch
    s, (b0, b1) = build_prompt(rec["text"], cfg, tok)
    assert s[b0:b1] == rec["text"] and s.startswith("<im_start>user")
    enc = tok(s, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0]
    with torch.no_grad():
        hs = model(**enc, output_hidden_states=True).hidden_states
    m = (off[:, 0] >= b0) & (off[:, 1] <= b1)
    for k in range(3):
        assert np.allclose(A.get("last", k, [rec["id"]])[0], hs[k][0, -1].numpy(), atol=2e-2)
        assert np.allclose(A.get("mean", k, [rec["id"]])[0], hs[k][0, m].mean(0).numpy(), atol=2e-2)

    out = _run(ROOT / "scripts" / "run_probes.py", "--variant", "matched", "--ladder_dir", lad, "--acts_dir",
               acts / "tiny", "--out_dir", rep / "tiny", "--min_pos", "10", "--n_boot", "50")
    for f in ("degradation.csv", "degradation.png", "degradation_centred.png", "layers.csv", "layers.png",
              "per_analyte.csv", "ladder.md", "run.json"):
        assert (rep / "tiny" / f).exists(), f
    deg = list(csv.DictReader(open(rep / "tiny" / "degradation.csv")))
    assert {r["method"] for r in deg} == {"probe_lr", "probe_mm", "baseline_lr"}
    assert {r["rung"] for r in deg} == {"R0", "R2", "R3", "R4", "R5", "R6", "R8", "R9"}
    # the synthetic DFT keeps QM9S's group bands, the synthetic SERS does not: the classical curve must fall
    base = {r["rung"]: float(r["auroc"]) for r in deg if r["label"] == "MEAN" and r["method"] == "baseline_lr"
            and r["centred"] in ("0", 0)}
    assert base["R0"] > 0.85 and base["R2"] > base["R9"] + 0.1
    run = json.loads((rep / "tiny" / "run.json").read_text())
    assert len(run["paired_analytes"]) == 20 and 0.3 < run["controls"]["shuffled_labels_val_auroc"] < 0.7
    assert "R1_agreement_baseline_lr" in run["controls"]


def test_probes_run_without_rdkit():
    """The GPU environment may not have RDKit: extraction and probes must import without it."""
    code = ("import sys; sys.modules['rdkit'] = None; sys.path.insert(0, 'src'); "
            "import importlib.util as u; "
            "[u.spec_from_file_location('m', p).loader.exec_module(u.module_from_spec(u.spec_from_file_location('m', p))) "
            "for p in ('scripts/run_probes.py', 'scripts/extract_activations.py')]")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
