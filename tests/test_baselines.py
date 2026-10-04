"""Step 2: baselines building blocks and a small end-to-end run of T1-T4."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from probe2circuit import baseline_tasks as T
from probe2circuit import baselines as B
from probe2circuit.io import apply_variant, load_config
from probe2circuit.pipeline import build_qm9s_records, build_sers_records, in_subsample, load_analytes, save_spectra
from probe2circuit.strings import to_string

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ metrics
def test_fast_metrics_match_sklearn():
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score)
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        s = np.round(rng.random(60), 1)            # ties on purpose
        assert B.fast_auroc(y, s) == pytest.approx(roc_auc_score(y, s))
        assert B.fast_ap(y, s) == pytest.approx(average_precision_score(y, s))
        yc = rng.choice(list("abcd"), 50)
        pc = rng.choice(list("abcde"), 50)
        m = B.fast_multiclass(yc, pc)
        assert m["bal_acc"] == pytest.approx(balanced_accuracy_score(yc, pc))
        assert m["macro_f1"] == pytest.approx(f1_score(yc, pc, average="macro", labels=np.unique(yc)))
    assert np.isnan(B.fast_auroc([1, 1, 1], [0.2, 0.3, 0.4]))


def test_permute_by_group_keeps_groups_and_strata():
    y = np.array(list("aaabbbcccddd"))
    g = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    strata = np.array(["x"] * 6 + ["y"] * 6)
    for seed in range(10):
        yp = B.permute_by_group(y, g, strata, np.random.default_rng(seed))
        for k in range(4):                              # one label per group
            assert len(set(yp[g == k])) == 1
        assert set(yp[strata == "x"]) == {"a", "b"}     # labels stay in their stratum
        assert set(yp[strata == "y"]) == {"c", "d"}
    with pytest.raises(ValueError):
        B.permute_by_group(np.array([0, 1]), np.array([0, 0]))


def test_grouped_bootstrap_resamples_groups():
    y = np.r_[np.ones(10), np.zeros(10)].astype(int)
    s = np.r_[np.full(10, 0.9), np.full(10, 0.1)]
    fn = lambda i: {"auroc": B.fast_auroc(y[i], s[i])}          # noqa: E731
    ci = B.bootstrap_all(fn, np.r_[np.zeros(10), np.ones(10)], 200)
    # only two groups: half the resamples have one class -> NaN, the rest are perfect
    assert ci["auroc"][0] == ci["auroc"][1] == 1.0 and ci["auroc"][2] < 200


def test_inner_cv_needs_two_groups_per_class():
    y = np.array(["a"] * 4 + ["b"] * 4)
    assert B.inner_splits(y, np.array([0] * 4 + [1] * 4)) is None           # one culture per class
    folds = B.inner_splits(y, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
    assert folds and all(len(set(y[tr])) == 2 for tr, _ in folds)


def test_threshold_maximises_balanced_accuracy():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.8])
    assert B.best_threshold(y, s) == pytest.approx(0.4)


def test_scaffold_split_keeps_scaffolds_apart():
    smi = ["CCO", "CCN", "CCC", "c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "C1CC1O", "C1CC1N", "c1ccncc1", "O=C1CCC1"]
    is_test, scaf = B.scaffold_split(smi, 0.3)
    for sc in set(scaf):
        sides = {bool(t) for t, s in zip(is_test, scaf) if s == sc}
        assert len(sides) == 1
    assert not any(t for t, s in zip(is_test, scaf) if s == "")               # acyclic -> train


# ------------------------------------------------------------------ features
def test_features(cfg):
    peaks = [{"position": 1000.0, "intensity": 0.8, "width": 12.0}, {"position": 1003.0, "intensity": 0.5, "width": 20}]
    rec = {"id": "x", "text": to_string(peaks, cfg)}
    f1 = B.feature_matrix([rec], "F1", cfg)[0]
    fp = B.feature_matrix([rec], "F1pos", cfg)[0]
    f2 = B.feature_matrix([rec], "F2", cfg)[0]
    assert f1.shape == (125,) and f1.max() == pytest.approx(0.8) and fp.max() == 1.0
    assert f2.shape == (1251,) and abs(np.argmax(f2) + 500 - 1000) <= 2
    empty = B.feature_matrix([{"id": "e", "text": ""}], "F2", cfg)
    assert not empty.any()


def test_variants(cfg):
    m = apply_variant(cfg, "matched")
    assert m["sim_extra_fwhm"] == 13.0 and m["substrate"]["reference_norm"]["rescale_to_max_after_discard"]
    assert m["schema"] == cfg["schema"] + "-matched" and cfg["sim_extra_fwhm"] == 0.0     # original untouched
    assert apply_variant(cfg, "native")["schema"] == cfg["schema"]
    with pytest.raises(ValueError):
        apply_variant(cfg, "nope")


def test_in_subsample_is_deterministic():
    ids = [f"qm9s_{i}" for i in range(20000)]
    a = [i for i in ids if in_subsample(i, 0.2)]
    assert a == [i for i in ids if in_subsample(i, 0.2)]
    assert 0.18 < len(a) / len(ids) < 0.22 and all(in_subsample(i, 1.0) for i in ids[:5])


# ------------------------------------------------------------------ signal vs no signal
@pytest.mark.parametrize("signal", [True, False])
def test_run_folds_finds_signal_and_only_signal(cfg, signal):
    from conftest import synth_records
    recs = synth_records(cfg, n_classes=5, per_class=12, signal=signal, seed=1)
    X = B.feature_matrix(recs, "F1", cfg)
    y = np.array([r["analyte"] for r in recs], dtype=object)
    half = np.array([int(r["id"].split("_")[1]) % 2 for r in recs])      # two "strains"
    g = np.array([f"{r['analyte']}_{h}" for r, h in zip(recs, half)], dtype=object)
    folds = [(f"test={h}", np.where(half != h)[0], np.where(half == h)[0]) for h in (0, 1)]
    res = B.run_folds(X, y, g, X, folds, "lr", False)
    ba = B.fast_multiclass(y, res["score"])["bal_acc"]
    assert (ba > 0.8) if signal else (ba < 0.5)
    assert {f["hp_source"] for f in res["folds"]} == {"default"}       # one group per class per fold


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def synth_data(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "tests"))
    from synth_step2 import make_qm9s, make_sers_dir
    root = tmp_path_factory.mktemp("step2")
    cfg = load_config(ROOT / "configs" / "string_v1.yaml")
    cfg["substrate"]["generic_cb"]["path"] = str(root / "none.txt")
    analytes = load_analytes(ROOT / "data" / "analytes.csv")
    have = {x["name"] for x in analytes}
    for name, code, smi in (("8-hydroxyquinoline", "8hq", "Oc1cccc2cccnc12"),       # in case your csv
                            ("isatin", "isatin", "O=C1Nc2ccccc2C1=O"),               # predates them
                            ("tryptophol", "tryptophol", "OCCc1c[nH]c2ccccc12")):
        if name not in have:
            analytes.append({"name": name, "code": code, "smiles": smi, "formula": "", "source": "test"})
    sers = make_sers_dir(root, n_scans=3)
    csv_p, map_p = make_qm9s(root, n=300)
    out = root / "processed"
    from probe2circuit.io import find_sers_files, load_number_smiles, write_jsonl
    from probe2circuit.labels import add_labels
    from probe2circuit.pipeline import assign_splits
    spectra = {}
    recs, _ = build_sers_records(find_sers_files(sers), cfg, analytes, spectra_out=spectra)
    add_labels(recs)
    assign_splits([r for r in recs if r["kind"] != "substrate_only"], cfg)
    for r in recs:
        r.setdefault("split", "control")
    write_jsonl(recs, out / "sers_records.jsonl")
    from probe2circuit.preprocess import make_grid
    save_spectra(out / "sers_spectra.npz", make_grid(cfg), spectra, cfg)
    qs = {}
    q, _ = build_qm9s_records(csv_p, load_number_smiles(map_p), cfg, spectra_out=qs, spectra_frac=1.0)
    add_labels(q)
    assign_splits(q, {**cfg, "split": {**cfg["split"], "group_by": "id"}}, label_key="source")
    write_jsonl(q, out / "qm9s_records.jsonl")
    save_spectra(out / "qm9s_spectra.npz", make_grid(cfg), qs, cfg)
    return cfg, out, root


def _ctx(cfg, out, **kw):
    base = dict(cfg=cfg, data_dir=out, out_dir=out, features=("F1", "F3"), models=("majority", "lr"),
                n_boot=50, n_perm=10, min_perm=5, n_jobs=1, qm9s_frac=1.0, min_pos=10, min_pos_test=3,
                log=lambda *a: None)
    base.update(kw)
    return T.Ctx(**base)


def _get(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def test_T1_end_to_end(synth_data):
    cfg, out, _ = synth_data
    res = T.task_T1(_ctx(cfg, out, models=("majority", "lr", "ncm")))
    rows = res["rows"]
    loso = _get(rows, scheme="loso", test_set="all", metric="bal_acc", feature="F1", model="lr")[0]
    maj = _get(rows, scheme="loso", test_set="all", metric="bal_acc", feature="F1", model="majority")[0]
    assert loso["value"] > 0.8 and loso["ci_lo"] > loso["chance"] and loso["p_perm"] != ""
    assert maj["value"] == pytest.approx(0.05)
    feats = {r["feature"] for r in rows}
    assert {"F1", "F3", "F1.bg", "F3.bg", "F3.d1", "F3.snv", "F3.bg.d1", "F1.ctrl", "F3.ctrl"} <= feats   # 2b variants
    ctl = _get(rows, scheme="loso", test_set="all", metric="bal_acc", feature="F3.ctrl", model="ncm")[0]
    assert ctl["value"] > 0.8                                # each strain's control culture subtracted
    cult = _get(rows, scheme="loso", test_set="all@culture", metric="bal_acc", feature="F3.bg", model="ncm")[0]
    assert cult["n_test"] == 40 and cult["value"] > 0.8                                       # one row per culture
    assert {r["scheme"] for r in rows} >= {"loso", "step1_split", "cells_to_standards"}
    assert {r["check"] for r in res["confound_rows"]} == {"metadata_only", "substrate_only"}
    sub = _get(res["confound_rows"], scheme="loso:CB->CB", metric="bal_acc", model="lr", test_set="all")
    assert sub and all(r["value"] < 0.3 for r in sub)       # substrate-only spectra carry no analyte
    det = {d["analyte"]: d for d in res["extra"]["detectability"]}
    assert len(det) == 20 and all(d["r_cross"] > 0.5 and d["rank_BW_in_536"] == 1 for d in det.values())
    assert all("r_cross_ctrl" in d and "r_std_BW" in d for d in det.values())               # control + standards used


def test_strain_background_needs_bg(tmp_path):
    """A strain-specific background kills leave-one-strain-out; '.bg' recovers it."""
    sys.path.insert(0, str(ROOT / "tests"))
    from synth_step2 import make_sers_dir
    from probe2circuit.io import find_sers_files
    cfg = load_config(ROOT / "configs" / "string_v1.yaml")
    cfg["substrate"]["generic_cb"]["path"] = str(tmp_path / "none.txt")
    sers = make_sers_dir(tmp_path, n_scans=3, signal=0.15, strain_bg=1.5, standards=False)
    spectra = {}
    recs, _ = build_sers_records(find_sers_files(sers), cfg, load_analytes(ROOT / "data" / "analytes.csv"),
                                 spectra_out=spectra)
    cells = [r for r in recs if r["kind"] == "cell" and r["analyte"]]
    F3 = np.stack([spectra[r["id"]][0] for r in cells])
    y = np.array([r["analyte"] for r in cells], dtype=object)
    g = np.array([r["group"] for r in cells], dtype=object)
    st = np.array([T.strain_of(r["id"]) for r in cells])
    folds = [(s, np.where(st != s)[0], np.where(st == s)[0]) for s in sorted(set(st))]
    raw = B.fast_multiclass(y, B.run_folds(F3, y, g, F3, folds, "ncm", False)["score"])["bal_acc"]
    Fb = B.center_by_domain(F3, st, g)
    bg = B.fast_multiclass(y, B.run_folds(Fb, y, g, Fb, folds, "ncm", False)["score"])["bal_acc"]
    assert raw < 0.3 and bg > 0.8
    # the same with each strain's own no-analyte control culture as the background
    ctrl = [r for r in recs if r["kind"] == "cell" and not r["analyte"]]
    assert {r["group"] for r in ctrl} == {"Cell_BW_ctrl", "Cell_536_ctrl"}
    Fc = F3.copy()
    for s in set(st):
        Fc[st == s] -= np.mean([spectra[r["id"]][0] for r in ctrl if T.strain_of(r["id"]) == s], axis=0)
    ct = B.fast_multiclass(y, B.run_folds(Fc, y, g, Fc, folds, "ncm", False)["score"])["bal_acc"]
    assert ct > 0.8


def test_leave_pair_out_is_unbiased_for_a_constant_model():
    rng = np.random.default_rng(0)
    ak = np.repeat([f"a{i:02d}" for i in range(10)], 6)
    y = np.array([int(a < "a03") for a in ak])
    X = rng.normal(size=(60, 8))
    r = B.leave_pair_out(X, y, ak, "majority")
    assert np.nanmean(r["W"]) == 0.5 and r["W"].shape == (3, 7)      # pooled leave-one-out gives 0.0 here
    X[y == 1, 0] += 3
    r = B.leave_pair_out(X, y, ak, "lr", tune=False)
    assert np.nanmean(r["W"]) == 1.0 and B.lpo_ci(r["W"], 50) == (1.0, 1.0)


def test_preps_and_new_models(cfg):
    X = np.abs(np.random.default_rng(0).normal(size=(12, 1251)))
    d = np.array(["a"] * 6 + ["b"] * 6)
    Xc = B.center_by_domain(X, d, np.repeat([0, 1, 2, 3, 4, 5], 2))
    assert np.allclose(Xc[d == "a"].mean(0), 0, atol=1e-9) and np.allclose(Xc[d == "b"].mean(0), 0, atol=1e-9)
    assert B.apply_preps(X, ("lt1700",), "F3", cfg).shape[1] == 1200
    assert B.apply_preps(X[:, :125], ("lt1700",), "F1", cfg).shape[1] == 120
    assert np.allclose(B.prep_snv(X).std(1), 1)
    y = np.repeat(list("abc"), 4)
    Xs = X.copy()
    for k, c in enumerate("abc"):
        Xs[y == c, 100 * k] += 5
    for m in ("ncm", "plsda", "pca_lr"):
        mod = B._fit(m, B.DEFAULT_HP[m], Xs, y, 0)
        assert (mod.predict(Xs) == y).mean() == 1.0 and mod.predict_proba(Xs).shape == (12, 3)


def test_T2_T3_T4_end_to_end(synth_data):
    cfg, out, _ = synth_data
    r2 = T.task_T2(_ctx(cfg, out, labels=("fg_amide", "fg_phenol", "fg_thiol", "fg_carboxylic_acid", "fg_indole"),
                        models=("majority", "knn"), features=("F1",)))
    tab = {(t["matrix"], t["label"]): t for t in r2["extra"]["labels"]}
    assert tab[("all", "fg_amide")]["testable"] and not tab[("all", "fg_thiol")]["testable"]      # Cys only
    assert tab[("all", "fg_phenol")]["testable"] and not tab[("cells", "fg_phenol")]["testable"]  # Tyr only in cells
    assert not tab[("cells", "fg_carboxylic_acid")]["testable"]                    # every cell analyte has one
    assert {r["scheme"] for r in r2["rows"]} >= {"lpo:all", "lpo:cells", "lpo:standards"}
    maj = _get(r2["rows"], scheme="lpo:all", metric="auroc", model="majority", feature="F1")
    assert maj and all(r["value"] == 0.5 for r in maj)                             # no pooling bias
    # carboxylic acid = amino acids (cells) vs indole standards: the matrix alone explains it
    mm = _get(r2["rows"], scheme="lpo:all", label="fg_carboxylic_acid", feature="META_matrix", metric="auroc")[0]
    assert mm["value"] > 0.9
    assert "indole+indole-d6" in {p["analyte"] for p in r2["extra"]["per_analyte"]}      # held out together
    s2c = _get(r2["rows"], scheme="std_to_cells", label="fg_indole", metric="auroc", test_set="all@analyte")
    assert s2c                                                    # indole standards -> tryptophan cells

    r3 = T.task_T3(_ctx(cfg, out, labels=("fg_aromatic_ring", "fg_hydroxyl_aliphatic")))
    auc = _get(r3["rows"], scheme="random", metric="auroc", feature="F1", model="lr")
    assert auc and all(r["value"] > 0.8 for r in auc)
    assert _get(r3["rows"], scheme="scaffold", metric="auroc")
    assert {"F1.sqrt", "F3.lt1700"} <= {r["feature"] for r in r3["rows"]}

    r4 = T.task_T4(_ctx(cfg, out, min_pos=3, labels=("fg_aromatic_ring", "fg_carboxylic_acid", "fg_imidazole_ring")))
    tab = {t["label"]: t for t in r4["extra"]["labels"]}
    assert not tab["fg_carboxylic_acid"]["testable"]            # no QM9S support
    assert tab["fg_imidazole_ring"]["low_support"]              # His only
    ts = {r["test_set"] for r in r4["rows"]}
    assert {"qm9s_test", "sers_all", "sers_cells@analyte"} <= ts and "qm9s_test@analyte" not in ts
    assert _get(r4["rows"], metric="ece", test_set="sers_all")
    assert {"F1.bg", "F3.bg"} <= {r["feature"] for r in r4["rows"]}
    his = [p for p in r4["extra"]["per_analyte"] if p["label"] == "fg_imidazole_ring" and p["analyte"] == "L-histidine"]
    assert his and all(1 <= p["rank"] <= p["n_analytes"] for p in his)
    p = [r for r in r4["rows"] if r["p_perm"] != ""]
    assert p and all(r["test_set"] == "sers_all" for r in p)


def test_script_writes_reports(synth_data):
    cfg, out, root = synth_data
    rep = root / "rep"
    cmd = [sys.executable, str(ROOT / "scripts" / "run_baselines.py"), "--task", "T1", "--data_dir", str(out),
           "--out_dir", str(rep), "--quick", "--n_jobs", "1", "--features", "F1,F3", "--models", "majority,lr",
           "--no_figures"]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    for f in ("T1.csv", "T1.md", "T1_confounds.csv", "T1_per_class.csv", "T1_run.json", "T1_predictions.csv.gz",
              "T1_detectability.csv"):
        assert (rep / f).exists(), f
    run = json.loads((rep / "T1_run.json").read_text())
    assert run["inputs"] and run["n_rows"] > 0
    md = (rep / "T1.md").read_text()
    assert "## Gate" in md and "## Detectability" in md
