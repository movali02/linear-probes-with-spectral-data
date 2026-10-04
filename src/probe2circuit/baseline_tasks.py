"""Step 2 tasks (roadmap 2.1-2.5) on top of baselines.py.

  T1  SERS cells, analyte (20 classes). Schemes:
        loso               leave-one-strain-out: train BW, test 536 and the
                           reverse; the scans of a culture always stay together
        step1_split        the step-1 split field (the split the LLM uses)
        cells_to_standards train on every cell culture, test on the standards
                           (reported separately, as the roadmap asks)
      Confounds: metadata-only, substrate-only (cells -> CB and CB -> CB), and
      the old train/test split with CB rows removed (for the 58/174 comparison).
  T2  SERS cells + standards, binary structural labels, leave-one-analyte-out.
      Analytes are merged by constitution (indole and indole-d6 are one analyte).
      A label is testable only with >= 2 analytes on each side.
  T3  QM9S structural labels: step-1 random molecule split, and a Bemis-Murcko
      scaffold split.
  T4  Train on QM9S, test on SERS (cells, standards) on the shared labels;
      thresholds are chosen on QM9S only. Run once per --variant.

Everything is evaluated through Spec + run_specs, so every number gets the
same grouped bootstrap CI and the same permutation test."""
from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import baselines as B
from .io import read_jsonl
from .labels import FG_KEYS, canonical
from .pipeline import in_subsample, normalise_id

PRIMARY = {True: "auroc", False: "bal_acc"}
# models per task: the chemometrics / one-shot models only where the data are small (SERS)
TASK_MODELS = {"T1": B.MODELS, "T2": B.MODELS, "T3": ("majority", "lr", "hgb", "knn"),
               "T4": ("majority", "lr", "hgb", "knn")}
# preprocessing variants (base feature, chain); see baselines.apply_preps
TASK_PREPS = {"T1": [("F1", ("bg",)), ("F2", ("bg",)), ("F3", ("bg",)), ("F3", ("d1",)), ("F3", ("snv",)),
                     ("F3", ("bg", "d1"))],
              "T2": [("F1", ("bg",)), ("F3", ("bg",)), ("F3", ("d1",))],
              "T3": [("F1", ("sqrt",)), ("F2", ("lt1700",)), ("F3", ("lt1700",))],
              "T4": [("F1", ("bg",)), ("F3", ("bg",)), ("F1", ("sqrt",))]}


# ================================================================== context / spec
@dataclass
class Ctx:
    cfg: dict
    variant: str = "native"
    data_dir: Path = Path("data/processed")
    out_dir: Path = Path("reports/baselines/native")
    features: tuple = B.FEATURES
    models: tuple | None = None        # None -> TASK_MODELS[task]
    preps: bool = True                 # step 2b preprocessing variants (F3.bg, F3.d1, ...)
    control_regex: str = r"(?i)(?:^|_)(ctrl|control|blank|noaa|no-aa|none|neg|medium)(?:_|$)"
    n_boot: int = 1000
    n_perm: int = 1000
    perm_for: str = "primary,best"     # "primary,best" | "all" | "none"
    n_jobs: int = 1
    seed: int = 42
    qm9s_frac: float = 0.2
    min_pos: int | None = None         # QM9S training positives needed; None -> 50 (T3), 30 (T4)
    min_pos_test: int = 10
    labels: tuple | None = None
    default_hp: dict = field(default_factory=lambda: {k: dict(v) for k, v in B.DEFAULT_HP.items()})
    old_manifest: str | None = None
    old_predictions: str | None = None
    meta_csv: str | None = None
    figures: bool = True
    perm_budget_min: float = 120.0     # wall-clock minutes for all permutation tests of one task
    min_perm: int = 100                # never fewer shuffles than this (min p = 1/101)
    log: object = print


@dataclass
class Spec:
    task: str
    scheme: str
    label: str
    binary: bool
    Xtr: dict                 # feature -> matrix over the training source
    Xte: dict                 # feature -> matrix over the test source (same objects if shared)
    ytr: np.ndarray
    gtr: np.ndarray           # inner-CV groups over the training source
    yte: np.ndarray
    folds: list               # [(name, train idx, test idx)]
    subsets: dict             # name -> bool mask over the test source
    boot_groups: np.ndarray   # bootstrap unit over the test source
    perm_groups: np.ndarray   # permutation unit over the training source
    perm_strata: np.ndarray | None = None
    shared: bool = True       # train and test source are the same records
    perm_subset: str = "all"
    chance: float = 0.5
    with_ece: bool = False
    features: tuple | None = None
    models: tuple | None = None
    max_splits: int = 5
    te_ids: list | None = None
    te_analyte: np.ndarray | None = None
    note: str = ""
    agg_groups: np.ndarray | None = None   # also report metrics on one row per group (culture / analyte)
    agg_name: str = "group"
    agg_subsets: tuple | None = None       # subsets to aggregate (None = all)


# ================================================================== workers
def _limit_threads(n_jobs):
    if n_jobs == 1:
        from contextlib import nullcontext
        return nullcontext()
    from threadpoolctl import threadpool_limits
    return threadpool_limits(1)


def _metric_fn(yte, res, binary, with_ece):
    if binary:
        s = res["score"].astype(float)
        p = res["pred"]
        y = yte.astype(int)
        return lambda i: B.fast_binary(y[i], s[i], p[i], with_ece)
    pr = res["score"]
    return lambda i: B.fast_multiclass(yte[i], pr[i])


def _aggregate(yte, res, idx, groups, binary):
    """One row per group: mean score / majority vote. Returns arrays for the metric functions."""
    uniq, inv = np.unique(groups[idx].astype(str), return_inverse=True)
    n = np.bincount(inv).astype(float)
    if binary:
        s = np.bincount(inv, weights=res["score"][idx].astype(float)) / n
        y = (np.bincount(inv, weights=yte[idx].astype(float)) / n >= 0.5).astype(int)
        p = (np.bincount(inv, weights=(res["pred"][idx] == 1).astype(float)) / n >= 0.5).astype(int)
        return y, s, p
    y, p = [], []
    for k in range(len(uniq)):
        m = idx[inv == k]
        vals, counts = np.unique(res["score"][m].astype(str), return_counts=True)
        p.append(vals[np.argmax(counts)])
        y.append(str(yte[m][0]))
    return np.array(y, dtype=object), None, np.array(p, dtype=object)


def _eval_job(Xtr, Xte, ytr, gtr, folds, model, binary, seed, default_hp, max_splits, yte, subsets,
              boot_groups, n_boot, with_ece, n_jobs, agg_groups=None, agg_name="group", agg_subsets=None):
    t0 = time.time()
    with _limit_threads(n_jobs):
        res = B.run_folds(Xtr, ytr, gtr, Xte, folds, model, binary, seed, max_splits=max_splits,
                          default_hp=default_hp)
    mets = {}
    fn = _metric_fn(yte, res, binary, with_ece)
    for sname, mask in subsets.items():
        idx = np.where(mask & (res["fold"] >= 0))[0]
        if not len(idx):
            continue
        point = fn(idx)
        ci = B.bootstrap_all(lambda j, idx=idx: fn(idx[j]), boot_groups[idx], n_boot, seed)
        mets[sname] = {k: (v, *ci.get(k, (np.nan, np.nan, 0))) for k, v in point.items()}
        mets[sname]["_n"] = (len(idx), len(set(boot_groups[idx].tolist())))
        if agg_groups is not None and (agg_subsets is None or sname in agg_subsets):
            ya, sa, pa = _aggregate(yte, res, idx, agg_groups, binary)
            fa = ((lambda j: B.fast_binary(ya[j], sa[j], pa[j], False)) if binary
                  else (lambda j: B.fast_multiclass(ya[j], pa[j])))
            pt = fa(np.arange(len(ya)))
            cia = B.bootstrap_all(fa, np.arange(len(ya)), n_boot, seed)
            key = f"{sname}@{agg_name}"
            mets[key] = {k: (v, *cia.get(k, (np.nan, np.nan, 0))) for k, v in pt.items()}
            mets[key]["_n"] = (len(ya), len(ya))
    res["seconds"] = round(time.time() - t0, 2)
    return res, mets


def _perm_job(Xtr, Xte, ytr, gtr, folds, model, binary, seed, fixed, yte, perm_groups, strata, shared,
              stat_idx, seeds, n_jobs):
    out = []
    with _limit_threads(n_jobs):
        for ps in seeds:
            yp = B.permute_by_group(ytr, perm_groups, strata, np.random.default_rng(ps))
            yt = yp if shared else yte
            res = B.run_folds(Xtr, yp, gtr, Xte, folds, model, binary, seed, fixed=fixed)
            if binary:
                out.append(B.fast_auroc(yt[stat_idx].astype(int), res["score"][stat_idx].astype(float)))
            else:
                out.append(B.fast_multiclass(yt[stat_idx], res["score"][stat_idx])["bal_acc"])
    return out


def _parallel(ctx, calls):
    if ctx.n_jobs == 1 or len(calls) == 1:
        return [f(*a) for f, a in calls]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=ctx.n_jobs, verbose=0)(delayed(f)(*a) for f, a in calls)


# ================================================================== engine
def _chance(metric, spec, y_sub):
    if metric in ("auroc",):
        return 0.5
    if metric == "bal_acc":
        return 0.5 if spec.binary else spec.chance
    if metric == "auprc":
        return float(np.mean(y_sub.astype(int)))
    if metric == "accuracy":
        return B.majority_rate(y_sub)
    return np.nan


def run_specs(specs: list[Spec], ctx: Ctx, tag: str = "") -> tuple[list[dict], dict]:
    """Evaluate every spec x feature x model, add CIs and permutation p-values.
    Returns (rows, results) with results[(i, feature, model)] = (res, mets)."""
    if not specs:
        ctx.log(f"[{tag}] nothing to evaluate (no testable label/scheme); see the *_labels.csv table")
        return [], {}
    jobs = []
    for i, sp in enumerate(specs):
        for f in (sp.features or tuple(sp.Xtr)):
            if f not in sp.Xtr or f not in sp.Xte:
                continue
            for m in (sp.models or ctx.models or TASK_MODELS[sp.task]):
                jobs.append((i, f, m))
    t0 = time.time()
    ctx.log(f"[{tag}] {len(jobs)} fits ({len(specs)} label/scheme combinations) on {ctx.n_jobs} job(s)")
    calls = []
    for i, f, m in jobs:
        sp = specs[i]
        calls.append((_eval_job, (sp.Xtr[f], sp.Xte[f], sp.ytr, sp.gtr, sp.folds, m, sp.binary, ctx.seed,
                                  ctx.default_hp, sp.max_splits, sp.yte, sp.subsets, sp.boot_groups,
                                  ctx.n_boot, sp.with_ece, ctx.n_jobs, sp.agg_groups, sp.agg_name,
                                  sp.agg_subsets)))
    outs = _parallel(ctx, calls)
    results = {k: o for k, o in zip(jobs, outs)}
    ctx.log(f"[{tag}] fits + bootstrap done in {time.time() - t0:.0f} s")

    # ---- which combinations get a permutation test
    pvals = {}
    if ctx.n_perm > 0 and ctx.perm_for != "none":
        perm_jobs = []
        for i, sp in enumerate(specs):
            pm = PRIMARY[sp.binary]
            cand = {(f, m): results[(i, f, m)][1].get(sp.perm_subset, {}).get(pm, (np.nan,))[0]
                    for (j, f, m) in results if j == i and m != "majority"}
            cand = {k: v for k, v in cand.items() if np.isfinite(v)}
            if not cand:
                continue
            chosen = set(cand) if ctx.perm_for == "all" else set()
            if ctx.perm_for != "all":
                if ("F1", "lr") in cand:
                    chosen.add(("F1", "lr"))
                chosen.add(max(cand, key=cand.get))
            for f, m in sorted(chosen):
                perm_jobs.append((i, f, m))
        if perm_jobs:
            t1 = time.time()
            calls, index = [], []
            share = 60.0 * ctx.perm_budget_min * max(1, ctx.n_jobs) / len(perm_jobs)   # CPU-seconds each
            for i, f, m in perm_jobs:
                sp = specs[i]
                res = results[(i, f, m)][0]
                fixed = [(fo["hp"], fo["threshold"]) for fo in res["folds"]]
                stat_idx = np.where(sp.subsets[sp.perm_subset] & (res["fold"] >= 0))[0]
                base = 10_000_019 * (ctx.seed + 1) + 1009 * i
                # time one shuffle, then fit the number of shuffles to the budget
                tt = time.time()
                _perm_job(sp.Xtr[f], sp.Xte[f], sp.ytr, sp.gtr, sp.folds, m, sp.binary, ctx.seed, fixed, sp.yte,
                          sp.perm_groups, sp.perm_strata, sp.shared, stat_idx, [base - 1], 1)
                per = max(time.time() - tt, 1e-3)
                n_perm = int(min(ctx.n_perm, max(min(ctx.min_perm, ctx.n_perm), share / per)))
                if n_perm < ctx.n_perm:
                    ctx.log(f"[{tag}] {sp.scheme}/{sp.label} {f}/{m}: {per:.1f} s per shuffle -> {n_perm} "
                            f"shuffles to stay within --perm_budget_min {ctx.perm_budget_min:g}")
                seeds = [base + k for k in range(n_perm)]
                n_chunks = max(1, min(ctx.n_jobs, n_perm // 5 or 1))
                for c in range(n_chunks):
                    calls.append((_perm_job, (sp.Xtr[f], sp.Xte[f], sp.ytr, sp.gtr, sp.folds, m, sp.binary,
                                              ctx.seed, fixed, sp.yte, sp.perm_groups, sp.perm_strata, sp.shared,
                                              stat_idx, seeds[c::n_chunks], ctx.n_jobs)))
                    index.append((i, f, m))
            ctx.log(f"[{tag}] permutation tests: {len(perm_jobs)} combination(s), up to {ctx.n_perm} shuffles each")
            outs = _parallel(ctx, calls)
            null = {}
            for key, vals in zip(index, outs):
                null.setdefault(key, []).extend(vals)
            for (i, f, m), vals in null.items():
                sp = specs[i]
                real = results[(i, f, m)][1][sp.perm_subset][PRIMARY[sp.binary]][0]
                v = np.asarray([x for x in vals if np.isfinite(x)])
                pvals[(i, f, m)] = ((1 + int((v >= real - 1e-12).sum())) / (1 + len(v)), len(v),
                                    float(v.mean()) if len(v) else np.nan)
            ctx.log(f"[{tag}] permutations done in {time.time() - t1:.0f} s")

    # ---- rows
    rows = []
    for (i, f, m), (res, mets) in results.items():
        sp = specs[i]
        hp = json.dumps([{"fold": fo["fold"], **fo["hp"], "src": fo["hp_source"], "thr": fo["threshold"]}
                         for fo in res["folds"]])
        for sname, md in mets.items():
            n_test, n_groups = md["_n"]
            agg = "@" in sname
            idx = np.where(sp.subsets[sname.split("@")[0]] & (res["fold"] >= 0))[0]
            for metric, v in md.items():
                if metric == "_n":
                    continue
                val, lo, hi, n_ok = v
                row = {"task": sp.task, "variant": ctx.variant, "scheme": sp.scheme, "test_set": sname,
                       "label": sp.label, "feature": f, "model": m, "metric": metric,
                       "value": _r(val), "ci_lo": _r(lo), "ci_hi": _r(hi),
                       "chance": "" if agg and metric in ("auprc", "accuracy") else _r(_chance(metric, sp, sp.yte[idx])),
                       "p_perm": "", "n_perm": "", "perm_null_mean": "",
                       "n_test": n_test, "n_test_groups": n_groups, "n_boot_ok": n_ok,
                       "n_train": int(np.mean([fo["n_train"] for fo in res["folds"]])),
                       "hp_source": ",".join(sorted({fo["hp_source"] for fo in res["folds"]})),
                       "hp": hp, "seconds": res["seconds"], "note": sp.note}
                if (i, f, m) in pvals and sname == sp.perm_subset and metric == PRIMARY[sp.binary]:
                    p, n, nm = pvals[(i, f, m)]
                    row.update(p_perm=_r(p), n_perm=n, perm_null_mean=_r(nm))
                rows.append(row)
    return rows, results


def _r(v, d=4):
    try:
        return "" if v is None or not np.isfinite(v) else round(float(v), d)
    except TypeError:
        return v


def best_combo(specs, results, i, subset=None):
    sp = specs[i]
    subset = subset or sp.perm_subset
    pm = PRIMARY[sp.binary]
    cand = {(f, m): mets.get(subset, {}).get(pm, (np.nan,))[0]
            for (j, f, m), (_, mets) in results.items() if j == i and m != "majority"}
    cand = {k: v for k, v in cand.items() if np.isfinite(v)}
    return max(cand, key=cand.get) if cand else None


# ================================================================== data
def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sers(ctx):
    p = Path(ctx.data_dir) / "sers_records.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run build_strings.py --variant {ctx.variant} --sers_dir ... first")
    recs = read_jsonl(p)
    schemas = {r.get("schema") for r in recs}
    ctx.log(f"[data] {p}: {len(recs)} SERS records, schema {sorted(map(str, schemas))}")
    return recs


def load_qm9s(ctx):
    p = Path(ctx.data_dir) / "qm9s_records.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run build_strings.py --variant {ctx.variant} --qm9s_csv ... first")
    recs = [r for r in read_jsonl(p) if in_subsample(r["id"], ctx.qm9s_frac) and r.get("labels") and r.get("text")]
    ctx.log(f"[data] {p}: {len(recs)} QM9S molecules in the {ctx.qm9s_frac:.0%} subsample")
    return recs


def spectra_store(ctx, which):
    paths = [Path(ctx.data_dir) / f"{w}_spectra.npz" for w in which]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        if "F3" in ctx.features:
            ctx.log(f"[F3] WARNING: {missing} missing -> F3 skipped. Rebuild with build_strings.py --save_spectra")
        return None
    return B.SpectraStore(paths, ctx.cfg)


def feature_set(ctx, records, store, tag):
    out = {}
    for f in ctx.features:
        if f == "F3" and store is None:
            continue
        try:
            out[f] = B.feature_matrix(records, f, ctx.cfg, store)
        except KeyError as e:
            ctx.log(f"[{tag}] {f} skipped: {e}")
    return out


def with_preps(ctx, task, X, domains=None, groups=None):
    """Add the task's preprocessing variants: 'F3.bg' = F3 centred per domain, etc."""
    out = dict(X)
    if not ctx.preps:
        return out
    for base, chain in TASK_PREPS[task]:
        if base not in X or ("bg" in chain and domains is None):
            continue
        out[base + "." + ".".join(chain)] = B.apply_preps(X[base], chain, base, ctx.cfg, domains, groups)
    return out


def _with_ctrl(ctx, X, strain, ctrl, store, tag, bases=("F1", "F3")):
    """Add '<base>.ctrl' = feature minus the mean of the same strain's control spectra."""
    if not ctx.preps or not ctrl:
        return X
    cs = np.array([strain_of(r["id"]) for r in ctrl])
    if not set(np.unique(strain)) <= set(cs.tolist()):
        return X                                           # a strain without a control: no '.ctrl' features
    Xc = feature_set(ctx, ctrl, store, tag + " controls")
    out = dict(X)
    for b in bases:
        if b in X and b in Xc:
            M = np.asarray(X[b], float).copy()
            for s_ in np.unique(strain):
                M[strain == s_] -= Xc[b][cs == s_].mean(0)
            out[b + ".ctrl"] = M.astype(np.float32)
    return out


def strain_of(rid: str) -> str:
    parts = rid.split("_")
    return parts[1] if len(parts) >= 3 else "?"


def _arr(v):
    return np.array(v, dtype=object)


# ================================================================== T1
def task_T1(ctx):
    recs = load_sers(ctx)
    cells = [r for r in recs if r["kind"] == "cell" and r.get("analyte")]
    if len(cells) < 4:
        raise SystemExit("T1: fewer than 4 cell spectra with an analyte")
    classes = sorted({r["analyte"] for r in cells})
    K = len(classes)
    stds = [r for r in recs if r["kind"] == "standard" and r.get("analyte") in classes]
    cbs = [r for r in recs if r["kind"] == "substrate_only" and r.get("analyte") in classes]
    ctrl = [r for r in recs if r["kind"] == "cell" and not r.get("analyte") and re.search(ctx.control_regex, r["id"])]
    store = spectra_store(ctx, ["sers"])
    y, g = _arr([r["analyte"] for r in cells]), _arr([r["group"] for r in cells])
    strain = np.array([strain_of(r["id"]) for r in cells])
    # '.bg' variants: each strain's mean spectrum (mean of culture means) is subtracted. No labels used.
    X = with_preps(ctx, "T1", feature_set(ctx, cells, store, "T1"), strain, g)
    # '.ctrl' variants: the strain's own no-analyte control culture is subtracted. Unlike '.bg' this
    # uses no spectra of the cultures being classified. Needs a control for every strain.
    X = _with_ctrl(ctx, X, strain, ctrl, store, "T1")
    strains = sorted(set(strain))
    ids = [r["id"] for r in cells]
    ctx.log(f"[T1] {len(cells)} cell spectra, {len(set(g))} cultures, {K} analytes, strains "
            f"{ {str(s): int((strain == s).sum()) for s in strains} }; {len(stds)} standards; {len(cbs)} CB-only")
    per_class_cultures = {c: len(set(g[y == c])) for c in classes}

    specs = []
    if len(strains) >= 2:
        folds = [(f"test={s}", np.where(strain != s)[0], np.where(strain == s)[0]) for s in strains]
        subs = {"all": np.ones(len(cells), bool), **{f"test={s}": strain == s for s in strains}}
        specs.append(Spec("T1", "loso", "analyte", False, X, X, y, g, y, folds, subs, g, g, strain,
                          chance=1 / K, te_ids=ids, te_analyte=y, agg_groups=g, agg_name="culture",
                          agg_subsets=("all",),
                          note="one culture per strain x analyte: no grouped inner CV, default hyperparameters"))
    else:
        ctx.log("[T1] only one strain: leave-one-strain-out skipped")
    sp1 = np.array([r.get("split", "") for r in cells])
    if (sp1 == "train").any() and (sp1 == "test").any():
        n_te = len(set(y[sp1 == "test"].tolist()))
        if n_te < K:
            ctx.log(f"[T1] NOTE: the step-1 split tests cells of only {n_te}/{K} analytes (for the others the "
                    "held-out group is the standard, so both cultures are in training). LOSO is the primary result.")
        specs.append(Spec("T1", "step1_split", "analyte", False, X, X, y, g, y,
                          [("step1", np.where(sp1 == "train")[0], np.where(sp1 == "test")[0])],
                          {"all": sp1 == "test"}, g, g, sp1, chance=1 / K, te_ids=ids, te_analyte=y,
                          note=f"the split in sers_records.jsonl; test cells cover {n_te}/{K} analytes"))
    if stds:
        ys, gs = _arr([r["analyte"] for r in stds]), _arr([r["group"] for r in stds])
        Xs = with_preps(ctx, "T1", feature_set(ctx, stds, store, "T1-standards"), np.array(["std"] * len(stds)), gs)
        Xc = {f: X[f] for f in Xs if f in X}
        specs.append(Spec("T1", "cells_to_standards", "analyte", False, Xc, Xs, y, g, ys,
                          [("cells->standards", np.arange(len(cells)), np.arange(len(stds)))],
                          {"all": np.ones(len(stds), bool)}, gs, g, strain, shared=False, chance=1 / K,
                          te_ids=[r["id"] for r in stds], te_analyte=ys,
                          note="train: all cultures; test: pure standards (different matrix, no cells)"))
    rows, results = run_specs(specs, ctx, "T1")
    extra = {"per_class": _per_class_rows(specs, results, ctx), "per_class_cultures": per_class_cultures,
             "codes": {r["analyte"]: r["id"].split("_")[2] for r in cells if len(r["id"].split("_")) >= 4}}

    # ---- confusion matrices: LR-F1 and the best combination, per scheme
    conf = {}
    for i, sp in enumerate(specs):
        for key in {("F1", "lr"), best_combo(specs, results, i)} - {None}:
            if (i, *key) not in results:
                continue
            res = results[(i, *key)][0]
            idx = np.where(sp.subsets["all"] & (res["fold"] >= 0))[0]
            conf[(sp.scheme, *key)] = _confusion(sp.yte[idx], res["score"][idx], classes)
    extra["confusion"] = conf

    # ---- confounds
    loso_i = next((i for i, sp in enumerate(specs) if sp.scheme == "loso"), None)
    best = best_combo(specs, results, loso_i) if loso_i is not None else ("F1", "lr")
    extra["best_loso"] = best
    crows = []
    crows += _metadata_only(ctx, cells, y, g, strain, sp1, K)
    cb_ctrl = [r for r in recs if r["kind"] == "substrate_only" and not r.get("analyte")
               and re.search(ctx.control_regex, r["id"])]
    crows += _substrate_only(ctx, cells, cbs, X, y, g, strain, store, K, best, cb_ctrl)
    if ctx.old_manifest:
        crows += _old_split(ctx, cells, X, y, K, best)
    if ctx.old_predictions:
        crows += _old_prediction_rows(ctx, cells, X, y, g, K, best)
    if ctrl:
        ctx.log(f"[T1] {len(ctrl)} no-analyte control spectra: {sorted({r['group'] for r in ctrl})}")
    else:
        ctx.log("[T1] no control spectra matched --control_regex (kind=cell, no analyte); detectability "
                "uses the strain mean as background only")
    recall = {}
    if loso_i is not None and best is not None:
        res = results[(loso_i, *best)][0]
        recall = B.per_class_recall(y, res["score"])
    extra["detectability"] = detectability(ctx, cells, stds, ctrl, store, y, g, strain, recall)
    return {"rows": rows, "confound_rows": crows, "specs": specs, "results": results, "extra": extra,
            "classes": classes}


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def detectability(ctx, cells, stds, ctrl, store, y, g, strain, recall):
    """Which analytes leave a reproducible signature in the cell spectra?

    For each strain, an analyte's signature is its culture-mean spectrum minus
    the strain mean (mean of culture means). No classifier is involved.
      r_cross       correlation of the signature between the two strains
      rank_*        where the matching analyte ranks among all analytes of the
                    other strain (1 = its own signature is the best match)
      r_std_<s>     correlation with the pure standard's signature
      snr_<s>       signature size / scan-to-scan noise within the culture
      r_cross_ctrl  as r_cross, with each strain's no-analyte control culture as the
                    background instead of the strain mean
      top_bands_cm  the three strongest positive bands of the mean signature
    The 829 reference band (1.0 by construction) is excluded."""
    if store is None or not cells:
        return []
    grid = store.grid
    rn = ctx.cfg["substrate"]["reference_norm"]
    keep = np.abs(grid - float(rn["ref_cm"])) > 15.0
    F = store.matrix(cells).astype(float)
    strains = sorted(set(strain))
    analytes = sorted(set(y.tolist()))
    mean, sd, n = {}, {}, {}
    for s_ in strains:
        for a in analytes:
            m = (strain == s_) & (y == a)
            if m.any():
                mean[(s_, a)], sd[(s_, a)], n[(s_, a)] = F[m].mean(0), F[m].std(0), int(m.sum())
    bg = {s_: np.mean([v for (t, _), v in mean.items() if t == s_], axis=0) for s_ in strains}
    d = {k: v - bg[k[0]] for k, v in mean.items()}
    Ds = {}
    if stds:
        S = store.matrix(stds).astype(float)
        sa = np.array([r["analyte"] for r in stds])
        smean = S[np.isin(sa, analytes)].mean(0) if np.isin(sa, analytes).any() else S.mean(0)
        Ds = {a: S[sa == a].mean(0) - smean for a in analytes if (sa == a).any()}
    cbg = {}
    if ctrl:
        C = store.matrix(ctrl).astype(float)
        cs = np.array([strain_of(r["id"]) for r in ctrl])
        cbg = {s_: C[cs == s_].mean(0) for s_ in strains if (cs == s_).any()}
    rows = []
    for a in analytes:
        row = {"analyte": a}
        for s_ in strains:
            if (s_, a) in d:
                row[f"n_{s_}"] = n[(s_, a)]
                noise = float(np.median(sd[(s_, a)][keep])) or np.nan
                row[f"snr_{s_}"] = round(float(np.sqrt(np.mean(d[(s_, a)][keep] ** 2)) / noise), 2)
                if a in Ds:
                    row[f"r_std_{s_}"] = round(_corr(d[(s_, a)][keep], Ds[a][keep]), 3)
        if len(strains) == 2 and all((s_, a) in d for s_ in strains):
            s1, s2 = strains
            row["r_cross"] = round(_corr(d[(s1, a)][keep], d[(s2, a)][keep]), 3)
            for u, v in ((s1, s2), (s2, s1)):
                rs = {b: _corr(d[(u, a)][keep], d[(v, b)][keep]) for b in analytes if (v, b) in d}
                row[f"rank_{u}_in_{v}"] = 1 + sum(r > rs[a] for b, r in rs.items() if b != a)
            if s1 in cbg and s2 in cbg:        # both strains have a control: signature = culture - control
                row["r_cross_ctrl"] = round(_corr((mean[(s1, a)] - cbg[s1])[keep], (mean[(s2, a)] - cbg[s2])[keep]), 3)
            elif s1 in cbg or s2 in cbg:       # one control: control-referenced vs strain-mean-referenced
                s_, o = (s1, s2) if s1 in cbg else (s2, s1)
                row["r_cross_ctrl"] = round(_corr((mean[(s_, a)] - cbg[s_])[keep], d[(o, a)][keep]), 3)
            dm = (d[(s1, a)] + d[(s2, a)]) / 2
            dm = np.where(keep, dm, -np.inf)
            top = []
            for i in np.argsort(-dm):
                if all(abs(grid[i] - t) > 15 for t in top):
                    top.append(float(grid[i]))
                if len(top) == 3:
                    break
            row["top_bands_cm"] = " ".join(f"{t:.0f}" for t in top)
        if a in recall:
            row["loso_recall_best"] = round(recall[a], 2)
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("r_cross") if r.get("r_cross") is not None else -9))
    return rows


def _confusion(y, pred, classes):
    idx = {c: k for k, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, pred):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    return M


def _per_class_rows(specs, results, ctx):
    out = []
    for (i, f, m), (res, _) in results.items():
        sp = specs[i]
        if sp.binary:
            continue
        idx = np.where(sp.subsets["all"] & (res["fold"] >= 0))[0]
        for c, rec in B.per_class_recall(sp.yte[idx], res["score"][idx]).items():
            out.append({"task": sp.task, "variant": ctx.variant, "scheme": sp.scheme, "feature": f, "model": m,
                        "analyte": c, "recall": round(rec, 4), "n": int((sp.yte[idx] == c).sum())})
    return out


def _metadata_only(ctx, cells, y, g, strain, sp1, K):
    """Predict the analyte from metadata alone. 'design' = strain (+ any columns
    of --meta_csv, e.g. date); 'acquisition' = absolute CB[5] 829 counts, 829
    band position (calibration drift) and number of CB peaks discarded. None of
    these reach the model; if they predict the analyte, analyte is confounded
    with how or when the spectrum was taken."""
    design = [[1.0 if s == t else 0.0 for t in sorted(set(strain))] for s in strain]
    if ctx.meta_csv:
        extra = {normalise_id(Path(r["id"]).stem): r for r in csv.DictReader(open(ctx.meta_csv))}
        cols = [c for c in (next(iter(extra.values())).keys() if extra else []) if c != "id"]
        for c in cols:
            vals = [extra.get(r["id"], {}).get(c, "") for r in cells]
            try:
                num = [float(v) for v in vals]
                design = [d + [v] for d, v in zip(design, num)]
            except ValueError:
                levels = sorted(set(vals))
                design = [d + [1.0 if v == lv else 0.0 for lv in levels] for d, v in zip(design, vals)]
        ctx.log(f"[T1 meta] --meta_csv columns {cols} added to the design metadata")
    acq = []
    for r in cells:
        s = r.get("substrate") or {}
        h = s.get("ref_height")
        acq.append([np.log10(h) if h else np.nan, s.get("ref_pos_cm") or np.nan, float(s.get("n_masked") or 0)])
    acq = np.asarray(acq, float)
    med = np.nanmedian(acq, axis=0)
    acq = np.where(np.isnan(acq), med, acq)
    X = {"META_design": np.asarray(design, np.float32), "META_acq": acq.astype(np.float32)}
    X["META_all"] = np.hstack([X["META_design"], X["META_acq"]])
    specs = []
    strains = sorted(set(strain))
    if len(strains) >= 2:
        folds = [(f"test={s}", np.where(strain != s)[0], np.where(strain == s)[0]) for s in strains]
        specs.append(Spec("T1", "loso", "analyte", False, X, X, y, g, y, folds, {"all": np.ones(len(y), bool)},
                          g, g, strain, chance=1 / K, features=tuple(X), models=("majority", "lr", "hgb"),
                          note="metadata-only confound check"))
    if (sp1 == "train").any() and (sp1 == "test").any():
        n_te = len(set(y[sp1 == "test"].tolist()))
        if n_te < K:
            ctx.log(f"[T1] NOTE: the step-1 split tests cells of only {n_te}/{K} analytes (for the others the "
                    "held-out group is the standard, so both cultures are in training). LOSO is the primary result.")
        specs.append(Spec("T1", "step1_split", "analyte", False, X, X, y, g, y,
                          [("step1", np.where(sp1 == "train")[0], np.where(sp1 == "test")[0])],
                          {"all": sp1 == "test"}, g, g, sp1, chance=1 / K, features=tuple(X),
                          models=("majority", "lr", "hgb"), note="metadata-only confound check"))
    rows, _ = run_specs(specs, ctx, "T1 metadata-only")
    for r in rows:
        r["check"] = "metadata_only"
    return rows


def _substrate_only(ctx, cells, cbs, X, y, g, strain, store, K, best, cb_ctrl=None):
    """CB-only spectra contain no analyte. (a) cells -> CB: a classifier trained
    on cell spectra labels the CB-only spectra of the held-out strain (the step-1
    control, now with the best step-2 model). (b) CB -> CB: can CB-only spectra
    alone identify which analyte's session they came from? Both must sit at chance."""
    if not cbs:
        ctx.log("[T1 substrate-only] no CB-only records: skipped")
        return []
    feats = tuple(dict.fromkeys(["F1", best[0]]))
    models = tuple(dict.fromkeys(["majority", "lr", best[1]]))
    sub_ctx_feats = [f for f in feats if f in X]
    ycb, gcb = _arr([r["analyte"] for r in cbs]), _arr([r["group"] for r in cbs])
    scb = np.array([strain_of(r["id"]) for r in cbs])
    Xcb_all = with_preps(ctx, "T1", feature_set(ctx, cbs, store, "T1 substrate-only"), scb, gcb)
    Xcb_all = _with_ctrl(ctx, Xcb_all, scb, cb_ctrl or [], store, "T1 substrate-only")
    Xcb = {f: Xcb_all[f] for f in sub_ctx_feats if f in Xcb_all}
    Xc = {f: X[f] for f in Xcb}
    strains = sorted(set(strain) & set(scb))
    specs = []
    if len(strains) >= 2:
        folds = [(f"test={s}", np.where(strain != s)[0], np.where(scb == s)[0]) for s in strains]
        specs.append(Spec("T1", "loso:cells->CB", "analyte", False, Xc, Xcb, y, g, ycb, folds,
                          {"all": np.isin(scb, strains)}, gcb, g, strain, shared=False, chance=1 / K,
                          features=tuple(Xcb), models=models, note="substrate-only control (a)"))
        folds = [(f"test={s}", np.where(scb != s)[0], np.where(scb == s)[0]) for s in strains]
        specs.append(Spec("T1", "loso:CB->CB", "analyte", False, Xcb, Xcb, ycb, gcb, ycb, folds,
                          {"all": np.isin(scb, strains)}, gcb, gcb, scb, chance=1 / K,
                          features=tuple(Xcb), models=models, note="substrate-only control (b)"))
    rows, _ = run_specs(specs, ctx, "T1 substrate-only")
    for r in rows:
        r["check"] = "substrate_only"
    return rows


def _old_split(ctx, cells, X, y, K, best):
    """The old fine-tune's train/test split (split_manifest.csv), CB rows removed.
    Not grouped by culture, so expect it to be optimistic: that is the point of
    putting it next to the grouped numbers."""
    rows_m = list(csv.DictReader(open(ctx.old_manifest)))
    split = {normalise_id(Path(r["fname"].replace("\\", "/")).stem): r.get("split", "") for r in rows_m}
    grp = {normalise_id(Path(r["fname"].replace("\\", "/")).stem): r.get("group", "") for r in rows_m}
    ids = [r["id"] for r in cells]
    sp = np.array([split.get(i, "") for i in ids])
    if not ((sp == "train").any() and (sp == "test").any()):
        ctx.log(f"[T1 old split] no overlap between {ctx.old_manifest} and the cell records: skipped")
        return []
    n_cb = sum(1 for r in rows_m if normalise_id(Path(r["fname"]).stem).startswith("CB_"))
    ctx.log(f"[T1 old split] {int((sp == 'train').sum())} train / {int((sp == 'test').sum())} test cell spectra "
            f"({n_cb} CB rows in the manifest ignored)")
    gold = _arr([grp.get(i) or f"old:{i}" for i in ids])
    feats = tuple(dict.fromkeys(["F1", best[0]]))
    Xo = {f: X[f] for f in feats if f in X}
    spec = Spec("T1", "old_split(no CB)", "analyte", False, Xo, Xo, y, gold, y,
                [("old", np.where(sp == "train")[0], np.where(sp == "test")[0])], {"all": sp == "test"},
                gold, gold, sp, chance=1 / K, features=tuple(Xo),
                models=tuple(dict.fromkeys(["majority", "lr", best[1]])),
                note="old fine-tune split, CB rows removed; groups = old sample ids")
    rows, _ = run_specs([spec], ctx, "T1 old split")
    for r in rows:
        r["check"] = "old_split"
    return rows


def _pred_ids(ctx):
    rows = list(csv.DictReader(open(ctx.old_predictions)))
    cols = list(rows[0]) if rows else []
    fcol = next((c for c in cols if c.lower() in ("file", "fname", "filename", "id")), cols[0] if cols else None)
    return rows, fcol, [normalise_id(Path(str(r[fcol]).replace("\\", "/")).stem) for r in rows]


def _old_prediction_rows(ctx, cells, X, y, g, K, best):
    """Like-for-like with the old LLM: the TEST set is exactly the non-CB rows of
    the old predictions file; everything else is training. (The manifest's split
    is not the one behind that file: most of its rows are manifest-train.)"""
    prows, fcol, pids = _pred_ids(ctx)
    if not prows:
        return []
    ids = np.array([r["id"] for r in cells])
    is_test = np.isin(ids, pids)
    out = []
    if ctx.old_manifest:
        split = {normalise_id(Path(r["fname"].replace("\\", "/")).stem): r.get("split", "")
                 for r in csv.DictReader(open(ctx.old_manifest))}
        from collections import Counter
        c = Counter(split.get(i, "not in manifest") for i in pids)
        ctx.log(f"[T1 old predictions] rows of the predictions file by manifest split: {dict(c)}")
    if is_test.sum() >= 5 and (~is_test).sum() >= 5:
        feats = tuple(dict.fromkeys(["F1", best[0], "F3"]))
        Xo = {f: X[f] for f in feats if f in X}
        sp = np.where(is_test, "test", "train")
        spec = Spec("T1", "old_prediction_rows", "analyte", False, Xo, Xo, y, g, y,
                    [("old", np.where(~is_test)[0], np.where(is_test)[0])], {"all": is_test}, g, g, None,
                    chance=1 / K, features=tuple(Xo), models=tuple(dict.fromkeys(["majority", "lr", "knn", best[1]])),
                    note="test = non-CB rows of the old predictions file; train = all other cell spectra")
        ctx.log(f"[T1 old predictions] {int((~is_test).sum())} train / {int(is_test.sum())} test cell spectra")
        out, _ = run_specs([spec], ctx, "T1 old predictions")
    for r in out:
        r["check"] = "old_predictions"
    return out + _old_llm_rows(ctx, set(ids[is_test].tolist()))


def _truthy(v):
    return str(v).strip().lower() in {"true", "correct", "1", "yes", "y", "right", "t", "✓", "✅"}


def _old_llm_rows(ctx, test_ids):
    rows = list(csv.DictReader(open(ctx.old_predictions)))
    if not rows:
        return []
    cols = list(rows[0])
    fcol = next((c for c in cols if c.lower() in ("file", "fname", "filename", "id")), cols[0])
    use_result = "Result" in cols and all(str(r["Result"]).strip().lower() in
                                          {"true", "false", "correct", "incorrect", "1", "0", "yes", "no", "right",
                                           "wrong", "t", "f", "✓", "✗", "✅", "❌"} for r in rows)

    def norm(s):
        s = str(s).strip().lower()
        return s[2:] if s.startswith("l-") else s

    out = []
    stats = {}
    for r in rows:
        rid = normalise_id(Path(str(r[fcol]).replace("\\", "/")).stem)
        if use_result:
            ok = _truthy(r["Result"])
        else:
            t = r.get("true_name") or r.get("Extracted amino acid from filename", "")
            p = r.get("predicted_name") or r.get("Predicted_name") or r.get("Predicted", "")
            ok = norm(t) == norm(p)
        for subset, keep in (("all rows", True), ("non-CB rows", not rid.startswith("CB_")),
                             ("non-CB rows matched to a cell spectrum", rid in test_ids)):
            if keep:
                stats.setdefault(subset, []).append(ok)
    for subset, oks in stats.items():
        out.append({"task": "T1", "variant": ctx.variant, "scheme": "old_prediction_rows", "test_set": subset,
                    "label": "analyte", "feature": "old string", "model": "old_llm", "metric": "accuracy",
                    "value": round(float(np.mean(oks)), 4), "n_test": len(oks), "check": "old_predictions",
                    "note": f"{sum(oks)}/{len(oks)} correct in {Path(ctx.old_predictions).name}"
                            f" ({'Result column' if use_result else 'true vs predicted name'})"})
    return out


# ================================================================== T2
def _lpo_job(X, y, akey, model, seed, default_hp, n_boot, n_jobs, tune=True):
    t0 = time.time()
    with _limit_threads(n_jobs):
        r = B.leave_pair_out(X, y, akey, model, seed, tune=tune, default_hp=default_hp)
    r["auroc"] = float(np.nanmean(r["W"])) if np.isfinite(r["W"]).any() else np.nan
    r["auroc_spectra"] = float(np.nanmean(r["S"])) if np.isfinite(r["S"]).any() else np.nan
    r["ci"] = B.lpo_ci(r["W"], n_boot, seed)
    r["ci_spectra"] = B.lpo_ci(r["S"], n_boot, seed)
    r["seconds"] = round(time.time() - t0, 2)
    return r


def _lpo_perm_job(X, y, akey, model, seed, default_hp, seeds, n_jobs):
    """Null for the leave-pair-out AUROC: labels shuffled between analytes,
    default hyperparameters (no tuning inside the shuffles)."""
    out = []
    with _limit_threads(n_jobs):
        for ps in seeds:
            yp = B.permute_by_group(y, akey, None, np.random.default_rng(ps))
            w = B.leave_pair_out(X, yp, akey, model, seed, tune=False, default_hp=default_hp)["W"]
            out.append(float(np.nanmean(w)) if np.isfinite(w).any() else np.nan)
    return out


def _t2_testable(yl, akey, aname, min_analytes, min_spectra):
    pos = sorted({aname[k] for k, v in zip(akey, yl) if v == 1})
    neg = sorted({aname[k] for k, v in zip(akey, yl) if v == 0})
    npos, nneg = int((yl == 1).sum()), int((yl == 0).sum())
    ok = len(pos) >= min_analytes and len(neg) >= min_analytes and npos >= min_spectra and nneg >= min_spectra
    reason = "" if ok else (f"{len(pos)} analyte(s) / {npos} spectra with the label" if len(pos) < min_analytes
                            or npos < min_spectra else f"{len(neg)} analyte(s) / {nneg} spectra without it")
    return ok, reason, pos, neg, npos


def task_T2(ctx):
    """Structural labels on SERS, holding analytes out.

    lpo:cells / lpo:standards   leave-pair-out AUROC within one matrix (no
                                cells-vs-standards confound)
    lpo:all                     both matrices pooled, with a matrix-only control
                                (feature META_matrix) to show how much of the
                                AUROC the sample type alone explains
    std_to_cells                train on the standards of analytes that have no cell
                                spectra (the indole family), test on cell spectra
    Metric 'auroc' = analyte-level leave-pair-out AUROC (does the positive
    analyte's mean score beat the negative's, under a model that saw neither);
    'auroc_spectra' = the same within-fold comparison at spectrum level."""
    recs = [r for r in load_sers(ctx) if r["kind"] in ("cell", "standard") and r.get("smiles") and r.get("labels")]
    if not recs:
        raise SystemExit("T2: no cell or standard records with SMILES")
    akey = _arr([canonical(r["smiles"]) for r in recs])
    names = {}
    for r, k in zip(recs, akey):
        names.setdefault(k, set()).add(r["analyte"] or k)
    aname = {k: "+".join(sorted(v)) for k, v in names.items()}
    kind = np.array([r["kind"] for r in recs])
    domain = np.array([strain_of(r["id"]) if r["kind"] == "cell" else "std" for r in recs])
    grp = _arr([r["group"] for r in recs])
    store = spectra_store(ctx, ["sers"])
    X = with_preps(ctx, "T2", feature_set(ctx, recs, store, "T2"), domain, grp)
    doms = sorted(set(domain))
    meta = np.array([[1.0 if d == t else 0.0 for t in doms] for d in domain], np.float32)
    labels = list(ctx.labels or FG_KEYS)
    models = ctx.models or TASK_MODELS["T2"]
    matrices = {"all": np.ones(len(recs), bool), "cells": kind == "cell", "standards": kind == "standard"}
    need = {"all": (2, 5), "cells": (2, 5), "standards": (3, 3)}      # (analytes, spectra) per side

    table, jobs = [], []
    for mname, mask in matrices.items():
        if not mask.any():
            continue
        for lab in labels:
            yl = np.array([int(r["labels"].get(lab, 0)) for r in recs])
            ok, reason, pos, neg, npos = _t2_testable(yl[mask], akey[mask], aname, *need[mname])
            table.append({"matrix": mname, "label": lab, "n_pos_analytes": len(pos), "n_neg_analytes": len(neg),
                          "pos_analytes": "; ".join(pos), "n_pos_spectra": npos, "n_spectra": int(mask.sum()),
                          "testable": ok, "reason": reason})
            if not ok:
                continue
            for f in X:
                for m in models:
                    jobs.append((mname, lab, f, m))
            if mname == "all":
                jobs.append((mname, lab, "META_matrix", "lr"))
    ctx.log(f"[T2] {len(recs)} spectra, {len(set(akey))} analytes. Testable: "
            + "; ".join(f"{m}: {[t['label'].replace('fg_', '') for t in table if t['matrix'] == m and t['testable']]}"
                        for m in matrices))
    ys = {lab: np.array([int(r["labels"].get(lab, 0)) for r in recs]) for lab in labels}

    def args(mname, lab, f):
        mask = matrices[mname]
        Xf = meta if f == "META_matrix" else X[f]
        return Xf[mask], ys[lab][mask], akey[mask]

    t0 = time.time()
    ctx.log(f"[T2] {len(jobs)} leave-pair-out evaluations on {ctx.n_jobs} job(s)")
    outs = _parallel(ctx, [(_lpo_job, (*args(mn, lab, f), m, ctx.seed, ctx.default_hp, ctx.n_boot, ctx.n_jobs))
                           for mn, lab, f, m in jobs])
    res = dict(zip(jobs, outs))
    ctx.log(f"[T2] leave-pair-out done in {time.time() - t0:.0f} s")

    # ---- permutation tests: LR-F1 and the best combination per matrix x label
    pvals = {}
    if ctx.n_perm > 0 and ctx.perm_for != "none":
        chosen = []
        for mn, lab in dict.fromkeys((j[0], j[1]) for j in jobs):
            cand = {(f, m): res[(mn, lab, f, m)]["auroc"] for (a, b, f, m) in jobs
                    if (a, b) == (mn, lab) and m != "majority" and f != "META_matrix"}
            cand = {k: v for k, v in cand.items() if np.isfinite(v)}
            if not cand:
                continue
            keys = set(cand) if ctx.perm_for == "all" else ({("F1", "lr")} & set(cand)) | {max(cand, key=cand.get)}
            chosen += [(mn, lab, f, m) for f, m in sorted(keys)]
        if chosen:
            t1 = time.time()
            share = 60.0 * ctx.perm_budget_min * max(1, ctx.n_jobs) / len(chosen)
            calls, index, real0 = [], [], {}
            for k, (mn, lab, f, m) in enumerate(chosen):
                a = args(mn, lab, f)
                tt = time.time()
                # real labels, default hyperparameters: the statistic the shuffles are compared with
                w0 = B.leave_pair_out(*a, m, ctx.seed, tune=False, default_hp=ctx.default_hp)["W"]
                real0[(mn, lab, f, m)] = float(np.nanmean(w0))
                per = max(time.time() - tt, 1e-3)
                n_perm = int(min(ctx.n_perm, max(min(ctx.min_perm, ctx.n_perm), share / per)))
                if n_perm < ctx.n_perm:
                    ctx.log(f"[T2] {mn}/{lab} {f}/{m}: {per:.1f} s per shuffle -> {n_perm} shuffles")
                seeds = [10_000_019 * (ctx.seed + 1) + 1009 * k + i for i in range(n_perm)]
                n_chunks = max(1, min(ctx.n_jobs, n_perm // 5 or 1))
                for c in range(n_chunks):
                    calls.append((_lpo_perm_job, (*a, m, ctx.seed, ctx.default_hp, seeds[c::n_chunks], ctx.n_jobs)))
                    index.append((mn, lab, f, m))
            null = {}
            for key, vals in zip(index, _parallel(ctx, calls)):
                null.setdefault(key, []).extend(vals)
            for key, vals in null.items():
                v = np.asarray([x for x in vals if np.isfinite(x)])
                real = real0[key]        # shuffles use default hyperparameters: compare like with like
                pvals[key] = ((1 + int((v >= real - 1e-12).sum())) / (1 + len(v)), len(v), float(v.mean()))
            ctx.log(f"[T2] permutations done in {time.time() - t1:.0f} s")

    rows, per_analyte = [], []
    for (mn, lab, f, m), r in res.items():
        base = {"task": "T2", "variant": ctx.variant, "scheme": f"lpo:{mn}", "test_set": "all", "label": lab,
                "feature": f, "model": m, "chance": 0.5, "p_perm": "", "n_perm": "", "perm_null_mean": "",
                "n_test": int(matrices[mn].sum()), "n_test_groups": len(r["pos"]) + len(r["neg"]),
                "n_boot_ok": ctx.n_boot, "n_train": "", "hp_source": "inner_cv(per fold)",
                "hp": json.dumps(dict(r["hp"])), "seconds": r["seconds"],
                "note": f"{len(r['pos'])} positive x {len(r['neg'])} negative analytes"
                        + ("; matrix-only control" if f == "META_matrix" else "")}
        for metric, ci in (("auroc", "ci"), ("auroc_spectra", "ci_spectra")):
            row = dict(base, metric=metric, value=_r(r[metric]), ci_lo=_r(r[ci][0]), ci_hi=_r(r[ci][1]))
            if metric == "auroc" and (mn, lab, f, m) in pvals:
                pv, n, nm = pvals[(mn, lab, f, m)]
                row.update(p_perm=_r(pv), n_perm=n, perm_null_mean=_r(nm))
            rows.append(row)
        W = r["W"]
        for i, a in enumerate(r["pos"]):
            per_analyte.append({"task": "T2", "scheme": f"lpo:{mn}", "label": lab, "feature": f, "model": m,
                                "analyte": aname.get(a, a), "true": 1, "win_rate": _r(np.nanmean(W[i]))})
        for j, a in enumerate(r["neg"]):
            per_analyte.append({"task": "T2", "scheme": f"lpo:{mn}", "label": lab, "feature": f, "model": m,
                                "analyte": aname.get(a, a), "true": 0, "win_rate": _r(1 - np.nanmean(W[:, j]))})

    # ---- standards -> cells
    specs = []
    cmask, smask = kind == "cell", kind == "standard"
    # train only on standards of analytes that have NO cell spectra, so a hit cannot come from
    # recognising the analyte itself (e.g. indole standards -> tryptophan cells)
    cell_analytes = set(akey[cmask].tolist())
    smask = smask & ~np.isin(akey, list(cell_analytes))
    if cmask.any() and smask.any():
        tr, te = np.where(smask)[0], np.where(cmask)[0]
        Xtr = {f: M[tr] for f, M in X.items()}
        Xte = {f: M[te] for f, M in X.items()}
        an = _arr([aname[k] for k in akey])
        for lab in labels:
            ok_s = _t2_testable(ys[lab][tr], akey[tr], aname, 2, 2)[0]
            ok_c = _t2_testable(ys[lab][te], akey[te], aname, 1, 5)[0]
            if ok_s and ok_c:
                specs.append(Spec("T2", "std_to_cells", lab, True, Xtr, Xte, ys[lab][tr], akey[tr], ys[lab][te],
                                  [("standards->cells", np.arange(len(tr)), np.arange(len(te)))],
                                  {"all": np.ones(len(te), bool)}, akey[te], akey[tr], shared=False,
                                  te_ids=[recs[i]["id"] for i in te], te_analyte=an[te],
                                  agg_groups=akey[te], agg_name="analyte",
                                  note="train: standards of analytes with no cell spectra; test: cell spectra"))
    srows, sres = run_specs(specs, ctx, "T2 standards->cells") if specs else ([], {})
    per_analyte += [dict(r, win_rate="") for r in _per_analyte_rows(specs, sres, ctx)]
    return {"rows": rows + srows, "specs": specs, "results": sres,
            "extra": {"labels": table, "per_analyte": per_analyte}}


def _per_analyte_rows(specs, results, ctx):
    out = []
    for (i, f, m), (res, _) in results.items():
        sp = specs[i]
        if not sp.binary or sp.te_analyte is None:
            continue
        tested = res["fold"] >= 0
        rows = []
        for a in sorted(set(sp.te_analyte[tested].tolist())):
            k = tested & (sp.te_analyte == a)
            rows.append({"task": sp.task, "variant": ctx.variant, "scheme": sp.scheme, "label": sp.label,
                         "feature": f, "model": m, "analyte": a, "true": int(np.round(sp.yte[k].astype(float).mean())),
                         "n": int(k.sum()), "mean_score": round(float(np.mean(res["score"][k].astype(float))), 4),
                         "frac_pred_pos": round(float(np.mean(res["pred"][k] == 1)), 4)})
        # rank among the SERS analytes (1 = highest mean score); QM9S rows are not ranked
        order = sorted((r for r in rows if r["analyte"] != "qm9s"), key=lambda r: -r["mean_score"])
        for k, r in enumerate(order):
            r["rank"], r["n_analytes"] = k + 1, len(order)
        out += rows
    return out


# ================================================================== T3
def _fg_labels(ctx):
    return list(ctx.labels or FG_KEYS)


def task_T3(ctx):
    q = load_qm9s(ctx)
    store = spectra_store(ctx, ["qm9s"])
    X = with_preps(ctx, "T3", feature_set(ctx, q, store, "T3"))
    ids = _arr([r["id"] for r in q])
    is_test_rand = np.array([r.get("split") == "test" for r in q])
    is_test_scaf, scaf = B.scaffold_split([r["smiles"] for r in q], 0.2)
    scaf = _arr(scaf)
    schemes = {"random": (is_test_rand, ids), "scaffold": (is_test_scaf, scaf)}
    table, specs = [], []
    for scheme, (is_test, boot) in schemes.items():
        ctx.log(f"[T3] {scheme}: {int((~is_test).sum())} train / {int(is_test.sum())} test molecules, "
                f"{len(set(boot[is_test].tolist()))} test groups")
        for lab in _fg_labels(ctx):
            yl = np.array([int(r["labels"].get(lab, 0)) for r in q])
            ptr, pte = int(yl[~is_test].sum()), int(yl[is_test].sum())
            mp = ctx.min_pos or 50
            ok = ptr >= mp and pte >= ctx.min_pos_test and (~is_test).sum() - ptr >= mp
            table.append({"scheme": scheme, "label": lab, "n_pos_train": ptr, "n_pos_test": pte,
                          "prevalence": round(float(yl.mean()), 4), "testable": ok,
                          "reason": "" if ok else f"needs >= {mp} train / {ctx.min_pos_test} test positives"})
            if ok:
                specs.append(Spec("T3", scheme, lab, True, X, X, yl, ids, yl,
                                  [(scheme, np.where(~is_test)[0], np.where(is_test)[0])],
                                  {"all": is_test}, boot, ids, max_splits=3, te_ids=list(ids),
                                  note=f"bootstrap unit: {'molecule' if scheme == 'random' else 'scaffold'}"))
    rows, results = run_specs(specs, ctx, "T3")
    return {"rows": rows, "specs": specs, "results": results, "extra": {"labels": table}}


# ================================================================== T4
def task_T4(ctx):
    q = load_qm9s(ctx)
    s = [r for r in load_sers(ctx) if r["kind"] in ("cell", "standard") and r.get("smiles") and r.get("labels")]
    qtr = [r for r in q if r.get("split") != "test"]
    qte = [r for r in q if r.get("split") == "test"]
    test_src = qte + s
    store = spectra_store(ctx, ["qm9s", "sers"])
    # '.bg': every domain centred on its own mean (QM9S; each strain's cells; the standards). No labels.
    dom_te = np.array(["qm9s"] * len(qte) + [strain_of(r["id"]) if r["kind"] == "cell" else "std" for r in s])
    grp_te = _arr([r["id"] for r in qte] + [r["group"] for r in s])
    Xtr = with_preps(ctx, "T4", feature_set(ctx, qtr, store, "T4 train"), np.array(["qm9s"] * len(qtr)))
    Xte = with_preps(ctx, "T4", feature_set(ctx, test_src, store, "T4 test"), dom_te, grp_te)
    keep = [f for f in Xtr if f in Xte]
    Xtr, Xte = {f: Xtr[f] for f in keep}, {f: Xte[f] for f in keep}
    is_q = np.r_[np.ones(len(qte), bool), np.zeros(len(s), bool)]
    kind = np.array(["qm9s"] * len(qte) + [r["kind"] for r in s])
    akey = [canonical(r["smiles"]) for r in s]
    names = {}
    for r, k in zip(s, akey):
        names.setdefault(k, set()).add(r["analyte"] or k)
    aname = {k: "+".join(sorted(v)) for k, v in names.items()}
    boot = _arr([r["id"] for r in qte] + [f"sers:{k}" for k in akey])
    te_analyte = _arr(["qm9s"] * len(qte) + [aname[k] for k in akey])
    subs = {"qm9s_test": is_q, "sers_all": ~is_q, "sers_cells": kind == "cell", "sers_standards": kind == "standard"}
    subs = {k: v for k, v in subs.items() if v.any()}
    gtr = _arr([r["id"] for r in qtr])
    table, specs = [], []
    for lab in _fg_labels(ctx):
        ytr = np.array([int(r["labels"].get(lab, 0)) for r in qtr])
        yte = np.array([int(r["labels"].get(lab, 0)) for r in test_src])
        ys = yte[~is_q]
        pos = sorted({aname[k] for k, v in zip(akey, ys) if v == 1})
        neg = sorted({aname[k] for k, v in zip(akey, ys) if v == 0})
        mp = ctx.min_pos or 30
        ok = int(ytr.sum()) >= mp and len(pos) >= 1 and len(neg) >= 1
        reason = "" if ok else (f"{int(ytr.sum())} QM9S training positives (< {mp})" if ytr.sum() < mp
                                else "SERS side has only one class")
        low = ok and (len(pos) < 2 or len(neg) < 2)
        table.append({"label": lab, "n_pos_qm9s_train": int(ytr.sum()), "n_pos_sers_analytes": len(pos),
                      "n_neg_sers_analytes": len(neg), "pos_analytes": "; ".join(pos), "testable": ok,
                      "low_support": low, "reason": reason})
        if ok:
            specs.append(Spec("T4", "qm9s_to_sers", lab, True, Xtr, Xte, ytr, gtr, yte,
                              [("qm9s->sers", np.arange(len(qtr)), np.arange(len(test_src)))], subs, boot, gtr,
                              shared=False, perm_subset="sers_all", with_ece=True, max_splits=3,
                              te_ids=[r["id"] for r in test_src], te_analyte=te_analyte,
                              agg_groups=boot, agg_name="analyte",
                              agg_subsets=tuple(k for k in subs if k.startswith("sers")),
                              note=("SINGLE ANALYTE: <2 SERS analytes on one side, read the rank table; " if low else "")
                                   + "threshold chosen on QM9S (inner CV)"))
    ctx.log(f"[T4] train {len(qtr)} QM9S, test {len(qte)} QM9S + {len(s)} SERS; shared labels "
            f"{[sp.label for sp in specs]}")
    rows, results = run_specs(specs, ctx, "T4")
    pa = [r for r in _per_analyte_rows(specs, results, ctx) if r["analyte"] != "qm9s"]
    return {"rows": rows, "specs": specs, "results": results, "extra": {"labels": table, "per_analyte": pa}}


TASKS = {"T1": task_T1, "T2": task_T2, "T3": task_T3, "T4": task_T4}
