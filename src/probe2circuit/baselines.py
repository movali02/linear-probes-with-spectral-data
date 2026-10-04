"""Step 2: classical baselines on exactly the information the LLM sees.

Building blocks only; the four tasks are assembled in baseline_tasks.py.

Features (roadmap 2.2), from least to most information:
  F1     max intensity per 10 cm-1 bin of the parsed string (what the LLM sees)
  F1pos  F1 with every intensity set to 1 (positions only; reused in step 5)
  F2     every string peak redrawn as a Lorentzian with its FWHM on the 1 cm-1 grid
  F3     the processed spectrum before peak picking (saved by
         build_strings.py --save_spectra); the upper bound. F3 >> F1 means the
         string format throws information away.

Models (2.3): majority class, logistic regression (standardised, L2, balanced,
C by inner grouped CV), histogram gradient boosting (class-weighted), kNN on
cosine distance. Hyperparameters and the binary decision threshold are chosen
by inner grouped CV on the training fold only. When the training fold cannot
be split by group with every class on both sides (T1: one culture per strain x
amino acid), the defaults are used and the result says hp_source = "default".

Statistics (2.4): grouped bootstrap CIs (resample groups, not spectra) and
group-respecting permutation p-values (labels shuffled between groups, within
strata)."""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np

from .preprocess import make_grid
from .strings import parse_string, raster

FEATURES = ("F1", "F1pos", "F2", "F3")
MODELS = ("majority", "lr", "pca_lr", "plsda", "ncm", "hgb", "knn")
GRIDS = {"lr": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
         "pca_lr": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
         "plsda": [{"n": n} for n in (2, 5, 10, 15)],
         "knn": [{"k": k} for k in (1, 3, 5, 10)],
         "hgb": [{}], "majority": [{}], "ncm": [{}]}
DEFAULT_HP = {"lr": {"C": 0.1}, "pca_lr": {"C": 0.1}, "plsda": {"n": 10}, "knn": {"k": 5}, "hgb": {},
              "majority": {}, "ncm": {}}


# ================================================================== preprocessing (step 2b)
def feature_axis(name: str, cfg: dict, bin_cm: float = 10.0) -> np.ndarray:
    """Wavenumber of every column of a feature matrix (bin lower edge for F1)."""
    lo, hi = cfg["window"]
    base = name.split(".")[0]
    return np.arange(lo, hi, bin_cm) if base in ("F1", "F1pos") else make_grid(cfg)


def prep_snv(X):
    """Standard normal variate: each spectrum to zero mean, unit variance."""
    X = np.asarray(X, float)
    return (X - X.mean(1, keepdims=True)) / np.maximum(X.std(1, keepdims=True), 1e-12)


def prep_d1(X, window: int = 11, poly: int = 2):
    """Savitzky-Golay first derivative along the wavenumber axis (removes slow background)."""
    from scipy.signal import savgol_filter
    return savgol_filter(np.asarray(X, float), window, poly, deriv=1, axis=1)


def prep_sqrt(X):
    return np.sqrt(np.clip(np.asarray(X, float), 0, None))


def center_by_domain(X, domains, groups=None):
    """Subtract each domain's mean (e.g. each strain's mean spectrum). The mean
    is the mean of GROUP means, so a culture with 20 scans counts as much as one
    with 10. Uses no labels; it is the simplest domain alignment there is."""
    X = np.asarray(X, float).copy()
    domains = np.asarray(domains).astype(str)
    groups = np.arange(len(X)).astype(str) if groups is None else np.asarray(groups).astype(str)
    for d in np.unique(domains):
        m = domains == d
        _, inv = np.unique(groups[m], return_inverse=True)
        sums = np.zeros((inv.max() + 1, X.shape[1]))
        np.add.at(sums, inv, X[m])
        X[m] -= (sums / np.bincount(inv)[:, None]).mean(0)
    return X


def apply_preps(M, chain, name, cfg, domains=None, groups=None):
    """chain: tuple of 'bg' (centre by domain), 'd1', 'snv', 'sqrt', 'lt<cm>' (crop below <cm>)."""
    for p in chain:
        if p == "bg":
            M = center_by_domain(M, domains, groups)
        elif p == "d1":
            M = prep_d1(M)
        elif p == "snv":
            M = prep_snv(M)
        elif p == "sqrt":
            M = prep_sqrt(M)
        elif p.startswith("lt"):
            M = np.asarray(M)[:, feature_axis(name, cfg) < float(p[2:])]
        else:
            raise ValueError(f"unknown preprocessing step {p!r}")
    return np.asarray(M, dtype=np.float32)


# ================================================================== extra models (step 2b)
class NearestCentroid:
    """Nearest class mean by correlation, after centring on the training mean.
    The natural classifier when there is one culture per class."""

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.mu_ = X.mean(0)
        C = np.stack([X[y == c].mean(0) for c in self.classes_]) - self.mu_
        self.C_ = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        return self

    def _sim(self, X):
        Z = np.asarray(X, float) - self.mu_
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return Z @ self.C_.T

    def predict_proba(self, X):
        s = 10.0 * self._sim(X)
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._sim(X), axis=1)]


class PLSDA:
    """PLS discriminant analysis: PLS regression onto one-hot classes. The
    standard chemometrics baseline for Raman classification."""

    def __init__(self, n=10):
        self.n = n

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        k = int(max(1, min(self.n, X.shape[0] - 1, X.shape[1])))
        self.sd_ = X.std(0)
        self.keep_ = self.sd_ > 1e-12                     # constant columns break PLS scaling
        self.pls_ = PLSRegression(n_components=k, scale=True).fit(X[:, self.keep_], Y)
        return self

    def predict_proba(self, X):
        P = np.clip(self.pls_.predict(np.asarray(X, float)[:, self.keep_]), 1e-6, None)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.pls_.predict(np.asarray(X, float)[:, self.keep_]), axis=1)]


# ================================================================== features
def peaks_of(record: dict, cfg: dict) -> list[dict]:
    """Parsed peaks; an empty string (no peak survived) is an empty list."""
    t = record.get("text") or ""
    return parse_string(t, cfg) if t else []


def render_lorentzian(peaks: list[dict], grid: np.ndarray) -> np.ndarray:
    """F2: each peak as a Lorentzian of its own FWHM, peak height = its intensity."""
    y = np.zeros_like(grid, dtype=float)
    for p in peaks:
        hw = max(float(p["width"]), 1.0) / 2.0
        y += p["intensity"] * hw ** 2 / ((grid - p["position"]) ** 2 + hw ** 2)
    return y


class SpectraStore:
    """Processed spectra saved by build_strings.py --save_spectra (F3).

    SERS npz holds x829 (829 = 1.0 scale, CB bands still in: exactly what peak
    picking sees) and xsub (x829 minus the CB reference). QM9S npz holds x
    (window-normalised). F3 is x829 for native SERS; for the matched variant it
    is xsub rescaled to max = 1, the same scale as QM9S."""

    def __init__(self, paths, cfg: dict):
        self.rows, self.grid = {}, None
        matched = bool(cfg["substrate"]["reference_norm"].get("rescale_to_max_after_discard"))
        self.schemas = set()
        for path in paths:
            d = np.load(path, allow_pickle=False)
            self.schemas.add(str(d["schema"]))
            self.grid = d["grid"]
            if "x" in d:
                X = d["x"]
            elif matched:
                X = d["xsub"] / np.maximum(d["xsub"].max(1, keepdims=True), 1e-9)
            else:
                X = d["x829"]
            for i, rid in enumerate(d["ids"]):
                self.rows[str(rid)] = X[i]

    def matrix(self, records) -> np.ndarray:
        missing = [r["id"] for r in records if r["id"] not in self.rows]
        if missing:
            raise KeyError(f"{len(missing)} records have no saved spectrum (e.g. {missing[:3]}); rerun "
                           "build_strings.py with --save_spectra (same --variant, same --spectra_frac)")
        return np.stack([self.rows[r["id"]] for r in records]).astype(np.float32)


def feature_matrix(records: list[dict], name: str, cfg: dict, spectra: SpectraStore | None = None,
                   bin_cm: float = 10.0) -> np.ndarray:
    if name == "F3":
        if spectra is None:
            raise KeyError("F3 needs saved spectra (build_strings.py --save_spectra)")
        return spectra.matrix(records)
    grid = make_grid(cfg)
    out = []
    for r in records:
        pk = peaks_of(r, cfg)
        if name == "F1":
            out.append(raster(pk, cfg, bin_cm))
        elif name == "F1pos":
            out.append(raster([{**p, "intensity": 1.0} for p in pk], cfg, bin_cm))
        elif name == "F2":
            out.append(render_lorentzian(pk, grid))
        else:
            raise ValueError(f"unknown feature {name!r}; choose from {FEATURES}")
    return np.asarray(out, dtype=np.float32)


# ================================================================== models
def make_model(name: str, hp: dict, seed: int = 0):
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000))
    if name == "pca_lr":
        from sklearn.decomposition import PCA
        return make_pipeline(StandardScaler(), PCA(n_components=0.95, svd_solver="full"),
                             LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000))
    if name == "plsda":
        return PLSDA(hp["n"])
    if name == "ncm":
        return NearestCentroid()
    if name == "hgb":
        # min_samples_leaf 5 (sklearn default 20): T1 trains on ~10-20 scans per
        # class, and with 20 a leaf can't isolate one class at all
        return HistGradientBoostingClassifier(class_weight="balanced", max_iter=100, learning_rate=0.1,
                                              max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=1.0,
                                              early_stopping=False, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=hp["k"], metric="cosine", weights="distance", algorithm="brute")
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def _fit(name, hp, X, y, seed):
    if name == "knn":
        hp = {"k": max(1, min(hp["k"], len(y) - 1))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_model(name, hp, seed).fit(X, y)


def _scores(model, X, binary):
    """binary: P(label = 1). Multi-class: predicted class."""
    if not binary:
        return model.predict(X)
    cls = list(model.classes_)
    if 1 not in cls:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, cls.index(1)]


# ================================================================== inner CV
def inner_splits(y, groups, max_splits: int = 5, seed: int = 0):
    """Grouped, stratified inner folds, or None when some class has fewer than 2
    groups (then no fold can keep every class on both sides)."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    y, groups = np.asarray(y), np.asarray(groups)
    per_class = {c: len(set(groups[y == c])) for c in np.unique(y)}
    if len(per_class) < 2 or min(per_class.values()) < 2:
        return None
    k = min(max_splits, min(per_class.values()))
    if len(set(groups)) == len(groups):                      # every sample its own group
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups) if len(np.unique(y[tr])) == len(per_class)]
    return folds or None


def _score_fn(binary):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    if binary:
        return lambda y, s: roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return lambda y, s: balanced_accuracy_score(y, s)


def best_threshold(y, s) -> float:
    """Threshold on P(1) that maximises balanced accuracy (training-side OOF scores)."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return 0.5
    cand = np.unique(np.r_[s, 0.5])
    best, thr = -1.0, 0.5
    pos, neg = (y == 1), (y == 0)
    for t in cand:
        p = s >= t
        ba = 0.5 * (p[pos].mean() + (~p[neg]).mean())
        if ba > best + 1e-12:
            best, thr = ba, float(t)
    return thr


def select_hp(name, X, y, groups, binary, seed=0, max_splits=5, default_hp=None, need_threshold=True):
    """Returns (hp, hp_source, threshold). Nested: only the training fold is seen."""
    default_hp = default_hp or DEFAULT_HP
    grid = GRIDS[name]
    need_oof = binary and name != "majority" and need_threshold
    if len(grid) == 1 and not need_oof:
        return dict(grid[0]) or dict(default_hp[name]), "fixed", 0.5
    folds = inner_splits(y, groups, max_splits, seed)
    if folds is None:
        return dict(default_hp[name]), "default", 0.5
    if len(grid) == 1 and len(y) > 5000:
        folds = folds[:1]          # threshold only, large training set: one inner split is enough
    score = _score_fn(binary)
    best = None
    for hp in grid:
        hp = hp or dict(default_hp[name])
        oof = np.full(len(y), np.nan) if binary else np.empty(len(y), dtype=object)
        seen = np.zeros(len(y), bool)
        for tr, te in folds:
            m = _fit(name, hp, X[tr], y[tr], seed)
            oof[te] = _scores(m, X[te], binary)
            seen[te] = True
        val = score(y[seen], oof[seen])
        if best is None or (np.isfinite(val) and (not np.isfinite(best[0]) or val > best[0] + 1e-9)):
            best = (val, hp, oof, seen)
    val, hp, oof, seen = best
    thr = best_threshold(y[seen], oof[seen].astype(float)) if binary else 0.5
    return hp, "inner_cv", thr


# ================================================================== one evaluation
def leave_pair_out(X, y, akey, model, seed=0, tune=True, max_splits=3, default_hp=None):
    """Leave-pair-out AUROC over analytes (Airola et al. 2011).

    Pooling leave-one-analyte-out scores biases AUROC downward: every fold's
    model has a different prior, and holding an analyte out always lowers the
    training rate of its own class (a constant "majority" model scores 0.0).
    Here each fold holds out one positive and one negative analyte, and only the
    two scores from the SAME model are compared.

    Returns {"pos", "neg": analyte keys, "W": [n_pos, n_neg] with 1 if the
    positive analyte's mean score is above the negative's (0.5 tie, 0 below),
    "S": the within-fold spectrum-level AUROC, "hp": Counter of chosen hp}."""
    default_hp = default_hp or DEFAULT_HP
    y, akey = np.asarray(y).astype(int), np.asarray(akey).astype(str)
    pos, neg = sorted(set(akey[y == 1])), sorted(set(akey[y == 0]))
    W = np.full((len(pos), len(neg)), np.nan)
    S = np.full((len(pos), len(neg)), np.nan)
    hps = Counter()
    members = {a: np.where(akey == a)[0] for a in pos + neg}
    for i, pa in enumerate(pos):
        for j, na in enumerate(neg):
            te = np.r_[members[pa], members[na]]
            tr = np.setdiff1d(np.arange(len(y)), te)
            if len(np.unique(y[tr])) < 2:
                continue
            if tune:
                hp, _, _ = select_hp(model, X[tr], y[tr], akey[tr], True, seed, max_splits, default_hp,
                                     need_threshold=False)
            else:
                hp = dict(default_hp[model])
            hps[str(hp)] += 1
            s = np.asarray(_scores(_fit(model, hp, X[tr], y[tr], seed), X[te], True), float)
            sp, sn = s[:len(members[pa])], s[len(members[pa]):]
            diff = float(sp.mean() - sn.mean())
            W[i, j] = 0.5 if abs(diff) <= 1e-12 else (1.0 if diff > 0 else 0.0)   # tolerance: constant models tie
            S[i, j] = fast_auroc(y[te], s)
    return {"pos": pos, "neg": neg, "W": W, "S": S, "hp": hps}


def lpo_ci(W, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap CI of the leave-pair-out AUROC: resample positive and negative
    analytes with replacement (no refitting)."""
    rng = np.random.default_rng(seed)
    P, N = W.shape
    vals = []
    for _ in range(n_boot):
        v = np.nanmean(W[np.ix_(rng.integers(0, P, P), rng.integers(0, N, N))])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_folds(Xtr_all, ytr_all, gtr_all, Xte_all, folds, model, binary, seed=0, fixed=None,
              max_splits=5, default_hp=None):
    """Fit on each fold's training indices, predict its test indices, pool.

    folds: list of (name, tr_idx, te_idx); tr indexes the *_tr_all arrays, te the
    Xte_all array (the same arrays for within-domain tasks; QM9S -> SERS for T4).
    fixed: per-fold [(hp, threshold)] to skip the inner CV (permutation runs).
    Returns {"score": pooled scores/predictions over Xte_all (NaN/None where not
    tested), "pred": thresholded predictions (binary), "fold": fold index,
    "folds": [per-fold info]}."""
    n = len(Xte_all)
    score = np.full(n, np.nan) if binary else np.empty(n, dtype=object)
    pred = np.full(n, -1)
    fold_of = np.full(n, -1)
    info = []
    for k, (fname, tr, te) in enumerate(folds):
        ytr = ytr_all[tr]
        if fixed is not None:
            hp, thr = fixed[k]
            src = "frozen"
        else:
            hp, src, thr = select_hp(model, Xtr_all[tr], ytr, gtr_all[tr], binary, seed, max_splits, default_hp)
        if len(np.unique(ytr)) < 2:
            s = np.full(len(te), float(np.unique(ytr)[0])) if binary else np.array([ytr[0]] * len(te), dtype=object)
        else:
            m = _fit(model, hp, Xtr_all[tr], ytr, seed)
            s = _scores(m, Xte_all[te], binary)
        score[te] = s
        if binary:
            pred[te] = (np.asarray(s, float) >= thr).astype(int)
        fold_of[te] = k
        info.append({"fold": fname, "hp": hp, "hp_source": src, "threshold": round(float(thr), 4),
                     "n_train": int(len(tr)), "n_test": int(len(te))})
    return {"score": score, "pred": pred, "fold": fold_of, "folds": info}


# ================================================================== metrics
def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on P(1)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary_metrics(y, score, pred, with_ece=False) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y).astype(int)
    out = {"auroc": np.nan, "auprc": np.nan, "bal_acc": np.nan}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, score))
        out["auprc"] = float(average_precision_score(y, score))
        out["bal_acc"] = float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()))
    if with_ece:
        out["ece"] = ece(y, score)
    return out


def multiclass_metrics(y, pred) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"bal_acc": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0)),
                "accuracy": float(accuracy_score(y, pred))}


def fast_auroc(y, s) -> float:
    """Mann-Whitney AUROC with average ranks for ties (= sklearn roc_auc_score)."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.asarray(s, float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_ap(y, s) -> float:
    """Average precision with sklearn's tie handling (one point per distinct score)."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return np.nan
    s = np.asarray(s, float)
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    thr_idx = np.r_[np.where(np.diff(ss))[0], len(ys) - 1]
    tps = np.cumsum(ys)[thr_idx]
    fps = 1 + thr_idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_binary(y, s, p, with_ece=False) -> dict:
    y = np.asarray(y).astype(int)
    out = {"auroc": np.nan, "auprc": np.nan, "bal_acc": np.nan}
    if 0 < y.sum() < len(y):
        out["auroc"] = fast_auroc(y, s)
        out["auprc"] = fast_ap(y, s)
        out["bal_acc"] = float(0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()))
    if with_ece:
        out["ece"] = ece(y, s)
    return out


def fast_multiclass(y, pred) -> dict:
    """Balanced accuracy, macro-F1 (over classes present in y) and accuracy."""
    y, pred = np.asarray(y), np.asarray(pred)
    rec, f1 = [], []
    for c in np.unique(y):
        t, pp = y == c, pred == c
        tp = float((t & pp).sum())
        rec.append(tp / t.sum())
        f1.append(2 * tp / (t.sum() + pp.sum()) if (t.sum() + pp.sum()) else 0.0)
    return {"bal_acc": float(np.mean(rec)), "macro_f1": float(np.mean(f1)), "accuracy": float((y == pred).mean())}


def bootstrap_all(fn, groups, n_boot=1000, seed=0, alpha=0.05) -> dict:
    """Grouped bootstrap of every metric fn(idx) returns: {metric: (lo, hi, n_ok)}."""
    uniq, _, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    acc: dict[str, list] = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        for k, v in fn(np.concatenate([members[i] for i in pick])).items():
            if v is not None and np.isfinite(v):
                acc.setdefault(k, []).append(v)
    out = {}
    for k, vals in acc.items():
        if len(vals) < max(10, n_boot // 10):
            out[k] = (np.nan, np.nan, len(vals))
        else:
            lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[k] = (float(lo), float(hi), len(vals))
    return out


def per_class_recall(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    return {c: float((pred[y == c] == c).mean()) for c in sorted(set(y))}


def grouped_bootstrap(metric_fn, groups, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI of metric_fn(indices) over resampled groups. metric_fn must
    accept an index array (with repeats) and return a float (NaN = skip)."""
    uniq, _, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        v = metric_fn(np.concatenate([members[i] for i in pick]))
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 10):
        return np.nan, np.nan, len(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(vals)


def _group_index(groups):
    """(unique groups, inverse, members): members[k] = indices of group k. O(n log n)."""
    uniq, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1]
    return uniq, inv, np.split(order, cuts)


def permute_by_group(y, groups, strata=None, rng=None):
    """Shuffle labels between groups (every member of a group keeps one shared
    label), separately within each stratum. The null: no association between a
    group's spectra and its label, with the group structure intact."""
    rng = rng or np.random.default_rng()
    y = np.asarray(y)
    uniq, inv, members = _group_index(groups)
    first = np.array([m[0] for m in members])
    glab = y[first]
    if not np.array_equal(glab[inv], y):
        bad = uniq[np.unique(inv[glab[inv] != y])[:3]]
        raise ValueError(f"label not constant within group(s) {list(bad)}")
    gstr = np.zeros(len(uniq), int) if strata is None else np.asarray(strata)[first]
    new = glab.copy()
    for st in np.unique(gstr):
        k = np.where(gstr == st)[0]
        new[k] = glab[k][rng.permutation(len(k))]
    return new[inv]


def majority_rate(y) -> float:
    c = Counter(np.asarray(y).tolist())
    return max(c.values()) / len(y)


def scaffold_split(smiles: list[str], test_frac: float = 0.2) -> tuple[np.ndarray, list[str]]:
    """Bemis-Murcko scaffold split (DeepChem style): scaffold sets sorted from
    largest to smallest fill the training set up to 1 - test_frac; the rest are
    test. Acyclic molecules share the empty scaffold, the largest set, so they
    land in training. Returns (is_test, scaffold per molecule)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaf = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m is not None else "?")
    sets: dict[str, list[int]] = {}
    for i, s in enumerate(scaf):
        sets.setdefault(s, []).append(i)
    order = sorted(sets.values(), key=lambda v: (-len(v), v[0]))
    n_train_max = (1 - test_frac) * len(smiles)
    is_test = np.zeros(len(smiles), bool)
    n = 0
    for members in order:
        if n + len(members) <= n_train_max:
            n += len(members)
        else:
            is_test[members] = True
    return is_test, scaf
