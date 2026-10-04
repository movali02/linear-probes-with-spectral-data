"""Leakage checks. Each returns {"check", "status", "detail", ...}.

status: PASS / FAIL / WARN / SKIP. A FAIL means the dataset should not be used
for training or for reporting a number until it is fixed."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from .labels import canonical
from .strings import count_tokens, grammar, parse_string, raster, to_string


def _res(check, status, detail, **kw):
    return {"check": check, "status": status, "detail": detail, **kw}


# ------------------------------------------------------------ 1. grammar
def check_grammar(records, cfg):
    """Whitelist check: every model input must parse under the schema grammar and
    re-serialise to itself. Numbers and separators only, so no names, SMILES,
    formulas, file ids, media or assignments can be present."""
    g = grammar(cfg)
    bad = []
    for r in records:
        t = r.get("text", "")
        if not t:
            bad.append((r["id"], "empty string"))
            continue
        if not g.match(t):
            bad.append((r["id"], t[:60]))
            continue
        if to_string(parse_string(t, cfg), cfg) != t:
            bad.append((r["id"], "does not round-trip"))
    if bad:
        return _res("grammar_whitelist", "FAIL", f"{len(bad)}/{len(records)} inputs break the schema",
                    examples=bad[:5])
    return _res("grammar_whitelist", "PASS", f"all {len(records)} inputs are pure peak lists and round-trip")


# ------------------------------------------------------------ 2. forbidden terms
def label_terms(analytes, records=()):
    terms = set()
    for a in analytes:
        name = a["name"]
        terms |= {name, name.replace("L-", ""), name.replace(" ", "_"), a.get("formula") or "",
                  a["smiles"]}
        if len(a.get("code", "")) >= 3:
            terms.add(a["code"])
        try:
            terms.add(canonical(a["smiles"]))
        except ValueError:
            pass
    terms |= {"SMILES", "medium", "scaffold", "M9", "formula", "functional"}
    return sorted(t for t in terms if len(t) >= 3)


def _term_regex(terms):
    alpha = [re.escape(t) for t in terms if re.fullmatch(r"[A-Za-z][A-Za-z\- ]*", t)]
    other = [re.escape(t) for t in terms if not re.fullmatch(r"[A-Za-z][A-Za-z\- ]*", t)]
    parts = []
    if alpha:
        parts.append(r"(?<![A-Za-z])(?:" + "|".join(alpha) + r")(?![A-Za-z])")
    if other:
        parts.append("(?:" + "|".join(other) + ")")
    return re.compile("|".join(parts), re.IGNORECASE)


def check_forbidden_terms(records, analytes, field="text"):
    """Blacklist check (belt and braces on top of the grammar)."""
    rx = _term_regex(label_terms(analytes))
    # file ids / groups: checked as whole tokens (a regex alternation over tens of
    # thousands of ids is far too slow, and inputs are whitespace/|-delimited)
    ids = {r["id"] for r in records} | {r.get("group") for r in records if r.get("group")}
    hits = []
    for r in records:
        t = r.get(field, "") or ""
        m = rx.search(t)
        if m:
            hits.append((r["id"], m.group(0)))
            continue
        bad = next((tok for tok in re.split(r"[\s|:,;]+", t) if tok in ids), None)
        if bad:
            hits.append((r["id"], bad))
    if hits:
        return _res("forbidden_terms", "FAIL", f"{len(hits)} inputs contain label/id terms", examples=hits[:5])
    return _res("forbidden_terms", "PASS", "no analyte names, codes, SMILES, formulas or ids in inputs")


def check_prompt_template(template, analytes):
    if template.count("{text}") != 1:
        return _res("prompt_template", "FAIL", "template must contain {text} exactly once")
    m = _term_regex([t for t in label_terms(analytes) if t not in {"formula"}]).search(template)
    if m:
        return _res("prompt_template", "FAIL", f"template contains label term {m.group(0)!r}")
    return _res("prompt_template", "PASS", "fixed instruction, no label terms")


# ------------------------------------------------------------ 3. splits
def check_split_disjoint(records):
    if not all("split" in r for r in records):
        return _res("split_disjoint", "SKIP", "records have no split field")
    ids = Counter(r["id"] for r in records)
    dup_ids = [k for k, v in ids.items() if v > 1]
    gs = defaultdict(set)
    for r in records:
        gs[r["group"]].add(r["split"])
    straddle = [g for g, s in gs.items() if len(s) > 1]
    if dup_ids or straddle:
        return _res("split_disjoint", "FAIL",
                    f"{len(dup_ids)} duplicated ids, {len(straddle)} groups in both splits",
                    examples=(dup_ids + straddle)[:5])
    return _res("split_disjoint", "PASS", f"{len(gs)} groups, none spans train and test")


# ------------------------------------------------------------ 4. duplicates
def _get(r, key):
    """label_key may be nested: 'labels.fg_aromatic_ring'."""
    v = r
    for part in key.split("."):
        v = v.get(part) if isinstance(v, dict) else None
    return v


def check_duplicates(records, cfg, near=0.98, label_key="analyte", max_test=3000, seed=0):
    """`label_key` here is the IDENTITY of the sample (analyte for SERS, SMILES
    for QM9S), not a coarse probe label: identical input + same identity across
    splits is leakage; identical input + different identity is an ambiguous input."""
    """Exact duplicates across splits are a FAIL. Near-duplicates (cosine of the
    binned peak vector >= `near`) across splits are reported: same label means
    replicate leakage (fix with grouping), different label means label noise or a
    spectrum dominated by substrate."""
    if not all("split" in r for r in records):
        return _res("duplicates", "SKIP", "records have no split field")
    by_text = defaultdict(list)
    for r in records:
        by_text[r["text"]].append(r)
    # identical input, same label, both splits = the test item is in training: FAIL.
    # identical input, different labels = the input can't determine the label: WARN.
    exact_same, exact_diff = [], []
    for v in by_text.values():
        if len(v) > 1 and len({x["split"] for x in v}) > 1:
            labs = {str(_get(x, label_key)) for x in v}
            (exact_same if len(labs) < len(v) else exact_diff).append([x["id"] for x in v][:4])
    tr = [r for r in records if r["split"] == "train"]
    te = [r for r in records if r["split"] == "test"]
    sampled = ""
    if len(te) > max_test:
        idx = np.random.default_rng(seed).choice(len(te), max_test, replace=False)
        te = [te[i] for i in idx]
        sampled = f" (near-dup scan on a random {max_test} test items)"
    near_same = near_diff = 0
    near_ex = []
    if tr and te:
        A = np.stack([raster(parse_string(r["text"], cfg), cfg) for r in tr]).astype(np.float32)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        for b0 in range(0, len(te), 500):
            chunk = te[b0:b0 + 500]
            B = np.stack([raster(parse_string(r["text"], cfg), cfg) for r in chunk]).astype(np.float32)
            B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
            S = B @ A.T
            for i, j in zip(*np.where(S >= near)):
                same = _get(chunk[i], label_key) == _get(tr[j], label_key)
                near_same += same
                near_diff += not same
                if len(near_ex) < 5:
                    near_ex.append((chunk[i]["id"], tr[j]["id"], round(float(S[i, j]), 3)))
    detail = (f"exact cross-split duplicates: {len(exact_same)} same-label, {len(exact_diff)} "
              f"different-label; near-duplicate pairs (cos>={near}): {near_same} same-label, "
              f"{near_diff} different-label{sampled}")
    if exact_same:
        return _res("duplicates", "FAIL", detail, examples=exact_same[:5])
    status = "WARN" if (exact_diff or near_same or near_diff) else "PASS"
    return _res("duplicates", status, detail, examples=(exact_diff[:3] + near_ex)[:5])


# ------------------------------------------------------------ 5. token budget
def check_token_budget(records, cfg, tokenizer=None):
    prompts = [cfg["prompt_template"].format(text=r["text"]) for r in records]
    n = np.array(count_tokens(prompts, tokenizer))
    limit = cfg["max_seq_length"] - cfg["answer_token_reserve"]
    over = [(r["id"], int(k)) for r, k in zip(records, n) if k > limit]
    how = "exact (" + tokenizer + ")" if tokenizer else "estimated (1 token per digit)"
    detail = f"prompt tokens {how}: median {int(np.median(n))}, max {int(n.max())}, limit {limit}"
    if over:
        return _res("token_budget", "FAIL", detail + f"; {len(over)} over the limit", examples=over[:5])
    return _res("token_budget", "PASS", detail, max_tokens=int(n.max()))


# ------------------------------------------------------------ 6/7. controls
def _xy(records, cfg, label_key):
    X = np.stack([raster(parse_string(r["text"], cfg), cfg) for r in records])
    y = np.array([_get(r, label_key) for r in records])
    g = np.array([r["group"] for r in records])
    return X, y, g


def _cv_acc(X, y, g, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    counts = Counter(y)
    k = min(5, min(counts.values()))
    if k < 2 or len(counts) < 2:
        return None
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    correct = 0
    for tr, te in cv.split(X, y, g):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        correct += int((clf.predict(X[te]) == y[te]).sum())
    return correct / len(y)


def shuffled_label_control(records, cfg, label_key="analyte", n_perm=30, seed=0, max_records=4000):
    """Logistic regression on binned peak features, group-aware CV.
    Real labels -> the LR baseline the LLM has to beat.
    Shuffled labels -> must sit at chance; if not, the pipeline or the CV leaks."""
    rs = [r for r in records if _get(r, label_key) is not None and r.get("kind") != "substrate_only"
          and r.get("text")]
    if len(rs) > max_records:
        idx = np.random.default_rng(seed).choice(len(rs), max_records, replace=False)
        rs = [rs[i] for i in idx]
    if len(rs) < 6:
        return _res("shuffled_label_control", "SKIP", f"only {len(rs)} labelled records")
    X, y, g = _xy(rs, cfg, label_key)
    real = _cv_acc(X, y, g, seed)
    if real is None:
        return _res("shuffled_label_control", "SKIP", "need >=2 samples in every class and >=2 classes")
    rng = np.random.default_rng(seed)
    perm = [a for a in (_cv_acc(X, rng.permutation(y), g, seed + i + 1) for i in range(n_perm)) if a is not None]
    majority = Counter(y).most_common(1)[0][1] / len(y)
    pmean, psd = float(np.mean(perm)), float(np.std(perm))
    p_value = (1 + sum(a >= real for a in perm)) / (1 + len(perm))
    status = "FAIL" if pmean > majority + 2 * psd + 0.05 else "PASS"
    return _res("shuffled_label_control", status,
                f"[{label_key}, n={len(rs)}] LR baseline acc {real:.3f} (majority class {majority:.3f}); "
                f"shuffled-label acc {pmean:.3f} +/- {psd:.3f}; permutation p = {p_value:.3f}",
                lr_accuracy=real, majority_rate=majority, shuffled_mean=pmean, p_value=p_value)


def substrate_only_control(records, cfg, label_key="analyte", margin=0.10):
    """Train on analyte spectra, predict the label of CB-only spectra from the same
    runs. Substrate-only spectra contain no analyte, so accuracy must be at chance.
    Above chance means the classifier (and probably the LLM) is reading something
    other than the analyte: session, day, substrate batch, medium."""
    from sklearn.linear_model import LogisticRegression
    tr = [r for r in records if _get(r, label_key) and r.get("kind") != "substrate_only"]
    te = [r for r in records if _get(r, label_key) and r.get("kind") == "substrate_only"]
    labs = {_get(r, label_key) for r in tr}
    te = [r for r in te if _get(r, label_key) in labs]
    if len(te) < 5 or len(labs) < 2:
        return _res("substrate_only_control", "SKIP",
                    f"{len(te)} substrate-only spectra with a trained label (need >=5)")
    Xtr, ytr, _ = _xy(tr, cfg, label_key)
    Xte, yte, _ = _xy(te, cfg, label_key)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    chance = Counter(yte).most_common(1)[0][1] / len(yte)
    status = "FAIL" if acc > chance + margin else "PASS"
    return _res("substrate_only_control", status,
                f"accuracy on substrate-only spectra {acc:.3f} vs chance {chance:.3f}",
                accuracy=acc, chance=chance)


def run_all(records, cfg, analytes, tokenizer=None, label_key="analyte", identity_key=None):
    return [
        check_grammar(records, cfg),
        check_forbidden_terms(records, analytes),
        check_prompt_template(cfg["prompt_template"], analytes),
        check_split_disjoint(records),
        check_duplicates(records, cfg, label_key=identity_key or label_key),
        check_token_budget(records, cfg, tokenizer),
        shuffled_label_control(records, cfg, label_key),
        substrate_only_control(records, cfg, label_key),
    ]


# ------------------------------------------------------------ legacy audit
_OLD_PREFIX_GROUPS = ["Laser", "Substrate", "Spectrum", "DFT", "Molecule"]
_OLD_PEAK_FIELDS = ["peak_start", "peak_position", "peak_end", "intensity_number", "intensity_area",
                    "intensity_bin", "mode_assignment", "mode_assignment_source", "mode_assignment_references"]


def legacy_serialise(d: dict) -> str:
    """Re-implementation of the old notebook's serialise() (key-tagged prefix +
    9 positional fields per peak) so we can audit exactly what the model saw."""
    na = "NA"
    val = lambda v: na if v is None or v == "" else str(v)
    prefix = " ".join(f"{k}:{val(v)}" for g in _OLD_PREFIX_GROUPS if g in d for k, v in d[g].items())
    peaks = " | ".join(" ".join(val(p.get(f)) for f in _OLD_PEAK_FIELDS) for p in d.get("Peaks", []))
    return prefix + ("  PEAKS  " + peaks if peaks else "")


def audit_legacy_json(docs: dict[str, dict], analytes: list[dict], new_records: list[dict] | None = None,
                      substrate_catalogue: list[float] | None = None, tol: float = 4.0) -> list[dict]:
    """docs: {file_stem: parsed JSON}. Reports every field that carries the label
    or lets the model tell spectrum types apart without reading peaks, and puts
    the old string next to the new one for the same spectrum."""
    from .strings import estimate_tokens
    new_by_id = {r["id"]: r for r in (new_records or [])}
    rx = _term_regex([t for t in label_terms(analytes) if t not in {"SMILES", "medium", "scaffold",
                                                                    "M9", "formula", "functional"}])
    rows = []
    for stem, d in docs.items():
        s = legacy_serialise(d)
        leaks = []
        for g in _OLD_PREFIX_GROUPS:
            for k, v in (d.get(g) or {}).items():
                if v is not None and rx.search(str(v)):
                    leaks.append(f"{g}.{k}={v}")
        mol = d.get("Molecule") or {}
        if any(v is not None for v in mol.values()):
            leaks.append("Molecule block filled (SMILES/formula/scaffold/functional_groups)")
        n_assign = sum(p.get("mode_assignment") is not None for p in d.get("Peaks", []))
        if n_assign:
            leaks.append(f"{n_assign} peaks carry mode_assignment looked up by analyte name")
        meta = d.get("_meta") or d.get("meta") or {}
        if meta and rx.search(" ".join(str(v) for v in meta.values())):
            leaks.append("meta name/path contains label (safe only if never serialised)")
        old_pos = [p["peak_position"] for p in d.get("Peaks", [])]
        n_sub_old = sum(any(abs(p - q) <= tol for q in substrate_catalogue) for p in old_pos) \
            if substrate_catalogue else None
        nr = new_by_id.get(stem)
        rows.append({"file": stem, "old_peaks": len(old_pos), "old_substrate_peaks": n_sub_old,
                     "old_tokens_est": _est(s),
                     "new_peaks": nr["n_peaks"] if nr else None,
                     "new_tokens_est": estimate_tokens(nr["text"]) if nr else None,
                     "leaks_in_old": "; ".join(leaks) or "none found"})
    return rows


def _est(s):
    from .strings import estimate_tokens
    return estimate_tokens(s)
