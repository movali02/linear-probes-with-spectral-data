"""Coverage checks: is the SERS analyte set inside what QM9S can teach?

Three questions, each answered with a table:
  1. Chemical: is each analyte inside QM9's domain (<=9 heavy atoms, C/N/O/F),
     is it literally in QM9, and how close is its nearest QM9 neighbour?
  2. Labels: for each structural label, how many QM9 molecules carry it? A probe
     trained on QM9S cannot learn a label QM9 barely contains (thiol: zero).
  3. Spectral: do the two sources produce strings of similar shape (peak counts,
     widths, positions, token lengths)? And which peaks are shared by spectra of
     many DIFFERENT analytes (a sign of substrate / medium / session, not analyte)?
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .labels import FG_KEYS, canonical, mol_from_smiles, strip_isotopes, structural_labels
from .strings import estimate_tokens, parse_string

_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(smi):
    return _FPGEN.GetFingerprint(strip_isotopes(mol_from_smiles(smi)))


def chemical_coverage(analytes: list[dict], qm9_smiles: list[str], sim_threshold=0.5) -> list[dict]:
    qm9_canon = {}
    fps, keep = [], []
    for s in qm9_smiles:
        try:
            qm9_canon[canonical(s)] = s
            fps.append(_fp(s))
            keep.append(s)
        except ValueError:
            continue
    rows = []
    for a in analytes:
        lab = structural_labels(a["smiles"])
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(_fp(a["smiles"]), fps)) if fps else np.array([])
        k = int(sims.argmax()) if sims.size else None
        rows.append({
            "analyte": a["name"],
            "n_heavy": lab["n_heavy"],
            "elements": lab["elements"],
            "in_qm9_domain": lab["in_qm9_domain"],
            "exact_in_qm9": int(canonical(a["smiles"]) in qm9_canon),
            "nn_tanimoto": round(float(sims[k]), 3) if k is not None else None,
            "nn_smiles": keep[k] if k is not None else None,
            f"n_qm9_tanimoto_ge_{sim_threshold}": int((sims >= sim_threshold).sum()) if sims.size else 0,
            "why_out": _why_out(lab),
        })
    return rows


def _why_out(lab):
    r = []
    if lab["n_heavy"] > 9:
        r.append(f"{lab['n_heavy']} heavy atoms > 9")
    extra = set(lab["elements"]) - {"C", "N", "O", "F"}
    if extra:
        r.append("contains " + ",".join(sorted(extra)))
    if lab["has_isotope"]:
        r.append("isotopically labelled")
    return "; ".join(r)


def label_support(analytes: list[dict], qm9_smiles: list[str], min_support=100,
                  usable_smiles: set | None = None) -> list[dict]:
    """Count QM9 molecules carrying each label. If `usable_smiles` is given (QM9S
    molecules whose window string has enough peaks to be informative), count
    those too: that is the support a probe actually gets."""
    qm9_counts = defaultdict(int)
    use_counts = defaultdict(int)
    n_ok = 0
    for s in qm9_smiles:
        try:
            lab = structural_labels(s)
        except ValueError:
            continue
        n_ok += 1
        u = usable_smiles is not None and s in usable_smiles
        for k in FG_KEYS:
            qm9_counts[k] += lab[k]
            use_counts[k] += lab[k] * u
    an_with = defaultdict(list)
    for a in analytes:
        lab = structural_labels(a["smiles"])
        for k in FG_KEYS:
            if lab[k]:
                an_with[k].append(a["name"])
    rows = []
    for k in FG_KEYS:
        n = qm9_counts[k]
        nu = use_counts[k] if usable_smiles is not None else n
        row = {
            "label": k[3:],
            "n_qm9": n,
            "frac_qm9": round(n / max(n_ok, 1), 4),
        }
        if usable_smiles is not None:
            row["n_qm9s_usable"] = nu
        row.update({
            "n_analytes": len(an_with[k]),
            "analytes": ", ".join(an_with[k]),
            "status": ("unused" if not an_with[k] else
                       "NO QM9 SUPPORT" if nu == 0 else
                       "thin" if nu < min_support else "ok"),
        })
        rows.append(row)
    return rows


def spectral_summary(records: list[dict], cfg: dict, bin_cm=25.0) -> tuple[list[dict], dict]:
    by_src = defaultdict(list)
    for r in records:
        if r.get("text"):
            by_src[(r["source"], r.get("kind"))].append(r)
    lo, hi = cfg["window"]
    edges = np.arange(lo, hi + bin_cm, bin_cm)
    hists, rows = {}, []
    for key, rs in sorted(by_src.items()):
        peaks = [parse_string(r["text"], cfg) for r in rs]
        flat = [p for ps in peaks for p in ps]
        pos = np.array([p["position"] for p in flat])
        wid = np.array([p["width"] for p in flat])
        toks = np.array([estimate_tokens(r["text"]) for r in rs])
        h, _ = np.histogram(pos, edges)
        hists[key] = h / max(h.sum(), 1)
        rows.append({
            "source": key[0], "kind": key[1], "n_spectra": len(rs),
            "peaks_median": float(np.median([len(p) for p in peaks])),
            "peaks_range": f"{min(len(p) for p in peaks)}-{max(len(p) for p in peaks)}",
            "fwhm_median": round(float(np.median(wid)), 1) if wid.size else None,
            "fwhm_iqr": (f"{np.percentile(wid, 25):.0f}-{np.percentile(wid, 75):.0f}" if wid.size else None),
            "tokens_median_est": int(np.median(toks)),
            "tokens_max_est": int(toks.max()),
        })
    overlap = {}
    keys = sorted(hists)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            overlap[f"{a[0]}/{a[1]} vs {b[0]}/{b[1]}"] = round(float(np.minimum(hists[a], hists[b]).sum()), 3)
    return rows, overlap


def shared_peak_audit(records: list[dict], cfg: dict, tol=5.0, min_frac=0.5) -> list[dict]:
    """Peak positions that appear in spectra of many DIFFERENT analytes. A band
    shared by >= min_frac of analytes is more likely substrate, medium or session
    than chemistry, and a model can use it to tell sessions apart."""
    rs = [r for r in records if r.get("text") and r.get("analyte") and r.get("source") == "sers"
          and r.get("kind") != "substrate_only"]
    analytes = sorted({r["analyte"] for r in rs})
    if len(analytes) < 3:
        return []
    pos_by_an = defaultdict(list)
    for r in rs:
        pos_by_an[r["analyte"]] += [p["position"] for p in parse_string(r["text"], cfg)]
    allpos = np.sort(np.concatenate([np.array(v) for v in pos_by_an.values()]))
    # greedy clustering of positions within tol
    clusters, cur = [], [allpos[0]]
    for p in allpos[1:]:
        if p - cur[-1] <= tol:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    out = []
    for c in clusters:
        centre = float(np.median(c))
        hit = [a for a in analytes if any(abs(q - centre) <= tol for q in pos_by_an[a])]
        frac = len(hit) / len(analytes)
        if frac >= min_frac:
            out.append({"position": round(centre), "n_analytes": len(hit),
                        "frac_analytes": round(frac, 2), "analytes": ", ".join(hit)})
    return out


def to_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_(none)_\n"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
