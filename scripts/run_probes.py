#!/usr/bin/env python3
"""Probes along the shift ladder: the degradation figure (roadmap 4.2 + 5.3).

  python scripts/run_probes.py --variant matched --acts_dir data/activations/matched/Qwen3-8B
  python scripts/run_probes.py --variant matched --baseline_only        # no GPU, no activations

For every structural label that QM9S supports and the amino acids can test:
  1. fit a linear probe on QM9S train, for every saved layer and position
     (lr = logistic regression; mm = mass-mean, the difference of class means);
  2. choose C and the layer on a QM9S validation split (nothing about SERS is
     used to choose anything);
  3. score that probe on QM9S test (R0) and on every rung of the ladder, where
     rungs R2-R9 are the same amino acids, compared analyte by analyte.
The classical baseline (the binned string, F1) goes through the same procedure,
so the figure shows probe and baseline side by side. If --acts_random is given
(activations of the untrained model), that control is drawn as well.

Outputs in reports/ladder/<variant>/<model tag>/:
  degradation.csv       label x method x rung: AUROC, 95% CI (bootstrap over analytes), layer used
  degradation.png       the figure; degradation_centred.png = the same with each rung centred on
                        its own mean (a label-free recalibration)
  layers.csv / .png     AUROC against layer, for QM9S validation and every rung
  per_analyte.csv       the probe's mean score for every analyte on every rung
  ladder.md             summary table
AUROC on a rung with one positive analyte is that analyte's rank among the 20;
n_pos is in the table, read those rows as a rank."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit import ladder as L  # noqa: E402
from probe2circuit import probes as P  # noqa: E402
from probe2circuit.baselines import fast_auroc, feature_matrix  # noqa: E402
from probe2circuit.io import load_config, read_jsonl  # noqa: E402
from probe2circuit.pipeline import in_subsample  # noqa: E402

SERIES = {  # method key -> (legend, colour, line style, marker). Colours: validated categorical slots 1-3;
    # the untrained-model control is a recessive grey, not a data series.
    "probe_lr": ("probe: logistic regression", "#2a78d6", "-", "o"),
    "probe_mm": ("probe: mass-mean", "#eb6834", "-", "s"),
    "baseline_lr": ("classifier on the string (F1)", "#1baf7a", "-", "^"),
    "random_lr": ("probe on the untrained model", "#8a8a85", "--", "x"),
}


def strain_of(rid):
    p = rid.split(":")[-1].split("_")
    return p[1] if len(p) >= 3 else "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="matched")
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--ladder_dir", default=None)
    ap.add_argument("--acts_dir", default=None, help="data/activations/<variant>/<model tag>")
    ap.add_argument("--acts_random", default=None, help="activations of the untrained model (control)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--baseline_only", action="store_true")
    ap.add_argument("--baseline_feature", default="F1", choices=["F1", "F1pos"],
                    help="F1 = max intensity per 10 cm-1 bin; F1pos = which bins hold a peak (positions only: "
                         "the like-for-like baseline for activations extracted with --fields pos)")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--min_pos", type=int, default=30, help="QM9S training positives needed")
    ap.add_argument("--Cs", default="0.001,0.01,0.1")
    ap.add_argument("--positions", default="last,mean")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.acts_dir and not a.baseline_only:
        raise SystemExit("give --acts_dir, or --baseline_only for the classical curve alone")

    cfg = load_config(a.config, a.variant)
    ldir = Path(a.ladder_dir or f"data/ladder/{a.variant}")
    tag = Path(a.acts_dir).name if a.acts_dir else "baseline" + ("" if a.baseline_feature == "F1" else "-" + a.baseline_feature)
    out = Path(a.out_dir or f"reports/ladder/{a.variant}/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    Cs = [float(c) for c in a.Cs.split(",")]

    q = [r for r in read_jsonl(ldir / "qm9s_records.jsonl") if r.get("text")]
    lad = [r for r in read_jsonl(ldir / "ladder_records.jsonl") if r.get("text") and r.get("labels")]
    is_te = np.array([r.get("split") == "test" for r in q])
    is_va = np.array([(not t) and in_subsample(r["id"] + ":val", 0.15) for r, t in zip(q, is_te)])
    is_tr = ~is_te & ~is_va
    rung = np.array([r["rung"] for r in lad])
    analyte = np.array([r["analyte"] for r in lad], dtype=object)
    rungs = [x for x in L.ORDER[1:] if (rung == x).any()]
    paired = sorted(set.intersection(*[set(analyte[rung == x].tolist()) for x in rungs])) if rungs else []
    print(f"[probes] QM9S {int(is_tr.sum())} train / {int(is_va.sum())} validation / {int(is_te.sum())} test; "
          f"rungs {rungs}; {len(paired)} analytes present on every rung")
    if len(paired) < 4:
        raise SystemExit("fewer than 4 analytes are shared by all rungs: check data/ladder and the DFT sticks")
    in_pair = np.isin(analyte, paired)

    # ---- labels: enough QM9S positives, and both classes among the paired analytes
    cand = a.labels.split(",") if a.labels else [k for k in q[0]["labels"] if k.startswith("fg_")]
    lab_of = {}
    for r in lad:
        if r["analyte"] in paired:
            lab_of.setdefault(r["analyte"], r["labels"])
    labels, table = [], []
    for k in cand:
        ntr = sum(int(r["labels"].get(k, 0)) for r, t in zip(q, is_tr) if t)
        npos = sum(int(lab_of[x].get(k, 0)) for x in paired)
        ok = ntr >= a.min_pos and 0 < npos < len(paired)
        table.append({"label": k, "n_pos_qm9s_train": ntr, "n_pos_analytes": npos, "n_analytes": len(paired),
                      "pos_analytes": "; ".join(x for x in paired if lab_of[x].get(k)), "used": ok})
        if ok:
            labels.append(k)
    if not labels:
        raise SystemExit("no label has QM9S support and both classes among the analytes")
    K = len(labels)
    print(f"[probes] labels: {[k.replace('fg_', '') for k in labels]}")
    Yq = np.array([[int(r["labels"].get(k, 0)) for k in labels] for r in q], dtype=np.float32)
    Yl = np.array([[int(r["labels"].get(k, 0)) for k in labels] for r in lad], dtype=np.float32)
    dom = np.array([r["rung"] + ("|" + strain_of(r["id"]) if r["rung"] == "R9" else "") for r in lad])
    grp = np.array([r.get("group") or r["id"] for r in lad], dtype=object)

    def rung_auc(S):
        """{rung: (auroc[K], S_analyte, Y_analyte, names)} on the paired analytes."""
        res = {}
        for x in rungs:
            m = (rung == x) & in_pair
            Sa, Ya, names = P.by_analyte(S[m], Yl[m], analyte[m])
            res[x] = (P.auroc_cols(Ya, Sa), Sa, Ya, names)
        return res

    layer_rows, store = [], {}

    def run_source(group, position, layer, Xq, Xl):
        """Fit on QM9S train, choose C on validation, score everything. group: probe|random|baseline."""
        (Xtr, Xva, Xte, Xls), _ = P.standardise(Xq[is_tr], Xq[is_va], Xq[is_te], Xl)
        Xlc = P.centre_domains(Xls, dom, grp)
        fits = {}
        best = None
        for C in Cs:
            W, b = P.fit_logreg(Xtr, Yq[is_tr], C)
            va = P.auroc_cols(Yq[is_va], Xva @ W + b)
            if best is None or np.nanmean(va) > np.nanmean(best[0]):
                best = (va, W, b, C)
        fits["lr"] = best
        if group != "baseline":
            W, b = P.mass_mean(Xtr, Yq[is_tr])
            fits["mm"] = (P.auroc_cols(Yq[is_va], Xva @ W + b), W, b, None)
        for method, (va, W, b, C) in fits.items():
            key = f"{group}_{method}"
            Ste = Xte @ W + b
            r0 = P.auroc_cols(Yq[is_te], Ste)
            Sl, Slc = Xls @ W + b, Xlc @ W + b
            ra, rac = rung_auc(Sl), rung_auc(Slc)
            store[(key, position, layer)] = {"val": va, "R0": r0, "Ste": Ste, "ra": ra, "rac": rac, "C": C,
                                            "Sl": Sl, "W": W, "b": b}
            for j, k in enumerate(labels):
                base = {"method": key, "position": position, "layer": layer, "label": k, "C": C}
                layer_rows.append({**base, "set": "qm9s_val", "centred": 0, "auroc": va[j]})
                layer_rows.append({**base, "set": "R0", "centred": 0, "auroc": r0[j]})
                for x in rungs:
                    layer_rows.append({**base, "set": x, "centred": 0, "auroc": ra[x][0][j]})
                    layer_rows.append({**base, "set": x, "centred": 1, "auroc": rac[x][0][j]})

    t0 = time.time()
    bf = a.baseline_feature
    SERIES["baseline_lr"] = (f"classifier on the string ({bf})",) + SERIES["baseline_lr"][1:]
    run_source("baseline", "-", -1, feature_matrix(q, bf, cfg), feature_matrix(lad, bf, cfg))
    print(f"[probes] baseline (binned string, {bf}) done in {time.time() - t0:.0f} s")
    meta = {}
    for group, d in (("probe", a.acts_dir), ("random", a.acts_random)):
        if not d or a.baseline_only:
            continue
        from probe2circuit.activations import Acts
        Aq, Al = Acts(Path(d) / "qm9s"), Acts(Path(d) / "ladder")
        meta[group] = Aq.meta
        qi, li = [r["id"] for r in q], [r["id"] for r in lad]
        miss = [i for i in qi if i not in Aq.row][:3] + [i for i in li if i not in Al.row][:3]
        if miss:
            raise SystemExit(f"{d}: activations missing for {miss} ...: rerun extract_activations.py (no --limit)")
        for pos in a.positions.split(","):
            for k, layer in enumerate(Aq.layers):
                run_source(group, pos, layer, Aq.get(pos, k, qi), Al.get(pos, k, li))
            print(f"[probes] {group} / {pos}: {len(Aq.layers)} layers done ({time.time() - t0:.0f} s)")

    # ---- per label and method: the position and layer with the best QM9S validation AUROC
    methods = [m for m in SERIES if any(k[0] == m for k in store)]
    rng = np.random.default_rng(a.seed)
    deg, per_analyte, chosen = [], [], {}
    for m in methods:
        keys = [k for k in store if k[0] == m]
        for j, lab in enumerate(labels):
            kb = max(keys, key=lambda k: np.nan_to_num(store[k]["val"][j], nan=-1))
            chosen[(m, lab)] = kb
            s = store[kb]
            base = {"label": lab, "method": m, "position": kb[1], "layer": kb[2], "C": s["C"],
                    "qm9s_val_auroc": round(float(s["val"][j]), 4)}
            lo, hi = P.boot_ci(Yq[is_te][:, j], s["Ste"][:, j], a.n_boot, a.seed)
            deg.append({**base, "rung": "R0", "centred": 0, "auroc": s["R0"][j], "ci_lo": lo, "ci_hi": hi,
                        "n": int(is_te.sum()), "n_pos": int(Yq[is_te][:, j].sum())})
            for cen, src in ((0, "ra"), (1, "rac")):
                for x in rungs:
                    au, Sa, Ya, names = s[src][x]
                    lo, hi = P.boot_ci(Ya[:, j], Sa[:, j], a.n_boot, a.seed)
                    deg.append({**base, "rung": x, "centred": cen, "auroc": au[j], "ci_lo": lo, "ci_hi": hi,
                                "n": len(names), "n_pos": int(Ya[:, j].sum())})
                    if cen == 0:
                        order = np.argsort(-Sa[:, j])
                        rank = {names[i]: r + 1 for r, i in enumerate(order)}
                        for i, nm in enumerate(names):
                            per_analyte.append({"label": lab, "method": m, "rung": x, "analyte": nm,
                                                "true": int(Ya[i, j]), "score": round(float(Sa[i, j]), 4),
                                                "rank": rank[nm], "n": len(names)})
        # mean over labels, with a joint bootstrap over analytes
        for cen, src in ((0, "ra"), (1, "rac")):
            for x in ["R0"] + rungs:
                if x == "R0":
                    if cen:
                        continue
                    v = np.nanmean([store[chosen[(m, lab)]]["R0"][j] for j, lab in enumerate(labels)])
                    deg.append({"label": "MEAN", "method": m, "rung": "R0", "centred": 0, "auroc": v,
                                "ci_lo": np.nan, "ci_hi": np.nan, "n": int(is_te.sum())})
                    continue
                cols = [(store[chosen[(m, lab)]][src][x][1][:, j], store[chosen[(m, lab)]][src][x][2][:, j])
                        for j, lab in enumerate(labels)]
                v = np.nanmean([fast_auroc(yy, ss) for ss, yy in cols])
                n = len(cols[0][0])
                bs = []
                for _ in range(a.n_boot):
                    i = rng.integers(0, n, n)
                    aa = [fast_auroc(yy[i], ss[i]) for ss, yy in cols]
                    if np.isfinite(aa).any():
                        bs.append(np.nanmean(aa))
                lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 20 else (np.nan, np.nan))
                deg.append({"label": "MEAN", "method": m, "rung": x, "centred": cen, "auroc": v, "ci_lo": lo,
                            "ci_hi": hi, "n": n})

    # ---- paired differences between methods: same analytes resampled for both, so the CI is on the gap itself
    diffs = []
    for m1, m2 in (("probe_lr", "baseline_lr"), ("probe_lr", "random_lr"), ("probe_lr", "probe_mm")):
        if m1 not in methods or m2 not in methods:
            continue
        for x in rungs:
            c1 = [(store[chosen[(m1, lab)]]["ra"][x][1][:, j], store[chosen[(m1, lab)]]["ra"][x][2][:, j])
                  for j, lab in enumerate(labels)]
            c2 = [(store[chosen[(m2, lab)]]["ra"][x][1][:, j], store[chosen[(m2, lab)]]["ra"][x][2][:, j])
                  for j, lab in enumerate(labels)]
            mean = lambda cols, i: np.nanmean([fast_auroc(yy[i], ss[i]) for ss, yy in cols])  # noqa: E731
            n = len(c1[0][0])
            full = np.arange(n)
            d0 = mean(c1, full) - mean(c2, full)
            bs = []
            for _ in range(a.n_boot):
                i = rng.integers(0, n, n)
                with np.errstate(all="ignore"):
                    v = mean(c1, i) - mean(c2, i)
                if np.isfinite(v):
                    bs.append(v)
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 20 else (np.nan, np.nan))
            diffs.append({"a": m1, "b": m2, "rung": x, "diff": d0, "ci_lo": lo, "ci_hi": hi, "n": n})

    # ---- controls: shuffled labels at the chosen layer; R1 agreement (same molecule, QM9S vs your DFT)
    controls = {}
    if "probe_lr" in methods:
        from collections import Counter
        (m, pos, layer), _ = Counter(chosen[("probe_lr", lab)] for lab in labels).most_common(1)[0]
        from probe2circuit.activations import Acts
        Aq = Acts(Path(a.acts_dir) / "qm9s")
        Xq = Aq.get(pos, Aq.layers.index(layer), [r["id"] for r in q])
        (Xtr, Xva), _ = P.standardise(Xq[is_tr], Xq[is_va])
        W, b = P.fit_logreg(Xtr, Yq[is_tr][rng.permutation(int(is_tr.sum()))], store[(m, pos, layer)]["C"])
        controls["shuffled_labels_val_auroc"] = float(np.nanmean(P.auroc_cols(Yq[is_va], Xva @ W + b)))
        controls["shuffled_at"] = f"{pos} / layer {layer}"
    r1, r1q = rung == "R1", rung == "R1q"
    if r1.any() and r1q.any():
        from scipy.stats import spearmanr
        for m in [x for x in ("probe_lr", "baseline_lr") if x in methods]:
            rho = []
            for j, lab in enumerate(labels):
                S = store[chosen[(m, lab)]]["Sl"][:, j]
                common = sorted(set(analyte[r1].tolist()) & set(analyte[r1q].tolist()))
                x1 = [S[r1 & (analyte == c)].mean() for c in common]
                x2 = [S[r1q & (analyte == c)].mean() for c in common]
                if len(common) >= 4:
                    rho.append(spearmanr(x1, x2).correlation)
            if rho:
                controls[f"R1_agreement_{m}"] = float(np.nanmean(rho))
                controls["R1_n_molecules"] = len(common)

    # ---- write
    def w(path, rows):
        if rows:
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=cols)
                wr.writeheader()
                for r in rows:
                    wr.writerow({k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                                     else ("" if isinstance(v, (float, np.floating)) else v)) for k, v in r.items()})
    w(out / "degradation.csv", deg)
    w(out / "layers.csv", layer_rows)
    w(out / "per_analyte.csv", per_analyte)
    w(out / "labels.csv", table)
    w(out / "paired_differences.csv", diffs)
    (out / "run.json").write_text(json.dumps({"args": vars(a), "schema": cfg["schema"], "labels": labels,
                                              "paired_analytes": paired, "rungs": rungs, "controls": controls,
                                              "activations": meta, "seconds": round(time.time() - t0, 1)},
                                             indent=1, default=str))

    xs = ["R0"] + rungs
    md = [f"# Shift ladder: {tag} ({a.variant})", "",
          "AUROC of a probe fitted on QM9S, scored on each rung. R2 onwards are the same "
          f"{len(paired)} amino acids, one score per analyte. Layer and C were chosen on QM9S validation only.", ""]
    for cen in (0, 1):
        md += [f"## {'Each rung centred on its own mean (label-free recalibration)' if cen else 'As trained'}", "",
               "| label | n pos | method | layer | " + " | ".join(xs) + " |", "|---|---|---|---|" + "---|" * len(xs)]
        for lab in ["MEAN"] + labels:
            for m in methods:
                rr = {r["rung"]: r for r in deg if r["label"] == lab and r["method"] == m
                      and (r["centred"] == cen or r["rung"] == "R0")}
                if not rr:
                    continue
                any_r = next(iter(rr.values()))
                npos = next((r["n_pos"] for r in rr.values() if r["rung"] != "R0" and "n_pos" in r), "")
                where = f"{any_r.get('position', '')} {any_r.get('layer', '')}" if lab != "MEAN" else ""
                cells = []
                for x in xs:
                    r = rr.get(x)
                    if r is None or not np.isfinite(r["auroc"]):
                        cells.append("")
                    elif np.isfinite(r.get("ci_lo", np.nan)):
                        cells.append(f"{r['auroc']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")
                    else:
                        cells.append(f"{r['auroc']:.2f}")
                md.append(f"| {lab.replace('fg_', '')} | {npos} | {SERIES[m][0]} | {where} | " + " | ".join(cells) + " |")
        md.append("")
    if diffs:
        md += ["## Paired differences in the mean (same analytes resampled for both methods)", "",
               "A difference is resolved when its interval excludes 0.", "",
               "| a - b | " + " | ".join(rungs) + " |", "|---|" + "---|" * len(rungs)]
        for m1, m2 in dict.fromkeys((d["a"], d["b"]) for d in diffs):
            cells = []
            for x in rungs:
                d = next(d for d in diffs if (d["a"], d["b"], d["rung"]) == (m1, m2, x))
                res = np.isfinite(d["ci_lo"]) and (d["ci_lo"] > 0 or d["ci_hi"] < 0)
                c_ = f"{d['diff']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]"
                cells.append(f"**{c_}**" if res else c_)
            md.append(f"| {SERIES[m1][0]} - {SERIES[m2][0]} | " + " | ".join(cells) + " |")
        md.append("")
    single = [lab for lab in labels if next((t["n_pos_analytes"] for t in table if t["label"] == lab), 0) == 1]
    if single:
        md += ["## Labels with one positive analyte, as that analyte's rank", "",
               f"Rank 1 = the probe scores it highest of the {len(paired)} analytes. An AUROC here is only this rank.", "",
               "| label | analyte | method | " + " | ".join(rungs) + " |", "|---|---|---|" + "---|" * len(rungs)]
        for lab in single:
            for m in methods:
                rk = {r["rung"]: r for r in per_analyte if r["label"] == lab and r["method"] == m and r["true"] == 1}
                if rk:
                    md.append(f"| {lab.replace('fg_', '')} | {next(iter(rk.values()))['analyte']} | {SERIES[m][0]} | "
                              + " | ".join(str(rk[x]["rank"]) if x in rk else "" for x in rungs) + " |")
        md.append("")
    md += ["## Rungs", ""] + [f"- **{x}**: {L.RUNG_LABEL[x]}" for x in xs]
    if controls:
        md += ["", "## Controls", ""] + [f"- {k}: {v if isinstance(v, str) else round(v, 3)}" for k, v in controls.items()]
    sim = out.parent / "similarity_summary.csv"
    if sim.exists():
        md += ["", "## Spectral similarity to SERS (no model)", "", "| rung | vs | mean cosine | own standard best match | top 3 |",
               "|---|---|---|---|---|"]
        md += [f"| {r['rung']} | {r['target']} | {r['mean_cosine']} | {float(r['top1']):.0%} | {float(r['top3']):.0%} |"
               for r in csv.DictReader(open(sim))]
    (out / "ladder.md").write_text("\n".join(md) + "\n")
    print("\n".join(l for l in md if l.startswith(("#", "| label", "| MEAN"))))
    for k, v in controls.items():
        print(f"control  {k}: {v if isinstance(v, str) else round(v, 3)}")

    figures(out, deg, layer_rows, labels, methods, xs, tag)
    print(f"\n-> {out}/degradation.png, degradation.csv, ladder.md ({time.time() - t0:.0f} s)")


def figures(out, deg, layer_rows, labels, methods, xs, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig] matplotlib not installed: figures skipped")
        return
    ink, muted, grid = "#1f1f1e", "#6b6b66", "#e4e3dd"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": muted, "xtick.color": muted,
                         "ytick.color": muted, "text.color": ink, "axes.titlesize": 10, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    panels = ["MEAN"] + labels
    ncol = min(4, len(panels))
    nrow = -(-len(panels) // ncol)
    for cen, name in ((0, "degradation.png"), (1, "degradation_centred.png")):
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow + 0.6), squeeze=False, sharey=True)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        for ax, lab in zip(axes.ravel(), panels):
            for m in methods:
                rr = {r["rung"]: r for r in deg if r["label"] == lab and r["method"] == m
                      and (r["centred"] == cen or r["rung"] == "R0")}
                pts = [(i, rr[x]) for i, x in enumerate(xs) if x in rr and np.isfinite(rr[x]["auroc"])]
                if not pts:
                    continue
                leg, col, ls, mk = SERIES[m]
                xi = [i for i, _ in pts]
                ctl = m == "random_lr"          # the control sits behind the data series
                ax.plot(xi, [r["auroc"] for _, r in pts], color=col, ls=ls, marker=mk, lw=1.5 if ctl else 2, ms=6,
                        markeredgecolor=col if ctl else "#fcfcfb", markeredgewidth=1, label=leg, zorder=2 if ctl else 3)
                if m == "probe_lr":
                    ci = [(i, r) for i, r in pts if np.isfinite(r.get("ci_lo", np.nan))]
                    if ci:
                        ax.fill_between([i for i, _ in ci], [r["ci_lo"] for _, r in ci], [r["ci_hi"] for _, r in ci],
                                        color=col, alpha=0.13, lw=0, zorder=1)
                    i, r = pts[-1]
                    ax.annotate(f"{r['auroc']:.2f}", (i, r["auroc"]), xytext=(5, 0), textcoords="offset points",
                                va="center", fontsize=8, color=ink)
            ax.axhline(0.5, color=muted, lw=1, ls=":", zorder=0)
            npos = next((r.get("n_pos") for r in deg if r["label"] == lab and r["rung"] == xs[-1]), None)
            title = "mean over labels" if lab == "MEAN" else lab.replace("fg_", "").replace("_", " ")
            ax.set_title(title + (f"  ({npos} of {next(r['n'] for r in deg if r['label'] == lab and r['rung'] == xs[-1])} analytes)"
                                  if lab != "MEAN" and npos is not None else ""), loc="left")
            ax.set_xticks(range(len(xs)), xs)
            ax.set_xlim(-0.3, len(xs) - 0.3)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", color=grid, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for r_ in range(nrow):
            axes[r_][0].set_ylabel("AUROC")
        h, l_ = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l_, loc="lower center", ncol=len(h), frameon=False, fontsize=9)
        fig.suptitle(f"Probe trained on QM9S, scored along the ladder to SERS  ({tag}"
                     + (", each rung centred" if cen else "") + ")", x=0.01, ha="left", fontsize=11)
        fig.text(0.01, 0.955 - 0.0, "R0 QM9S held out · R2 amino acids, gas · R3 + water · R4 + protonation · "
                 "R5 + 785 nm intensity · R6 + gap field · R8 SERS standards · R9 SERS cells.  Dotted line: chance. "
                 "Band: 95% CI over analytes.", fontsize=7.5, color=muted, va="top")
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        fig.savefig(out / name, dpi=160)
        plt.close(fig)

    # AUROC against layer: QM9S validation and the SERS rungs, logistic-regression probe
    pr = [r for r in layer_rows if r["method"] == "probe_lr" and r["centred"] == 0]
    if pr:
        sets = [s for s in ("qm9s_val", "R4", "R8", "R9") if any(r["set"] == s for r in pr)]
        cols = dict(zip(("qm9s_val", "R4", "R8", "R9"), ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")))
        positions = sorted({r["position"] for r in pr})
        fig, axes = plt.subplots(len(positions), ncol, figsize=(3.6 * ncol, 2.7 * len(positions) + 0.6),
                                 squeeze=False, sharey=True)
        show = (["MEAN"] + labels)[:ncol]
        for ri, pos in enumerate(positions):
            for ax, lab in zip(axes[ri], show):
                for s in sets:
                    rr = [r for r in pr if r["position"] == pos and r["set"] == s and (lab == "MEAN" or r["label"] == lab)]
                    lay = sorted({r["layer"] for r in rr})
                    ys = [np.nanmean([r["auroc"] for r in rr if r["layer"] == L_]) for L_ in lay]
                    ax.plot(lay, ys, color=cols[s], lw=2, label={"qm9s_val": "QM9S validation"}.get(s, s))
                ax.axhline(0.5, color=muted, lw=1, ls=":")
                ax.set_title(f"{'mean over labels' if lab == 'MEAN' else lab.replace('fg_', '')} · position: {pos}", loc="left")
                ax.set_xlabel("layer")
                ax.set_ylim(0, 1.02)
                ax.grid(axis="y", color=grid, lw=0.8)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            axes[ri][0].set_ylabel("AUROC")
        h, l_ = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l_, loc="lower center", ncol=len(h), frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out / "layers.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
