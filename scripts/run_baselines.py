#!/usr/bin/env python3
"""Step 2: classical baselines (roadmap section 2).

  python scripts/run_baselines.py --task T1 --variant native
  python scripts/run_baselines.py --task all --variant native --n_jobs 16
  python scripts/run_baselines.py --task T4 --variant matched
  python scripts/run_baselines.py --task T1 --old_manifest data/split_manifest.csv \
      --old_predictions data/test_predictions_example.csv

Inputs (from build_strings.py; add --save_spectra there for F3):
  <data_dir>/sers_records.jsonl, sers_spectra.npz        T1, T2, T4
  <data_dir>/qm9s_records.jsonl, qm9s_spectra.npz        T3, T4
  data_dir defaults to data/processed (native) or data/processed_<variant>.

Outputs in reports/baselines/<variant>/:
  <task>.csv            one row per scheme x test set x label x feature x model x metric:
                        value, 95% grouped-bootstrap CI, chance, permutation p (primary
                        metric, for LR-F1 and the best combination), n, hyperparameters
  <task>.md             headline table + gate check
  <task>.png            figure
  <task>_run.json       arguments, input hashes, timings
  T1_confounds.csv, T1_per_class.csv, T1_confusion_*.csv
  T2_labels.csv, T2_per_analyte.csv, T3_labels.csv, T4_labels.csv, T4_per_analyte.csv
  <task>_predictions.csv.gz   out-of-fold predictions of LR-F1 and the best combination
                              (for the paired comparison with the LLM in step 3)

Runtime: T1/T2 are minutes with --n_perm 1000 on a CPU node when the best model
is linear; a gradient-boosting winner makes the permutation test the slow part
(use --n_jobs). T3/T4 default to --n_perm 100 (all their p-values sit at the
floor; min p = 1/101) and a 20 % QM9S subsample (--qm9s_frac)."""
import argparse
import csv
import gzip
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit import baseline_tasks as T  # noqa: E402
from probe2circuit.baselines import FEATURES, MODELS  # noqa: E402
from probe2circuit.io import default_data_dir, load_config  # noqa: E402

PERM_DEFAULT = {"T1": 1000, "T2": 1000, "T3": 100, "T4": 100}
MAIN_COLS = ["task", "variant", "scheme", "test_set", "label", "feature", "model", "metric", "value", "ci_lo",
             "ci_hi", "chance", "p_perm", "n_perm", "perm_null_mean", "n_test", "n_test_groups", "n_boot_ok",
             "n_train", "hp_source", "hp", "seconds", "note"]


def write_csv(rows, path, cols=None):
    if not rows:
        return
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(r):
    if r is None or r.get("value", "") == "":
        return "n/a"
    ci = f" [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if r.get("ci_lo", "") != "" else ""
    p = f", p={r['p_perm']:.3g}" if r.get("p_perm", "") != "" else ""
    return f"{r['value']:.3f}{ci}{p}"


def pick(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return out


def fam(r, base):
    """Feature family: 'F3' matches F3, F3.bg, F3.d1 ...; 'F1' matches F1, F1.bg, F1.sqrt (not F1pos)."""
    return r["feature"] == base or r["feature"].startswith(base + ".")


def best_row(rows, family=None):
    c = [r for r in rows if r["model"] != "majority" and r["value"] != "" and not r["feature"].startswith("META")
         and (family is None or fam(r, family))]
    return max(c, key=lambda r: r["value"]) if c else None


def tag(r, feature=True):
    if not r:
        return ""
    return f" ({r['feature']}/{r['model']})" if feature else f" ({r['model']})"


def summary_md(task, out, ctx, args):
    rows = out["rows"]
    pm = "bal_acc" if task == "T1" else "auroc"
    L = [f"# {task} baselines ({ctx.variant})", ""]
    L.append(f"Primary metric: {'balanced accuracy' if task == 'T1' else 'AUROC'}. "
             "Values: point [95% bootstrap CI over cultures / analytes / molecules], p = group-permutation p-value.")
    L.append("\"Best\" columns pick the best non-majority model over a feature family (F3 = F3, F3.bg, F3.d1 ...). "
             "They are chosen on the test result, so read them as optimistic; LR on F1 is the pre-specified number.")
    if task == "T2":
        L.append("lpo:* rows are leave-pair-out AUROC at analyte level: one positive and one negative analyte are "
                 "held out together and only their two scores, from the same model, are compared. This replaces the "
                 "pooled leave-one-analyte-out AUROC of the first run, which was biased downward.")
    L += ["", "| scheme | test set | label | LR on F1 (string) | best F1 family | best F3 family | best overall |"
          + (" matrix-only |" if task == "T2" else ""), "|---|---|---|---|---|---|---|" + ("---|" if task == "T2" else "")]
    single = []
    keys = list(dict.fromkeys((r["scheme"], r["test_set"], r["label"]) for r in rows if r["metric"] == pm))
    for sc, ts, lab in keys:
        rr = pick(rows, scheme=sc, test_set=ts, label=lab, metric=pm)
        if rr and str(rr[0].get("note", "")).startswith("SINGLE ANALYTE"):
            single.append((sc, ts, lab))
            continue
        lr1 = next((r for r in rr if r["model"] == "lr" and r["feature"] == "F1"), None)
        b1, b3, bb = best_row(rr, "F1"), best_row(rr, "F3"), best_row(rr)
        line = (f"| {sc} | {ts} | {lab.replace('fg_', '')} | {fmt(lr1)} | {fmt(b1)}{tag(b1)} | {fmt(b3)}{tag(b3)} | "
                f"{fmt(bb)}{tag(bb)} |")
        if task == "T2":
            mm = next((r for r in rr if r["feature"] == "META_matrix"), None)
            line += f" {fmt(mm) if mm else ''} |"
        L.append(line)
    L.append("")
    if task == "T1":
        L += gate_T1(rows)
        det = out["extra"].get("detectability") or []
        if det:
            cols = [c for c in ("analyte", "r_cross", "r_cross_ctrl", "loso_recall_best", "top_bands_cm") if any(c in d for d in det)]
            rk = [c for c in det[0] if c.startswith("rank_")]
            sn = [c for c in det[0] if c.startswith("snr_")]
            rs = [c for c in det[0] if c.startswith("r_std_")]
            cols = cols[:2] + rk + rs + sn + cols[2:]
            L += ["", "## Detectability (no classifier involved)", "",
                  "An analyte's signature = its culture mean minus the strain mean. r_cross = correlation of the "
                  "signature between the two strains; rank = where the matching analyte ranks in the other strain "
                  "(1 = best match); r_std = correlation with the pure standard; snr = signature size / scan noise.", "",
                  "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            for d in det:
                L.append("| " + " | ".join(str(d.get(c, "")) for c in cols) + " |")
        crows = out.get("confound_rows") or []
        if crows:
            L += ["", "## Confound checks and the old result", "",
                  "| check | scheme | test set | feature | model | metric | value | chance | n |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for r in crows:
                show = (r["metric"] == "bal_acc" and r.get("test_set") == "all") or \
                       (r.get("check", "").startswith("old") and r["metric"] == "accuracy")
                if show:
                    L.append(f"| {r.get('check')} | {r['scheme']} | {r.get('test_set', '')} | {r['feature']} | "
                             f"{r['model']} | {r['metric']} | {fmt(r)} | {r.get('chance', '')} | {r.get('n_test', '')} |"
                             + (f" {r['note']}" if r.get('model') == 'old_llm' else ""))
            L += ["", "Metadata-only and substrate-only rows must sit at chance: a CI entirely above "
                  "chance means analyte is confounded with how the spectrum was taken."]
    if task == "T2":
        L += ["matrix-only = the same leave-pair-out AUROC from the sample type alone (BW cell / 536 cell / standard). "
              "When it is far from 0.5 in lpo:all, the label is confounded with sample type: read the lpo:cells and "
              "lpo:standards rows instead."]
    if single:
        L += ["", "## Labels with a single SERS analyte on one side", "",
              "AUROC here only ranks one molecule, so these are reported as that analyte's rank among all analytes "
              f"(`{task}_per_analyte.csv`, columns rank / n_analytes), not in the table above:", ""]
        L += [f"- {lab}" for lab in dict.fromkeys(x[2] for x in single)]
    if task in ("T2", "T3", "T4"):
        bad = [t for t in out["extra"].get("labels", []) if not t.get("testable") and t.get("reason")]
        if bad:
            L += ["", "## Labels not tested", ""]
            for t in bad:
                where = t.get("scheme") or t.get("matrix") or ""
                L.append(f"- {where + ': ' if where else ''}{t['label']}: {t['reason']}")
    L += ["", f"Arguments: `{' '.join(sys.argv[1:])}`"]
    return "\n".join(L) + "\n"


def gate_T1(rows):
    L = ["## Gate (roadmap 2.6)", ""]
    rr = pick(rows, scheme="loso", test_set="all", metric="bal_acc")
    if not rr:
        return L + ["No leave-one-strain-out result (needs two strains)."]
    ch = rr[0]["chance"]
    raw3 = best_row([r for r in rr if r["feature"] == "F3"])
    b3, b1 = best_row(rr, "F3"), best_row(rr, "F1")
    lr1 = next((r for r in rr if r["model"] == "lr" and r["feature"] == "F1"), None)
    pre3 = next((r for r in rr if r["feature"] == "F3.bg" and r["model"] == "ncm"), None)
    prec = next((r for r in rr if r["feature"] == "F3.ctrl" and r["model"] == "ncm"), None)
    pre1 = next((r for r in rr if r["feature"] == "F1.bg" and r["model"] == "lr"), None)
    for name, r in (("F3 as in the first run (no preprocessing)", raw3),
                    ("F3.bg with nearest centroid (pre-specified for 2b)", pre3),
                    ("F3.ctrl with nearest centroid (pre-specified for 2b; control culture subtracted)", prec),
                    ("best F3 family", b3),
                    ("LR on F1 (string, pre-specified)", lr1), ("LR on F1.bg (pre-specified for 2b)", pre1),
                    ("best F1 family", b1)):
        if r is None:
            L.append(f"- {name}: not run")
            continue
        above = r["ci_lo"] != "" and r["ci_lo"] > ch
        L.append(f"- {name}, leave-one-strain-out: {fmt(r)}{tag(r)}, chance {ch:.3f} -> "
                 + ("CI above chance." if above else "**CI includes chance.**"))
    rc = pick(rows, scheme="loso", test_set="all@culture", metric="bal_acc")
    bc = best_row(rc, "F3")
    if bc:
        L.append(f"- per culture (majority vote over a culture's scans, n = {bc['n_test']}): {fmt(bc)}{tag(bc)}")
    gate = [r for r in (raw3, pre3, prec) if r is not None]
    if gate and all(r["ci_lo"] == "" or r["ci_lo"] <= ch for r in gate):
        L.append("- Neither raw F3 nor the pre-specified F3.bg / F3.ctrl is distinguishable from chance under the grouped "
                 "split: per the roadmap, the SERS arm becomes a negative control and claims A/B rest on QM9S and "
                 "the standards.")
    if b1 is not None and b3 is not None:
        L.append(f"- best F3 family - best F1 family = {b3['value'] - b1['value']:+.3f}: how much the string format loses.")
    return L


# ------------------------------------------------------------------ figures
def figures(task, out, ctx, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        ctx.log("[fig] matplotlib not installed: figures skipped (pip install matplotlib)")
        return
    rows = [r for r in out["rows"] if "@" not in r["test_set"]]
    if not rows:
        ctx.log(f"[fig] {task}: no results, no figure")
        return
    if task == "T1":
        feats = list(dict.fromkeys(r["feature"] for r in rows))
        models = list(dict.fromkeys(r["model"] for r in rows))
        cmap = plt.get_cmap("tab20")
        colors = {f: cmap(i % 20) for i, f in enumerate(feats)}
        schemes = list(dict.fromkeys(r["scheme"] for r in rows))
        conf = out["extra"]["confusion"]
        best = out["extra"].get("best_loso")
        ncol = len(schemes) + (1 if conf else 0)
        fig, axes = plt.subplots(1, ncol, figsize=(6.4 * ncol, 4.6), squeeze=False)
        for ax, sc in zip(axes[0], schemes):
            rr = pick(rows, scheme=sc, test_set="all", metric="bal_acc")
            w = 0.85 / max(1, len(feats))
            for k, f in enumerate(feats):
                xs, ys, lo, hi = [], [], [], []
                for j, m in enumerate(models):
                    r = next((r for r in rr if r["feature"] == f and r["model"] == m), None)
                    if r and r["value"] != "":
                        xs.append(j + (k - (len(feats) - 1) / 2) * w)
                        ys.append(r["value"])
                        lo.append(r["value"] - (r["ci_lo"] if r["ci_lo"] != "" else r["value"]))
                        hi.append((r["ci_hi"] if r["ci_hi"] != "" else r["value"]) - r["value"])
                ax.bar(xs, ys, w, yerr=[lo, hi], color=colors[f], label=f, capsize=1, error_kw={"lw": 0.6})
            if rr:
                ax.axhline(rr[0]["chance"], color="k", ls="--", lw=1, label="chance")
            ax.set_xticks(range(len(models)), models)
            ax.set_ylim(0, 1)
            ax.set_title(f"T1 {sc}")
            ax.set_ylabel("balanced accuracy")
        axes[0][0].legend(fontsize=6, ncol=2)
        if conf:
            key = next((k for k in conf if k[0] == "loso" and best and k[1:] == tuple(best)), next(iter(conf)))
            M = conf[key].astype(float)
            M = M / np.maximum(M.sum(1, keepdims=True), 1)
            ax = axes[0][-1]
            ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
            codes = out["extra"].get("codes", {})
            cl = [codes.get(c, c.replace("L-", "")[:6]) for c in out["classes"]]
            ax.set_xticks(range(len(cl)), cl, rotation=90, fontsize=6)
            ax.set_yticks(range(len(cl)), cl, fontsize=6)
            ax.set_title(f"confusion {key[0]} {key[1]}/{key[2]}", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    else:
        pm = "auroc"
        panels = list(dict.fromkeys((r["scheme"], r["test_set"]) for r in rows if r["metric"] == pm))
        if task == "T4":
            panels = [p for p in panels if p[1] != "sers_all"] or panels
        series = [("LR on F1", "#1f77b4"), ("best F1 family", "#9467bd"), ("best F3 family", "#d62728")]
        n_lab = max(len({r["label"] for r in pick(rows, scheme=sc, test_set=ts)}) for sc, ts in panels)
        ncol = min(len(panels), 4)
        nrow = -(-len(panels) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, nrow * (0.42 * n_lab + 1.3)), squeeze=False)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        for ax, (sc, ts) in zip(axes.ravel(), panels):
            labels = list(dict.fromkeys(r["label"] for r in pick(rows, scheme=sc, test_set=ts, metric=pm)))
            for li, lab in enumerate(labels):
                rr = pick(rows, scheme=sc, test_set=ts, metric=pm, label=lab)
                picks = [next((r for r in rr if r["model"] == "lr" and r["feature"] == "F1"), None),
                         best_row(rr, "F1"), best_row(rr, "F3")]
                for k, (r, (name, col)) in enumerate(zip(picks, series)):
                    if not r or r["value"] == "":
                        continue
                    yv = li + (k - 1) * 0.22
                    if r["ci_lo"] != "":
                        ax.plot([r["ci_lo"], r["ci_hi"]], [yv, yv], color=col, lw=1)
                    ax.plot(r["value"], yv, "o", color=col, ms=4, label=name if li == 0 else None)
            ax.axvline(0.5, color="k", ls="--", lw=1)
            ax.set_yticks(range(len(labels)), [l.replace("fg_", "") for l in labels], fontsize=7)
            ax.set_xlim(0, 1.0)
            ax.invert_yaxis()
            ax.set_title(f"{task} {sc} / {ts}", fontsize=9)
            ax.set_xlabel("AUROC")
        axes[0][0].legend(fontsize=6, loc="best", framealpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ predictions
def save_predictions(task, out, ctx, path):
    specs, results = out["specs"], out["results"]
    n = 0
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "variant", "scheme", "label", "feature", "model", "id", "analyte", "y_true", "score",
                    "pred", "fold"])
        for i, sp in enumerate(specs):
            keep = {("F1", "lr"), T.best_combo(specs, results, i)}
            b3 = max(((f, m) for (j, f, m) in results if j == i and f.split(".")[0] == "F3" and m != "majority"),
                     key=lambda k: results[(i, *k)][1].get(sp.perm_subset, {}).get(T.PRIMARY[sp.binary], (-1,))[0],
                     default=None)
            keep.add(b3)
            for key in keep - {None}:
                if (i, *key) not in results:
                    continue
                res = results[(i, *key)][0]
                for k in np.where(res["fold"] >= 0)[0]:
                    pred = res["pred"][k] if sp.binary else res["score"][k]
                    score = round(float(res["score"][k]), 5) if sp.binary else ""
                    w.writerow([task, ctx.variant, sp.scheme, sp.label, key[0], key[1],
                                sp.te_ids[k] if sp.te_ids else k,
                                sp.te_analyte[k] if sp.te_analyte is not None else "",
                                sp.yte[k], score, pred, res["folds"][res["fold"][k]]["fold"]])
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", default=["T1"], help="T1 T2 T3 T4 or all")
    ap.add_argument("--variant", default="native")
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None, help="default reports/baselines/<variant>")
    ap.add_argument("--features", default=",".join(FEATURES))
    ap.add_argument("--models", default=None, help=f"comma list from {MODELS}; default per task "
                                                   "(all for T1/T2; majority,lr,hgb,knn for T3/T4)")
    ap.add_argument("--no_preps", action="store_true",
                    help="skip the preprocessing variants (F3.bg, F3.d1, ...): reproduces the first step-2 run")
    ap.add_argument("--control_regex", default=None,
                    help="regex on the spectrum id that marks no-analyte control cultures (T1 detectability)")
    ap.add_argument("--labels", default=None, help="comma list of fg_* labels (T2-T4); default all")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=None, help=f"default per task {PERM_DEFAULT}")
    ap.add_argument("--perm_budget_min", type=float, default=120.0,
                    help="wall-clock minutes for all permutation tests of a task; slow combinations get fewer "
                         "shuffles (never fewer than 100); the n used is in the n_perm column")
    ap.add_argument("--perm_for", default="primary,best", choices=["primary,best", "all", "none"])
    ap.add_argument("--n_jobs", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qm9s_frac", type=float, default=None,
                    help="QM9S subsample, default 0.2 (deterministic by id; must match build_strings "
                         "--spectra_frac for F3)")
    ap.add_argument("--min_pos", type=int, default=None,
                    help="QM9S training positives needed to test a label (default 50 for T3, 30 for T4)")
    ap.add_argument("--lr_C", type=float, default=None, help="LR C when no grouped inner CV is possible (default 0.1)")
    ap.add_argument("--knn_k", type=int, default=None, help="kNN k when no grouped inner CV is possible (default 5)")
    ap.add_argument("--old_manifest", default=None, help="old fine-tune split_manifest.csv (T1 comparison)")
    ap.add_argument("--old_predictions", default=None, help="old test_predictions csv (T1 comparison)")
    ap.add_argument("--meta_csv", default=None, help="extra metadata per spectrum (id,<cols>), e.g. date (T1)")
    ap.add_argument("--quick", action="store_true", help="smoke test: n_boot 100, n_perm 20, qm9s_frac 0.02")
    ap.add_argument("--no_figures", action="store_true")
    a = ap.parse_args()

    tasks = ["T1", "T2", "T3", "T4"] if "all" in a.task else a.task
    for opt in ("old_manifest", "old_predictions", "meta_csv"):
        if getattr(a, opt) and not Path(getattr(a, opt)).exists():
            raise SystemExit(f"--{opt} {getattr(a, opt)}: file not found")
    cfg = load_config(a.config, a.variant)
    data_dir = Path(a.data_dir or default_data_dir(a.variant))
    out_dir = Path(a.out_dir or f"reports/baselines/{a.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = tuple(f for f in a.features.split(",") if f)
    models = tuple(m for m in a.models.split(",") if m) if a.models else None
    bad = [f for f in feats if f not in FEATURES] + [m for m in (models or ()) if m not in MODELS]
    if bad:
        raise SystemExit(f"unknown feature/model {bad}; features {FEATURES}, models {MODELS}")

    for task in tasks:
        n_perm = a.n_perm if a.n_perm is not None else PERM_DEFAULT[task]
        ctx = T.Ctx(cfg=cfg, variant=a.variant, data_dir=data_dir, out_dir=out_dir, features=feats, models=models,
                    n_boot=a.n_boot, n_perm=n_perm, perm_for=a.perm_for, n_jobs=a.n_jobs, seed=a.seed,
                    qm9s_frac=a.qm9s_frac if a.qm9s_frac is not None else 0.2, min_pos=a.min_pos,
                    labels=tuple(a.labels.split(",")) if a.labels else None,
                    perm_budget_min=a.perm_budget_min, old_manifest=a.old_manifest, old_predictions=a.old_predictions, meta_csv=a.meta_csv,
                    figures=not a.no_figures, preps=not a.no_preps)
        if a.control_regex:
            ctx.control_regex = a.control_regex
        if a.lr_C is not None:
            ctx.default_hp["lr"] = {"C": a.lr_C}
            ctx.default_hp["pca_lr"] = {"C": a.lr_C}
        if a.knn_k is not None:
            ctx.default_hp["knn"] = {"k": a.knn_k}
        if a.quick:
            ctx.n_boot, ctx.n_perm, ctx.min_perm = min(ctx.n_boot, 100), min(ctx.n_perm, 20), 5
            if a.qm9s_frac is None:
                ctx.qm9s_frac = 0.02
        t0 = time.time()
        print(f"\n=== {task} ({a.variant}) -> {out_dir}")
        try:
            out = T.TASKS[task](ctx)
        except FileNotFoundError as e:
            print(f"[{task}] SKIPPED: {e}")
            continue
        write_csv(out["rows"], out_dir / f"{task}.csv", MAIN_COLS)
        ex = out.get("extra", {})
        if task == "T1":
            write_csv(out.get("confound_rows"), out_dir / "T1_confounds.csv", MAIN_COLS[:8] + ["check"] + MAIN_COLS[8:])
            write_csv(ex.get("per_class"), out_dir / "T1_per_class.csv")
            write_csv(ex.get("detectability"), out_dir / "T1_detectability.csv")
            for (sc, f, m), M in ex.get("confusion", {}).items():
                safe = sc.replace(":", "_").replace(">", "").replace("(", "_").replace(")", "")
                p = out_dir / f"T1_confusion_{safe}_{f}_{m}.csv"
                with open(p, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["true\\pred"] + out["classes"])
                    for c, row in zip(out["classes"], M):
                        w.writerow([c] + list(row))
        for key in ("labels", "per_analyte"):
            if ex.get(key):
                write_csv(ex[key], out_dir / f"{task}_{key}.csv")
        n_pred = save_predictions(task, out, ctx, out_dir / f"{task}_predictions.csv.gz")
        (out_dir / f"{task}.md").write_text(summary_md(task, out, ctx, a))
        if ctx.figures:
            figures(task, out, ctx, out_dir / f"{task}.png")
        inputs = {}
        for name in ("sers_records.jsonl", "qm9s_records.jsonl", "sers_spectra.npz", "qm9s_spectra.npz"):
            p = data_dir / name
            if p.exists():
                inputs[str(p)] = {"sha256": T._sha(p), "bytes": p.stat().st_size}
        import sklearn
        run = {"task": task, "variant": a.variant, "schema": cfg["schema"], "args": vars(a), "n_perm": ctx.n_perm,
               "n_boot": ctx.n_boot, "qm9s_frac": ctx.qm9s_frac, "default_hp": ctx.default_hp, "inputs": inputs,
               "n_rows": len(out["rows"]), "n_predictions": n_pred, "seconds": round(time.time() - t0, 1),
               "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}
        (out_dir / f"{task}_run.json").write_text(json.dumps(run, indent=1, default=str))
        print((out_dir / f"{task}.md").read_text())
        print(f"[{task}] done in {time.time() - t0:.0f} s -> {out_dir}/{task}.csv")


if __name__ == "__main__":
    main()
