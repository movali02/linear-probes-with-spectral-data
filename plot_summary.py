"""
Summary figures for the probe2circuit README.

Reads the matched-molecule ladder results under reports/ladder/matched/ and
writes three PNGs into assets/. Run from the repo root, inside mv_env:

    python plot_summary.py

What it draws:
  Figure 1  assets/layer_sweep.png      probe AUROC vs layer, in-domain, per concept, with the raw-peak baseline
  Figure 2  assets/degradation.png      probe AUROC vs shift-ladder rung, per concept, with CIs
  Figure 3  assets/transfer_scatter.png in-domain vs most-shifted-rung AUROC, per concept

Point REPORTS at a different run (e.g. "reports/ladder/v2mf") to plot that one.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path("reports/ladder/matched")
PROBE   = REPORTS / "Qwen3-8B"
BASE    = REPORTS / "baseline"
ASSETS  = Path("assets"); ASSETS.mkdir(exist_ok=True)

PROBE_METHOD, BASE_METHOD, POSITION = "probe_lr", "baseline_lr", "mean"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#000000", "#999999"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def pretty(label: str) -> str:
    return label.replace("fg_", "").replace("_", " ")

def rung_order(rungs):
    """Sort rung names like R0, R2, R3 by their numeric part."""
    return sorted(set(rungs), key=lambda r: int(re.sub(r"\D", "", str(r)) or 0))

def concept_colors(labels):
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}

# ----------------------------------------------------------------------
# Figure 1 — layer sweep, in-domain, with the raw-peak baseline
# ----------------------------------------------------------------------
def fig_layer_sweep():
    df = pd.read_csv(PROBE / "layers.csv")
    probe = df[(df.method == PROBE_METHOD) & (df["set"] == "qm9s_val")]
    if "position" in probe.columns:
        probe = probe[probe.position == POSITION]
    # best AUROC per (label, layer) across any C; layers are numeric here
    probe = probe[pd.to_numeric(probe.layer, errors="coerce").notna()].copy()
    probe.layer = probe.layer.astype(float)
    piv = probe.groupby(["label", "layer"]).auroc.max().reset_index()
    labels = sorted(piv.label.unique())
    colors = concept_colors(labels)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for lab in labels:
        s = piv[piv.label == lab].sort_values("layer")
        ax.plot(s.layer, s.auroc, color=colors[lab], lw=1.8, label=pretty(lab))

    # raw-peak baseline (logistic regression on the spectrum, no LLM): mean across concepts
    base = df[(df.method == BASE_METHOD) & (df["set"] == "qm9s_val")]
    if len(base):
        b = base.auroc.mean()
        ax.axhline(b, ls="--", lw=1.2, color="#777777")
        ax.text(piv.layer.max(), b, f"  raw-peak baseline ({b:.2f})",
                va="center", fontsize=8, color="#555555")
    ax.axhline(0.5, ls=":", lw=1, color="#bbbbbb")

    ax.set_xlabel("Layer"); ax.set_ylabel("In-domain AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax.set_title("Where each concept is linearly readable (QM9S validation)")
    fig.tight_layout(); fig.savefig(ASSETS / "layer_sweep.png", dpi=200); plt.close(fig)
    return labels

# ----------------------------------------------------------------------
# Figure 2 — degradation across the shift ladder (probe, at its chosen layer)
# ----------------------------------------------------------------------
def fig_degradation(colors):
    df = pd.read_csv(PROBE / "degradation.csv")
    rungs = rung_order(df.rung)
    x = {r: i for i, r in enumerate(rungs)}

    def mean_curve(method):
        q = df[(df.method == method) & (df.label == "MEAN")]
        if "centred" in q.columns:
            q = q.groupby("rung", as_index=False).auroc.mean()
        return [float(q[q.rung == r].auroc.mean()) if len(q[q.rung == r]) else np.nan for r in rungs]

    probe = df[(df.method == PROBE_METHOD) & (df.position == POSITION) & (df.label != "MEAN")]
    labels = sorted(probe.label.unique())

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    # per-concept probe lines, faded so the aggregates stand out
    for lab in labels:
        s = probe[probe.label == lab].copy()
        s["xi"] = s.rung.map(x); s = s.sort_values("xi")
        ax.plot(s.xi, s.auroc, marker="o", ms=3, lw=1.0, alpha=0.20,
                color=colors.get(lab, "#999999"), label=pretty(lab))
    # aggregate lines
    xs = range(len(rungs))
    ax.plot(xs, mean_curve(PROBE_METHOD), color="#111111", lw=3.0, marker="o",
            label="mean probe (LLM)", zorder=5)
    ax.plot(xs, mean_curve("baseline_lr"), color="#D55E00", lw=3.0, ls="--", marker="s",
            label="mean raw-peak baseline", zorder=5)
    ax.plot(xs, mean_curve("random_lr"), color="#7a7a7a", lw=2.0, ls=":", marker="^",
            label="mean random-init", zorder=4)
    ax.axhline(0.5, ls=":", lw=1, color="#bbbbbb")

    ax.set_xticks(list(xs)); ax.set_xticklabels(rungs)
    ax.set_xlabel("Shift-ladder rung  (R0 = simulated test \u2192 measured SERS)")
    ax.set_ylabel("AUROC"); ax.set_ylim(0.0, 1.0)
    # two-part legend: aggregates first, concepts after
    h, l = ax.get_legend_handles_labels()
    agg = ["mean probe (LLM)", "mean raw-peak baseline", "mean random-init"]
    order = [l.index(a) for a in agg] + [i for i, lab in enumerate(l) if lab not in agg]
    ax.legend([h[i] for i in order], [l[i] for i in order],
              frameon=False, fontsize=7.5, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.50))
    ax.set_title("Probes trained on simulation vs a raw-peak baseline under the sim\u2192real shift")
    fig.tight_layout(); fig.savefig(ASSETS / "degradation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rungs

# ----------------------------------------------------------------------
# Figure 3 — in-domain vs most-shifted-rung AUROC
# ----------------------------------------------------------------------
def fig_transfer(colors, rungs):
    df = pd.read_csv(PROBE / "degradation.csv")
    df = df[df.method == PROBE_METHOD]
    real = rungs[-1]                       # most-shifted rung
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#777777")
    for lab in sorted(df.label.unique()):
        s = df[(df.label == lab) & (df.rung == real)]
        if not len(s):
            continue
        xi = float(s.qm9s_val_auroc.iloc[0]); yi = float(s.auroc.iloc[0])
        ax.scatter(xi, yi, s=70, color=colors.get(lab, "#444444"), zorder=3)
        ax.annotate(pretty(lab), (xi, yi), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.0, 1.1)
    ax.set_xlabel("In-domain AUROC (QM9S validation)")
    ax.set_ylabel(f"Measured-SERS AUROC (rung {real})")
    ax.set_title("Sim→real transfer per concept")
    ax.text(0.98, 0.02, "on the line = perfect transfer",
            transform=ax.transAxes, ha="right", fontsize=8, color="#777777")
    fig.tight_layout(); fig.savefig(ASSETS / "transfer_scatter.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    labels = fig_layer_sweep()
    colors = concept_colors(labels)
    rungs = fig_degradation(colors)
    fig_transfer(colors, rungs)
    print("wrote assets/layer_sweep.png, assets/degradation.png, assets/transfer_scatter.png")
    print(f"concepts: {[pretty(l) for l in labels]}")
    print(f"rungs: {rungs}  (most-shifted = {rungs[-1]})")
