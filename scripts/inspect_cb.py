#!/usr/bin/env python3
"""Why did some CB-only spectra get no 829 cm-1 reference band?

In the 30 Sep run, 52 CB-only files (13 conditions) were flagged
`max_in_window(ref_missing)`: no real peak within ref_cm +/- ref_tol. For every
CB-only file this prints what is actually there:

  top_cm / top_counts  position and height of the tallest band in the window
  h829_rel             tallest point within 829 +/- ref_tol, relative to the top band
  near_cm / near_off   nearest real peak to 829 within +/- 20 cm-1, and its offset
  r_good               correlation with the mean of the CB files that DO have the band
  verdict              ok | shifted (band is there, outside +/- ref_tol)
                       | weak (band is there, under the prominence threshold)
                       | absent (no CB[5] signature, or r_good < 0.5: check the file / acquisition)

  python scripts/inspect_cb.py --sers_dir data/raw_sers --out reports/cb_inspect
Writes <out>.csv and <out>.png (flagged conditions overlaid on the good-CB mean)."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit import preprocess as pp  # noqa: E402
from probe2circuit import substrate as sub  # noqa: E402
from probe2circuit.io import find_sers_files, load_config, load_sers_txt  # noqa: E402
from probe2circuit.pipeline import group_of, kind_from_id, normalise_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--sers_dir", default="data/raw_sers")
    ap.add_argument("--out", default="reports/cb_inspect")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rn = cfg["substrate"]["reference_norm"]
    ref, tol, min_rel = float(rn["ref_cm"]), float(rn.get("ref_tol", 4.0)), float(rn.get("ref_min_rel_prominence", 0.02))

    spec, grid = {}, None
    for p in find_sers_files(a.sers_dir):
        sid = normalise_id(p.stem)
        if kind_from_id(sid) != "substrate_only":
            continue
        x, y = load_sers_txt(p)
        grid, yc = pp.prepare_experimental(x, y, cfg)
        spec[sid] = pp.smooth(yc, cfg)
    if not spec:
        raise SystemExit(f"no CB_*.txt under {a.sers_dir}")
    has = {s: sub.reference_height(grid, ys, ref, tol, min_rel)[0] is not None for s, ys in spec.items()}
    good = np.mean([spec[s] / spec[s].max() for s in spec if has[s]], axis=0) if any(has.values()) else None

    rows = []
    for sid, ys in sorted(spec.items()):
        top = float(ys.max())
        idx, props = find_peaks(ys, prominence=0.005 * top)
        win = (grid >= ref - tol) & (grid <= ref + tol)
        near = [(abs(grid[i] - ref), i, pr) for i, pr in zip(idx, props["prominences"]) if abs(grid[i] - ref) <= 20]
        row = {"id": sid, "group": group_of(sid, cfg), "has_ref": has[sid], "top_cm": float(grid[np.argmax(ys)]),
               "top_counts": round(top, 1), "h829_rel": round(float(ys[win].max() / top), 3) if top > 0 else "",
               "near_cm": "", "near_off": "", "near_prom_rel": "",
               "r_good": round(float(np.corrcoef(ys, good)[0, 1]), 3) if good is not None else ""}
        if near:
            _, i, pr = min(near)
            row.update(near_cm=float(grid[i]), near_off=round(float(grid[i] - ref), 1), near_prom_rel=round(float(pr / top), 4))
        if row["r_good"] != "" and row["r_good"] < 0.5:
            row["verdict"] = "absent"            # does not look like the other CB spectra at all
        elif has[sid]:
            row["verdict"] = "ok"
        elif near and abs(row["near_off"]) > tol and row["near_prom_rel"] >= min_rel:
            row["verdict"] = "shifted"
        elif near and row["near_prom_rel"] < min_rel:
            row["verdict"] = "weak"
        else:
            row["verdict"] = "absent"
        rows.append(row)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} CB-only files: {dict(Counter(r['verdict'] for r in rows))}")
    bad = [r for r in rows if r["verdict"] != "ok"]
    print(f"\n{'group':16s} {'n':>3s}  verdicts                 top band (cm-1)   h829/top   r_good")
    for g in sorted({r["group"] for r in bad}):
        rs = [r for r in bad if r["group"] == g]
        v = dict(Counter(r["verdict"] for r in rs))
        print(f"{g:16s} {len(rs):3d}  {str(v):24s} {np.median([r['top_cm'] for r in rs]):8.0f}       "
              f"{np.median([r['h829_rel'] for r in rs]):6.2f}   {np.median([r['r_good'] for r in rs]):6.2f}")
    print("\nshifted -> raise substrate.reference_norm.ref_tol; weak -> the CB signal is low in that well;"
          "\nabsent  -> the file has no CB[5] signature: open it, it may not be a CB-only spectrum.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = sorted({r["group"] for r in bad})
        if groups and good is not None:
            n = len(groups)
            fig, axes = plt.subplots(-(-n // 3), 3, figsize=(15, 2.6 * -(-n // 3)), squeeze=False)
            for ax in axes.ravel()[n:]:
                ax.axis("off")
            for ax, g in zip(axes.ravel(), groups):
                ax.plot(grid, good, color="0.6", lw=1, label="mean of good CB files")
                for r in bad:
                    if r["group"] == g:
                        ax.plot(grid, spec[r["id"]] / spec[r["id"]].max(), lw=0.8)
                ax.axvspan(ref - tol, ref + tol, color="C3", alpha=0.15)
                ax.set_title(g, fontsize=9)
                ax.set_xlim(grid[0], grid[-1])
            axes[0][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out.with_suffix(".png"), dpi=130)
            print(f"-> {out.with_suffix('.csv')}, {out.with_suffix('.png')}")
    except ImportError:
        print(f"-> {out.with_suffix('.csv')} (matplotlib not installed: no figure)")


if __name__ == "__main__":
    main()
