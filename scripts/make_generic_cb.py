#!/usr/bin/env python3
"""Build the generic CB[5] reference from the cell-assay CB-only spectra.

The amino-acid and indole standards have no CB-only file of their own. They use
this generic reference instead: every CB file goes through the same steps as a
paired reference (window, ALS baseline, smoothing, 829 band = 1.0), files whose
shape disagrees with the rest are left out, and the mean and 90th percentile
are saved.

  python scripts/make_generic_cb.py --sers_dir data/raw_sers \
      --manifest data/split_manifest.csv --out data/generic_cb.txt

--manifest is optional: with it, only CB files listed there are used (your
cell-assay set); without it, every CB_*.txt in --sers_dir is used.

Writes:
  <out>                 grid, mean, p90 (829 = 1.0 scale); read by build_strings.py
  <out stem>_report.json  files used / rejected, and a band table: for every CB
                          band in the window, its height across files (p10, p50,
                          p90, CV), inside or outside the discard regions
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit import preprocess as pp  # noqa: E402
from probe2circuit import substrate as sub  # noqa: E402
from probe2circuit.io import find_sers_files, load_config, load_sers_txt  # noqa: E402
from probe2circuit.pipeline import kind_from_id, normalise_id, normalised_cb_references  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402


def cb_files(sers_dir, manifest):
    files = {normalise_id(p.stem): p for p in find_sers_files(sers_dir)}   # subfolders too
    files = {k: v for k, v in files.items() if kind_from_id(k) == "substrate_only"}
    if manifest:
        listed = {normalise_id(Path(r["fname"]).stem) for r in csv.DictReader(open(manifest))}
        missing = sorted(k for k in listed if kind_from_id(k) == "substrate_only" and k not in files)
        if missing:
            print(f"[warn] {len(missing)} CB files in the manifest are not in {sers_dir}, e.g. {missing[:5]}")
        files = {k: v for k, v in files.items() if k in listed}
    return files


def band_table(grid, M, regions, prom, step):
    """Height of every CB band across files (829 = 1.0)."""
    mean = M.mean(0)
    idx, _ = find_peaks(mean, prominence=prom, distance=max(1, int(round(4 / step))))
    inside = sub._region_mask(grid, regions)
    rows = []
    for i in idx:
        lo, hi = max(0, i - 3), min(len(grid), i + 4)
        h = M[:, lo:hi].max(1)
        rows.append({"position": round(float(grid[i]), 1),
                     "p10": round(float(np.percentile(h, 10)), 3),
                     "p50": round(float(np.percentile(h, 50)), 3),
                     "p90": round(float(np.percentile(h, 90)), 3),
                     "cv": round(float(h.std() / max(h.mean(), 1e-9)), 2),
                     "in_discard_regions": bool(inside[i])})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--sers_dir", default="data/raw_sers")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None, help="default: substrate.generic_cb.path in the config")
    a = ap.parse_args()

    cfg = load_config(a.config)
    gc = cfg["substrate"].get("generic_cb") or {}
    out = Path(a.out or gc.get("path", "data/generic_cb.txt"))
    files = cb_files(a.sers_dir, a.manifest)
    if not files:
        raise SystemExit(f"no CB_*.txt files found in {a.sers_dir}")

    loaded, grid = {}, None
    for sid, p in sorted(files.items()):
        x, y = load_sers_txt(p)
        grid, loaded[sid] = pp.prepare_experimental(x, y, cfg)
    _, ref_at, refs_norm = normalised_cb_references(loaded, grid, cfg)
    no_ref = sorted(k for k in loaded if k not in refs_norm)
    g, report = sub.build_generic_cb(refs_norm, float(gc.get("min_corr", 0.90)))
    report["no_829_band"] = no_ref
    report["ref_pos_cm"] = {"median": float(np.median([ref_at[k][1] for k in refs_norm])),
                            "min": float(min(ref_at[k][1] for k in refs_norm)),
                            "max": float(max(ref_at[k][1] for k in refs_norm))}
    used = [k for k in sorted(refs_norm) if k not in {r[0] for r in report["rejected"]}]
    M = np.stack([refs_norm[k] for k in used])
    rn = cfg["substrate"]["reference_norm"]
    prom = float(cfg["substrate"].get("catalogue_min_prominence", 0.10))
    report["bands"] = band_table(grid, M, rn["cb_regions"], prom, cfg["grid_step"])
    # strain check: are the BW and 536 CB spectra the same substrate?
    by_strain = {}
    for k in used:
        parts = k.split("_")
        if len(parts) >= 3:
            by_strain.setdefault(parts[1], []).append(refs_norm[k])
    if len(by_strain) >= 2:
        means = {s: np.mean(v, axis=0) for s, v in by_strain.items()}
        ks = sorted(means)
        report["strain_mean_corr"] = {f"{ks[i]} vs {ks[j]}": round(float(np.corrcoef(means[ks[i]], means[ks[j]])[0, 1]), 4)
                                      for i in range(len(ks)) for j in range(i + 1, len(ks))}
        report["n_by_strain"] = {s: len(v) for s, v in by_strain.items()}

    sub.save_generic_cb(out, grid, g, {k: report[k] for k in ("n_files", "n_used")})
    rep_path = out.with_name(out.stem + "_report.json")
    rep_path.write_text(json.dumps(report, indent=1))

    print(f"generic CB from {report['n_used']}/{report['n_files']} CB files "
          f"({len(report['rejected'])} rejected by shape, {len(no_ref)} without an 829 band)")
    print(f"829 band position across files: median {report['ref_pos_cm']['median']:.0f}, "
          f"range {report['ref_pos_cm']['min']:.0f}-{report['ref_pos_cm']['max']:.0f} cm-1")
    if "strain_mean_corr" in report:
        print(f"strain means: {report['n_by_strain']}  correlation {report['strain_mean_corr']}")
    print("\nCB bands (829 = 1.0): position  p10   p50   p90   CV   discarded in standards?")
    for b in report["bands"]:
        print(f"   {b['position']:7.0f}  {b['p10']:5.2f} {b['p50']:5.2f} {b['p90']:5.2f} {b['cv']:4.2f}   "
              f"{'yes (in cb_regions)' if b['in_discard_regions'] else 'NO, outside cb_regions'}")
    print(f"\n-> {out}\n-> {rep_path}")


if __name__ == "__main__":
    main()
