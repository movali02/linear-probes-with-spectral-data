#!/usr/bin/env python3
"""Build the shift-ladder strings (roadmap 5.3) and the QM9S probe set.

  python scripts/build_ladder.py --variant matched

Needs (see scripts/run_ladder.sh, which does all of it in order):
  data/dft_sticks/<set>/{activity,int785,gapiso785}/   from dft/dft_to_sticks.py
  data/processed_matched/{qm9s,sers}_records.jsonl     from build_strings.py --variant matched
  reports/ladder/calibration.txt (optional)            printed output of dft/calibrate_vs_qm9s.py:
                                                       sets the DFT frequency scale and which
                                                       intensity convention matches QM9S
Writes to data/ladder/<variant>/:
  qm9s_records.jsonl     probe set: a deterministic QM9S subsample, topped up with extra
                         positives of rare groups (training = split train, R0 = split test)
  ladder_records.jsonl   R1-R6 (your DFT), R8 (SERS standards), R9 (SERS cells), and R1q
                         (the QM9S spectra of the calibration molecules)
and to reports/ladder/<variant>/: similarity.csv, similarity_summary.csv (how close
each simulated rung is to the SERS standards and cells; no model involved)."""
import argparse
import csv
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit import ladder as L  # noqa: E402
from probe2circuit.io import default_data_dir, load_config, read_jsonl, write_jsonl  # noqa: E402
from probe2circuit.labels import FG_KEYS  # noqa: E402
from probe2circuit.pipeline import in_subsample  # noqa: E402
from probe2circuit.strings import estimate_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="matched")
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--sticks_dir", default="data/dft_sticks")
    ap.add_argument("--molecules", default="dft/molecules.csv")
    ap.add_argument("--calibration", default="reports/ladder/calibration.txt")
    ap.add_argument("--dft_scale", type=float, default=None, help="overrides the calibration file / config")
    ap.add_argument("--base_variant", default=None, choices=["activity", "int785"],
                    help="stick convention for R1-R4 (default: the one that matches QM9S in the calibration)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--report_dir", default=None)
    ap.add_argument("--qm9s_frac", type=float, default=0.1, help="QM9S molecules in the probe set")
    ap.add_argument("--enrich", type=int, default=300,
                    help="top up every functional group to this many positives where QM9S has them")
    a = ap.parse_args()

    cfg = load_config(a.config, a.variant)
    data_dir = Path(a.data_dir or default_data_dir(a.variant))
    out = Path(a.out_dir or f"data/ladder/{a.variant}")
    rep = Path(a.report_dir or f"reports/ladder/{a.variant}")
    rep.mkdir(parents=True, exist_ok=True)

    cal_scale, cal_conv = L.read_calibration(a.calibration)
    scale = a.dft_scale or cal_scale or float(cfg["dft"]["scale"])
    base = a.base_variant or cal_conv or "activity"
    src = "--dft_scale" if a.dft_scale else ("calibration" if cal_scale else "config default (run the calibration!)")
    print(f"[ladder] variant {a.variant} (schema {cfg['schema']}); DFT frequency scale {scale:.3f} from {src}; "
          f"stick convention for R1-R4: {base}")
    cfg["dft"]["scale"] = scale
    fs = float(cfg.get("sim_freq_scale", 1.0))
    if fs != 1.0:
        print(f"[ladder] simulated frequencies x {fs} (QM9S and DFT); DFT total factor {scale * fs:.4f}")

    dft, notes = L.dft_rung_records(a.sticks_dir, a.molecules, cfg, base)
    for n in notes:
        print(f"[ladder] {n}")
    sers_p = data_dir / "sers_records.jsonl"
    if not sers_p.exists():
        raise SystemExit(f"{sers_p} missing: run build_strings.py --variant {a.variant} --sers_dir ... --save_spectra")
    sers = L.sers_rung_records(read_jsonl(sers_p))

    # ---- QM9S probe set
    qp = data_dir / "qm9s_records.jsonl"
    if not qp.exists():
        raise SystemExit(f"{qp} missing: run build_strings.py --variant {a.variant} --qm9s_csv ...")
    keep, extra = [], {k: [] for k in FG_KEYS}
    for line in open(qp):
        r = json.loads(line)
        if not r.get("labels") or not r.get("text"):
            continue
        if in_subsample(r["id"], a.qm9s_frac):
            keep.append(r)
        else:
            for k in FG_KEYS:
                if r["labels"].get(k):
                    extra[k].append(r)
    have = Counter(k for r in keep for k in FG_KEYS if r["labels"].get(k))
    added = {}
    for k in FG_KEYS:
        need = a.enrich - have[k]
        if need > 0 and extra[k]:
            pool = sorted(extra[k], key=lambda r: zlib.crc32(r["id"].encode()))[:need]
            got = {r["id"] for r in keep}
            new = [r for r in pool if r["id"] not in got]
            keep += new
            added[k.replace("fg_", "")] = len(new)
            for r in new:
                for kk in FG_KEYS:
                    if r["labels"].get(kk):
                        have[kk] += 1
    cal = {r["name"]: r["smiles"] for r in csv.DictReader(open(a.molecules)) if r["set"] == "calibration_gas"}
    r1q = L.calibration_qm9s_records(qp, cal)
    write_jsonl(keep, out / "qm9s_records.jsonl")
    ladder = dft + r1q + sers
    write_jsonl(ladder, out / "ladder_records.jsonl")

    n_te = sum(r.get("split") == "test" for r in keep)
    print(f"[ladder] QM9S probe set: {len(keep)} molecules ({len(keep) - n_te} train, {n_te} test = R0)"
          + (f"; topped up: {added}" if added else ""))
    c = Counter(r["rung"] for r in ladder)
    for rung in sorted(c):
        print(f"[ladder]   {rung:4s} {c[rung]:5d} strings   {L.RUNG_LABEL.get(rung, 'QM9S spectra of the calibration molecules')}")
    toks = [estimate_tokens(cfg["prompt_template"].format(text=r["text"])) for r in keep[:2000] + ladder]
    print(f"[ladder] prompt length (estimate, 1 token per digit): median {sorted(toks)[len(toks) // 2]}, max {max(toks)}")

    groups = {"R0 (QM9S)": keep[:3000]}
    for rung in sorted(c):
        groups[rung] = [r for r in ladder if r["rung"] == rung]
    st = L.string_stats(groups, cfg)
    if st:
        with open(rep / "string_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(st[0]))
            w.writeheader()
            w.writerows(st)
        print("\n[ladder] what the strings look like on each rung (no model):")
        print(f"  {'rung':10s} {'n':>5s} {'peaks':>6s} {'median int':>10s} {'int<0.2':>8s} {'median fwhm':>11s} "
              f"{'peaks<900':>9s} {'strongest<900':>13s}")
        for r in st:
            print(f"  {r['rung']:10s} {r['n_strings']:5d} {r['peaks_median']:6.0f} {r['intensity_median']:10.2f} "
                  f"{r['frac_intensity_below_0.2']:8.0%} {r['fwhm_median']:11.0f} {r['frac_peaks_below_900']:9.0%} "
                  f"{r['frac_strongest_below_900']:13.0%}")

    rows = L.similarity_table(ladder, cfg)
    if rows:
        summ = L.similarity_summary(rows)
        for name, rr in (("similarity.csv", rows), ("similarity_summary.csv", summ)):
            with open(rep / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0]))
                w.writeheader()
                w.writerows(rr)
        print("\n[ladder] spectral similarity of each simulated rung to SERS (cosine of the re-drawn strings):")
        print(f"  {'rung':5s} {'vs':3s} {'mean cos':>8s} {'own standard is the best match':>31s} {'in top 3':>9s}")
        for s in summ:
            print(f"  {s['rung']:5s} {s['target']:3s} {s['mean_cosine']:8.3f} {s['top1']:31.0%} {s['top3']:9.0%}")
    print(f"\n-> {out}/qm9s_records.jsonl, {out}/ladder_records.jsonl, {rep}/similarity*.csv, {rep}/string_stats.csv")


if __name__ == "__main__":
    main()
