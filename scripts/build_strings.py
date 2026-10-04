#!/usr/bin/env python3
"""Build harmonised string records for SERS, QM9S and/or DFT with one config.

  python scripts/build_strings.py --config configs/string_v1.yaml \
      --sers_dir data/raw_sers --out_dir data/processed
  python scripts/build_strings.py --qm9s_csv /path/raman_boraden.csv \
      --qm9s_smiles data/qm9s_number_smiles.csv --out_dir data/processed

  python scripts/build_strings.py --variant matched ...   # -> data/processed_matched
  add --save_spectra to also write the processed spectra (step-2 F3 features):
  sers_spectra.npz (all SERS) and qm9s_spectra.npz (a --spectra_frac subset).

Outputs <source>_records.jsonl. `text` is the model input; everything else is
metadata. SERS records get a group-level split; QM9S records get one too (by
molecule) so a QM9S probe can be evaluated on held-out molecules."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe2circuit.io import (default_data_dir, find_sers_files, load_config, load_number_smiles,  # noqa: E402
                              subset_of, write_jsonl)
from probe2circuit.preprocess import make_grid  # noqa: E402
from probe2circuit.labels import add_labels  # noqa: E402
from probe2circuit.pipeline import (assign_splits, build_dft_records, build_qm9s_records,  # noqa: E402
                                    build_qm9s_records_from_peaklists, build_sers_records,
                                    load_analytes, save_spectra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/string_v1.yaml")
    ap.add_argument("--analytes", default="data/analytes.csv")
    ap.add_argument("--sers_dir")
    ap.add_argument("--sers_glob", default="*.txt")
    ap.add_argument("--qm9s_csv")
    ap.add_argument("--qm9s_smiles")
    ap.add_argument("--qm9s_jsonl", help="make_views.py peak-list jsonl (e.g. qm9s_raman.jsonl)")
    ap.add_argument("--qm9s_limit", type=int, default=None)
    ap.add_argument("--dft_dir")
    ap.add_argument("--dft_smiles", help="JSON {stem: SMILES}")
    ap.add_argument("--variant", default="native", help="native | matched (see `variants` in the config)")
    ap.add_argument("--out_dir", default=None, help="default: data/processed (native), data/processed_<variant>")
    ap.add_argument("--save_spectra", action="store_true",
                    help="also save processed spectra for the step-2 F3 features")
    ap.add_argument("--spectra_frac", type=float, default=0.2,
                    help="fraction of QM9S molecules whose spectra are saved (deterministic subset)")
    a = ap.parse_args()

    cfg = load_config(a.config, a.variant)
    analytes = load_analytes(a.analytes)
    out = Path(a.out_dir or default_data_dir(a.variant))
    print(f"[variant] {a.variant}: schema {cfg['schema']} -> {out}")

    if a.sers_dir:
        paths = find_sers_files(a.sers_dir, a.sers_glob)       # includes AAs/, indoles/, Cell/ ...
        if not paths:
            raise SystemExit(f"no {a.sers_glob} files under {a.sers_dir}")
        spectra = {} if a.save_spectra else None
        recs, info = build_sers_records(paths, cfg, analytes, spectra_out=spectra)
        if spectra:
            save_spectra(out / "sers_spectra.npz", make_grid(cfg), spectra, cfg)
            print(f"[sers] {len(spectra)} processed spectra -> {out/'sers_spectra.npz'}")
        from probe2circuit.pipeline import normalise_id
        folder = {normalise_id(p.stem): subset_of(p, a.sers_dir) for p in paths}
        for r in recs:
            r["subset"] = folder.get(r["id"], ".")   # metadata only, never in the model input
        counts = {}
        for r in recs:
            counts[(r["subset"], r["kind"])] = counts.get((r["subset"], r["kind"]), 0) + 1
        print("[sers] files by folder: " + ", ".join(f"{f}/{k}={n}" for (f, k), n in sorted(counts.items())))
        miss = add_labels(recs)
        assign_splits([r for r in recs if r["kind"] != "substrate_only"], cfg)
        for r in recs:
            r.setdefault("split", "control")  # substrate-only spectra: never trained on
        write_jsonl(recs, out / "sers_records.jsonl")
        json.dump(info, open(out / "substrate_catalogue.json", "w"), indent=1)
        modes = {}
        for r in recs:
            modes[r["substrate"]["mode_used"]] = modes.get(r["substrate"]["mode_used"], 0) + 1
        print(f"[sers] {len(recs)} records ({miss} without SMILES), substrate modes {modes}, "
              f"catalogue {info['catalogue']} -> {out/'sers_records.jsonl'}")
        refs = {}
        for r in recs:
            ref = (r["substrate"].get("cb_ref") or "none")
            key = "own CB (condition mean)" if ref.startswith("mean(") else \
                  "generic CB" if ref.startswith("generic(") else ref
            refs.setdefault((r["kind"], key), 0)
            refs[(r["kind"], key)] += 1
        for (kind, key), n in sorted(refs.items()):
            print(f"[sers]   {kind:15s} {key:25s} {n}")
        ax = info.get("axis_shift_cm")
        if ax:
            print(f"[sers] axis aligned to the CB[5] band: {ax['n']} spectra shifted by {ax['min']:+.0f} to "
                  f"{ax['max']:+.0f} cm-1 (median {ax['median']:+.0f})")
        g = info.get("generic_cb") or {}
        if g.get("source"):
            print(f"[sers] generic CB: {g['source']}, {g['n']} CB files, expected heights = {g['height']}")
        if info.get("n_cells_on_generic"):
            print(f"[sers] WARNING: {info['n_cells_on_generic']} CELL spectra have no CB file of their own "
                  "and used the generic CB")
        unmatched = [r["id"] for r in recs if r["analyte"] is None]
        if unmatched:
            print(f"[sers] WARNING: no analyte matched for {unmatched[:10]}")

    if a.qm9s_csv or a.qm9s_jsonl:
        if a.qm9s_csv:
            n2s = load_number_smiles(a.qm9s_smiles)
            spectra = {} if a.save_spectra else None
            recs, stats = build_qm9s_records(a.qm9s_csv, n2s, cfg, a.qm9s_limit,
                                             spectra_out=spectra, spectra_frac=a.spectra_frac)
            if spectra:
                save_spectra(out / "qm9s_spectra.npz", make_grid(cfg), spectra, cfg)
                print(f"[qm9s] {len(spectra)} processed spectra (frac {a.spectra_frac}) -> {out/'qm9s_spectra.npz'}")
        else:
            recs, stats = build_qm9s_records_from_peaklists(a.qm9s_jsonl, cfg, a.qm9s_limit)
            print("[qm9s] WARNING: built from global-normalised peak lists; see effective_floor. "
                  "Rebuild from raman_boraden.csv for a like-for-like comparison.")
        add_labels(recs)
        for r in recs:  # every molecule is its own group; split by molecule
            r["analyte"] = r["smiles"]
        assign_splits(recs, {**cfg, "split": {**cfg["split"], "group_by": "id"}}, label_key="source")
        write_jsonl(recs, out / "qm9s_records.jsonl")
        print(f"[qm9s] {stats} -> {len(recs)} records -> {out/'qm9s_records.jsonl'}")

    if a.dft_dir:
        smap = json.load(open(a.dft_smiles)) if a.dft_smiles else None
        recs = build_dft_records(sorted(p for p in Path(a.dft_dir).glob("*") if p.is_file()), cfg, smap)
        add_labels(recs)
        write_jsonl(recs, out / "dft_records.jsonl")
        print(f"[dft] {len(recs)} records -> {out/'dft_records.jsonl'}")


if __name__ == "__main__":
    main()
