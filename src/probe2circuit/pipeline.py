"""Build records {id, source, kind, analyte, text, ...} from any source.

`text` is the model input. Every other field is metadata and must never be
concatenated into a prompt; the leakage checks enforce that."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from . import preprocess as pp
from . import substrate as sub
from .io import iter_peaklist_jsonl, iter_qm9s_csv, load_dft_sticks, load_sers_txt
from .peaks import detect_peaks
from .strings import to_string


def in_subsample(record_id: str, frac: float) -> bool:
    """Deterministic, seed-free subset (CRC32 of the id). The same ids are picked
    in every run and every process, so spectra saved by build_strings.py and the
    QM9S subsample in run_baselines.py always agree."""
    import zlib
    return frac >= 1.0 or zlib.crc32(record_id.encode()) / 2**32 < frac


def save_spectra(path, grid, spectra: dict, cfg: dict) -> None:
    """npz for the F3 features. SERS entries are (x829, xsub) tuples, QM9S plain arrays."""
    ids = sorted(spectra)
    first = spectra[ids[0]]
    arrays = {"ids": np.array(ids), "grid": np.asarray(grid, float),
              "schema": np.array(cfg["schema"]), "variant": np.array(cfg.get("variant", "native"))}
    if isinstance(first, tuple):
        arrays["x829"] = np.stack([spectra[i][0] for i in ids]).astype(np.float32)
        arrays["xsub"] = np.stack([spectra[i][1] for i in ids]).astype(np.float32)
    else:
        arrays["x"] = np.stack([spectra[i] for i in ids]).astype(np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ----------------------------------------------------------------- analytes
def load_analytes(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def analyte_from_id(sample_id: str, analytes: list[dict]) -> dict | None:
    """Match a file id to an analyte by full name (standards: 'L-tyrosine') or by
    the code token in cell ids ('Cell_BW_tyr_1' -> tyr)."""
    low = sample_id.lower()
    for a in analytes:
        if low == a["name"].lower() or low == a["name"].lower().replace(" ", "_"):
            return a
    contained = [a for a in analytes if a["name"].lower() in low]   # CB_L-tyrosine_1
    if contained:
        return max(contained, key=lambda a: len(a["name"]))
    tokens = set(re.split(r"[_\-\s]+", low))
    hits = [a for a in analytes if a["code"].lower() in tokens]
    return hits[0] if len(hits) == 1 else None


def normalise_id(stem: str) -> str:
    """Undo the extra prefixes seen in the old outputs:
    Cell_Cell_BW_pro_18 -> Cell_BW_pro_18, Cell_CB_536_trp_4 -> CB_536_trp_4,
    AAs_L-threonine -> L-threonine, indoles_oxindole -> oxindole."""
    stem = re.sub(r"^(?:AAs|indoles)_", "", stem)
    return re.sub(r"^(?:Cell_|AAs_)+(?=(?:Cell|CB)_|L-)", "", stem)


def kind_from_id(sample_id: str) -> str:
    if sample_id.startswith("CB_"):
        return "substrate_only"
    if sample_id.startswith("Cell_"):
        return "cell"
    return "standard"


def group_of(sample_id: str, cfg: dict) -> str:
    if cfg["split"]["group_by"] == "strip_replicate":
        return re.sub(r"_\d+$", "", sample_id)
    return sample_id


# ----------------------------------------------------------------- core
def _finish(grid, y, cfg, mask_positions=None, top=None):
    """Smooth, normalise, detect peaks.

    top=None -> divide by the maximum inside the window (QM9S, DFT, and the SERS
    subtraction/mask modes). A number -> divide by that instead (SERS
    reference_norm passes the CB[5] 829 band height, so 829 = 1.0 and other
    peaks may exceed 1.0)."""
    ys = pp.smooth(y, cfg)
    if top is None:
        top = float(ys.max())
    if top <= 0:
        return [], 0
    ys = ys / top
    noise = pp.robust_noise(y / top)
    peaks = detect_peaks(grid, ys, cfg, noise=noise)
    n_masked = 0
    if mask_positions:
        peaks, n_masked = sub.mask_peaks(peaks, mask_positions, cfg["substrate"]["tol_cm"])
        if peaks:  # re-normalise to the strongest remaining (analyte) peak
            top = max(p["intensity"] for p in peaks)
            for p in peaks:
                p["intensity"] /= top
    return peaks, n_masked


def _sers_record(sid, kind, a, peaks, info, cfg):
    return {
        "id": sid, "source": "sers", "kind": kind,
        "analyte": a["name"] if a else None,
        "smiles": a["smiles"] if a else None,
        "group": group_of(sid, cfg),
        "text": to_string(peaks, cfg), "n_peaks": len(peaks),
        "substrate": info, "schema": cfg["schema"],
    }


def normalised_cb_references(loaded: dict, grid, cfg: dict) -> tuple[dict, dict, dict]:
    """Smooth every spectrum and find its 829 reference band. Returns
    (smoothed, ref_at, refs_norm) where refs_norm holds the CB-only spectra on
    the 829 = 1.0 scale. Shared with scripts/make_generic_cb.py so the generic
    reference is built exactly the way paired references are."""
    rn = cfg["substrate"]["reference_norm"]
    ref_cm, ref_tol = float(rn["ref_cm"]), float(rn.get("ref_tol", 4.0))
    min_rel = float(rn.get("ref_min_rel_prominence", 0.02))
    smoothed = {sid: pp.smooth(yc, cfg) for sid, yc in loaded.items()}
    ref_at = {sid: sub.reference_height(grid, ys, ref_cm, ref_tol, min_rel) for sid, ys in smoothed.items()}
    refs_norm = {sid: smoothed[sid] / ref_at[sid][0] for sid in loaded
                 if kind_from_id(sid) == "substrate_only" and ref_at[sid][0]}
    return smoothed, ref_at, refs_norm


def align_axis(loaded: dict, grid, cfg: dict) -> tuple[dict, dict]:
    """Shift every spectrum along the wavenumber axis so that its own CB[5]
    reference band sits at ref_cm (substrate.reference_norm.align_axis).

    The band is found exactly as for the normalisation (tallest local maximum
    within ref_cm +/- ref_tol of the smoothed spectrum). Across your cultures it
    sits anywhere from 822 to 828 cm-1: an axis offset between wells / sessions
    of half a 10 cm-1 bin, which also turns a control-culture subtraction into
    derivative-shaped residue. Spectra with no usable band are left alone.
    Returns (aligned spectra, {id: shift in cm-1 or None}).

    Caveat: tyrosine's own ~830 band overlaps the reference band and can pull
    the detected position by a cm-1 or two."""
    rn = cfg["substrate"]["reference_norm"]
    ref_cm, ref_tol = float(rn["ref_cm"]), float(rn.get("ref_tol", 4.0))
    min_rel = float(rn.get("ref_min_rel_prominence", 0.02))
    out, shifts = {}, {}
    for sid, yc in loaded.items():
        _, pos = sub.reference_height(grid, pp.smooth(yc, cfg), ref_cm, ref_tol, min_rel)
        if pos is None:
            out[sid], shifts[sid] = yc, None
            continue
        d = ref_cm - pos
        out[sid] = yc if d == 0 else np.interp(grid, grid + d, yc)
        shifts[sid] = float(d)
    return out, shifts


def _resolve_generic(cfg, grid, refs_norm, generic):
    """Generic CB for spectra with no CB file of their own. In order: the one
    passed in, the file named in substrate.generic_cb.path, or the mean of the
    CB-only spectra in this run."""
    gc = cfg["substrate"].get("generic_cb") or {}
    if generic is not None:
        return generic, "given"
    path = gc.get("path")
    if path and Path(path).exists():
        return sub.load_generic_cb(path, grid), f"file:{Path(path).name}"
    if refs_norm:
        try:
            g, _ = sub.build_generic_cb(refs_norm, float(gc.get("min_corr", 0.90)))
            return g, "in-run"
        except ValueError:
            return None, None
    return None, None


def _build_sers_reference_norm(loaded: dict, grid, cfg: dict, analytes: list[dict], generic=None,
                               spectra_out: dict | None = None):
    """substrate.mode = reference_norm.

    1. Scale each spectrum so its CB[5] band (max within ref_cm +/- ref_tol) = 1.0.
    2. Detect peaks on that scale.
    3. Drop peaks the substrate explains: near a CB band in `cb_regions` of the
       paired CB-only spectrum (same strain+medium, also 829-scaled) and not
       clearly taller than it. See substrate.discard_substrate_peaks.
       Spectra with no CB-only file of their own (amino-acid and indole
       standards) use the GENERIC CB reference instead: band positions from the
       generic mean, expected heights from its p90 by default.
    No subtraction, so no negative intensities.

    spectra_out: optional dict, filled with {id: (x829, xsub)} for the step-2 F3
    features: x829 = smoothed spectrum on the 829 = 1.0 scale (exactly what peak
    picking sees), xsub = x829 minus the CB reference used for that spectrum
    (condition mean, or the generic MEAN for unpaired spectra)."""
    sc = cfg["substrate"]
    rn = sc["reference_norm"]
    gc = sc.get("generic_cb") or {}
    tol = float(rn.get("match_tol_cm", 5.0))
    min_excess = float(rn.get("keep_min_excess", 0.10))
    min_ratio = float(rn.get("keep_min_ratio", 1.5))
    pair_by = sc.get("pair_by", "condition_mean")
    prom = float(sc.get("catalogue_min_prominence", 0.10))
    band_args = (grid, rn["cb_regions"], prom, cfg["grid_step"])

    shifts = {}
    if rn.get("align_axis"):
        loaded, shifts = align_axis(loaded, grid, cfg)
    smoothed, ref_at, refs_norm = normalised_cb_references(loaded, grid, cfg)
    generic, generic_src = _resolve_generic(cfg, grid, refs_norm, generic)
    height = gc.get("height", "p90")
    g_bands = sub.generic_bands(generic, *band_args, height=height) if generic is not None else []
    g_label = f"generic({generic_src}, n={generic['n']}, {height})" if generic is not None else None

    records = []
    n_cells_generic = 0
    for sid, yc in loaded.items():
        kind = kind_from_id(sid)
        a = analyte_from_id(sid, analytes)
        h, ref_pos = ref_at[sid]
        info = {"mode_used": "reference_norm", "ref_height": None, "ref_pos_cm": None,
                "cb_ref": None, "cb_bands": [], "n_masked": 0, "discarded": []}
        if shifts:
            info["axis_shift_cm"] = shifts.get(sid)
        if h is None:
            # No positive CB[5] band: can't put this spectrum on the 829 scale, and
            # without that scale the CB bands can't be judged. Keep it visible.
            peaks, _ = _finish(grid, yc, cfg)
            info["mode_used"] = "max_in_window(ref_missing)"
            if spectra_out is not None:
                top = float(smoothed[sid].max())
                x = smoothed[sid] / top if top > 0 else smoothed[sid]
                spectra_out[sid] = (x, x)
            records.append(_sers_record(sid, kind, a, peaks, info, cfg))
            continue

        peaks, _ = _finish(grid, yc, cfg, top=h)
        info.update(ref_height=round(h, 6), ref_pos_cm=ref_pos)

        if kind == "substrate_only" and pair_by != "condition_mean" and sid in refs_norm:
            ref, lab = refs_norm[sid], sid
        else:
            ref, lab = sub.find_reference(sid, refs_norm, pair_by)
        if spectra_out is not None:
            ref_curve = ref if ref is not None else (generic["mean"] if generic is not None else 0.0)
            spectra_out[sid] = (smoothed[sid] / h, smoothed[sid] / h - ref_curve)
        if ref is not None:
            bands, info["cb_ref"] = sub.cb_bands(ref, *band_args), lab
        elif g_bands:
            bands, info["cb_ref"] = list(g_bands), g_label
            if kind == "cell":
                n_cells_generic += 1
                info["cb_ref_warning"] = "cell spectrum without its own CB file"
        else:
            bands, info["cb_ref"] = [], "none"
        # The reference band is 1.0 by construction and carries no information:
        # make sure it is always in the discard list, even if find_peaks missed it.
        if not any(abs(pos - ref_pos) <= tol for pos, _ in bands):
            bands.append((ref_pos, 1.0))

        peaks, dropped = sub.discard_substrate_peaks(peaks, bands, tol, min_excess, min_ratio)
        if rn.get("rescale_to_max_after_discard") and peaks:
            top = max(p["intensity"] for p in peaks)
            for p in peaks:
                p["intensity"] /= top
            info["mode_used"] = "reference_norm+rescaled"
        info["cb_bands"] = [[round(p, 1), round(e, 3)] for p, e in sorted(bands)]
        info["n_masked"] = len(dropped)
        info["discarded"] = [[round(d["position"], 1), round(d["intensity"], 3), round(d["cb_expected"], 3)]
                             for d in dropped]
        records.append(_sers_record(sid, kind, a, peaks, info, cfg))
    return records, {"mode": "reference_norm", "catalogue": [round(p, 1) for p, _ in g_bands],
                     "n_refs": len(refs_norm),
                     "generic_cb": {"source": generic_src, "n": generic["n"] if generic is not None else 0,
                                    "height": height,
                                    "bands": [[round(p, 1), round(e, 3)] for p, e in g_bands]},
                     "n_cells_on_generic": n_cells_generic,
                     "axis_shift_cm": ({"n": len(sv), "median": float(np.median(sv)), "min": float(min(sv)),
                                        "max": float(max(sv))}
                                       if (sv := [v for v in shifts.values() if v is not None]) else None)}


def build_sers_records(paths: list[Path], cfg: dict, analytes: list[dict], generic_cb=None,
                       spectra_out: dict | None = None) -> list[dict]:
    """SERS spectra, with substrate handled per cfg['substrate']['mode'].
    CB_* files are treated as substrate-only references (and also emitted as
    records, kind='substrate_only', for the substrate-only control).
    `generic_cb`: optional dict from substrate.load_generic_cb, for spectra with
    no CB file of their own (otherwise substrate.generic_cb.path, then in-run)."""
    loaded = {}
    grid = None
    for p in paths:
        x, y = load_sers_txt(p)
        grid, yc = pp.prepare_experimental(x, y, cfg)
        sid = normalise_id(Path(p).stem)
        if sid in loaded:
            raise ValueError(f"two files map to the same id {sid!r} (check {p})")
        loaded[sid] = yc
    mode = cfg["substrate"]["mode"]
    if mode == "reference_norm":
        return _build_sers_reference_norm(loaded, grid, cfg, analytes, generic_cb, spectra_out)

    refs = {k: v for k, v in loaded.items() if kind_from_id(k) == "substrate_only"}
    mean_ref = np.mean(list(refs.values()), axis=0) if refs else None
    cat = sub.catalogue(list(refs.values()), grid, cfg) if refs else []

    records = []
    for sid, yc in loaded.items():
        kind = kind_from_id(sid)
        a = analyte_from_id(sid, analytes)
        info = {"mode_used": "none", "scale": None, "ref": None, "n_masked": 0}
        y, mask_pos = yc, None
        if kind != "substrate_only" and mode != "none":
            ref, ref_label = (sub.find_reference(sid, refs, cfg["substrate"].get("pair_by", "condition_mean"))
                              if mode == "paired" else (None, None))
            if ref is not None:
                y, s, sh = sub.subtract(yc, ref, grid, cfg)
                info.update(mode_used="paired", scale=round(s, 4), shift_cm=sh, ref=ref_label)
            elif mode == "mean_ref" and mean_ref is not None:
                y, s, sh = sub.subtract(yc, mean_ref, grid, cfg)
                info.update(mode_used="mean_ref", scale=round(s, 4), shift_cm=sh, ref="mean")
            elif cat:
                mask_pos = cat
                info["mode_used"] = "mask"
            if cfg["substrate"].get("mask_after_subtract") and info["mode_used"] != "mask" and cat:
                mask_pos = cat
        peaks, n_masked = _finish(grid, y, cfg, mask_pos)
        info["n_masked"] = n_masked
        records.append(_sers_record(sid, kind, a, peaks, info, cfg))
    return records, {"catalogue": cat, "n_refs": len(refs)}


def build_qm9s_records(csv_path, number_smiles: dict[int, str], cfg: dict, limit=None,
                       spectra_out: dict | None = None, spectra_frac: float = 0.2) -> tuple[list[dict], dict]:
    """spectra_out: optional dict, filled with {id: window-normalised smoothed
    spectrum} for the molecules selected by `in_subsample(id, spectra_frac)` (the
    same deterministic subset run_baselines.py uses for T3/T4)."""
    recs, stats = [], {"rows": 0, "no_smiles": 0, "empty": 0}
    for num, x, y in iter_qm9s_csv(csv_path):
        stats["rows"] += 1
        if limit and stats["rows"] > limit:
            break
        smi = number_smiles.get(num)
        if smi is None:
            stats["no_smiles"] += 1
            continue
        grid, yg = pp.prepare_simulated(x, y, cfg)
        if yg.max() <= 0:
            stats["empty"] += 1
            continue
        peaks, _ = _finish(grid, yg, cfg)
        if not peaks:
            stats["empty"] += 1
            continue
        if spectra_out is not None and in_subsample(f"qm9s_{num}", spectra_frac):
            ys = pp.smooth(yg, cfg)
            spectra_out[f"qm9s_{num}"] = ys / ys.max()
        recs.append({"id": f"qm9s_{num}", "source": "qm9s", "kind": "sim",
                     "analyte": None, "smiles": smi, "group": f"qm9s_{num}",
                     "text": to_string(peaks, cfg), "n_peaks": len(peaks),
                     "schema": cfg["schema"]})
    return recs, stats


def lorentz_width_to_fwhm(width, rel_height):
    """scipy peak_widths at rel_height r measures the full width at a fraction
    (1 - r) of the peak height. For a Lorentzian that is FWHM * sqrt(1/(1-r) - 1)."""
    return width / np.sqrt(1.0 / (1.0 - rel_height) - 1.0)


def build_qm9s_records_from_peaklists(path, cfg: dict, limit=None) -> tuple[list[dict], dict]:
    """QM9S from already-extracted peak lists (make_views.py output).

    Those lists were normalised over the FULL spectrum and cut at 0.1 of the
    global max, so inside 500-1750 every molecule is missing any peak weaker than
    `source_floor / window_max` of its strongest fingerprint peak. We record that
    effective floor per molecule so it can be filtered on or reported. Rebuilding
    from raman_boraden.csv (build_qm9s_records) avoids the problem entirely."""
    q = cfg["qm9s_peaklist"]
    lo, hi = cfg["window"]
    p = cfg["peaks"]
    recs = []
    stats = {"rows": 0, "no_peak_in_window": 0}
    floors = []
    for i, smi, x, y, w in iter_peaklist_jsonl(path):
        stats["rows"] += 1
        if limit and stats["rows"] > limit:
            break
        m = (x >= lo) & (x <= hi)
        if not m.any():
            stats["no_peak_in_window"] += 1
            continue
        wmax = float(y[m].max())
        eff_floor = q["source_floor"] / wmax          # in window-normalised units
        floors.append(eff_floor)
        fwhm = lorentz_width_to_fwhm(w[m], q["source_rel_height"])
        peaks = [{"position": float(a), "intensity": float(b) / wmax, "width": float(c)}
                 for a, b, c in zip(x[m], y[m], fwhm)]
        peaks = [pk for pk in peaks if pk["intensity"] >= max(p["min_height"], q.get("min_height", 0.0))]
        if len(peaks) > p["max_peaks"]:
            peaks = sorted(peaks, key=lambda d: -d["intensity"])[: p["max_peaks"]]
        peaks.sort(key=lambda d: d["position"])
        recs.append({"id": f"qm9s_{i+1}", "source": "qm9s", "kind": "sim", "analyte": None,
                     "smiles": smi, "group": f"qm9s_{i+1}", "text": to_string(peaks, cfg),
                     "n_peaks": len(peaks), "window_max_global": round(wmax, 3),
                     "effective_floor": round(min(eff_floor, 1.0), 3), "schema": cfg["schema"],
                     "qm9s_input": "peaklist_jsonl"})
    f = np.asarray(floors)
    if f.size:
        stats["effective_floor_pct_5_25_50_75_95"] = [round(float(v), 2) for v in np.percentile(np.minimum(f, 1), [5, 25, 50, 75, 95])]
    stats["n_peaks_median"] = float(np.median([r["n_peaks"] for r in recs])) if recs else 0
    return recs, stats


def build_dft_records(paths, cfg, smiles_map: dict[str, str] | None = None) -> list[dict]:
    recs = []
    for p in paths:
        sid = Path(p).stem
        f, i = load_dft_sticks(p)
        grid, y = pp.prepare_dft_sticks(f, i, cfg)
        if y.max() <= 0:
            continue
        peaks, _ = _finish(grid, y, cfg)
        recs.append({"id": f"dft_{sid}", "source": "dft", "kind": "sim",
                     "analyte": sid, "smiles": (smiles_map or {}).get(sid),
                     "group": f"dft_{sid}", "text": to_string(peaks, cfg),
                     "n_peaks": len(peaks), "schema": cfg["schema"]})
    return recs


def assign_splits(records: list[dict], cfg: dict, label_key: str = "analyte") -> None:
    """Group-level split, stratified by label, >=1 group per side where possible."""
    rng = np.random.default_rng(cfg["split"]["seed"])
    by_label: dict = {}
    for r in records:
        by_label.setdefault(r.get(label_key), set()).add(r["group"])
    split_of = {}
    for lab, groups in by_label.items():
        g = sorted(groups)
        rng.shuffle(g)
        n_test = int(round(cfg["split"]["test_fraction"] * len(g)))
        n_test = min(max(n_test, 1), len(g) - 1) if len(g) > 1 else 0
        for k, gg in enumerate(g):
            split_of[gg] = "test" if k < n_test else "train"
    for r in records:
        r["split"] = split_of[r["group"]]
