"""Substrate (MLagg / CB[5]) handling for SERS spectra.

The CB-only spectra give the full substrate signature, not just the 829 cm-1
band: in CB_536_trp_1 the substrate also contributes clear peaks near 617, 676,
755 and 880 cm-1. The old pipeline removed only 829 +/- 10, so 755 and 880 were
serialised as if they were analyte peaks in every spectrum.

Default mode is `reference_norm` (see `reference_height`, `cb_bands`,
`discard_substrate_peaks`): scale every SERS spectrum so the CB[5] ~829 band is
1.0, detect peaks, then drop the peaks the substrate explains. Nothing is
subtracted, so no negative intensities.

Spectra with no CB-only file of their own (the amino-acid and indole
standards) use a GENERIC CB reference: the mean of all cell-assay CB-only
spectra on the 829 = 1.0 scale, built once by scripts/make_generic_cb.py (see
`build_generic_cb`, `generic_bands`).

Caveat: tyrosine's 830/850 Fermi doublet overlaps the CB[5] 829 band. Under
reference_norm, tyrosine signal at ~830 inflates the reference height (every
other peak in that spectrum comes out smaller) and the ~830 peak is always
discarded with the reference band. Tyr 850 is outside the CB regions and kept."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks


def fit_scale(y_sample: np.ndarray, y_ref: np.ndarray, grid: np.ndarray, regions) -> float:
    """Non-negative least-squares scale of the reference onto the sample, fitted
    only on the substrate band regions (so analyte peaks elsewhere don't pull it)."""
    m = np.zeros_like(grid, dtype=bool)
    for lo, hi in regions:
        m |= (grid >= lo) & (grid <= hi)
    A = y_ref[m][:, None]
    s, _ = nnls(A, y_sample[m])
    return float(s[0])


def _region_mask(grid, regions):
    m = np.zeros_like(grid, dtype=bool)
    for lo, hi in regions:
        m |= (grid >= lo) & (grid <= hi)
    return m


def subtract(y_sample, y_ref, grid, cfg) -> tuple[np.ndarray, float, float]:
    """Align the reference by a small axis shift (calibration drift between
    acquisitions), fit a non-negative scale on the substrate bands, subtract.
    Returns (residual, scale, shift_cm)."""
    sc = cfg["substrate"]
    regions = sc["fit_regions"]
    m = _region_mask(grid, regions)
    best = None
    max_shift = float(sc.get("max_shift_cm", 0.0))
    for shift in np.arange(-max_shift, max_shift + 1e-9, 0.25):
        ref = np.interp(grid, grid + shift, y_ref)
        s = fit_scale(y_sample, ref, grid, regions)
        err = float(np.sum((y_sample[m] - s * ref[m]) ** 2))
        if best is None or err < best[0]:
            best = (err, s, float(shift), ref)
    _, s, shift, ref = best
    return y_sample - s * ref, s, shift


def catalogue(ref_spectra: list[np.ndarray], grid: np.ndarray, cfg: dict) -> list[float]:
    """Substrate peak positions from the mean of baseline-corrected CB-only spectra."""
    mean = np.mean([r / max(r.max(), 1e-12) for r in ref_spectra], axis=0)
    mean = mean / mean.max()
    idx, _ = find_peaks(mean, prominence=cfg["substrate"]["catalogue_min_prominence"],
                        distance=max(1, int(4 / cfg["grid_step"])))
    return [float(grid[i]) for i in idx]


def mask_peaks(peaks: list[dict], positions: list[float], tol: float) -> tuple[list[dict], int]:
    kept = [p for p in peaks if all(abs(p["position"] - q) > tol for q in positions)]
    return kept, len(peaks) - len(kept)


# ------------------------------------------------------------ reference_norm
def reference_height(grid: np.ndarray, y_smooth: np.ndarray, ref_cm: float, ref_tol: float,
                     min_rel_prominence: float = 0.02) -> tuple[float | None, float | None]:
    """(height, position) of the reference band: the tallest local maximum of the
    smoothed, baseline-corrected spectrum within ref_cm +/- ref_tol. A window
    rather than one grid point absorbs the 826-830 cm-1 drift between sessions.

    It must be a real peak: a local maximum whose prominence is at least
    `min_rel_prominence` x the spectrum's maximum. Otherwise (None, None). Without
    this, a spectrum with no CB band would be divided by the tail of a
    neighbouring peak or by noise, and every intensity would blow up."""
    top = float(y_smooth.max())
    if top <= 0:
        return None, None
    idx, props = find_peaks(y_smooth, prominence=min_rel_prominence * top)
    inwin = [(y_smooth[i], i) for i in idx if ref_cm - ref_tol <= grid[i] <= ref_cm + ref_tol]
    if not inwin:
        return None, None
    h, i = max(inwin)
    return float(h), float(grid[i])


def cb_bands(y_ref_norm: np.ndarray, grid: np.ndarray, regions, min_prominence: float,
             step: float) -> list[tuple[float, float]]:
    """CB[5] bands [(position, intensity)] inside `regions`, read off a smoothed
    CB-only spectrum that is already on the 829 = 1.0 scale. The intensity is
    the substrate's expected contribution at that position in an analyte
    spectrum on the same scale."""
    idx, _ = find_peaks(y_ref_norm, prominence=min_prominence, distance=max(1, int(round(4 / step))))
    m = _region_mask(grid, regions)
    return [(float(grid[i]), float(y_ref_norm[i])) for i in idx if m[i]]


def discard_substrate_peaks(peaks: list[dict], bands: list[tuple[float, float]], tol: float,
                            min_excess: float, min_ratio: float) -> tuple[list[dict], list[dict]]:
    """Drop the peaks the CB[5] substrate explains. Returns (kept, dropped).

    A peak is dropped when both hold:
      (a) it sits within `tol` cm-1 of a CB band from the paired CB-only spectrum;
      (b) it is not clearly taller than that band, i.e. NOT
          (intensity - expected >= min_excess AND intensity >= min_ratio * expected).

    Position alone can't separate Trp/indole 758 from CB ~755 (or Trp 877 from
    CB ~880): they are 3-4 cm-1 apart and SERS lines are ~19 cm-1 wide, so the
    detector usually sees one merged peak. Both spectra are on the 829 = 1.0
    scale, so the CB-only band height is how tall that peak would be with no
    analyte. A peak well above it carries analyte signal and is kept, at its
    full observed intensity (nothing is subtracted)."""
    kept, dropped = [], []
    for p in peaks:
        near = [(abs(p["position"] - pos), pos, e) for pos, e in bands if abs(p["position"] - pos) <= tol]
        if near:
            _, pos, e = min(near)
            analyte_signal = (p["intensity"] - e >= min_excess) and (p["intensity"] >= min_ratio * e)
            if not analyte_signal:
                dropped.append({**p, "cb_band": pos, "cb_expected": e})
                continue
        kept.append(p)
    return kept, dropped


# ------------------------------------------------------------ generic CB reference
def build_generic_cb(refs_norm: dict[str, np.ndarray], min_corr: float = 0.90) -> tuple[dict, dict]:
    """Generic CB[5] reference from many CB-only spectra, each already smoothed
    and on the 829 = 1.0 scale.

    Files whose shape correlates below `min_corr` with the median spectrum are
    left out (a bad acquisition, or a CB file that actually contains analyte).
    Returns ({"mean", "p90", "n"}, report). `p90` is the 90th percentile across
    files at every wavenumber: the height a CB band reaches in a high-substrate
    spectrum, used as the conservative expected height for unpaired spectra."""
    if not refs_norm:
        raise ValueError("no CB-only spectra with a usable 829 band")
    ids = sorted(refs_norm)
    M = np.stack([refs_norm[i] for i in ids])
    med = np.median(M, axis=0)
    r = np.array([np.corrcoef(m, med)[0, 1] for m in M])
    # with fewer than 3 files the median is not a meaningful consensus: keep all
    keep = r >= min_corr if len(ids) >= 3 else np.ones(len(ids), bool)
    if len(ids) >= 3 and keep.sum() < max(3, 0.5 * len(ids)):
        raise ValueError(f"only {keep.sum()}/{len(ids)} CB spectra pass min_corr={min_corr}; "
                         "check the files or lower min_corr")
    K = M[keep]
    g = {"mean": K.mean(0), "p90": np.percentile(K, 90, axis=0), "n": int(keep.sum())}
    report = {"n_files": len(ids), "n_used": int(keep.sum()),
              "rejected": [[ids[i], round(float(r[i]), 3)] for i in np.where(~keep)[0]],
              "corr_with_median": {"min": round(float(r.min()), 3), "median": round(float(np.median(r)), 3)}}
    return g, report


def generic_bands(g: dict, grid, regions, min_prominence, step, height="p90") -> list[tuple[float, float]]:
    """Band positions from the generic MEAN spectrum; expected heights from
    `height` ("p90" = conservative, "mean")."""
    pos = cb_bands(g["mean"], grid, regions, min_prominence, step)
    curve = g[height]
    return [(p, float(np.interp(p, grid, curve))) for p, _ in pos]


def save_generic_cb(path, grid, g: dict, report: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.c_[grid, g["mean"], g["p90"]], fmt="%.2f %.6f %.6f",
               header="wavenumber_cm-1 mean_829norm p90_829norm\n" + json.dumps({"n": g["n"], **report}))


def load_generic_cb(path, grid) -> dict:
    """Read a generic CB file and put it on `grid` (they normally match exactly)."""
    path = Path(path)
    meta = {}
    with open(path) as f:
        for line in f:
            if line.startswith("# {"):
                meta = json.loads(line[2:])
                break
    arr = np.loadtxt(path, comments="#")
    x = arr[:, 0]
    g = {"mean": np.interp(grid, x, arr[:, 1]), "p90": np.interp(grid, x, arr[:, 2]),
         "n": int(meta.get("n", 0)), "path": str(path)}
    return g


# ------------------------------------------------------------ pairing
def pair_name(sample_id: str) -> str:
    """Cell_536_trp_1 -> CB_536_trp_1 (your naming convention)."""
    return sample_id.replace("Cell_", "CB_", 1) if sample_id.startswith("Cell_") else "CB_" + sample_id


def condition_key(sample_or_ref_id: str) -> str:
    """Cell_536_trp_3 / CB_536_trp_7 -> 536_trp : strain + medium, repeat index
    and Cell/CB prefix dropped."""
    import re
    s = re.sub(r"^(Cell|CB)_", "", sample_or_ref_id)
    return re.sub(r"_\d+$", "", s)


def find_reference(sample_id: str, refs: dict, pair_by: str):
    """Return (reference spectrum, reference label) or (None, None).
    pair_by='condition_mean': mean of all CB repeats for the same strain+medium
    (less noise; the default). 'same_index': CB file with the same repeat number."""
    if pair_by == "condition_mean":
        key = condition_key(sample_id)
        members = sorted(k for k in refs if condition_key(k) == key)
        if members:
            return np.mean([refs[k] for k in members], axis=0), f"mean({key}, n={len(members)})"
    rid = pair_name(sample_id)
    if rid in refs:
        return refs[rid], rid
    return None, None
