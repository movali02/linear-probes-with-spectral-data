"""Shared preprocessing: window, common grid, baseline, smoothing, broadening."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


def make_grid(cfg: dict) -> np.ndarray:
    lo, hi = cfg["window"]
    step = cfg["grid_step"]
    return np.arange(lo, hi + step / 2, step)


def to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation onto the shared grid. Raises if the spectrum does not
    cover the window (silently extrapolating would invent flat regions)."""
    if x[0] > grid[0] + 1e-6 or x[-1] < grid[-1] - 1e-6:
        raise ValueError(
            f"spectrum covers {x[0]:.1f}-{x[-1]:.1f} cm-1, window needs "
            f"{grid[0]:.1f}-{grid[-1]:.1f}")
    return np.interp(grid, x, y)


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers & Boelens 2005)."""
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2), dtype=float)
    penalty = lam * d.dot(d.T)
    w = np.ones(n)
    z = y
    for _ in range(niter):
        wm = sparse.spdiags(w, 0, n, n)
        z = spsolve(sparse.csc_matrix(wm + penalty), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def smooth(y: np.ndarray, cfg: dict) -> np.ndarray:
    s = cfg["smooth"]
    npts = int(round(s["window_cm"] / cfg["grid_step"]))
    npts = max(npts | 1, s["polyorder"] + 2 | 1)  # odd, > polyorder
    return savgol_filter(y, npts, s["polyorder"])


def gaussian_broaden(y: np.ndarray, fwhm_cm: float, step: float) -> np.ndarray:
    if fwhm_cm <= 0:
        return y
    sigma_pts = fwhm_cm / (2 * np.sqrt(2 * np.log(2))) / step
    return gaussian_filter1d(y, sigma_pts, mode="nearest")


def lorentzian_broaden(freqs: np.ndarray, ints: np.ndarray, grid: np.ndarray, fwhm: float) -> np.ndarray:
    hw = fwhm / 2.0
    return (ints[None, :] * hw**2 / ((grid[:, None] - freqs[None, :]) ** 2 + hw**2)).sum(1)


def robust_noise(y: np.ndarray) -> float:
    """Robust std of white noise from first differences (MAD / sqrt 2).
    ~0 for simulated spectra, so they fall back to the prominence floor."""
    d = np.diff(y)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def normalise_max(y: np.ndarray) -> np.ndarray:
    m = float(np.max(y))
    if m <= 0:
        raise ValueError("spectrum has no positive signal in the window")
    return y / m


def prepare_experimental(x, y, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Crop to window (with a margin so the baseline has room), baseline-correct,
    then resample to the grid. Returns (grid, baseline-corrected y), NOT normalised:
    normalisation happens after substrate handling."""
    lo, hi = cfg["window"]
    margin = 50.0
    m = (x >= lo - margin) & (x <= hi + margin)
    xw, yw = x[m], y[m]
    b = cfg["baseline"]
    yc = yw - als_baseline(yw, float(b["lam"]), float(b["p"]), int(b["niter"]))
    grid = make_grid(cfg)
    return grid, to_grid(xw, yc, grid)


def prepare_simulated(x, y, cfg) -> tuple[np.ndarray, np.ndarray]:
    """QM9S broadened spectrum: resample to grid, optional extra broadening.
    No baseline step (simulated spectra have none)."""
    grid = make_grid(cfg)
    fs = float(cfg.get("sim_freq_scale", 1.0))      # 1.0134 in variant v2mf: QM9S sits 1.3 % below experiment
    yg = to_grid(np.asarray(x, float) * fs, y, grid)
    return grid, gaussian_broaden(yg, cfg.get("sim_extra_fwhm", 0.0), cfg["grid_step"])


def prepare_dft_sticks(freqs, ints, cfg) -> tuple[np.ndarray, np.ndarray]:
    grid = make_grid(cfg)
    d = cfg["dft"]
    # dft.scale maps your DFT onto QM9S (calibration set); sim_freq_scale then moves both onto experiment
    y = lorentzian_broaden(freqs * d["scale"] * float(cfg.get("sim_freq_scale", 1.0)), ints, grid, d["fwhm"])
    return grid, gaussian_broaden(y, cfg.get("sim_extra_fwhm", 0.0), cfg["grid_step"])
