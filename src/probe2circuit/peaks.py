"""One peak detector for every source."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, peak_widths


def detect_peaks(grid: np.ndarray, y_norm: np.ndarray, cfg: dict, noise: float = 0.0) -> list[dict]:
    """Peaks on a max-normalised, smoothed spectrum on the shared grid.

    Returns [{"position", "intensity", "width"}] sorted by position, keeping at
    most `max_peaks` of the most intense."""
    p = cfg["peaks"]
    step = cfg["grid_step"]
    prom = max(p["min_prominence"], p.get("noise_k", 0.0) * noise)
    idx, props = find_peaks(
        y_norm,
        height=p["min_height"],
        prominence=prom,
        distance=max(1, int(round(p["min_distance_cm"] / step))),
        width=p.get("min_width_cm", 0.0) / step,
        rel_height=p["width_rel_height"],
    )
    if len(idx) == 0:
        return []
    w = peak_widths(y_norm, idx, rel_height=p["width_rel_height"])
    ar = np.arange(len(grid))
    left = np.interp(w[2], ar, grid)
    right = np.interp(w[3], ar, grid)
    peaks = [
        {"position": float(grid[i]), "intensity": float(y_norm[i]), "width": float(r - l)}
        for i, l, r in zip(idx, left, right)
    ]
    if len(peaks) > p["max_peaks"]:
        peaks = sorted(peaks, key=lambda d: -d["intensity"])[: p["max_peaks"]]
    return sorted(peaks, key=lambda d: d["position"])
