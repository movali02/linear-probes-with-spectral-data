#!/usr/bin/env python3
"""
gapmode.py
==========
Orientation rung of the DFT -> SERS ladder, from ORCA's polarisability
derivatives in the .hess file.

In a nanoparticle-on-mirror / MLagg gap the local field is almost entirely along
the gap axis z, for both the incident and the scattered light. A molecule with
fixed orientation R then scatters mode k with intensity

    I_k(R)  ~  f(nu_k) * [ (R alpha'_k R^T)_zz ]^2

instead of the orientation-averaged activity 45 a^2 + 7 gamma^2. Averaged over
random orientations this gives a^2 + 4 gamma^2 / 45 (the `gapiso785` sticks);
a preferred orientation reweights the bands further. Modes with large
anisotropy gamma^2 are the ones that move most, which is why your anisotropy
ranking picks the molecules where this rung is testable.

The .hess parser and the tensor-component convention check are from your
stage-B parser (1.parse_out_files.py): the convention that reproduces ORCA's
own activities (constant ratio) and depolarisation ratios is chosen.

  # per-mode table + orientation ensemble for later fitting
  python dft/gapmode.py --hess tryptophan.hess --out gap/tryptophan
  # fit orientation to a measured spectrum, scored on HELD-OUT bands
  python dft/gapmode.py --hess tryptophan.hess --out gap/tryptophan \
      --target sers_trp_mean.txt --scale 0.967 --fwhm 19

The fit has 3 degrees of freedom, so a good in-sample match means little. The
report fits on one half of the window and scores the other half, both ways,
and compares with the isotropic average and with random orientations.
"""
from __future__ import annotations

import argparse
import re

import numpy as np

CANDIDATE_ORDERS = [
    ["xx", "yy", "zz", "xy", "xz", "yz"],
    ["xx", "xy", "yy", "xz", "yz", "zz"],
    ["xx", "yy", "zz", "xy", "yz", "xz"],
]


# ---------------------------------------------------------------- .hess parsing
def _read_counted_block(txt, name):
    m = re.search(rf"\${name}\s*\n\s*(\d+)\s*\n(.*?)(?:\n\s*\$|\Z)", txt, re.DOTALL)
    if not m:
        raise ValueError(f"No ${name} block found")
    count = int(m.group(1))
    floats = [float(x) for x in re.findall(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?", m.group(2))]
    return count, floats


def parse_hess(path):
    txt = open(path, errors="ignore").read()
    n_pd, pd = _read_counted_block(txt, "polarizability_derivatives")
    n_rs, rs = _read_counted_block(txt, "raman_spectrum")
    if len(pd) != n_pd * 6 or len(rs) != n_rs * 3 or n_pd != n_rs:
        raise ValueError(f"unexpected block sizes: pd={len(pd)}/{n_pd} rs={len(rs)}/{n_rs}")
    return np.array(pd).reshape(n_pd, 6), np.array(rs).reshape(n_rs, 3)


def tensor(row, order):
    d = dict(zip(order, row))
    return np.array([[d["xx"], d["xy"], d["xz"]],
                     [d["xy"], d["yy"], d["yz"]],
                     [d["xz"], d["yz"], d["zz"]]])


def invariants(A):
    a = np.trace(A) / 3.0
    xx, yy, zz = A[0, 0], A[1, 1], A[2, 2]
    xy, xz, yz = A[0, 1], A[0, 2], A[1, 2]
    g2 = 0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy ** 2 + xz ** 2 + yz ** 2)
    return a, g2


def pick_convention(pd, rs):
    freq, act, dep = rs[:, 0], rs[:, 1], rs[:, 2]
    real = freq > 1.0
    best = None
    for order in CANDIDATE_ORDERS:
        S, rho = [], []
        for row in pd[real]:
            a, g2 = invariants(tensor(row, order))
            S.append(45 * a * a + 7 * g2)
            den = 45 * a * a + 4 * g2
            rho.append(3 * g2 / den if den > 0 else np.nan)
        S, rho = np.array(S), np.array(rho)
        good = (S > 1e-8) & (act[real] > 1e-8)
        ratio = act[real][good] / S[good]
        cv = np.std(ratio) / np.mean(ratio) if ratio.size else np.inf
        mae = np.nanmean(np.abs(dep[real] - rho))
        if best is None or cv + mae < best[1] + best[2]:
            best = (order, cv, mae)
    return best


def load_modes(hess_path):
    pd, rs = parse_hess(hess_path)
    order, cv, mae = pick_convention(pd, rs)
    ok = cv < 0.02 and mae < 0.03
    real = rs[:, 0] > 1.0
    tensors = np.stack([tensor(r, order) for r in pd[real]])
    return rs[real, 0], rs[real, 1], tensors, {"order": "-".join(order), "activity_cv": cv,
                                               "depol_mae": mae, "verified": ok}


# ---------------------------------------------------------------- physics
def iso_zz(tensors):
    """Orientation average of (alpha'_ZZ)^2 = a^2 + 4 gamma^2 / 45 (analytic)."""
    out = []
    for A in tensors:
        a, g2 = invariants(A)
        out.append(a * a + 4.0 * g2 / 45.0)
    return np.array(out)


def random_rotations(n, seed=0):
    from scipy.spatial.transform import Rotation
    return Rotation.random(n, random_state=seed)


def zz_matrix(tensors, rotations):
    """[n_orient, n_modes] of (R A R^T)_zz^2. (R A R^T)_zz = r3 . A . r3, r3 = 3rd row of R."""
    r3 = rotations.as_matrix()[:, 2, :]                      # [n_orient, 3]
    v = np.einsum("oi,mij,oj->om", r3, tensors, r3)          # [n_orient, n_modes]
    return v ** 2


def freq_factor(nu, laser_nm=785.0, T=298.15):
    nu0 = 1e7 / laser_nm
    return (nu0 - nu) ** 4 / nu / (1.0 - np.exp(-1.4387769 * nu / T))


def render(freqs, heights, grid, fwhm):
    hw = fwhm / 2.0
    return (heights[None, :] * hw ** 2 / ((grid[:, None] - freqs[None, :]) ** 2 + hw ** 2)).sum(1)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def fit_orientation(freqs, zz, fac, target_x, target_y, fwhm, window=(500, 1750)):
    """Held-out score: fit on one half of the window, score on the other, both ways."""
    grid = np.arange(window[0], window[1] + 1.0)
    ty = np.interp(grid, target_x, target_y)
    mid = 0.5 * (window[0] + window[1])
    halves = [grid < mid, grid >= mid]
    sims = np.stack([render(freqs, fac * zz[o], grid, fwhm) for o in range(zz.shape[0])])
    iso = render(freqs, fac * zz.mean(0), grid, fwhm)
    report = []
    for fit_h, test_h in [(0, 1), (1, 0)]:
        fm, tm = halves[fit_h], halves[test_h]
        fit_scores = np.array([_cos(s[fm], ty[fm]) for s in sims])
        best = int(np.argmax(fit_scores))
        test_scores = np.array([_cos(s[tm], ty[tm]) for s in sims])
        report.append({
            "fit_half": "low" if fit_h == 0 else "high",
            "best_orientation": best,
            "fit_cos_best": float(fit_scores[best]),
            "heldout_cos_best": float(test_scores[best]),
            "heldout_cos_isotropic": _cos(iso[tm], ty[tm]),
            "heldout_cos_random_median": float(np.median(test_scores)),
            "heldout_percentile_of_best": float((test_scores < test_scores[best]).mean() * 100),
        })
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hess", required=True)
    ap.add_argument("--out", required=True, help="output prefix, e.g. gap/tryptophan")
    ap.add_argument("--n_orient", type=int, default=5000)
    ap.add_argument("--laser_nm", type=float, default=785.0)
    ap.add_argument("--target", help="two-column measured spectrum to fit (baseline-corrected)")
    ap.add_argument("--scale", type=float, default=1.0, help="frequency scale factor for the DFT modes")
    ap.add_argument("--fwhm", type=float, default=19.0)
    a = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    freqs, S, tensors, conv = load_modes(a.hess)
    print(f"convention {conv['order']}  activity CV={conv['activity_cv']:.2%}  "
          f"depol MAE={conv['depol_mae']:.4f}  {'VERIFIED' if conv['verified'] else 'NOT VERIFIED'}")
    if not conv["verified"]:
        print("tensor convention does not reproduce ORCA's activities: fix the parser before using this")
    R = random_rotations(a.n_orient)
    zz = zz_matrix(tensors, R)
    iso = iso_zz(tensors)
    rel_err = np.abs(zz.mean(0) - iso) / np.maximum(iso, 1e-12)
    print(f"self-check: sampled orientation average vs analytic a^2+4g^2/45, median rel. error "
          f"{np.median(rel_err):.2%} over {a.n_orient} orientations")
    a_s, g2 = zip(*(invariants(A) for A in tensors))
    a_s, g2 = np.array(a_s), np.array(g2)
    fac = freq_factor(freqs * a.scale, a.laser_nm)
    p05, p50, p95 = np.percentile(zz, [5, 50, 95], axis=0)
    np.savetxt(f"{a.out}_modes.csv",
               np.c_[freqs, freqs * a.scale, S, a_s ** 2, g2, iso, p05, p50, p95,
                     g2 / np.maximum(45 * a_s ** 2, 1e-12)],
               delimiter=",", fmt="%.6g", comments="",
               header="freq_unscaled,freq_scaled,orca_activity,a2,gamma2,zz2_isotropic,"
                      "zz2_p05,zz2_p50,zz2_p95,anisotropy_g2_over_45a2")
    np.savez_compressed(f"{a.out}_orient.npz", quat=R.as_quat(), zz2=zz.astype(np.float32),
                        freq=freqs, scale=a.scale)
    print(f"wrote {a.out}_modes.csv and {a.out}_orient.npz")
    if a.target:
        t = np.loadtxt(a.target, comments="#")
        t = t[np.argsort(t[:, 0])]
        for r in fit_orientation(freqs * a.scale, zz, fac, t[:, 0], t[:, 1], a.fwhm):
            print(f"fit on {r['fit_half']:4s} half: held-out cos best={r['heldout_cos_best']:.3f} "
                  f"isotropic={r['heldout_cos_isotropic']:.3f} random median={r['heldout_cos_random_median']:.3f} "
                  f"(best orientation at held-out percentile {r['heldout_percentile_of_best']:.0f})")


if __name__ == "__main__":
    main()
