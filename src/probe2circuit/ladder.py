"""The shift ladder from simulated to real spectra (roadmap 5.3).

Every rung is the same string format from the same pipeline; each changes one
thing. Rungs R2-R9 are the SAME 20 amino acids, so a probe's score can be
followed analyte by analyte from simulation to SERS.

  R0  QM9S, held-out molecules            where the probe is trained (in-domain)
  R1  calibration set: your DFT, gas      the same 9 molecules as in QM9S -> method/basis only
  R2  amino acids, neutral, gas           the molecules leave the QM9 domain
  R3  amino acids, neutral, water         + implicit solvent
  R4  amino acids, pH 7 species, water    + protonation state (zwitterion)
  R5  R4 as Raman intensity at 785 nm     + frequency factor (activity -> intensity)
  R6  R4 with the gap field along z       + plasmonic gap, molecule randomly oriented
  R8  SERS, pure standards                + substrate, surface binding, anharmonicity
  R9  SERS, cell supernatant              + biological matrix
(R7, a fitted orientation in the gap, needs a per-molecule fit against the
standard and is left for the mechanism step.)

Labels always come from the NEUTRAL parent molecule, as for the SERS spectra:
the question is whether the probe still finds "has an amine" when the amine is
NH3+ on a gold surface."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .baselines import render_lorentzian
from .pipeline import build_dft_records
from .preprocess import make_grid
from .strings import parse_string

RUNGS = [  # (rung, label, DFT set, stick variant or None = base variant)
    ("R1", "calibration molecules, your DFT (gas)", "calibration_gas", None),
    ("R2", "amino acids: neutral, gas", "neutral_gas", None),
    ("R3", "+ water (implicit)", "neutral_water", None),
    ("R4", "+ pH 7 protonation", "zwitterion_water", None),
    ("R5", "+ 785 nm intensities", "zwitterion_water", "int785"),
    ("R6", "+ gap field along z", "zwitterion_water", "gapiso785"),
]
RUNG_LABEL = {"R0": "QM9S (held out)", **{r: lab for r, lab, _, _ in RUNGS},
              "R8": "SERS standards", "R9": "SERS cells"}
ORDER = ["R0", "R2", "R3", "R4", "R5", "R6", "R8", "R9"]


def read_calibration(path) -> tuple[float | None, str | None]:
    """(frequency scale, intensity convention) from the printed output of
    dft/calibrate_vs_qm9s.py; (None, None) if the file is missing or unreadable."""
    p = Path(path)
    if not p.exists():
        return None, None
    txt = p.read_text()
    m = re.search(r"frequency scale[^:]*:\s*([0-9.]+)", txt)
    rho = dict(re.findall(r"intensity convention (\w+)\s*: Spearman rho with QM9S = (-?[0-9.]+|nan)", txt))
    conv = None
    try:
        vals = {k: float(v) for k, v in rho.items() if v != "nan"}
        if vals:        # activity unless int785 is clearly better (a 0.05 margin guards against noise)
            conv = "int785" if vals.get("int785", -9) > vals.get("activity", -9) + 0.05 else "activity"
    except ValueError:
        pass
    return (float(m.group(1)) if m else None), conv


def dft_rung_records(sticks_dir, molecules_csv, cfg, base_variant="activity") -> tuple[list[dict], list[str]]:
    """Records for R1-R6 from dft_to_sticks.py output (<sticks_dir>/<set>/<variant>/<name>.txt)."""
    mols = list(csv.DictReader(open(molecules_csv)))
    analyte = {(r["set"], r["name"]): r["analyte"] for r in mols}
    out, notes = [], []
    for rung, label, dset, variant in RUNGS:
        variant = variant or base_variant
        if rung == "R5" and base_variant == "int785":
            notes.append("R5 skipped: the base variant already is int785 (QM9S matched that convention)")
            continue
        d = Path(sticks_dir) / dset / variant
        files = sorted(d.glob("*.txt")) if d.exists() else []
        if not files:
            notes.append(f"{rung} missing: no sticks in {d} (run dft/dft_to_sticks.py for {dset})")
            continue
        smap_p = Path(sticks_dir) / dset / "smiles_map.json"
        smap = json.loads(smap_p.read_text()) if smap_p.exists() else {}
        recs = build_dft_records(files, cfg, smap)
        for r in recs:
            name = r["id"][len("dft_"):]
            r.update(id=f"{rung}:{name}", rung=rung, source="dft", dft_set=dset, stick_variant=variant,
                     analyte=analyte.get((dset, name)) or name, group=f"{rung}:{name}", split="ladder")
        n_empty = sum(1 for r in recs if not r["text"])
        out += [r for r in recs if r["text"]]
        notes.append(f"{rung}: {len(recs) - n_empty} molecules from {dset}/{variant}"
                     + (f" ({n_empty} with no peak in the window dropped)" if n_empty else ""))
    from .labels import add_labels          # RDKit only where labels are computed
    add_labels(out)
    return out, notes


def sers_rung_records(sers_records: list[dict]) -> list[dict]:
    """R8 (standards) and R9 (cells) from the SERS records of the same variant. CB-only and
    no-analyte control spectra are left out."""
    out = []
    for r in sers_records:
        if r["kind"] not in ("standard", "cell") or not r.get("analyte") or not r.get("smiles") or not r.get("text"):
            continue
        rung = "R8" if r["kind"] == "standard" else "R9"
        out.append({**r, "id": f"{rung}:{r['id']}", "sers_id": r["id"], "rung": rung, "source": "sers"})
    return out


def calibration_qm9s_records(qm9s_records_path, cal_smiles: dict[str, str]) -> list[dict]:
    """The QM9S spectra of the calibration molecules (rung 'R1q'), to compare probe scores with R1."""
    from .labels import canonical
    want = {canonical(s): n for n, s in cal_smiles.items()}
    out, seen = [], set()
    for line in open(qm9s_records_path):
        r = json.loads(line)
        if not r.get("smiles") or not r.get("text"):
            continue
        c = canonical(r["smiles"])
        if c in want and c not in seen:
            seen.add(c)
            out.append({**r, "id": f"R1q:{want[c]}", "qm9s_id": r["id"], "rung": "R1q", "analyte": want[c],
                        "group": f"R1q:{want[c]}", "split": "ladder"})
        if len(seen) == len(want):
            break
    return out


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def similarity_table(records: list[dict], cfg: dict) -> list[dict]:
    """Direct spectral similarity of every simulated rung to the SERS standards
    (and cells): cosine between the strings re-drawn as spectra, per analyte, and
    whether an analyte's simulated spectrum is closer to its OWN standard than to
    the other analytes' standards (rank 1 = yes)."""
    grid = make_grid(cfg)
    mean = {}
    for rung in {r["rung"] for r in records}:
        by = {}
        for r in records:
            if r["rung"] == rung:
                by.setdefault(r["analyte"], []).append(render_lorentzian(parse_string(r["text"], cfg), grid))
        mean[rung] = {a: _unit(np.mean(v, axis=0)) for a, v in by.items()}
    rows = []
    for rung in sorted(mean):
        if rung in ("R8", "R9", "R1", "R1q"):
            continue
        for target in ("R8", "R9"):
            if target not in mean:
                continue
            common = sorted(set(mean[rung]) & set(mean[target]))
            for a in common:
                sims = {b: float(mean[rung][a] @ mean[target][b]) for b in common}
                rows.append({"rung": rung, "target": target, "analyte": a, "cosine": round(sims[a], 4),
                             "rank": 1 + sum(v > sims[a] for b, v in sims.items() if b != a), "n": len(common)})
    return rows


def similarity_summary(rows) -> list[dict]:
    out = []
    for rung, target in dict.fromkeys((r["rung"], r["target"]) for r in rows):
        rr = [r for r in rows if r["rung"] == rung and r["target"] == target]
        out.append({"rung": rung, "target": target, "n_analytes": len(rr),
                    "mean_cosine": round(float(np.mean([r["cosine"] for r in rr])), 4),
                    "median_rank": float(np.median([r["rank"] for r in rr])),
                    "top1": round(float(np.mean([r["rank"] == 1 for r in rr])), 3),
                    "top3": round(float(np.mean([r["rank"] <= 3 for r in rr])), 3)})
    return out


def string_stats(groups: dict[str, list[dict]], cfg: dict) -> list[dict]:
    """What the strings of each rung look like, without any model: peaks per
    string and the distribution of the three numbers. A probe (or classifier)
    can only stay stable along the ladder where these stay comparable.
    groups: {rung name: records}."""
    rows = []
    for name, recs in groups.items():
        pk = [parse_string(r["text"], cfg) for r in recs if r.get("text")]
        if not pk:
            continue
        flat = [p for q in pk for p in q]
        inten = np.array([p["intensity"] for p in flat])
        width = np.array([p["width"] for p in flat])
        pos = np.array([p["position"] for p in flat])
        strongest = np.array([max(q, key=lambda p: p["intensity"])["position"] for q in pk])
        rows.append({"rung": name, "n_strings": len(pk),
                     "peaks_median": float(np.median([len(q) for q in pk])),
                     "peaks_p10": float(np.percentile([len(q) for q in pk], 10)),
                     "peaks_p90": float(np.percentile([len(q) for q in pk], 90)),
                     "intensity_median": round(float(np.median(inten)), 3),
                     "frac_intensity_below_0.2": round(float((inten < 0.2).mean()), 3),
                     "fwhm_median": float(np.median(width)),
                     "fwhm_p10": float(np.percentile(width, 10)), "fwhm_p90": float(np.percentile(width, 90)),
                     "frac_peaks_below_900": round(float((pos < 900).mean()), 3),
                     "strongest_peak_median_cm": float(np.median(strongest)),
                     "frac_strongest_below_900": round(float((strongest < 900).mean()), 3)})
    return rows
