"""Loaders for every spectrum source. All return ascending (x, y) float arrays."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml


def load_config(path: str | Path, variant: str | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return apply_variant(cfg, variant) if variant else cfg


def apply_variant(cfg: dict, name: str) -> dict:
    """Return a copy of cfg with the overrides in cfg['variants'][name] applied
    (dotted keys, e.g. 'substrate.reference_norm.rescale_to_max_after_discard').
    'native' changes nothing. Other variants get schema '<schema>-<name>' so their
    strings can never be mistaken for native ones."""
    import copy
    variants = cfg.get("variants") or {"native": {}}
    if name not in variants:
        raise ValueError(f"unknown variant {name!r}; config defines {sorted(variants)}")
    out = copy.deepcopy(cfg)
    for key, val in (variants[name] or {}).items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise KeyError(f"variant {name!r}: {key!r} is not a config path")
            node = node[p]
        node[parts[-1]] = val
    out["variant"] = name
    if name != "native":
        out["schema"] = f"{cfg['schema']}-{name}"
    return out


def default_data_dir(variant: str) -> str:
    """Where build_strings.py writes each variant (native keeps the step-1 path)."""
    return "data/processed" if variant in (None, "native") else f"data/processed_{variant}"


def _ascending(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    return x[order], y[order]


def load_sers_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Two-column wavenumber/intensity text file.

    Handles a '#Wave  #Intensity' header, comma or whitespace separators, and
    descending axes (your WiRE exports run 3200 -> 100)."""
    xs, ys = [], []
    for line in open(path):
        line = line.strip()
        if not line or line[0] in "#;/":
            continue
        parts = line.replace(",", " ").split()
        try:
            a, b = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            continue
        xs.append(a)
        ys.append(b)
    if not xs:
        raise ValueError(f"no numeric rows in {path}")
    return _ascending(np.asarray(xs, float), np.asarray(ys, float))


def find_sers_files(sers_dir: str | Path, pattern: str = "*.txt") -> list[Path]:
    """Every spectrum under sers_dir, in subfolders too (e.g. AAs/, indoles/,
    Cell/), following symlinked folders. Skips README files and hidden
    files/folders."""
    import fnmatch
    import os
    root = Path(sers_dir)
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in filenames:
            if fn.startswith(".") or fn.lower().startswith("readme") or not fnmatch.fnmatch(fn, pattern):
                continue
            out.append(Path(dirpath) / fn)
    return sorted(out)


def subset_of(path: str | Path, sers_dir: str | Path) -> str:
    """Top-level subfolder a spectrum sits in ('AAs', 'indoles', 'Cell'), or '.'."""
    try:
        rel = Path(path).relative_to(Path(sers_dir))
    except ValueError:
        return "."
    return rel.parts[0] if len(rel.parts) > 1 else "."


def iter_qm9s_csv(path: str | Path) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Rows of QM9S `raman_boraden.csv`: first column = molecule number (1-based),
    header of the remaining columns = wavenumber axis. Streams row by row."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header and header[0].startswith("PLACEHOLDER"):
            raise FileNotFoundError(f"{path} is the placeholder, not QM9S. Symlink the real file there.")
        try:
            x = np.asarray([float(h) for h in header[1:]], float)
        except ValueError as e:
            raise ValueError(
                f"{path}: header is not a numeric wavenumber axis. Refusing to guess "
                "an axis (the SpectraLLM linspace(4000,400,n) fallback is wrong for "
                "Raman). Pass the true axis." ) from e
        order = np.argsort(x)
        x = x[order]
        for row in reader:
            if not row:
                continue
            y = np.asarray(row[1:], float)[order]
            yield int(float(row[0])), x, y


def load_number_smiles(path: str | Path) -> dict[int, str]:
    """CSV with columns number,smiles (export from qm9s.pt with
    scripts/export_qm9s_smiles.py on Aura)."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["number"])] = r["smiles"]
    return out


def load_dft_sticks(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """ORCA stick spectrum: frequency, activity/intensity per line."""
    f_, i_ = [], []
    for line in open(path):
        parts = line.strip().replace(",", " ").split()
        try:
            f_.append(float(parts[0]))
            i_.append(float(parts[1]))
        except (ValueError, IndexError):
            continue
    return np.asarray(f_, float), np.asarray(i_, float)


_PEAKLIST_RE = None


def iter_peaklist_jsonl(path: str | Path):
    """Rows of a make_views.py output (qm9s_raman.jsonl):
    {"prompt": "Raman spectroscopy {json Wavenumbers/Intensities/Widths}",
     "response": "##SMILES: <smi>"}. Yields (row_index, smiles, x, y, w)."""
    import re
    global _PEAKLIST_RE
    if _PEAKLIST_RE is None:
        _PEAKLIST_RE = re.compile(r"^[^{]*(\{.*\})\s*$")
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        d = json.loads(line)
        m = _PEAKLIST_RE.match(d["prompt"])
        if not m:
            raise ValueError(f"{path}:{i+1}: prompt has no peak JSON")
        j = json.loads(m.group(1))
        smi = d["response"].split("##SMILES:", 1)[-1].strip()
        arr = lambda k: np.asarray([float(v) for v in j[k].split(",")], float) if j.get(k) else np.array([])
        yield i, smi, arr("Wavenumbers"), arr("Intensities"), arr("Widths")


def write_jsonl(records, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]
