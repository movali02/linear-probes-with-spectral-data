"""Serialise peak lists to the model string, and parse them back.

The string is the ONLY thing the model sees. It contains numbers and separators
and nothing else, so a strict grammar check is the strongest leakage test we
have: if a string parses and re-serialises to itself, it cannot carry a name,
SMILES, filename or medium."""
from __future__ import annotations

import re

import numpy as np


def _fmt(v: float, dec: int) -> str:
    return f"{round(v, dec):.{dec}f}" if dec > 0 else str(int(round(v)))


def to_string(peaks: list[dict], cfg: dict) -> str:
    s = cfg["serialise"]
    fs, ps = s["field_sep"], s["peak_sep"]
    parts = []
    for p in sorted(peaks, key=lambda d: d["position"]):
        # Clip at 0 only. SERS under substrate.mode=reference_norm is scaled so the
        # CB[5] 829 band = 1.0, and analyte peaks can legitimately be taller than
        # that; an upper clip at 1.0 would flatten them all to "1.00".
        # Max-normalised sources (QM9S, DFT) never exceed 1.0 anyway.
        inten = max(p["intensity"], 0.0)
        parts.append(fs.join([
            _fmt(p["position"], s["pos_decimals"]),
            _fmt(inten, s["int_decimals"]),
            _fmt(abs(p["width"]), s["width_decimals"]),
        ]))
    return ps.join(parts)


def _num(dec: int) -> str:
    return r"\d+" if dec == 0 else rf"\d+\.\d{{{dec}}}"


def grammar(cfg: dict) -> re.Pattern:
    s = cfg["serialise"]
    one = re.escape(s["field_sep"]).join(
        [_num(s["pos_decimals"]), _num(s["int_decimals"]), _num(s["width_decimals"])])
    return re.compile(rf"^(?:{one})(?:{re.escape(s['peak_sep'])}{one})*$")


def parse_string(text: str, cfg: dict) -> list[dict]:
    if not grammar(cfg).match(text):
        raise ValueError(f"string does not match schema {cfg['schema']}: {text[:80]!r}")
    s = cfg["serialise"]
    out = []
    for chunk in text.split(s["peak_sep"]):
        a, b, c = chunk.split(s["field_sep"])
        out.append({"position": float(a), "intensity": float(b), "width": float(c)})
    return out


def raster(peaks: list[dict], cfg: dict, bin_cm: float = 10.0) -> np.ndarray:
    """Fixed-length feature vector from a peak list: max intensity per bin.
    Used for the baselines, the shuffled-label control and near-duplicate search."""
    lo, hi = cfg["window"]
    edges = np.arange(lo, hi + bin_cm, bin_cm)
    v = np.zeros(len(edges) - 1)
    for p in peaks:
        k = int(np.clip(np.searchsorted(edges, p["position"], side="right") - 1, 0, len(v) - 1))
        v[k] = max(v[k], p["intensity"])
    return v


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for Qwen2/2.5-style tokenisers, which split
    every digit into its own token. Letters are counted at ~3 chars per token.
    Use `count_tokens` with the real tokenizer on Aura for exact numbers."""
    n = 0
    for tok in re.findall(r"\d|[A-Za-z]+|\s+|[^\sA-Za-z\d]", text):
        if tok[0].isalpha():
            n += -(-len(tok) // 3)
        else:
            n += 1
    return n


def count_tokens(texts: list[str], tokenizer_name: str | None) -> list[int]:
    if tokenizer_name is None:
        return [estimate_tokens(t) for t in texts]
    from transformers import AutoTokenizer  # only on the GPU box
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
