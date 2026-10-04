"""Synthetic raw data with the real file layout, for the step-2 tests.

SERS: data/raw_sers/{Cell,AAs,indoles}/ with Cell_<strain>_<code>_<n>.txt, the
paired CB_<strain>_<code>_<n>.txt, and one file per standard. Every analyte has
its own peaks (optionally shifted per strain), every spectrum carries the CB[5]
bands (755, 829, 880), an ALS-removable baseline and noise.

QM9S: raman_boraden.csv (number, then a wavenumber-axis header) and the number
-> SMILES map, with spectra whose peaks depend on the functional groups."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

AAS = ["ala", "arg", "asn", "asp", "cys", "glu", "gln", "gly", "his", "ile",
       "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val"]
STD_NAMES = {"ala": "L-alanine", "arg": "L-arginine", "asn": "L-asparagine", "asp": "L-aspartic acid",
             "cys": "L-cysteine", "glu": "L-glutamic acid", "gln": "L-glutamine", "gly": "L-glycine",
             "his": "L-histidine", "ile": "L-isoleucine", "leu": "L-leucine", "lys": "L-lysine",
             "met": "L-methionine", "phe": "L-phenylalanine", "pro": "L-proline", "ser": "L-serine",
             "thr": "L-threonine", "trp": "L-tryptophan", "tyr": "L-tyrosine", "val": "L-valine"}
INDOLES = ["indole", "indole-d6", "oxindole", "tryptophol", "isatin", "8-hydroxyquinoline"]
CB = [(617, 0.12, 20), (676, 0.10, 22), (755, 0.18, 14), (829, 1.0, 12), (880, 0.30, 14)]


def _lor(x, peaks):
    y = np.zeros_like(x)
    for p, h, w in peaks:
        y += h * (w / 2) ** 2 / ((x - p) ** 2 + (w / 2) ** 2)
    return y


def _write(path, x, y):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("#Wave\t#Intensity\n")
        for a, b in zip(x[::-1], y[::-1]):          # descending axis, like WiRE exports
            f.write(f"{a:.2f}\t{b:.3f}\n")


def make_sers_dir(root: Path, n_scans: int = 4, signal: float = 1.0, seed: int = 0, strains=("BW", "536"),
                  strain_shift: float = 0.0, standards: bool = True, strain_bg: float = 0.0) -> Path:
    """strain_bg > 0 adds a strain-specific background (the same bands in every
    culture of a strain), the situation the '.bg' preprocessing is for."""
    rng = np.random.default_rng(seed)
    x = np.linspace(300, 1900, 1601)
    fp = {a: [(float(p), float(h), 16.0) for p, h in zip(rng.uniform(950, 1700, 4), rng.uniform(0.4, 1.2, 4))]
          for a in AAS + INDOLES}
    base = 400 + 0.3 * (x - 300)
    d = Path(root) / "raw_sers"
    bgp = {st: [(float(p), strain_bg * float(h), 25.0) for p, h in zip(rng.uniform(900, 1700, 6), rng.uniform(0.5, 1.5, 6))]
           for st in strains}
    for si, st in enumerate(strains):
        for a in AAS:
            for n in range(1, n_scans + 1):
                amp = 3000 * rng.uniform(0.8, 1.2)
                pk = [(p + strain_shift * si, h * signal, w) for p, h, w in fp[a]]
                y = amp * (_lor(x, CB) + _lor(x, pk) + _lor(x, bgp[st])) + base + rng.normal(0, 25, len(x))
                _write(d / "Cell" / f"Cell_{st}_{a}_{n}.txt", x, y)
                ycb = amp * _lor(x, CB) + base + rng.normal(0, 25, len(x))
                _write(d / "Cell" / f"CB_{st}_{a}_{n}.txt", x, ycb)
    for st in strains:                                      # no-analyte control cultures, numbered from 0
        for n in range(3):
            amp = 3000 * rng.uniform(0.8, 1.2)
            _write(d / "Cell" / f"Cell_{st}_ctrl_{n}.txt", x,
                   amp * (_lor(x, CB) + _lor(x, bgp[st])) + base + rng.normal(0, 25, len(x)))
            _write(d / "Cell" / f"CB_{st}_ctrl_{n}.txt", x, amp * _lor(x, CB) + base + rng.normal(0, 25, len(x)))
    if standards:
        for a in AAS:
            y = 3000 * (_lor(x, CB) + _lor(x, fp[a])) + base + rng.normal(0, 25, len(x))
            _write(d / "AAs" / f"{STD_NAMES[a]}.txt", x, y)
        for a in INDOLES:
            y = 3000 * (_lor(x, CB) + _lor(x, fp[a])) + base + rng.normal(0, 25, len(x))
            _write(d / "indoles" / f"{a}.txt", x, y)
    return d


QM9_SMILES = ["CO", "CCO", "CC(C)O", "OCCO", "CN", "CCN", "NCCN", "CC(N)C", "c1ccccc1", "Cc1ccccc1",
              "Oc1ccccc1", "Nc1ccccc1", "c1ccncc1", "c1cc[nH]c1", "CC(=O)C", "CC=O", "CC(=O)N", "NC(=O)C=O",
              "CC#N", "C#C", "C=CC", "COC", "CCOC", "CC(=O)OC", "OCC#N", "NCC=O", "OC1CC1", "NC1CC1",
              "O=C1CCC1", "C1CCOC1", "c1cnc[nH]1", "Cc1cc[nH]c1", "OCc1ccccc1", "CC(O)C#N", "CCCC", "CC(C)C",
              "FCC", "FC(F)C", "OCC=O", "NCCO"]
FG_PEAKS = {"hydroxyl": 1050, "amine": 1600, "aromatic": 1000, "carbonyl": 1700, "nitrile": 1740,
            "ether": 1120, "alkene": 1650, "fluoro": 1150}


def _qm9_fg(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    pats = {"hydroxyl": "[OX2H]", "amine": "[NX3;H2]", "aromatic": "a", "carbonyl": "C=O", "nitrile": "C#N",
            "ether": "[OD2]([#6])[#6]", "alkene": "C=C", "fluoro": "F"}
    return [k for k, p in pats.items() if m.HasSubstructMatch(Chem.MolFromSmarts(p))]


def make_qm9s(root: Path, n: int = 800, seed: int = 0) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    x = np.arange(400.0, 4000.0 + 1, 2.0)
    root = Path(root) / "qm9s"
    root.mkdir(parents=True, exist_ok=True)
    csv_p, map_p = root / "raman_boraden.csv", root / "number_smiles.csv"
    with open(csv_p, "w", newline="") as f, open(map_p, "w", newline="") as g:
        w, wm = csv.writer(f), csv.writer(g)
        w.writerow(["number"] + [f"{v:.1f}" for v in x])
        wm.writerow(["number", "smiles", "list_pos", "qm9_number"])
        for i in range(n):
            smi = QM9_SMILES[i % len(QM9_SMILES)]
            peaks = [(FG_PEAKS[k] + rng.normal(0, 4), rng.uniform(0.5, 1.0), 10.0) for k in _qm9_fg(smi)]
            peaks += [(p, rng.uniform(0.1, 0.6), 10.0) for p in rng.uniform(550, 1700, 3)]
            peaks += [(2950, 2.0, 10.0)]                                     # C-H stretch outside the window
            y = _lor(x, peaks)
            w.writerow([i + 1] + [f"{v:.5f}" for v in y])
            wm.writerow([i + 1, smi, i, i + 1])
    return csv_p, map_p


# ---------------------------------------------------------------- ladder (DFT sticks, tiny language model)
def make_dft_sticks(root: Path, molecules_csv, analytes_csv, seed: int = 0, scale: float = 0.98) -> Path:
    """Stick files laid out like dft/dft_to_sticks.py output. Band positions follow the
    functional groups of the neutral parent (as in make_qm9s), written UNSCALED."""
    rng = np.random.default_rng(seed)
    import json
    parent = {r["name"]: r["smiles"] for r in csv.DictReader(open(analytes_csv))}
    out = Path(root) / "dft_sticks"
    smap = {}
    for r in csv.DictReader(open(molecules_csv)):
        if r["set"] == "pI_water":
            continue
        smi = parent.get(r["analyte"], r["smiles"])
        smap.setdefault(r["set"], {})[r["name"]] = smi
        peaks = [(FG_PEAKS[k], rng.uniform(0.5, 1.0)) for k in _qm9_fg(smi)] + \
                [(p, rng.uniform(0.1, 0.6)) for p in rng.uniform(550, 1700, 3)] + [(2950.0, 2.0)]
        for variant, tilt in (("activity", 0.0), ("int785", 0.3), ("gapiso785", 0.5)):
            d = out / r["set"] / variant
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{r['name']}.txt", "w") as f:
                f.write(f"# freq_cm-1_unscaled {variant}\n")
                for p, h in peaks:
                    f.write(f"{p / scale:.4f} {h * (1 + tilt * (1700 - p) / 1200):.6e}\n")
    for s, m in smap.items():
        (out / s / "smiles_map.json").write_text(json.dumps(m))
    return out


def make_tiny_lm(path: Path, n_layer: int = 2, d: int = 24) -> Path:
    """A random 2-layer GPT-2 with a character-level tokenizer and a ChatML template, saved to
    `path`: enough to test activation extraction offline."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    chars = sorted(set("0123456789 .|:,()?-_<>\n" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    vocab = {"<pad>": 0, "<unk>": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}<im_start>{{ m['role'] }}\n{{ m['content'] }}<im_end>\n{% endfor %}"
                         "{% if add_generation_prompt %}<im_start>assistant\n{% endif %}")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=2048, n_embd=d, n_layer=n_layer, n_head=2))
    Path(path).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(path)
    model.save_pretrained(path)
    return Path(path)
