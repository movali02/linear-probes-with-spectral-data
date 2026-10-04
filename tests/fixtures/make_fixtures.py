"""Regenerate the tiny SYNTHETIC QM9S fixture used by the tests.

The spectra are made-up Lorentzian sums in the raman_boraden.csv layout (first
column = molecule number, header = wavenumber axis). They test the plumbing
only; they are not QM9S data."""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SMILES = ["C", "CO", "CC(=O)O", "NCC(=O)O", "CC(N)C(=O)O", "NC(CO)C(=O)O", "CC(O)C(N)C(=O)O",
          "c1ccccc1", "Oc1ccccc1", "c1c[nH]cn1", "c1ccc2[nH]ccc2c1", "CC(C)C(N)C(=O)O", "O=C(O)C1CCCN1"]


def main():
    rng = np.random.default_rng(0)
    x = np.arange(400.0, 4000.0 + 0.5, 2.0)
    out = HERE / "qm9s_mini"
    out.mkdir(exist_ok=True)
    with open(out / "raman_boraden.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number"] + [f"{v:.1f}" for v in x])
        for num in range(1, len(SMILES) + 2):  # last number has no SMILES on purpose
            pos = np.concatenate([rng.uniform(500, 1750, rng.integers(3, 12)), rng.uniform(2850, 3100, 3)])
            amp = np.concatenate([rng.uniform(0.05, 0.4, len(pos) - 3), [1.0, 0.8, 0.6]])  # C-H dominates
            hw = 8.0
            y = (amp[None] * hw**2 / ((x[:, None] - pos[None]) ** 2 + hw**2)).sum(1)
            w.writerow([num] + [f"{v:.5f}" for v in y])
    with open(out / "number_smiles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "smiles"])
        for i, s in enumerate(SMILES, 1):
            w.writerow([i, s])


if __name__ == "__main__":
    main()
