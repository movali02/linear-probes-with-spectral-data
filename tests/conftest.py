import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from probe2circuit.io import load_config  # noqa: E402
from probe2circuit.pipeline import load_analytes  # noqa: E402
from probe2circuit.strings import to_string  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "string_v1.yaml")


@pytest.fixture(scope="session")
def analytes():
    return load_analytes(ROOT / "data" / "analytes.csv")


@pytest.fixture(scope="session")
def root():
    return ROOT


def synth_records(cfg, n_classes=4, per_class=8, signal=True, seed=0, drift=0.0):
    """Synthetic peak-list records. With signal=True each class has its own
    characteristic peaks; with signal=False all classes share one distribution."""
    rng = np.random.default_rng(seed)
    base = {c: rng.uniform(550, 1700, 5) for c in range(n_classes)}
    recs = []
    for c in range(n_classes):
        for k in range(per_class):
            pos = base[c] if signal else base[0]
            pos = pos + rng.normal(0, 2, len(pos)) + drift
            extra = rng.uniform(550, 1700, 3)
            peaks = [{"position": p, "intensity": rng.uniform(0.3, 1.0), "width": 12} for p in pos]
            peaks += [{"position": p, "intensity": rng.uniform(0.05, 0.3), "width": 12} for p in extra]
            top = max(p["intensity"] for p in peaks)
            for p in peaks:
                p["intensity"] /= top
            recs.append({"id": f"s{c}_{k}", "group": f"s{c}_{k}", "source": "sers", "kind": "cell",
                         "analyte": f"class{c}", "text": to_string(peaks, cfg),
                         "split": "test" if k < 2 else "train"})
    return recs
