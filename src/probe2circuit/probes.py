"""Linear probes on saved activations, and their scores along the shift ladder.

Two probes per label, layer and position (roadmap 4.2):
  lr   logistic regression: standardised activations, L2, class-balanced.
       All labels are fitted at once (one linear layer with K outputs), with
       L-BFGS in torch, so a 4096-wide layer takes seconds.
  mm   mass-mean: the difference of the class means (Marks & Tegmark 2023),
       no fitting; reported to generalise better across distributions.

Hygiene: probes are fitted on QM9S train; the regularisation C and the layer
are chosen on a QM9S validation split; QM9S test (rung R0) and every other rung
are only scored. Nothing about SERS is used to choose anything.

The same functions take any feature matrix, so the classical baseline (F1,
the binned string) goes through the identical procedure."""
from __future__ import annotations

import numpy as np

from .baselines import fast_auroc


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return [((X - mu) / sd).astype(np.float32) for X in (Xtr, *others)], (mu, sd)


def fit_logreg(X, Y, C: float = 0.01, max_iter: int = 300, device: str | None = None):
    """Multi-label logistic regression. X [n, d] standardised, Y [n, K] in {0, 1}.
    Minimises  mean_i sum_k w_ik BCE  +  ||W||^2 / (2 C n),  w = class-balanced.
    Returns (W [d, K], b [K]) as numpy."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    n, d = Xt.shape
    K = Yt.shape[1]
    npos = Yt.sum(0).clamp(min=1)
    nneg = (n - Yt.sum(0)).clamp(min=1)
    wt = Yt * (n / (2 * npos)) + (1 - Yt) * (n / (2 * nneg))
    W = torch.zeros(d, K, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def closure():
        opt.zero_grad()
        loss = (bce(Xt @ W + b, Yt, reduction="none") * wt).sum(1).mean() + (W ** 2).sum() / (2 * C * n)
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().cpu().numpy(), b.detach().cpu().numpy()


def mass_mean(X, Y):
    """Direction per label = mean(positives) - mean(negatives); threshold at the midpoint."""
    Y = np.asarray(Y, float)
    npos, nneg = np.maximum(Y.sum(0), 1), np.maximum((1 - Y).sum(0), 1)
    mp, mn = (X.T @ Y) / npos, (X.T @ (1 - Y)) / nneg
    W = mp - mn
    b = -((mp + mn) / 2 * W).sum(0)
    return W.astype(np.float32), b.astype(np.float32)


def auroc_cols(Y, S) -> np.ndarray:
    return np.array([fast_auroc(Y[:, k], S[:, k]) for k in range(Y.shape[1])])


def by_analyte(scores, labels, analytes):
    """Mean score and label per analyte: [A, K] arrays and the analyte names."""
    names = sorted(set(analytes))
    a = np.asarray(analytes)
    S = np.stack([scores[a == n].mean(0) for n in names])
    Y = np.stack([(labels[a == n].mean(0) >= 0.5).astype(int) for n in names])
    return S, Y, names


def boot_ci(Y, S, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap over rows (analytes or molecules) of the AUROC of one label."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = fast_auroc(Y[i], S[i])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(10, n_boot // 10):
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def centre_domains(X, domains, groups=None):
    """Subtract each domain's mean (mean of group means): the label-free recalibration."""
    from .baselines import center_by_domain
    return center_by_domain(X, domains, groups).astype(np.float32)
