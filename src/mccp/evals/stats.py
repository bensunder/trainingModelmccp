"""Small, dependency-light statistics for paired eval comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


def paired_bootstrap_ci(
    a: Sequence[float], b: Sequence[float], *, iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float | int]:
    """Bootstrap CI for mean(a) - mean(b) over paired items."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def bootstrap_mean_ci(
    xs: Sequence[float], *, iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float | int]:
    n = len(xs)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    arr = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def percentile(xs: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else 0.0


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
