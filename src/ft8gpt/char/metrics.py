from __future__ import annotations

import math
import numpy as np
from typing import Iterable, Sequence


def recall_at_k(truth_indices: Iterable[float], candidate_scores: Sequence[tuple[float, float]], tol: float) -> float:
    """Compute recall@K over frequency-like axis.

    - truth_indices: list of ground-truth target values (e.g., Hz)
    - candidate_scores: list of (value, score) sorted by descending score
    - tol: absolute tolerance for a hit (same units as value)
    """
    if not truth_indices:
        return 0.0
    if not candidate_scores:
        return 0.0
    truths = list(truth_indices)
    cvals = [v for (v, _s) in candidate_scores]
    hits = 0
    for t in truths:
        ok = any(abs(cv - t) <= tol for cv in cvals)
        hits += 1 if ok else 0
    return float(hits) / float(len(truths))


def rmse(errors: Iterable[float]) -> float:
    arr = np.array(list(errors), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def bit_mutual_information(llrs: np.ndarray, bits: np.ndarray) -> float:
    """Estimate bit mutual information from LLRs and ground-truth bits.

    BMI ≈ 1 - (1/N) * sum_i log2(1 + exp(-(-1)^b_i * L_i))
    """
    if llrs.size == 0 or bits.size == 0:
        return 0.0
    b = bits.astype(np.float64)
    s = 1.0 - 2.0 * b  # (-1)^b = 1 for b=0; -1 for b=1 -> use sign trick via sgn
    z = np.log2(1.0 + np.exp(-(s * llrs.astype(np.float64))))
    return float(1.0 - np.mean(z))