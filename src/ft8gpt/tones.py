from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
import math

from .constants import FSK_TONES, FT8_GRAY_MAP


def extract_symbol_llrs(mag_bins: NDArray[np.float64]) -> Tuple[float, float, float]:
    """Order-correct per-symbol LLRs using Gray group log-sum metrics.
    Returns (l2, l1, l0) to match (b2, b1, b0) ordering.
    mag_bins: shape [8], magnitudes/energies at tone bins in natural frequency order (0..7).
    """
    return extract_symbol_llrs_gray_lse(mag_bins)


def extract_symbol_llrs_gray_lse(mag_bins: NDArray[np.float64]) -> Tuple[float, float, float]:
    """Compute per-symbol LLRs in bit order (l2,l1,l0) = (b2,b1,b0).

    Steps:
      1) Remap to Gray-ordered energies s[j] = mag_bins[FT8_GRAY_MAP[j]] with ±1 neighbor tolerance.
      2) Compute LLRs via log-sum over Gray groups for each bit.
    """
    # Build Gray-ordered energies with tolerance for ±1-bin leakage
    s = np.empty(8, dtype=np.float64)
    for j in range(8):
        tone = FT8_GRAY_MAP[j]
        e0 = float(mag_bins[tone])
        em = float(mag_bins[tone - 1]) if tone - 1 >= 0 else 0.0
        ep = float(mag_bins[tone + 1]) if tone + 1 < 8 else 0.0
        s[j] = max(e0, em, ep, 1e-20)

    # Bit grouping in Gray-index space j ∈ [0..7]
    g2_0 = (0, 1, 2, 3); g2_1 = (4, 5, 6, 7)
    g1_0 = (0, 1, 4, 5); g1_1 = (2, 3, 6, 7)
    g0_0 = (0, 2, 4, 6); g0_1 = (1, 3, 5, 7)

    def lse(idx: tuple[int, ...]) -> float:
        return math.log(sum(s[k] for k in idx))

    l2 = lse(g2_0) - lse(g2_1)
    l1 = lse(g1_0) - lse(g1_1)
    l0 = lse(g0_0) - lse(g0_1)
    return l2, l1, l0


