from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import FSK_TONES, FT8_GRAY_MAP


def extract_symbol_llrs(mag_bins: NDArray[np.float64]) -> Tuple[float, float, float]:
    """Compute unnormalized LLRs for the 3 bits using max-log metric.
    mag_bins: shape [8], magnitudes at tone bins in natural frequency order (0..7).

    We first remap to Gray-ordered indexing s_gray[j] = mag_bins[FT8_GRAY_MAP[j]],
    such that index j corresponds to binary bits (b2,b1,b0) of the tone.
    """
    # Build Gray-ordered energies with tolerance for ±1-bin leakage
    # Use a small penalty for neighbors to discourage large shifts while allowing slight offsets.
    s = np.empty(8, dtype=np.float64)
    for j in range(8):
        fidx = FT8_GRAY_MAP[j]
        candidates = [float(mag_bins[fidx])]
        if fidx - 1 >= 0:
            candidates.append(float(mag_bins[fidx - 1]))
        if fidx + 1 < 8:
            candidates.append(float(mag_bins[fidx + 1]))
        s[j] = max(candidates)

    # LLRs following the simple max-of-groups scheme.
    # Define LLR = logP(bit=0) - logP(bit=1) so that negative implies bit=1.
    llr0 = max(s[0], s[1], s[2], s[3]) - max(s[4], s[5], s[6], s[7])
    llr1 = max(s[0], s[1], s[4], s[5]) - max(s[2], s[3], s[6], s[7])
    llr2 = max(s[0], s[2], s[4], s[6]) - max(s[1], s[3], s[5], s[7])
    return llr0, llr1, llr2


