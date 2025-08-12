from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import FSK_TONES


def extract_symbol_llrs(mag_bins: NDArray[np.float64]) -> Tuple[float, float, float]:
    """Compute unnormalized LLRs for the 3 bits using max-log metric over Gray-mapped tones.
    mag_bins: shape [8], magnitudes at tone bins (already aligned and Gray-ordered).
    """
    s = mag_bins
    # LLRs following the simple max-of-groups scheme used widely
    llr0 = max(s[4], s[5], s[6], s[7]) - max(s[0], s[1], s[2], s[3])
    llr1 = max(s[2], s[3], s[6], s[7]) - max(s[0], s[1], s[4], s[5])
    llr2 = max(s[1], s[3], s[5], s[7]) - max(s[0], s[2], s[4], s[6])
    return llr0, llr1, llr2


