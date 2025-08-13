from __future__ import annotations

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import FSK_TONES, gray_to_bits


def extract_symbol_llrs(mag_bins: NDArray[np.float64]) -> Tuple[float, float, float]:
    """Compute unnormalized LLRs (b2,b1,b0) using max-log over Gray-mapped tones.
    mag_bins: shape [8], magnitudes at tone bins in frequency order (tone index == Gray code value).
    """
    s = mag_bins
    # Build bit masks once per call (small fixed size of 8)
    bit_masks_one = [[], [], []]  # indices where respective bit is 1
    bit_masks_zero = [[], [], []]  # indices where respective bit is 0
    for tone_idx in range(FSK_TONES):
        b2, b1, b0 = gray_to_bits(tone_idx)
        bits = (b2, b1, b0)
        for i in range(3):
            if bits[i] == 1:
                bit_masks_one[i].append(tone_idx)
            else:
                bit_masks_zero[i].append(tone_idx)

    # Max over groups for each bit position (order: b2,b1,b0)
    llrs = []
    for i in range(3):
        m1 = max(s[idx] for idx in bit_masks_one[i])
        m0 = max(s[idx] for idx in bit_masks_zero[i])
        llrs.append(m1 - m0)

    return float(llrs[0]), float(llrs[1]), float(llrs[2])


