from __future__ import annotations
import numpy as np
from .ldpc_data import FT8_GRAY

# Inverse gray map: tone -> bits (0..7)
INV_GRAY = {tone: idx for idx, tone in enumerate(FT8_GRAY)}


def symbol_llrs(block_mags: np.ndarray) -> np.ndarray:
    """Compute 3 LLRs per FT8 data symbol from 8 magnitude bins using max-log.
    Returns array shape (3,).
    """
    s2 = np.zeros(8, dtype=np.float32)
    for j in range(8):
        s2[j] = block_mags[FT8_GRAY[j]]
    llr0 = max(s2[4], s2[5], s2[6], s2[7]) - max(s2[0], s2[1], s2[2], s2[3])
    llr1 = max(s2[2], s2[3], s2[6], s2[7]) - max(s2[0], s2[1], s2[4], s2[5])
    llr2 = max(s2[1], s2[3], s2[5], s2[7]) - max(s2[0], s2[2], s2[4], s2[6])
    return np.array([llr0, llr1, llr2], dtype=np.float32)
