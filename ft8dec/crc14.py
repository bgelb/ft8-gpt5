from __future__ import annotations
import numpy as np

# FT8 uses a 14-bit CRC; we will verify the polynomial by cross-checking against known vectors.
# Placeholder polynomial; unit tests will guide correction if needed.
POLY = 0x2757
WIDTH = 14
MASK = (1 << WIDTH) - 1
INIT = 0


def crc14(bits: np.ndarray) -> int:
    """Compute FT8 CRC-14 over bit array of length >= 1.

    bits: numpy array of shape (N,), dtype uint8 or bool, values in {0,1}
    Returns integer CRC value in range [0, 2^14-1].
    """
    if bits.ndim != 1:
        raise ValueError("bits must be 1-D array")
    reg = int(INIT)
    for b in bits.astype(np.uint8):
        reg ^= (int(b) & 1) << (WIDTH - 1)
        msb = (reg >> (WIDTH - 1)) & 1
        reg = ((reg << 1) & MASK)
        if msb:
            reg ^= POLY & MASK
    return int(reg & MASK)
