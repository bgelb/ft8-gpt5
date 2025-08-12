from __future__ import annotations
import numpy as np

# FT8 uses CRC-14 with polynomial x^14 + x^13 + x^12 + x^11 + x^8 + x^6 + x^5 + x^4 + x^2 + x + 1
# Polynomial (no top bit) as integer: 0x2757 (per WSJT-X/FT8 spec)
CRC14_POLY = 0x2757
CRC14_MASK = (1 << 14) - 1


def crc14(bits: np.ndarray) -> int:
    """Compute CRC-14 over a bit array (numpy array of {0,1}), MSB-first."""
    reg = 0
    for bit in bits.astype(np.uint8):
        reg ^= (bit & 1) << 13
        feedback = (reg >> 13) & 1
        reg = ((reg << 1) & CRC14_MASK)
        if feedback:
            reg ^= CRC14_POLY
    return reg & CRC14_MASK


def crc14_check(bits_with_crc: np.ndarray) -> bool:
    """Return True if CRC-14 over payload equals appended 14-bit CRC (MSB-first)."""
    if bits_with_crc.size < 14:
        return False
    payload = bits_with_crc[:-14]
    crc_bits = bits_with_crc[-14:]
    expected = crc14(payload)
    got = 0
    for b in crc_bits.astype(np.uint8):
        got = (got << 1) | (b & 1)
    return expected == got


