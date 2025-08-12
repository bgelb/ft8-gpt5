from __future__ import annotations
import numpy as np

# FT8 uses CRC-14 with polynomial x^14 + x^13 + x^12 + x^11 + x^8 + x^6 + x^5 + x^4 + x^2 + x + 1
# Polynomial (no top bit) as integer: 0x2757 (per WSJT-X/FT8 spec)
CRC14_POLY = 0x2757
CRC14_MASK = (1 << 14) - 1


def crc14(bits_77: np.ndarray) -> int:
    """
    FT8 CRC-14 over 82 bits: the 77-bit payload zero-extended by 5 zero bits, MSB-first.
    Returns the 14-bit CRC value as an int.
    """
    reg: int = 0
    total_bits = 77 + 5
    for i in range(total_bits):
        bit = int(bits_77[i]) if i < 77 else 0
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
    for b in crc_bits:
        got = (got << 1) | (int(b) & 1)
    return expected == got


