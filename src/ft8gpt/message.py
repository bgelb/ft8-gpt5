from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Payload77:
    bits: NDArray[np.uint8]  # length 77, MSB-first semantics when packed


def pack_bits_msb_first(bits: NDArray[np.uint8]) -> bytes:
    nbits = bits.size
    nbytes = (nbits + 7) // 8
    out = np.zeros(nbytes, dtype=np.uint8)
    mask = 0x80
    i_byte = 0
    for i in range(nbits):
        if bits[i] & 1:
            out[i_byte] |= mask
        mask >>= 1
        if mask == 0:
            mask = 0x80
            i_byte += 1
    return out.tobytes()


