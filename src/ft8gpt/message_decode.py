from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DecodedStd:
    call_to: str
    call_de: str
    extra: str


# Constants per FT8 spec
MAX22 = 4194304
NTOKENS = 2063592


def _ch_alphanum_space(v: int) -> str:
    """Inverse of packer's nchar_alphanum_space:
    0 -> ' ', 1..10 -> '0'..'9', 11..36 -> 'A'..'Z'.
    """
    if v == 0:
        return ' '
    if 1 <= v <= 10:
        return chr(ord('0') + (v - 1))
    if 11 <= v <= 36:
        return chr(ord('A') + (v - 11))
    return ' '


def _ch_alphanum(v: int) -> str:
    """Inverse of packer's nchar_alphanum: 0..9 -> '0'..'9', 10..35 -> 'A'..'Z'."""
    if 0 <= v <= 9:
        return chr(ord('0') + v)
    if 10 <= v <= 35:
        return chr(ord('A') + (v - 10))
    return ' '


def _ch_lspace(v: int) -> str:
    """Inverse of packer's nchar_letters_space: 0 -> ' ', 1..26 -> 'A'..'Z'."""
    if v == 0:
        return ' '
    if 1 <= v <= 26:
        return chr(ord('A') + (v - 1))
    return ' '


def _unpack_basecall(n: int) -> str:
    c = [' '] * 6
    c[5] = _ch_lspace(n % 27); n //= 27
    c[4] = _ch_lspace(n % 27); n //= 27
    c[3] = _ch_lspace(n % 27); n //= 27
    c[2] = chr(ord('0') + (n % 10)); n //= 10
    d = n % 36; n //= 36
    c[1] = _ch_alphanum(d)
    c[0] = _ch_alphanum_space(n % 37)
    return ''.join(c).strip()


def _unpack28(n28: int) -> str:
    """Decode a 28-bit token into a callsign or special token (CQ/DE/QRZ).

    Returns an empty string for unsupported tokens (e.g., hashed or non-standard).
    """
    # Tokens CQ/DE/QRZ
    if n28 < NTOKENS:
        if n28 == 2:
            return 'CQ'
        if n28 == 0:
            return 'DE'
        if n28 == 1:
            return 'QRZ'
        return ''
    # Basecalls
    n = n28 - NTOKENS - MAX22
    return _unpack_basecall(n)


def unpack_standard_payload(a10: bytes) -> DecodedStd:
    """Unpack 77-bit standard message (i3 in {1,2}) into calls and grid/report.
    Implements FT8 standard message reconstruction for tokens and base calls.
    """
    n29a = int.from_bytes(bytes([a10[0], a10[1], a10[2], a10[3] & 0xF8]), 'big') >> 3
    n29b = ((a10[3] & 0x07) << 26) | (a10[4] << 18) | (a10[5] << 10) | (a10[6] << 2) | (a10[7] >> 6)
    igrid4 = ((a10[7] & 0x3F) << 10) | (a10[8] << 2) | (a10[9] >> 6)

    # Drop suffix bit
    n28a = n29a >> 1
    n28b = n29b >> 1

    call_to = _unpack28(n28a)
    call_de = _unpack28(n28b)

    # grid
    if igrid4 <= 32400:
        g = igrid4
        d0 = g % 10; g //= 10
        d1 = g % 10; g //= 10
        l1 = chr(ord('A') + (g % 18)); g //= 18
        l0 = chr(ord('A') + (g % 18))
        extra = f"{l0}{l1}{d1}{d0}"
    else:
        extra = ''

    return DecodedStd(call_to=call_to.strip(), call_de=call_de.strip(), extra=extra)


