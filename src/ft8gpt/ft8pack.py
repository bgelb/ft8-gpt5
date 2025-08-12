from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

MAX22 = 4194304  # 2^22
NTOKENS = 2063592


def _is_letter(c: str) -> bool:
    return 'A' <= c <= 'Z'


def _is_digit(c: str) -> bool:
    return '0' <= c <= '9'


def pack_basecall(callsign: str) -> int:
    callsign = callsign.upper()
    length = len(callsign)
    if length < 3:
        return -1
    c6 = [' ', ' ', ' ', ' ', ' ', ' ']
    if length > 2:
        if length <= 6 and _is_digit(callsign[1]):
            # A0XYZ -> " A0XYZ"
            for i, ch in enumerate(callsign):
                c6[i + 1] = ch
        elif length <= 6 and _is_digit(callsign[2]):
            # AB0XYZ
            for i, ch in enumerate(callsign):
                c6[i] = ch
        else:
            return -1
    else:
        return -1

    def nchar_alphanum_space(ch: str) -> int:
        if ch == ' ':
            return 36
        if _is_digit(ch):
            return ord(ch) - ord('0') + 26
        if _is_letter(ch):
            return ord(ch) - ord('A')
        return -1

    def nchar_alphanum(ch: str) -> int:
        if _is_digit(ch):
            return ord(ch) - ord('0') + 26
        if _is_letter(ch):
            return ord(ch) - ord('A')
        return -1

    def nchar_numeric(ch: str) -> int:
        if _is_digit(ch):
            return ord(ch) - ord('0')
        return -1

    def nchar_letters_space(ch: str) -> int:
        if ch == ' ':
            return 0
        if _is_letter(ch):
            return (ord(ch) - ord('A')) + 1
        return -1

    i0 = nchar_alphanum_space(c6[0])
    i1 = nchar_alphanum(c6[1])
    i2 = nchar_numeric(c6[2])
    i3 = nchar_letters_space(c6[3])
    i4 = nchar_letters_space(c6[4])
    i5 = nchar_letters_space(c6[5])
    if min(i0, i1, i2, i3, i4, i5) < 0:
        return -1
    n = i0
    n = n * 36 + i1
    n = n * 10 + i2
    n = n * 27 + i3
    n = n * 27 + i4
    n = n * 27 + i5
    return n


def pack28(callsign: str) -> Tuple[int, int]:
    """Return (n28, ip) encoding for a callsign/token. Supports tokens CQ/DE/QRZ and standard calls.
    ip (1 bit) is suffix flag, always 0 here.
    """
    callsign = callsign.upper()
    if callsign == 'DE':
        return 0, 0
    if callsign == 'QRZ':
        return 1, 0
    if callsign == 'CQ':
        return 2, 0
    ip = 0
    n28 = pack_basecall(callsign)
    if n28 < 0:
        raise ValueError('Only standard callsigns supported in this packer')
    return NTOKENS + MAX22 + n28, ip


def packgrid(grid4: str) -> Tuple[int, int]:
    grid4 = grid4.upper()
    # returns (igrid4, ir)
    if len(grid4) == 4 and ('A' <= grid4[0] <= 'R') and ('A' <= grid4[1] <= 'R') and grid4[2].isdigit() and grid4[3].isdigit():
        n = (ord(grid4[0]) - ord('A'))
        n = n * 18 + (ord(grid4[1]) - ord('A'))
        n = n * 10 + (ord(grid4[2]) - ord('0'))
        n = n * 10 + (ord(grid4[3]) - ord('0'))
        return n, 0
    raise ValueError('Unsupported grid format for this minimal packer')


def pack_standard_payload(call_to: str, call_de: str, grid4: str) -> bytes:
    """Pack a standard FT8 message (type i3=1) into 77 bits stored as 10 bytes (MSB-first)."""
    n28a, ipa = pack28(call_to)
    n28b, ipb = pack28(call_de)
    igrid4, ir = packgrid(grid4)
    n29a = (n28a << 1) | (ipa & 1)
    n29b = (n28b << 1) | (ipb & 1)

    a = bytearray(10)
    a[0] = (n29a >> 21) & 0xFF
    a[1] = (n29a >> 13) & 0xFF
    a[2] = (n29a >> 5) & 0xFF
    a[3] = ((n29a << 3) & 0xF8) | ((n29b >> 26) & 0x07)
    a[4] = (n29b >> 18) & 0xFF
    a[5] = (n29b >> 10) & 0xFF
    a[6] = (n29b >> 2) & 0xFF
    a[7] = ((n29b << 6) & 0xC0) | ((igrid4 >> 10) & 0x3F)
    a[8] = (igrid4 >> 2) & 0xFF
    a[9] = ((igrid4 << 6) & 0xC0) | ((1 & 0x07) << 3)  # i3 = 1 (standard)
    return bytes(a)


