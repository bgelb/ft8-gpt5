from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DecodedStd:
    call_to: str
    call_de: str
    extra: str


def unpack_standard_payload(a10: bytes) -> DecodedStd:
    """Unpack 77-bit standard message (i3 in {1,2}) into calls and grid/report.
    Minimal implementation for smoke tests; not complete for all corner cases.
    """
    b = np.unpackbits(np.frombuffer(a10, dtype=np.uint8))
    n29a = int.from_bytes(bytes([a10[0], a10[1], a10[2], a10[3] & 0xF8]), 'big') >> 3
    n29b = ((a10[3] & 0x07) << 26) | (a10[4] << 18) | (a10[5] << 10) | (a10[6] << 2) | (a10[7] >> 6)
    igrid4 = ((a10[7] & 0x3F) << 10) | (a10[8] << 2) | (a10[9] >> 6)
    # Drop suffix bit
    n28a = n29a >> 1
    n28b = n29b >> 1
    # Only support standard calls here
    MAX22 = 4194304
    NTOKENS = 2063592
    def unpack28_to_call(n28: int) -> str:
        if n28 < NTOKENS:
            return "CQ" if n28 == 2 else "<TOK>"
        n = n28 - NTOKENS - MAX22
        # decode basecall
        c = [' '] * 6
        c[5] = _ch_lspace(n % 27); n //= 27
        c[4] = _ch_lspace(n % 27); n //= 27
        c[3] = _ch_lspace(n % 27); n //= 27
        c[2] = chr(ord('0') + (n % 10)); n //= 10
        d = n % 36; n //= 36
        c[1] = _ch_alphanum(d)
        c[0] = _ch_alphanum_space(n % 37)
        return ''.join(c).strip()
    def _ch_alphanum_space(v: int) -> str:
        if v == 36: return ' '
        return _ch_alphanum(v)
    def _ch_alphanum(v: int) -> str:
        if v < 26: return chr(ord('A') + v)
        return chr(ord('0') + (v - 26))
    def _ch_lspace(v: int) -> str:
        if v == 0: return ' '
        return chr(ord('A') + (v - 1))

    call_to = unpack28_to_call(n28a)
    call_de = unpack28_to_call(n28b)
    # grid
    if igrid4 <= 32400:
        g = igrid4
        d0 = g % 10; g //= 10
        d1 = g % 10; g //= 10
        l1 = chr(ord('A') + (g % 18)); g //= 18
        l0 = chr(ord('A') + (g % 18))
        extra = f"{l0}{l1}{d1}{d0}"
    else:
        extra = ""
    return DecodedStd(call_to=call_to, call_de=call_de, extra=extra)


