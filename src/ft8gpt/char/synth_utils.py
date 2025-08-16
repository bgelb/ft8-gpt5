from __future__ import annotations

import numpy as np

from ft8gpt.ft8pack import pack_standard_payload
from ft8gpt.crc import crc14
from ft8gpt.ldpc_encode import encode174_bits_systematic
from ft8gpt.synth import tones_from_codeword, synthesize_ft8_audio


def make_bits91_from_msg(call_to: str, call_de: str, grid4: str) -> np.ndarray:
    """Return 91-bit payload [77 payload | 14 CRC] as np.uint8 array of 0/1 (MSB-first CRC).

    The 77-bit payload is packed MSB-first into 10 bytes by pack_standard_payload.
    """
    a10 = pack_standard_payload(call_to, call_de, grid4)
    # Unpack MSB-first and take first 77 bits
    bits77 = np.unpackbits(np.frombuffer(a10, dtype=np.uint8), bitorder="big")[:77].astype(np.uint8)
    c = int(crc14(bits77))
    crc_bits = np.array([(c >> i) & 1 for i in range(13, -1, -1)], dtype=np.uint8)
    return np.concatenate([bits77, crc_bits]).astype(np.uint8)


def make_clean_signal(
    call_to: str,
    call_de: str,
    grid4: str,
    sample_rate_hz: float,
    base_freq_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a clean FT8 slot and return (audio, tones).

    - audio: real-valued float32 time series
    - tones: int32 array of length 79 with tone indices 0..7
    """
    a91 = make_bits91_from_msg(call_to, call_de, grid4)
    cw = encode174_bits_systematic(a91)
    tones = tones_from_codeword(cw)
    x = synthesize_ft8_audio(tones, sample_rate_hz, base_freq_hz)
    return x, tones