from __future__ import annotations

import numpy as np

from ft8gpt.ft8pack import pack_standard_payload
from ft8gpt.crc import crc14
from ft8gpt.ldpc_encode import encode174_bits_systematic
from ft8gpt.synth import tones_from_codeword, synthesize_ft8_audio


def make_bits91_from_msg(call_to: str, call_de: str, grid4: str) -> np.ndarray:
	"""Pack a standard FT8 message into 91 bits (77 payload + 14 CRC), MSB-first."""
	a10 = pack_standard_payload(call_to, call_de, grid4)
	bits77 = np.unpackbits(np.frombuffer(a10, dtype=np.uint8), bitorder="big")[:77]
	c = crc14(bits77)
	crc_bits = np.array([(c >> i) & 1 for i in range(13, -1, -1)], dtype=np.uint8)
	return np.concatenate([bits77.astype(np.uint8), crc_bits])


def make_clean_signal(
	call_to: str,
	call_de: str,
	grid4: str,
	sr: float,
	base_freq_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return (audio, tones) for a clean FT8 frame at given sample rate and base frequency."""
	a91 = make_bits91_from_msg(call_to, call_de, grid4)
	cw = encode174_bits_systematic(a91)
	tones = tones_from_codeword(cw)
	x = synthesize_ft8_audio(tones, sr, base_freq_hz)
	return x.astype(np.float32, copy=False), tones