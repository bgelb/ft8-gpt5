import numpy as np

from ft8gpt.constants import gray_to_bits
from ft8gpt.tones import extract_symbol_llrs_gray_lse


def test_llr_signs_match_gray_bits_peak_tones():
	# For a strong single tone, LLR signs must match (b2,b1,b0)
	for tone in range(8):
		mag = np.zeros(8, dtype=np.float64)
		mag[tone] = 10.0
		l2, l1, l0 = extract_symbol_llrs_gray_lse(mag)
		b2, b1, b0 = gray_to_bits(tone)
		assert (l2 > 0) if b2 == 0 else (l2 < 0), f"bit2 sign mismatch for tone {tone}"
		assert (l1 > 0) if b1 == 0 else (l1 < 0), f"bit1 sign mismatch for tone {tone}"
		assert (l0 > 0) if b0 == 0 else (l0 < 0), f"bit0 sign mismatch for tone {tone}"


def test_llr_robust_to_neighbor_leakage():
	# With adjacent leakage, signs should remain consistent
	for tone in range(8):
		mag = np.zeros(8, dtype=np.float64)
		mag[tone] = 10.0
		if tone - 1 >= 0:
			mag[tone - 1] = 2.0
		if tone + 1 < 8:
			mag[tone + 1] = 2.0
		l2, l1, l0 = extract_symbol_llrs_gray_lse(mag)
		b2, b1, b0 = gray_to_bits(tone)
		assert (l2 > 0) if b2 == 0 else (l2 < 0), f"[nbr] bit2 sign mismatch for tone {tone}"
		assert (l1 > 0) if b1 == 0 else (l1 < 0), f"[nbr] bit1 sign mismatch for tone {tone}"
		assert (l0 > 0) if b0 == 0 else (l0 < 0), f"[nbr] bit0 sign mismatch for tone {tone}"