from __future__ import annotations

import numpy as np


def rmse(errors: np.ndarray) -> float:
	"""Root-mean-square of an array of errors."""
	errors = np.asarray(errors, dtype=np.float64)
	if errors.size == 0:
		return 0.0
	return float(np.sqrt(np.mean(errors ** 2)))


def bmi_from_llrs(bits: np.ndarray, llrs: np.ndarray) -> float:
	"""Bit Mutual Information estimate from LLRs and true bits.

	BMI ≈ 1 - mean(log2(1 + exp(-(-1)^{b_i} * L_i)))
	"""
	b = np.asarray(bits, dtype=np.uint8).astype(np.float64)
	L = np.asarray(llrs, dtype=np.float64)
	if b.size == 0 or L.size == 0:
		return 0.0
	m = min(b.size, L.size)
	b = b[:m]
	L = L[:m]
	s = (1.0 - 2.0 * b) * L
	val = np.log2(1.0 + np.exp(-s))
	return float(1.0 - np.mean(val))