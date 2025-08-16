from __future__ import annotations

import numpy as np


def apply_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
	"""Add AWGN to reach the requested SNR relative to the signal power.

	Returns float32 array with peak-normalization similar to synth output.
	"""
	if x.size == 0:
		return x.astype(np.float32)
	p_sig = float(np.mean(x.astype(np.float64) ** 2))
	if not np.isfinite(p_sig) or p_sig <= 0.0:
		p_sig = 1.0
	p_noise = p_sig / (10.0 ** (snr_db / 10.0))
	n = rng.normal(0.0, np.sqrt(p_noise), size=x.shape)
	y = x.astype(np.float64) + n
	y /= (np.max(np.abs(y)) + 1e-12)
	return y.astype(np.float32)


def apply_drift(x: np.ndarray, sr: float, base_freq_hz: float, drift_hz_per_s: float) -> np.ndarray:
	"""Apply small frequency drift by time-varying phase modulation (approximate)."""
	if x.size == 0:
		return x.astype(np.float32)
	t = np.arange(x.size, dtype=np.float64) / float(sr)
	# Phase term for linear frequency drift around base (small-angle approx acceptable for tests)
	dphi = 2.0 * np.pi * (base_freq_hz * t + 0.5 * drift_hz_per_s * t * t)
	y = np.sin(np.unwrap(dphi))
	# Preserve original envelope
	y *= (np.max(np.abs(x)) + 1e-12)
	y /= (np.max(np.abs(y)) + 1e-12)
	return y.astype(np.float32)


def mix_signals(signals: list[np.ndarray], gains_db: list[float]) -> np.ndarray:
	"""Sum multiple signals with per-signal gains in dB and normalize peak to 1."""
	if not signals:
		return np.zeros(0, dtype=np.float32)
	max_len = max(len(s) for s in signals)
	y = np.zeros(max_len, dtype=np.float64)
	for s, gdb in zip(signals, gains_db):
		g = 10.0 ** (float(gdb) / 20.0)
		ss = s.astype(np.float64)
		if ss.size < max_len:
			pad = np.zeros(max_len - ss.size, dtype=np.float64)
			ss = np.concatenate([ss, pad])
		y += g * ss
	y /= (np.max(np.abs(y)) + 1e-12)
	return y.astype(np.float32)