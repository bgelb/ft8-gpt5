from __future__ import annotations

import numpy as np


def apply_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
	"""Add AWGN to achieve target SNR in dB relative to signal power.

	Returns float32 array, optionally peak-normalized to ~1.
	"""
	xp = x.astype(np.float64, copy=False)
	p_sig = float(np.mean(xp * xp)) if xp.size else 1.0
	if not np.isfinite(p_sig) or p_sig <= 0.0:
		p_sig = 1.0
	p_noise = p_sig / (10.0 ** (snr_db / 10.0))
	n = rng.normal(0.0, np.sqrt(p_noise), size=xp.shape)
	y = xp + n
	# Normalize to avoid clipping differences across mixes
	peak = float(np.max(np.abs(y))) if y.size else 1.0
	if not np.isfinite(peak) or peak <= 1e-12:
		peak = 1.0
	y = (y / peak).astype(np.float32)
	return y


def apply_drift(x: np.ndarray, sample_rate_hz: float, drift_hz_per_s: float) -> np.ndarray:
	"""Apply small linear frequency drift via phase modulation approximation.

	For characterization only; for accurate simulation, re-synthesize per-symbol.
	"""
	n = np.arange(x.size, dtype=np.float64)
	t = n / float(sample_rate_hz)
	# Quadratic phase: φ(t) = π * drift * t^2 yields f(t) ≈ drift * t
	phi = np.pi * drift_hz_per_s * (t * t)
	c = np.cos(phi).astype(np.float64)
	y = x.astype(np.float64) * c  # ignore Hilbert for speed; adequate for small drift
	y = y.astype(np.float32)
	# Re-normalize gently
	peak = float(np.max(np.abs(y))) if y.size else 1.0
	if peak > 1e-12:
		y /= peak
	return y


def apply_sample_rate_error(x: np.ndarray, ppm: float) -> np.ndarray:
	"""Apply sample-rate error by re-sampling to emulate clock skew.

	ppm: parts-per-million; positive ppm compresses time (faster clock).
	"""
	if x.size == 0 or abs(ppm) < 1e-6:
		return x.astype(np.float32, copy=False)
	scale = 1.0 + float(ppm) * 1e-6
	n_old = int(x.size)
	n_new = max(1, int(round(n_old / scale)))
	t_old = np.linspace(0.0, 1.0, n_old, endpoint=False)
	t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
	y = np.interp(t_new, t_old, x.astype(np.float64))
	# Normalize peak
	peak = float(np.max(np.abs(y))) if y.size else 1.0
	if peak > 1e-12:
		y /= peak
	return y.astype(np.float32)


def apply_qsb_fading(x: np.ndarray, sample_rate_hz: float, f_hz: float, rician_k: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
	"""Apply slow Rayleigh/Rician fading envelope.

	f_hz: Doppler-like envelope frequency (0.1..2 Hz typical)
	rician_k: if provided, mixes a LOS component (K-factor); else Rayleigh.
	"""
	if rng is None:
		rng = np.random.default_rng(0)
	n = np.arange(x.size, dtype=np.float64) / float(sample_rate_hz)
	phi = 2.0 * np.pi * f_hz * n
	# Sum of sinusoids for envelope randomness
	env = (np.sin(phi) + 0.5 * np.sin(2 * phi + 0.7) + 0.25 * np.sin(3 * phi + 1.3))
	env = (env - np.mean(env)) / (np.std(env) + 1e-12)
	rayleigh = np.sqrt(0.5 * (rng.standard_normal(x.size) ** 2 + rng.standard_normal(x.size) ** 2))
	# Lowpass the Rayleigh noise by moving average to get slow fading
	win = max(1, int(round(sample_rate_hz * 0.2)))
	kernel = np.ones(win, dtype=np.float64) / float(win)
	rayleigh_slow = np.convolve(rayleigh, kernel, mode='same')
	rayleigh_slow = (rayleigh_slow - np.mean(rayleigh_slow)) / (np.std(rayleigh_slow) + 1e-12)
	# Combine deterministic and random to form envelope in [0,1]
	e = 0.6 + 0.2 * env + 0.2 * rayleigh_slow
	if rician_k is not None:
		los = np.sqrt(rician_k / (rician_k + 1.0))
		rnd = np.sqrt(1.0 / (rician_k + 1.0)) * e
		e = np.clip(los + rnd, 0.05, 2.0)
	y = x.astype(np.float64) * e
	# Normalize rms
	rms = float(np.sqrt(np.mean(y * y))) if y.size else 1.0
	if rms > 1e-12:
		y /= rms
	return y.astype(np.float32)


def apply_impulse_noise(x: np.ndarray, sample_rate_hz: float, burst_snr_db: float, burst_prob: float, rng: np.random.Generator) -> np.ndarray:
	"""Inject random impulse noise bursts.

	burst_snr_db: relative to signal peak; larger => stronger impulses
	burst_prob: probability per symbol-length to insert a burst
	"""
	y = x.astype(np.float64).copy()
	n = y.size
	if n == 0:
		return x.astype(np.float32)
	# Determine symbol length approximately at 6.25 baud
	sym_len = max(1, int(round(sample_rate_hz / 6.25)))
	k = 0
	while k < n:
		if rng.random() < burst_prob:
			L = int(rng.integers(low=sym_len // 8, high=sym_len))
			idx = int(rng.integers(low=k, high=min(n, k + 4 * sym_len)))
			amp = (10.0 ** (burst_snr_db / 20.0)) * (2.0 * rng.random() - 1.0)
			y[idx: idx + L] += amp
		k += sym_len
	# Normalize peak
	peak = float(np.max(np.abs(y)))
	if peak > 1e-12:
		y /= peak
	return y.astype(np.float32)


def mix_signals(signals: list[np.ndarray], gains_db: list[float]) -> np.ndarray:
	"""Mix multiple time-aligned signals with per-signal gain in dB.

	Shorter signals are zero-padded implicitly by slicing.
	"""
	if not signals:
		return np.zeros(0, dtype=np.float32)
	max_len = max(int(s.size) for s in signals)
	acc = np.zeros(max_len, dtype=np.float64)
	for s, gdb in zip(signals, gains_db):
		g = 10.0 ** (float(gdb) / 20.0)
		L = int(s.size)
		if L == 0:
			continue
		acc[:L] += g * s.astype(np.float64)
	peak = float(np.max(np.abs(acc))) if acc.size else 1.0
	if not np.isfinite(peak) or peak <= 1e-12:
		peak = 1.0
	y = (acc / peak).astype(np.float32)
	return y