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
    s = np.sin(phi).astype(np.float64)
    y = x.astype(np.float64) * c  # ignore Hilbert for speed; adequate for small drift
    y = y.astype(np.float32)
    # Re-normalize gently
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 1e-12:
        y /= peak
    return y


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