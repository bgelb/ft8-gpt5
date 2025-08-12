from __future__ import annotations

import numpy as np

from .constants import SYMBOL_PERIOD_S, NN, FSK_TONES, TONE_SPACING_HZ


def goertzel_magnitudes(
    samples: np.ndarray,
    sample_rate_hz: float,
    start_sample: int,
    num_symbols: int,
    center_freq_hz: float,
) -> np.ndarray:
    """Compute magnitude per symbol and tone using Goertzel at FT8 tone frequencies.

    Returns array of shape [num_symbols, 8] of magnitudes (linear scale).
    """
    N = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    out = np.zeros((num_symbols, FSK_TONES), dtype=np.float64)
    n = np.arange(N)
    for s in range(num_symbols):
        i0 = start_sample + s * N
        seg = samples[i0 : i0 + N]
        if seg.size < N:
            seg = np.pad(seg, (0, N - seg.size))
        # Optionally apply a window (Hann) to control sidelobes
        # win = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
        # x = seg * win
        x = seg
        for k in range(FSK_TONES):
            f = center_freq_hz + k * TONE_SPACING_HZ
            w = np.exp(-1j * 2 * np.pi * f / sample_rate_hz)
            s_prev = 0.0 + 0.0j
            s_prev2 = 0.0 + 0.0j
            coeff = 2 * np.cos(2 * np.pi * f / sample_rate_hz)
            for xi in x:
                s = xi + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            # Power approx
            power = (s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2).real
            out[s, k] = max(power, 0.0)
    return out


