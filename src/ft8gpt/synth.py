from __future__ import annotations

from pathlib import Path
import numpy as np
import math

from .constants import SYMBOL_PERIOD_S, NN, LENGTH_SYNC, SYNC_OFFSET, FT8_COSTAS_PATTERN, FT8_GRAY_MAP, FSK_TONES, TONE_SPACING_HZ


def tones_from_codeword(codeword_bits: np.ndarray) -> np.ndarray:
    """Map 174 bits into 79 FT8 tones with Costas sync blocks.
    Data layout: S7 D29 S7 D29 S7, each data symbol maps 3 bits via Gray map.
    """
    tones = np.zeros(NN, dtype=np.int32)
    # Fill Costas
    tones[0:7] = FT8_COSTAS_PATTERN
    tones[36:43] = FT8_COSTAS_PATTERN
    tones[72:79] = FT8_COSTAS_PATTERN
    # Fill data
    i = 0
    for k in range(NN):
        if (0 <= k < 7) or (36 <= k < 43) or (72 <= k < 79):
            continue
        b2 = int(codeword_bits[i]); b1 = int(codeword_bits[i+1]); b0 = int(codeword_bits[i+2])
        idx = (b2 << 2) | (b1 << 1) | b0
        tones[k] = FT8_GRAY_MAP[idx]
        i += 3
        if i >= codeword_bits.shape[0]:
            break
    return tones


def synthesize_ft8_audio(tones: np.ndarray, sample_rate_hz: float, base_freq_hz: float = 1000.0) -> np.ndarray:
    """Synthesize FT8 audio using GFSK-like phase shaping for spectral compliance.

    Produces a float32 mono waveform in [-1, 1] at sample_rate_hz.
    """
    symbol_samples = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    n_total = symbol_samples * NN
    # Precompute Gaussian pulse (BT≈2.0 as in reference)
    bt = 2.0
    K = np.pi * np.sqrt(2.0 / np.log(2.0))
    def gfsk_pulse(idx):
        t = idx / float(symbol_samples) - 1.5
        arg1 = K * bt * (t + 0.5)
        arg2 = K * bt * (t - 0.5)
        return (math.erf(arg1) - math.erf(arg2)) * 0.5

    pulse = np.array([gfsk_pulse(i) for i in range(3 * symbol_samples)], dtype=np.float64)

    # Frequency increment per sample (radians per sample)
    dphi_peak = 2 * np.pi / symbol_samples
    dphi = np.full(n_total + 2 * symbol_samples, 2 * np.pi * base_freq_hz / sample_rate_hz, dtype=np.float64)
    for i in range(NN):
        ib = i * symbol_samples
        dphi[ib:ib + 3 * symbol_samples] += dphi_peak * tones[i] * pulse
    # Extend edges with first and last tones
    dphi[:2 * symbol_samples] += dphi_peak * pulse[symbol_samples:] * tones[0]
    dphi[NN * symbol_samples:NN * symbol_samples + 2 * symbol_samples] += dphi_peak * pulse[:2 * symbol_samples] * tones[-1]

    # Integrate phase and synthesize
    x = np.zeros(n_total, dtype=np.float64)
    phi = 0.0
    for k in range(n_total):
        x[k] = np.sin(phi)
        phi = (phi + dphi[k + symbol_samples]) % (2 * np.pi)

    # Apply gentle ramp at start/end
    ramp = symbol_samples // 8
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(ramp) / (2 * ramp)))
    x[:ramp] *= win
    x[-ramp:] *= win[::-1]

    # Normalize
    x /= np.max(np.abs(x)) + 1e-12
    return x.astype(np.float32)


