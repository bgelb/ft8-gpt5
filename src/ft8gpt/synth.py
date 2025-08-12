from __future__ import annotations

from pathlib import Path
import numpy as np

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
    """Synthesize baseband FT8 audio for a single transmission."""
    symbol_samples = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    n = symbol_samples * NN
    t = np.arange(n) / sample_rate_hz
    x = np.zeros(n, dtype=np.float64)
    for s in range(NN):
        f = base_freq_hz + tones[s] * TONE_SPACING_HZ
        start = s * symbol_samples
        end = start + symbol_samples
        # simple phase-continuous synthesis by resetting phase per symbol is acceptable for strong-signal test
        x[start:end] = np.sin(2 * np.pi * f * t[start:end])
    # Normalize
    x /= np.max(np.abs(x)) + 1e-12
    return x.astype(np.float32)


