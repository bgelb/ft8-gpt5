from __future__ import annotations
import numpy as np
from .ldpc_data import FT8_COSTAS
from .constants import FT8Constants


def stft_waterfall(samples: np.ndarray, fs_hz: int, center_hz: float = 1500.0,
                    time_osr: int = 2, freq_osr: int = 2) -> tuple[np.ndarray, int, int]:
    """Compute a simple waterfall: [time_blocks, num_bins, tones] magnitudes.
    Returns (waterfall, block_stride, num_bins). Here we use a simplified grid matching FT8 symbol period.
    """
    const = FT8Constants()
    sym_len = int(round(const.symbol_period_s * fs_hz))
    nfft = 512
    hop = sym_len // time_osr
    # frequency bins near baseband 0.. ?; we will use 8 bins around center
    win = np.hanning(sym_len)
    num_blocks = max(0, (len(samples) - sym_len) // hop)
    # Precompute FFT bins for tones spaced by tone_spacing_hz
    tone_bins = np.arange(8) * (const.tone_spacing_hz / (fs_hz / nfft))
    base_bin = int(round((center_hz) / (fs_hz / nfft)))
    bins = (base_bin + tone_bins).astype(int)
    mags = np.zeros((num_blocks, 8), dtype=np.float32)
    for b in range(num_blocks):
        s = b * hop
        frame = samples[s:s+sym_len]
        if len(frame) < sym_len:
            break
        spec = np.fft.rfft(win * frame, n=nfft)
        psd = np.abs(spec)
        mags[b, :] = psd[bins[:8]]
    return mags, 1, 8


def costas_sync_score(mags: np.ndarray) -> np.ndarray:
    """Compute a simple sync score over time by sliding 7-symbol Costas pattern."""
    pattern = FT8_COSTAS
    T = mags.shape[0]
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T - 7):
        sub = mags[t:t+7]
        s = 0.0
        for k in range(7):
            sm = pattern[k]
            cur = sub[k, sm]
            neigh = 0.0
            if sm > 0:
                neigh += sub[k, sm-1]
            if sm < 7:
                neigh += sub[k, sm+1]
            s += max(0.0, cur - 0.5 * neigh)
        scores[t] = s / 7.0
    return scores
