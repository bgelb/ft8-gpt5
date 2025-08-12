from __future__ import annotations
import numpy as np
from .ldpc_data import FT8_COSTAS
from .constants import FT8Constants


def stft_waterfall(samples: np.ndarray, fs_hz: int, time_osr: int = 2) -> tuple[np.ndarray, int, float, int]:
    """Compute STFT magnitude spectra per FT8-sized time block.

    Returns (psd_blocks [num_blocks, nfft_bins], hop_samples, bin_hz, time_osr).
    """
    const = FT8Constants()
    sym_len = int(round(const.symbol_period_s * fs_hz))
    nfft = sym_len  # exact bin alignment to 6.25 Hz at 12 kHz
    hop = max(1, sym_len // time_osr)
    win = np.hanning(sym_len).astype(np.float32)
    num_blocks = max(0, (len(samples) - sym_len) // hop)
    bin_hz = fs_hz / nfft
    psd_blocks = np.zeros((num_blocks, nfft // 2 + 1), dtype=np.float32)
    for b in range(num_blocks):
        s = b * hop
        frame = samples[s:s+sym_len]
        if len(frame) < sym_len:
            break
        spec = np.fft.rfft(win * frame, n=nfft)
        psd_blocks[b, :] = np.abs(spec)
    return psd_blocks, hop, float(bin_hz), int(max(1, time_osr))


def extract_8tone_mags(psd_row: np.ndarray, base_bin: int, bin_step: int) -> np.ndarray:
    """Slice 8-bin magnitudes for tones at indices base_bin + j*bin_step."""
    idx = base_bin + (np.arange(8, dtype=int) * bin_step)
    if np.any((idx < 0) | (idx >= len(psd_row))):
        return None  # out of bounds
    return psd_row[idx]


def costas_sync_score(psd: np.ndarray, bin_step: int, base_bin: int) -> np.ndarray:
    """Compute a sync score over time by sliding 7-symbol Costas pattern at a specific base_bin."""
    pattern = FT8_COSTAS
    T = psd.shape[0]
    scores = np.zeros(T, dtype=np.float32)
    for t in range(T - 7):
        s = 0.0
        for k in range(7):
            mags = extract_8tone_mags(psd[t + k], base_bin, bin_step)
            if mags is None:
                continue
            sm = pattern[k]
            cur = mags[sm]
            neigh = 0.0
            if sm > 0:
                neigh += mags[sm-1]
            if sm < 7:
                neigh += mags[sm+1]
            s += max(0.0, cur - 0.5 * neigh)
        scores[t] = s / 7.0
    return scores
