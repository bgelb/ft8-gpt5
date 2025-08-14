from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import LENGTH_SYNC, NUM_SYNC, SYNC_OFFSET, FT8_COSTAS_PATTERN, FSK_TONES, SYMBOL_PERIOD_S
from scipy.signal import get_window


"""
Legacy waterfall-based sync score routines were removed in favor of a robust STFT
Costas matched-filter candidate search. This module now exposes only the STFT-based
search entry point and supporting structures.
"""


@dataclass(frozen=True)
class StftCandidate:
    # Index of the STFT frame that aligns with the first Costas symbol
    frame_start: int
    # Integer base bin index (k0) of the 8-tone group
    base_bin: int
    # Fractional bin offset in multiples of tone/bin spacing (supports 0.0 or 0.5 currently)
    frac: float
    # Matched filter score
    score: float


def _compute_stft_magnitude_db(signal: NDArray[np.float64], sample_rate_hz: float) -> Tuple[NDArray[np.float64], int, int]:
    """
    Compute a tapered STFT magnitude spectrogram in dB with Hann window and hop ≈ T/2.

    Returns (mags_db, n_fft, hop) where mags_db has shape [num_frames, n_fft//2+1].
    """
    n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    n_fft = max(16, n_fft)
    hop = max(1, n_fft // 2)
    win = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))

    n = signal.size
    if n < n_fft:
        # zero-pad to at least one frame
        pad = np.pad(signal, (0, n_fft - n))
        n = pad.size
        x = pad
    else:
        x = signal

    num_frames = 1 + max(0, (n - n_fft) // hop)
    mags = np.empty((num_frames, n_fft // 2 + 1), dtype=np.float64)
    for i in range(num_frames):
        i0 = i * hop
        seg = x[i0 : i0 + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        sw = seg * win
        spec = np.fft.rfft(sw)
        m = np.abs(spec).astype(np.float64)
        # Simple per-frame noise normalization for better contrast
        m = np.maximum(m, 1e-12)
        med = np.median(m)
        m = np.maximum(m - med, 1e-12)
        mags[i] = 20.0 * np.log10(m)

    return mags, n_fft, hop


def find_sync_candidates_stft(signal: NDArray[np.float64], sample_rate_hz: float, top_k: int = 300) -> Tuple[List[StftCandidate], int, int]:
    """
    Perform a Costas matched-filter search over time (via STFT frames) and fractional
    frequency offsets (0.0 and +0.5 bin) to produce a ranked list of candidate starts.

    Returns (candidates, n_fft, hop).
    """
    mags_db, n_fft, hop = _compute_stft_magnitude_db(signal, sample_rate_hz)
    num_frames, num_bins = mags_db.shape

    # Prepare linear power for fractional-bin interpolation
    pwr = 10.0 ** (mags_db / 10.0)

    # Integer-aligned 8-tone groups
    num_bases = max(0, num_bins - FSK_TONES)
    # +0.5 fractional alignment groups
    num_bases_half = max(0, num_bins - (FSK_TONES + 1))

    # Precompute per-frame tone-bank slices for 0.0 and +0.5 fractional offsets
    bank0 = np.empty((num_frames, num_bases, FSK_TONES), dtype=np.float64)
    for k0 in range(num_bases):
        bank0[:, k0, :] = pwr[:, k0:k0 + FSK_TONES]

    bankh = np.empty((num_frames, num_bases_half, FSK_TONES), dtype=np.float64)
    for k0 in range(num_bases_half):
        interp = 0.5 * (pwr[:, k0:k0 + FSK_TONES] + pwr[:, k0 + 1:k0 + 1 + FSK_TONES])
        bankh[:, k0, :] = interp

    # Matched filter search across frame times aligned to symbols: 2 frames per symbol (hop = T/2)
    frames_per_symbol = max(1, int(round((SYMBOL_PERIOD_S * sample_rate_hz) / hop)))
    # For a true T/2 hop this should be 2; guard anyway
    frames_per_symbol = max(frames_per_symbol, 1)

    window_len_symbols = LENGTH_SYNC + (NUM_SYNC - 1) * SYNC_OFFSET
    max_start_frame = num_frames - (window_len_symbols - 1) * frames_per_symbol
    if max_start_frame <= 0:
        return ([], n_fft, hop)

    candidates: List[StftCandidate] = []

    def score_at(bank: NDArray[np.float64], base_limit: int, frac_val: float) -> None:
        for t0 in range(0, max_start_frame):
            # Times for the three sync blocks and their 7 symbols
            score_row = 0.0
            count = 0
            for base_idx in range(base_limit):
                # Accumulate Costas energies across three blocks
                score = 0.0
                cnt = 0
                for m in range(NUM_SYNC):
                    base_frame = t0 + (SYNC_OFFSET * m) * frames_per_symbol
                    for k in range(LENGTH_SYNC):
                        f = base_frame + k * frames_per_symbol
                        if f < 0 or f >= num_frames:
                            continue
                        tone = FT8_COSTAS_PATTERN[k]
                        score += float(bank[f, base_idx, tone])
                        cnt += 1
                if cnt > 0:
                    # Use linear power already; keep score linear
                    candidates.append(StftCandidate(frame_start=t0, base_bin=base_idx, frac=frac_val, score=score / cnt))

    # Integer bins
    if num_bases > 0:
        score_at(bank0, num_bases, 0.0)
    # Half-bin bins
    if num_bases_half > 0:
        score_at(bankh, num_bases_half, 0.5)

    candidates.sort(key=lambda c: c.score, reverse=True)
    if top_k is not None and top_k > 0:
        candidates = candidates[:top_k]
    return (candidates, n_fft, hop)


