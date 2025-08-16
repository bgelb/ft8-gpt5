from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import LENGTH_SYNC, NUM_SYNC, SYNC_OFFSET, FT8_COSTAS_PATTERN, FSK_TONES, SYMBOL_PERIOD_S
from scipy.signal import get_window
from numpy.lib.stride_tricks import as_strided


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


def _compute_stft_power_linear(signal: NDArray[np.float64], sample_rate_hz: float) -> Tuple[NDArray[np.float64], int, int]:
    """
    Compute a tapered STFT power spectrogram in linear units with Hann window and hop ≈ T/2.

    Returns (power, n_fft, hop) where power has shape [num_frames, n_fft//2+1].
    """
    n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    n_fft = max(16, n_fft)
    hop = max(1, n_fft // 2)
    win = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    # Normalize window to unit RMS to keep power scales stable across n_fft
    win /= np.sqrt(np.sum(win ** 2))

    x = signal if signal.size >= n_fft else np.pad(signal, (0, n_fft - signal.size))
    n = x.size
    num_frames = 1 + max(0, (n - n_fft) // hop)
    if num_frames <= 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float64), n_fft, hop
    # Build strided frame matrix [num_frames, n_fft]
    frames = as_strided(x, shape=(num_frames, n_fft), strides=(x.strides[0] * hop, x.strides[0]))
    frames_w = frames * win[None, :]
    spec = np.fft.rfft(frames_w, axis=1)
    # Linear power
    p = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)
    # Robust per-frame noise floor removal in linear domain
    med = np.median(p, axis=1, keepdims=True)
    p = np.maximum(p - med, 0.0)

    return p, n_fft, hop


def find_sync_candidates_stft(signal: NDArray[np.float64], sample_rate_hz: float, top_k: int = 300) -> Tuple[List[StftCandidate], int, int]:
    """
    Perform a Costas matched-filter search over time (via STFT frames) and fractional
    frequency offsets (0.0 and +0.5 bin) to produce a ranked list of candidate starts.

    Returns (candidates, n_fft, hop).
    """
    pwr, n_fft, hop = _compute_stft_power_linear(signal, sample_rate_hz)
    num_frames, num_bins = pwr.shape

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

    # Build per-start frame offsets for the 3x7 Costas symbols
    offs = np.array([SYNC_OFFSET * m + k for m in range(NUM_SYNC) for k in range(LENGTH_SYNC)], dtype=np.int64)
    offs *= frames_per_symbol  # in frames
    tone_idx = np.array([FT8_COSTAS_PATTERN[k] for _m in range(NUM_SYNC) for k in range(LENGTH_SYNC)], dtype=np.int64)

    def score_grid(bank: NDArray[np.float64], base_limit: int) -> NDArray[np.float64]:
        if base_limit <= 0:
            return np.zeros((0, 0), dtype=np.float64)
        grid = np.full((max_start_frame, base_limit), -1e30, dtype=np.float64)
        # For each candidate start t0, gather 21 frames and sum expected tones
        for t0 in range(max_start_frame):
            frames = t0 + offs  # shape [21]
            valid = (frames >= 0) & (frames < num_frames)
            if not np.any(valid):
                continue
            frames = np.clip(frames, 0, num_frames - 1)
            # bank[frames] -> [21, base_limit, 8]
            B = bank[frames, :base_limit, :]
            # select expected tones -> [21, base_limit]
            E = np.take_along_axis(B, tone_idx[:, None, None], axis=2)[..., 0]
            mask = valid[:, None].astype(np.float64)
            s = (E * mask).sum(axis=0)
            c = mask.sum(axis=0)
            c = np.maximum(c, 1.0)
            grid[t0, :] = s / c
        return grid

    grid0 = score_grid(bank0, num_bases)
    gridh = score_grid(bankh, num_bases_half)

    def peaks_along_bases(grid: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Detect local maxima along the base-bin axis per time frame.

        A peak at (t, k) satisfies grid[t, k] > grid[t, k-1] and grid[t, k] > grid[t, k+1].
        """
        if grid.size == 0:
            return np.zeros_like(grid, dtype=bool)
        H, W = grid.shape
        if W < 3 or H == 0:
            return np.zeros_like(grid, dtype=bool)
        left = grid[:, :-2]
        center = grid[:, 1:-1]
        right = grid[:, 2:]
        mask_mid = (center > left) & (center > right)
        peaks = np.zeros_like(grid, dtype=bool)
        peaks[:, 1:-1] = mask_mid
        return peaks

    cand_list: List[StftCandidate] = []
    if grid0.size:
        peaks0 = peaks_along_bases(grid0)
        ys, xs = np.nonzero(peaks0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cand_list.append(StftCandidate(frame_start=int(y), base_bin=int(x), frac=0.0, score=float(grid0[y, x])))
    if gridh.size:
        peaksh = peaks_along_bases(gridh)
        ys, xs = np.nonzero(peaksh)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cand_list.append(StftCandidate(frame_start=int(y), base_bin=int(x), frac=0.5, score=float(gridh[y, x])))

    # Fallback: if no peaks are found (e.g., flat spectra), take best-scoring candidates directly
    if not cand_list:
        items: List[Tuple[float, int, int, float]] = []  # (score, y, x, frac)
        if grid0.size:
            # Exclude edge bases to mimic peak constraint
            g = grid0.copy()
            if g.shape[1] >= 2:
                g[:, 0] = -1e30
                g[:, -1] = -1e30
            ys, xs = np.unravel_index(np.argsort(g, axis=None)[-max(1, (top_k or 50)) :], g.shape)
            for s, y, x in zip(g[ys, xs], ys, xs):
                items.append((float(s), int(y), int(x), 0.0))
        if gridh.size:
            gh = gridh.copy()
            if gh.shape[1] >= 2:
                gh[:, 0] = -1e30
                gh[:, -1] = -1e30
            ys, xs = np.unravel_index(np.argsort(gh, axis=None)[-max(1, (top_k or 50)) :], gh.shape)
            for s, y, x in zip(gh[ys, xs], ys, xs):
                items.append((float(s), int(y), int(x), 0.5))
        items.sort(key=lambda t: t[0], reverse=True)
        for s, y, x, frac in items[: (top_k or 50)]:
            cand_list.append(StftCandidate(frame_start=y, base_bin=x, frac=frac, score=s))

    cand_list.sort(key=lambda c: c.score, reverse=True)
    if top_k is not None and top_k > 0:
        cand_list = cand_list[:top_k]
    return (cand_list, n_fft, hop)


