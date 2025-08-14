from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window

from .constants import (
    LENGTH_SYNC,
    NUM_SYNC,
    SYNC_OFFSET,
    FT8_COSTAS_PATTERN,
    SYMBOL_PERIOD_S,
    FSK_TONES,
)


@dataclass(frozen=True)
class RefinementResult:
    # Fractional bin offset in [0, 1) relative to integer base_bin
    frac_bin: float
    # Timing offset in STFT frames (hop = n_fft//2) relative to coarse alignment
    delta_frames: float
    # Peak coherent correlation magnitude (arbitrary units)
    peak_metric: float


def _compute_stft_complex(signal: NDArray[np.float64], n_fft: int, hop: int) -> NDArray[np.complex128]:
    """Compute complex STFT with a Hann window and given hop.
    Returns array of shape [num_frames, n_fft//2+1].
    """
    win = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))
    n = signal.size
    if n < n_fft:
        x = np.pad(signal, (0, n_fft - n))
    else:
        x = signal
    num_frames = 1 + max(0, (x.size - n_fft) // hop)
    out = np.empty((num_frames, n_fft // 2 + 1), dtype=np.complex128)
    for i in range(num_frames):
        i0 = i * hop
        seg = x[i0 : i0 + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        sw = seg * win
        out[i] = np.fft.rfft(sw)
    return out


def _interp_complex_bin(spec_row: NDArray[np.complex128], k: int, frac: float) -> complex:
    """Linear complex interpolation between adjacent FFT bins.
    k is the lower bin index; frac in [0,1).
    """
    if frac <= 1e-12:
        return complex(spec_row[k])
    k1 = min(k + 1, spec_row.size - 1)
    return complex((1.0 - frac) * spec_row[k] + frac * spec_row[k1])


def refine_cfo_and_timing(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    start_sample: int,
    start_symbol: int,
    base_bin: int,
    initial_frac: float,
) -> Tuple[RefinementResult, int, int]:
    """Refine fractional CFO (as fractional bin) and timing (in STFT frames) using coherent
    correlation over the three Costas blocks.

    Returns (RefinementResult, n_fft, hop).
    """
    n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    hop = max(1, n_fft // 2)

    # Map coarse per-symbol segmentation to STFT frame index baseline
    t0_sample = start_sample + start_symbol * n_fft
    t0_frame_est = int(round(t0_sample / hop))
    frames_per_symbol = max(1, int(round((SYMBOL_PERIOD_S * sample_rate_hz) / hop)))

    stft = _compute_stft_complex(signal, n_fft=n_fft, hop=hop)
    num_frames, num_bins = stft.shape

    # Build list of base frame indices for the 21 Costas symbols
    base_frames: List[int] = []
    base_bins: List[int] = []
    for m in range(NUM_SYNC):
        block_base = t0_frame_est + (SYNC_OFFSET * m) * frames_per_symbol
        for k in range(LENGTH_SYNC):
            base_frames.append(block_base + k * frames_per_symbol)
            base_bins.append(base_bin + FT8_COSTAS_PATTERN[k])

    # Small search grids
    delta_frame_grid = np.array([-1, 0, +1], dtype=np.int64)
    # Frequency around initial_frac within ±0.5 in 0.25 steps, clipped to [0, 1)
    df_steps = np.array([-0.5, -0.25, 0.0, +0.25, +0.5], dtype=np.float64)
    frac_candidates = []
    for d in df_steps:
        f = initial_frac + d
        # Keep in [0,1). Note: wrapping by +/-1 doesn't change physical tone set; we clamp here
        if f < 0.0 or f >= 1.0:
            continue
        frac_candidates.append(f)
    if not frac_candidates:
        frac_candidates = [max(0.0, min(1.0 - 1e-9, initial_frac))]
    frac_candidates = np.array(frac_candidates, dtype=np.float64)

    # Evaluate coherent correlation grid
    grid = np.zeros((delta_frame_grid.size, frac_candidates.size), dtype=np.float64)

    for ti, dt in enumerate(delta_frame_grid):
        # Collect complex tones for this time offset
        corr_sum = np.complex128(0.0)
        # We will accumulate per-frequency candidate separately to reuse bin fetches
        for fi, frac in enumerate(frac_candidates):
            cs = np.complex128(0.0)
            valid = True
            for idx in range(len(base_frames)):
                fidx = base_frames[idx] + int(dt)
                if fidx < 0 or fidx >= num_frames:
                    valid = False
                    break
                kbin = base_bins[idx]
                if kbin < 0 or kbin + 1 >= num_bins:
                    valid = False
                    break
                val = _interp_complex_bin(stft[fidx], kbin, frac)
                cs += val
            if not valid:
                grid[ti, fi] = -1e30
            else:
                grid[ti, fi] = float(np.abs(cs))

    # Find coarse maximum
    flat_idx = int(np.argmax(grid))
    ti_coarse, fi_coarse = np.unravel_index(flat_idx, grid.shape)
    peak = float(grid[ti_coarse, fi_coarse])

    # Parabolic interpolation along frequency axis (if interior)
    frac_refined = float(frac_candidates[fi_coarse])
    if 0 < fi_coarse < frac_candidates.size - 1:
        y_m = grid[ti_coarse, fi_coarse - 1]
        y_0 = grid[ti_coarse, fi_coarse]
        y_p = grid[ti_coarse, fi_coarse + 1]
        denom = (y_m - 2.0 * y_0 + y_p)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_m - y_p) / denom
            step = float(frac_candidates[fi_coarse] - frac_candidates[fi_coarse - 1])
            frac_refined = float(frac_candidates[fi_coarse] + np.clip(delta, -1.0, 1.0) * step)

    # Parabolic interpolation along time axis (if interior)
    dt_refined = float(delta_frame_grid[ti_coarse])
    if 0 < ti_coarse < delta_frame_grid.size - 1:
        y_m = grid[ti_coarse - 1, fi_coarse]
        y_0 = grid[ti_coarse, fi_coarse]
        y_p = grid[ti_coarse + 1, fi_coarse]
        denom = (y_m - 2.0 * y_0 + y_p)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_m - y_p) / denom
            dt_refined = float(delta_frame_grid[ti_coarse] + np.clip(delta, -1.0, 1.0))

    # Clamp refined fractional bin to [0,1)
    frac_refined = float(min(max(frac_refined, 0.0), 1.0 - 1e-9))

    return RefinementResult(frac_bin=frac_refined, delta_frames=dt_refined, peak_metric=peak), n_fft, hop


def extract_derotated_symbol_magnitudes(
    signal: NDArray[np.float64],
    sample_rate_hz: float,
    start_sample: int,
    symbol_indices: List[int],
    base_bin: int,
    frac_bin: float,
    n_fft: int,
    delta_frames: float,
) -> NDArray[np.float64]:
    """Compute per-symbol 8-tone magnitudes (in dB, with simple per-symbol median noise normalization)
    using a coherent complex correlator at fractional-bin frequency and small timing offset.

    Returns an array of shape [len(symbol_indices), 8].
    """
    hop = max(1, n_fft // 2)
    delta_samples = int(round(delta_frames * hop))

    win = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))

    # Precompute time vector for complex exponential
    n = np.arange(n_fft, dtype=np.float64)
    t = n / float(sample_rate_hz)
    bin_hz = sample_rate_hz / float(n_fft)

    out = np.empty((len(symbol_indices), FSK_TONES), dtype=np.float64)

    for i, sym in enumerate(symbol_indices):
        s0 = start_sample + sym * n_fft + delta_samples
        seg = signal[s0 : s0 + n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        xw = seg * win

        mags = np.empty(FSK_TONES, dtype=np.float64)
        for k in range(FSK_TONES):
            freq_hz = (base_bin + k + frac_bin) * bin_hz
            phi = np.exp(-1j * 2.0 * np.pi * freq_hz * t)
            val = np.sum(xw * phi)
            mags[k] = float(np.abs(val))

        # Simple per-symbol noise normalization and convert to dB
        mags = np.maximum(mags, 1e-12)
        med = float(np.median(mags))
        mags = np.maximum(mags - med, 1e-12)
        out[i, :] = 20.0 * np.log10(mags)

    return out