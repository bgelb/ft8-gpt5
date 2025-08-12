from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window

from .constants import SYMBOL_PERIOD_S, NN, FSK_TONES


@dataclass(frozen=True)
class Waterfall:
    # magnitudes per [symbol, base_bin, tone]
    mag: NDArray[np.float64]  # shape [num_symbols, num_bases, 8]
    sample_rate_hz: float
    n_fft: int
    base_bin0_hz: float


def compute_waterfall_symbols(signal: NDArray[np.float64], sample_rate_hz: float, start_sample: int,
                               num_symbols: int = NN) -> Waterfall:
    """
    Compute per-symbol FFT magnitudes at tone-aligned bin grid and arrange 8-tone groups per base bin.

    - Uses window length = round(fs * SYMBOL_PERIOD_S)
    - Frequency resolution equals tone spacing when fs is 12000 Hz
    """
    n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    if start_sample + num_symbols * n_fft > signal.size:
        num_symbols = max(0, (signal.size - start_sample) // n_fft)
    win = get_window("hann", n_fft, fftbins=True).astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))  # energy-normalized

    # Compute spectra for each symbol
    spectra = np.empty((num_symbols, n_fft // 2 + 1), dtype=np.complex128)
    for s in range(num_symbols):
        seg = signal[start_sample + s * n_fft: start_sample + (s + 1) * n_fft]
        if seg.size < n_fft:
            seg = np.pad(seg, (0, n_fft - seg.size))
        xw = seg * win
        spectra[s, :] = np.fft.rfft(xw)

    mags = np.abs(spectra).astype(np.float64)
    mags = np.maximum(mags, 1e-12)
    # Convert to log magnitude for robustness
    mags_db = 20.0 * np.log10(mags)

    # Construct base bins (k0..k0+7 must be in range)
    num_bins = mags_db.shape[1]
    num_bases = max(0, num_bins - FSK_TONES)
    wf = np.empty((num_symbols, num_bases, FSK_TONES), dtype=np.float64)
    for k0 in range(num_bases):
        wf[:, k0, :] = mags_db[:, k0:k0 + FSK_TONES]

    base_bin0_hz = 0.0  # rfft bin 0
    return Waterfall(mag=wf, sample_rate_hz=sample_rate_hz, n_fft=n_fft, base_bin0_hz=base_bin0_hz)


