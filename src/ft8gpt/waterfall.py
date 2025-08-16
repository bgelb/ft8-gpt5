from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.signal import get_window

from .constants import SYMBOL_PERIOD_S, NN, FSK_TONES


@dataclass(frozen=True)
class Waterfall:
    # magnitudes per [symbol, base_bin, tone]
    mag: NDArray[np.float64]       # shape [num_symbols, num_bases, 8] at integer bin alignment
    mag_half: NDArray[np.float64]  # shape [num_symbols, num_bases-1, 8] at +0.5 bin alignment (approx)
    sample_rate_hz: float
    n_fft: int
    base_bin0_hz: float


def compute_waterfall_symbols(signal: NDArray[np.float64], sample_rate_hz: float, start_sample: int,
                               num_symbols: int = NN) -> Waterfall:
    """
    Compute per-symbol FFT magnitudes at tone-aligned bin grid and arrange 8-tone groups per base bin.

    - Window length n_fft = round(sample_rate_hz * SYMBOL_PERIOD_S)
    - Frequency resolution equals tone spacing when sample_rate_hz is 12000 Hz
    - Returns magnitude in dB with per-symbol median subtracted (simple noise normalization)
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
    # Apply simple noise floor normalization per symbol for better discrimination
    med = np.median(mags, axis=1, keepdims=True)
    mags = np.maximum(mags - med, 1e-12)
    mags_db = 20.0 * np.log10(mags)

    # Construct base bins (k0..k0+7 must be in range)
    num_bins = mags_db.shape[1]
    num_bases = max(0, num_bins - FSK_TONES)
    wf = np.empty((num_symbols, num_bases, FSK_TONES), dtype=np.float64)
    for k0 in range(num_bases):
        wf[:, k0, :] = mags_db[:, k0:k0 + FSK_TONES]

    # Approximate +0.5-bin alignment using adjacent-bin averaging in linear power domain before converting back to dB
    # Convert mags_db back to linear power for interpolation
    pwr = 10.0 ** (mags_db / 10.0)
    num_bases_half = max(0, num_bins - (FSK_TONES + 1))
    wf_half = np.empty((num_symbols, num_bases_half, FSK_TONES), dtype=np.float64)
    for k0 in range(num_bases_half):
        # half-step group spans bins [k0+0.5 ... k0+7.5]
        interp = 0.5 * (pwr[:, k0:k0 + FSK_TONES] + pwr[:, k0 + 1:k0 + 1 + FSK_TONES])
        wf_half[:, k0, :] = 10.0 * np.log10(np.maximum(interp, 1e-20))

    base_bin0_hz = 0.0  # rfft bin 0
    return Waterfall(mag=wf, mag_half=wf_half, sample_rate_hz=sample_rate_hz, n_fft=n_fft, base_bin0_hz=base_bin0_hz)


