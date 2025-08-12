from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from .constants import SYMBOL_PERIOD_S


@dataclass(frozen=True)
class Signal:
    samples: NDArray[np.float64]
    sample_rate_hz: float


def to_mono_float64(samples: NDArray[np.float_]) -> NDArray[np.float64]:
    if samples.ndim == 1:
        x = samples
    else:
        x = samples[:, 0]
    x = np.asarray(x, dtype=np.float64)
    # Normalize to [-1, 1] if integer type ranges are suspected
    if np.max(np.abs(x)) > 2.0:
        xmax = np.max(np.abs(x))
        if xmax > 0:
            x = x / xmax
    return x


def resample_to_symbol_aligned_rate(signal: Signal, target_osr: int = 2) -> Signal:
    """
    Resample to a rate where symbol boundaries align to integer samples:
    target_rate = target_osr / SYMBOL_PERIOD_S.
    """
    target_rate_hz = target_osr / SYMBOL_PERIOD_S
    # Use rational approximation for polyphase resampling
    from fractions import Fraction

    frac = Fraction(target_rate_hz / signal.sample_rate_hz).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y = resample_poly(signal.samples, up, down).astype(np.float64)
    return Signal(samples=y, sample_rate_hz=signal.sample_rate_hz * up / down)


def frame_symbol_count(duration_s: float) -> int:
    return int(np.round(duration_s / SYMBOL_PERIOD_S))


