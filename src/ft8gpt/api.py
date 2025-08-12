from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import soundfile as sf
from pathlib import Path

from .decoder_e2e import decode_block

@dataclass(frozen=True)
class DecodeResult:
    start_time_s: float
    frequency_hz: float
    snr_db: float
    message: str
    crc14_ok: bool


def decode_wav(path: str) -> List[DecodeResult]:
    """
    Decode all FT8 signals in a 15-second WAV file.

    Minimal placeholder that just loads audio and returns no decodes yet.
    """
    samples, sample_rate_hz = sf.read(path, always_2d=False)
    x = samples[:, 0] if getattr(samples, "ndim", 1) > 1 else samples
    x = np.asarray(x, dtype=np.float64)
    # Run a basic pipeline using reference parity table
    parity = Path(__file__).resolve().parents[2] / "external" / "ft8_lib" / "ft4_ft8_public" / "parity.dat"
    decs = decode_block(x, float(sample_rate_hz), parity)
    results: List[DecodeResult] = []
    for d in decs:
        results.append(
            DecodeResult(
                start_time_s=0.0, frequency_hz=0.0, snr_db=0.0, message="", crc14_ok=True
            )
        )
    return results


