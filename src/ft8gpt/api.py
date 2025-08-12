from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
import soundfile as sf


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
    # TODO: implement full pipeline: waterfall, sync candidates, LLRs, LDPC, CRC, message unpack
    return []


