from __future__ import annotations
import numpy as np
import soundfile as sf


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Return mono float32 samples in range [-1, 1] and sample rate in Hz."""
    data, fs = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32, copy=False)
    # Normalize if needed
    max_abs = np.max(np.abs(data))
    if max_abs > 1.0:
        data = data / max_abs
    return data, int(fs)
