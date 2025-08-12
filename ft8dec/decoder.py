from __future__ import annotations
from typing import List
import numpy as np
from .types import Decode
from .io import read_wav_mono


def decode_wav_file(path: str) -> List[Decode]:
    """Decode an FT8 WAV file and return decodes.

    Initial implementation is a stub that returns no decodes.
    """
    samples, fs_hz = read_wav_mono(path)
    _ = (samples, fs_hz)
    return []
