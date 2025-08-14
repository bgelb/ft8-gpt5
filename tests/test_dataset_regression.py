import os
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from ft8gpt import decode_wav


DATASET_DIR = Path(__file__).resolve().parent.parent / "external" / "ft8_lib" / "test" / "wav"


@pytest.mark.regression
@pytest.mark.slow
def test_minimal_dataset_available():
    # Ensure dataset exists and has at least a few wavs
    if not DATASET_DIR.exists():
        pytest.skip("dataset not available")
    wavs = sorted([p for p in DATASET_DIR.glob("*.wav")])
    assert len(wavs) > 0


@pytest.mark.regression
@pytest.mark.slow
def test_decode_one_strong_sample_runtime(benchmark):
    # Pick a likely-strong sample file; fall back to skip if not present
    if not DATASET_DIR.exists():
        pytest.skip("dataset not available")
    candidates = [p for p in DATASET_DIR.glob("*.wav")]
    if not candidates:
        pytest.skip("no wav files found")
    wav_path = str(candidates[0])

    def run_decode():
        return decode_wav(wav_path)

    results = benchmark(run_decode)
    assert isinstance(results, list)
    # Not asserting decode counts yet; baseline ensures non-regression in crash/runtimes


