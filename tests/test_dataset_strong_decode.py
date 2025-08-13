from pathlib import Path
import pytest

from ft8gpt import decode_wav


DATASET_DIR = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"


def test_dataset_has_at_least_one_decode():
    assert DATASET_DIR.exists(), "Required dataset directory is missing: external/ft8_lib/test/wav (ensure submodules are checked out)"
    preferred = DATASET_DIR / "websdr_test1.wav"
    if preferred.exists():
        wav_path = preferred
    else:
        candidates = sorted(DATASET_DIR.glob("*.wav"))
        assert candidates, "No WAV files found in dataset directory"
        wav_path = candidates[0]

    results = decode_wav(str(wav_path))
    assert isinstance(results, list)
    assert len(results) > 0