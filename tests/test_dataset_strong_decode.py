from pathlib import Path
import pytest

from ft8gpt import decode_wav


DATASET_DIR = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"


def test_dataset_has_at_least_one_decode():
    if not DATASET_DIR.exists():
        pytest.skip("dataset not available")
    preferred = DATASET_DIR / "websdr_test1.wav"
    if preferred.exists():
        wav_path = preferred
    else:
        candidates = sorted(DATASET_DIR.glob("*.wav"))
        if not candidates:
            pytest.skip("no wav files found")
        wav_path = candidates[0]

    results = decode_wav(str(wav_path))
    assert isinstance(results, list)
    assert len(results) > 0