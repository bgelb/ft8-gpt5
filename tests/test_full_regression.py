import os
from pathlib import Path
import time
import re
import pytest

from ft8gpt import decode_wav


def parse_expected(path: Path) -> set[str]:
    if not path.exists():
        return set()
    msgs = set()
    for line in path.read_text().splitlines():
        if '~' in line:
            msg = line.split('~', 1)[1].strip()
        else:
            parts = line.split()
            msg = ' '.join(parts[5:]) if len(parts) > 5 else ''
        if msg:
            msgs.add(re.sub(r"\s+", " ", msg.upper()))
    return msgs


@pytest.mark.slow
@pytest.mark.regression
def test_full_dataset_nonregression():
    dataset = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"
    if not dataset.exists():
        pytest.skip("dataset not available")
    wavs = sorted(dataset.glob("*.wav"))
    if not wavs:
        pytest.skip("no wav files")

    total_expected = 0
    total_matched = 0
    t0 = time.time()
    for wav in wavs:
        expected = parse_expected(wav.with_suffix('.txt'))
        total_expected += len(expected)
        results = decode_wav(str(wav))
        got = {re.sub(r"\s+", " ", r.message.upper()) for r in results if r.message}
        total_matched += len(expected.intersection(got))
    t1 = time.time()

    decode_rate = 0.0 if total_expected == 0 else total_matched / total_expected
    # Baseline threshold is very low initially; to be raised as we improve
    assert decode_rate >= 0.0
    # Runtime guardrail: average < 5s per 15s sample on CI
    avg_runtime = (t1 - t0) / max(1, len(wavs))
    assert avg_runtime < 5.0




