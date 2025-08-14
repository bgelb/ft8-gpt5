from pathlib import Path
import re
import numpy as np
import pytest

from ft8gpt import decode_wav


def parse_wsjt_lines(path: Path) -> set[str]:
    lines = []
    if not path.exists():
        return set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: time snr dt freq ~  MESSAGE
        # Extract the trailing message
        if '~' in line:
            msg = line.split('~', 1)[1].strip()
        else:
            parts = line.split()
            msg = ' '.join(parts[5:]) if len(parts) > 5 else ''
        if msg:
            lines.append(msg)
    return set(lines)


def normalize_msg(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


@pytest.mark.slow
def test_dataset_compare_smoke():
    dataset = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"
    if not dataset.exists():
        return
    # Use a small subset quickly; full regression will be pinned later
    wavs = sorted(dataset.glob("*.wav"))[:3]
    for wav in wavs:
        expected = parse_wsjt_lines(wav.with_suffix('.txt'))
        got = decode_wav(str(wav))
        got_msgs = {normalize_msg(r.message) for r in got if r.message}
        # Not asserting rate yet; ensure we at least produce a set and don't crash
        assert isinstance(got, list)

