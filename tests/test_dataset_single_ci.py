from pathlib import Path
import re
import pytest

from ft8gpt import decode_wav


DATASET_DIR = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"


def _expected_messages_set(txt_path: Path) -> set[str]:
    msgs: set[str] = set()
    if not txt_path.exists():
        return msgs
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        msg = line.split("~", 1)[1].strip() if "~" in line else " ".join(line.split()[5:])
        msg = re.sub(r"\s+", " ", msg.upper())
        if re.match(r"^(CQ)\s+[A-Z0-9/]+\s+[A-R]{2}[0-9]{2}$", msg) or \
           re.match(r"^[A-Z0-9/]+\s+[A-Z0-9/]+\s+[A-R]{2}[0-9]{2}$", msg):
            msgs.add(msg)
    return msgs


def _normalize_msg(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def test_dataset_single_decode_ci():
    if not DATASET_DIR.exists():
        pytest.skip("dataset not available")

    # Try a small set of candidate WAVs deterministically
    candidates = [
        DATASET_DIR / "websdr_test1.wav",
        DATASET_DIR / "websdr_test10.wav",
        DATASET_DIR / "191111_110615.wav",
    ]
    candidates = [p for p in candidates if p.exists() and p.with_suffix('.txt').exists()]
    if not candidates:
        pytest.skip("no suitable WAV+TXT pairs found in dataset directory")

    matched = False
    for wav in candidates:
        expected_msgs = _expected_messages_set(wav.with_suffix('.txt'))
        if not expected_msgs:
            continue
        results = decode_wav(str(wav))
        if not isinstance(results, list):
            continue
        good = [r for r in results if r.crc14_ok and r.ldpc_errors == 0 and r.message]
        got_msgs = {_normalize_msg(r.message) for r in good}
        if got_msgs & expected_msgs:
            matched = True
            break

    assert matched, "No CRC-valid, zero-syndrome text match found in small dataset subset"