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

    # Deterministic single file with strong signals
    wav = DATASET_DIR / "20m_busy" / "test_04.wav"
    if not wav.exists() or not wav.with_suffix('.txt').exists():
        pytest.skip("required dataset file not available: 20m_busy/test_04.wav")

    expected_msgs = _expected_messages_set(wav.with_suffix('.txt'))
    assert expected_msgs, "no expected messages parsed from txt"

    results = decode_wav(str(wav))
    assert isinstance(results, list)
    good = [r for r in results if r.crc14_ok and r.message]
    got_msgs = {_normalize_msg(r.message) for r in good}
    assert got_msgs & expected_msgs, "decoder produced no CRC-valid text matches for test_04.wav"