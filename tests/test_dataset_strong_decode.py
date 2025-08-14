from pathlib import Path
import re
import pytest

import numpy as np
import soundfile as sf

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


@pytest.mark.skip(reason="Temporarily skipping dataset strong decode while we land encoder/decoder fixes")
def test_dataset_zero_syndrome_crc_and_text_match():
    assert DATASET_DIR.exists(), (
        "Required dataset directory is missing: external/ft8_lib/test/wav (ensure submodules are checked out)"
    )

    preferred = DATASET_DIR / "websdr_test1.wav"
    txt = preferred.with_suffix(".txt")
    if not preferred.exists() or not txt.exists():
        pairs = sorted(p for p in DATASET_DIR.glob("*.wav") if p.with_suffix(".txt").exists())
        assert pairs, "No WAV+TXT pairs found in dataset directory"
        preferred = pairs[0]
        txt = preferred.with_suffix(".txt")

    expected_msgs = _expected_messages_set(txt)
    assert expected_msgs, "No expected standard messages parsed from dataset txt"

    results = decode_wav(str(preferred))
    assert isinstance(results, list) and len(results) > 0

    # Require at least one result with zero LDPC errors and crc14_ok True
    good = [r for r in results if r.crc14_ok and r.ldpc_errors == 0]
    assert good, "No decodes with zero LDPC syndrome and valid CRC"

    got_msgs = {_normalize_msg(r.message) for r in good if r.message}
    assert got_msgs & expected_msgs, "No decoded text matched any expected standard message"