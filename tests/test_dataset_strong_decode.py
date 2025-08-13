from pathlib import Path
import re

import numpy as np
import soundfile as sf
import pytest

from ft8gpt.decoder_e2e import decode_block
from ft8gpt.crc import crc14_check
from ft8gpt.message_decode import unpack_standard_payload


DATASET_DIR = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "test" / "wav"
PARITY_PATH = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "ft4_ft8_public" / "parity.dat"


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
        # Restrict to simple standard messages that our minimal unpacker can represent
        if re.match(r"^(CQ)\s+[A-Z0-9/]+\s+[A-R]{2}[0-9]{2}$", msg) or \
           re.match(r"^[A-Z0-9/]+\s+[A-Z0-9/]+\s+[A-R]{2}[0-9]{2}$", msg):
            msgs.add(msg)
    return msgs


@pytest.mark.xfail(reason="Decoder produces valid LDPC/CRC but text mapping not aligned with dataset yet")
def test_dataset_zero_syndrome_crc_and_text_match():
    assert DATASET_DIR.exists(), (
        "Required dataset directory is missing: external/ft8_lib/test/wav (ensure submodules are checked out)"
    )
    assert PARITY_PATH.exists(), "Missing parity.dat file"

    # Prefer a sample with expected text available
    preferred = DATASET_DIR / "websdr_test1.wav"
    txt = preferred.with_suffix(".txt")
    if not preferred.exists() or not txt.exists():
        # Fallback: choose any wav with a matching .txt
        pairs = sorted(p for p in DATASET_DIR.glob("*.wav") if p.with_suffix(".txt").exists())
        assert pairs, "No WAV+TXT pairs found in dataset directory"
        preferred = pairs[0]
        txt = preferred.with_suffix(".txt")

    expected_msgs = _expected_messages_set(txt)
    assert expected_msgs, "No expected standard messages parsed from dataset txt"

    # Load audio
    samples, fs = sf.read(str(preferred), always_2d=False)
    x = samples[:, 0] if getattr(samples, "ndim", 1) > 1 else samples
    x = np.asarray(x, dtype=np.float64)

    # Decode candidates
    cands = decode_block(x, float(fs), PARITY_PATH)
    assert isinstance(cands, list) and len(cands) > 0

    # Check LDPC/CRC and reconstruct messages; require at least one match
    matched = False
    for c in cands:
        assert c.ldpc_errors == 0
        assert crc14_check(c.bits_with_crc)
        payload_bits = np.concatenate([c.bits_with_crc[:77], np.zeros(3, dtype=np.uint8)])
        a10 = np.packbits(payload_bits)[:10].tobytes()
        try:
            dec = unpack_standard_payload(a10)
            msg = f"{dec.call_to} {dec.call_de} {dec.extra}".strip().upper()
            msg = re.sub(r"\s+", " ", msg)
        except Exception:
            msg = ""
        if msg and msg in expected_msgs:
            matched = True
            break

    assert matched, "No decoded text matched any expected standard message"