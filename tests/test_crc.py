import numpy as np

from ft8gpt.crc import CRC14_POLY as POLY
from ft8gpt.crc import crc14, crc14_check


def test_crc14_roundtrip():
    payload = np.zeros(77, dtype=np.uint8)
    c = crc14(payload)
    bits = np.concatenate([payload, np.array([(c >> i) & 1 for i in range(13, -1, -1)], dtype=np.uint8)])
    assert crc14_check(bits)


def test_crc14_variation():
    # Randomized patterns to ensure roundtrip across a variety of payloads
    rng = np.random.default_rng(123)
    for _ in range(20):
        payload = rng.integers(0, 2, size=77, dtype=np.uint8)
        c = crc14(payload)
        bits = np.concatenate([payload, np.array([(c >> i) & 1 for i in range(13, -1, -1)], dtype=np.uint8)])
        assert crc14_check(bits)
        # Flip a bit to ensure failure
        bits = bits.copy()
        bits[0] ^= 1
        assert not crc14_check(bits)

