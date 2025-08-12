import numpy as np
from ft8dec.crc14 import crc14, POLY


def test_crc14_smoke():
    # Random bits; just ensure it returns 14-bit value
    bits = np.random.randint(0, 2, size=77, dtype=np.uint8)
    val = crc14(bits)
    assert 0 <= val < (1<<14)


def test_crc14_known_vector_placeholder():
    # Placeholder test: adjust when authoritative poly is confirmed
    bits = np.zeros(77, dtype=np.uint8)
    v0 = crc14(bits)
    assert isinstance(v0, int)

