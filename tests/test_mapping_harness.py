import numpy as np
from dataclasses import dataclass

from ft8gpt.constants import FT8_COSTAS_PATTERN, FT8_GRAY_MAP
from ft8gpt.synth import tones_from_codeword

# ---------- Spec constants (QEX) ----------
COSTAS = np.array(FT8_COSTAS_PATTERN, dtype=int)
GRAY_TONE_OF = {
    (0,0,0): 0,
    (0,0,1): 1,
    (0,1,1): 2,
    (0,1,0): 3,
    (1,1,0): 4,
    (1,0,0): 5,
    (1,0,1): 6,
    (1,1,1): 7,
}
BITS_OF_TONE = {v:k for k,v in GRAY_TONE_OF.items()}
DATA_BLOCK_1 = np.arange(7, 7+29)
DATA_BLOCK_2 = np.arange(43, 43+29)
DATA_SLOTS = np.concatenate([DATA_BLOCK_1, DATA_BLOCK_2])

# ---------- Reference bit->tones (data-only) ----------
def bits174_to_data_tones58_ref(bits174: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits174, dtype=int)
    assert bits.size == 174
    triads = bits.reshape(58, 3)
    tones = np.empty(58, dtype=int)
    for i, (b0,b1,b2) in enumerate(triads):
        tones[i] = GRAY_TONE_OF[(b0,b1,b2)]
    return tones


def data_tones58_to_frame79_ref(tones58: np.ndarray) -> np.ndarray:
    tones58 = np.asarray(tones58, dtype=int)
    assert tones58.size == 58
    frame = -np.ones(79, dtype=int)
    frame[0:7]   = COSTAS
    frame[7:36]  = tones58[0:29]
    frame[36:43] = COSTAS
    frame[43:72] = tones58[29:58]
    frame[72:79] = COSTAS
    return frame


# ---------- Property tests (pure reference) ----------
def test_gray_adjacency_ref():
    def hd(x,y): return sum(int(a!=b) for a,b in zip(BITS_OF_TONE[x], BITS_OF_TONE[y]))
    for a,b in zip(range(0,7), range(1,8)):
        assert hd(a,b) == 1


def test_frame_layout_ref():
    assert list(DATA_BLOCK_1) == list(range(7,36))
    assert list(DATA_BLOCK_2) == list(range(43,72))


# ---------- Counterexamples (pure reference) ----------
def test_counterexample_endianness_ref():
    bits = np.zeros(174, dtype=int)
    bits[0:3] = [0,0,1]
    tones = bits174_to_data_tones58_ref(bits)
    assert tones[0] == 1


def test_counterexample_block_boundary_ref():
    bits = np.zeros(174, dtype=int)
    bits[28*3:28*3+3] = [1,1,1]
    bits[29*3:29*3+3] = [0,0,0]
    frame = data_tones58_to_frame79_ref(bits174_to_data_tones58_ref(bits))
    assert frame[35] == 7
    assert frame[43] == 0


def test_counterexample_non_gray_ref():
    # FT8 Gray table spot-check: tone 2 must be 011
    assert BITS_OF_TONE[2] == (0,1,1)


# ---------- Cross-check our implementation vs reference ----------
@dataclass
class Adapter:
    pass


def _extract_data_tones_from_frame(frame79: np.ndarray) -> np.ndarray:
    frame79 = np.asarray(frame79, dtype=int)
    return frame79[DATA_SLOTS]


def test_adapter_matches_reference_random():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        bits = rng.integers(0, 2, size=174, dtype=np.uint8)
        # Reference
        tones_ref = bits174_to_data_tones58_ref(bits)
        frame_ref = data_tones58_to_frame79_ref(tones_ref)
        # Adapter via library
        frame_lib = tones_from_codeword(bits)
        tones_lib = _extract_data_tones_from_frame(frame_lib)
        assert np.array_equal(tones_lib, tones_ref)
        assert np.array_equal(frame_lib, frame_ref)


def test_adapter_counterexample_endianness():
    bits = np.zeros(174, dtype=int)
    bits[0:3] = [0,0,1]
    frame = tones_from_codeword(bits)
    assert frame[7] == 1


def test_adapter_counterexample_block_boundary():
    bits = np.zeros(174, dtype=int)
    bits[28*3:28*3+3] = [1,1,1]
    bits[29*3:29*3+3] = [0,0,0]
    frame = tones_from_codeword(bits)
    assert frame[35] == 7
    assert frame[43] == 0


def test_gray_table_consistency_constant():
    # Ensure our FT8_GRAY_MAP constant encodes the exact table expected.
    expected = [0,1,3,2,5,6,4,7]
    assert list(FT8_GRAY_MAP) == expected