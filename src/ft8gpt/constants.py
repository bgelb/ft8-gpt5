from __future__ import annotations

"""
Core FT8 protocol constants and small lookup tables.

Values follow the public FT8/FT4 description (Franke, Somerville, Taylor) and
match the de-facto standard used by interoperable implementations.
"""

# Timing
SYMBOL_PERIOD_S = 0.160  # seconds per FT8 symbol (6.25 baud)
SLOT_TIME_S = 15.0       # seconds per T/R slot

# Frame layout
ND = 58                  # number of data symbols (carry 3 bits each)
NN = 79                  # total channel symbols (includes sync symbols)
LENGTH_SYNC = 7          # symbols per Costas sync block
NUM_SYNC = 3             # number of Costas sync blocks
SYNC_OFFSET = 36         # distance between starts of successive sync blocks

# Modulation
FSK_TONES = 8
TONE_SPACING_HZ = 6.25

# LDPC(174,91)
LDPC_N = 174
LDPC_K = 91
LDPC_M = 83

# CRC-14
CRC14_POLY = 0x2757

# Costas 7x7 sync tone pattern and Gray map (tones -> bits)
# These small tables are public and protocol-defined.
FT8_COSTAS_PATTERN = (3, 1, 4, 0, 6, 5, 2)
FT8_GRAY_MAP = (0, 1, 3, 2, 5, 6, 4, 7)


def gray_to_bits(gray_symbol: int) -> tuple[int, int, int]:
    """Return the three bits (b2,b1,b0) corresponding to a Gray-coded tone index."""
    v = FT8_GRAY_MAP[gray_symbol]
    return (v >> 2) & 1, (v >> 1) & 1, v & 1


def bits_to_gray(b2: int, b1: int, b0: int) -> int:
    """Return Gray-coded tone index for the three bits (b2,b1,b0)."""
    idx = (b2 & 1) << 2 | (b1 & 1) << 1 | (b0 & 1)
    return FT8_GRAY_MAP[idx]


