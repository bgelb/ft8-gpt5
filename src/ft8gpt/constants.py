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

# Costas 7x7 sync tone pattern and Gray map (bits -> tone index)
# These small tables are public and protocol-defined.
FT8_COSTAS_PATTERN = (3, 1, 4, 0, 6, 5, 2)
FT8_GRAY_MAP = (0, 1, 3, 2, 5, 6, 4, 7)

# Precompute inverse mapping from tone index -> 3-bit binary index (b2b1b0 as integer 0..7)
_FT8_INV_GRAY_MAP = tuple(int(i) for i, tone in sorted([(i, t) for i, t in enumerate(FT8_GRAY_MAP)], key=lambda x: x[1]))


def gray_to_bits(gray_symbol: int) -> tuple[int, int, int]:
    """Return (b2,b1,b0) for the FT8-specific Gray map given a tone index 0..7.

    Note: FT8 uses a non-standard Gray ordering; do NOT use the generic XOR inverse.
    """
    g = gray_symbol & 7
    idx = _FT8_INV_GRAY_MAP[g]
    return (idx >> 2) & 1, (idx >> 1) & 1, idx & 1


def bits_to_gray(b2: int, b1: int, b0: int) -> int:
    """Return Gray-coded tone index for the three bits (b2,b1,b0)."""
    idx = (b2 & 1) << 2 | (b1 & 1) << 1 | (b0 & 1)
    # Binary to Gray: g = b ^ (b >> 1) equals FT8_GRAY_MAP[idx]
    return FT8_GRAY_MAP[idx]


