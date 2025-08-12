from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np

from .constants import LDPC_K, LDPC_M, LDPC_N


def load_generator_from_file(path: Path) -> np.ndarray:
    """Load generator.dat as a boolean matrix of shape [LDPC_M, LDPC_K]."""
    lines = path.read_text().splitlines()
    # Skip header lines until a full 91-bit row is found
    rows: list[list[int]] = []
    for line in lines:
        s = line.strip()
        if len(s) >= LDPC_K and set(s[:LDPC_K]).issubset({"0", "1"}):
            rows.append([1 if c == "1" else 0 for c in s[:LDPC_K]])
    if len(rows) < LDPC_M:
        raise ValueError("generator.dat appears incomplete")
    G = np.array(rows[:LDPC_M], dtype=np.uint8)
    return G


def encode174_bits(a91_bits: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Encode 91 payload+CRC bits to 174-bit codeword using generator matrix G (LDPC_M x LDPC_K).

    codeword = [a91_bits (K bits)] + [parity (M bits)], parity[i] = sum(G[i,j]*a91[j]) mod 2
    """
    if a91_bits.shape[0] != LDPC_K:
        raise ValueError("a91_bits must have length LDPC_K")
    # Parity bits vector p = G @ a91 (mod 2)
    p = (G @ a91_bits.astype(np.uint8)) % 2
    codeword = np.concatenate([a91_bits.astype(np.uint8), p.astype(np.uint8)])
    assert codeword.shape[0] == LDPC_N
    return codeword


