from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from .constants import LDPC_K, LDPC_M, LDPC_N
from .ldpc_tables_embedded import get_parity_matrices


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


# Systematic encoder derived from embedded parity matrices (no external files)
_P_cached: Optional[np.ndarray] = None
_enc_cached: Optional[tuple[np.ndarray, np.ndarray, list[int], list[int]]] = None


def _invert_binary_matrix_mod2(B: np.ndarray) -> np.ndarray:
    """Invert a binary (0/1) square matrix modulo 2 using Gauss-Jordan elimination."""
    m = B.shape[0]
    A = (B.copy() & 1).astype(np.uint8)
    I = np.eye(m, dtype=np.uint8)
    # Augment [A | I]
    aug = np.concatenate([A, I], axis=1)
    r = 0
    for c in range(m):
        # Find a pivot
        pivot = None
        for rr in range(r, m):
            if aug[rr, c] & 1:
                pivot = rr
                break
        if pivot is None:
            raise ValueError("Matrix is singular in GF(2)")
        # Swap rows
        if pivot != r:
            aug[[r, pivot]] = aug[[pivot, r]]
        # Eliminate other rows
        for rr in range(m):
            if rr == r:
                continue
            if aug[rr, c] & 1:
                aug[rr, :] ^= aug[r, :]
        r += 1
        if r == m:
            break
    inv = aug[:, m:]
    return inv


def _build_systematic_parity_matrix() -> np.ndarray:
    """Compute parity matrix P (shape [LDPC_M, LDPC_K]) such that p = P @ a91 (mod 2).
    Uses embedded Mn/Nm to construct H = [A|B] and returns P = B^{-1} A.
    """
    Mn, Nm = get_parity_matrices()
    # Build full H (LDPC_M x LDPC_N)
    H = np.zeros((LDPC_M, LDPC_N), dtype=np.uint8)
    # Nm stores 1-origin indices or -1 for padding; set H[row, idx-1] = 1
    for r in range(LDPC_M):
        for v in Nm[r]:
            vi = int(v)
            if vi < 1:
                continue
            H[r, vi - 1] ^= 1
    A = H[:, :LDPC_K]  # (M x K)
    B = H[:, LDPC_K:]  # (M x M)
    Binv = _invert_binary_matrix_mod2(B)
    P = (Binv @ A) & 1
    return P.astype(np.uint8)


def get_systematic_parity() -> np.ndarray:
    global _P_cached
    if _P_cached is None:
        _P_cached = _build_systematic_parity_matrix()
    return _P_cached


def encode174_bits_systematic(a91_bits: np.ndarray) -> np.ndarray:
    """Encode 91-bit payload+CRC into 174-bit codeword using systematic parity from embedded tables.

    Produces the codeword in original codeword bit order (0..173) matching the H matrix columns.
    """
    if a91_bits.shape[0] != LDPC_K:
        raise ValueError("a91_bits must have length LDPC_K")
    # Build full encoder based on column permutation so that Hpiv is invertible
    Br_inv, Arest, rest_cols, piv_cols = get_encoder_structures()
    a = a91_bits.astype(np.uint8)
    # Compute parity for chosen pivot columns: Hpiv * p = Arest * a
    rhs = (Arest @ a) % 2
    p = (Br_inv @ rhs) % 2
    # Assemble codeword in original codeword order (0..173)
    cw = np.zeros(LDPC_N, dtype=np.uint8)
    cw[np.array(rest_cols, dtype=np.int64)] = a
    cw[np.array(piv_cols, dtype=np.int64)] = p
    return cw


def encode174_bits_consecutive(a91_bits: np.ndarray) -> np.ndarray:
    """Encode 91-bit payload+CRC into 174-bit codeword in reference order [a|p].

    This matches the bit ordering expected by the reference implementation's ft8_encode/encode174,
    where the first LDPC_K bits are the message (a91) followed by LDPC_M parity bits.
    """
    if a91_bits.shape[0] != LDPC_K:
        raise ValueError("a91_bits must have length LDPC_K")
    P = get_systematic_parity()  # shape [M,K]
    a = a91_bits.astype(np.uint8)
    p = (P @ a) % 2
    cw = np.concatenate([a.astype(np.uint8), p.astype(np.uint8)])
    assert cw.shape[0] == LDPC_N
    return cw

def get_encoder_structures() -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Return (Hpiv_inv, Hrest, rest_cols, piv_cols) for H column permutation where Hpiv is invertible.
    Cached after first computation.
    """
    global _enc_cached
    if _enc_cached is not None:
        return _enc_cached
    Mn, Nm = get_parity_matrices()
    H = np.zeros((LDPC_M, LDPC_N), dtype=np.uint8)
    for r in range(LDPC_M):
        for v in Nm[r]:
            vi = int(v)
            if vi < 1:
                continue
            H[r, vi - 1] ^= 1
    # Gaussian elimination to pick pivot columns
    H_work = H.copy()
    piv_cols: list[int] = []
    r = 0
    for c in range(LDPC_N):
        if r >= LDPC_M:
            break
        # Find pivot in column c
        pivot = None
        for rr in range(r, LDPC_M):
            if H_work[rr, c] & 1:
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            H_work[[r, pivot]] = H_work[[pivot, r]]
        # Eliminate other rows
        for rr in range(LDPC_M):
            if rr != r and (H_work[rr, c] & 1):
                H_work[rr, :] ^= H_work[r, :]
        piv_cols.append(c)
        r += 1
    if len(piv_cols) != LDPC_M:
        raise ValueError("Could not find full-rank pivot columns for H")
    rest_cols = [c for c in range(LDPC_N) if c not in set(piv_cols)]
    # Partition H into [Hrest | Hpiv]
    Hrest = H[:, rest_cols]
    Hpiv = H[:, piv_cols]
    # Invert Hpiv in GF(2)
    Hpiv_inv = _invert_binary_matrix_mod2(Hpiv)
    _enc_cached = (Hpiv_inv, Hrest, rest_cols, piv_cols)
    return _enc_cached

