from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .constants import LDPC_N, LDPC_M


def _generate_synthetic_parity() -> Tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic synthetic (Mn, Nm) mapping with 3 rows per column.
    This is a fallback used only when external parity.dat is unavailable.
    """
    rows_for_col = np.zeros((LDPC_N, 3), dtype=np.int32)
    # Spread connections quasi-uniformly across rows
    for n in range(LDPC_N):
        r0 = (n * 3 + 0) % LDPC_M
        r1 = (n * 3 + LDPC_M // 3) % LDPC_M
        r2 = (n * 3 + (2 * LDPC_M) // 3) % LDPC_M
        rows_for_col[n, :] = (r0, r1, r2)
    # Build Nm lists
    cols_for_row: List[List[int]] = [[] for _ in range(LDPC_M)]
    for n in range(LDPC_N):
        for r in rows_for_col[n]:
            cols_for_row[r].append(n)
    max_deg = max(len(lst) for lst in cols_for_row)
    Nm = np.full((LDPC_M, max_deg), -1, dtype=np.int32)
    for r, lst in enumerate(cols_for_row):
        Nm[r, : len(lst)] = np.array(lst, dtype=np.int32)
    return rows_for_col, Nm


def load_parity_from_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load WSJT-X style parity.dat (83x174) where each of 174 lines lists 3 row indices (1..83)
    for the column. Returns (Mn, Nm) as 0-based integer arrays:
      - Mn: shape [LDPC_N, 3] mapping column -> three row indices
      - Nm: list of rows, each containing the list of column indices in that row
    If the file does not exist, returns a deterministic synthetic mapping suitable for tests.
    """
    if not path.exists():
        return _generate_synthetic_parity()

    rows_for_col = np.zeros((LDPC_N, 3), dtype=np.int32)
    lines: List[str] = path.read_text().splitlines()
    start = 0
    # Skip header lines until numbers begin (line with three integers)
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and all(p.lstrip("-+").isdigit() for p in parts):
            start = i
            break
    idx = 0
    for line in lines[start: start + LDPC_N]:
        parts = [int(p) for p in line.split()]
        if len(parts) != 3:
            raise ValueError("Invalid parity.dat format line: " + line)
        rows_for_col[idx, :] = np.array(parts, dtype=np.int32) - 1
        idx += 1
    # Build Nm lists
    cols_for_row: List[List[int]] = [[] for _ in range(LDPC_M)]
    for n in range(LDPC_N):
        for r in rows_for_col[n]:
            cols_for_row[r].append(n)
    # Pad rows to max degree 7 with -1
    max_deg = max(len(lst) for lst in cols_for_row)
    Nm = np.full((LDPC_M, max_deg), -1, dtype=np.int32)
    for r, lst in enumerate(cols_for_row):
        Nm[r, : len(lst)] = np.array(lst, dtype=np.int32)
    return rows_for_col, Nm


