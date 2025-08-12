from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import LDPC_N, LDPC_M


@dataclass(frozen=True)
class BeliefPropagationConfig:
    max_iterations: int = 30
    early_stop_no_improve: int = 5
    damping: float = 0.0
    alpha: float = 0.8  # normalized min-sum scaling factor (0.7..0.9 typical)


def _build_row_index_of_col_in_row(Mn: NDArray[np.int32], Nm: NDArray[np.int32]) -> NDArray[np.int32]:
    """Precompute, for each variable n and its three connected checks (rows), the index j such that
    Nm[row, j] == n. Missing entries are set to -1.
    """
    row_index = np.full((LDPC_N, 3), -1, dtype=np.int32)
    for n in range(LDPC_N):
        for m_idx in range(3):
            r = Mn[n, m_idx]
            if r < 0:
                continue
            # search n in row r once
            ridxs = Nm[r]
            where = np.where(ridxs == n)[0]
            if where.size:
                row_index[n, m_idx] = int(where[0])
    return row_index


def min_sum_decode(
    llr_174: NDArray[np.float64],
    Mn: NDArray[np.int32],
    Nm: NDArray[np.int32],
    config: BeliefPropagationConfig,
) -> Tuple[int, NDArray[np.uint8]]:
    """
    Min-sum LDPC decoder for (174,91) code using sparse connectivity.

    Returns (num_unsatisfied_checks, hard_bits)
    """
    # Initialize messages variable->check
    tov = np.zeros((LDPC_N, 3), dtype=np.float64)
    # toc allocated with per-row variable degrees
    row_deg = np.sum(Nm >= 0, axis=1)
    max_deg = Nm.shape[1]
    toc = np.zeros((LDPC_M, max_deg), dtype=np.float64)

    # Precompute index j per (n, m_idx) to avoid np.where in inner loop
    row_index = _build_row_index_of_col_in_row(Mn, Nm)

    best_errors = LDPC_M + 1
    best_bits = np.zeros(LDPC_N, dtype=np.uint8)
    no_improve = 0

    for _ in range(config.max_iterations):
        # Hard decision
        total = llr_174 + tov.sum(axis=1)
        bits = (total < 0).astype(np.uint8)  # 1 if LLR negative
        # Check parity
        errors = 0
        for r in range(LDPC_M):
            idxs = Nm[r, : row_deg[r]]
            if idxs.size == 0:
                continue
            parity = np.bitwise_xor.reduce(bits[idxs])
            if parity != 0:
                errors += 1
        if errors < best_errors:
            best_errors = errors
            best_bits = bits.copy()
            no_improve = 0
        else:
            no_improve += 1
        if best_errors == 0 or no_improve >= config.early_stop_no_improve:
            break

        # Check node update: messages from check r to each connected variable
        for r in range(LDPC_M):
            deg = int(row_deg[r])
            if deg == 0:
                continue
            idxs = Nm[r, :deg]
            incoming = np.empty(deg, dtype=np.float64)
            # Build incoming messages for each connected variable
            for i in range(deg):
                n = int(idxs[i])
                # Sum tov from other checks connected to n (exclude r)
                # Identify which of the three entries refers to row r
                where = np.where(Mn[n] == r)[0]
                mpos = int(where[0]) if where.size else 0
                incoming[i] = llr_174[n] + tov[n, (np.arange(3) != mpos)].sum()

            signs = np.sign(incoming)
            absvals = np.abs(incoming)
            prod_sign = np.prod(signs) if signs.size > 0 else 1.0
            min1 = np.min(absvals) if deg > 0 else 0.0
            # Normalized min-sum scaling
            min1 *= config.alpha
            for i in range(deg):
                s_excl = prod_sign / (signs[i] if signs[i] != 0 else 1.0)
                new_msg = s_excl * min1
                if config.damping > 0.0:
                    toc[r, i] = (1 - config.damping) * new_msg + config.damping * toc[r, i]
                else:
                    toc[r, i] = new_msg

        # Variable node update: messages from var n to check m_idx
        for n in range(LDPC_N):
            for m_idx in range(3):
                r = Mn[n, m_idx]
                if r < 0:
                    continue
                j = int(row_index[n, m_idx])
                if j < 0:
                    continue
                deg = int(row_deg[r])
                # Sum all check messages to n excluding from r
                # Compute total once and subtract toc[r, j]
                sum_checks = float(np.sum(toc[r, :deg])) - float(toc[r, j])
                new_val = llr_174[n] + sum_checks
                if config.damping > 0.0:
                    tov[n, m_idx] = (1 - config.damping) * new_val + config.damping * tov[n, m_idx]
                else:
                    tov[n, m_idx] = new_val

    return best_errors, best_bits


