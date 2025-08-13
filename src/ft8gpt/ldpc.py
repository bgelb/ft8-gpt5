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

    # Precompute, for each row r and edge position j, the index mpos (0..2) of r in Mn[n]
    mpos_by_row = np.full((LDPC_M, max_deg), -1, dtype=np.int32)
    for r in range(LDPC_M):
        deg = int(row_deg[r])
        if deg == 0:
            continue
        ridxs = Nm[r, :deg]
        for j, n in enumerate(ridxs):
            # Find which of the 3 neighbors of n is row r
            # We do this once up-front to avoid np.where in the hot loop
            where = np.where(Mn[n] == r)[0]
            mpos_by_row[r, j] = int(where[0]) if where.size else 0

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
            deg = int(row_deg[r])
            if deg == 0:
                continue
            idxs = Nm[r, :deg]
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
        sum_tov_by_var = tov.sum(axis=1)  # shape [N]
        for r in range(LDPC_M):
            deg = int(row_deg[r])
            if deg == 0:
                continue
            idxs = Nm[r, :deg]
            mpos_row = mpos_by_row[r, :deg]
            # Incoming messages to check r from each connected var n
            incoming = llr_174[idxs] + (sum_tov_by_var[idxs] - tov[idxs, mpos_row])

            signs = np.sign(incoming)
            absvals = np.abs(incoming)
            prod_sign = np.prod(signs) if signs.size > 0 else 1.0
            min1 = np.min(absvals) if absvals.size > 0 else 0.0

            # Exclude-edge sign via division (handle 0 sign by substituting 1)
            safe_signs = np.where(signs == 0.0, 1.0, signs)
            new_msgs = (prod_sign / safe_signs) * min1

            if config.damping > 0.0:
                toc[r, :deg] = (1 - config.damping) * new_msgs + config.damping * toc[r, :deg]
            else:
                toc[r, :deg] = new_msgs

        # Variable node update: messages from var n to check m_idx
        # Use row-wise sums to avoid inner loops
        row_sums = np.array([toc[r, : int(row_deg[r])].sum() for r in range(LDPC_M)], dtype=np.float64)
        for r in range(LDPC_M):
            deg = int(row_deg[r])
            if deg == 0:
                continue
            idxs = Nm[r, :deg]
            mpos_row = mpos_by_row[r, :deg]
            sum_checks_excl = row_sums[r] - toc[r, :deg]
            new_vals = llr_174[idxs] + sum_checks_excl
            if config.damping > 0.0:
                tov[idxs, mpos_row] = (1 - config.damping) * new_vals + config.damping * tov[idxs, mpos_row]
            else:
                tov[idxs, mpos_row] = new_vals

    return best_errors, best_bits


