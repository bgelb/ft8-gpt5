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

    # Precompute fast lookups:
    # mpos_lookup[n, r] = position index 0..2 of check r in Mn[n]
    mpos_lookup = np.full((LDPC_N, LDPC_M), -1, dtype=np.int8)
    for n in range(LDPC_N):
        for m_idx in range(3):
            r = int(Mn[n, m_idx])
            if r >= 0:
                mpos_lookup[n, r] = np.int8(m_idx)
    # var_index_in_row[r, n] = j such that Nm[r, j] == n
    var_index_in_row = np.full((LDPC_M, LDPC_N), -1, dtype=np.int16)
    for r in range(LDPC_M):
        deg = int(row_deg[r])
        for j in range(deg):
            n = int(Nm[r, j])
            if n >= 0:
                var_index_in_row[r, n] = np.int16(j)

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
            if deg <= 0:
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
        for r in range(LDPC_M):
            deg = int(row_deg[r])
            if deg <= 0:
                continue
            idxs = Nm[r, :deg]
            # Build incoming messages efficiently
            incoming = np.empty(deg, dtype=np.float64)
            for i in range(deg):
                n = int(idxs[i])
                mpos = int(mpos_lookup[n, r])
                # Sum tov for variable n excluding the message from this check (mpos)
                total_tov = tov[n, 0] + tov[n, 1] + tov[n, 2]
                incoming[i] = llr_174[n] + (total_tov - tov[n, mpos])

            signs = np.sign(incoming)
            absvals = np.abs(incoming)
            prod_sign = np.prod(signs) if signs.size > 0 else 1.0
            min1 = np.min(absvals)
            for i in range(deg):
                s_excl = prod_sign / (signs[i] if signs[i] != 0 else 1.0)
                new_msg = s_excl * min1
                if config.damping > 0.0:
                    toc[r, i] = (1 - config.damping) * new_msg + config.damping * toc[r, i]
                else:
                    toc[r, i] = new_msg

        # Variable node update: messages from var n to check m_idx
        # Precompute per-row sums once to avoid inner-loop summations
        row_sums = np.zeros(LDPC_M, dtype=np.float64)
        for r in range(LDPC_M):
            deg = int(row_deg[r])
            if deg > 0:
                row_sums[r] = float(np.sum(toc[r, :deg]))
        for n in range(LDPC_N):
            for m_idx in range(3):
                r = int(Mn[n, m_idx])
                if r < 0:
                    continue
                j = int(var_index_in_row[r, n])
                if j < 0:
                    continue
                # Sum all check messages to n excluding from r
                sum_checks = row_sums[r] - toc[r, j]
                new_val = llr_174[n] + sum_checks
                if config.damping > 0.0:
                    tov[n, m_idx] = (1 - config.damping) * new_val + config.damping * tov[n, m_idx]
                else:
                    tov[n, m_idx] = new_val

    return best_errors, best_bits


