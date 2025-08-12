from __future__ import annotations
import numpy as np
from typing import Tuple
from .ldpc_data import Nm, Mn

# Build parity-check matrix structure from adjacency
M = len(Nm)
N = len(Mn)

# Convert Nm (rows list of indices 1-origin with padding zeros) to Python lists of 0-origin column indices
H_rows = [ [c-1 for c in row if c>0] for row in Nm ]
# For bp decoding we need row-wise and col-wise neighborhoods
ROW_NEIGHBORS = H_rows
COL_NEIGHBORS = [ [r-1 for r in rows] for rows in Mn ]  # Mn lists parity check numbers 1-origin


def bp_decode(llr174: np.ndarray, max_iters: int = 50) -> Tuple[np.ndarray, int]:
    """Belief propagation decoding for LDPC(174,91) using min-sum with damping.

    llr174: log-likelihood ratios for N bits (positive favors 1)
    Returns hard bits (0/1) and number of unsatisfied parity checks.
    """
    Nloc = N
    Mloc = M
    assert llr174.shape[0] == Nloc
    # Messages: variable->check and check->variable
    # Initialize v2c with channel LLR
    v2c = {}
    c2v = {}
    for ci, cols in enumerate(ROW_NEIGHBORS):
        for v in cols:
            v2c[(v, ci)] = float(llr174[v])
            c2v[(ci, v)] = 0.0

    damping = 0.5
    for _ in range(max_iters):
        # Check node update (min-sum)
        for ci, cols in enumerate(ROW_NEIGHBORS):
            sgn = 1.0
            mins = 1e9
            second = 1e9
            min_idx = -1
            for v in cols:
                m = v2c[(v, ci)]
                if m < 0:
                    sgn *= -1.0
                a = abs(m)
                if a < mins:
                    second = mins
                    mins = a
                    min_idx = v
                elif a < second:
                    second = a
            for v in cols:
                a = mins if v != min_idx else second
                msg = (sgn * ( -1.0 if v2c[(v, ci)] < 0 else 1.0)) * a
                # damping
                c2v[(ci, v)] = (1-damping) * msg + damping * c2v[(ci, v)]

        # Variable node update
        for v in range(Nloc):
            neigh = COL_NEIGHBORS[v]
            sum_in = float(llr174[v]) + sum(c2v[(ci, v)] for ci in neigh)
            for ci in neigh:
                v2c[(v, ci)] = sum_in - c2v[(ci, v)]

        # Hard decision and syndrome check
        bits = np.zeros(Nloc, dtype=np.uint8)
        for v in range(Nloc):
            llr = float(llr174[v]) + sum(c2v[(ci, v)] for ci in COL_NEIGHBORS[v])
            bits[v] = 1 if llr < 0 else 0  # convention: positive favors 1 above; match mapping downstream
        # Compute parity checks
        unsat = 0
        for ci, cols in enumerate(ROW_NEIGHBORS):
            s = 0
            for v in cols:
                s ^= int(bits[v])
            if s != 0:
                unsat += 1
        if unsat == 0:
            return bits, 0
    return bits, unsat
