from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import NDArray

from .constants import LENGTH_SYNC, NUM_SYNC, SYNC_OFFSET, FT8_COSTAS_PATTERN


@dataclass(frozen=True)
class SyncHit:
    time_symbol: int
    score: float


def costas_score(waterfall: NDArray[np.float64], time_symbol: int) -> float:
    """
    Compute a simple sync score at a given symbol time index using local contrast.
    waterfall shape: [num_symbols, num_bins], num_bins>=8
    """
    num_symbols, num_bins = waterfall.shape
    score_sum = 0.0
    count = 0
    for m in range(NUM_SYNC):
        for k in range(LENGTH_SYNC):
            block = time_symbol + (SYNC_OFFSET * m) + k
            if block < 0 or block >= num_symbols:
                continue
            row = waterfall[block]
            sm = FT8_COSTAS_PATTERN[k]
            s = row[sm]
            if sm > 0:
                score_sum += s - row[sm - 1]
                count += 1
            if sm < 7:
                score_sum += s - row[sm + 1]
                count += 1
    if count == 0:
        return -1e9
    return score_sum / max(1, count)


def find_sync_positions(waterfall: NDArray[np.float64], min_score: float = 0.0) -> List[SyncHit]:
    hits: List[SyncHit] = []
    num_symbols = waterfall.shape[0]
    # Search a plausible window; here scan broadly
    for t in range(0, max(0, num_symbols - (LENGTH_SYNC + (NUM_SYNC - 1) * SYNC_OFFSET)) + 1):
        s = costas_score(waterfall, t)
        if s >= min_score:
            hits.append(SyncHit(time_symbol=t, score=s))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


