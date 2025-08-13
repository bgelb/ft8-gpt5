from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import LENGTH_SYNC, NUM_SYNC, SYNC_OFFSET, FT8_COSTAS_PATTERN


@dataclass(frozen=True)
class SyncHit:
    time_symbol: int
    score: float
    base_index: int = -1


def _vectorized_sync_scores_2d(waterfall_2d: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute sync scores for all plausible start times on a 2D waterfall [T, 8].

    Uses neighbor-contrast matched filtering across the three Costas sync blocks,
    vectorized over all candidate start times.
    Returns array of shape [T_valid] with one score per start time.
    """
    T, K = waterfall_2d.shape
    assert K >= 8, "Expected at least 8 tones on last axis"

    # Normalize per-time row to suppress wideband variations
    x = waterfall_2d - np.median(waterfall_2d, axis=1, keepdims=True)

    # Neighbor contrast along tone axis (left/right), with zero at edges
    left = x - np.pad(x[:, :-1], ((0, 0), (1, 0)), mode="constant")
    right = x - np.pad(x[:, 1:], ((0, 0), (0, 1)), mode="constant")

    max_offset = (NUM_SYNC - 1) * SYNC_OFFSET + LENGTH_SYNC - 1
    num_positions = max(0, T - max_offset)
    if num_positions == 0:
        return np.zeros((0,), dtype=np.float64)

    # Build time and tone index sequences for the 21 sync positions
    block_offsets = np.arange(LENGTH_SYNC)[None, :] + SYNC_OFFSET * np.arange(NUM_SYNC)[:, None]
    flat_time_offsets = block_offsets.reshape(-1)  # length = NUM_SYNC * LENGTH_SYNC
    flat_tones = np.tile(np.array(FT8_COSTAS_PATTERN, dtype=int), NUM_SYNC)

    # Accumulate scores across all 21 positions
    base_t = np.arange(num_positions)[:, None]  # [T_valid, 1]
    scores = np.zeros((num_positions,), dtype=np.float64)

    for off, tone in zip(flat_time_offsets, flat_tones):
        t_idx = (base_t + off).ravel()
        scores += left[t_idx, tone] + right[t_idx, tone]

    # Normalize by effective neighbor count: 13 per block (tone 0 has only one neighbor)
    denom = NUM_SYNC * (2 * LENGTH_SYNC - 1)
    scores /= float(denom)
    return scores


def _vectorized_sync_scores_3d(waterfall_3d: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Compute sync scores for all plausible start times on a 3D waterfall [T, B, 8].

    Returns:
      - best_scores: [T_valid] best score across base bins for each start time
      - best_base_indices: [T_valid] argmax base index per start time
    """
    T, B, K = waterfall_3d.shape
    assert K >= 8, "Expected at least 8 tones on last axis"

    # Normalize per-time, per-base across tones
    x = waterfall_3d - np.median(waterfall_3d, axis=2, keepdims=True)

    # Neighbor contrast along tone axis (left/right), with zero at edges
    left = x - np.pad(x[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode="constant")
    right = x - np.pad(x[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode="constant")

    max_offset = (NUM_SYNC - 1) * SYNC_OFFSET + LENGTH_SYNC - 1
    num_positions = max(0, T - max_offset)
    if num_positions == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    block_offsets = np.arange(LENGTH_SYNC)[None, :] + SYNC_OFFSET * np.arange(NUM_SYNC)[:, None]
    flat_time_offsets = block_offsets.reshape(-1)
    flat_tones = np.tile(np.array(FT8_COSTAS_PATTERN, dtype=int), NUM_SYNC)

    base_t = np.arange(num_positions)[:, None]  # [T_valid, 1]
    scores_tb = np.zeros((num_positions, B), dtype=np.float64)

    for off, tone in zip(flat_time_offsets, flat_tones):
        t_idx = (base_t + off).ravel()
        # Gather per-base contributions and accumulate
        scores_tb += left[t_idx, :, tone] + right[t_idx, :, tone]

    denom = NUM_SYNC * (2 * LENGTH_SYNC - 1)
    scores_tb /= float(denom)

    best_base_indices = np.argmax(scores_tb, axis=1)
    best_scores = scores_tb[np.arange(num_positions), best_base_indices]
    return best_scores, best_base_indices.astype(np.int64)


def costas_score(waterfall: NDArray[np.float64], time_symbol: int) -> float:
    """
    Reference scalar implementation retained for validation and potential unit tests.
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
        return 0.0
    return score_sum / count


def find_sync_positions(
    waterfall: NDArray[np.float64],
    min_score: float = 0.0,
    top_n: int | None = 50,
) -> List[SyncHit]:
    """
    Find likely sync start positions using a vectorized matched filter.

    Accepts either a collapsed waterfall [T, 8] or full base waterfall [T, B, 8].
    Returns SyncHit list sorted by score desc. If top_n is provided, returns only best N.
    """
    if waterfall.ndim == 3:
        scores, best_bases = _vectorized_sync_scores_3d(waterfall)
    elif waterfall.ndim == 2:
        scores = _vectorized_sync_scores_2d(waterfall)
        best_bases = None
    else:
        raise ValueError("waterfall must be 2D [T,8] or 3D [T,B,8]")

    # Threshold and select candidates
    t_indices = np.arange(scores.shape[0], dtype=int)
    mask = scores >= float(min_score)
    t_indices = t_indices[mask]
    scores = scores[mask]
    if best_bases is not None:
        best_bases = best_bases[mask]

    if t_indices.size == 0:
        return []

    # Non-maximum suppression in a small window to avoid clustered duplicates
    # Use a simple 3-wide max filter implemented via comparisons
    keep = np.ones_like(scores, dtype=bool)
    for i in range(scores.size):
        if not keep[i]:
            continue
        # Suppress neighbors if they are not strictly better
        if i - 1 >= 0 and scores[i - 1] <= scores[i]:
            keep[i - 1] = False
        if i + 1 < scores.size and scores[i + 1] <= scores[i]:
            keep[i + 1] = False
    t_indices = t_indices[keep]
    scores = scores[keep]
    if best_bases is not None:
        best_bases = best_bases[keep]

    # Sort by score desc
    order = np.argsort(-scores)
    if top_n is not None:
        order = order[:top_n]
    t_sorted = t_indices[order]
    s_sorted = scores[order]
    if best_bases is not None:
        b_sorted = best_bases[order]
        return [SyncHit(time_symbol=int(t), score=float(s), base_index=int(b)) for t, s, b in zip(t_sorted, s_sorted, b_sorted)]

    return [SyncHit(time_symbol=int(t), score=float(s)) for t, s in zip(t_sorted, s_sorted)]


