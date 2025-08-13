from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .constants import NN, ND, LENGTH_SYNC, SYNC_OFFSET
from .waterfall import compute_waterfall_symbols, Waterfall
from .sync import find_sync_positions
from .tones import extract_symbol_llrs
from .ldpc import min_sum_decode, BeliefPropagationConfig
from .ldpc_tables import load_parity_from_file
from .crc import crc14_check
from .message_decode import unpack_standard_payload


@dataclass(frozen=True)
class CandidateDecode:
    start_symbol: int
    base_bin: int
    ldpc_errors: int
    bits_with_crc: np.ndarray


def _slice_data_symbol_rows(wf: Waterfall, start_symbol: int, base_bin: int) -> np.ndarray:
    symbol_rows = []
    for k in range(ND):
        sym_idx = start_symbol + (k + (7 if k < 29 else 14))
        if sym_idx < 0 or sym_idx >= wf.mag.shape[0]:
            break
        if base_bin < 0 or base_bin >= wf.mag.shape[1]:
            break
        symbol_rows.append(wf.mag[sym_idx, base_bin, :])
    if len(symbol_rows) < ND:
        return np.empty((0, 8), dtype=np.float64)
    return np.stack(symbol_rows, axis=0)


def llrs_from_waterfall(wf_group: np.ndarray) -> np.ndarray:
    # wf_group shape [num_symbols, 8]
    llrs: List[float] = []
    for s in range(wf_group.shape[0]):
        l2, l1, l0 = extract_symbol_llrs(wf_group[s])
        # Maintain bit order consistent with encoder (b2,b1,b0)
        llrs.extend([l2, l1, l0])
    return np.array(llrs[:174], dtype=np.float64)


def _score_candidate_block(wf: Waterfall, start_symbol: int, base_bin: int) -> float:
    # Simple heuristic: average contrast across sync rows at this base
    # Use Costas positions for a crude SNR-like metric
    from .constants import FT8_COSTAS_PATTERN, NUM_SYNC
    score_sum = 0.0
    count = 0
    for m in range(NUM_SYNC):
        for k in range(LENGTH_SYNC):
            t = start_symbol + (SYNC_OFFSET * m) + k
            if t < 0 or t >= wf.mag.shape[0]:
                continue
            row = wf.mag[t, base_bin, :]
            sm = FT8_COSTAS_PATTERN[k]
            s = row[sm]
            left = row[sm - 1] if sm > 0 else row[sm]
            right = row[sm + 1] if sm < 7 else row[sm]
            score_sum += s - 0.5 * (left + right)
            count += 1
    return score_sum / max(count, 1)


def decode_block(samples: np.ndarray, sample_rate_hz: float, parity_path: Path) -> List[CandidateDecode]:
    wf = compute_waterfall_symbols(samples, sample_rate_hz, 0, num_symbols=NN)

    # For each base bin, compute sync hits independently and keep top few time candidates
    # Aggregate unique (time_symbol, base_bin) pairs with a composite score
    candidate_list: List[Tuple[int, int, float]] = []  # (time_symbol, base_bin, score)
    for base in range(wf.mag.shape[1]):
        wf_collapsed = wf.mag[:, base, :]  # [num_symbols, 8]
        hits = find_sync_positions(wf_collapsed, min_score=0.0)
        top_hits = hits[:5]
        for h in top_hits:
            score = _score_candidate_block(wf, h.time_symbol, base)
            candidate_list.append((h.time_symbol, base, score))

    # Sort global candidates by score and try top-N across all bases
    candidate_list.sort(key=lambda x: x[2], reverse=True)

    Mn, Nm = load_parity_from_file(parity_path)
    config = BeliefPropagationConfig(max_iterations=30, early_stop_no_improve=7, damping=0.2)

    results: List[CandidateDecode] = []
    for (t0, base, _score) in candidate_list[:40]:
        rows = _slice_data_symbol_rows(wf, t0, base)
        if rows.size == 0:
            continue
        llrs = llrs_from_waterfall(rows)
        errors, bits = min_sum_decode(llrs, Mn, Nm, config)
        payload_with_crc = bits[:91]
        bits_with_crc = np.concatenate([payload_with_crc[:77], payload_with_crc[77:91]])
        if crc14_check(bits_with_crc):
            results.append(CandidateDecode(start_symbol=t0, base_bin=base, ldpc_errors=errors, bits_with_crc=bits_with_crc))
    return results


