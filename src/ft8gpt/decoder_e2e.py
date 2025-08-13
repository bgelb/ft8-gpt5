from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from .constants import NN, ND, LENGTH_SYNC, SYNC_OFFSET, FT8_GRAY_MAP
from .waterfall import compute_waterfall_symbols
from .sync import find_sync_positions
from .tones import extract_symbol_llrs
from .ldpc import min_sum_decode, BeliefPropagationConfig
from .ldpc_tables import load_parity_from_file
from .crc import crc14_check
from .message_decode import unpack_standard_payload


@dataclass(frozen=True)
class CandidateDecode:
    start_symbol: int
    ldpc_errors: int
    bits_with_crc: np.ndarray


def llrs_from_waterfall(wf_group: np.ndarray) -> np.ndarray:
    # wf_group shape [num_symbols, 8] in tone-index order; reorder to Gray-bit order
    llrs: List[float] = []
    for s in range(wf_group.shape[0]):
        row_gray = wf_group[s][list(FT8_GRAY_MAP)]
        l0, l1, l2 = extract_symbol_llrs(row_gray)
        llrs.extend([l0, l1, l2])
    return np.array(llrs[: 174], dtype=np.float64)


def _base_sync_score(wf_mag: np.ndarray, start_symbol: int, base_idx: int) -> float:
    # wf_mag shape [num_symbols, num_bases, 8]
    num_symbols = wf_mag.shape[0]
    score_sum = 0.0
    count = 0
    for m in range(3):
        for k in range(LENGTH_SYNC):
            t = start_symbol + m * SYNC_OFFSET + k
            if t < 0 or t >= num_symbols:
                continue
            row = wf_mag[t, base_idx, :]
            sm = (3, 1, 4, 0, 6, 5, 2)[k]
            s = row[sm]
            if sm > 0:
                score_sum += s - row[sm - 1]
                count += 1
            if sm < 7:
                score_sum += s - row[sm + 1]
                count += 1
    if count == 0:
        return -1e9
    return score_sum / count


def decode_block(samples: np.ndarray, sample_rate_hz: float, parity_path: Path) -> List[CandidateDecode]:
    wf = compute_waterfall_symbols(samples, sample_rate_hz, 0, num_symbols=NN)
    # Collapse base bins for sync search
    wf_collapsed = wf.mag.max(axis=1)  # [num_symbols, 8]
    hits = find_sync_positions(wf_collapsed, min_score=0.0)
    Mn, Nm = load_parity_from_file(parity_path)

    config = BeliefPropagationConfig(max_iterations=20, early_stop_no_improve=5)
    results: List[CandidateDecode] = []
    for h in hits[:8]:
        start = max(0, h.time_symbol)
        # Rank base bins by sync score
        num_bases = wf.mag.shape[1]
        scores = np.array([_base_sync_score(wf.mag, start, b) for b in range(num_bases)], dtype=np.float64)
        top_idx = np.argsort(scores)[-8:][::-1]
        for base_idx in top_idx:
            # Assemble data rows skipping syncs using known layout: S7 D29 S7 D29 S7
            symbol_rows = []
            # First D29 starts at 7, second starts at 7+29+7 = 43
            for k in range(29):
                t = start + 7 + k
                if t < 0 or t >= wf.mag.shape[0]:
                    symbol_rows = []
                    break
                symbol_rows.append(wf.mag[t, base_idx, :])
            if not symbol_rows:
                continue
            for k in range(29):
                t = start + 43 + k
                if t < 0 or t >= wf.mag.shape[0]:
                    symbol_rows = []
                    break
                symbol_rows.append(wf.mag[t, base_idx, :])
            if len(symbol_rows) != 58:
                continue
            llrs = llrs_from_waterfall(np.stack(symbol_rows, axis=0))
            errors, bits = min_sum_decode(llrs, Mn, Nm, config)
            payload_with_crc = bits[:91]
            bits_with_crc = np.concatenate([payload_with_crc[:77], payload_with_crc[77:91]])
            if not crc14_check(bits_with_crc):
                continue
            # Reject trivial all-zero payloads
            if np.all(bits_with_crc[:77] == 0):
                continue
            results.append(CandidateDecode(start_symbol=start, ldpc_errors=errors, bits_with_crc=bits_with_crc))
    return results


