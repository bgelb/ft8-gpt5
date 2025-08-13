from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from .constants import NN, ND, LENGTH_SYNC, SYNC_OFFSET
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
    # wf_group shape [num_symbols, 8]
    llrs: List[float] = []
    for s in range(wf_group.shape[0]):
        l0, l1, l2 = extract_symbol_llrs(wf_group[s])
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

    config = BeliefPropagationConfig(max_iterations=30, early_stop_no_improve=8)
    results: List[CandidateDecode] = []
    for h in hits[:6]:
        start = max(0, h.time_symbol)
        # Compute energy per base across data symbol times
        num_bases = wf.mag.shape[1]
        data_times = list(range(start + 7, start + 36)) + list(range(start + 43, start + 72))
        valid_times = [t for t in data_times if 0 <= t < wf.mag.shape[0]]
        if len(valid_times) < 58:
            continue
        energy = np.zeros(num_bases, dtype=np.float64)
        for b in range(num_bases):
            # sum energy over all tones and all data symbols
            e = 0.0
            for t in valid_times:
                e += float(np.sum(wf.mag[t, b, :]))
            energy[b] = e
        top_idx = np.argsort(energy)[-10:][::-1]
        for base_idx in top_idx:
            symbol_rows = [wf.mag[t, base_idx, :] for t in valid_times]
            wf_group = np.stack(symbol_rows, axis=0)
            llrs = llrs_from_waterfall(wf_group)
            errors, bits = min_sum_decode(llrs, Mn, Nm, config)
            payload_with_crc = bits[:91]
            bits_with_crc = np.concatenate([payload_with_crc[:77], payload_with_crc[77:91]])
            if not crc14_check(bits_with_crc):
                continue
            results.append(CandidateDecode(start_symbol=start, ldpc_errors=errors, bits_with_crc=bits_with_crc))
            if len(results) >= 10:
                return results
    return results


