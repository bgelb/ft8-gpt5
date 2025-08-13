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
    if wf_group.size == 0:
        return np.zeros(0, dtype=np.float64)
    s = wf_group
    llr0 = np.max(s[:, 4:8], axis=1) - np.max(s[:, 0:4], axis=1)
    llr1 = np.max(s[:, [2, 3, 6, 7]], axis=1) - np.max(s[:, [0, 1, 4, 5]], axis=1)
    llr2 = np.max(s[:, [1, 3, 5, 7]], axis=1) - np.max(s[:, [0, 2, 4, 6]], axis=1)
    llrs = np.stack([llr0, llr1, llr2], axis=1).reshape(-1)
    return llrs[:174].astype(np.float64)


def decode_block(samples: np.ndarray, sample_rate_hz: float, parity_path: Path) -> List[CandidateDecode]:
    wf = compute_waterfall_symbols(samples, sample_rate_hz, 0, num_symbols=NN)
    # Collapse base bins by taking max across frequency, simple heuristic for now
    wf_collapsed = wf.mag.max(axis=1)  # [num_symbols, 8]
    hits = find_sync_positions(wf_collapsed, min_score=0.0)
    Mn, Nm = load_parity_from_file(parity_path)

    config = BeliefPropagationConfig(max_iterations=12, early_stop_no_improve=3)
    results: List[CandidateDecode] = []
    for h in hits[:5]:
        start = max(0, h.time_symbol)
        # Build data symbol subarray while skipping syncs
        symbol_rows = []
        for k in range(ND):
            sym_idx = start + (k + (7 if k < 29 else 14))
            if sym_idx < 0 or sym_idx >= wf_collapsed.shape[0]:
                break
            symbol_rows.append(wf_collapsed[sym_idx])
        if len(symbol_rows) < ND:
            continue
        llrs = llrs_from_waterfall(np.stack(symbol_rows, axis=0))
        errors, bits = min_sum_decode(llrs, Mn, Nm, config)
        # Pack first 91 bits + compute and check CRC
        payload_with_crc = bits[:91]
        # Convert to explicit bits array (77 + 14)
        bits_with_crc = np.concatenate([payload_with_crc[:77], payload_with_crc[77:91]])
        # Note: crc14_check expects 77+14 MSB-first bits
        if crc14_check(bits_with_crc):
            results.append(CandidateDecode(start_symbol=start, ldpc_errors=errors, bits_with_crc=bits_with_crc))
    return results


