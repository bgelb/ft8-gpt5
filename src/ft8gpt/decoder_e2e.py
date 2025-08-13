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


def decode_block(samples: np.ndarray, sample_rate_hz: float, parity_path: Path) -> List[CandidateDecode]:
    wf = compute_waterfall_symbols(samples, sample_rate_hz, 0, num_symbols=NN)
    # Use full base-aware waterfall for sync search to improve accuracy
    hits = find_sync_positions(wf.mag, min_score=0.0)
    Mn, Nm = load_parity_from_file(parity_path)

    config = BeliefPropagationConfig(max_iterations=20, early_stop_no_improve=5)
    results: List[CandidateDecode] = []
    for h in hits[:10]:
        start = max(0, h.time_symbol)
        base_idx = getattr(h, "base_index", -1)
        # Build data symbol subarray while skipping syncs
        symbol_rows = []
        for k in range(ND):
            sym_idx = start + (k + (7 if k < 29 else 14))
            if sym_idx < 0 or sym_idx >= wf.mag.shape[0]:
                break
            if 0 <= base_idx < wf.mag.shape[1]:
                symbol_rows.append(wf.mag[sym_idx, base_idx, :])
            else:
                # Fallback: max across bases if base index not available
                symbol_rows.append(wf.mag[sym_idx].max(axis=0))
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


