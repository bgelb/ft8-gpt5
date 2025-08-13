from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from .constants import NN, ND, LENGTH_SYNC, SYNC_OFFSET, NUM_SYNC
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


def _data_symbol_offsets() -> List[int]:
    # Compute frame symbol indices [0..NN-1] that carry data (skip sync blocks)
    sync_idxs = set()
    for m in range(NUM_SYNC):
        start = m * SYNC_OFFSET
        for k in range(LENGTH_SYNC):
            sync_idxs.add(start + k)
    return [i for i in range(NN) if i not in sync_idxs]


def decode_block(samples: np.ndarray, sample_rate_hz: float, parity_path: Path) -> List[CandidateDecode]:
    wf = compute_waterfall_symbols(samples, sample_rate_hz, 0, num_symbols=NN)
    # Collapse base bins by taking max across frequency, simple heuristic for now
    wf_collapsed = wf.mag.max(axis=1)  # [num_symbols, 8]
    hits = find_sync_positions(wf_collapsed, min_score=0.0)
    Mn, Nm = load_parity_from_file(parity_path)

    data_offsets = _data_symbol_offsets()
    assert len(data_offsets) == ND

    config = BeliefPropagationConfig(max_iterations=20, early_stop_no_improve=5)
    results: List[CandidateDecode] = []
    for h in hits[:10]:
        start = max(0, h.time_symbol)
        # Build data symbol subarray while skipping syncs
        symbol_rows = []
        for k in range(ND):
            sym_idx = start + data_offsets[k]
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


