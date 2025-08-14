from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from .constants import NN, ND, LENGTH_SYNC, SYNC_OFFSET, SYMBOL_PERIOD_S, NUM_SYNC
from .waterfall import compute_waterfall_symbols
from .tones import extract_symbol_llrs
from .ldpc import min_sum_decode, BeliefPropagationConfig
from .ldpc_tables_embedded import get_parity_matrices
from .crc import crc14_check
from .message_decode import unpack_standard_payload
from .constants import FT8_COSTAS_PATTERN
from .refine import refine_cfo_and_timing, extract_derotated_symbol_magnitudes


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


def _sync_score_for_base(wf_mag: np.ndarray, start: int, base_idx: int) -> float:
    # Sum tone energy at Costas pattern positions across three sync blocks
    num_symbols = wf_mag.shape[0]
    score = 0.0
    count = 0
    for block_start in (start, start + 36, start + 72):
        for k in range(LENGTH_SYNC):
            t = block_start + k
            if t < 0 or t >= num_symbols:
                continue
            tone = FT8_COSTAS_PATTERN[k]
            row = wf_mag[t, base_idx, :]
            score += float(row[tone])
            count += 1
    if count == 0:
        return -1e9
    return score / count


def decode_block(samples: np.ndarray, sample_rate_hz: float) -> List[CandidateDecode]:
    # Analyze the entire buffer at several fractional symbol alignments using STFT waterfall
    n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    if n_fft <= 0:
        return []

    Mn, Nm = get_parity_matrices()
    config = BeliefPropagationConfig(max_iterations=30, early_stop_no_improve=8)

    results: List[CandidateDecode] = []
    # Coarse timing sweep: 8 phases across one symbol
    step = max(1, n_fft // 8)
    for start_sample in range(0, min(n_fft, samples.size), step):
        total_symbols = max(0, int((samples.size - start_sample) // n_fft))
        if total_symbols < 79:
            continue
        wf = compute_waterfall_symbols(samples, sample_rate_hz, start_sample, num_symbols=total_symbols)
        num_symbols, num_bases, _ = wf.mag.shape
        window_len = LENGTH_SYNC + (NUM_SYNC - 1) * SYNC_OFFSET
        t_min = 0
        t_max = max(0, num_symbols - window_len)

        candidate_list: list[tuple[float, int, int]] = []  # (score, t, base_idx)
        for t in range(t_min, t_max + 1):
            for base_idx in range(num_bases):
                s = _sync_score_for_base(wf.mag, t, int(base_idx))
                candidate_list.append((s, t, int(base_idx)))

        candidate_list.sort(key=lambda x: x[0], reverse=True)
        candidate_list = candidate_list[:200]

        for score, start, base_idx in candidate_list:
            # Data symbol indices within this candidate (exclude Costas blocks)
            data_times = list(range(start + 7, start + 36)) + list(range(start + 43, start + 72))
            valid_times = [t for t in data_times if 0 <= t < num_symbols]
            if len(valid_times) < 58:
                continue

            # Try both integer-bin and +0.5-bin initial alignments; refine coherently then derotate
            for initial_frac, mag_cube in ((0.0, wf.mag), (0.5, wf.mag_half)):
                if base_idx >= mag_cube.shape[1]:
                    continue
                refine_res, n_fft_ref, hop_ref = refine_cfo_and_timing(
                    samples,
                    sample_rate_hz,
                    start_sample,
                    start_symbol=start,
                    base_bin=base_idx,
                    initial_frac=initial_frac,
                )
                # Use refined CFO/timing to compute derotated per-symbol magnitudes for data symbols
                mags = extract_derotated_symbol_magnitudes(
                    samples,
                    sample_rate_hz,
                    start_sample,
                    valid_times,
                    base_idx,
                    refine_res.frac_bin,
                    n_fft_ref,
                    refine_res.delta_frames,
                )
                llrs = llrs_from_waterfall(mags)
                errors, bits = min_sum_decode(llrs, Mn, Nm, config)
                payload_with_crc = bits[:91]
                bits_with_crc = np.concatenate([payload_with_crc[:77], payload_with_crc[77:91]])
                if not crc14_check(bits_with_crc):
                    continue
                results.append(
                    CandidateDecode(start_symbol=start, ldpc_errors=errors, bits_with_crc=bits_with_crc)
                )
                if len(results) >= 10:
                    return results
    return results


