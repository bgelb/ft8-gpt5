from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .constants import NN, ND, LENGTH_SYNC, SYNC_OFFSET, SYMBOL_PERIOD_S
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


def _compute_goertzel_group(samples: np.ndarray, fs: float, n_fft: int, start_symbol: int, base_freq_hz: float) -> np.ndarray:
    """Compute per-data-symbol, 8-tone magnitudes using Goertzel centered at base_freq_hz.
    Returns array of shape [ND, 8] in log magnitude domain.
    """
    symbol_len = n_fft
    # Energy-normalized Hann
    n = np.arange(symbol_len)
    win = 0.5 * (1 - np.cos(2 * np.pi * n / (symbol_len - 1)))
    win = win.astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))

    out = np.zeros((ND, 8), dtype=np.float64)
    for k in range(ND):
        sym_idx = start_symbol + (k + (7 if k < 29 else 14))
        i0 = sym_idx * symbol_len
        seg = samples[i0 : i0 + symbol_len]
        if seg.size < symbol_len:
            seg = np.pad(seg, (0, symbol_len - seg.size))
        xw = seg * win
        for tone in range(8):
            f = base_freq_hz + tone * (fs / n_fft)
            # Goertzel
            coeff = 2 * np.cos(2 * np.pi * f / fs)
            s_prev = 0.0
            s_prev2 = 0.0
            for xi in xw:
                s = xi + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            power = max(s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2, 1e-12)
            out[k, tone] = 20.0 * np.log10(power)
    return out


def _assemble_llrs_interleaved(mags: np.ndarray, perm: Tuple[int, int, int]) -> np.ndarray:
    """Build length-174 LLR vector in codeword bit order using 3-substream mapping.
    perm maps (b2,b1,b0) indices onto positions [s, s+58, s+116] respectively.
    Scale LLRs from dB differences to natural log units.
    """
    DB_TO_LN = np.log(10.0) / 20.0  # convert dB to natural log of amplitude ratio
    llr = np.zeros(174, dtype=np.float64)
    for s in range(min(ND, mags.shape[0])):
        l2, l1, l0 = extract_symbol_llrs(mags[s])
        triple = (l2 * DB_TO_LN, l1 * DB_TO_LN, l0 * DB_TO_LN)
        llr[s] = triple[perm[0]]
        llr[s + 58] = triple[perm[1]]
        llr[s + 116] = triple[perm[2]]
    return llr


def decode_block(samples: np.ndarray, sample_rate_hz: float, parity_path: Path) -> List[CandidateDecode]:
    # Compute over the entire buffer so we can slide frame start in symbol steps
    n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    total_symbols = max(0, samples.size // n_fft)
    wf = compute_waterfall_symbols(samples, sample_rate_hz, 0, num_symbols=total_symbols)

    # For each base bin, compute sync hits independently and keep top few time candidates
    # Aggregate unique (time_symbol, base_bin) pairs with a composite score
    candidate_list: List[Tuple[int, int, float]] = []  # (time_symbol, base_bin, score)
    frame_span = (SYNC_OFFSET * 2) + LENGTH_SYNC  # total span of sync blocks ~ 36*2+7 = 79
    for base in range(wf.mag.shape[1]):
        wf_collapsed = wf.mag[:, base, :]  # [num_symbols, 8]
        hits = find_sync_positions(wf_collapsed, min_score=0.0)
        # Filter to valid start times fully inside buffer
        hits = [h for h in hits if (h.time_symbol >= 0 and (h.time_symbol + frame_span) <= (wf.mag.shape[0] - 1))]
        top_hits = hits[:5]
        for h in top_hits:
            score = _score_candidate_block(wf, h.time_symbol, base)
            candidate_list.append((h.time_symbol, base, score))

    # Sort global candidates by score and try top-N across all bases
    candidate_list.sort(key=lambda x: x[2], reverse=True)

    Mn, Nm = load_parity_from_file(parity_path)
    config = BeliefPropagationConfig(max_iterations=40, early_stop_no_improve=10, damping=0.2)

    results: List[CandidateDecode] = []
    bin_hz = sample_rate_hz / wf.n_fft
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    for (t0, base, _score) in candidate_list[:80]:
        # Try small fractional offsets around the base frequency using Goertzel magnitudes
        base_freq_hz = base * bin_hz
        for delta_hz in (-3.125, 0.0, 3.125):
            mags = _compute_goertzel_group(samples, sample_rate_hz, wf.n_fft, t0, base_freq_hz + delta_hz)
            if mags.size == 0:
                continue
            for perm in perms:
                llrs = _assemble_llrs_interleaved(mags, perm)
                errors, bits = min_sum_decode(llrs, Mn, Nm, config)
                payload_with_crc = bits[:91]
                bits_with_crc = np.concatenate([payload_with_crc[:77], payload_with_crc[77:91]])
                if crc14_check(bits_with_crc):
                    results.append(CandidateDecode(start_symbol=t0, base_bin=base, ldpc_errors=errors, bits_with_crc=bits_with_crc))
                    break
            if results and results[-1].start_symbol == t0 and results[-1].base_bin == base:
                break
    return results


