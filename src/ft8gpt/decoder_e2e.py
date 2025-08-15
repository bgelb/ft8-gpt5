from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .constants import (
    NN,
    ND,
    LENGTH_SYNC,
    SYNC_OFFSET,
    SYMBOL_PERIOD_S,
    NUM_SYNC,
    FT8_COSTAS_PATTERN,
    TONE_SPACING_HZ,
    FT8_GRAY_MAP,
)
from .waterfall import compute_waterfall_symbols
from .tones import extract_symbol_llrs
from .ldpc import min_sum_decode, BeliefPropagationConfig
from .ldpc_tables_embedded import get_parity_matrices
from .crc import crc14_check
from .message_decode import unpack_standard_payload
from scipy.signal import decimate, get_window
from .sync import find_sync_candidates_stft
from .ldpc_encode import get_encoder_structures


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
        # Encoder packs bits per symbol as (b2,b1,b0); order LLRs accordingly
        llrs.extend([l2, l1, l0])
    return np.array(llrs[: 174], dtype=np.float64)


def _downmix_and_decimate(
    samples: np.ndarray,
    sample_rate_hz: float,
    carrier_hz: float,
    decim_factor: int,
) -> Tuple[np.ndarray, float]:
    """Mix real samples by exp(-j2π carrier t) and decimate by an integer factor.

    Returns (complex_baseband_decimated, new_sample_rate).
    """
    n = np.arange(samples.size, dtype=np.float64)
    phi = -2.0 * np.pi * carrier_hz * (n / sample_rate_hz)
    osc = np.cos(phi) + 1j * np.sin(phi)
    y = samples.astype(np.float64) * osc
    # Zero-phase IIR decimation; sample_rate expected 12000 Hz, factor 60 -> 200 Hz
    y_dec = decimate(y, decim_factor, ftype='iir', zero_phase=True)
    return y_dec.astype(np.complex128, copy=False), float(sample_rate_hz / decim_factor)


def _precompute_ctones(symbol_len: int, fs_hz: float) -> np.ndarray:
    """Return matrix C of shape [symbol_len, 8] with per-tone complex exponentials
    exp(-j2π tone*Δf/fs * n) for tone=0..7, Δf=TONE_SPACING_HZ.
    """
    n = np.arange(symbol_len, dtype=np.float64)
    tones = np.arange(8, dtype=np.float64) * TONE_SPACING_HZ
    phase = -2.0 * np.pi * (np.outer(n, tones) / fs_hz)
    return np.cos(phase) - 1j * np.sin(phase)


def refine_sync_fine(
    samples: np.ndarray,
    sample_rate_hz: float,
    base_freq_hz: float,
    coarse_abs_start_sample: int,
    time_search_ms: int = 40,
    time_step_ms: int = 5,
    freq_search_hz: float = 2.5,
    freq_step_hz: float = 0.5,
) -> Tuple[int, float, float]:
    """Refine time (±time_search in steps) and frequency (±freq_search in steps) using
    quasi-coherent cross-correlation against the Costas pattern.

    Returns (best_offset_samples_200Hz, best_df_hz, best_score).
    """
    # Downmix to base frequency and decimate to 200 Hz (assumes 12000/60)
    decim = max(1, int(round(sample_rate_hz / 200.0)))
    y2, fs2 = _downmix_and_decimate(samples, sample_rate_hz, base_freq_hz, decim)
    sym_len = int(round(SYMBOL_PERIOD_S * fs2))  # expect 32
    if sym_len <= 0:
        return 0, 0.0, -1e30

    # Absolute downsampled index of coarse start
    pos0_2 = int(round(coarse_abs_start_sample / decim))

    # Precompute tone exponentials and window
    ctones = _precompute_ctones(sym_len, fs2)
    win = get_window('hann', sym_len, fftbins=True).astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))

    # Build frequency tweak factors for each Δf
    n = np.arange(sym_len, dtype=np.float64)
    df_vals = np.arange(-freq_search_hz, freq_search_hz + 1e-9, freq_step_hz, dtype=np.float64)
    ctweaks = [np.cos(-2.0 * np.pi * df * n / fs2) - 1j * np.sin(-2.0 * np.pi * df * n / fs2) for df in df_vals]

    # Time offsets in downsampled samples (5 ms per step at 200 Hz)
    tsamp = int(round(time_step_ms * fs2 / 1000.0))
    max_tsamp = int(round(time_search_ms * fs2 / 1000.0))
    time_offsets = list(range(-max_tsamp, max_tsamp + 1, tsamp))

    best_score = -1e300
    best_df = 0.0
    best_off = 0

    sync_starts = (0, 36, 72)
    for df, ctweak in zip(df_vals, ctweaks):
        for off in time_offsets:
            T = 0.0
            for m in sync_starts:
                for k in range(LENGTH_SYNC):
                    tone = FT8_COSTAS_PATTERN[k]
                    pos = pos0_2 + off + (m + k) * sym_len
                    if pos < 0 or pos + sym_len > y2.size:
                        continue
                    seg = y2[pos: pos + sym_len]
                    # demod at tone freq with df tweak
                    z = np.vdot(win * seg * ctweak, ctones[:, tone])
                    T += float((z.real * z.real + z.imag * z.imag))
            if T > best_score:
                best_score = T
                best_df = float(df)
                best_off = int(off)

    return best_off, best_df, best_score


def coherent_symbol_energies(
    y2: np.ndarray,
    fs2: float,
    pos0_2: int,
    df_hz: float,
    symbol_indices: List[int],
) -> np.ndarray:
    """Return energies array of shape [len(symbol_indices), 8] via coherent demod at refined CFO.
    """
    sym_len = int(round(SYMBOL_PERIOD_S * fs2))
    ctones = _precompute_ctones(sym_len, fs2)
    n = np.arange(sym_len, dtype=np.float64)
    ctweak = np.cos(-2.0 * np.pi * df_hz * n / fs2) - 1j * np.sin(-2.0 * np.pi * df_hz * n / fs2)
    win = get_window('hann', sym_len, fftbins=True).astype(np.float64)
    win /= np.sqrt(np.sum(win ** 2))

    E = np.zeros((len(symbol_indices), 8), dtype=np.float64)
    for i, t in enumerate(symbol_indices):
        pos = pos0_2 + t * sym_len
        if pos < 0 or pos + sym_len > y2.size:
            continue
        seg = y2[pos: pos + sym_len]
        segw = win * seg * ctweak
        # energies for 8 tones
        for tone in range(8):
            z = np.vdot(segw, ctones[:, tone])
            E[i, tone] = float((z.real * z.real + z.imag * z.imag))
    return E


def count_costas_matches(E_sync: np.ndarray) -> int:
    """Count how many of the 21 sync symbols have argmax matching the Costas tone."""
    matches = 0
    # E_sync is shape [21, 8] ordered by actual sync times
    for i, k in enumerate(range(LENGTH_SYNC)):
        if int(np.argmax(E_sync[i])) == FT8_COSTAS_PATTERN[k]:
            matches += 1
    for i, k in enumerate(range(LENGTH_SYNC)):
        if int(np.argmax(E_sync[LENGTH_SYNC + i])) == FT8_COSTAS_PATTERN[k]:
            matches += 1
    for i, k in enumerate(range(LENGTH_SYNC)):
        if int(np.argmax(E_sync[2 * LENGTH_SYNC + i])) == FT8_COSTAS_PATTERN[k]:
            matches += 1
    return matches


def _llrs_from_linear_energies_gray_groups(energies_row: np.ndarray) -> Tuple[float, float, float]:
    """Compute per-symbol LLRs from linear tone energies using Gray group log-sum ratios.

    LLR(bit=0) - LLR(bit=1) = log(sum energies where bit=0) - log(sum energies where bit=1).
    """
    # Map to Gray order index j -> energies of Gray-coded symbol j
    s_gray = np.empty(8, dtype=np.float64)
    for j in range(8):
        tone = FT8_GRAY_MAP[j]
        s_gray[j] = float(max(energies_row[tone], 1e-20))
    # Group indices for each bit
    g0 = (0, 1, 2, 3)
    g1 = (4, 5, 6, 7)
    g_b1_0 = (0, 1, 4, 5)
    g_b1_1 = (2, 3, 6, 7)
    g_b0_0 = (0, 2, 4, 6)
    g_b0_1 = (1, 3, 5, 7)
    import math
    llr2 = math.log(sum(s_gray[i] for i in g0)) - math.log(sum(s_gray[i] for i in g1))
    llr1 = math.log(sum(s_gray[i] for i in g_b1_0)) - math.log(sum(s_gray[i] for i in g_b1_1))
    llr0 = math.log(sum(s_gray[i] for i in g_b0_0)) - math.log(sum(s_gray[i] for i in g_b0_1))
    # Return in (b2,b1,b0) order consistent with previous usage
    return llr2, llr1, llr0


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
    # Analyze the entire buffer using STFT-based Costas matched filter to find coarse candidates
    n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
    if n_fft <= 0:
        return []

    Mn, Nm = get_parity_matrices()
    # Stronger LDPC settings improve convergence with coherent LLRs
    config = BeliefPropagationConfig(max_iterations=80, early_stop_no_improve=25)

    # Compute codeword->payload mapping once based on embedded parity structure
    Br_inv, Hrest, rest_cols, piv_cols = get_encoder_structures()

    results: List[CandidateDecode] = []

    # Run STFT-based candidate search across the whole buffer
    candidates, stft_nfft, hop = find_sync_candidates_stft(samples, sample_rate_hz, top_k=200)

    # Process top candidates with fine refinement and coherent demod
    for cand in candidates:
        # Coarse absolute sample index corresponding to the first Costas symbol
        coarse_abs_start = cand.frame_start * hop
        if coarse_abs_start < 0:
            continue

        # Bin spacing and base frequency for both frac hypotheses (cand.frac is already 0.0 or 0.5)
        bin_hz = sample_rate_hz / stft_nfft if stft_nfft > 0 else TONE_SPACING_HZ
        base_idx = int(cand.base_bin)
        frac = float(cand.frac)
        base_freq_hz = (base_idx + frac) * bin_hz

        # Refine time and frequency via Costas correlation at 200 Hz
        best_off_2, df_hz, corr_score = refine_sync_fine(
            samples, sample_rate_hz, base_freq_hz, coarse_abs_start
        )

        # Downmix/decimate once more to reuse in demod
        decim = max(1, int(round(sample_rate_hz / 200.0)))
        y2, fs2 = _downmix_and_decimate(samples, sample_rate_hz, base_freq_hz, decim)
        sym_len2 = int(round(SYMBOL_PERIOD_S * fs2))
        pos0_2 = int(round(coarse_abs_start / decim)) + best_off_2
        if sym_len2 <= 0:
            continue

        # Build symbol index lists
        sync_times = (
            list(range(0, 7)) +
            list(range(36, 43)) +
            list(range(72, 79))
        )
        # Use times RELATIVE to candidate start for coherent demod
        data_times_rel = list(range(7, 36)) + list(range(43, 72))

        # Evaluate Costas match quality gate
        # Compute sync energies at refined alignment
        sync_E = coherent_symbol_energies(y2, fs2, pos0_2, df_hz, sync_times)
        if sync_E.shape[0] != 21:
            continue
        # Require a strong Costas lock for proceeding (strong synthetic should easily satisfy)
        if count_costas_matches(sync_E) < 18:
            continue

        # Coherent demod for data symbols
        E = coherent_symbol_energies(y2, fs2, pos0_2, df_hz, data_times_rel)
        # Convert linear energies to LLRs via Gray group log-sum ratios
        llrs: List[float] = []
        for row in E:
            l2, l1, l0 = _llrs_from_linear_energies_gray_groups(row)
            # Pack LLRs per symbol in (b2,b1,b0) order to match bit significance
            llrs.extend([l2, l1, l0])
        llrs_arr = np.array(llrs[:174], dtype=np.float64)
        # Empirical scaling for improved LDPC convergence with coherent LLRs
        llrs_arr *= 2.5

        errors, bits = min_sum_decode(llrs_arr, Mn, Nm, config)
        # Map decoded codeword bits to payload+CRC using encoder-derived systematic mapping
        Br_inv, Hrest, rest_cols, piv_cols = get_encoder_structures()
        a91 = bits[np.array(rest_cols, dtype=np.int64)].astype(np.uint8)
        bits_with_crc = np.concatenate([a91[:77], a91[77:91]])
        if not crc14_check(bits_with_crc):
            continue
        results.append(
            CandidateDecode(start_symbol=0, ldpc_errors=errors, bits_with_crc=bits_with_crc)
        )
        if len(results) >= 10:
            return results
    return results


