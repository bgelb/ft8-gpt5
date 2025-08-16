import numpy as np
from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.sync import find_sync_candidates_stft
from ft8gpt.decoder_e2e import refine_sync_fine, coherent_symbol_energies, _llrs_from_linear_energies_gray_groups, _normalize_llrs_inplace
from ft8gpt.constants import gray_to_bits
from ft8gpt.char.channel import apply_awgn


def test_llr_stats_on_slot():
    sr = 12000.0
    rng = np.random.default_rng(42)
    # Moderate SNR for stable stats
    x, tones = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, 1500.0)
    y = apply_awgn(x, -14.0, rng)

    # Coarse + fine align
    cands, nfft, hop = find_sync_candidates_stft(y.astype(np.float64), sr, top_k=20)
    assert len(cands) > 0
    best = cands[0]
    coarse_abs = best.frame_start * hop
    bin_hz = sr / nfft
    base_hz_est = (best.base_bin + best.frac) * bin_hz
    off2, df_est, score = refine_sync_fine(y, sr, base_hz_est, coarse_abs)

    # Coherent energies for data symbols
    decim = max(1, int(round(sr / 200.0)))
    fs2 = sr / decim
    pos0_2 = int(round(coarse_abs / decim)) + int(off2)
    data_times_rel = list(range(7, 36)) + list(range(43, 72))
    E = coherent_symbol_energies(y[::decim].astype(np.complex128), fs2, pos0_2, df_est, data_times_rel)
    assert E.shape[0] == 58

    # Build ground-truth bits from tones
    truth_bits = []
    for k in data_times_rel:
        b2, b1, b0 = gray_to_bits(int(tones[k]))
        truth_bits.extend([b2, b1, b0])
    truth_bits = np.array(truth_bits[:174], dtype=np.uint8)

    # LLRs across slot
    llrs = []
    for row in E:
        l2, l1, l0 = _llrs_from_linear_energies_gray_groups(row)
        llrs.extend([l2, l1, l0])
    llrs = np.array(llrs[:174], dtype=np.float64)

    # Normalize and check variance near ~24 and separation positive
    _normalize_llrs_inplace(llrs)
    var = float(np.var(llrs))
    sep = float(np.mean((1.0 - 2.0 * truth_bits.astype(np.float64)) * llrs))
    print("LLR var, sep:", var, sep)
    assert 18.0 <= var <= 30.0
    assert sep > 0.45