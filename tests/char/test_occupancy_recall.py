import numpy as np
import pytest

from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn, mix_signals
from ft8gpt.sync import find_sync_candidates_stft


@pytest.mark.parametrize("num_sigs", [10, 20, 40])
@pytest.mark.parametrize("snr_db", [-16])
def test_occupancy_recall_k(num_sigs, snr_db):
    sr = 12000.0
    rng = np.random.default_rng(99)
    freqs = np.linspace(500.0, 3500.0, num_sigs, endpoint=False) + rng.uniform(-1.5, 1.5, size=num_sigs)
    slot_signals = []
    truth = []
    for i in range(num_sigs):
        x, _ = make_clean_signal("K1AAA", "W9XYZ", "FN20", sr, freqs[i])
        xn = apply_awgn(x, snr_db, rng)
        slot_signals.append(xn)
        truth.append(freqs[i])
    slot = mix_signals(slot_signals, [0.0] * num_sigs)

    cands, nfft, hop = find_sync_candidates_stft(slot.astype(np.float64), sr, top_k=150)
    bin_hz = sr / nfft
    K = min(80, len(cands))
    found = 0
    for f in truth:
        ok = any(abs((c.base_bin + c.frac) * bin_hz - f) <= 6.25 for c in cands[:K])
        found += int(ok)
    recall = found / float(num_sigs)
    print("recall@K", recall)
    assert recall >= 0.4


@pytest.mark.slow
def test_occupancy_recall_k_80_sigs():
    sr = 12000.0
    snr_db = -16
    rng = np.random.default_rng(100)
    num_sigs = 80
    freqs = np.linspace(500.0, 3500.0, num_sigs, endpoint=False) + rng.uniform(-1.5, 1.5, size=num_sigs)
    slot_signals = []
    truth = []
    for i in range(num_sigs):
        x, _ = make_clean_signal("K1AAA", "W9XYZ", "FN20", sr, freqs[i])
        xn = apply_awgn(x, snr_db, rng)
        slot_signals.append(xn)
        truth.append(freqs[i])
    slot = mix_signals(slot_signals, [0.0] * num_sigs)
    cands, nfft, hop = find_sync_candidates_stft(slot.astype(np.float64), sr, top_k=150)
    bin_hz = sr / nfft
    K = min(80, len(cands))
    found = 0
    for f in truth:
        ok = any(abs((c.base_bin + c.frac) * bin_hz - f) <= 6.25 for c in cands[:K])
        found += int(ok)
    recall = found / float(num_sigs)
    print("recall@K", recall)
    assert recall >= 0.35