import numpy as np
import pytest

from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn, mix_signals
from ft8gpt.sync import find_sync_candidates_stft


def _make_callsign(i: int) -> str:
    # Use letter-digit-letters (length 5) to satisfy minimal packer A0XYZ pattern
    a = chr(ord('A') + (i % 26))
    b = chr(ord('A') + ((i // 26) % 26))
    c = chr(ord('A') + ((i // (26 * 26)) % 26))
    return f"K1{a}{b}{c}"


@pytest.mark.parametrize("num_sigs", [10, 20, 40])
@pytest.mark.parametrize("snr_db", [-18, -16, -14])
def test_coarse_recall_at_k(num_sigs, snr_db):
    sr = 12000.0
    rng = np.random.default_rng(1234)
    freqs = np.linspace(500.0, 3500.0, num_sigs, endpoint=False) + rng.uniform(-1.5, 1.5, size=num_sigs)
    signals = []
    metas = []
    for i in range(num_sigs):
        call = _make_callsign(i)
        x, tones = make_clean_signal(call, "W9XYZ", "FN20", sr, freqs[i])
        x_awgn = apply_awgn(x, snr_db, rng)
        signals.append(x_awgn)
        metas.append({"f": freqs[i]})
    slot = mix_signals(signals, [0.0] * num_sigs)

    cands, nfft, hop = find_sync_candidates_stft(slot.astype(np.float64), sr, top_k=150)
    bin_hz = sr / nfft if nfft > 0 else 6.25
    K = min(80, len(cands))
    found = 0
    for m in metas:
        ok = False
        for c in cands[:K]:
            f_est = (c.base_bin + c.frac) * bin_hz
            if abs(f_est - m["f"]) <= 6.25:
                ok = True
                break
        found += int(ok)
    recall_at_k = found / num_sigs
    # Do not assert a hard bound yet; print for visibility and ensure non-zero
    print("recall@K", recall_at_k)
    assert recall_at_k >= 0.4