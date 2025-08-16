import numpy as np
import pytest

from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn, mix_signals
from ft8gpt.sync import find_sync_candidates_stft


@pytest.mark.parametrize("num_sigs", [10, 20])
@pytest.mark.parametrize("snr_db", [-18, -16])
def test_coarse_recall_at_k(num_sigs, snr_db):
	sr = 12000.0
	rng = np.random.default_rng(1234)
	freqs = np.linspace(500.0, 3500.0, num_sigs, endpoint=False) + rng.uniform(-1.5, 1.5, size=num_sigs)
	signals = []
	metas = []
	for i in range(num_sigs):
		# Use a fixed standard callsign that our packer supports reliably
		x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, float(freqs[i]))
		x_awgn = apply_awgn(x, float(snr_db), rng)
		signals.append(x_awgn)
		metas.append({"f": float(freqs[i])})
	slot = mix_signals(signals, [0.0] * num_sigs)

	cands, nfft, hop = find_sync_candidates_stft(slot, sr, top_k=150)
	bin_hz = sr / nfft if nfft > 0 else 6.25
	K = min(80, len(cands))
	found = 0
	for m in metas:
		ok = False
		for c in cands[:K]:
			f_est = (c.base_bin + c.frac) * bin_hz
			if abs(f_est - m["f"]) <= 6.25:  # within one tone
				ok = True
				break
		found += int(ok)
	recall_at_k = found / num_sigs
	assert recall_at_k >= 0.6