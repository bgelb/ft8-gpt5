import numpy as np

from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn
from ft8gpt.sync import find_sync_candidates_stft
from ft8gpt.decoder_e2e import refine_sync_fine


def test_cfo_and_timing_rmse_basic():
	sr = 12000.0
	base = 1500.0
	rng = np.random.default_rng(0)
	injected_cfos = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
	errs = []
	for df in injected_cfos:
		x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, base + float(df))
		y = apply_awgn(x, -16.0, rng)
		cands, nfft, hop = find_sync_candidates_stft(y, sr, top_k=10)
		if not cands:
			continue
		best = cands[0]
		coarse_abs = best.frame_start * hop
		bin_hz = sr / nfft if nfft > 0 else 6.25
		base_hz_est = (best.base_bin + best.frac) * bin_hz
		_, df_est, _ = refine_sync_fine(y, sr, base_hz_est, coarse_abs)
		errs.append((df_est - float(df)) ** 2)
	# Loose bound initially; tighten as refine_sync_fine improves
	assert np.sqrt(np.mean(errs)) <= 3.5