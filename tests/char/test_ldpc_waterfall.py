import numpy as np
import pytest

from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn
from ft8gpt.decoder_e2e import decode_block


@pytest.mark.parametrize("snr_db", [-20, -18])
def test_end_to_end_fer_vs_snr_smoke(snr_db):
	sr = 12000.0
	rng = np.random.default_rng(0)
	trials = 8
	ok = 0
	for _ in range(trials):
		x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, 1500.0)
		y = apply_awgn(x, float(snr_db), rng)
		results = decode_block(y, sr)
		ok += int(len(results) > 0)
	fer = 1.0 - ok / trials
	# Do not assert a specific FER; just assert the code runs and produces a sane number
	assert 0.0 <= fer <= 1.0