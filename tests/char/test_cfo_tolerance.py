import numpy as np
import pytest

from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn
from ft8gpt.decoder_e2e import decode_block


@pytest.mark.parametrize("snr_db", [-16])
def test_cfo_tolerance_decode_rate(snr_db):
    sr = 12000.0
    base = 1500.0
    rng = np.random.default_rng(0)
    cfos = np.array([-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5], dtype=float)
    ok = 0
    for df in cfos:
        x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, base + float(df))
        y = apply_awgn(x, snr_db, rng)
        res = decode_block(y, sr)
        ok += int(len(res) > 0)
    rate = ok / float(cfos.size)
    print("CFO decode rate:", rate)
    assert rate >= 0.5