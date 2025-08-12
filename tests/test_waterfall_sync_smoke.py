import numpy as np

from ft8gpt.waterfall import compute_waterfall_symbols
from ft8gpt.sync import find_sync_positions


def test_waterfall_and_sync_smoke():
    fs = 12000.0
    duration_s = 1.0
    t = np.arange(int(fs * duration_s)) / fs
    x = np.zeros_like(t)
    wf = compute_waterfall_symbols(x, fs, 0, num_symbols=20)
    hits = find_sync_positions(wf.mag.mean(axis=1))  # collapse base bins to single row
    assert isinstance(hits, list)

