import numpy as np
from ft8gpt.decoder_e2e import refine_sync_fine
from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.sync import find_sync_candidates_stft


def test_cfo_and_timing_rmse():
    sr = 12000.0
    base = 1500.0
    injected_cfos = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    errs = []
    for df in injected_cfos:
        x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, base + float(df))
        cands, nfft, hop = find_sync_candidates_stft(x.astype(np.float64), sr, top_k=10)
        assert len(cands) > 0, "Coarse search found no candidates"
        best = cands[0]
        coarse_abs = best.frame_start * hop
        bin_hz = sr / nfft
        base_hz_est = (best.base_bin + best.frac) * bin_hz
        off2, df_est, score = refine_sync_fine(x, sr, base_hz_est, coarse_abs)
        # Compare to injected CFO relative to candidate's base estimate
        errs.append((df_est - float(df)) ** 2)
    rmse = float(np.sqrt(np.mean(np.array(errs, dtype=np.float64))))
    print("CFO RMSE:", rmse)
    assert rmse <= 2.8