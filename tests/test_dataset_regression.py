from __future__ import annotations
import glob
import os
import time
from ft8dec.decoder import decode_wav_file

DATA_DIR = os.environ.get("FT8_WAV_DIR", "data/ft8_wav")


def iter_wavs():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))


def test_e2e_strong_sample_runtime_and_crc_smoke():
    # Pick one file; later we will validate CRC and text after decoder implemented
    files = iter_wavs()
    assert files, "dataset not fetched; run scripts/fetch_dataset.sh"
    path = files[0]
    t0 = time.perf_counter()
    decodes = decode_wav_file(path)
    dt = time.perf_counter() - t0
    # runtime guard (loose initially)
    assert dt < 10.0
    # shape
    assert isinstance(decodes, list)


def test_dataset_regression_counts_placeholder():
    # Placeholder expectations: decoder returns zero decodes
    files = iter_wavs()
    assert files, "dataset not fetched; run scripts/fetch_dataset.sh"
    total_decodes = 0
    for path in files:
        total_decodes += len(decode_wav_file(path))
    # Expect zero initially; will raise as implementation improves
    assert total_decodes == 0
