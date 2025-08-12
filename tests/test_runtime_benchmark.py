import pytest

from ft8gpt import decode_wav


@pytest.mark.slow
def test_runtime_placeholder(tmp_path, benchmark):
    # Placeholder runtime test on empty audio
    import numpy as np
    import soundfile as sf

    sr = 12000
    x = np.zeros(sr * 15, dtype=np.float32)
    wav = tmp_path / "silent.wav"
    sf.write(str(wav), x, sr)

    def run():
        return decode_wav(str(wav))

    results = benchmark(run)
    assert isinstance(results, list)


