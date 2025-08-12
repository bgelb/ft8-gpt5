from ft8gpt import decode_wav


def test_decode_smoke(tmp_path):
    # Create a silent 15s mono WAV and ensure API does not crash
    import numpy as np
    import soundfile as sf

    sr = 12000
    x = np.zeros(sr * 15, dtype=np.float32)
    wav = tmp_path / "silent.wav"
    sf.write(str(wav), x, sr)

    results = decode_wav(str(wav))
    assert isinstance(results, list)

