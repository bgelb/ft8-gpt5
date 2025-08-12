import numpy as np
import soundfile as sf
from pathlib import Path

from ft8gpt.decoder_e2e import decode_block


def test_e2e_strong_synthetic(tmp_path):
    # Create a 15s buffer at 12000 Hz with silence; this is a placeholder until tone synth is added
    sr = 12000
    x = np.zeros(sr * 15, dtype=np.float32)
    wav = tmp_path / "silence.wav"
    sf.write(str(wav), x, sr)

    parity = Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "ft4_ft8_public" / "parity.dat"
    results = decode_block(x, float(sr), parity)
    # No decodes on silence, but pipeline should run without error
    assert isinstance(results, list)

