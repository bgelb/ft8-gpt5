import numpy as np
import soundfile as sf

from ft8gpt.ft8pack import pack_standard_payload
from ft8gpt.crc import crc14
from ft8gpt.ldpc_encode import encode174_bits_systematic
from ft8gpt.synth import tones_from_codeword, synthesize_ft8_audio
from ft8gpt.decoder_e2e import decode_block
from ft8gpt.api import decode_wav


def test_e2e_synthetic_strong(tmp_path):
    # Build payload (77 bits) for a standard message
    a10 = pack_standard_payload("K1ABC", "W9XYZ", "FN20")
    # Append CRC14
    bits77 = np.unpackbits(np.frombuffer(a10, dtype=np.uint8))[:77]
    c = crc14(bits77)
    crc_bits = np.array([(c >> i) & 1 for i in range(13, -1, -1)], dtype=np.uint8)
    a91 = np.concatenate([bits77, crc_bits])

    # LDPC encode using generator
    codeword = encode174_bits_systematic(a91)

    tones = tones_from_codeword(codeword)
    sr = 12000.0
    x = synthesize_ft8_audio(tones, sr)
    wav = tmp_path / "strong.wav"
    sf.write(str(wav), x, int(sr))

    # Run through public API
    results = decode_wav(str(wav))
    assert isinstance(results, list)
    # Ensure we successfully decode at least one candidate from the strong synthetic sample
    assert len(results) > 0

