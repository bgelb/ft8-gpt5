import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ft8gpt.ft8pack import pack_standard_payload
from ft8gpt.crc import crc14
from ft8gpt.ldpc_encode import load_generator_from_file, encode174_bits
from ft8gpt.synth import tones_from_codeword, synthesize_ft8_audio


def _have_make_and_cc() -> bool:
    return shutil.which("make") is not None and shutil.which("cc") is not None


def _build_ft8lib_decode() -> Path | None:
    """Attempt to build external/ft8_lib/decode_ft8. Return path or None if unavailable."""
    root = Path(__file__).resolve().parents[1]
    libdir = root / "external" / "ft8_lib"
    if not _have_make_and_cc():
        return None
    try:
        subprocess.run(["make", "-C", str(libdir), "decode_ft8"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    exe = libdir / "decode_ft8"
    return exe if exe.exists() else None


def test_ft8lib_decodes_reference_synthetic(tmp_path: Path):
    exe = _build_ft8lib_decode()
    if exe is None:
        # Skip if toolchain not available in CI
        import pytest
        pytest.skip("ft8_lib not available to build in this environment")

    # Build a known-good synthetic WAV that the C library decodes
    call_to, call_de, grid = "K1ABC", "W9XYZ", "FN20"
    a10 = pack_standard_payload(call_to, call_de, grid)
    bits77 = np.unpackbits(np.frombuffer(a10, dtype=np.uint8))[:77]
    c = crc14(bits77)
    crc_bits = np.array([(c >> i) & 1 for i in range(13, -1, -1)], dtype=np.uint8)
    a91 = np.concatenate([bits77, crc_bits])
    G = load_generator_from_file(Path(__file__).resolve().parents[1] / "external" / "ft8_lib" / "ft4_ft8_public" / "generator.dat")
    codeword = encode174_bits(a91, G)
    tones = tones_from_codeword(codeword)
    sr = 12000.0
    x = synthesize_ft8_audio(tones, sr)

    # Write 15s PCM16 mono WAV with signal centered in slot
    import wave
    wav_path = tmp_path / "ref_synth.wav"
    slot = np.zeros(int(15 * sr), dtype=np.int16)
    start = int(0.9 * sr)
    xi = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    slot[start : start + xi.size] = xi
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(slot.tobytes())

    # Cross-check with external decoder
    proc = subprocess.run([str(exe), str(wav_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout
    assert "Decoded 1 messages" in out or "Decoded 2 messages" in out
    assert "K1ABC" in out and "W9XYZ" in out and "FN20" in out


