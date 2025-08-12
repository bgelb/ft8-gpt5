from __future__ import annotations
from typing import List
import numpy as np
from .types import Decode
from .io import read_wav_mono
from .sync import stft_waterfall, costas_sync_score, extract_8tone_mags
from .demod import symbol_llrs
from .ldpc import bp_decode
from .constants import FT8Constants
from .crc14 import crc14, POLY as CRC_POLY


def decode_wav_file(path: str) -> List[Decode]:
    """Decode an FT8 WAV file and return decodes.

    Minimal pipeline: waterfall -> sync score -> single candidate -> LLRs -> LDPC -> CRC check.
    """
    samples, fs_hz = read_wav_mono(path)
    psd, hop, bin_hz, time_osr = stft_waterfall(samples, fs_hz)
    if psd.shape[0] < 100:
        return []
    const = FT8Constants()
    # Determine bin step for 6.25 Hz spacing
    bin_step = max(1, int(round(const.tone_spacing_hz / bin_hz)))
    # Scan a range of base bins around ~reference frequency (search wide)
    center_hz = 1200.0
    center_bin = int(round(center_hz / bin_hz))
    best = None
    for delta in range(-400, 401, 1):
        base_bin = center_bin + delta
        scores = costas_sync_score(psd, bin_step, base_bin)
        t0 = int(np.argmax(scores)) if len(scores) > 0 else 0
        score = float(scores[t0]) if len(scores) > 0 else 0.0
        if best is None or score > best[0]:
            best = (score, t0, base_bin)
    if best is None or best[0] <= 0:
        return []
    score, start, base_bin = best
    # Collect LLRs for 58 data symbols using per-symbol 8-tone magnitudes
    llrs = []
    def add_llrs(sym_idx: int):
        if 0 <= sym_idx < psd.shape[0]:
            mags = extract_8tone_mags(psd[sym_idx], base_bin, bin_step)
            if mags is None:
                llrs.append(np.zeros(3, dtype=np.float32))
            else:
                llrs.append(symbol_llrs(mags))
        else:
            llrs.append(np.zeros(3, dtype=np.float32))
    for k in range(29):
        add_llrs(start + 7 + k)
    for k in range(29):
        add_llrs(start + 43 + k)
    if len(llrs) != 58:
        return []
    log174 = np.concatenate(llrs, axis=0)
    if np.max(np.abs(log174)) > 0:
        log174 = log174 / (np.max(np.abs(log174)) + 1e-9)
    bits, unsat = bp_decode(log174, max_iters=250)
    if unsat != 0:
        return []
    a91 = bits[:91]
    tmp_bits = np.zeros(96, dtype=np.uint8)
    tmp_bits[:91] = a91
    tmp_bits[77:82] = 0
    crc_val = crc14(tmp_bits[:82])
    recv_crc = 0
    for i in range(14):
        recv_crc = (recv_crc << 1) | int(a91[77 + i])
    crc_ok = (crc_val == recv_crc)
    if not crc_ok:
        return []
    dec = Decode(
        start_time_s=float(start)*const.symbol_period_s,
        snr_db=0.0,
        freq_hz=float(base_bin*bin_hz),
        drift_hz_per_s=0.0,
        message_text="<parsed message pending>",
        crc_ok=True,
        time_offset_s=0.0,
    )
    return [dec]
