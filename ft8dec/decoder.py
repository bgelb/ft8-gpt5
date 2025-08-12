from __future__ import annotations
from typing import List
import numpy as np
from .types import Decode
from .io import read_wav_mono
from .sync import stft_waterfall, costas_sync_score
from .demod import symbol_llrs
from .ldpc import bp_decode
from .constants import FT8Constants
from .crc14 import crc14, POLY as CRC_POLY


def decode_wav_file(path: str) -> List[Decode]:
    """Decode an FT8 WAV file and return decodes.

    Minimal pipeline: waterfall -> sync score -> single candidate -> LLRs -> LDPC -> CRC check.
    """
    samples, fs_hz = read_wav_mono(path)
    wf, stride, num_bins = stft_waterfall(samples, fs_hz)
    if wf.shape[0] < 100:
        return []
    scores = costas_sync_score(wf)
    if len(scores) == 0:
        return []
    start = int(np.argmax(scores))
    const = FT8Constants()
    # Collect LLRs for 58 data symbols (skip Costas at 0..6, 36..42, 72..78)
    llrs = []
    def add_llrs(sym_idx: int):
        if 0 <= sym_idx < wf.shape[0]:
            llrs.append(symbol_llrs(wf[sym_idx]))
        else:
            llrs.append(np.zeros(3, dtype=np.float32))
    # First block D1 symbols 7..35 (29 symbols)
    for k in range(29):
        add_llrs(start + 7 + k)
    # Second block D2 symbols 43..71 (29 symbols)
    for k in range(29):
        add_llrs(start + 43 + k)
    if len(llrs) != 58:
        return []
    log174 = np.concatenate(llrs, axis=0)
    # Normalize roughly
    if np.max(np.abs(log174)) > 0:
        log174 = log174 / (np.max(np.abs(log174)) + 1e-9)
    bits, unsat = bp_decode(-log174, max_iters=40)
    if unsat != 0:
        return []
    # Extract first 91 bits (payload + CRC)
    a91 = bits[:91]
    # Recompute CRC over 82 bits (77 + 5 zeros)
    # Pack into bytes MSB-first as ft8_lib does for crc
    def pack_bits(arr):
        by = []
        acc = 0
        cnt = 0
        for b in arr:
            acc = (acc << 1) | int(b)
            cnt += 1
            if cnt == 8:
                by.append(acc)
                acc = 0
                cnt = 0
        if cnt > 0:
            by.append(acc << (8-cnt))
        return np.array(by, dtype=np.uint8)
    a91_bytes = pack_bits(a91)
    # Zero out 14 CRC bits region (like add_crc did during compute) and compute CRC over first 82 bits
    # Create temp 96-bit buffer with CRC area zeroed
    tmp_bits = np.zeros(96, dtype=np.uint8)
    tmp_bits[:91] = a91
    tmp_bits[91:96] = 0
    tmp_bytes = pack_bits(tmp_bits)
    # Compute CRC-14; our crc14 expects bit array 82 long, so pass first 82 bits
    crc_val = crc14(tmp_bits[:82])
    # Extract received CRC from a91 bits [77..90] (14 bits)
    recv_crc = 0
    for i in range(14):
        recv_crc = (recv_crc << 1) | int(a91[77 + i])
    crc_ok = (crc_val == recv_crc)
    if not crc_ok:
        return []
    # For now, message parsing not implemented; return placeholder
    dec = Decode(
        start_time_s=float(start)*const.symbol_period_s,
        snr_db=0.0,
        freq_hz=0.0,
        drift_hz_per_s=0.0,
        message_text="<parsed message pending>",
        crc_ok=True,
        time_offset_s=0.0,
    )
    return [dec]
