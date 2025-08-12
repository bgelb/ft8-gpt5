from ft8dec.io import read_wav_mono
from ft8dec.sync import stft_waterfall, costas_sync_score, extract_8tone_mags
from ft8dec.constants import FT8Constants
from ft8dec.ldpc_data import FT8_GRAY
from ft8dec.crc14 import crc14
import numpy as np

path = "data/ft8_wav/191111_110115.wav"
samples, fs = read_wav_mono(path)
psd, hop, bin_hz = stft_waterfall(samples, fs)
const = FT8Constants()
print("psd", psd.shape, "bin_hz", bin_hz)
# use known freq 1234 Hz
base_bin = int(round(1234.0 / bin_hz))
bin_step = int(round(const.tone_spacing_hz / bin_hz))
scores = costas_sync_score(psd, bin_step, base_bin)
start = int(np.argmax(scores))
print("start", start, "score", float(np.max(scores)))
# Hard slice symbols
bits = []
for k in range(29):
    mags = extract_8tone_mags(psd[start + 7 + k], base_bin, bin_step)
    if mags is None:
        sym = 0
    else:
        tone = int(np.argmax(mags))
        sym = FT8_GRAY.index(tone)
    bits += [ (sym >> 2) & 1, (sym >> 1) & 1, sym & 1 ]
for k in range(29):
    mags = extract_8tone_mags(psd[start + 43 + k], base_bin, bin_step)
    if mags is None:
        sym = 0
    else:
        tone = int(np.argmax(mags))
        sym = FT8_GRAY.index(tone)
    bits += [ (sym >> 2) & 1, (sym >> 1) & 1, sym & 1 ]
print("len bits", len(bits))
a91 = np.array(bits[:91], dtype=np.uint8)
# compute CRC
tmp_bits = np.zeros(96, dtype=np.uint8)
tmp_bits[:91] = a91
# zero pad bits 77..81
tmp_bits[77:82] = 0
crc_val = crc14(tmp_bits[:82])
recv_crc = 0
for i in range(14):
    recv_crc = (recv_crc << 1) | int(a91[77 + i])
print("crc", crc_val, recv_crc, crc_val == recv_crc)
