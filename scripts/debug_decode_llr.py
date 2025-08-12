from ft8dec.io import read_wav_mono
from ft8dec.sync import stft_waterfall, costas_sync_score, extract_8tone_mags
from ft8dec.constants import FT8Constants
from ft8dec.demod import symbol_llrs
from ft8dec.ldpc import bp_decode
from ft8dec.crc14 import crc14
import numpy as np

path = "data/ft8_wav/191111_110115.wav"
samples, fs = read_wav_mono(path)
psd, hop, bin_hz = stft_waterfall(samples, fs)
const = FT8Constants()
print("psd", psd.shape, "bin_hz", bin_hz)
base_bin = int(round(1234.0 / bin_hz))
bin_step = int(round(const.tone_spacing_hz / bin_hz))
scores = costas_sync_score(psd, bin_step, base_bin)
start = int(np.argmax(scores))
print("start", start, "score", float(np.max(scores)))
llrs = []
for k in range(29):
    mags = extract_8tone_mags(psd[start + 7 + k], base_bin, bin_step)
    llrs.append(symbol_llrs(mags) if mags is not None else np.zeros(3,dtype=np.float32))
for k in range(29):
    mags = extract_8tone_mags(psd[start + 43 + k], base_bin, bin_step)
    llrs.append(symbol_llrs(mags) if mags is not None else np.zeros(3,dtype=np.float32))
log174 = np.concatenate(llrs, axis=0)
print('llr stats', float(np.max(log174)), float(np.min(log174)))
# normalize
if np.max(np.abs(log174))>0:
    log174 = log174/(np.max(np.abs(log174))+1e-9)
# decode
bits, unsat = bp_decode(log174, max_iters=200)
print('unsat', unsat)
a91 = bits[:91]
tmp_bits = np.zeros(96, dtype=np.uint8); tmp_bits[:91]=a91; tmp_bits[77:82]=0
crc_val = crc14(tmp_bits[:82])
recv_crc = 0
for i in range(14):
    recv_crc = (recv_crc<<1) | int(a91[77+i])
print('crc', crc_val, recv_crc, crc_val==recv_crc)
