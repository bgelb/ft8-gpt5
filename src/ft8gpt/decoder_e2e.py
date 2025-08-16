from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple
import numpy as np
import math

from .constants import (
	NN,
	ND,
	LENGTH_SYNC,
	SYNC_OFFSET,
	SYMBOL_PERIOD_S,
	NUM_SYNC,
	FT8_COSTAS_PATTERN,
	TONE_SPACING_HZ,
	FT8_GRAY_MAP,
	gray_to_bits,
)
from .waterfall import compute_waterfall_symbols
from .tones import extract_symbol_llrs
from .ldpc import min_sum_decode, BeliefPropagationConfig
from .ldpc_tables_embedded import get_parity_matrices
from .crc import crc14_check
from .message_decode import unpack_standard_payload
from scipy.signal import get_window, sosfiltfilt, cheby1
from numpy.lib.stride_tricks import as_strided
from .sync import find_sync_candidates_stft
from .ldpc_encode import get_encoder_structures


@dataclass(frozen=True)
class CandidateDecode:
	start_symbol: int
	ldpc_errors: int
	bits_with_crc: np.ndarray


def llrs_from_waterfall(wf_group: np.ndarray) -> np.ndarray:
	# wf_group shape [num_symbols, 8]
	llrs: List[float] = []
	for s in range(wf_group.shape[0]):
		l2, l1, l0 = extract_symbol_llrs(wf_group[s])
		# Encoder packs bits per symbol as (b2,b1,b0); order LLRs accordingly
		llrs.extend([l2, l1, l0])
	return np.array(llrs[: 174], dtype=np.float64)


def _downmix_and_decimate(
	samples: np.ndarray,
	sample_rate_hz: float,
	carrier_hz: float,
	decim_factor: int,
) -> Tuple[np.ndarray, float]:
	"""Mix real samples by exp(-j2π carrier t) and decimate by an integer factor.

	Returns (complex_baseband_decimated, new_sample_rate).
	"""
	n = np.arange(samples.size, dtype=np.float64)
	phi = -2.0 * np.pi * carrier_hz * (n / sample_rate_hz)
	osc = np.cos(phi) + 1j * np.sin(phi)
	y = samples.astype(np.float64) * osc
	# Zero-phase IIR decimation via cached SOS filter, then stride decimation
	sos = _cached_decimator_sos(int(decim_factor))
	yf = sosfiltfilt(sos, y)
	y_dec = yf[::decim_factor]
	return y_dec.astype(np.complex128, copy=False), float(sample_rate_hz / decim_factor)


@lru_cache(maxsize=16)
def _cached_decimator_sos(decim_factor: int):
	"""Return cached Chebyshev-I lowpass SOS for integer decimation.

	Matches SciPy decimate's default approximately: rp=0.05 dB, Wn=0.8/decim.
	"""
	rp_db = 0.05
	wn = 0.8 / float(max(1, decim_factor))
	N = 8
	return cheby1(N, rp_db, wn, output='sos')


def _precompute_ctones(symbol_len: int, fs_hz: float) -> np.ndarray:
	"""Return matrix C of shape [symbol_len, 8] with per-tone complex exponentials
	exp(-j2π tone*Δf/fs * n) for tone=0..7, Δf=TONE_SPACING_HZ.
	"""
	n = np.arange(symbol_len, dtype=np.float64)
	tones = np.arange(8, dtype=np.float64) * TONE_SPACING_HZ
	phase = -2.0 * np.pi * (np.outer(n, tones) / fs_hz)
	return np.cos(phase) - 1j * np.sin(phase)


def refine_sync_fine(
	samples: np.ndarray,
	sample_rate_hz: float,
	base_freq_hz: float,
	coarse_abs_start_sample: int,
	time_search_ms: int = 80,
	time_step_ms: int = 5,
	freq_search_hz: float = 3.2,
	freq_step_hz: float = 0.25,
) -> Tuple[int, float, float]:
	"""Refine time (±time_search in steps) and frequency (±freq_search in steps) using
	quasi-coherent cross-correlation against the Costas pattern.

	Returns (best_offset_samples_200Hz, best_df_hz, best_score).
	"""
	# Downmix to base frequency and decimate to 200 Hz (assumes 12000/60)
	decim = max(1, int(round(sample_rate_hz / 200.0)))
	y2, fs2 = _downmix_and_decimate(samples, sample_rate_hz, base_freq_hz, decim)
	pos0_2 = int(round(coarse_abs_start_sample / decim))
	return refine_sync_fine_precomputed(
		y2,
		fs2,
		pos0_2,
		time_search_ms=time_search_ms,
		time_step_ms=time_step_ms,
		freq_search_hz=freq_search_hz,
		freq_step_hz=freq_step_hz,
	)


def refine_sync_fine_precomputed(
	y2: np.ndarray,
	fs2: float,
	pos0_2: int,
	time_search_ms: int = 80,
	time_step_ms: int = 5,
	freq_search_hz: float = 3.2,
	freq_step_hz: float = 0.25,
) -> Tuple[int, float, float]:
	"""Vectorized refinement given precomputed 200 Hz baseband.

	Returns (best_offset_samples_200Hz, best_df_hz, best_score).
	"""
	sym_len = int(round(SYMBOL_PERIOD_S * fs2))
	if sym_len <= 0 or y2.size < sym_len:
		return 0, 0.0, -1e30

	ctones = _precompute_ctones(sym_len, fs2)
	win = np.ones(sym_len, dtype=np.float64)
	win /= np.sqrt(np.sum(win ** 2))

	n = np.arange(sym_len, dtype=np.float64)
	df_vals = np.arange(-freq_search_hz, freq_search_hz + 1e-9, freq_step_hz, dtype=np.float64)
	if df_vals.size == 0:
		df_vals = np.array([0.0], dtype=np.float64)
	ctweaks = (np.cos(-2.0 * np.pi * df_vals[:, None] * n[None, :] / fs2)
		      - 1j * np.sin(-2.0 * np.pi * df_vals[:, None] * n[None, :] / fs2))

	tsamp = int(round(time_step_ms * fs2 / 1000.0))
	max_tsamp = int(round(time_search_ms * fs2 / 1000.0))
	if tsamp <= 0:
		tsamp = 1
	time_offsets = np.arange(-max_tsamp, max_tsamp + 1, tsamp, dtype=np.int64)
	if time_offsets.size == 0:
		time_offsets = np.array([0], dtype=np.int64)

	L = sym_len
	N = int(y2.size)
	num_frames = max(0, N - L + 1)
	if num_frames <= 0:
		return 0, 0.0, -1e30
	frames = as_strided(y2, shape=(num_frames, L), strides=(y2.strides[0], y2.strides[0]))

	sync_starts = (0, 36, 72)
	sym_rel = []
	for m in sync_starts:
		for k in range(LENGTH_SYNC):
			sym_rel.append((m + k) * L)
	sym_rel = np.array(sym_rel, dtype=np.int64)

	starts = pos0_2 + time_offsets[:, None] + sym_rel[None, :]
	valid = (starts >= 0) & (starts < num_frames)
	starts_clipped = np.clip(starts, 0, max(0, num_frames - 1))

	X = frames[starts_clipped]
	Xw = X * win[None, None, :]
	Y = (Xw[None, :, :, :] * ctweaks[:, None, None, :]).conj()
	Z = np.einsum('d o s n, n k -> d o s k', Y, ctones, optimize=True)
	E = (Z.real * Z.real + Z.imag * Z.imag)

	expected = np.array([FT8_COSTAS_PATTERN[k] for _m in sync_starts for k in range(LENGTH_SYNC)], dtype=np.int64)
	tm = np.maximum(expected - 1, 0)
	tp = np.minimum(expected + 1, 7)
	idx_exp = expected[None, None, :, None]
	idx_tm = tm[None, None, :, None]
	idx_tp = tp[None, None, :, None]
	E_exp = np.take_along_axis(E, idx_exp, axis=3)[..., 0]
	E_m = np.take_along_axis(E, idx_tm, axis=3)[..., 0]
	E_p = np.take_along_axis(E, idx_tp, axis=3)[..., 0]
	En = 0.5 * (E_m + E_p)

	valid_f = valid.astype(np.float64)[None, :, :]
	scores = (E_exp / (En + 1e-20)) * valid_f
	cnts = np.sum(valid_f, axis=2)
	cnts = np.maximum(cnts, 1.0)
	T = np.sum(scores, axis=2) / cnts

	best_flat = int(np.argmax(T))
	best_d, best_o = np.unravel_index(best_flat, T.shape)
	best_score = float(T[best_d, best_o])
	best_df = float(df_vals[best_d])
	best_off = int(time_offsets[best_o])
	return best_off, best_df, best_score


def coherent_symbol_energies(
	y2: np.ndarray,
	fs2: float,
	pos0_2: int,
	df_hz: float,
	symbol_indices: List[int],
) -> np.ndarray:
	"""Vectorized coherent energies for a set of symbol indices.

	Returns [num_symbols, 8] energies.
	"""
	sym_len = int(round(SYMBOL_PERIOD_S * fs2))
	if sym_len <= 0 or y2.size < sym_len or not symbol_indices:
		return np.zeros((0, 8), dtype=np.float64)
	ctones = _precompute_ctones(sym_len, fs2)
	n = np.arange(sym_len, dtype=np.float64)
	ctweak = np.cos(-2.0 * np.pi * df_hz * n / fs2) - 1j * np.sin(-2.0 * np.pi * df_hz * n / fs2)
	win = np.ones(sym_len, dtype=np.float64)
	win /= np.sqrt(np.sum(win ** 2))

	starts = np.array([pos0_2 + t * sym_len for t in symbol_indices], dtype=np.int64)
	valid = (starts >= 0) & (starts + sym_len <= y2.size)
	if not np.any(valid):
		return np.zeros((0, 8), dtype=np.float64)
	starts_valid = starts[valid]
	frames = as_strided(y2, shape=(y2.size - sym_len + 1, sym_len), strides=(y2.strides[0], y2.strides[0]))
	S = frames[starts_valid]
	Sw = S * (win * ctweak)[None, :]
	den = np.sum(Sw.real * Sw.real + Sw.imag * Sw.imag, axis=1)
	den = np.where((den > 1e-20) & np.isfinite(den), den, 1.0)
	Z = np.einsum('s n, n k -> s k', Sw.conj(), ctones, optimize=True)
	E = (Z.real * Z.real + Z.imag * Z.imag) / den[:, None]
	out = np.zeros((len(symbol_indices), 8), dtype=np.float64)
	out[np.where(valid)[0]] = E
	return out


def _costas_snr_score(
	y2: np.ndarray,
	fs2: float,
	pos0_2: int,
	df_hz: float,
) -> float:
	"""Compute noise-normalized Costas correlation energy across 3 sync blocks.

	Uses the same SNR-like metric as refine_sync_fine, with rectangular window and adjacent-tone noise proxy.
	"""
	sym_len = int(round(SYMBOL_PERIOD_S * fs2))
	ctones = _precompute_ctones(sym_len, fs2)
	n = np.arange(sym_len, dtype=np.float64)
	ctweak = np.cos(-2.0 * np.pi * df_hz * n / fs2) - 1j * np.sin(-2.0 * np.pi * df_hz * n / fs2)
	win = np.ones(sym_len, dtype=np.float64)
	win /= np.sqrt(np.sum(win ** 2))

	T = 0.0
	cnt = 0
	for m in (0, 36, 72):
		for k in range(LENGTH_SYNC):
			tone = FT8_COSTAS_PATTERN[k]
			pos = pos0_2 + (m + k) * sym_len
			if pos < 0 or pos + sym_len > y2.size:
				continue
			segw = win * y2[pos: pos + sym_len] * ctweak
			zc = np.vdot(segw, ctones[:, tone])
			Es = float((zc.real * zc.real + zc.imag * zc.imag))
			# adjacent tones as noise proxy
			tm = max(0, tone - 1)
			tp = min(7, tone + 1)
			zm = np.vdot(segw, ctones[:, tm])
			zp = np.vdot(segw, ctones[:, tp])
			En = 0.5 * float((zm.real * zm.real + zm.imag * zm.imag) + (zp.real * zp.real + zp.imag * zp.imag))
			T += Es / (En + 1e-20)
			cnt += 1
	return T if cnt > 0 else -1e300


def count_costas_matches(E_sync: np.ndarray) -> int:
	"""Count how many of the 21 sync symbols have argmax matching the Costas tone."""
	matches = 0
	# E_sync is shape [21, 8] ordered by actual sync times
	for i, k in enumerate(range(LENGTH_SYNC)):
		if int(np.argmax(E_sync[i])) == FT8_COSTAS_PATTERN[k]:
			matches += 1
	for i, k in enumerate(range(LENGTH_SYNC)):
		if int(np.argmax(E_sync[LENGTH_SYNC + i])) == FT8_COSTAS_PATTERN[k]:
			matches += 1
	for i, k in enumerate(range(LENGTH_SYNC)):
		if int(np.argmax(E_sync[2 * LENGTH_SYNC + i])) == FT8_COSTAS_PATTERN[k]:
			matches += 1
	return matches


def _llrs_from_linear_energies_gray_groups(energies_row: np.ndarray) -> Tuple[float, float, float]:
	"""Compute per-symbol LLRs from linear tone energies using Gray group log-sum ratios.

	LLR(bit=0) - LLR(bit=1) = log(sum energies where bit=0) - log(sum energies where bit=1).
	"""
	# Map to Gray order index j -> energies of Gray-coded symbol j
	s_gray = np.empty(8, dtype=np.float64)
	for j in range(8):
		tone = FT8_GRAY_MAP[j]
		s_gray[j] = float(max(energies_row[tone], 1e-20))
	# Group indices for each bit
	g0 = (0, 1, 2, 3)
	g1 = (4, 5, 6, 7)
	g_b1_0 = (0, 1, 4, 5)
	g_b1_1 = (2, 3, 6, 7)
	g_b0_0 = (0, 2, 4, 6)
	g_b0_1 = (1, 3, 5, 7)
	import math
	llr2 = math.log(sum(s_gray[i] for i in g0)) - math.log(sum(s_gray[i] for i in g1))
	llr1 = math.log(sum(s_gray[i] for i in g_b1_0)) - math.log(sum(s_gray[i] for i in g_b1_1))
	llr0 = math.log(sum(s_gray[i] for i in g_b0_0)) - math.log(sum(s_gray[i] for i in g_b0_1))
	# Return in (b2,b1,b0) order consistent with previous usage
	return llr2, llr1, llr0


def _llrs_from_linear_energies_gray_groups_max(energies_row: np.ndarray) -> Tuple[float, float, float]:
	"""Compute LLRs via max-of-groups per FT8 reference (ft8_extract_symbol)."""
	# Map to Gray-ordered energies
	s2 = np.empty(8, dtype=np.float64)
	for j in range(8):
		s2[j] = float(max(energies_row[FT8_GRAY_MAP[j]], 1e-20))
	def max4(a, b, c, d):
		return max(a, b, c, d)
	b0 = max4(s2[4], s2[5], s2[6], s2[7]) - max4(s2[0], s2[1], s2[2], s2[3])
	b1 = max4(s2[2], s2[3], s2[6], s2[7]) - max4(s2[0], s2[1], s2[4], s2[5])
	b2 = max4(s2[1], s2[3], s2[5], s2[7]) - max4(s2[0], s2[2], s2[4], s2[6])
	return b2, b1, b0


def _normalize_llrs_inplace(llrs: np.ndarray) -> None:
	"""Scale LLRs so that their variance matches the reference (~24) as in ftx_normalize_logl.

	This improves LDPC convergence across a wide range of SNRs.
	"""
	if llrs.size == 0:
		return
	# Compute population variance
	sum_val = float(np.sum(llrs))
	sum2_val = float(np.sum(llrs * llrs))
	inv_n = 1.0 / float(llrs.size)
	variance = (sum2_val - (sum_val * sum_val * inv_n)) * inv_n
	if variance <= 1e-12:
		return
	norm_factor = math.sqrt(24.0 / variance)
	llrs *= norm_factor


def _sync_score_for_base(wf_mag: np.ndarray, start: int, base_idx: int) -> float:
	# Sum tone energy at Costas pattern positions across three sync blocks
	num_symbols = wf_mag.shape[0]
	score = 0.0
	count = 0
	for block_start in (start, start + 36, start + 72):
		for k in range(LENGTH_SYNC):
			t = block_start + k
			if t < 0 or t >= num_symbols:
				continue
			tone = FT8_COSTAS_PATTERN[k]
			row = wf_mag[t, base_idx, :]
			score += float(row[tone])
			count += 1
	if count == 0:
		return -1e9
	return score / count


def decode_block(samples: np.ndarray, sample_rate_hz: float) -> List[CandidateDecode]:
	# Analyze the entire buffer using STFT-based Costas matched filter to find coarse candidates
	n_fft = int(round(sample_rate_hz * SYMBOL_PERIOD_S))
	if n_fft <= 0:
		return []

	Mn, Nm = get_parity_matrices()
	# Stronger LDPC settings improve convergence with coherent LLRs
	config = BeliefPropagationConfig(max_iterations=80, early_stop_no_improve=25)

	# Compute codeword->payload mapping once based on embedded parity structure
	Br_inv, Hrest, rest_cols, piv_cols = get_encoder_structures()

	results: List[CandidateDecode] = []
	# Track unique decodes by packed bits to avoid duplicates across candidates
	seen_keys = set()

	# Run 2D matched-filter candidate search across the whole buffer with time padding
	pad_pre = int(round(0.5 * sample_rate_hz))
	pad_post = int(round(1.5 * sample_rate_hz))
	xpad = np.pad(samples.astype(np.float64, copy=False), (pad_pre, pad_post))
	# Candidate search
	candidates, stft_nfft, hop = find_sync_candidates_stft(xpad, sample_rate_hz, top_k=400)

	# Process top candidates with fine refinement and coherent demod
	for cand in candidates:
		# Coarse absolute sample index corresponding to the first Costas symbol
		coarse_abs_start = cand.frame_start * hop - pad_pre
		if coarse_abs_start < 0:
			continue

		# Bin spacing and base frequency for both frac hypotheses (cand.frac is already 0.0 or 0.5)
		bin_hz = sample_rate_hz / stft_nfft if stft_nfft > 0 else TONE_SPACING_HZ
		base_idx = int(cand.base_bin)
		frac = float(cand.frac)
		base_freq_hz = (base_idx + frac) * bin_hz

		# Refine time and frequency via Costas correlation at 200 Hz
		best_off_2, df_hz, corr_score = refine_sync_fine(
			samples, sample_rate_hz, base_freq_hz, coarse_abs_start
		)

		# Downmix/decimate once more to reuse in demod
		decim = max(1, int(round(sample_rate_hz / 200.0)))
		y2, fs2 = _downmix_and_decimate(samples, sample_rate_hz, base_freq_hz, decim)
		sym_len2 = int(round(SYMBOL_PERIOD_S * fs2))
		pos0_2 = int(round(coarse_abs_start / decim)) + best_off_2
		if sym_len2 <= 0:
			continue

		# Build symbol index lists
		sync_times = (
			list(range(0, 7)) +
			list(range(36, 43)) +
			list(range(72, 79))
		)
		# Use times RELATIVE to candidate start for coherent demod
		data_times_rel = list(range(7, 36)) + list(range(43, 72))

		# Evaluate Costas match quality gate
		# Compute sync energies at refined alignment
		sync_E = coherent_symbol_energies(y2, fs2, pos0_2, df_hz, sync_times)
		if sync_E.shape[0] != 21:
			continue
		# Require a strong Costas lock for proceeding (slightly relaxed)
		if count_costas_matches(sync_E) < 16:
			continue

		# Micro-refine alignment using Costas SNR as objective (do not use bit matches)
		best_snr = -1e300
		best_pos = pos0_2
		best_df = df_hz
		cfo_grid = np.arange(df_hz - 0.8, df_hz + 0.8001, 0.2, dtype=np.float64)
		# Explore small offsets: integer symbol shifts ±3 and sample offsets ±8
		sym_len2 = int(round(SYMBOL_PERIOD_S * fs2))
		for sym_shift in range(-3, 4):
			base_pos = pos0_2 + sym_shift * sym_len2
			for samp_shift in range(-8, 9):
				pos_try = base_pos + int(samp_shift)
				for df_try in cfo_grid:
					S = _costas_snr_score(y2, fs2, pos_try, float(df_try))
					if S > best_snr:
						best_snr = S
						best_pos = pos_try
						best_df = float(df_try)

		# With the best Costas alignment, demod and attempt LDPC and CRC
		E = coherent_symbol_energies(y2, fs2, best_pos, best_df, data_times_rel)
		llrs: List[float] = []
		for row in E:
			l2, l1, l0 = _llrs_from_linear_energies_gray_groups(row)
			llrs.extend([l2, l1, l0])
		llrs_arr = np.array(llrs[:174], dtype=np.float64)
		_normalize_llrs_inplace(llrs_arr)
		errors, bits = min_sum_decode(llrs_arr, Mn, Nm, config)
		if bits.shape[0] >= 91 and errors == 0:
			# Try reference mapping first (consecutive a91)
			a91_ref = bits[:91].astype(np.uint8)
			bwc_ref = np.concatenate([a91_ref[:77], a91_ref[77:91]])
			if crc14_check(bwc_ref):
				key = np.packbits(bwc_ref).tobytes()
				if key not in seen_keys:
					seen_keys.add(key)
					results.append(CandidateDecode(start_symbol=0, ldpc_errors=errors, bits_with_crc=bwc_ref))
			# Also try encoder-derived systematic mapping (compat with internal synthetic)
			a91_sys = bits[np.array(rest_cols, dtype=np.int64)].astype(np.uint8)
			bwc_sys = np.concatenate([a91_sys[:77], a91_sys[77:91]])
			if crc14_check(bwc_sys):
				key = np.packbits(bwc_sys).tobytes()
				if key not in seen_keys:
					seen_keys.add(key)
					results.append(CandidateDecode(start_symbol=0, ldpc_errors=errors, bits_with_crc=bwc_sys))
		# If not successful, try alternative LLR computation (max-of-groups)
		if not results:
			llrs_alt: List[float] = []
			for row in E:
				b2, b1, b0 = _llrs_from_linear_energies_gray_groups_max(row)
				llrs_alt.extend([b2, b1, b0])
			llrs_alt_arr = np.array(llrs_alt[:174], dtype=np.float64)
			_normalize_llrs_inplace(llrs_alt_arr)
			errors2, bits2 = min_sum_decode(llrs_alt_arr, Mn, Nm, config)
			if bits2.shape[0] >= 91 and errors2 == 0:
				# Reference mapping
				a91_ref2 = bits2[:91].astype(np.uint8)
				bwc_ref2 = np.concatenate([a91_ref2[:77], a91_ref2[77:91]])
				if crc14_check(bwc_ref2):
					key = np.packbits(bwc_ref2).tobytes()
					if key not in seen_keys:
						seen_keys.add(key)
						results.append(CandidateDecode(start_symbol=0, ldpc_errors=errors2, bits_with_crc=bwc_ref2))
				# Systematic mapping
				a91_sys2 = bits2[np.array(rest_cols, dtype=np.int64)].astype(np.uint8)
				bwc_sys2 = np.concatenate([a91_sys2[:77], a91_sys2[77:91]])
				if crc14_check(bwc_sys2):
					key = np.packbits(bwc_sys2).tobytes()
					if key not in seen_keys:
						seen_keys.add(key)
						results.append(CandidateDecode(start_symbol=0, ldpc_errors=errors2, bits_with_crc=bwc_sys2))
		# Also try hard-decision once at the best alignment for robustness
		cw_bits: List[int] = []
		for row in E:
			tone = int(np.argmax(row))
			b2, b1, b0 = gray_to_bits(tone)
			cw_bits.extend([b2, b1, b0])
		cw = np.array(cw_bits[:174], dtype=np.uint8)
		if cw.shape[0] == 174:
			# Reference mapping
			a91_hd_ref = cw[:91].astype(np.uint8)
			bwc_hd_ref = np.concatenate([a91_hd_ref[:77], a91_hd_ref[77:91]])
			if crc14_check(bwc_hd_ref):
				key = np.packbits(bwc_hd_ref).tobytes()
				if key not in seen_keys:
					seen_keys.add(key)
					results.append(CandidateDecode(start_symbol=0, ldpc_errors=0, bits_with_crc=bwc_hd_ref))
			# Systematic mapping compatibility
			a91_hd_sys = cw[np.array(rest_cols, dtype=np.int64)].astype(np.uint8)
			bwc_hd_sys = np.concatenate([a91_hd_sys[:77], a91_hd_sys[77:91]])
			if crc14_check(bwc_hd_sys):
				key = np.packbits(bwc_hd_sys).tobytes()
				if key not in seen_keys:
					seen_keys.add(key)
					results.append(CandidateDecode(start_symbol=0, ldpc_errors=0, bits_with_crc=bwc_hd_sys))
	return results
