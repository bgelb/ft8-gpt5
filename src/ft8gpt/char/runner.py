from __future__ import annotations

import time
import numpy as np

from ft8gpt.sync import find_sync_candidates_stft
from ft8gpt.decoder_e2e import refine_sync_fine, coherent_symbol_energies, _llrs_from_linear_energies_gray_groups, _normalize_llrs_inplace
from ft8gpt.constants import SYMBOL_PERIOD_S, FT8_GRAY_MAP, gray_to_bits
from ft8gpt.ldpc import min_sum_decode, BeliefPropagationConfig
from ft8gpt.ldpc_tables_embedded import get_parity_matrices


def run_decoder(slot: np.ndarray, sample_rate_hz: float, which: str = "ft8gpt"):
	if which == "ft8gpt":
		from ft8gpt.decoder_e2e import decode_block
		return decode_block(slot, sample_rate_hz)
	elif which == "external":
		raise NotImplementedError("External decoder hook not implemented in this repo")
	else:
		raise ValueError(f"Unknown decoder selector: {which}")


def decode_with_stage_times(samples: np.ndarray, sample_rate_hz: float, top_k: int = 50) -> dict:
	"""Run a single-shot decode with coarse->fine->demod->ldpc stage timing.

	Returns dict with timings and success flag.
	"""
	Mn, Nm = get_parity_matrices()
	config = BeliefPropagationConfig(max_iterations=80, early_stop_no_improve=25)

	out: dict = {"coarse_ms": 0.0, "fine_ms": 0.0, "demod_ms": 0.0, "ldpc_ms": 0.0, "decoded": 0}
	# Coarse
	t0 = time.perf_counter()
	cands, nfft, hop = find_sync_candidates_stft(samples.astype(np.float64), sample_rate_hz, top_k=top_k)
	out["coarse_ms"] = (time.perf_counter() - t0) * 1000.0
	if not cands:
		return out
	cand = cands[0]
	coarse_abs = int(cand.frame_start) * int(hop)
	bin_hz = sample_rate_hz / float(nfft) if nfft > 0 else 6.25
	base_hz_est = (float(cand.base_bin) + float(cand.frac)) * bin_hz

	# Fine
	t0 = time.perf_counter()
	off2, df_hz, score = refine_sync_fine(samples, sample_rate_hz, base_hz_est, coarse_abs)
	out["fine_ms"] = (time.perf_counter() - t0) * 1000.0

	# Demod
	t0 = time.perf_counter()
	decim = max(1, int(round(sample_rate_hz / 200.0)))
	pos0_2 = int(round(coarse_abs / decim)) + int(off2)
	fs2 = float(sample_rate_hz / decim)
	data_times_rel = list(range(7, 36)) + list(range(43, 72))
	E = coherent_symbol_energies(
		samples[::decim].astype(np.complex128),  # consistent with decoder's decimate path
		fs2,
		pos0_2,
		df_hz,
		data_times_rel,
	)
	llrs: list[float] = []
	for row in E:
		l2, l1, l0 = _llrs_from_linear_energies_gray_groups(row)
		llrs.extend([l2, l1, l0])
	llrs_arr = np.array(llrs[:174], dtype=np.float64)
	_normalize_llrs_inplace(llrs_arr)
	out["demod_ms"] = (time.perf_counter() - t0) * 1000.0

	# LDPC
	t0 = time.perf_counter()
	errors, bits = min_sum_decode(llrs_arr, Mn, Nm, config)
	out["ldpc_ms"] = (time.perf_counter() - t0) * 1000.0
	if bits.shape[0] >= 91 and errors == 0:
		out["decoded"] = 1
	return out