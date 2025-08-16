from __future__ import annotations

import math
import numpy as np
from typing import Iterable, Sequence


def recall_at_k(truth_indices: Iterable[float], candidate_scores: Sequence[tuple[float, float]], tol: float) -> float:
	"""Compute recall@K over frequency-like axis.

	- truth_indices: list of ground-truth target values (e.g., Hz)
	- candidate_scores: list of (value, score) sorted by descending score
	- tol: absolute tolerance for a hit (same units as value)
	"""
	if not truth_indices:
		return 0.0
	if not candidate_scores:
		return 0.0
	truths = list(truth_indices)
	cvals = [v for (v, _s) in candidate_scores]
	hits = 0
	for t in truths:
		ok = any(abs(cv - t) <= tol for cv in cvals)
		hits += 1 if ok else 0
	return float(hits) / float(len(truths))


def precision_fp_at_k(truth_indices: Iterable[float], candidate_scores: Sequence[tuple[float, float]], tol: float) -> tuple[float, int]:
	"""Return (precision@K, false_positives@K) for top-K candidates provided.
	K = len(candidate_scores).
	"""
	K = len(candidate_scores)
	if K == 0:
		return 0.0, 0
	truths = list(truth_indices)
	cvals = [v for (v, _s) in candidate_scores]
	tp = 0
	matched = [False] * len(truths)
	for cv in cvals:
		for i, t in enumerate(truths):
			if not matched[i] and abs(cv - t) <= tol:
				matched[i] = True
				tp += 1
				break
	fp = max(0, K - tp)
	precision = float(tp) / float(K) if K > 0 else 0.0
	return precision, fp


def rmse(errors: Iterable[float]) -> float:
	arr = np.array(list(errors), dtype=np.float64)
	if arr.size == 0:
		return 0.0
	return float(np.sqrt(np.mean(arr * arr)))


def bit_mutual_information(llrs: np.ndarray, bits: np.ndarray) -> float:
	"""Estimate bit mutual information from LLRs and ground-truth bits.

	BMI ≈ 1 - (1/N) * sum_i log2(1 + exp(-(-1)^b_i * L_i))
	"""
	if llrs.size == 0 or bits.size == 0:
		return 0.0
	b = bits.astype(np.float64)
	s = 1.0 - 2.0 * b  # (-1)^b = 1 for b=0; -1 for b=1 -> use sign trick via sgn
	z = np.log2(1.0 + np.exp(-(s * llrs.astype(np.float64))))
	return float(1.0 - np.mean(z))


def llr_separation(llrs: np.ndarray, bits: np.ndarray) -> float:
	"""Mean signed LLR, positive when correct bits have larger magnitude with correct sign."""
	if llrs.size == 0 or bits.size == 0:
		return 0.0
	signs = (1.0 - 2.0 * bits.astype(np.float64))
	return float(np.mean(signs * llrs.astype(np.float64)))


def auc_from_scores(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
	"""Compute AUC given arrays of positive and negative scores."""
	if scores_pos.size == 0 or scores_neg.size == 0:
		return 0.0
	total = 0
	wins = 0
	for sp in scores_pos:
		for sn in scores_neg:
			total += 1
			if sp > sn:
				wins += 1
			elif sp == sn:
				wins += 0.5
	return float(wins) / float(total) if total > 0 else 0.0