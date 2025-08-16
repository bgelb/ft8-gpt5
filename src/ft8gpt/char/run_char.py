from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from ft8gpt.char.scenarios import get_default_scenarios, load_scenarios
from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn, mix_signals
from ft8gpt.char.runner import run_decoder, decode_with_stage_times
from ft8gpt.char.metrics import precision_fp_at_k
from ft8gpt.sync import find_sync_candidates_stft


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def run_awgn_snr_sweep(cfg: dict, outdir: Path) -> None:
	sr = float(cfg.get("sr", 12000.0))
	base_freq_hz = float(cfg.get("base_freq_hz", 1500.0))
	snr_list = list(cfg.get("snr_db", [-22, -20, -18, -16, -14, -12]))
	trials = int(cfg.get("trials", 20))
	seed = int(cfg.get("seed", 123))
	rng = np.random.default_rng(seed)

	rows = [("snr_db", "trials", "decode_rate")]
	for snr_db in snr_list:
		ok = 0
		for _ in range(trials):
			x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, base_freq_hz)
			y = apply_awgn(x, float(snr_db), rng)
			res = run_decoder(y, sr)
			ok += 1 if len(res) > 0 else 0
		decode_rate = ok / float(trials)
		rows.append((snr_db, trials, f"{decode_rate:.3f}"))
	ensure_dir(outdir)
	with (outdir / "ldpc_waterfall.csv").open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerows(rows)


def run_cfo_sweep(cfg: dict, outdir: Path) -> None:
	sr = float(cfg.get("sr", 12000.0))
	base_freq_hz = float(cfg.get("base_freq_hz", 1500.0))
	cfo_vals = list(cfg.get("cfo_hz", [-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5]))
	snr_db = float(cfg.get("snr_db", -16))
	trials = int(cfg.get("trials", 10))
	seed = int(cfg.get("seed", 321))
	rng = np.random.default_rng(seed)

	rows = [("cfo_hz", "trials", "decode_rate")]
	for df in cfo_vals:
		ok = 0
		for _ in range(trials):
			x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, base_freq_hz + float(df))
			y = apply_awgn(x, snr_db, rng)
			res = run_decoder(y, sr)
			ok += 1 if len(res) > 0 else 0
		rows.append((df, trials, f"{ok/float(trials):.3f}"))
	ensure_dir(outdir)
	with (outdir / "cfo_sweep.csv").open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerows(rows)


def run_occupancy(cfg: dict, outdir: Path) -> None:
	sr = float(cfg.get("sr", 12000.0))
	snr_db = float(cfg.get("snr_db", -16))
	top_k_raw = int(cfg.get("top_k_raw", 800))
	K_eval = int(cfg.get("K_eval", 80))
	seed = int(cfg.get("seed", 1234))
	nsigs_list = list(cfg.get("num_sigs", [10, 20, 40, 80]))
	rng = np.random.default_rng(seed)

	rows = [("num_sigs", "snr_db", "K_eval", "recall", "precision", "fp_per_slot", "time_ms")]
	for num_sigs in nsigs_list:
		freqs = np.linspace(500.0, 3500.0, num_sigs, endpoint=False) + rng.uniform(-1.5, 1.5, size=num_sigs)
		signals = []
		truth = []
		for i in range(num_sigs):
			x, _ = make_clean_signal("K1AAA", "W9XYZ", "FN20", sr, float(freqs[i]))
			xn = apply_awgn(x, snr_db, rng)
			signals.append(xn)
			truth.append(float(freqs[i]))
			slot = mix_signals(signals, [0.0] * num_sigs)
			cands, nfft, hop = find_sync_candidates_stft(slot.astype(np.float64), sr, top_k=top_k_raw)
			bin_hz = sr / float(nfft) if nfft > 0 else 6.25
			# Top-K value list
			cand_vals = [((c.base_bin + c.frac) * bin_hz, c.score) for c in cands[:K_eval]]
		prec, fp = precision_fp_at_k(truth, cand_vals, tol=6.25)
		recall = 0.0
		if len(truth) > 0:
			found = 0
			for t in truth:
				if any(abs(v - t) <= 6.25 for (v, _s) in cand_vals):
					found += 1
			recall = found / float(len(truth))
		rows.append((num_sigs, snr_db, K_eval, f"{recall:.3f}", f"{prec:.3f}", fp, 0.0))
	ensure_dir(outdir)
	with (outdir / "coarse_recall.csv").open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerows(rows)


def run_end_to_end(cfg: dict, outdir: Path) -> None:
	sr = float(cfg.get("sr", 12000.0))
	base_freq_hz = float(cfg.get("base_freq_hz", 1500.0))
	snr_list = list(cfg.get("snr_db", [-20, -18, -16, -14]))
	trials = int(cfg.get("trials", 20))
	seed = int(cfg.get("seed", 7))
	rng = np.random.default_rng(seed)

	rows = [("snr_db", "decode_rate", "fp_rate", "coarse_ms", "fine_ms", "demod_ms", "ldpc_ms")]
	for snr_db in snr_list:
		ok = 0
		fp = 0
		coarse_ms = 0.0
		fine_ms = 0.0
		demod_ms = 0.0
		ldpc_ms = 0.0
		truth_keys = []
		for _ in range(trials):
			x, _ = make_clean_signal("K1ABC", "W9XYZ", "FN20", sr, base_freq_hz)
			y = apply_awgn(x, float(snr_db), rng)
			stats = decode_with_stage_times(y, sr)
			coarse_ms += stats["coarse_ms"]
			fine_ms += stats["fine_ms"]
			demod_ms += stats["demod_ms"]
			ldpc_ms += stats["ldpc_ms"]
			ok += stats["decoded"]
		rows.append((snr_db, f"{ok/float(trials):.3f}", f"{fp/float(trials):.3f}", f"{coarse_ms/trials:.2f}", f"{fine_ms/trials:.2f}", f"{demod_ms/trials:.2f}", f"{ldpc_ms/trials:.2f}"))
	ensure_dir(outdir)
	with (outdir / "end_to_end.csv").open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerows(rows)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run FT8 characterization scenarios")
	parser.add_argument("scenario", nargs="?", default="awgn_snr_sweep", help="Scenario name or 'list'")
	parser.add_argument("--config", default=None, help="Path to JSON with scenarios")
	parser.add_argument("--outdir", default="reports", help="Output directory for CSV/JSON")
	args = parser.parse_args()

	scenarios = get_default_scenarios() if args.config is None else load_scenarios(args.config)

	if args.scenario == "list":
		print("Available scenarios:")
		for k in scenarios.keys():
			print(" -", k)
		return

	outdir = Path(args.outdir)
	cfg = scenarios.get(args.scenario)
	if cfg is None:
		raise SystemExit(f"Unknown scenario: {args.scenario}")

	if args.scenario == "awgn_snr_sweep":
		run_awgn_snr_sweep(cfg, outdir)
	elif args.scenario == "cfo_sweep":
		run_cfo_sweep(cfg, outdir)
	elif args.scenario == "occupancy":
		run_occupancy(cfg, outdir)
	elif args.scenario == "end_to_end":
		run_end_to_end(cfg, outdir)
	else:
		raise SystemExit(f"Scenario not implemented: {args.scenario}")


if __name__ == "__main__":
	main()
