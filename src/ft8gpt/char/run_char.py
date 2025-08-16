from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List
import numpy as np

from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn, mix_signals
from ft8gpt.char.scenarios import SCENARIOS
from ft8gpt.sync import find_sync_candidates_stft
from ft8gpt.decoder_e2e import decode_block, refine_sync_fine

REPORTS_DIR = Path("reports")


def ensure_reports_dir() -> None:
	REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_awgn_snr_sweep(cfg: dict) -> Path:
	ensure_reports_dir()
	out = REPORTS_DIR / "ldpc_waterfall.csv"
	sr = float(cfg["sr"]) ; base = float(cfg["base_freq_hz"]) ; trials = int(cfg["trials"]) ; seed = int(cfg.get("seed", 0))
	rng = np.random.default_rng(seed)
	snrs = list(cfg["snr_db"]) ; call_to = cfg["call_to"] ; call_de = cfg["call_de"] ; grid4 = cfg["grid4"]
	with out.open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["snr_db", "trials", "ok", "fer"])
		for snr in snrs:
			ok = 0
			for _ in range(trials):
				x, _ = make_clean_signal(call_to, call_de, grid4, sr, base)
				y = apply_awgn(x, float(snr), rng)
				res = decode_block(y, sr)
				ok += int(len(res) > 0)
			fer = 1.0 - ok / float(trials)
			w.writerow([snr, trials, ok, f"{fer:.6f}"])
	return out


def run_cfo_sweep(cfg: dict) -> Path:
	ensure_reports_dir()
	out = REPORTS_DIR / "fine_sync_rmse.csv"
	sr = float(cfg["sr"]) ; base = float(cfg["base_freq_hz"]) ; trials = int(cfg["trials"]) ; seed = int(cfg.get("seed", 0))
	rng = np.random.default_rng(seed)
	cfos = list(cfg["cfo_hz"]) ; snr = float(cfg["snr_db"]) ; call_to = cfg["call_to"] ; call_de = cfg["call_de"] ; grid4 = cfg["grid4"]
	with out.open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["cfo_hz", "rmse_hz"])
		sqerrs: List[float] = []
		for df in cfos:
			se_local = []
			for _ in range(trials):
				x, _ = make_clean_signal(call_to, call_de, grid4, sr, base + float(df))
				y = apply_awgn(x, snr, rng)
				cands, nfft, hop = find_sync_candidates_stft(y, sr, top_k=20)
				if not cands:
					continue
				best = cands[0]
				coarse_abs = best.frame_start * hop
				bin_hz = sr / nfft if nfft > 0 else 6.25
				base_hz_est = (best.base_bin + best.frac) * bin_hz
				_, df_est, _ = refine_sync_fine(y, sr, base_hz_est, coarse_abs)
				se_local.append((df_est - float(df)) ** 2)
			if se_local:
				sqerrs.append(float(np.mean(se_local)))
		for df, se in zip(cfos, sqerrs):
			w.writerow([df, f"{np.sqrt(se):.6f}"])
	return out


def run_occupancy(cfg: dict) -> Path:
	ensure_reports_dir()
	out = REPORTS_DIR / "coarse_recall.csv"
	sr = float(cfg["sr"]) ; num_sigs_list = list(cfg["num_sigs"]) ; snr_db = float(cfg["snr_db"]) ; seed = int(cfg.get("seed", 0))
	rng = np.random.default_rng(seed)
	top_k = int(cfg.get("top_k", 150)) ; K_eval = int(cfg.get("K_eval", 80))
	call_de = cfg["call_de"] ; grid4 = cfg["grid4"]
	with out.open("w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["num_sigs", "snr_db", "K", "recall_at_k"])
		for ns in num_sigs_list:
			freqs = np.linspace(500.0, 3500.0, int(ns), endpoint=False)
			freqs += rng.uniform(-1.5, 1.5, size=int(ns))
			signals = []
			metas = []
			for i in range(int(ns)):
				call_to = f"C{i:02d}AAA"
				x, _ = make_clean_signal(call_to, call_de, grid4, sr, float(freqs[i]))
				y = apply_awgn(x, snr_db, rng)
				signals.append(y)
				metas.append({"f": float(freqs[i])})
				slot = mix_signals(signals, [0.0] * int(ns))
			cands, nfft, hop = find_sync_candidates_stft(slot, sr, top_k=top_k)
			bin_hz = sr / nfft if nfft > 0 else 6.25
			found = 0
			for m in metas:
				ok = False
				for c in cands[:K_eval]:
					f_est = (c.base_bin + c.frac) * bin_hz
					if abs(f_est - m["f"]) <= 6.25:
						ok = True
						break
				found += int(ok)
			recall = found / float(int(ns)) if ns else 0.0
			w.writerow([ns, snr_db, K_eval, f"{recall:.6f}"])
	return out


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("scenario", choices=list(SCENARIOS.keys()))
	args = p.parse_args()
	cfg = SCENARIOS[args.scenario]
	if args.scenario == "awgn_snr_sweep":
		path = run_awgn_snr_sweep(cfg)
		print(f"wrote {path}")
	elif args.scenario == "cfo_sweep":
		path = run_cfo_sweep(cfg)
		print(f"wrote {path}")
	elif args.scenario == "occupancy":
		path = run_occupancy(cfg)
		print(f"wrote {path}")


if __name__ == "__main__":
	main()