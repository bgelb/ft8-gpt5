from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from ft8gpt.char.scenarios import get_default_scenarios, load_scenarios
from ft8gpt.char.synth_utils import make_clean_signal
from ft8gpt.char.channel import apply_awgn
from ft8gpt.char.runner import run_decoder


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
    else:
        raise SystemExit("This CLI currently supports only 'awgn_snr_sweep'. Extend as needed.")


if __name__ == "__main__":
    main()