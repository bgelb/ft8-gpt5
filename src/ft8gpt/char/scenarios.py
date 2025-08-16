from __future__ import annotations

from pathlib import Path
import json


def get_default_scenarios() -> dict:
    return {
        "awgn_snr_sweep": {
            "sr": 12000.0,
            "base_freq_hz": 1500.0,
            "snr_db": [-22, -20, -18, -16, -14, -12],
            "trials": 20,
            "seed": 123,
        },
        "cfo_sweep": {
            "sr": 12000.0,
            "base_freq_hz": 1500.0,
            "cfo_hz": [-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5],
            "snr_db": -16,
            "trials": 10,
        },
        "occupancy": {
            "sr": 12000.0,
            "num_sigs": [10, 20, 40, 80],
            "snr_db": -16,
            "top_k": 150,
            "K_eval": 80,
        },
    }


def load_scenarios(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data