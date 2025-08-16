from __future__ import annotations

SCENARIOS = {
	"awgn_snr_sweep": {
		"sr": 12000.0,
		"base_freq_hz": 1500.0,
		"snr_db": [-22, -20, -18, -16, -14, -12],
		"trials": 10,
		"seed": 123,
		"call_to": "K1ABC",
		"call_de": "W9XYZ",
		"grid4": "FN20",
	},
	"cfo_sweep": {
		"sr": 12000.0,
		"base_freq_hz": 1500.0,
		"cfo_hz": [-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5],
		"snr_db": -16,
		"trials": 10,
		"seed": 456,
		"call_to": "K1ABC",
		"call_de": "W9XYZ",
		"grid4": "FN20",
	},
	"occupancy": {
		"sr": 12000.0,
		"num_sigs": [10, 20],
		"snr_db": -16,
		"top_k": 120,
		"K_eval": 80,
		"seed": 789,
		"call_de": "W9XYZ",
		"grid4": "FN20",
	},
}