# Expose key helpers for external import convenience
from .synth_utils import make_bits91_from_msg, make_clean_signal
from .channel import apply_awgn, apply_drift, mix_signals
from .metrics import recall_at_k, rmse, bit_mutual_information

__all__ = [
    "make_bits91_from_msg",
    "make_clean_signal",
    "apply_awgn",
    "apply_drift",
    "mix_signals",
    "recall_at_k",
    "rmse",
    "bit_mutual_information",
]