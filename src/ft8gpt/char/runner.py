from __future__ import annotations

import numpy as np


def run_decoder(slot: np.ndarray, sample_rate_hz: float, which: str = "ft8gpt"):
    if which == "ft8gpt":
        from ft8gpt.decoder_e2e import decode_block
        return decode_block(slot, sample_rate_hz)
    elif which == "external":
        raise NotImplementedError("External decoder hook not implemented in this repo")
    else:
        raise ValueError(f"Unknown decoder selector: {which}")