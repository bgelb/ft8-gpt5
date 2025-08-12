from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .constants import LDPC_N, LDPC_M


@dataclass(frozen=True)
class BeliefPropagationConfig:
    max_iterations: int = 30
    early_stop_no_improve: int = 5


def bp_decode(log_likelihood_174: NDArray[np.float64], config: BeliefPropagationConfig) -> Tuple[int, NDArray[np.uint8]]:
    """
    Minimal placeholder for LDPC(174,91) belief propagation decoder.
    For now returns a hard decision on the input and zero errors count.
    """
    x = (log_likelihood_174 > 0).astype(np.uint8)
    # TODO: implement true BP using sparse Nm/Mn.
    errors = 0
    return errors, x


