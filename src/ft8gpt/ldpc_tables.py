from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import LDPC_N, LDPC_M
from .ldpc_tables_embedded import get_parity_matrices


def load_parity_from_file(path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible signature; ignores path and returns embedded matrices.
    """
    return get_parity_matrices()


