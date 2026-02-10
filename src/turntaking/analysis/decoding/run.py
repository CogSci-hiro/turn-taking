"""turntaking.analysis.decoding.run

Decoding execution and scoring.
"""


import numpy as np


def run_decoding(
    X: np.ndarray,
    y: np.ndarray,
    *,
    clf_cfg: dict,
    cv_cfg: dict,
) -> dict:
    """Run decoding and return scores and diagnostics."""
    raise NotImplementedError
