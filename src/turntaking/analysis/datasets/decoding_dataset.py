"""
Decoding dataset construction (X, y).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_decoding_dataset(
    epoch_paths: list[Path],
    *,
    contrast: str,
    kind: str,
    selection_params: object,
    band: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build decoding arrays and metadata.
    """
    raise NotImplementedError
