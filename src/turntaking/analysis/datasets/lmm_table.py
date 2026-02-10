"""turntaking.analysis.datasets.lmm_table

Construct a tidy DataFrame for linear mixed-effects models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_lmm_table(
    epoch_paths: list[Path],
    *,
    selection_params: object,
    roi_definitions: dict,
    time_windows: dict,
    bands: dict,
) -> pd.DataFrame:
    """Build tidy dataframe for linear mixed-effects models."""
    raise NotImplementedError
