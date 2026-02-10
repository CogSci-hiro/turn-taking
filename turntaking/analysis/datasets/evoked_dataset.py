"""turntaking.analysis.datasets.evoked_dataset

ERP / TFR group-level dataset construction.

This module should build datasets from lists of epochs files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import pandas as pd

Kind = Literal["erp", "tfr"]


@dataclass
class EvokedDatasetResult:
    """Outputs of group-level evoked dataset construction."""

    evokeds_cond1: list[mne.Evoked]
    evokeds_cond2: list[mne.Evoked]
    difference: mne.Evoked
    metadata: pd.DataFrame
    n_trials: pd.DataFrame


def build_evoked_dataset(
    epoch_paths: list[Path],
    *,
    kind: Kind,
    contrast: str,
    selection_params: object,
    band: tuple[float, float] | None = None,
) -> EvokedDatasetResult:
    """Build group-level evoked dataset."""
    raise NotImplementedError
