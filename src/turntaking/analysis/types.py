from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpochBundle:
    """In-memory representation of a set of epochs to analyze."""

    epochs: mne.Epochs
    metadata: pd.DataFrame
    info: mne.Info

