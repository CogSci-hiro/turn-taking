
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Prevent numba cache issues triggered during mne import in sandboxed environments.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import mne


# Ensure local package imports work without requiring an editable install.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_epochs() -> mne.Epochs:
    """Return a tiny deterministic Epochs object with required metadata columns."""
    rng = np.random.default_rng(0)
    n_epochs, n_channels, n_times = 8, 2, 16
    sfreq = 32.0

    info = mne.create_info(ch_names=["Cz", "Pz"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    data = rng.normal(size=(n_epochs, n_channels, n_times))
    epochs = mne.EpochsArray(data, info, tmin=-0.25, verbose=False)

    epochs.metadata = pd.DataFrame(
        {
            "latency": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            "self_duration": [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24],
            "other_duration": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37],
        }
    )
    return epochs
