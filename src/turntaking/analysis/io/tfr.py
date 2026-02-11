# src/turntaking/analysis/io/tfr.py
# =============================================================================
#                     ########################################
#                     #        TFR OUTPUT CONTRACT (I/O)      #
#                     ########################################
# =============================================================================
#
# Writes the per-contrast × per-band induced-TFR artifacts to:
#   {io.out_dir}/tfr/{contrast}/{band}/
#
# Files (per contrast × band):
#   1) difference_ave.fif
#   2) induced-data.npy
#   3) {long|fast}_ave.fif
#   4) {short|slow}_ave.fif
#   5) n_trials.csv
#   6) metadata.hdf5
#
# Naming rule:
#   contrast="duration": long vs short
#   contrast="latency":  fast vs slow
#
# Notes
# -----
# This "TFR" is an ERP-like induced amplitude envelope (bandpass + Hilbert),
# saved as Evoked objects for compatibility with MNE adjacency / plotting.
#
# =============================================================================

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from turntaking.analysis.io.core import save_hdf5, save_npy, save_table_csv


# =============================================================================
#                     ########################################
#                     #          CONDITION NAME MAPPING        #
#                     ########################################
# =============================================================================

@dataclass(frozen=True)
class TfrConditionNames:
    cond_1: str
    cond_2: str


def get_tfr_condition_names(contrast: str) -> TfrConditionNames:
    """
    Map TFR contrast -> condition names used in filenames.

    Parameters
    ----------
    contrast
        "duration" or "latency".

    Returns
    -------
    TfrConditionNames
        For duration: (long, short)
        For latency:  (fast, slow)

    Usage example
    -------------
        names = get_tfr_condition_names("latency")
        assert names.cond_1 == "fast"
        assert names.cond_2 == "slow"
    """
    if contrast == "duration":
        return TfrConditionNames(cond_1="long", cond_2="short")
    if contrast == "latency":
        return TfrConditionNames(cond_1="fast", cond_2="slow")
    raise ValueError(f"Unknown contrast: {contrast!r}. Expected 'duration' or 'latency'.")


# =============================================================================
#                     ########################################
#                     #             PUBLIC WRITER              #
#                     ########################################
# =============================================================================

def write_tfr_outputs(
    out_dir: Path,
    *,
    contrast: str,
    band: str,
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
    induced_data: np.ndarray,
    n_trials: pd.DataFrame,
    metadata: Mapping[str, Any],
    overwrite: bool = True,
) -> None:
    """
    Write all induced-TFR artifacts for one contrast × one band.

    Parameters
    ----------
    out_dir
        Output directory for the contrast×band, e.g. ".../tfr/duration/alpha".
    contrast
        "duration" or "latency".
    band
        Band label used in the directory level (e.g., "alpha", "beta").
    evokeds_cond_1
        Per-subject Evoked list for condition 1 (long/fast).
    evokeds_cond_2
        Per-subject Evoked list for condition 2 (short/slow).
    evokeds_difference
        Per-subject Evoked list for the difference wave (cond_1 - cond_2).
    induced_data
        Array saved as "induced-data.npy". Recommended shape:
        (n_subjects, 3, n_channels, n_times) with order [cond_1, cond_2, diff].
    n_trials
        Trial counts table saved to "n_trials.csv".

        Example table format
        --------------------
        | subject  | long | short |
        |----------|------|-------|
        | sub-004  | 120  | 118   |
    metadata
        Mapping saved to "metadata.hdf5" (dataset-level metadata, not cluster results).
    overwrite
        Overwrite FIF/HDF5 outputs if present.

    Returns
    -------
    None

    Usage example
    -------------
        write_tfr_outputs(
            Path("/tmp/tfr/duration/alpha"),
            contrast="duration",
            band="alpha",
            evokeds_cond_1=ev_long,
            evokeds_cond_2=ev_short,
            evokeds_difference=ev_diff,
            induced_data=induced_data,
            n_trials=n_trials,
            metadata={"band": "alpha", "times": times},
        )
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = get_tfr_condition_names(contrast)

    # Filenames (single source of truth)
    path_diff = out_dir / "difference_ave.fif"
    path_c1 = out_dir / f"{names.cond_1}_ave.fif"
    path_c2 = out_dir / f"{names.cond_2}_ave.fif"
    path_induced_npy = out_dir / "induced-data.npy"
    path_n_trials = out_dir / "n_trials.csv"
    path_metadata = out_dir / "metadata.hdf5"

    # Basic validation
    if len(evokeds_cond_1) != len(evokeds_cond_2):
        raise ValueError(
            f"Cond lists must match in length: {len(evokeds_cond_1)} vs {len(evokeds_cond_2)}"
        )
    if len(evokeds_difference) != len(evokeds_cond_1):
        raise ValueError(
            f"Difference list must match subject count: {len(evokeds_difference)} vs {len(evokeds_cond_1)}"
        )
    if induced_data.ndim < 3:
        raise ValueError(f"induced_data looks wrong (expected >=3 dims), got shape={induced_data.shape}")

    # Enrich metadata deterministically (does not overwrite user-provided keys)
    meta = dict(metadata)
    meta.setdefault("kind", "tfr")
    meta.setdefault("contrast", str(contrast))
    meta.setdefault("band", str(band))
    meta.setdefault("condition_1", names.cond_1)
    meta.setdefault("condition_2", names.cond_2)
    meta.setdefault("induced_data_shape", np.array(induced_data.shape, dtype=int))

    # Write FIF files
    mne.write_evokeds(path_c1.as_posix(), list(evokeds_cond_1), overwrite=overwrite)
    mne.write_evokeds(path_c2.as_posix(), list(evokeds_cond_2), overwrite=overwrite)
    mne.write_evokeds(path_diff.as_posix(), list(evokeds_difference), overwrite=overwrite)

    # Write numeric/table outputs
    save_npy(induced_data, path_induced_npy)
    save_table_csv(n_trials, path_n_trials)

    if path_metadata.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite metadata: {path_metadata}")
    save_hdf5(path_metadata, meta)

    # Final sanity check: Snakemake completion criteria
    required = [path_diff, path_induced_npy, path_c1, path_c2, path_n_trials, path_metadata]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing TFR outputs after write: {missing} (out_dir={out_dir})")
