from __future__ import annotations

"""
TFR domain I/O helpers.

This module defines the file-output contract for induced TFR artifacts and
contains only I/O responsibilities (naming, validation, persistence).
Computation remains in ``turntaking.analysis.tfr.core``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from turntaking.analysis.utils.io import (
    ensure_dir_exists,
    save_array_nd,
    save_dataframe_csv,
    save_hdf5_dataset,
)

__all__ = [
    "TfrConditionNames",
    "get_tfr_condition_names",
    "write_tfr_outputs",
]


@dataclass(frozen=True)
class TfrConditionNames:
    cond_1: str
    cond_2: str


def get_tfr_condition_names(contrast: str) -> TfrConditionNames:
    """Map TFR contrast names to condition labels used in output filenames."""
    if contrast == "duration":
        return TfrConditionNames(cond_1="long", cond_2="short")
    if contrast == "latency":
        return TfrConditionNames(cond_1="fast", cond_2="slow")
    raise ValueError(f"Unknown contrast: {contrast!r}. Expected 'duration' or 'latency'.")


def _build_output_paths(out_dir: Path, names: TfrConditionNames) -> dict[str, Path]:
    return {
        "difference": out_dir / "difference_ave.fif",
        "condition_1": out_dir / f"{names.cond_1}_ave.fif",
        "condition_2": out_dir / f"{names.cond_2}_ave.fif",
        "induced": out_dir / "induced-data.npy",
        "n_trials": out_dir / "n_trials.csv",
        "metadata": out_dir / "metadata.hdf5",
    }


def _validate_write_inputs(
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
    induced_data: np.ndarray,
) -> None:
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


def _enrich_metadata(
    metadata: Mapping[str, Any],
    *,
    contrast: str,
    band: str,
    names: TfrConditionNames,
    induced_data: np.ndarray,
) -> dict[str, Any]:
    meta = dict(metadata)
    meta.setdefault("kind", "tfr")
    meta.setdefault("contrast", str(contrast))
    meta.setdefault("band", str(band))
    meta.setdefault("condition_1", names.cond_1)
    meta.setdefault("condition_2", names.cond_2)
    meta.setdefault("induced_data_shape", np.array(induced_data.shape, dtype=int))
    return meta


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
    """Write induced-TFR artifacts for one contrast × one band."""
    out_dir = ensure_dir_exists(out_dir)
    names = get_tfr_condition_names(contrast)
    paths = _build_output_paths(out_dir, names)
    _validate_write_inputs(evokeds_cond_1, evokeds_cond_2, evokeds_difference, induced_data)
    meta = _enrich_metadata(
        metadata,
        contrast=contrast,
        band=band,
        names=names,
        induced_data=induced_data,
    )

    mne.write_evokeds(paths["condition_1"].as_posix(), list(evokeds_cond_1), overwrite=overwrite)
    mne.write_evokeds(paths["condition_2"].as_posix(), list(evokeds_cond_2), overwrite=overwrite)
    mne.write_evokeds(paths["difference"].as_posix(), list(evokeds_difference), overwrite=overwrite)

    save_array_nd(induced_data, paths["induced"])
    save_dataframe_csv(n_trials, paths["n_trials"])

    if paths["metadata"].exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite metadata: {paths['metadata']}")
    save_hdf5_dataset(paths["metadata"], meta)

    required = list(paths.values())
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing TFR outputs after write: {missing} (out_dir={out_dir})")
