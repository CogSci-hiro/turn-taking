
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
    "ErpConditionNames",
    "get_erp_condition_names",
    "write_erp_outputs",
]


@dataclass(frozen=True)
class ErpConditionNames:
    """Condition labels used in ERP filenames for a given contrast."""

    cond_1: str
    cond_2: str


def get_erp_condition_names(contrast: str) -> ErpConditionNames:
    """
    Map ERP contrast names to condition labels used in output filenames.

    Mapping
    -------
    - ``duration`` -> ``long`` vs ``short``
    - ``latency``  -> ``fast`` vs ``slow``
    """
    if contrast == "duration":
        return ErpConditionNames(cond_1="long", cond_2="short")
    if contrast == "latency":
        return ErpConditionNames(cond_1="fast", cond_2="slow")
    raise ValueError(f"Unknown contrast: {contrast!r}. Expected 'duration' or 'latency'.")


def write_erp_outputs(
    out_dir: Path,
    *,
    contrast: str,
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
    evoked_data: np.ndarray,
    n_trials: pd.DataFrame,
    results: Mapping[str, Any],
    offsets: pd.DataFrame,
    overwrite: bool = True,
) -> None:
    """
    Write the full ERP artifact contract for one contrast.

    This function is a low-level file sink. It only validates write-time
    invariants and persists already computed in-memory artifacts.
    """
    ensure_dir_exists(out_dir)
    names = get_erp_condition_names(contrast)
    paths = _erp_output_paths(out_dir, names)
    _validate_evoked_lengths(evokeds_cond_1, evokeds_cond_2, evokeds_difference)

    _write_evoked_fifs(
        path_c1=paths["cond_1"],
        path_c2=paths["cond_2"],
        path_diff=paths["difference"],
        evokeds_cond_1=evokeds_cond_1,
        evokeds_cond_2=evokeds_cond_2,
        evokeds_difference=evokeds_difference,
        overwrite=overwrite,
    )
    save_array_nd(evoked_data, paths["evoked_data"])
    save_dataframe_csv(n_trials, paths["n_trials"])
    save_dataframe_csv(offsets, paths["offsets"])
    _write_results_hdf5(paths["results"], results, overwrite=overwrite)
    _assert_required_outputs_exist(paths, out_dir)


def _erp_output_paths(out_dir: Path, names: ErpConditionNames) -> dict[str, Path]:
    return {
        "difference": out_dir / "difference_ave.fif",
        "cond_1": out_dir / f"{names.cond_1}_ave.fif",
        "cond_2": out_dir / f"{names.cond_2}_ave.fif",
        "evoked_data": out_dir / "evoked-data.npy",
        "n_trials": out_dir / "n_trials.csv",
        "results": out_dir / "metadata.hdf5",
        "offsets": out_dir / "offsets.csv",
    }


def _validate_evoked_lengths(
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
) -> None:
    if len(evokeds_cond_1) != len(evokeds_cond_2):
        raise ValueError(
            f"Cond lists must match in length: {len(evokeds_cond_1)} vs {len(evokeds_cond_2)}"
        )
    if len(evokeds_difference) != len(evokeds_cond_1):
        raise ValueError(
            f"Difference list must match subject count: {len(evokeds_difference)} vs {len(evokeds_cond_1)}"
        )


def _write_evoked_fifs(
    *,
    path_c1: Path,
    path_c2: Path,
    path_diff: Path,
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
    overwrite: bool,
) -> None:
    mne.write_evokeds(path_c1.as_posix(), list(evokeds_cond_1), overwrite=overwrite)
    mne.write_evokeds(path_c2.as_posix(), list(evokeds_cond_2), overwrite=overwrite)
    mne.write_evokeds(path_diff.as_posix(), list(evokeds_difference), overwrite=overwrite)


def _write_results_hdf5(path_results: Path, results: Mapping[str, Any], *, overwrite: bool) -> None:
    if path_results.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite results: {path_results}")
    save_hdf5_dataset(path_results, results)


def _assert_required_outputs_exist(paths: Mapping[str, Path], out_dir: Path) -> None:
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing ERP outputs after write: {missing} (out_dir={out_dir})")
