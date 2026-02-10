# src/turntaking/analysis/io/erp.py
# =============================================================================
#                     ########################################
#                     #        ERP OUTPUT CONTRACT (I/O)       #
#                     ########################################
# =============================================================================
#
# Writes the 7 per-contrast ERP artifacts to:
#   {io.out_dir}/erp/{contrast}/
#
# Files (per contrast):
#   1) difference_ave.fif
#   2) evoked-data.npy
#   3) {long|fast}_ave.fif
#   4) {short|slow}_ave.fif
#   5) n_trials.csv
#   6) results.hdf5
#   7) offsets.csv
#
# Naming rule:
#   contrast="duration": long vs short
#   contrast="latency":  fast vs slow
#
# =============================================================================


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import mne


# =============================================================================
#                     ########################################
#                     #          CONDITION NAME MAPPING        #
#                     ########################################
# =============================================================================

@dataclass(frozen=True)
class ErpConditionNames:
    cond_1: str
    cond_2: str


def get_erp_condition_names(contrast: str) -> ErpConditionNames:
    """
    Map ERP contrast -> condition names used in filenames.

    Parameters
    ----------
    contrast
        "duration" or "latency".

    Returns
    -------
    ErpConditionNames
        For duration: (long, short)
        For latency:  (fast, slow)

    Usage example
    -------------
        names = get_erp_condition_names("duration")
        assert names.cond_1 == "long"
        assert names.cond_2 == "short"
    """
    if contrast == "duration":
        return ErpConditionNames(cond_1="long", cond_2="short")
    if contrast == "latency":
        return ErpConditionNames(cond_1="fast", cond_2="slow")
    raise ValueError(f"Unknown contrast: {contrast!r}. Expected 'duration' or 'latency'.")


# =============================================================================
#                     ########################################
#                     #              GENERIC I/O               #
#                     ########################################
# =============================================================================

def _save_table_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _save_hdf5(path: Path, results: Mapping[str, Any]) -> None:
    """
    Minimal HDF5 serializer: arrays -> datasets, scalars -> attributes.
    """
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in results.items():
            if value is None:
                continue
            if isinstance(value, (int, float, str, bytes, np.integer, np.floating)):
                f.attrs[str(key)] = value
            else:
                f.create_dataset(str(key), data=np.asarray(value))


# =============================================================================
#                     ########################################
#                     #             PUBLIC WRITER              #
#                     ########################################
# =============================================================================

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
    Write all ERP artifacts for one contrast.

    Parameters
    ----------
    out_dir
        Output directory for the contrast, e.g. ".../erp/duration".
    contrast
        "duration" or "latency".
    evokeds_cond_1
        Per-subject Evoked list for condition 1 (long/fast).
    evokeds_cond_2
        Per-subject Evoked list for condition 2 (short/slow).
    evokeds_difference
        Per-subject Evoked list for the difference wave.
    evoked_data
        Array to save as "evoked-data.npy" (semantics defined by upstream code).
    n_trials
        Trial counts table saved to "n_trials.csv".

        Example table format
        --------------------
        | subject  | n_cond_1 | n_cond_2 |
        |----------|----------|----------|
        | sub-004  |  120     |  118     |
    results
        Mapping saved to "results.hdf5".
    offsets
        Legacy offsets table (as in old script). Saved to "offsets.csv".
    overwrite
        Overwrite FIF/HDF5 outputs if present.

    Returns
    -------
    None

    Usage example
    -------------
        write_erp_outputs(
            Path("/tmp/erp/duration"),
            contrast="duration",
            evokeds_cond_1=ev_long,
            evokeds_cond_2=ev_short,
            evokeds_difference=ev_diff,
            evoked_data=evoked_data,
            n_trials=n_trials,
            results={"times": times, "pvals": pvals},
            offsets=offsets,
        )
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = get_erp_condition_names(contrast)

    # Filenames (single source of truth)
    path_diff = out_dir / "difference_ave.fif"
    path_c1 = out_dir / f"{names.cond_1}_ave.fif"
    path_c2 = out_dir / f"{names.cond_2}_ave.fif"
    path_evoked_npy = out_dir / "evoked-data.npy"
    path_n_trials = out_dir / "n_trials.csv"
    path_results = out_dir / "results.hdf5"
    path_offsets = out_dir / "offsets.csv"

    # Basic validation (cheap, catches lots of silent bugs)
    if len(evokeds_cond_1) != len(evokeds_cond_2):
        raise ValueError(
            f"Cond lists must match in length: {len(evokeds_cond_1)} vs {len(evokeds_cond_2)}"
        )
    if len(evokeds_difference) != len(evokeds_cond_1):
        raise ValueError(
            f"Difference list must match subject count: {len(evokeds_difference)} vs {len(evokeds_cond_1)}"
        )

    # Write FIF files (lists of Evoked -> one .fif each)
    mne.write_evokeds(path_c1.as_posix(), list(evokeds_cond_1), overwrite=overwrite)
    mne.write_evokeds(path_c2.as_posix(), list(evokeds_cond_2), overwrite=overwrite)
    mne.write_evokeds(path_diff.as_posix(), list(evokeds_difference), overwrite=overwrite)

    # Write numeric/table outputs
    _save_npy(evoked_data, path_evoked_npy)
    _save_table_csv(n_trials, path_n_trials)
    _save_table_csv(offsets, path_offsets)
    if path_results.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite results: {path_results}")
    _save_hdf5(path_results, results=results)

    # Final sanity check: Snakemake completion criteria
    required = [path_diff, path_evoked_npy, path_c1, path_c2, path_n_trials, path_results, path_offsets]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing ERP outputs after write: {missing} (out_dir={out_dir})")
