from pathlib import Path
from typing import Sequence

import pandas as pd
import mne

from .constants import (
    DROP_COLUMN_EXACT,
    DROP_COLUMN_PREFIXES,
    DROP_COLUMN_SUBSTRINGS,
    EPOCH_FILE_PATTERN,
)
from .eeg_features import compute_run_eeg_features
from .schema import MixedEffectTableParams
from turntaking.analysis.selection import select_epochs


def make_mixed_effect_table(
    epoch_dir: Path,
    *,
    params: MixedEffectTableParams,
    anterior_picks: Sequence[str],
    posterior_picks: Sequence[str],
) -> pd.DataFrame:
    """
    Build the trial-level CSV table consumed by R mixed-effects models.

    Parameters
    ----------
    epoch_dir
        Directory containing epoch files.
    params
        Table construction parameters (windows + selection thresholds).
    anterior_picks, posterior_picks
        ROI channel lists.

    Returns
    -------
    pd.DataFrame
        Trial-level table (one row per epoch/trial), with:
        - EEG summaries (ERP means + alpha/beta power in tw1/tw2; baseline ERP)
        - behavioral metadata columns from epochs.metadata
        - subject as "sub-XXX" (string)
        - run as int

    Example table
    -------------
    | subject | run | latency | self_duration | other_duration | tw1_mean_anterior |
    |---|---:|---:|---:|---:|---:|
    | sub-004 | 3 | 0.182 | 1.240 | 0.980 | -0.83 |

    Usage example
    -------------
        df = make_mixed_effect_table(
            Path(".../epochs"),
            params=params,
            anterior_picks=ANTERIOR,
            posterior_picks=POSTERIOR,
        )
    """
    epoch_files = sorted([p for p in epoch_dir.iterdir() if p.is_file()])
    if len(epoch_files) == 0:
        raise FileNotFoundError(f"No files found in: {epoch_dir}")

    df_list: list[pd.DataFrame] = []

    for epoch_file in epoch_files:
        match = EPOCH_FILE_PATTERN.search(epoch_file.stem)
        if match is None:
            continue

        subject_id = f"sub-{match.group(1)}"
        run = int(match.group(2))

        epochs = mne.read_epochs(epoch_file, preload=False)
        epochs = select_epochs(epochs, params.selection)

        if epochs.metadata is None:
            raise ValueError(f"epochs.metadata is None for file: {epoch_file}")

        metadata = epochs.metadata.copy()

        eeg_df = compute_run_eeg_features(
            epochs,
            tw1_tmin=params.tw1_tmin,
            tw1_tmax=params.tw1_tmax,
            tw2_tmin=params.tw2_tmin,
            tw2_tmax=params.tw2_tmax,
            baseline_tmin=params.baseline_tmin,
            baseline_tmax=params.baseline_tmax,
            anterior_picks=anterior_picks,
            posterior_picks=posterior_picks,
        )

        # IMPORTANT: avoid reset_index artifacts and duplicated index columns.
        # Align by row order (epochs are already aligned with metadata).
        out = pd.concat([eeg_df, metadata.reset_index(drop=True)], axis=1)

        out["subject"] = subject_id
        out["run"] = run

        out = _drop_unwanted_columns(out)

        df_list.append(out)

    if len(df_list) == 0:
        raise RuntimeError(
            f"No valid epoch files matched pattern in {epoch_dir}. "
            f"Pattern: {EPOCH_FILE_PATTERN.pattern}"
        )

    combined = pd.concat(df_list, axis=0, ignore_index=True)
    combined = _drop_unwanted_columns(combined)  # safety pass
    return combined


def write_mixed_effect_table(
    epoch_dir: Path,
    *,
    out_csv: Path,
    params: MixedEffectTableParams,
    anterior_picks: Sequence[str],
    posterior_picks: Sequence[str],
) -> None:
    """
    Convenience wrapper that writes the mixed-effect table to CSV.

    Usage example
    -------------
        write_mixed_effect_table(
            epoch_dir=Path(".../epochs"),
            out_csv=Path(".../mixed_effect/table.csv"),
            params=params,
            anterior_picks=ANTERIOR,
            posterior_picks=POSTERIOR,
        )
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = make_mixed_effect_table(
        epoch_dir,
        params=params,
        anterior_picks=anterior_picks,
        posterior_picks=posterior_picks,
    )
    df.to_csv(out_csv, index=False)


def _drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Drop explicit known bad columns
    keep_cols: list[str] = []
    for col in df.columns:
        if col in DROP_COLUMN_EXACT:
            continue
        if any(col.startswith(prefix) for prefix in DROP_COLUMN_PREFIXES):
            continue
        if any(substr in col for substr in DROP_COLUMN_SUBSTRINGS):
            continue
        keep_cols.append(col)

    filtered = df.loc[:, keep_cols].copy()

    # Drop any duplicated columns (defensive)
    filtered = filtered.loc[:, ~filtered.columns.duplicated()].copy()

    return filtered
