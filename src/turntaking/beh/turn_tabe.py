import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# ======================================================================================================================
# Constants
# ======================================================================================================================

PLOT_WINDOW_S: float = 4.0
ANALYSIS_WINDOW_S: float = 1.0


# ======================================================================================================================
# Public API
# ======================================================================================================================

def build_offsets_csv_from_metadata_tsvs(
    metadata_tsv_paths: Iterable[Path],
    out_csv: Path,
) -> pd.DataFrame:
    """
    Build offsets CSV used for behavior figure(s) from per-run metadata TSVs.

    Parameters
    ----------
    metadata_tsv_paths
        Iterable of metadata TSV paths (one per subject/run), e.g.
        /Volumes/work-4T/hyperscanning/derived/beh/metadata/sub-004_task-conversation_run-3_metadata.tsv
    out_csv
        Output CSV path.

    Returns
    -------
    pandas.DataFrame
        Concatenated offsets table written to disk.

    Notes
    -----
    Required TSV columns:
    - latency
    - self_duration
    - other_duration (allowed to be NaN)
    """
    rows: list[pd.DataFrame] = []
    for tsv_path in metadata_tsv_paths:
        df = pd.read_csv(tsv_path, sep="\t")

        _require_columns(df, required=["latency", "self_duration", "other_duration"])

        subject, run = _parse_subject_run(tsv_path)

        out = pd.DataFrame(
            {
                "subject": subject,
                "run": run,
                "latency": df["latency"].astype(float),
                "self_duration": df["self_duration"].astype(float),
                "other_duration": df["other_duration"].astype(float),
            }
        )

        out["in_plot_window"] = out["latency"].notna() & (out["latency"] > -PLOT_WINDOW_S) & (out["latency"] < PLOT_WINDOW_S)
        out["in_analysis_window"] = out["latency"].notna() & (out["latency"] > -ANALYSIS_WINDOW_S) & (out["latency"] < ANALYSIS_WINDOW_S)

        rows.append(out)

    all_df = pd.concat(rows, axis=0, ignore_index=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)

    return all_df


# ======================================================================================================================
# Helpers
# ======================================================================================================================

def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")


def _parse_subject_run(path: Path) -> tuple[str, str]:
    """
    Parse BIDS-ish subject/run from filename.

    Example: sub-004_task-conversation_run-3_metadata.tsv -> ("sub-004", "run-3")
    """
    name = path.name
    subj_m = re.search(r"(sub-\d+)", name)
    run_m = re.search(r"(run-\d+)", name)

    subject = subj_m.group(1) if subj_m else "unknown"
    run = run_m.group(1) if run_m else "unknown"
    return subject, run
