from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ======================================================================================================================
# Constants
# ======================================================================================================================

PLOT_WINDOW_S: float = 4.0
ANALYSIS_WINDOW_S: float = 1.0

DEFAULT_OUTPUT_NAME: str = "turn_table.csv"
TSV_GLOB_PATTERN: str = "*_metadata.tsv"


# ======================================================================================================================
# Public API
# ======================================================================================================================

@dataclass(frozen=True)
class TurnTablePaths:
    beh_dir: Path
    out_csv: Path


def build_turn_table(paths: TurnTablePaths) -> pd.DataFrame:
    """
    Build a canonical turn-level table for visualization.

    Parameters
    ----------
    paths
        Input/output paths.

    Returns
    -------
    pandas.DataFrame
        Turn-level table.

    Notes
    -----
    Expected TSV columns:
    - latency
    - self_duration
    - other_duration

    Output columns:
    - latency
    - self_duration
    - other_duration
    - in_plot_window        (latency in (-4, 4))
    - in_analysis_window    (latency in (-1, 1))
    - source_file           (filename only; helpful for debugging)
    """
    tsv_paths = sorted(paths.beh_dir.glob(TSV_GLOB_PATTERN))
    if len(tsv_paths) == 0:
        raise FileNotFoundError(f"No metadata TSV files found under: {paths.beh_dir} (pattern: {TSV_GLOB_PATTERN})")

    frames: list[pd.DataFrame] = []
    for tsv_path in tsv_paths:
        df = pd.read_csv(tsv_path, sep="\t")
        _require_columns(df, required=["latency", "self_duration", "other_duration"])

        out = pd.DataFrame(
            {
                "latency": df["latency"].astype(float),
                "self_duration": df["self_duration"].astype(float),
                "other_duration": df["other_duration"].astype(float),
                "source_file": tsv_path.name,
            }
        )

        out["in_plot_window"] = (out["latency"] > -PLOT_WINDOW_S) & (out["latency"] < PLOT_WINDOW_S)
        out["in_analysis_window"] = (out["latency"] > -ANALYSIS_WINDOW_S) & (out["latency"] < ANALYSIS_WINDOW_S)

        frames.append(out)

    turn_table = pd.concat(frames, axis=0, ignore_index=True)

    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
    turn_table.to_csv(paths.out_csv, index=False)

    return turn_table


# ======================================================================================================================
# Helpers
# ======================================================================================================================

def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")
