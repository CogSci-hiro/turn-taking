from dataclasses import dataclass
from pathlib import Path

from turntaking.analysis.selection import SelectionParams


@dataclass(frozen=True)
class MixedEffectTableParams:
    """
    Parameters controlling mixed-effect table construction.

    Notes
    -----
    This stage *only* constructs the trial-level table for R.
    It does not fit any model.

    Usage example
    -------------
        params = MixedEffectTableParams(
            tw1_tmin=0.05, tw1_tmax=0.15,
            tw2_tmin=0.20, tw2_tmax=0.35,
            baseline_tmin=-0.20, baseline_tmax=0.0,
            min_latency=0.05, max_latency=2.0,
            min_response_duration=0.10,
        )
    """
    tw1_tmin: float
    tw1_tmax: float
    tw2_tmin: float
    tw2_tmax: float
    baseline_tmin: float
    baseline_tmax: float
    selection: SelectionParams


@dataclass(frozen=True)
class MixedEffectTablePaths:
    """
    IO paths for mixed-effect table export.

    Usage example
    -------------
        paths = MixedEffectTablePaths(
            epoch_dir=Path(".../epochs"),
            out_csv=Path(".../mixed_effect/table.csv"),
        )
    """

    epoch_dir: Path
    out_csv: Path
