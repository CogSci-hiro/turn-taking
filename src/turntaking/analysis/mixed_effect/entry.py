
"""Service entrypoint for mixed-effect table generation."""

import argparse
from pathlib import Path
from typing import Any, Sequence

from turntaking.analysis.constants import ANTERIOR, POSTERIOR
from turntaking.analysis.mixed_effect.make_table import write_mixed_effect_table
from turntaking.analysis.mixed_effect.schema import MixedEffectTableParams
from turntaking.analysis.selection import SelectionParams


def run_mixed_effect(args: argparse.Namespace, cfg: Any) -> None:
    epoch_dir = Path(cfg.io.epoch_dir)
    out_dir = Path(cfg.io.out_dir)
    out_csv = Path(args.out_csv) if getattr(args, "out_csv", None) is not None else out_dir / "mixed_effect" / "table.csv"

    me_cfg = cfg.analysis.mixed
    params = MixedEffectTableParams(
        tw1_tmin=float(me_cfg.tw1[0]),
        tw1_tmax=float(me_cfg.tw1[1]),
        tw2_tmin=float(me_cfg.tw2[0]),
        tw2_tmax=float(me_cfg.tw2[1]),
        baseline_tmin=float(me_cfg.baseline[0]),
        baseline_tmax=float(me_cfg.baseline[1]),
        selection=SelectionParams(
            min_latency=float(me_cfg.selection.min_latency),
            max_latency=float(me_cfg.selection.max_latency),
            min_self_duration=float(me_cfg.selection.min_self_duration),
        ),
    )
    write_mixed_effect_table(
        epoch_dir=epoch_dir,
        out_csv=out_csv,
        params=params,
        anterior_picks=list(ANTERIOR),  # type: Sequence[str]
        posterior_picks=list(POSTERIOR),  # type: Sequence[str]
    )
