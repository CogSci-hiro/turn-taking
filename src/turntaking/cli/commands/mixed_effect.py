
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from turntaking.analysis.mixed_effect.make_table import write_mixed_effect_table
from turntaking.analysis.mixed_effect.schema import MixedEffectTableParams
from turntaking.analysis.selection import SelectionParams


@dataclass(frozen=True)
class MixedEffectCliArgs:
    out_csv: Path | None


def add_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "mixed-effect",
        help="Export trial-level CSV table for R mixed-effects models.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to workflow config YAML.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional override for output CSV path.",
    )


def run(args, cfg) -> None:
    cli_args = MixedEffectCliArgs(out_csv=args.out_csv)

    epoch_dir = Path(cfg.io.epoch_dir)
    out_dir = Path(cfg.io.out_dir)
    out_csv = cli_args.out_csv if cli_args.out_csv is not None else out_dir / "mixed_effect" / "table.csv"

    me_cfg = cfg.analysis.mixed

    tw1_tmin, tw1_tmax = float(me_cfg.tw1[0]), float(me_cfg.tw1[1])
    tw2_tmin, tw2_tmax = float(me_cfg.tw2[0]), float(me_cfg.tw2[1])
    baseline_tmin, baseline_tmax = float(me_cfg.baseline[0]), float(me_cfg.baseline[1])

    selection = SelectionParams(
        min_latency=float(me_cfg.selection.min_latency),
        max_latency=float(me_cfg.selection.max_latency),
        min_self_duration=float(me_cfg.selection.min_self_duration),
    )

    params = MixedEffectTableParams(
        tw1_tmin=tw1_tmin,
        tw1_tmax=tw1_tmax,
        tw2_tmin=tw2_tmin,
        tw2_tmax=tw2_tmax,
        baseline_tmin=baseline_tmin,
        baseline_tmax=baseline_tmax,
        selection=selection,
    )

    # ROIs are defined in code (not config) to stay consistent across analyses
    from turntaking.analysis.constants import ANTERIOR, POSTERIOR  # adjust import path if needed

    anterior_picks: Sequence[str] = list(ANTERIOR)
    posterior_picks: Sequence[str] = list(POSTERIOR)

    write_mixed_effect_table(
        epoch_dir=epoch_dir,
        out_csv=out_csv,
        params=params,
        anterior_picks=anterior_picks,
        posterior_picks=posterior_picks,
    )
