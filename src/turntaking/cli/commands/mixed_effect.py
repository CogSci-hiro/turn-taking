from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from turntaking.analysis.mixed_effect.make_table import write_mixed_effect_table
from turntaking.analysis.mixed_effect.schema import MixedEffectTableParams
from turntaking.analysis.selection import SelectionParams


@dataclass(frozen=True)
class MixedEffectCliArgs:
    """CLI args for the mixed-effect table export.

    Usage example
    -------------
        python -m turntaking.cli.main mixed-effect --config workflow/config.yaml
    """
    config: Path
    out_csv: Path | None


def add_subparser(subparsers) -> None:
    """
    Required entrypoint for CLI command modules.

    The dispatcher looks for `add_subparser()` in each command module.
    """
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
        help="Optional override for output CSV path (index is never written).",
    )

    parser.set_defaults(func=run)


def run(args) -> None:
    cli_args = MixedEffectCliArgs(config=args.config, out_csv=args.out_csv)

    # IMPORTANT: use the *same* config loader you use elsewhere (e.g., decoding).
    # Swap this import if your project uses a different path.
    from turntaking.config.load import load_config  # type: ignore

    cfg = load_config(cli_args.config)

    # ---- Paths (adapt keys if needed) ----
    epoch_dir = Path(cfg.io.epoch_dir) if hasattr(cfg, "io") and hasattr(cfg.io, "epoch_dir") else Path(cfg.paths.epoch_dir)  # type: ignore
    out_dir = Path(cfg.io.out_dir) if hasattr(cfg, "io") and hasattr(cfg.io, "out_dir") else Path(cfg.paths.out_dir)  # type: ignore

    out_csv = cli_args.out_csv if cli_args.out_csv is not None else out_dir / "mixed_effect" / "table.csv"

    # ---- Selection ----
    # If you already store selection params under a single place, use that.
    # These attribute checks keep this robust across small refactors.
    if hasattr(cfg.analysis, "selection"):
        sel_cfg = cfg.analysis.selection
    else:
        sel_cfg = cfg.analysis.erp.selection  # type: ignore

    selection = SelectionParams(
        min_latency=float(sel_cfg.min_latency),
        max_latency=float(sel_cfg.max_latency),
        min_self_duration=float(sel_cfg.min_self_duration),
    )

    # ---- Mixed-effect windows ----
    me_cfg = cfg.analysis.mixed_effect

    params = MixedEffectTableParams(
        tw1_tmin=float(me_cfg.tw1_tmin),
        tw1_tmax=float(me_cfg.tw1_tmax),
        tw2_tmin=float(me_cfg.tw2_tmin),
        tw2_tmax=float(me_cfg.tw2_tmax),
        baseline_tmin=float(me_cfg.baseline_tmin),
        baseline_tmax=float(me_cfg.baseline_tmax),
        selection=selection,
    )

    # ---- ROIs ----
    roi_cfg = cfg.analysis.rois if hasattr(cfg.analysis, "rois") else cfg.rois  # type: ignore
    anterior_picks: Sequence[str] = list(roi_cfg.anterior)
    posterior_picks: Sequence[str] = list(roi_cfg.posterior)

    write_mixed_effect_table(
        epoch_dir=epoch_dir,
        out_csv=out_csv,
        params=params,
        anterior_picks=anterior_picks,
        posterior_picks=posterior_picks,
    )
