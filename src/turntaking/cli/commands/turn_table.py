import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


# ##################################################################################################
# Build turn table (behavior)
# ##################################################################################################


@dataclass(frozen=True)
class TurnTableCliConfig:
    beh_dir: Path
    out_dir: Path
    out_name: str


def _run_impl(cfg: TurnTableCliConfig) -> Path:
    from turntaking.beh.turn_table import TurnTablePaths, build_turn_table

    out_csv = cfg.out_dir / "beh" / cfg.out_name
    paths = TurnTablePaths(beh_dir=cfg.beh_dir, out_csv=out_csv)
    build_turn_table(paths)
    return out_csv


def run(args: argparse.Namespace, cfg) -> None:
    """
    Build turn_table.csv from metadata TSV files.

    This command reads workflow/config.yaml directly because 'paths:' is not part of the typed TurntakingConfig.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        d = yaml.safe_load(f)

    paths_d = d.get("paths", None)
    if not isinstance(paths_d, dict):
        raise KeyError("Missing or invalid 'paths' section in config YAML. Expected:\npaths:\n  beh_dir: ...\n  out_dir: ...")

    beh_dir = Path(paths_d["beh_dir"])
    out_dir = Path(paths_d["out_dir"])

    cli_cfg = TurnTableCliConfig(
        beh_dir=beh_dir,
        out_dir=out_dir,
        out_name=str(getattr(args, "out_name", "turn_table.csv")),
    )

    out_csv = _run_impl(cli_cfg)
    print(f"Wrote: {out_csv}")


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "beh-turn-table",
        help="Build a canonical turn-level CSV for behavior visualization.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out-name",
        default="turn_table.csv",
        help="Output filename under OUT_DIR/beh/ (default: turn_table.csv).",
    )

