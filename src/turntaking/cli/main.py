# =============================================================================
#                                   CLI
# =============================================================================
#
# Entry point for the `turntaking` command-line interface.
#
# Thin dispatcher:
# - parse global + subcommand arguments
# - load project config once
# - call a single library function per subcommand
#
# Scientific logic must live in `turntaking.analysis.*`, not here.
#
# =============================================================================


import argparse
from pathlib import Path
from typing import Sequence

from turntaking.config.loader import load_config

import turntaking.cli.commands.erp as cmd_erp
import turntaking.cli.commands.tfr as cmd_tfr
import turntaking.cli.commands.cluster as cmd_cluster
import turntaking.cli.commands.decoding as cmd_decoding
import turntaking.cli.commands.mixed_effect as cmd_mixed_effect

import turntaking.cli.commands.viz.erp_timecourse as cmd_viz_erp_timecourse
import turntaking.cli.commands.viz.behavior as cmd_viz_behavior
import turntaking.cli.commands.viz.erp_topo as cmd_viz_erp_topo

_COMMANDS = {"erp": cmd_erp,
             "tfr": cmd_tfr,
             "decoding": cmd_decoding,
             "cluster": cmd_cluster,
             "mixed-effect": cmd_mixed_effect,
             "viz-erp-timecourse": cmd_viz_erp_timecourse,
             "viz-behavior": cmd_viz_behavior,
             "viz-erp-topo": cmd_viz_erp_topo}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="turntaking",
        description="Turn-taking analysis pipeline (already-preprocessed data).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for name, module in _COMMANDS.items():
        if not hasattr(module, "add_subparser"):
            raise RuntimeError(f"CLI command module for '{name}' is missing add_subparser().")
        module.add_subparser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "config"):
        raise RuntimeError("Internal error: subcommand args missing --config.")

    cfg = load_config(Path(args.config))

    command_name = str(args.command)
    module = _COMMANDS.get(command_name)
    if module is None:
        raise RuntimeError(f"Unknown command: {command_name}")

    if not hasattr(module, "run"):
        raise RuntimeError(f"CLI command module for '{command_name}' is missing run().")

    module.run(args, cfg)


if __name__ == "__main__":
    main()
