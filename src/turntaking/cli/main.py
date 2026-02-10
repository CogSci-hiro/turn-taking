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

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

from turntaking.cli.types import CliCommand
from turntaking.config.loader import load_config

# Analysis command handlers (thin; no scientific logic here either)
from turntaking.cli.commands.analysis import erp_generate as cmd_erp_generate


_COMMANDS: Dict[str, CliCommand] = {
    "erp-generate": cmd_erp_generate,
}


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
