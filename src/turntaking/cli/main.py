import argparse
from pathlib import Path
from typing import Dict, Sequence

from turntaking.config import load_project_config
from turntaking.cli.types import CliCommand

from turntaking.cli.commands.analysis import (
    erp_generate as cmd_erp_generate,
    tfr_generate as cmd_tfr_generate,
    lmm_table as cmd_lmm_table,
    decode_prepare as cmd_decode_prepare,
    decode_run as cmd_decode_run,
)

_COMMANDS: Dict[str, CliCommand] = {
    "erp-generate": cmd_erp_generate,
    "tfr-generate": cmd_tfr_generate,
    "lmm-table": cmd_lmm_table,
    "decode-prepare": cmd_decode_prepare,
    "decode-run": cmd_decode_run,
}

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turntaking", description="Turn-taking analysis CLI")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name, module in _COMMANDS.items():
        module.add_subparser(subparsers)
    return parser

def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = load_project_config(Path(args.config))
    module = _COMMANDS[str(args.command)]
    module.run(args, cfg)


if __name__ == "__main__":
    main()
