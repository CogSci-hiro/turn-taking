
from typing import Protocol, Any


class CliCommand(Protocol):
    """Protocol for CLI command modules.

    Each command module must define:
    - add_subparser(subparsers): register args
    - run(args, cfg): execute command

    Notes
    -----
    Command modules must be import-only (no runnable entrypoints).
    """

    def add_subparser(self, subparsers: Any) -> None: ...
    def run(self, args: Any, cfg: Any) -> None: ...
