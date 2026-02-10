from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from turntaking.config.loader import load_config
from turntaking.preprocessing.pipeline import run_preprocessing_to_epochs

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def preprocess(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    subject: Optional[str] = typer.Option(None, "--subject", help="e.g., sub-005"),
    run: Optional[str] = typer.Option(None, "--run", help="e.g., run-1"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """
    Run preprocessing up to Epochs and save outputs.

    Intended to be the stable, reproducible entrypoint.
    """
    cfg = load_config(config)
    console.print(f"[bold]Config:[/bold] {config}")
    console.print(f"[bold]Subject/run:[/bold] {subject} / {run}")

    run_preprocessing_to_epochs(
        cfg=cfg,
        subject=subject,
        run=run,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
