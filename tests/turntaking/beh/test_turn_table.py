
"""Tests for behavioral turn-table construction from TSV metadata."""

import pandas as pd
import pytest

from turntaking.beh.turn_table import TurnTablePaths, build_turn_table


def test_build_turn_table_merges_files_and_computes_windows(tmp_path):
    """Verifies turn-table assembly merges all TSVs and computes expected boolean window flags."""
    beh_dir = tmp_path / "beh"
    beh_dir.mkdir()
    pd.DataFrame(
        {
            "latency": [-2.0, -0.5, 2.5],
            "self_duration": [0.1, 0.2, 0.3],
            "other_duration": [0.4, 0.5, 0.6],
        }
    ).to_csv(beh_dir / "sub-001_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "latency": [0.2, 1.2],
            "self_duration": [0.11, 0.21],
            "other_duration": [0.41, 0.51],
        }
    ).to_csv(beh_dir / "sub-002_metadata.tsv", sep="\t", index=False)

    out_csv = tmp_path / "out" / "turn_table.csv"
    table = build_turn_table(TurnTablePaths(beh_dir=beh_dir, out_csv=out_csv))

    assert len(table) == 5
    assert out_csv.exists()
    assert table["in_plot_window"].sum() == 5
    assert table["in_analysis_window"].sum() == 2
    assert set(table["source_file"].unique()) == {"sub-001_metadata.tsv", "sub-002_metadata.tsv"}


def test_build_turn_table_rejects_missing_input_files(tmp_path):
    """Ensures missing metadata is reported early instead of yielding empty or misleading outputs."""
    with pytest.raises(FileNotFoundError, match="No metadata TSV files"):
        build_turn_table(TurnTablePaths(beh_dir=tmp_path / "empty", out_csv=tmp_path / "x.csv"))


def test_build_turn_table_rejects_missing_required_columns(tmp_path):
    """Checks strict column validation for behavioral TSVs so downstream analysis can trust the schema."""
    beh_dir = tmp_path / "beh"
    beh_dir.mkdir()
    pd.DataFrame({"latency": [0.1], "self_duration": [0.2]}).to_csv(
        beh_dir / "sub-001_metadata.tsv", sep="\t", index=False
    )

    with pytest.raises(KeyError, match="Missing required columns"):
        build_turn_table(TurnTablePaths(beh_dir=beh_dir, out_csv=tmp_path / "out.csv"))
