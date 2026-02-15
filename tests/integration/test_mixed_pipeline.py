"""Scientific integration test.

These tests verify that outputs produced by Snakemake match our frozen baselines
within tight numerical tolerances. They guard against unintended scientific
changes during refactoring.

Do not relax tolerances without an experimental justification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_mixed_pipeline_regression(
    runtime_config_path: Path,
    out_dir: Path,
    baseline_path,
    tmp_path: Path,
    run_snakemake,
    compare_helpers: dict,
):
    """Compares mixed-effects design table against baseline and validates invariants."""
    run_snakemake("mixed_all", runtime_config_path, tmp_path)

    assert_csv_equal = compare_helpers["assert_csv_equal"]
    atol = compare_helpers["atol"]
    rtol = compare_helpers["rtol"]

    out_table = out_dir / "mixed_effect" / "table.csv"
    exp_table = baseline_path("mixed_effect/table.csv")

    assert out_table.exists(), f"Missing mixed-effects output table: {out_table}"
    assert_csv_equal(out_table, exp_table, "mixed_effect.table")

    out_df = pd.read_csv(out_table)
    exp_df = pd.read_csv(exp_table)

    required_columns = {
        "tw1_erp_anterior",
        "tw1_erp_posterior",
        "tw2_erp_anterior",
        "tw2_erp_posterior",
        "timestamp",
        "self_duration",
        "other_duration",
        "latency",
        "subject",
        "run",
    }
    missing = sorted(required_columns - set(out_df.columns))
    assert not missing, f"Missing required mixed-effects columns: {missing}"

    assert out_df.shape == exp_df.shape, (
        f"mixed_effect.table shape changed: actual={out_df.shape} expected={exp_df.shape}."
    )

    assert not out_df.isna().any().any(), "mixed_effect.table contains NaNs."

    numeric_cols = out_df.select_dtypes(include=[np.number]).columns.tolist()
    assert numeric_cols, "mixed_effect.table has no numeric columns to compare."

    for col in numeric_cols:
        av = out_df[col].to_numpy(dtype=float)
        ev = exp_df[col].to_numpy(dtype=float)
        max_diff = float(np.abs(av - ev).max()) if av.size else 0.0
        assert np.allclose(av, ev, atol=atol, rtol=rtol), (
            "Mixed-effects numeric column diverged beyond tolerance "
            f"for '{col}' (max diff={max_diff:.3e}, atol={atol}, rtol={rtol})."
        )

    assert set(out_df["subject"].astype(str)) == {"sub-006"}, (
        "mixed_effect.table contains unexpected subject IDs."
    )
    assert set(out_df["run"].astype(str)) == {"1"}, "mixed_effect.table contains unexpected run IDs."
