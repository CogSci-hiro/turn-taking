"""Integration smoke test for joint ERP-alpha-behavior mixed models."""


from pathlib import Path

import pandas as pd


def test_integrative_pipeline_outputs(
    runtime_config_path: Path,
    out_dir: Path,
    tmp_path: Path,
    run_snakemake,
):
    run_snakemake("integrative_all", runtime_config_path, tmp_path)

    integration_dir = out_dir / "mixed_effect" / "integration"
    outputs = [
        integration_dir / "joint_model.csv",
        integration_dir / "interactions.csv",
        integration_dir / "random_slope.csv",
        integration_dir / "partial_correlations.csv",
    ]
    required_columns = {"estimate", "std.error", "statistic", "p.value"}

    for csv_path in outputs:
        assert csv_path.exists(), f"Missing integrative output: {csv_path}"
        table = pd.read_csv(csv_path)
        assert not table.empty, f"Integrative output is empty: {csv_path}"
        missing = sorted(required_columns - set(table.columns))
        assert not missing, f"{csv_path} missing columns: {missing}"
