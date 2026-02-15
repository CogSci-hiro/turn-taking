
import numpy as np
import pandas as pd


def zscore_column(df: pd.DataFrame, col: str) -> pd.Series:
    vals = df[col].to_numpy(dtype=float)
    return pd.Series((vals - vals.mean()) / vals.std(ddof=0), index=df.index)


def test_dataframe_transformation_required_columns_no_nans_and_shape():
    n = 8
    df = pd.DataFrame(
        {
            "subject": ["sub-006"] * n,
            "run": [1] * n,
            "latency": np.linspace(-0.4, 0.4, n),
            "self_duration": np.linspace(0.1, 1.0, n),
            "other_duration": np.linspace(0.2, 1.2, n),
            "tw1_erp_anterior": np.linspace(-1.0, 1.0, n),
        }
    )
    df["tw1_erp_anterior_z"] = zscore_column(df, "tw1_erp_anterior")

    required = {"subject", "run", "latency", "self_duration", "other_duration", "tw1_erp_anterior_z"}
    assert required.issubset(df.columns)
    assert not df[list(required)].isna().any().any()
    assert df.shape[0] == n

    z = df["tw1_erp_anterior_z"].to_numpy()
    assert abs(float(z.mean())) < 1e-12
    assert abs(float(z.std(ddof=0)) - 1.0) < 1e-12
