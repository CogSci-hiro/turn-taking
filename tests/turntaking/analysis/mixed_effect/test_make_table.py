
"""Tests for mixed-effect table assembly helpers and file-level guards."""

from pathlib import Path

import pandas as pd
import pytest

from turntaking.analysis.mixed_effect.make_table import (
    _drop_unwanted_columns,
    _rename_and_select_columns,
    make_mixed_effect_table,
    write_mixed_effect_table,
)
from turntaking.analysis.mixed_effect.schema import MixedEffectTableParams
from turntaking.analysis.selection import SelectionParams


def _params() -> MixedEffectTableParams:
    return MixedEffectTableParams(
        tw1_tmin=0.0,
        tw1_tmax=0.1,
        tw2_tmin=0.2,
        tw2_tmax=0.3,
        baseline_tmin=-0.1,
        baseline_tmax=0.0,
        selection=SelectionParams(min_latency=0.1, max_latency=1.0, min_self_duration=0.1),
    )


def test_drop_unwanted_columns_filters_known_noise_columns():
    """Ensures mixed-effect output cleanup drops index artifacts and known irrelevant columns."""
    df = pd.DataFrame(
        {
            "index": [0],
            "Unnamed: 0": [0],
            "self_n_words": [3],
            "speech_rate": [1.2],
            "latency": [0.4],
            "latency": [0.4],  # duplicate column intent; pandas keeps latest
            "keep_me": [1],
        }
    )
    out = _drop_unwanted_columns(df)
    assert "index" not in out.columns
    assert "Unnamed: 0" not in out.columns
    assert "self_n_words" not in out.columns
    assert "speech_rate" not in out.columns
    assert "keep_me" in out.columns


def test_rename_and_select_columns_validates_required_schema(tmp_path):
    """Checks rename-to-final-schema step fails loudly when required columns are missing."""
    base = {
        "tw1_mean_anterior": [1.0],
        "tw1_mean_posterior": [1.0],
        "tw2_mean_anterior": [1.0],
        "tw2_mean_posterior": [1.0],
        "tw1_alpha_anterior": [1.0],
        "tw1_alpha_posterior": [1.0],
        "tw2_alpha_anterior": [1.0],
        "tw2_alpha_posterior": [1.0],
        "tw1_beta_anterior": [1.0],
        "tw1_beta_posterior": [1.0],
        "tw2_beta_anterior": [1.0],
        "tw2_beta_posterior": [1.0],
        "baseline_mean_anterior": [1.0],
        "baseline_mean_posterior": [1.0],
        "timestamp": [0.0],
        "self_duration": [0.3],
        "other_duration": [0.2],
        "latency": [0.1],
        "subject": ["sub-001"],
        "run": [1],
    }
    out = _rename_and_select_columns(pd.DataFrame(base), source_path=tmp_path / "x.fif")
    assert "tw1_erp_anterior" in out.columns
    assert "baseline_erp_posterior" in out.columns

    bad = pd.DataFrame({k: v for k, v in base.items() if k != "tw1_beta_anterior"})
    with pytest.raises(RuntimeError, match="missing required columns"):
        _rename_and_select_columns(bad, source_path=tmp_path / "x.fif")


def test_make_mixed_effect_table_rejects_empty_or_unmatched_input(tmp_path):
    """Prevents silent success when there are no candidate files or no pattern matches."""
    with pytest.raises(FileNotFoundError):
        make_mixed_effect_table(
            tmp_path / "missing",
            params=_params(),
            anterior_picks=["Fz"],
            posterior_picks=["Pz"],
        )

    epoch_dir = tmp_path / "epochs"
    epoch_dir.mkdir()
    (epoch_dir / "not_matching_name.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="No valid epoch files matched pattern"):
        make_mixed_effect_table(
            epoch_dir,
            params=_params(),
            anterior_picks=["Fz"],
            posterior_picks=["Pz"],
        )


def test_write_mixed_effect_table_writes_csv(monkeypatch, tmp_path):
    """Ensures wrapper writes the DataFrame produced by make_mixed_effect_table."""
    expected = pd.DataFrame({"subject": ["sub-001"], "run": [1]})
    monkeypatch.setattr("turntaking.analysis.mixed_effect.make_table.make_mixed_effect_table", lambda *a, **k: expected)

    out_csv = tmp_path / "mixed" / "table.csv"
    write_mixed_effect_table(
        epoch_dir=tmp_path,
        out_csv=out_csv,
        params=_params(),
        anterior_picks=["Fz"],
        posterior_picks=["Pz"],
    )
    pd.testing.assert_frame_equal(pd.read_csv(out_csv), expected)
