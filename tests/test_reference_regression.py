from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.reference_compare import (
    SimilarityThresholds,
    assert_similarity,
    load_supported_artifact,
)


def _load_reference_spec_from_env() -> dict:
    spec_path = os.getenv("TURNTAKING_REFERENCE_SPEC")
    if not spec_path:
        pytest.skip(
            "Set TURNTAKING_REFERENCE_SPEC to a JSON spec file to run reference regression checks."
        )

    path = Path(spec_path)
    if not path.exists():
        pytest.skip(f"Reference spec does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def test_reference_artifacts_are_similar_to_expected_results():
    """Compares generated artifacts against reference files using configurable numeric tolerances."""
    spec = _load_reference_spec_from_env()
    comparisons = spec.get("comparisons", [])
    if not comparisons:
        pytest.skip("Reference spec has no comparisons.")

    for item in comparisons:
        name = item["name"]
        actual_path = Path(item["actual"])
        reference_path = Path(item["reference"])
        thresholds = SimilarityThresholds(
            max_abs_error=float(item.get("max_abs_error", 1e-6)),
            mean_abs_error=float(item.get("mean_abs_error", 1e-8)),
            min_pearson_r=float(item.get("min_pearson_r", 0.999)),
        )

        if not actual_path.exists():
            pytest.fail(f"[{name}] Actual artifact is missing: {actual_path}")
        if not reference_path.exists():
            pytest.fail(f"[{name}] Reference artifact is missing: {reference_path}")

        actual = load_supported_artifact(actual_path)
        reference = load_supported_artifact(reference_path)
        report = assert_similarity(actual, reference, thresholds)

        # Keep useful details in pytest output for debugging tolerance choices.
        assert report.shape == actual.shape, f"[{name}] Shape mismatch after loading."
