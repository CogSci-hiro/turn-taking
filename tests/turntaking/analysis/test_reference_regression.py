from __future__ import annotations

"""Data-driven regression test entrypoint for reference artifact checks."""

import json
import os
from pathlib import Path

from .reference_compare import (
    SimilarityThresholds,
    assert_similarity,
    load_supported_artifact,
)


def _load_reference_spec_from_env() -> dict:
    spec_path = os.getenv("TURNTAKING_REFERENCE_SPEC")
    assert spec_path, (
        "TURNTAKING_REFERENCE_SPEC is required. "
        "Point it to a JSON file with reference comparisons."
    )

    path = Path(spec_path)
    assert path.exists(), f"Reference spec does not exist: {path}"

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Reference spec is not valid JSON: {path}") from exc

    assert isinstance(raw, dict), "Reference spec root must be a JSON object."
    return raw


def test_reference_artifacts_are_similar_to_expected_results():
    """Compares generated artifacts against reference files using configurable numeric tolerances."""
    spec = _load_reference_spec_from_env()
    comparisons = spec.get("comparisons", [])
    assert isinstance(comparisons, list), "Reference spec field 'comparisons' must be a list."
    assert len(comparisons) > 0, "Reference spec must include at least one comparison."

    for item in comparisons:
        assert isinstance(item, dict), "Each comparison entry must be a JSON object."
        for required_key in ("name", "actual", "reference"):
            assert required_key in item, f"Comparison missing required key: {required_key}"
            assert str(item[required_key]).strip(), f"Comparison key '{required_key}' cannot be empty."

        name = item["name"]
        actual_path = Path(item["actual"])
        reference_path = Path(item["reference"])
        thresholds = SimilarityThresholds(
            max_abs_error=float(item.get("max_abs_error", 1e-6)),
            mean_abs_error=float(item.get("mean_abs_error", 1e-8)),
            min_pearson_r=float(item.get("min_pearson_r", 0.999)),
        )

        assert thresholds.max_abs_error >= 0.0, f"[{name}] max_abs_error must be >= 0."
        assert thresholds.mean_abs_error >= 0.0, f"[{name}] mean_abs_error must be >= 0."
        assert -1.0 <= thresholds.min_pearson_r <= 1.0, (
            f"[{name}] min_pearson_r must be between -1 and 1."
        )

        assert actual_path.exists(), f"[{name}] Actual artifact is missing: {actual_path}"
        assert reference_path.exists(), f"[{name}] Reference artifact is missing: {reference_path}"

        actual = load_supported_artifact(actual_path)
        reference = load_supported_artifact(reference_path)
        report = assert_similarity(actual, reference, thresholds)

        # Keep useful details in pytest output for debugging tolerance choices.
        assert report.shape == actual.shape, f"[{name}] Shape mismatch after loading."
