from __future__ import annotations

"""Data-driven regression test entrypoint for reference artifact checks."""

import json
import os
from pathlib import Path

import yaml

from .reference_compare import (
    SimilarityThresholds,
    assert_similarity,
    load_supported_artifact,
)


DEFAULT_MAX_ABS_ERROR = 1e-6
DEFAULT_MEAN_ABS_ERROR = 1e-8
DEFAULT_MIN_PEARSON_R = 0.999
DEFAULT_REFERENCE_SPEC_PATH = Path("/Users/hiro/PycharmProjects/turn-taking-working/dev/reference_spec.json")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_reference_spec_from_env() -> dict:
    spec_path = os.getenv("TURNTAKING_REFERENCE_SPEC")
    path = Path(spec_path) if spec_path else DEFAULT_REFERENCE_SPEC_PATH
    assert path.exists(), f"Reference spec does not exist: {path}"

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Reference spec is not valid JSON: {path}") from exc

    assert isinstance(raw, dict), "Reference spec root must be a JSON object."
    return raw


def _load_snakemake_out_dir() -> Path:
    config_env = os.getenv("TURNTAKING_SNAKEMAKE_CONFIG")
    if config_env:
        config_path = Path(config_env)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
    else:
        config_path = PROJECT_ROOT / "workflow/config.yaml"

    assert config_path.exists(), f"Snakemake config not found: {config_path}"

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"Expected mapping at Snakemake config root: {config_path}"

    io = raw.get("io")
    if isinstance(io, dict) and io.get("out_dir"):
        return Path(str(io["out_dir"]))

    paths = raw.get("paths")
    if isinstance(paths, dict) and paths.get("out_dir"):
        return Path(str(paths["out_dir"]))

    raise AssertionError(
        f"Could not resolve output directory from {config_path}. "
        "Expected `io.out_dir` or `paths.out_dir`."
    )


def _resolve_reference_root(spec: dict) -> Path:
    # Allow environment override for CI or local ad-hoc runs.
    env_root = os.getenv("TURNTAKING_REFERENCE_ROOT")
    ref_root_raw = env_root or spec.get("reference_root")
    assert ref_root_raw, (
        "Global reference root is required. Set TURNTAKING_REFERENCE_ROOT or "
        "define `reference_root` in the reference spec."
    )

    ref_root = Path(str(ref_root_raw))
    assert ref_root.exists(), f"Reference root does not exist: {ref_root}"
    assert ref_root.is_dir(), f"Reference root is not a directory: {ref_root}"
    return ref_root


def _resolve_actual_path(item: dict, out_dir: Path) -> Path:
    assert "actual" in item, "Comparison missing required key: actual"
    actual_raw = str(item["actual"]).strip()
    assert actual_raw, "Comparison key 'actual' cannot be empty."

    actual = Path(actual_raw)
    return actual if actual.is_absolute() else out_dir / actual


def _resolve_reference_path(item: dict, *, actual_path: Path, out_dir: Path, reference_root: Path) -> Path:
    # Optional explicit per-file override.
    if "reference" in item and str(item["reference"]).strip():
        ref = Path(str(item["reference"]).strip())
        return ref if ref.is_absolute() else reference_root / ref

    # Default: map current output path to reference root by preserving out_dir-relative structure.
    try:
        rel = actual_path.relative_to(out_dir)
    except ValueError as exc:
        raise AssertionError(
            f"[{item.get('name', '<unnamed>')}] Cannot infer reference path because "
            f"actual path is outside Snakemake out_dir.\n"
            f"actual={actual_path}\n"
            f"out_dir={out_dir}\n"
            "Provide an explicit `reference` for this item."
        ) from exc

    return reference_root / rel


def _thresholds_for_item(spec: dict, item: dict) -> SimilarityThresholds:
    global_tol = spec.get("global_tolerances", {})
    assert isinstance(global_tol, dict), "Spec field `global_tolerances` must be an object when provided."

    item_tol = item.get("tolerances", {})
    assert isinstance(item_tol, dict), (
        f"[{item.get('name', '<unnamed>')}] `tolerances` must be an object when provided."
    )

    max_abs_error = float(
        item_tol.get(
            "max_abs_error",
            item.get(
                "max_abs_error",
                global_tol.get("max_abs_error", DEFAULT_MAX_ABS_ERROR),
            ),
        )
    )
    mean_abs_error = float(
        item_tol.get(
            "mean_abs_error",
            item.get(
                "mean_abs_error",
                global_tol.get("mean_abs_error", DEFAULT_MEAN_ABS_ERROR),
            ),
        )
    )
    min_pearson_r = float(
        item_tol.get(
            "min_pearson_r",
            item.get(
                "min_pearson_r",
                global_tol.get("min_pearson_r", DEFAULT_MIN_PEARSON_R),
            ),
        )
    )

    return SimilarityThresholds(
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        min_pearson_r=min_pearson_r,
    )


def test_reference_artifacts_are_similar_to_expected_results():
    """Compares generated artifacts against references using global defaults and per-file overrides."""
    spec = _load_reference_spec_from_env()
    out_dir = _load_snakemake_out_dir()
    reference_root = _resolve_reference_root(spec)

    comparisons = spec.get("comparisons", [])
    assert isinstance(comparisons, list), "Reference spec field 'comparisons' must be a list."
    assert len(comparisons) > 0, "Reference spec must include at least one comparison."

    for item in comparisons:
        assert isinstance(item, dict), "Each comparison entry must be a JSON object."
        for required_key in ("name", "actual"):
            assert required_key in item, f"Comparison missing required key: {required_key}"
            assert str(item[required_key]).strip(), f"Comparison key '{required_key}' cannot be empty."

        name = item["name"]
        actual_path = _resolve_actual_path(item, out_dir)
        reference_path = _resolve_reference_path(
            item,
            actual_path=actual_path,
            out_dir=out_dir,
            reference_root=reference_root,
        )
        thresholds = _thresholds_for_item(spec, item)

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
