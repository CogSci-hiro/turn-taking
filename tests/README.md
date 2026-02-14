# Test Suite Overview

This suite is designed to satisfy three goals:

1. Catch functional regressions in core analysis/data I/O logic.
2. Compare run outputs against reference artifacts with explicit numeric tolerances.
3. Keep every test self-documenting by including a short "what/why" docstring.

## Layout

Tests mirror the source package shape under `src/turntaking`:

- `tests/turntaking/analysis/...`
- `tests/turntaking/beh/...`
- `tests/turntaking/config/...`
- `tests/turntaking/stats/...`

## Basic usage

Run all non-reference tests:

```bash
pytest -q
```

Run only reference similarity test:

```bash
pytest -q tests/turntaking/analysis/test_reference_regression.py
```

This reference test is strict: it fails if required inputs are missing.

## Reference similarity: what to provide

The reference regression test compares two files per item:

- `actual`: output produced by your current code run
- `reference`: trusted baseline output (previous run, published result, or manually approved output)

You provide these file paths in a JSON spec file and point the test to it with
`TURNTAKING_REFERENCE_SPEC`.

If `TURNTAKING_REFERENCE_SPEC` is unset, invalid, or points to a missing file, the test fails.

### Supported artifact types

Currently supported:

- `.npy` (NumPy arrays)
- `.csv` (numeric tables)

Rules:

- `actual` and `reference` must have the same shape.
- For `.csv`, column order must match exactly.
- For `.csv`, values should be numeric.
- The spec must contain at least one comparison item.

## Where to put your files

You can store output files anywhere. Absolute paths are the safest.

Recommended structure (example only):

```text
/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/current/
/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/
/Users/hiro/PycharmProjects/turn-taking-working/dev/reference_spec.json
```

Example:

- Current run output:  
  `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/current/duration_scores.npy`
- Reference output:  
  `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/duration_scores.npy`

## Spec file format

Create a JSON file (for example: `dev/reference_spec.json`) like this:

```json
{
  "comparisons": [
    {
      "name": "duration_scores",
      "actual": "/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/current/duration_scores.npy",
      "reference": "/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/duration_scores.npy",
      "max_abs_error": 0.01,
      "mean_abs_error": 0.001,
      "min_pearson_r": 0.99
    },
    {
      "name": "latency_offsets",
      "actual": "/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/current/latency_offsets.csv",
      "reference": "/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/latency_offsets.csv",
      "max_abs_error": 0.02,
      "mean_abs_error": 0.002,
      "min_pearson_r": 0.98
    }
  ]
}
```

Field meanings:

- `name`: label shown in failures
- `actual`: current output file path
- `reference`: baseline output file path
- `max_abs_error`: max allowed absolute pointwise error
- `mean_abs_error`: max allowed mean absolute error
- `min_pearson_r`: minimum allowed Pearson correlation

Required keys in each comparison object:

- `name`
- `actual`
- `reference`

## How to run with your spec

```bash
TURNTAKING_REFERENCE_SPEC=/Users/hiro/PycharmProjects/turn-taking-working/dev/reference_spec.json pytest -q
```

Or only the reference test:

```bash
TURNTAKING_REFERENCE_SPEC=/Users/hiro/PycharmProjects/turn-taking-working/dev/reference_spec.json \
pytest -q tests/turntaking/analysis/test_reference_regression.py
```

## Troubleshooting

- Failure: `Actual artifact is missing`
  - The `actual` path in JSON does not exist.
- Failure: `Reference artifact is missing`
  - The `reference` path in JSON does not exist.
- Failure: `TURNTAKING_REFERENCE_SPEC is required`
  - Environment variable was not set.
- Failure: `Reference spec is not valid JSON`
  - JSON syntax is invalid.
- Failure: `Reference spec must include at least one comparison`
  - `comparisons` is empty or missing.
- Failure: `Shape mismatch`
  - Files contain different array/table shapes.
- Failure on correlation/error thresholds
  - Relax tolerances or investigate regression.

## Practical workflow

1. Run pipeline to generate current outputs.
2. Save/collect trusted reference outputs.
3. Write `dev/reference_spec.json`.
4. Run pytest with `TURNTAKING_REFERENCE_SPEC`.
5. If it fails, inspect which `name` failed and why.
