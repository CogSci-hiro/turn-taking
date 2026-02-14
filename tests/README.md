# Test Suite Overview

This suite is designed to satisfy three goals:

1. Catch functional regressions in core analysis/data I/O logic.
2. Compare run outputs against reference artifacts with explicit numeric tolerances.
3. Keep every test self-documenting by including a short "what/why" docstring.

## Running Tests

```bash
pytest -q
```

## Reference Result Similarity Checks

`tests/test_reference_regression.py` is data-driven and activated by an environment variable:

```bash
TURNTAKING_REFERENCE_SPEC=/absolute/path/to/reference_spec.json pytest -q
```

Example spec:

```json
{
  "comparisons": [
    {
      "name": "duration_scores",
      "actual": "/path/to/current/duration_scores.npy",
      "reference": "/path/to/reference/duration_scores.npy",
      "max_abs_error": 0.01,
      "mean_abs_error": 0.001,
      "min_pearson_r": 0.99
    },
    {
      "name": "latency_offsets",
      "actual": "/path/to/current/latency_offsets.csv",
      "reference": "/path/to/reference/latency_offsets.csv",
      "max_abs_error": 0.02,
      "mean_abs_error": 0.002,
      "min_pearson_r": 0.98
    }
  ]
}
```
