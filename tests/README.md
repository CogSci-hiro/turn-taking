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

The test resolves paths using:

1. Snakemake output root (`io.out_dir` from `workflow/config.yaml`, or from `TURNTAKING_SNAKEMAKE_CONFIG`)
2. Global reference root (`TURNTAKING_REFERENCE_ROOT` env var, or `reference_root` in spec)
3. Per-item values in `comparisons`

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

If your Snakemake `io.out_dir` is `/Users/hiro/.../workflow/results/current` and your
`reference_root` is `/Users/hiro/.../workflow/results/reference`, then:

- actual: `decoding/erp/duration/scores.npy`
- inferred reference: `decoding/erp/duration/scores.npy` under `reference_root`

## Full reference data checklist

Important rule:

- The test requires reference files for every item in `comparisons`.
- If a comparison is listed and the inferred/overridden reference file is missing, the test fails.

Recommended full list (default practical setup):

- `decoding/erp/duration/scores.npy`
- `decoding/erp/duration/times.npy`
- `decoding/erp/latency/scores.npy`
- `decoding/erp/latency/times.npy`
- `erp/duration/offsets.csv`
- `erp/latency/offsets.csv`
- `beh/turn_table.csv`

So if your `reference_root` is:

- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference`

then expected files are:

- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/decoding/erp/duration/scores.npy`
- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/decoding/erp/duration/times.npy`
- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/decoding/erp/latency/scores.npy`
- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/decoding/erp/latency/times.npy`
- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/erp/duration/offsets.csv`
- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/erp/latency/offsets.csv`
- `/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference/beh/turn_table.csv`

Notes:

- Supported formats in this test are currently `.npy` and `.csv` only.
- Files like `.hdf5` are not compared unless you extend `load_supported_artifact`.

## Spec file format

Create a JSON file (for example: `dev/reference_spec.json`) like this:

```json
{
  "reference_root": "/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference",
  "global_tolerances": {
    "max_abs_error": 0.01,
    "mean_abs_error": 0.001,
    "min_pearson_r": 0.99
  },
  "comparisons": [
    {
      "name": "duration_scores",
      "actual": "decoding/erp/duration/scores.npy"
    },
    {
      "name": "latency_offsets",
      "actual": "erp/latency/offsets.csv",
      "tolerances": {
        "max_abs_error": 0.02,
        "mean_abs_error": 0.002,
        "min_pearson_r": 0.98
      }
    },
    {
      "name": "duration_scores_from_custom_reference_file",
      "actual": "decoding/erp/duration/scores.npy",
      "reference": "custom/duration_scores_v2.npy"
    }
  ]
}
```

Field meanings:

- `name`: label shown in failures
- `reference_root` (global): base directory for all reference files
- `global_tolerances` (global): defaults for all comparisons
- `actual`: path to current output file
  - if absolute: used as-is
  - if relative: resolved as `<snakemake_out_dir>/<actual>`
- `reference` (optional per item): reference path override
  - if omitted: inferred as `<reference_root>/<actual relative to snakemake out_dir>`
  - if relative: resolved as `<reference_root>/<reference>`
- `tolerances` (optional per item): per-file override values

Tolerance precedence:

1. `comparisons[i].tolerances.*`
2. legacy top-level per-item fields (`max_abs_error`, `mean_abs_error`, `min_pearson_r`)
3. `global_tolerances.*`
4. hard defaults in test code (`1e-6`, `1e-8`, `0.999`)

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

Custom Snakemake config path:

```bash
TURNTAKING_REFERENCE_SPEC=/Users/hiro/PycharmProjects/turn-taking-working/dev/reference_spec.json \
TURNTAKING_SNAKEMAKE_CONFIG=/Users/hiro/PycharmProjects/turn-taking-working/workflow/config.yaml \
pytest -q tests/turntaking/analysis/test_reference_regression.py
```

Override reference root without editing spec:

```bash
TURNTAKING_REFERENCE_SPEC=/Users/hiro/PycharmProjects/turn-taking-working/dev/reference_spec.json \
TURNTAKING_REFERENCE_ROOT=/Users/hiro/PycharmProjects/turn-taking-working/workflow/results/reference \
pytest -q tests/turntaking/analysis/test_reference_regression.py
```

## Troubleshooting

- Failure: `Actual artifact is missing`
  - The `actual` path in JSON does not exist.
- Failure: `Reference artifact is missing`
  - The inferred or overridden reference path does not exist.
- Failure: `TURNTAKING_REFERENCE_SPEC is required`
  - Environment variable was not set.
- Failure: `Reference spec is not valid JSON`
  - JSON syntax is invalid.
- Failure: `Reference spec must include at least one comparison`
  - `comparisons` is empty or missing.
- Failure: `Cannot infer reference path because actual path is outside Snakemake out_dir`
  - Use a relative `actual` or provide explicit `reference` for that item.
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
