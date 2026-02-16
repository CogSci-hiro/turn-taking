"""
turntaking: analysis + visualization pipeline for conversational turn-taking EEG.

This repository provides a small, opinionated pipeline for running group-level
EEG analyses on *already-epoched* MNE ``Epochs`` files. The intended usage is
either via:

- the Python API (mostly under ``turntaking.analysis``), or
- the CLI/Snakemake workflow (``python -m turntaking.cli.main ...`` and
  ``workflow/Snakefile``).

High-level outputs (written under ``io.out_dir``) include:

- ERP and induced-TFR evoked artifacts (FIF + NumPy + CSV + HDF5 metadata),
- temporal-generalization decoding scores (NumPy),
- cluster-based permutation statistics (HDF5 + CSV summary),
- mixed-effect tables (CSV) and downstream model fits (R scripts in workflow),
- publication-ready figures (TIFF/EPS/PNG) via ``turntaking.viz``.

The project is intentionally structured with clear boundaries:

- ``turntaking.cli`` parses arguments and dispatches to library functions.
- ``turntaking.analysis`` holds computation and I/O contracts (domain modules).
- ``turntaking.viz`` renders figures from the saved artifact contracts.
"""

from __future__ import annotations
