"""Integration fixtures and scientific comparison helpers.

These fixtures standardize how Snakemake is executed in tests and how artifacts
are compared against frozen baselines. All helpers default to tight numeric
thresholds to catch subtle scientific regressions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import h5py
import mne
import numpy as np
import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
INTEGRATION_SNAKEFILE = REPO_ROOT / "tests" / "integration" / "Snakefile"

ATOL = 1e-10
RTOL = 1e-8


@pytest.fixture
def fixture_epoch_path() -> Path:
    p = FIXTURES / "epochs" / "sub-006_task-diapix_run-1_epo.fif"
    assert p.exists(), f"Missing fixture epoch: {p}"
    return p


@pytest.fixture
def test_config_template_path() -> Path:
    p = FIXTURES / "test_config.yaml"
    assert p.exists(), f"Missing test config template: {p}"
    return p


@pytest.fixture
def runtime_config_path(tmp_path: Path, fixture_epoch_path: Path, test_config_template_path: Path) -> Path:
    template = test_config_template_path.read_text(encoding="utf-8")
    epoch_dir = fixture_epoch_path.parent.resolve()
    out_dir = (tmp_path / "out").resolve()
    rendered = template.replace("__EPOCH_DIR__", str(epoch_dir)).replace("__OUT_DIR__", str(out_dir))
    cfg = tmp_path / "test_config_runtime.yaml"
    cfg.write_text(rendered, encoding="utf-8")
    return cfg


@pytest.fixture
def runtime_config(runtime_config_path: Path) -> dict:
    return yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))


@pytest.fixture
def out_dir(runtime_config: dict) -> Path:
    p = Path(runtime_config["io"]["out_dir"])
    assert p.is_absolute(), f"Configured out_dir must be absolute: {p}"
    return p


@pytest.fixture
def expected_root() -> Path:
    p = FIXTURES / "expected"
    assert p.exists(), f"Missing expected fixture root: {p}"
    return p


@pytest.fixture
def baseline_path(expected_root: Path) -> Callable[[str], Path]:
    def _baseline_path(relative: str) -> Path:
        p = expected_root / relative
        assert p.exists(), f"Missing baseline artifact: {p}"
        return p

    return _baseline_path


def _run_snakemake(target: str, config_path: Path, workdir: Path) -> None:
    """Run one Snakemake target in a hermetic temp environment.

    Scientific rule: integration tests must execute pipeline code through
    Snakemake, not through direct Python calls, so the orchestration layer is
    regression-tested as well.
    """
    snakemake_exe = shutil.which("snakemake")
    assert snakemake_exe, "snakemake not found in PATH."

    cmd = [
        snakemake_exe,
        "-s",
        str(INTEGRATION_SNAKEFILE),
        target,
        "--cores",
        "1",
        "--config",
        f"test_config={config_path}",
        f"python_exe={sys.executable}",
        f"repo_root={REPO_ROOT}",
        "--forceall",
    ]

    cache_root = workdir / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["HOME"] = str(workdir)
    env["XDG_CACHE_HOME"] = str(cache_root)
    env["SNAKEMAKE_HOME"] = str(workdir / ".snakemake_home")
    env["MPLCONFIGDIR"] = str(workdir / ".mplconfig")
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("MNE_DONTWRITE_HOME", "true")

    src_dir = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_dir if not env.get("PYTHONPATH") else f"{src_dir}:{env['PYTHONPATH']}"

    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise AssertionError(
            f"Snakemake failed for target={target}\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}"
        )


@pytest.fixture
def run_snakemake() -> Callable[[str, Path, Path], None]:
    return _run_snakemake


def load_hdf5_dataset(path: Path, key: str):
    """Load one HDF5 dataset value for scientific equality checks."""
    with h5py.File(path, "r") as h5:
        assert key in h5, f"Missing dataset '{key}' in {path}"
        return h5[key][()]


def hdf5_dataset_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    with h5py.File(path, "r") as h5:
        h5.visititems(lambda name, obj: keys.add(name) if isinstance(obj, h5py.Dataset) else None)
    return keys


def read_evoked(path: Path) -> mne.Evoked:
    return mne.read_evokeds(path, condition=0, verbose="ERROR")


def assert_no_nans_array(arr: np.ndarray, label: str) -> None:
    assert not np.isnan(arr).any(), f"{label} contains NaNs (n={np.isnan(arr).sum()})."


def assert_allclose_arrays(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    assert actual.shape == expected.shape, (
        f"{label} shape mismatch: actual {actual.shape} vs expected {expected.shape}."
    )
    diff = np.abs(actual - expected)
    finite_diff = diff[np.isfinite(diff)]
    max_diff = float(finite_diff.max()) if finite_diff.size else 0.0
    assert np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True), (
        f"{label} deviated beyond tolerance (atol={ATOL}, rtol={RTOL}, max_diff={max_diff:.3e})."
    )


def assert_csv_equal(actual: Path, expected: Path, label: str, *, allow_nans: bool = False) -> None:
    a = pd.read_csv(actual)
    e = pd.read_csv(expected)
    assert list(a.columns) == list(e.columns), (
        f"{label} columns differ: actual={list(a.columns)} expected={list(e.columns)}."
    )
    assert a.shape == e.shape, f"{label} shape differs: actual={a.shape} expected={e.shape}."

    for col in a.columns:
        ac = a[col]
        ec = e[col]
        if pd.api.types.is_numeric_dtype(ac) and pd.api.types.is_numeric_dtype(ec):
            av = ac.to_numpy(dtype=float)
            ev = ec.to_numpy(dtype=float)
            diff = np.abs(av - ev)
            finite_diff = diff[np.isfinite(diff)]
            max_diff = float(finite_diff.max()) if finite_diff.size else 0.0
            assert np.allclose(av, ev, rtol=RTOL, atol=ATOL, equal_nan=True), (
                f"{label}.{col} deviated beyond tolerance "
                f"(atol={ATOL}, rtol={RTOL}, max_diff={max_diff:.3e})."
            )
            if not allow_nans:
                assert not np.isnan(av).any(), f"{label}.{col} contains NaNs."
        else:
            assert ac.astype(str).tolist() == ec.astype(str).tolist(), f"{label}.{col} values differ."


@pytest.fixture
def hdf5_helpers():
    return {
        "load_hdf5_dataset": load_hdf5_dataset,
        "hdf5_dataset_keys": hdf5_dataset_keys,
    }


@pytest.fixture
def load_hdf5_dataset_helper() -> Callable[[Path, str], object]:
    return load_hdf5_dataset


@pytest.fixture
def compare_helpers():
    return {
        "assert_no_nans_array": assert_no_nans_array,
        "assert_allclose_arrays": assert_allclose_arrays,
        "assert_csv_equal": assert_csv_equal,
        "read_evoked": read_evoked,
        "atol": ATOL,
        "rtol": RTOL,
    }
