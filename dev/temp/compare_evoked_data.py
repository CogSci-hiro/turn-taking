#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_subjects_from_n_trials(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "subject" not in df.columns:
        raise ValueError(f"{path} has no 'subject' column. Columns={list(df.columns)}")
    subjects = df["subject"].astype(str).tolist()
    if len(subjects) != len(set(subjects)):
        dupes = df["subject"][df["subject"].duplicated()].unique().tolist()
        raise ValueError(f"Duplicate subjects in {path}: {dupes}")
    return subjects


def _as_old_style_difference(new_arr: np.ndarray) -> np.ndarray:
    """
    Convert new evoked-data.npy into the old difference-only format.

    Old format:
        (n_subjects, n_times, n_channels)

    New format (expected):
        (n_subjects, 3, n_channels, n_times)
        order: [cond_1, cond_2, diff]

    Returns
    -------
    np.ndarray
        (n_subjects, n_times, n_channels)
    """
    if new_arr.ndim != 4 or new_arr.shape[1] != 3:
        raise ValueError(f"Unexpected new array shape: {new_arr.shape} (expected (N,3,C,T))")

    diff = new_arr[:, 2, :, :]              # (N, C, T)
    diff_t = np.transpose(diff, (0, 2, 1))  # (N, T, C)
    return diff_t


def _basic_stats(a: np.ndarray, b: np.ndarray) -> None:
    diff = a - b
    abs_diff = np.abs(diff)

    print("Global diff stats")
    print(f"  max abs diff   : {abs_diff.max():.6e}")
    print(f"  mean abs diff  : {abs_diff.mean():.6e}")
    print(f"  median abs diff: {np.median(abs_diff):.6e}")

    # Useful “where is it worst?” diagnostics
    per_subj_max = abs_diff.reshape(abs_diff.shape[0], -1).max(axis=1)
    worst_idx = int(np.argmax(per_subj_max))
    print(f"\nWorst subject index: {worst_idx} (max abs diff={per_subj_max[worst_idx]:.6e})")

    # Optional: report top 5 worst
    topk = np.argsort(per_subj_max)[-5:][::-1]
    print("\nTop 5 worst subject indices:")
    for i in topk:
        print(f"  idx={int(i):03d} max_abs_diff={per_subj_max[int(i)]:.6e}")


def main(
    old_npy: Path,
    new_npy: Path,
    old_n_trials: Path | None,
    new_n_trials: Path | None,
) -> None:
    old_arr = np.load(old_npy)
    new_arr = np.load(new_npy)

    print("=== Shapes ===")
    print("old:", old_arr.shape, f"({old_npy})")
    print("new:", new_arr.shape, f"({new_npy})")

    if old_arr.ndim != 3:
        raise ValueError(f"Old array must be 3D (N,T,C). Got {old_arr.shape}")

    new_oldstyle = _as_old_style_difference(new_arr)
    print("new→oldstyle(diff):", new_oldstyle.shape)

    # Align by subjects if n_trials provided
    if old_n_trials is not None and new_n_trials is not None:
        old_subjects = _load_subjects_from_n_trials(old_n_trials)
        new_subjects = _load_subjects_from_n_trials(new_n_trials)

        print("\n=== Subject alignment (from n_trials.csv) ===")
        print("n old subjects:", len(old_subjects))
        print("n new subjects:", len(new_subjects))

        old_set = set(old_subjects)
        new_set = set(new_subjects)
        shared = sorted(old_set.intersection(new_set))
        only_old = sorted(old_set - new_set)
        only_new = sorted(new_set - old_set)

        print("n shared:", len(shared))
        if only_old:
            print("only in old:", only_old)
        if only_new:
            print("only in new:", only_new)

        if len(shared) == 0:
            raise ValueError("No shared subjects to compare.")

        # Build index maps
        old_idx = {s: i for i, s in enumerate(old_subjects)}
        new_idx = {s: i for i, s in enumerate(new_subjects)}

        old_aligned = np.stack([old_arr[old_idx[s]] for s in shared], axis=0)
        new_aligned = np.stack([new_oldstyle[new_idx[s]] for s in shared], axis=0)

        print("\nAligned shapes")
        print("old_aligned:", old_aligned.shape)
        print("new_aligned:", new_aligned.shape)

        a = old_aligned
        b = new_aligned
        shared_subjects = shared
    else:
        print("\n(no n_trials.csv provided; comparing by raw index order)")
        n = min(old_arr.shape[0], new_oldstyle.shape[0])
        if old_arr.shape[0] != new_oldstyle.shape[0]:
            print(f"⚠️ Different N: old={old_arr.shape[0]} new={new_oldstyle.shape[0]} -> comparing first n={n}")
        a = old_arr[:n]
        b = new_oldstyle[:n]
        shared_subjects = None

    # Check remaining dims match
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch after conversion/alignment: old={a.shape}, new={b.shape}")

    # Exact equality?
    if np.array_equal(a, b):
        print("\n✅ Exact match (bitwise) for compared data")
        return

    # Numeric closeness?
    atol = 1e-12
    rtol = 1e-7
    close = np.allclose(a, b, atol=atol, rtol=rtol)

    print("\n=== Comparison ===")
    print(f"allclose(atol={atol}, rtol={rtol}) -> {close}")

    _basic_stats(a, b)

    # If aligned, report the actual subject IDs for the worst cases
    if shared_subjects is not None:
        abs_diff = np.abs(a - b)
        per_subj_max = abs_diff.reshape(abs_diff.shape[0], -1).max(axis=1)
        topk = np.argsort(per_subj_max)[-5:][::-1]
        print("\nTop 5 worst subjects:")
        for i in topk:
            s = shared_subjects[int(i)]
            print(f"  {s}: max_abs_diff={per_subj_max[int(i)]:.6e}")


if __name__ == "__main__":
    if len(sys.argv) not in {3, 5}:
        print(
            "Usage:\n"
            "  python compare_evoked_data_old_vs_new.py OLD_evoked-data.npy NEW_evoked-data.npy\n"
            "  python compare_evoked_data_old_vs_new.py OLD_evoked-data.npy NEW_evoked-data.npy OLD_n_trials.csv NEW_n_trials.csv\n"
        )
        sys.exit(1)

    old_npy = Path(sys.argv[1])
    new_npy = Path(sys.argv[2])

    old_n_trials = Path(sys.argv[3]) if len(sys.argv) == 5 else None
    new_n_trials = Path(sys.argv[4]) if len(sys.argv) == 5 else None

    main(old_npy, new_npy, old_n_trials, new_n_trials)
