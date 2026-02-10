
#!/usr/bin/env python3
"""
Quick sanity check for evoked-data.npy between two projects.

Usage
-----
python compare_evoked_data.py \
    /path/to/old/evoked-data.npy \
    /path/to/new/evoked-data.npy

Assumptions
-----------
Array shape:
    (n_subjects, 3, n_channels, n_times)
    order = [cond1, cond2, difference]
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np


def main(old_path: Path, new_path: Path) -> None:
    old = np.load(old_path)
    new = np.load(new_path)

    print("=== Evoked data comparison ===")
    print(f"Old: {old_path}")
    print(f"New: {new_path}")
    print()

    print("Shapes")
    print(f"  old: {old.shape}")
    print(f"  new: {new.shape}")

    if old.shape != new.shape:
        print("❌ Shape mismatch — aborting numeric comparison")
        return

    print("✓ Shapes match\n")

    # Exact equality (strict)
    if np.array_equal(old, new):
        print("✅ Arrays are exactly equal (bitwise)")
        return

    print("⚠️ Arrays differ (not bitwise equal)")
    print()

    diff = old - new
    abs_diff = np.abs(diff)

    print("Global difference stats")
    print(f"  max abs diff : {abs_diff.max():.6e}")
    print(f"  mean abs diff: {abs_diff.mean():.6e}")
    print(f"  median abs diff: {np.median(abs_diff):.6e}")
    print()

    # Per-condition summary
    cond_names = ["cond1", "cond2", "difference"]
    print("Per-condition max abs diff")
    for ci, name in enumerate(cond_names):
        max_diff = abs_diff[:, ci, :, :].max()
        mean_diff = abs_diff[:, ci, :, :].mean()
        print(f"  {name:>10s}: max={max_diff:.6e}, mean={mean_diff:.6e}")

    print()

    # Per-subject sanity (often very revealing)
    print("Per-subject max abs diff")
    for si in range(old.shape[0]):
        max_diff = abs_diff[si].max()
        print(f"  subject[{si:02d}]: {max_diff:.6e}")

    print()

    # Tolerance-based verdict (adjust if needed)
    atol = 1e-12
    rtol = 1e-7
    if np.allclose(old, new, atol=atol, rtol=rtol):
        print(f"✅ Arrays are numerically close (atol={atol}, rtol={rtol})")
    else:
        print(f"❌ Arrays are NOT numerically close (atol={atol}, rtol={rtol})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_evoked_data.py OLD.npy NEW.npy")
        sys.exit(1)

    main(Path(sys.argv[1]), Path(sys.argv[2]))
