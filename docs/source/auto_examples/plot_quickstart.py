"""
Quickstart: ERP difference wave
===============================

This example shows the shape conventions used by the turntaking ERP pipeline and
generates a tiny diagnostic plot from synthetic data.

The real workflow operates on MNE ``Epochs`` files on disk, but the numerical
core functions can be demonstrated on arrays directly.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from turntaking.analysis.erp.core import compute_contrast


# Create two synthetic condition ERPs (channels x times).
rng = np.random.default_rng(0)
n_channels = 5
n_times = 200
times_s = np.linspace(-0.2, 0.5, n_times)

erp_a = rng.normal(scale=1.0, size=(n_channels, n_times))
erp_b = rng.normal(scale=1.0, size=(n_channels, n_times))

# Contrast is defined as condition1 - condition2 across the full array.
diff = compute_contrast(erp_a, erp_b)

# Plot a single channel's difference wave.
fig, ax = plt.subplots(figsize=(5, 2.5), constrained_layout=True)
ax.plot(times_s, diff[0], lw=1.5)
ax.axvline(0.0, color="k", lw=0.8, alpha=0.5)
ax.set(title="Synthetic ERP difference (channel 0)", xlabel="Time (s)", ylabel="Amplitude (a.u.)")

