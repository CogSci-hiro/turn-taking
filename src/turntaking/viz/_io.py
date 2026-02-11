"""I/O helpers for saving / exporting standalone legend assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

def plot_colorbar(vmin: float, vmax: float, out_file: str | Path, dpi: int):

    plt.figure(figsize=(1, 5))

    array = np.array([[vmin, vmax]])
    plt.imshow(array, cmap="RdBu_r")

    plt.gca().set_visible(False)
    cax = plt.axes((0.45, 0.1, 0.1, 0.8))
    plt.colorbar(orientation="vertical", cax=cax)
    plt.savefig(out_file, dpi=dpi)
