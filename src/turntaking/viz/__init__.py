"""Turn-taking visualization subpackage.

Public API
----------
This package re-exports the same figure functions that previously lived in
`visualization.py`, so you can migrate imports with minimal churn.

All functions are intended to keep figure outputs identical to the legacy script.
"""

from .figures.behavior import plot_behavior, plot_other_duration
from .figures.erp import (
    plot_electrode_time_course,
    plot_joint_erps,
    plot_latency_erp_with_histograms,
    plot_topo_selection,
)
from .figures.tfr import plot_tfr_electrode_time_course, plot_tfr_topo
from .figures.decoding import plot_decoding
from ._io import plot_colorbar

__all__ = [
    "plot_behavior",
    "plot_other_duration",
    "plot_topo_selection",
    "plot_electrode_time_course",
    "plot_latency_erp_with_histograms",
    "plot_joint_erps",
    "plot_tfr_topo",
    "plot_tfr_electrode_time_course",
    "plot_decoding",
    "plot_colorbar",
]
