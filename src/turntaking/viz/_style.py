"""Shared style and constants for turn-taking visualizations.

This module intentionally contains only *stable* plotting constants so that figures
remain visually identical across refactors.
"""

P_THRESHOLD = 0.05  # default significance threshold used in plots

DURATION_COLOR_1 = "#010fcc"  # true blue
DURATION_COLOR_2 = "#8ab8fe"  # carolina blue
LATENCY_COLOR_1 = "#e50000"  # red
LATENCY_COLOR_2 = "#ffb19a"  # pale salmon

JOINT_TIMES = (-2.0, -0.8, 0.0)  # times for joint plot

WIDTH = 8.27  # A4
TITLE_FONT_SIZE = 14
FONT_SIZE = 11
MARKER_SIZE = 15
SMALLER_MARKER_SIZE = 5
FACE_COLOR = "darkgray"
