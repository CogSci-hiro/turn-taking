"""turntaking.analysis

Core analysis logic for the turn-taking project.

Design principles
-----------------
- No CLI parsing here.
- Avoid hard-coded filesystem assumptions; accept paths/objects as inputs.
- Prefer pure functions returning structured outputs (arrays, DataFrames, Evokeds).

CLI wrappers live in :mod:`turntaking.cli.commands`.
"""
