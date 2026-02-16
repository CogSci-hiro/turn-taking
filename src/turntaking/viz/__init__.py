"""Visualization package entrypoints."""

from turntaking.viz.decoding.entry import run as run_decoding
from turntaking.viz.erp.entry import run as run_erp
from turntaking.viz.tfr.entry import run as run_tfr

__all__ = ["run_erp", "run_tfr", "run_decoding"]
