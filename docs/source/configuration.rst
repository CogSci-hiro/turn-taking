Configuration
=============

The canonical config file lives at ``workflow/config.yaml`` and is loaded with:

.. code-block:: python

   from pathlib import Path
   from turntaking.config.loader import load_config

   cfg = load_config(Path("workflow/config.yaml"))

Schema
------

The typed schema is defined in ``turntaking.config.analysis_schema``.
At a minimum you must provide:

- ``io.epoch_dir``: directory containing epoch FIFs
- ``io.epoch_pattern``: filename pattern with ``{subject}``, ``{task}``, ``{run}``
- ``io.out_dir``: output root directory
- ``dataset``: subject/task/run expansion controls
- ``analysis``: parameters for ERP/TFR/decoding/mixed-effect stages
- ``viz``: figure rendering parameters and paths

Input assumptions
-----------------

Epoch files must be MNE epochs FIF files and include a ``pandas.DataFrame`` in
``epochs.metadata`` with (at least):

- ``latency`` (float seconds)
- ``self_duration`` (float seconds)

