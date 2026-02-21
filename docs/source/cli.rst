CLI
===

The CLI is implemented as a Python module entrypoint:

.. code-block:: bash

   python -m turntaking.cli.main --help

Common commands
---------------

Run all analysis stages:

.. code-block:: bash

   python -m turntaking.cli.main analyze --config workflow/config.yaml all

Run a single domain:

.. code-block:: bash

   python -m turntaking.cli.main analyze --config workflow/config.yaml erp
   python -m turntaking.cli.main analyze --config workflow/config.yaml tfr
   python -m turntaking.cli.main analyze --config workflow/config.yaml decoding --contrast duration
   python -m turntaking.cli.main analyze --config workflow/config.yaml mixed

Run figures:

.. code-block:: bash

   python -m turntaking.cli.main viz --config workflow/config.yaml erp --mode timecourse

