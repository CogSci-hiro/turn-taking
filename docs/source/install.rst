Install
=======

Editable install
----------------

From the repository root:

.. code-block:: bash

   python -m pip install -e .

Documentation dependencies
--------------------------

To build the documentation locally:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   make -C docs html

Tests
-----

.. code-block:: bash

   python -m pip install -e ".[test]"
   pytest

