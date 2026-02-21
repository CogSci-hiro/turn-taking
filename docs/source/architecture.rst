Architecture
============

The codebase is organized around explicit boundaries:

- ``turntaking.cli``: argument parsing and dispatch to library entrypoints.
- ``turntaking.config``: typed YAML configuration schema and loader.
- ``turntaking.analysis``: domain logic and artifact contracts (ERP/TFR/decoding/mixed).
- ``turntaking.viz``: figure rendering from saved artifacts.
- ``workflow/``: Snakemake orchestration for analysis -> stats -> figures.

Pipeline overview
-----------------

.. mermaid::

   flowchart LR
     classDef input fill:#1d4ed8,stroke:#1e3a8a,color:#ffffff
     classDef compute fill:#10b981,stroke:#065f46,color:#052e1a
     classDef stats fill:#f59e0b,stroke:#92400e,color:#1f2937
     classDef viz fill:#a855f7,stroke:#6b21a8,color:#ffffff
     classDef out fill:#111827,stroke:#374151,color:#ffffff

     EPOCHS["Epochs FIF files"]:::input --> SEL["Selection + median split"]:::compute
     CFG["YAML config"]:::input --> SEL
     SEL --> ERP["ERP artifacts"]:::compute --> OUT["io.out_dir"]:::out
     SEL --> TFR["Induced TFR artifacts"]:::compute --> OUT
     SEL --> DEC["Decoding TG scores"]:::compute --> OUT
     OUT --> FIG["Figures"]:::viz --> OUT

