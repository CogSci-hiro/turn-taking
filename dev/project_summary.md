# Turntaking Project — Analysis Pipeline Overview

This document summarizes the **current structure, responsibilities, and interfaces** of the turn-taking project *after refactoring preprocessing out of Snakemake*. It is intended as a **shared reference** for future discussions and implementation decisions.

---

## 1. High-level design philosophy

The project is now split into **two clearly separated layers**:

1. **Python pipeline (authoritative, stateful)**
   - Handles all preprocessing up to **epochs**
   - Exposed via a Typer-based CLI
   - Responsible for correctness, reproducibility, and explicit control

2. **Snakemake workflow (stateless, batch analysis)**
   - Consumes *already-generated epochs*
   - Produces ERP / TFR / mixed-effects / decoding analyses
   - Treated purely as an execution backend, not a user interface

The **CLI is the only supported user-facing interface**.
Snakemake is invoked programmatically by the CLI (or manually by developers).

---

## 2. Repository structure (current)

```
turntaking/
├── src/turntaking/
│   ├── cli/
│   │   └── main.py              # Typer dispatcher (currently only `preprocess`)
│   ├── preprocessing/
│   │   └── pipeline.py          # run_preprocessing_to_epochs()
│   ├── config/
│   │   └── loader.py            # load_config()
│   └── ...
│
├── workflow/
│   ├── Snakefile                # Analysis-only Snakemake entry
│   ├── config.yaml              # Analysis configuration (new schema)
│   └── rules/
│       ├── analysis.smk         # ERP / TFR / Mixed / Decoding data generation
│       ├── stats.smk            # Cluster tests / statistical tests
│       ├── viz.smk              # ERP / TFR visualizations
│       └── targets.smk          # Explicit *_all CLI-facing targets
│
├── scripts/
│   ├── analysis/
│   │   ├── erp_data.py
│   │   ├── tfr_data.py
│   │   ├── mixed_effect_data.py
│   │   ├── decoding_data.py
│   │   └── decoding_score.py
│   ├── stats/
│   │   ├── erp_test.py
│   │   ├── tfr_test.py
│   │   └── test_decoding.py
│   └── visualization/
│       ├── plot_erp.py
│       └── plot_tfr.py
│
└── README.md (optional)
```

---

## 3. Responsibilities by component

### 3.1 Python CLI (`src/turntaking/cli`)

**Authoritative control layer**.

Currently implemented:
- `turntaking preprocess`
  - Loads a preprocessing config
  - Runs preprocessing → epochs
  - Writes epochs to disk

Planned / implied:
- `turntaking analyze erp`
- `turntaking analyze tfr`
- `turntaking analyze mixed`
- `turntaking analyze decoding`

Each analysis command will:
1. Validate analysis config
2. Validate epoch directory
3. Invoke Snakemake with a specific target rule (e.g. `erp_all`)

The CLI **never passes file paths** to Snakemake — only target rule names.

---

### 3.2 Snakemake workflow (`workflow/`)

**Batch execution backend**. No user logic.

Key principles:
- No preprocessing rules
- No config-driven `target:` switching
- No subject-specific CLI logic
- Explicit, named entry rules

Snakemake assumes:
- Epochs already exist
- Epoch naming follows a declared pattern
- Missing runs are explicitly listed

---

## 4. Analysis configuration (`workflow/config.yaml`)

The config is now **analysis-only** and defines:

### 4.1 I/O contract

- `io.epoch_dir` — where epochs live
- `io.epoch_pattern` — filename template
- `io.out_dir` — where results go

Snakemake does not guess paths; everything is explicit.

---

### 4.2 Dataset definition

- How subjects are discovered (BIDS root or explicit list)
- Which tasks and runs exist
- Which `(subject, run)` pairs are invalid / missing

This replaces all previous regex-based filtering.

---

### 4.3 Analysis plan

- Explicit list of contrasts (no `"both"` special case)
- Explicit frequency bands
- ERP / TFR / mixed / decoding parameters

Each analysis block maps 1:1 to a Python script in `scripts/`.

---

### 4.4 Execution defaults (optional)

- Thread counts
- Memory presets

These are defaults only; Snakemake CLI flags may override them.

---

## 5. Epochs as the sole upstream dependency

All analysis rules depend on a single helper:

- `epoch_inputs()`

Which expands:

```
{epoch_dir}/{subject}_{task}_run-{run}_epo.fif
```

Epochs are treated as **external, immutable inputs**.
If epochs are missing, the workflow fails early.

---

## 6. Entry targets (`workflow/rules/targets.smk`)

These are the **only intended Snakemake entrypoints**:

- `erp_all`       → full ERP pipeline (data → stats → plots)
- `tfr_all`       → full TFR pipeline
- `mixed_all`     → mixed-effects CSV
- `decoding_all`  → decoding data → scores → tests
- `all_all`       → everything above

Each target rule **declares final output files only**.
Snakemake resolves the dependency graph automatically.

---

## 7. Key invariants (important assumptions)

These are relied upon throughout the project:

1. Preprocessing and analysis are **strictly separated**
2. Epochs are the only handshake between the two layers
3. Snakemake never modifies raw or epoch data
4. All subject/run exclusions are explicit and centralized
5. CLI, not Snakemake, defines user intent

Violating any of these should be a deliberate design change.

---

## 8. Typical execution flow

1. User runs:
   ```
   turntaking preprocess --config preprocess.yaml
   ```
2. Epochs are written to `io.epoch_dir`
3. User runs:
   ```
   turntaking analyze erp --config analysis.yaml
   ```
4. CLI invokes:
   ```
   snakemake erp_all --configfile analysis.yaml
   ```
5. Results appear under `io.out_dir`

---

## 9. Open design extensions (future)

- Subject-level analysis targets (e.g. `erp_subject_all`)
- Fast vs full analysis modes
- Per-contrast or per-band CLI flags
- Caching or partial invalidation strategies

These should be added **without reintroducing preprocessing into Snakemake**.

---

## 10. Summary (one paragraph)

The project now treats preprocessing as a **Python-owned, stateful pipeline** and analysis as a **Snakemake-owned, stateless batch workflow**. The CLI is the single point of truth for user intent, while Snakemake exposes explicit `*_all` targets that declare completion criteria. Epochs form the sole contract between the two layers, enabling clean refactoring, reproducibility, and future extension without entangling concerns.

