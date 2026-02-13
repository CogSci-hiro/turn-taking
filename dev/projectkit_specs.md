# Project Creation & Automation Framework — Design Specification

## Purpose

Define a standardized, reusable framework for creating and extending research projects with minimal boilerplate overhead. The primary goal is to reduce repeated setup work and allow focus on scientific logic rather than infrastructure.

This framework should:

* Provide consistent structure across projects
* Support different project types with shared foundations
* Automate common scaffolding tasks
* Improve communication and context transfer between development sessions (including AI collaboration)
* Enable progressive visualization and introspection of project structure

---

# 1. Project Types

Three primary project categories are defined.

## A) Generic Project

Minimal structure suitable for reusable software or utilities.

Characteristics:

* CLI-based entrypoints
* Logging and configuration infrastructure
* Modular source layout
* Minimal dependencies

Typical components:

* CLI interface
* Config handling
* Logging setup
* Utilities

---

## B) Scientific Project

Primary research or analysis repository accompanying a paper or study.

Characteristics:

* Snakemake-based workflow orchestration
* Analysis pipelines separated from CLI layer
* Reusable IO, visualization, and statistics modules
* Strong reproducibility guarantees

Additional components:

* workflow/ directory
* analysis pipelines
* statistical utilities
* report generation

---

## C) Data Curation Project

Focused on dataset preparation, annotation, preprocessing, and validation.

Characteristics:

* Annotation schemas and validation
* Data-specific preprocessing pipelines
* Quality control tools
* Dataset documentation

Additional components:

* preprocessing modules
* annotation schemas
* dataset metadata / provenance tracking

---

# 2. Canonical Project Structure

```
project-root/
├── pyproject.toml
├── config/
├── src/<pkg_name>/
│   ├── cli/
│   ├── analysis/
│   │   ├── io/
│   │   ├── features/
│   │   ├── models/
│   │   └── pipelines/
│   ├── viz/
│   ├── stats/
│   ├── utils/
│   └── logging.py
├── workflow/            (scientific projects)
├── tests/
├── docs/
├── reports/
├── notebooks/
└── .snapshots/
```

---

# 3. Layered Architecture Philosophy

Dependencies should generally flow downward:

```
CLI layer        → entrypoints only
Workflow layer   → orchestration only
Analysis layer   → core scientific logic
IO layer         → loading/saving conventions
Stats layer      → reusable inference tools
Viz layer        → visualization functions
Utils            → small helpers only
```

Key rules:

* CLI contains no scientific logic.
* Workflow orchestrates but does not implement logic.
* Analysis modules are reusable and testable independently.
* Visualization separated from computation.

---

# 4. Repeating Tasks to Automate

## CLI Infrastructure

* Standard location: `cli/main.py` and `cli/commands/*`
* Command registration scaffold
* Automatic test stub generation

## Logging

* Debug mode:

  * full tracebacks
  * verbose console logging

* Production mode:

  * structured logging
  * continue processing where safe
  * aggregated error reporting

## Configuration

* YAML configuration files
* Schema validation (e.g., pydantic/msgspec/dataclasses)
* Default config generation

## Snakemake Templates

* Modular rules (one rule per file recommended)
* Standardized log and benchmark paths
* Separation between workflow and analysis logic

## IO Templates

* Standard load/save conventions
* Atomic writes
* Metadata sidecars

## Visualization Helpers

* Standard figure saving
* Shared styles

## Statistical Utilities

* Reusable statistical methods
* Standard result format

## Documentation Automation

* Config documentation from schema
* API documentation from signatures

## Schema / Contract Plumbing

* Reusable validation layers
* Interface consistency between modules

---

# 5. ProjectKit Tooling (Conceptual)

A central tool provides automation and scaffolding.

Example commands:

```
proj new <name>
proj add feature <feature_name>
proj add command <command_name>
proj add rule <rule_name>
proj add io <entity>
proj snapshot <mode>
proj viz <mode>
```

ProjectKit contains:

* project generators
* scaffolding wizards
* shared runtime helpers (optional)
* snapshot and visualization tools

Individual projects contain only generated structure and domain logic.

---

# 6. Project Creation Workflow

Example:

```
proj new turntaking --type scientific \
  --features cli,config,logging,io,viz,stats,snakemake \
  --extras mne,pandas,sklearn
```

Result:

* Fully runnable CLI skeleton
* Workflow scaffold
* Config and logging setup
* Test stubs

---

# 7. Project Extension Workflow

Two types of extension:

## A) Feature Extension

```
proj add snakemake
proj add stats
```

Adds missing structure safely without overwriting existing work.

## B) Work Unit Generation

Examples:

* Add CLI command
* Add Snakemake rule
* Add IO module
* Add analysis pipeline skeleton

---

# 8. Communication Snapshot System

Purpose:

Provide compact project context for efficient collaboration.

Snapshot modes:

## API Snapshot

* Public function signatures only
* Module locations
* First-line docstrings

## Structure Snapshot

* Directory tree
* Entrypoints
* Workflow overview

## Dependency Slice

* Function call graph starting from seed symbol
* Limited depth

## Goal Snapshot

Given a task description:

* Identify relevant modules
* Relevant workflow rules
* Config keys
* Suggested test locations

Output stored in `.snapshots/`.

---

# 9. Visualization System

Avoid large unreadable diagrams by using layered views.

Supported visualization modes:

* Architecture overview (package-level)
* Module interaction view
* Dependency slice graph
* Class structure (public methods only)
* CLI command flow diagram

Design constraints:

* Node count limits
* Auto-collapsing of modules
* Progressive drill-down navigation
* Optional Graphviz or interactive HTML output

---

# 10. Design Principles

* Consistency over flexibility at project start
* Thin template, shared tooling logic
* Explicit layering reduces complexity
* Progressive visualization instead of monolithic diagrams
* Focus developer effort on scientific logic

---

# Current Status

Conceptual specification agreed upon.

Next Steps:

1. Define minimal “golden template” files.
2. Decide runtime dependency strategy for shared tooling.
3. Implement ProjectKit core commands incrementally.
4. Build snapshot generation first (high impact).
5. Add visualization tooling on top of snapshot metadata.
