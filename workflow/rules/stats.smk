# workflow/rules/stats.smk

from pathlib import Path

conda:
    CONDA_PY_ENV


def stats_out_dir() -> Path:
    return out_dir() / "stats" / "erp"


def heavy_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10_000))


# IMPORTANT: Keep this list in sync with analysis.smk ERP outputs
ERP_OUT = [
    str(out_dir() / "erp" / "manifest.json"),
    str(out_dir() / "erp" / "grand_average-ave.fif"),
    str(out_dir() / "erp" / "stats.csv"),
]


rule test_erp:
    """
    Run ERP unit/integration tests and store the log as an output artifact.
    """
    input:
        erp_outputs=ERP_OUT,
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        log=str(stats_out_dir() / "pytest.log"),
    threads: 1
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.log}")"

        # Recommended: narrow to ERP-specific tests once you have them,
        # e.g. `python -m pytest -q tests/test_erp.py`
        python -m pytest -q tests > "{output.log}"
        """
