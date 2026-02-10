# workflow/rules/stats.smk

from pathlib import Path

conda:
    CONDA_PY_ENV


def stats_erp_out_dir() -> Path:
    return out_dir() / "stats" / "erp"

def stats_tfr_out_dir() -> Path:
    return out_dir() / "stats" / "tfr"

def stats_decoding_out_dir() -> Path:
    return out_dir() / "stats" / "decoding"

def heavy_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10_000))


T1_LMM = str(out_dir() / "tables" / "table_T1_lmm_results.csv")

# IMPORTANT: Keep this list in sync with analysis.smk ERP outputs
ERP_OUT = [
    str(out_dir() / "erp" / "manifest.json"),
    str(out_dir() / "erp" / "grand_average-ave.fif"),
    str(out_dir() / "erp" / "stats.csv"),
]

TFR_OUT = [
    str(out_dir() / "tfr" / "manifest.json"),
    str(out_dir() / "tfr" / "grand_average-tfr.h5"),
    str(out_dir() / "tfr" / "stats.csv"),
]

DECODING_OUT = [
    str(out_dir() / "decoding" / "manifest.json"),
    str(out_dir() / "decoding" / "scores.csv"),
    str(out_dir() / "decoding" / "confusion_matrix.csv"),
]


rule test_erp:
    """
    Run ERP unit/integration tests and store the log as an output artifact.
    """
    input:
        erp_outputs=ERP_OUT,
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        log=str(stats_erp_out_dir() / "pytest.log"),
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



rule test_tfr:
    """
    Run TFR-related tests and store the pytest log as a tracked artifact.
    """
    input:
        tfr_outputs=TFR_OUT,
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        log=str(stats_tfr_out_dir() / "pytest.log"),
    threads: 1
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.log}")"

        # Narrow this once you have TFR-specific tests
        python -m pytest -q tests > "{output.log}"
        """


rule test_decoding:
    """
    Run decoding-related tests and store pytest log as a tracked artifact.
    """
    input:
        decoding_outputs=DECODING_OUT,
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        log=str(stats_decoding_out_dir() / "pytest.log"),
    threads: 1
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.log}")"

        # Narrow to decoding tests when available
        python -m pytest -q tests > "{output.log}"
        """


rule table_lmm:
    """
    Build Table T1: LMM results (CSV).
    R implementation can replace the backend later without changing the output contract.
    """
    input:
        epochs=epoch_inputs(),
        erp=ERP_OUT,
        tfr=TFR_OUT,
        decoding=DECODING_OUT,
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        T1_LMM
    threads: 1
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" stats lmm --config "{input.config}" --out "{output}"
        """