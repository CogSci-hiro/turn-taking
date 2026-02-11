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


def erp_contrasts() -> list[str]:
    return list(config.get("analysis", {}).get("contrasts", []))

def erp_outputs_for_contrast(contrast: str) -> list[str]:
    base = out_dir() / "erp" / contrast
    if contrast == "duration":
        cond1 = "long_ave.fif"
        cond2 = "short_ave.fif"
    elif contrast == "latency":
        cond1 = "fast_ave.fif"
        cond2 = "slow_ave.fif"
    else:
        raise ValueError(f"Unknown ERP contrast: {contrast}")

    return [
        str(base / "difference_ave.fif"),
        str(base / "evoked-data.npy"),
        str(base / cond1),
        str(base / cond2),
        str(base / "n_trials.csv"),
        str(base / "metadata.hdf5"),
        str(base / "offsets.csv"),
    ]

ERP_OUT = [p for c in erp_contrasts() for p in erp_outputs_for_contrast(c)]

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
    input:
        erp_outputs=ERP_OUT,  # or just the specific per-contrast inputs
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        hdf5=str(stats_erp_out_dir() / "{contrast}" / "cluster_results.hdf5"),
        summary=str(stats_erp_out_dir() / "{contrast}" / "cluster_summary.csv"),
    params:
        entrypoint=str(entrypoint()),
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" cluster --config "{input.config}" --kind erp --contrast "{wildcards.contrast}"
        """


ERP_CLUSTER_OUT = [
    *expand(
        str(stats_erp_out_dir() / "{contrast}" / "cluster_results.hdf5"),
        contrast=erp_contrasts(),
    ),
    *expand(
        str(stats_erp_out_dir() / "{contrast}" / "cluster_summary.csv"),
        contrast=erp_contrasts(),
    ),
]


rule test_erp_all:
    """
    Run ERP cluster tests for all configured contrasts.
    Wildcard-free target rule.
    """
    input:
        ERP_CLUSTER_OUT


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