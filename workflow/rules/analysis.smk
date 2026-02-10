# rules/analysis.smk

conda:
    CONDA_PY_ENV

OUT = out_dir()

ERP_OUT = [
    str(OUT / "erp" / "manifest.json"),
    str(OUT / "erp" / "grand_average-ave.fif"),
    str(OUT / "erp" / "stats.csv"),
]

TFR_OUT = [
    str(OUT / "tfr" / "manifest.json"),
    str(OUT / "tfr" / "grand_average-tfr.h5"),
    str(OUT / "tfr" / "stats.csv"),
]

MIXED_OUT = [
    str(OUT / "mixed" / "manifest.json"),
    str(OUT / "mixed" / "lmm_results.csv"),
]

DECODING_OUT = [
    str(OUT / "decoding" / "manifest.json"),
    str(OUT / "decoding" / "scores.csv"),
    str(OUT / "decoding" / "confusion_matrix.csv"),
]


def heavy_threads() -> int:
    return int(config.get("execution", {}).get("threads_heavy", 10))

def light_threads() -> int:
    return int(config.get("execution", {}).get("threads_light", 1))

def heavy_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10000))

def light_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_light", 1000))



rule erp:
    input:
        epochs=epoch_inputs(),
        config=str(Path(workflow.basedir) / "config.yaml")
    output:
        ERP_OUT
    params:
        entrypoint=str(entrypoint())
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" erp --config "{input.config}"
        """


rule tfr:
    input:
        epochs=epoch_inputs(),
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        TFR_OUT
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" analyze tfr --config "{input.config}"
        """
