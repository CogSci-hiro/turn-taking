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

rule erp_group:
    input:
        epochs=epoch_inputs(),
        config=str(Path(workflow.basedir) / "config.yaml")
    output:
        ERP_OUT
    threads:
        _heavy_threads()
    resources:
        mem_mb=_heavy_mem()
    shell:
        r"""
        set -euo pipefail
        python "{entrypoint()}" analyze erp --config "{input.config}"
        """


# same for tfr_group, mixed_group, decoding_group
