# rules/analysis.smk

conda:
    CONDA_PY_ENV

OUT = out_dir()

TFR_OUT = [
    str(OUT / "tfr" / "manifest.json"),
    str(OUT / "tfr" / "grand_average-tfr.h5"),
    str(OUT / "tfr" / "stats.csv"),
]

MIXED_OUT = [
    str(OUT / "mixed" / "manifest.json"),
    str(OUT / "mixed" / "lmm_table.csv"),
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


CONTRASTS = config["analysis"]["contrasts"]

ERP_COND_NAMES = {
    "duration": ("long", "short"),
    "latency": ("fast", "slow"),
}
ERP_ROOT = config["io"]["out_dir"] + "/erp"



rule erp:
    input:
        config="workflow/config.yaml"   # or your actual config input
    output:
        expand(ERP_ROOT + "/{contrast}/difference_ave.fif",contrast=CONTRASTS),
        expand(ERP_ROOT + "/{contrast}/evoked-data.npy",contrast=CONTRASTS),
        expand(ERP_ROOT + "/{contrast}/n_trials.csv",contrast=CONTRASTS),
        expand(ERP_ROOT + "/{contrast}/results.hdf5",contrast=CONTRASTS),
        expand(ERP_ROOT + "/{contrast}/offsets.csv",contrast=CONTRASTS),

        # Condition-specific filenames (2 per contrast)
        ERP_ROOT + "/duration/long_ave.fif",
        ERP_ROOT + "/duration/short_ave.fif",
        ERP_ROOT + "/latency/fast_ave.fif",
        ERP_ROOT + "/latency/slow_ave.fif"
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main erp --config "{input.config}"
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


rule decoding:
    input:
        epochs=epoch_inputs(),
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        DECODING_OUT
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" analyze decoding --config "{input.config}"
        """


rule mixed:
    input:
        epochs=epoch_inputs(),
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        MIXED_OUT
    threads:
        light_threads()
    resources:
        mem_mb=light_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" analyze mixed --config "{input.config}"
        """
