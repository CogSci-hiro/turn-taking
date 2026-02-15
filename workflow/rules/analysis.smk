# rules/analysis.smk

OUT = out_dir()


def heavy_threads() -> int:
    return int(config.get("execution", {}).get("threads_heavy", 10))


def light_threads() -> int:
    return int(config.get("execution", {}).get("threads_light", 1))


def heavy_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10000))


def light_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_light", 1000))


def active_configfile() -> str:
    cfgs = list(getattr(workflow, "overwrite_configfiles", []))
    if not cfgs:
        cfgs = list(getattr(workflow, "configfiles", []))
    if not cfgs:
        raise ValueError("No configfile is available in Snakemake workflow context.")
    return str(cfgs[0])


CONFIGFILE = active_configfile()

CONTRASTS = config["analysis"]["contrasts"]
BANDS = config["analysis"]["bands"]

ERP_ROOT = str(OUT / "erp")
TFR_ROOT = str(OUT / "tfr")
DECODING_ROOT = str(OUT / "decoding/erp")
MIXED_ROOT = str(OUT / "mixed_effect")


rule erp:
    input:
        configfile=CONFIGFILE
    output:
        expand(f"{ERP_ROOT}/{{contrast}}/difference_ave.fif", contrast=CONTRASTS),
        expand(f"{ERP_ROOT}/{{contrast}}/evoked-data.npy", contrast=CONTRASTS),
        expand(f"{ERP_ROOT}/{{contrast}}/n_trials.csv", contrast=CONTRASTS),
        expand(f"{ERP_ROOT}/{{contrast}}/metadata.hdf5", contrast=CONTRASTS),
        expand(f"{ERP_ROOT}/{{contrast}}/offsets.csv", contrast=CONTRASTS),
        f"{ERP_ROOT}/duration/long_ave.fif",
        f"{ERP_ROOT}/duration/short_ave.fif",
        f"{ERP_ROOT}/latency/fast_ave.fif",
        f"{ERP_ROOT}/latency/slow_ave.fif"
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main erp --config "{input.configfile}"
        """


rule tfr:
    input:
        config="workflow/config.yaml"
    output:
        # Per contrast × band core files
        expand(TFR_ROOT + "/{contrast}/{band}/difference_ave.fif", contrast=CONTRASTS, band=BANDS),
        expand(TFR_ROOT + "/{contrast}/{band}/induced-data.npy", contrast=CONTRASTS, band=BANDS),
        expand(TFR_ROOT + "/{contrast}/{band}/n_trials.csv", contrast=CONTRASTS, band=BANDS),
        expand(TFR_ROOT + "/{contrast}/{band}/metadata.hdf5", contrast=CONTRASTS, band=BANDS),

        # Condition-specific averages (2 per contrast × band)
        expand(TFR_ROOT + "/duration/{band}/long_ave.fif", band=BANDS),
        expand(TFR_ROOT + "/duration/{band}/short_ave.fif", band=BANDS),
        expand(TFR_ROOT + "/latency/{band}/fast_ave.fif", band=BANDS),
        expand(TFR_ROOT + "/latency/{band}/slow_ave.fif", band=BANDS),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main tfr --config "{input.config}"
        """


rule decoding:
    input:
        config="workflow/config.yaml"
    output:
        scores=DECODING_ROOT + "/{contrast}/scores.npy",
        times=DECODING_ROOT + "/{contrast}/times.npy",
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main decoding \
          --config "{input.config}" \
          --contrast "{wildcards.contrast}"
        """


rule decoding_all:
    input:
        expand(DECODING_ROOT + "/{contrast}/scores.npy", contrast=CONTRASTS),
        expand(DECODING_ROOT + "/{contrast}/times.npy", contrast=CONTRASTS),


rule mixed_effect:
    input:
        config="workflow/config.yaml"
    output:
        table=MIXED_ROOT + "/table.csv"
    threads:
        light_threads()
    resources:
        mem_mb=light_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main mixed-effect --config "{input.config}" 

        """


rule mixed_effect_all:
    input:
        MIXED_ROOT + "/table.csv"