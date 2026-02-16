# workflow/rules/stats.smk

from pathlib import Path

conda:
    CONDA_PY_ENV


def active_configfile() -> str:
    cfgs = list(getattr(workflow, "overwrite_configfiles", []))
    if not cfgs:
        cfgs = list(getattr(workflow, "configfiles", []))
    if not cfgs:
        raise ValueError("No configfile is available in Snakemake workflow context.")
    return str(cfgs[0])


CONFIGFILE = active_configfile()


def stats_erp_out_dir() -> Path:
    return out_dir() / "stats" / "erp"

def stats_tfr_out_dir() -> Path:
    return out_dir() / "stats" / "tfr"

def stats_decoding_out_dir() -> Path:
    return out_dir() / "stats" / "decoding"

def heavy_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10_000))


T1_LMM = str(out_dir() / "mixed_effect" / "lmm" / "tables" / "models.csv")
INTEGRATIVE_LMM_OUT = [
    str(out_dir() / "mixed_effect" / "integration" / "joint_model.csv"),
    str(out_dir() / "mixed_effect" / "integration" / "interactions.csv"),
    str(out_dir() / "mixed_effect" / "integration" / "random_slope.csv"),
    str(out_dir() / "mixed_effect" / "integration" / "partial_correlations.csv"),
]


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


DECODING_OUT = [
    str(out_dir() / "decoding" / "manifest.json"),
    str(out_dir() / "decoding" / "scores.csv"),
    str(out_dir() / "decoding" / "confusion_matrix.csv"),
]


rule test_erp:
    input:
        erp_outputs=ERP_OUT,  # or just the specific per-contrast inputs
        config=CONFIGFILE
    output:
        hdf5=str(stats_erp_out_dir() / "{contrast}" / "cluster_results.hdf5"),
        summary=str(stats_erp_out_dir() / "{contrast}" / "cluster_summary.csv")
    params:
        entrypoint=str(entrypoint())
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


# =============================================================================
# TFR cluster tests (mirrors ERP pattern, but adds {band})
# =============================================================================

def tfr_contrasts() -> list[str]:
    return list(config.get("analysis", {}).get("contrasts", []))


def tfr_bands() -> list[str]:
    return list(config.get("analysis", {}).get("bands", []))


def tfr_outputs_for_contrast_band(contrast: str, band: str) -> list[str]:
    base = out_dir() / "tfr" / contrast / band

    if contrast == "duration":
        cond1 = "long_ave.fif"
        cond2 = "short_ave.fif"
    elif contrast == "latency":
        cond1 = "fast_ave.fif"
        cond2 = "slow_ave.fif"
    else:
        raise ValueError(f"Unknown TFR contrast: {contrast}")

    return [
        str(base / "difference_ave.fif"),
        str(base / "induced-data.npy"),
        str(base / cond1),
        str(base / cond2),
        str(base / "n_trials.csv"),
        str(base / "metadata.hdf5"),
    ]


TFR_OUT = [
    p
    for c in tfr_contrasts()
    for b in tfr_bands()
    for p in tfr_outputs_for_contrast_band(c, b)
]


rule test_tfr:
    """
    TFR cluster permutation test (per contrast × band).
    Do not target this rule directly (it contains wildcards).
    Target `test_tfr_all` instead.
    """
    input:
        tfr_outputs=lambda wc: tfr_outputs_for_contrast_band(wc.contrast, wc.band),
        config=CONFIGFILE
    output:
        hdf5=str(stats_tfr_out_dir() / "{contrast}" / "{band}" / "cluster_results.hdf5"),
        summary=str(stats_tfr_out_dir() / "{contrast}" / "{band}" / "cluster_summary.csv")
    threads: 1
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" cluster --config "{input.config}" --kind tfr --contrast "{wildcards.contrast}" --band "{wildcards.band}"
        """


TFR_CLUSTER_OUT = [
    *expand(
        str(stats_tfr_out_dir() / "{contrast}" / "{band}" / "cluster_results.hdf5"),
        contrast=tfr_contrasts(),
        band=tfr_bands(),
    ),
    *expand(
        str(stats_tfr_out_dir() / "{contrast}" / "{band}" / "cluster_summary.csv"),
        contrast=tfr_contrasts(),
        band=tfr_bands(),
    ),
]

rule test_tfr_all:
    """
    Wildcard-free TFR cluster test target (all configured contrasts × bands).
    """
    input:
        TFR_CLUSTER_OUT


# Add near the top with other OUT lists
def decoding_contrasts() -> list[str]:
    return list(config.get("analysis",{}).get("contrasts",[]))

rule test_decoding:
    input:
        scores=str(out_dir() / "decoding" / "erp" / "{contrast}" / "scores.npy"),
        times=str(out_dir() / "decoding" / "erp" / "{contrast}" / "times.npy"),
        config=CONFIGFILE,
    output:
        hdf5=str(stats_decoding_out_dir() / "erp" / "{contrast}" / "cluster_results.hdf5"),
        summary=str(stats_decoding_out_dir() / "erp" / "{contrast}" / "cluster_summary.csv"),
    threads: 1
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" decoding-cluster \
          --config "{input.config}" \
          --feature erp \
          --contrast "{wildcards.contrast}"
        """


DECODING_CLUSTER_OUT = [
    *expand(str(stats_decoding_out_dir() / "erp" / "{contrast}" / "cluster_results.hdf5"),contrast=decoding_contrasts()),
    *expand(str(stats_decoding_out_dir() / "erp" / "{contrast}" / "cluster_summary.csv"),contrast=decoding_contrasts()),
]

rule test_decoding_all:
    input:
        DECODING_CLUSTER_OUT


rule lmm_fit:
    input:
        table=MIXED_ROOT + "/table.csv"
    output:
        models_dir=MIXED_ROOT + "/lmm/models/.done",
        tables_dir=MIXED_ROOT + "/lmm/tables/models.csv",
    shell:
        r"""
        mkdir -p {MIXED_ROOT}/lmm/models
        Rscript workflow/scripts/fit_lmm.R \
          --in "{input.table}" \
          --out "{MIXED_ROOT}/lmm" \
          --zscore TRUE \
          --run_as_factor FALSE
        touch {MIXED_ROOT}/lmm/models/.done
        """


rule fit_integrative_lmm:
    input:
        table=MIXED_ROOT + "/table.csv"
    output:
        joint_model=MIXED_ROOT + "/integration/joint_model.csv",
        interactions=MIXED_ROOT + "/integration/interactions.csv",
        random_slope=MIXED_ROOT + "/integration/random_slope.csv",
        partial_correlations=MIXED_ROOT + "/integration/partial_correlations.csv",
    script:
        "../scripts/fit_integrative_lmm.R"
