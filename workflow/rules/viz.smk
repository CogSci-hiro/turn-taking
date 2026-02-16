# workflow/rules/viz.smk

from pathlib import Path


def active_configfile() -> str:
    cfgs = list(getattr(workflow, "overwrite_configfiles", []))
    if not cfgs:
        cfgs = list(getattr(workflow, "configfiles", []))
    if not cfgs:
        raise ValueError("No configfile is available in Snakemake workflow context.")
    return str(cfgs[0])


CONFIGFILE = active_configfile()


def heavy_threads() -> int:
    return int(config.get("execution", {}).get("threads_heavy", 10))


def heavy_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10_000))


CONTRASTS = list(config.get("analysis", {}).get("contrasts", []))
BANDS = list(config.get("analysis", {}).get("bands", []))

FIG_ROOT = out_dir() / "figures"
ERP_ROOT = out_dir() / "erp"
TFR_ROOT = out_dir() / "tfr"
DECODING_ROOT = out_dir() / "decoding" / "erp"
STATS_ROOT = out_dir() / "stats"


def _erp_condition_files(contrast: str) -> tuple[str, str]:
    if contrast == "duration":
        return "long_ave.fif", "short_ave.fif"
    if contrast == "latency":
        return "fast_ave.fif", "slow_ave.fif"
    raise ValueError(f"Unknown ERP contrast: {contrast}")


def _erp_outputs_for_contrast(contrast: str) -> list[str]:
    cond1, cond2 = _erp_condition_files(contrast)
    base = ERP_ROOT / contrast
    return [
        str(base / "difference_ave.fif"),
        str(base / "evoked-data.npy"),
        str(base / "n_trials.csv"),
        str(base / "metadata.hdf5"),
        str(base / "offsets.csv"),
        str(base / cond1),
        str(base / cond2),
    ]


def _tfr_condition_files(contrast: str) -> tuple[str, str]:
    if contrast == "duration":
        return "long_ave.fif", "short_ave.fif"
    if contrast == "latency":
        return "fast_ave.fif", "slow_ave.fif"
    raise ValueError(f"Unknown TFR contrast: {contrast}")


def _tfr_outputs_for_contrast_band(contrast: str, band: str) -> list[str]:
    cond1, cond2 = _tfr_condition_files(contrast)
    base = TFR_ROOT / contrast / band
    return [
        str(base / "difference_ave.fif"),
        str(base / "induced-data.npy"),
        str(base / "n_trials.csv"),
        str(base / "metadata.hdf5"),
        str(base / cond1),
        str(base / cond2),
    ]


ERP_OUT = [path for contrast in CONTRASTS for path in _erp_outputs_for_contrast(contrast)]
TFR_OUT = [
    path
    for contrast in CONTRASTS
    for band in BANDS
    for path in _tfr_outputs_for_contrast_band(contrast, band)
]
DECODING_OUT = [
    str(DECODING_ROOT / contrast / "scores.npy")
    for contrast in CONTRASTS
] + [
    str(DECODING_ROOT / contrast / "times.npy")
    for contrast in CONTRASTS
]
ERP_CLUSTER_OUT = expand(
    str(STATS_ROOT / "erp" / "{contrast}" / "cluster_results.hdf5"),
    contrast=CONTRASTS,
)
TFR_CLUSTER_OUT = expand(
    str(STATS_ROOT / "tfr" / "{contrast}" / "{band}" / "cluster_results.hdf5"),
    contrast=CONTRASTS,
    band=BANDS,
)
DECODING_CLUSTER_OUT = expand(
    str(STATS_ROOT / "decoding" / "erp" / "{contrast}" / "cluster_results.hdf5"),
    contrast=CONTRASTS,
)


FIG_MAIN = [
    str(FIG_ROOT / "main" / "F_behavior.tif"),
    str(FIG_ROOT / "main" / "F_erp_timecourse.tif"),
    str(FIG_ROOT / "main" / "F_erp_topomap.tif"),
    str(FIG_ROOT / "main" / "F_tfr_topomap.tif"),
    str(FIG_ROOT / "main" / "F_decoding.tif"),
]

FIG_SUPP = [
    str(FIG_ROOT / "supp" / "S1_response_duration_hist.tif"),
    str(FIG_ROOT / "supp" / "S2_previous_speech_duration_hist.tif"),
    str(FIG_ROOT / "supp" / "S3_long_joint.tif"),
    str(FIG_ROOT / "supp" / "S3_short_joint.tif"),
    str(FIG_ROOT / "supp" / "S3_fast_joint.tif"),
    str(FIG_ROOT / "supp" / "S3_slow_joint.tif"),
    str(FIG_ROOT / "main" / "F_erp_timecourse_hist.tif"),
    str(FIG_ROOT / "supp" / "F_erp_topo_duration.tif"),
    str(FIG_ROOT / "supp" / "F_erp_topo_latency.tif"),
    str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_duration.tif"),
    str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_latency.tif"),
    str(FIG_ROOT / "supp" / "F_tfr_topo_beta_duration.tif"),
    str(FIG_ROOT / "supp" / "F_tfr_topo_beta_latency.tif"),
]


rule figures_main:
    """
    Aggregate target for all main manuscript figures.
    """
    input:
        FIG_MAIN


rule figures_supp:
    """
    Aggregate target for all supplementary figures.
    """
    input:
        FIG_SUPP


rule fig_erp_timecourse:
    input:
        config=CONFIGFILE,
        erp=ERP_OUT,
    output:
        fig=str(FIG_ROOT / "main" / "F_erp_timecourse.tif"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" erp --mode timecourse
        """


rule fig_erp_topos:
    input:
        config=CONFIGFILE,
        erp=ERP_OUT,
        clusters=ERP_CLUSTER_OUT,
    output:
        duration=str(FIG_ROOT / "supp" / "F_erp_topo_duration.tif"),
        latency=str(FIG_ROOT / "supp" / "F_erp_topo_latency.tif"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" erp --mode topomap --format static
        """


rule fig_tfr_topos:
    input:
        config=CONFIGFILE,
        tfr=TFR_OUT,
        clusters=TFR_CLUSTER_OUT,
    output:
        alpha_duration=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_duration.tif"),
        alpha_latency=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_latency.tif"),
        beta_duration=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_duration.tif"),
        beta_latency=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_latency.tif"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" tfr --mode topomap --format static
        """


rule fig_behavior:
    input:
        config=CONFIGFILE,
        table=str(out_dir() / "beh" / "turn_table.csv"),
        duration_offsets=str(out_dir() / "erp" / "duration" / "offsets.csv"),
        latency_offsets=str(out_dir() / "erp" / "latency" / "offsets.csv"),
    output:
        main=str(FIG_ROOT / "main" / "F_behavior.tif"),
        s1=str(FIG_ROOT / "supp" / "S1_response_duration_hist.tif"),
        s2=str(FIG_ROOT / "supp" / "S2_previous_speech_duration_hist.tif"),
        s3_long=str(FIG_ROOT / "supp" / "S3_long_joint.tif"),
        s3_short=str(FIG_ROOT / "supp" / "S3_short_joint.tif"),
        s3_fast=str(FIG_ROOT / "supp" / "S3_fast_joint.tif"),
        s3_slow=str(FIG_ROOT / "supp" / "S3_slow_joint.tif"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" behavior
        """


rule fig_decoding:
    input:
        config=CONFIGFILE,
        decoding=DECODING_OUT,
        clusters=DECODING_CLUSTER_OUT,
    output:
        fig=str(FIG_ROOT / "main" / "F_decoding.tif"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" decoding --mode figure
        """


rule fig_erp_topomap_svg:
    input:
        template="workflow/templates/ERP-timeline.svg",
        config=CONFIGFILE,
        erp=ERP_OUT,
        clusters=ERP_CLUSTER_OUT,
    output:
        svg=str(FIG_ROOT / "main" / "F_erp_topomap.svg"),
    params:
        parts_dir=str(FIG_ROOT / "main" / "parts_erp_topomap"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" erp --mode topomap --format svg
        """


rule fig_erp_topomap_tif:
    input:
        svg=str(FIG_ROOT / "main" / "F_erp_topomap.svg"),
        config=CONFIGFILE,
    output:
        tif=str(FIG_ROOT / "main" / "F_erp_topomap.tif"),
    shell:
        r"""
        set -euo pipefail
        if [ -d /opt/homebrew/lib ]; then
          export DYLD_LIBRARY_PATH="/opt/homebrew/lib:${{DYLD_LIBRARY_PATH:-}}"
        fi
        if [ -d /usr/local/lib ]; then
          export DYLD_LIBRARY_PATH="/usr/local/lib:${{DYLD_LIBRARY_PATH:-}}"
        fi
        python -m turntaking.cli.main viz-svg-to-tiff \
          --config "{input.config}" \
          --in-svg "{input.svg}" \
          --out-tif "{output.tif}" \
          --dpi 300
        """


rule fig_tfr_topomap_svg:
    input:
        template="workflow/templates/TF-timeline.svg",
        config=CONFIGFILE,
        tfr=TFR_OUT,
        clusters=TFR_CLUSTER_OUT,
    output:
        svg=str(FIG_ROOT / "main" / "F_tfr_topomap.svg"),
    params:
        parts_dir=str(FIG_ROOT / "main" / "parts_tfr_topomap"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" tfr --mode topomap --format svg
        """


rule fig_tfr_topomap_tif:
    input:
        svg=str(FIG_ROOT / "main" / "F_tfr_topomap.svg"),
        config=CONFIGFILE,
    output:
        tif=str(FIG_ROOT / "main" / "F_tfr_topomap.tif"),
    shell:
        r"""
        set -euo pipefail
        if [ -d /opt/homebrew/lib ]; then
          export DYLD_LIBRARY_PATH="/opt/homebrew/lib:${{DYLD_LIBRARY_PATH:-}}"
        fi
        if [ -d /usr/local/lib ]; then
          export DYLD_LIBRARY_PATH="/usr/local/lib:${{DYLD_LIBRARY_PATH:-}}"
        fi
        python -m turntaking.cli.main viz-svg-to-tiff \
          --config "{input.config}" \
          --in-svg "{input.svg}" \
          --out-tif "{output.tif}" \
          --dpi 300
        """


rule fig_erp_latency_with_hist:
    input:
        config=CONFIGFILE,
        mixed_table=str(out_dir() / "mixed_effect" / "table.csv"),
        erp=ERP_OUT,
    output:
        fig=str(FIG_ROOT / "main" / "F_erp_timecourse_hist.tif"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" erp --mode hist
        """
