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
FIG_FORMATS = ("tif", "eps", "png")


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
    str((FIG_ROOT / "main" / "F_behavior").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "main" / "F_erp_timecourse").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "main" / "F_erp_topomap").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "main" / "F_tfr_topomap").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "main" / "F_decoding").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "main" / "F_lrt_comparisons_duration").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "main" / "F_lrt_comparisons_latency").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
]

FIG_SUPP = [
    str((FIG_ROOT / "supp" / "S1_response_duration_hist").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "S2_previous_speech_duration_hist").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "S3_long_joint").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "S3_short_joint").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "S3_fast_joint").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "S3_slow_joint").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "main" / "F_erp_timecourse_hist").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "F_erp_topo_duration").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "F_erp_topo_latency").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "F_tfr_topo_alpha_duration").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "F_tfr_topo_alpha_latency").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "F_tfr_topo_beta_duration").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
] + [
    str((FIG_ROOT / "supp" / "F_tfr_topo_beta_latency").with_suffix(f".{ext}"))
    for ext in FIG_FORMATS
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
        tif=str(FIG_ROOT / "main" / "F_erp_timecourse.tif"),
        eps=str(FIG_ROOT / "main" / "F_erp_timecourse.eps"),
        png=str(FIG_ROOT / "main" / "F_erp_timecourse.png"),
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
        duration_tif=str(FIG_ROOT / "supp" / "F_erp_topo_duration.tif"),
        duration_eps=str(FIG_ROOT / "supp" / "F_erp_topo_duration.eps"),
        duration_png=str(FIG_ROOT / "supp" / "F_erp_topo_duration.png"),
        latency_tif=str(FIG_ROOT / "supp" / "F_erp_topo_latency.tif"),
        latency_eps=str(FIG_ROOT / "supp" / "F_erp_topo_latency.eps"),
        latency_png=str(FIG_ROOT / "supp" / "F_erp_topo_latency.png"),
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
        alpha_duration_tif=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_duration.tif"),
        alpha_duration_eps=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_duration.eps"),
        alpha_duration_png=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_duration.png"),
        alpha_latency_tif=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_latency.tif"),
        alpha_latency_eps=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_latency.eps"),
        alpha_latency_png=str(FIG_ROOT / "supp" / "F_tfr_topo_alpha_latency.png"),
        beta_duration_tif=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_duration.tif"),
        beta_duration_eps=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_duration.eps"),
        beta_duration_png=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_duration.png"),
        beta_latency_tif=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_latency.tif"),
        beta_latency_eps=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_latency.eps"),
        beta_latency_png=str(FIG_ROOT / "supp" / "F_tfr_topo_beta_latency.png"),
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
        main_tif=str(FIG_ROOT / "main" / "F_behavior.tif"),
        main_eps=str(FIG_ROOT / "main" / "F_behavior.eps"),
        main_png=str(FIG_ROOT / "main" / "F_behavior.png"),
        s1_tif=str(FIG_ROOT / "supp" / "S1_response_duration_hist.tif"),
        s1_eps=str(FIG_ROOT / "supp" / "S1_response_duration_hist.eps"),
        s1_png=str(FIG_ROOT / "supp" / "S1_response_duration_hist.png"),
        s2_tif=str(FIG_ROOT / "supp" / "S2_previous_speech_duration_hist.tif"),
        s2_eps=str(FIG_ROOT / "supp" / "S2_previous_speech_duration_hist.eps"),
        s2_png=str(FIG_ROOT / "supp" / "S2_previous_speech_duration_hist.png"),
        s3_long_tif=str(FIG_ROOT / "supp" / "S3_long_joint.tif"),
        s3_long_eps=str(FIG_ROOT / "supp" / "S3_long_joint.eps"),
        s3_long_png=str(FIG_ROOT / "supp" / "S3_long_joint.png"),
        s3_short_tif=str(FIG_ROOT / "supp" / "S3_short_joint.tif"),
        s3_short_eps=str(FIG_ROOT / "supp" / "S3_short_joint.eps"),
        s3_short_png=str(FIG_ROOT / "supp" / "S3_short_joint.png"),
        s3_fast_tif=str(FIG_ROOT / "supp" / "S3_fast_joint.tif"),
        s3_fast_eps=str(FIG_ROOT / "supp" / "S3_fast_joint.eps"),
        s3_fast_png=str(FIG_ROOT / "supp" / "S3_fast_joint.png"),
        s3_slow_tif=str(FIG_ROOT / "supp" / "S3_slow_joint.tif"),
        s3_slow_eps=str(FIG_ROOT / "supp" / "S3_slow_joint.eps"),
        s3_slow_png=str(FIG_ROOT / "supp" / "S3_slow_joint.png"),
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
        tif=str(FIG_ROOT / "main" / "F_decoding.tif"),
        eps=str(FIG_ROOT / "main" / "F_decoding.eps"),
        png=str(FIG_ROOT / "main" / "F_decoding.png"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" decoding --mode figure
        """


rule fig_lrt_comparisons:
    """
    LRT comparisons table figure (duration + latency).
    """
    input:
        lrt_csv=str(out_dir() / "mixed_effect" / "lmm" / "tables" / "lrt_comparisons.csv"),
    output:
        duration_tif=str(FIG_ROOT / "main" / "F_lrt_comparisons_duration.tif"),
        duration_eps=str(FIG_ROOT / "main" / "F_lrt_comparisons_duration.eps"),
        duration_png=str(FIG_ROOT / "main" / "F_lrt_comparisons_duration.png"),
        latency_tif=str(FIG_ROOT / "main" / "F_lrt_comparisons_latency.tif"),
        latency_eps=str(FIG_ROOT / "main" / "F_lrt_comparisons_latency.eps"),
        latency_png=str(FIG_ROOT / "main" / "F_lrt_comparisons_latency.png"),
    threads: 1
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python scripts/make_lrt_table_figures.py \
          --lrt-csv "{input.lrt_csv}" \
          --out-dir "{FIG_ROOT}/main" \
          --out-stem "F_lrt_comparisons" \
          --profile jneuro_2col
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
        eps=str(FIG_ROOT / "main" / "F_erp_topomap.eps"),
        png=str(FIG_ROOT / "main" / "F_erp_topomap.png"),
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
          --out-png "{output.png}" \
          --out-eps "{output.eps}" \
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
        eps=str(FIG_ROOT / "main" / "F_tfr_topomap.eps"),
        png=str(FIG_ROOT / "main" / "F_tfr_topomap.png"),
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
          --out-png "{output.png}" \
          --out-eps "{output.eps}" \
          --dpi 300
        """


rule fig_erp_latency_with_hist:
    input:
        config=CONFIGFILE,
        mixed_table=str(out_dir() / "mixed_effect" / "table.csv"),
        erp=ERP_OUT,
    output:
        tif=str(FIG_ROOT / "main" / "F_erp_timecourse_hist.tif"),
        eps=str(FIG_ROOT / "main" / "F_erp_timecourse_hist.eps"),
        png=str(FIG_ROOT / "main" / "F_erp_timecourse_hist.png"),
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz --config "{input.config}" erp --mode hist
        """
