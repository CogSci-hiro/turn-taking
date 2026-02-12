# workflow/rules/viz.smk
from pathlib import Path

conda:
    CONDA_PY_ENV


def heavy_threads() -> int:
    return int(config.get("execution", {}).get("threads_heavy", 10))


def heavy_mem_mb() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10_000))


FIG_MAIN = [
    str(out_dir() / "figures" / "main" / "fig1_behavior.png"),
    str(out_dir() / "figures" / "main" / "fig2_erp_timecourse.png"),
    str(out_dir() / "figures" / "main" / "fig3_erp_topomaps.png"),
    str(out_dir() / "figures" / "main" / "fig4_tfr_topomaps.png"),
    str(out_dir() / "figures" / "main" / "fig5_decoding.png"),
]

FIG_SUPP = [
    str(out_dir() / "figures" / "supp" / "figS1_response_duration_hist.png"),
    str(out_dir() / "figures" / "supp" / "figS2_previous_speech_duration_hist.png"),
    str(out_dir() / "figures" / "supp" / "figS3_long_joint.png"),
    str(out_dir() / "figures" / "supp" / "figS3_short_joint.png"),
    str(out_dir() / "figures" / "supp" / "figS3_fast_joint.png"),
    str(out_dir() / "figures" / "supp" / "figS3_slow_joint.png"),
    str(out_dir() / "figures" / "supp" / "figS4_erp_timecourse_with_hist.png"),
]


# These should match whatever your analysis rules produce (Option B contracts)
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


FIG_ROOT = config["io"]["out_dir"] + "/figures"


rule figures_main:
    """
    Build all main manuscript figures (Fig 1–5).
    """
    input:
        epochs=epoch_inputs(),
        erp=ERP_OUT,
        tfr=TFR_OUT,
        decoding=DECODING_OUT,
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        FIG_MAIN
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" viz main --config "{input.config}"
        """


rule figures_supp:
    """
    Build all supplementary figures (S1–S4).
    """
    input:
        epochs=epoch_inputs(),
        erp=ERP_OUT,
        tfr=TFR_OUT,
        decoding=DECODING_OUT,
        config=str(Path(workflow.basedir) / "config.yaml"),
    output:
        FIG_SUPP
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    params:
        entrypoint=str(entrypoint())
    shell:
        r"""
        set -euo pipefail
        python "{params.entrypoint}" viz supp --config "{input.config}"
        """


rule fig_erp_timecourse:
    input:
        config="workflow/config.yaml"
    output:
        fig=FIG_ROOT + "/main" + "/F_erp_timecourse.tif"
    shell:
        r"""
        python -m turntaking.cli.main viz-erp-timecourse --config "{input.config}"
        """


rule fig_behavior:
    input:
        config="workflow/config.yaml",
        table=config["paths"]["out_dir"] + "/beh/turn_table.csv"
    output:
        main=FIG_ROOT + "/main" + "/F_behavior.tif",
        s1=FIG_ROOT + "/supp" + "/S1_response_duration_hist.tif",
        s2=FIG_ROOT + "/supp" + "/S2_previous_speech_duration_hist.tif",
        s3_long=FIG_ROOT + "/supp" + "/S3_long_joint.tif",
        s3_short=FIG_ROOT + "/supp" + "/S3_short_joint.tif",
        s3_fast=FIG_ROOT + "/supp" + "/S3_fast_joint.tif",
        s3_slow=FIG_ROOT + "/supp" + "/S3_slow_joint.tif",
    shell:
        r"""
        python -m turntaking.cli.main viz-behavior --config "{input.config}"
        """