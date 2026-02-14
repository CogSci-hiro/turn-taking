# workflow/rules/viz.smk
from pathlib import Path

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
    str(out_dir() / "figures" / "main" / "F_erp_topomap.tif"),
    str(out_dir() / "figures" / "main" / "F_tfr_topomap.tif"),
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
    # Decoding outputs (ERP)
    str(out_dir() / "decoding" / "erp" / "duration" / "scores.npy"),
    str(out_dir() / "decoding" / "erp" / "duration" / "times.npy"),
    str(out_dir() / "decoding" / "erp" / "latency" / "scores.npy"),
    str(out_dir() / "decoding" / "erp" / "latency" / "times.npy"),

    # Decoding cluster-test outputs (needed to draw significance)
    str(out_dir() / "stats" / "decoding" / "erp" / "duration" / "cluster_results.hdf5"),
    str(out_dir() / "stats" / "decoding" / "erp" / "latency" / "cluster_results.hdf5"),
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


rule fig_erp_topos:
    input:
        config="workflow/config.yaml"
    output:
        duration=FIG_ROOT + "/supp/F_erp_topo_duration.tif",
        latency=FIG_ROOT + "/supp/F_erp_topo_latency.tif",
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz-erp-topo --config "{input.config}"
        """


rule fig_tfr_topomaps:
    input:
        config="workflow/config.yaml"
    output:
        fig=FIG_ROOT + "/main" + "/F_tfr_topo.tif"
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz-tfr-topo --config "{input.config}"
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


rule fig_decoding:
    input:
        config="workflow/config.yaml",
        decoding=DECODING_OUT
    output:
        fig=FIG_ROOT + "/main" + "/F_decoding.tif"
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz-decoding \
          --config "{input.config}" \
          --out "{output.fig}"
        """


rule fig_erp_topomap_svg:
    input:
        template="workflow/templates/ERP-timeline.svg",
        config="workflow/config.yaml",
    output:
        svg=FIG_ROOT + "/main/F_erp_topomap.svg",
    params:
        parts_dir=FIG_ROOT + "/main/parts_erp_topomap"
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz-topomaps \
          --config "{input.config}" \
          --template "{input.template}" \
          --parts-dir "{params.parts_dir}" \
          --out-svg "{output.svg}"
        """


rule fig_erp_topomap_tif:
    input:
        svg=FIG_ROOT + "/main/F_erp_topomap.svg",
        config="workflow/config.yaml",
    output:
        tif=FIG_ROOT + "/main/F_erp_topomap.tif",
    shell:
        r"""
        set -euo pipefail

        # Make sure Python can locate libcairo on macOS (brew)
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
        config="workflow/config.yaml",
    output:
        svg=FIG_ROOT + "/main/F_tfr_topomap.svg",
    params:
        parts_dir=FIG_ROOT + "/main/parts_tfr_topomap"
    threads:
        heavy_threads()
    resources:
        mem_mb=heavy_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main viz-tfr-topomaps \
          --config "{input.config}" \
          --template "{input.template}" \
          --parts-dir "{params.parts_dir}" \
          --out-svg "{output.svg}"
        """


rule fig_tfr_topomap_tif:
    input:
        svg=FIG_ROOT + "/main/F_tfr_topomap.svg",
        config="workflow/config.yaml",
    output:
        tif=FIG_ROOT + "/main/F_tfr_topomap.tif",
    shell:
        r"""
        set -euo pipefail

        # Make sure Python can locate libcairo on macOS (brew)
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
