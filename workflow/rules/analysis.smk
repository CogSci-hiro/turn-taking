# =============================================================================
#                               Analysis rules
# =============================================================================
#
# Each rule delegates work to the single Python dispatcher:
#   python src/turntaking/cli/main.py analyze <analysis> --config workflow/config.yaml
#
# Completion is tracked using sentinel files:
#   io.out_dir/<analysis>/_DONE
#

conda:
    CONDA_PY_ENV


ERP_SENTINEL = sentinel_path("erp")
TFR_SENTINEL = sentinel_path("tfr")
MIXED_SENTINEL = sentinel_path("mixed")
DECODING_SENTINEL = sentinel_path("decoding")


def _heavy_threads() -> int:
    return int(config.get("execution", {}).get("threads_heavy", 10))


def _light_threads() -> int:
    return int(config.get("execution", {}).get("threads_light", 1))


def _heavy_mem() -> int:
    return int(config.get("execution", {}).get("mem_mb_heavy", 10000))


def _light_mem() -> int:
    return int(config.get("execution", {}).get("mem_mb_light", 1000))


rule erp_all:
    input:
        epoch_inputs()
    output:
        ERP_SENTINEL
    threads:
        _heavy_threads()
    resources:
        mem_mb=_heavy_mem()
    shell:
        r"""
        set -euo pipefail
        python "{entrypoint()}" analyze erp --config "{workflow.basedir}/config.yaml"
        mkdir -p "{out_dir()}/erp"
        touch "{output}"
        """


rule tfr_all:
    input:
        epoch_inputs()
    output:
        TFR_SENTINEL
    threads:
        _heavy_threads()
    resources:
        mem_mb=_heavy_mem()
    shell:
        r"""
        set -euo pipefail
        python "{entrypoint()}" analyze tfr --config "{workflow.basedir}/config.yaml"
        mkdir -p "{out_dir()}/tfr"
        touch "{output}"
        """


rule mixed_all:
    input:
        epoch_inputs()
    output:
        MIXED_SENTINEL
    threads:
        _light_threads()
    resources:
        mem_mb=_light_mem()
    shell:
        r"""
        set -euo pipefail
        python "{entrypoint()}" analyze mixed --config "{workflow.basedir}/config.yaml"
        mkdir -p "{out_dir()}/mixed"
        touch "{output}"
        """


rule decoding_all:
    input:
        epoch_inputs()
    output:
        DECODING_SENTINEL
    threads:
        _heavy_threads()
    resources:
        mem_mb=_heavy_mem()
    shell:
        r"""
        set -euo pipefail
        python "{entrypoint()}" analyze decoding --config "{workflow.basedir}/config.yaml"
        mkdir -p "{out_dir()}/decoding"
        touch "{output}"
        """


# Convenience: remove sentinels for clean reruns
rule clean_analysis:
    message:
        "Removing analysis sentinel files (forces re-run)."
    shell:
        r"""
        rm -f "{ERP_SENTINEL}" "{TFR_SENTINEL}" "{MIXED_SENTINEL}" "{DECODING_SENTINEL}"
        """
