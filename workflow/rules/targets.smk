# =============================================================================
#                               Targets
# =============================================================================

conda:
    CONDA_PY_ENV


rule all:
    input:
        [
            sentinel_path("erp"),
            sentinel_path("tfr"),
            sentinel_path("mixed"),
            sentinel_path("decoding"),
        ]
