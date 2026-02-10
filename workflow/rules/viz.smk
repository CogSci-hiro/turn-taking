
# =============================================================================
#                               VISUALIZATION
# =============================================================================

HEAVY_MEM_MB = 10_000


rule plot_erps:
    input:
        results=expand(str(OUT_DIR / "stats" / "erp" / "{contrast}" / "results.hdf5"), contrast=CONTRASTS)
    output:
        plots=expand(str(OUT_DIR / "plots" / "erp" / "{contrast}" / "time_courses.png"), contrast=CONTRASTS)
    threads: 1
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "visualization" / "plot_erp.py"


rule plot_tfrs:
    input:
        results=expand(
            str(OUT_DIR / "stats" / "tfr" / "{contrast}" / "{band}" / "results.hdf5"),
            contrast=CONTRASTS,
            band=BANDS,
        )
    output:
        plots=expand(
            str(OUT_DIR / "plots" / "tfr" / "{contrast}" / "{band}" / "topomap-alpha-0.1.png"),
            contrast=CONTRASTS,
            band=BANDS,
        )
    threads: 1
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "visualization" / "plot_tfr.py"
