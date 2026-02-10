
# =============================================================================
#                                  TARGETS
# =============================================================================
# "Entry rules" for the CLI. These rules do not run scripts; they just declare
# the final artifacts that define completion of each pipeline stage.
#
# CLI will call e.g.: snakemake erp_all
# =============================================================================

rule erp_all:
    input:
        # pick your definition of "done": plots, or results, or both
        expand(str(OUT_DIR / "plots" / "erp" / "{contrast}" / "time_courses.png"), contrast=CONTRASTS)

rule tfr_all:
    input:
        expand(
            str(OUT_DIR / "plots" / "tfr" / "{contrast}" / "{band}" / "topomap-alpha-0.1.png"),
            contrast=CONTRASTS,
            band=BANDS,
        )

rule mixed_all:
    input:
        str(OUT_DIR / "stats" / "summary-data.csv")

rule decoding_all:
    input:
        expand(
            str(OUT_DIR / "stats" / "decoding_results" / "{mode}-{contrast}-results.hdf5"),
            mode=["erp"],
            contrast=CONTRASTS,
        )

rule all_all:
    input:
        rules.erp_all.input,
        rules.tfr_all.input,
        rules.mixed_all.input,
        rules.decoding_all.input
