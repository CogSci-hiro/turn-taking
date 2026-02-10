# =============================================================================
#                          STATISTICAL TESTS (CLUSTER)
# =============================================================================

HEAVY_MEM_MB = 10_000


rule test_erp:
    input:
        evoked=expand(str(OUT_DIR / "stats" / "erp" / "{contrast}" / "evoked-data.npy"), contrast=CONTRASTS),
        diff=expand(str(OUT_DIR / "stats" / "erp" / "{contrast}" / "difference_ave.fif"), contrast=CONTRASTS),
    output:
        results=expand(str(OUT_DIR / "stats" / "erp" / "{contrast}" / "results.hdf5"), contrast=CONTRASTS)
    params:
        right_margin=config["erp"]["right_margin"],
        left_margin=config["erp"]["left_margin"],
        sfreq=config["sfreq"],
        threshold=config["erp"]["threshold"],
        n_permutations=config["erp"]["n_permutations"],
    threads: 10
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "stats" / "erp_test.py"


rule test_tfr:
    input:
        evoked=expand(
            str(OUT_DIR / "stats" / "tfr" / "{contrast}" / "{band}" / "evoked-data.npy"),
            contrast=CONTRASTS,
            band=BANDS,
        ),
        diff=expand(
            str(OUT_DIR / "stats" / "tfr" / "{contrast}" / "{band}" / "difference_ave.fif"),
            contrast=CONTRASTS,
            band=BANDS,
        )
    output:
        results=expand(
            str(OUT_DIR / "stats" / "tfr" / "{contrast}" / "{band}" / "results.hdf5"),
            contrast=CONTRASTS,
            band=BANDS,
        )
    params:
        right_margin=config["tfr"]["right_margin"],
        left_margin=config["tfr"]["left_margin"],
        sfreq=config["sfreq"],
        threshold=config["tfr"]["threshold"],
        n_permutations=config["tfr"]["n_permutations"],
    threads: 10
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "stats" / "tfr_test.py"


rule test_decoding:
    input:
        scores=expand(
            str(OUT_DIR / "stats" / "decoding_scores" / "{mode}-{contrast}.npy"),
            mode=["erp"],
            contrast=CONTRASTS,
        )
    output:
        results=expand(
            str(OUT_DIR / "stats" / "decoding_results" / "{mode}-{contrast}-results.hdf5"),
            mode=["erp"],
            contrast=CONTRASTS,
        )
    params:
        right_margin=config["decoding"]["right_margin"],
        left_margin=config["decoding"]["left_margin"],
        sfreq=config["decoding"]["sfreq"],
        threshold=config["decoding"]["threshold"],
        n_permutations=config["decoding"]["n_permutations"],
    threads: 10
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "stats" / "test_decoding.py"
