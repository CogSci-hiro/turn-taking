# =============================================================================
#                             ANALYSIS (DATA)
# =============================================================================

LIGHT_MEM_MB = 1_000
HEAVY_MEM_MB = 10_000

# Epochs exist already; use helper from main Snakefile
# - epoch_inputs()
# - CONTRASTS, BANDS
# - SUBJECTS
# - SCRIPT_DIR, OUT_DIR


rule make_erp_data:
    input:
        epoch_inputs()
    output:
        evoked=expand(str(OUT_DIR / "stats" / "erp" / "{contrast}" / "evoked-data.npy"), contrast=CONTRASTS),
        diff=expand(str(OUT_DIR / "stats" / "erp" / "{contrast}" / "difference_ave.fif"), contrast=CONTRASTS),
    params:
        contrast=config["erp"]["contrast"],
        min_latency=config["min_latency"],
        max_latency=config["max_latency"],
        min_response_duration=config["min_response_duration"],
    threads: 1
    resources:
        mem_mb=LIGHT_MEM_MB
    script:
        SCRIPT_DIR / "analysis" / "erp_data.py"


rule make_tfr_data:
    input:
        epoch_inputs()
    output:
        evoked=expand(
            str(OUT_DIR / "stats" / "tfr" / "{contrast}" / "{band}" / "evoked-data.npy"),
            contrast=CONTRASTS,
            band=BANDS,
        ),
        diff=expand(
            str(OUT_DIR / "stats" / "tfr" / "{contrast}" / "{band}" / "difference_ave.fif"),
            contrast=CONTRASTS,
            band=BANDS,
        ),
    params:
        contrast=config["tfr"]["contrast"],
        method=config["tfr"]["method"],
        min_latency=config["min_latency"],
        max_latency=config["max_latency"],
        min_response_duration=config["min_response_duration"],
    threads: 10
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "analysis" / "tfr_data.py"


rule make_mixed_effect_data:
    input:
        epoch_inputs()
    output:
        summary=str(OUT_DIR / "stats" / "summary-data.csv")
    params:
        tw1_tmin=config["mixed"]["tw1_tmin"],
        tw1_tmax=config["mixed"]["tw1_tmax"],
        tw2_tmin=config["mixed"]["tw2_tmin"],
        tw2_tmax=config["mixed"]["tw2_tmax"],
        baseline_tmin=config["mixed"]["baseline_tmin"],
        baseline_tmax=config["mixed"]["baseline_tmax"],
        min_latency=config["min_latency"],
        max_latency=config["max_latency"],
        min_response_duration=config["min_response_duration"],
    threads: 10
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "analysis" / "mixed_effect_data.py"


rule make_decoding_data:
    input:
        epoch_inputs()
    output:
        targets=expand(
            str(OUT_DIR / "stats" / "decoding" / "{mode}" / "{contrast}" / "{subject}-{kind}.npy"),
            mode=["erp"],
            contrast=CONTRASTS,
            subject=SUBJECTS,
            kind=["X", "y"],
        )
    params:
        min_latency=config["min_latency"],
        max_latency=config["max_latency"],
        min_response_duration=config["min_response_duration"],
        sfreq=config["decoding"]["sfreq"],
    threads: 10
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "analysis" / "decoding_data.py"


rule make_decoding_score:
    input:
        targets=expand(
            str(OUT_DIR / "stats" / "decoding" / "{mode}" / "{contrast}" / "{subject}-{kind}.npy"),
            mode=["erp"],
            contrast=CONTRASTS,
            subject=SUBJECTS,
            kind=["X", "y"],
        )
    output:
        scores=expand(
            str(OUT_DIR / "stats" / "decoding_scores" / "{mode}-{contrast}.npy"),
            mode=["erp"],
            contrast=CONTRASTS,
        )
    params:
        n_splits=config["decoding"]["n_splits"]
    threads: 10
    resources:
        mem_mb=HEAVY_MEM_MB
    script:
        SCRIPT_DIR / "analysis" / "decoding_score.py"
