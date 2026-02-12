import glob

BEH_DIR = config["paths"]["beh_dir"]
OUT_DIR = config["paths"]["out_dir"]

BEH_ROOT = OUT_DIR + "/beh"
BEH_TSVS = sorted(glob.glob(BEH_DIR + "/*_metadata.tsv"))

rule beh_turn_table:
    input:
        config="workflow/config.yaml",
        tsvs=BEH_TSVS
    output:
        csv=BEH_ROOT + "/turn_table.csv"
    threads:
        light_threads()
    resources:
        mem_mb=light_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main beh-turn-table --config "{input.config}"
        """
