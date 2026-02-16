import glob
from pathlib import Path


def active_configfile() -> str:
    cfgs = list(getattr(workflow, "overwrite_configfiles", []))
    if not cfgs:
        cfgs = list(getattr(workflow, "configfiles", []))
    if not cfgs:
        raise ValueError("No configfile is available in Snakemake workflow context.")
    return str(cfgs[0])


CONFIGFILE = active_configfile()

BEH_DIR = Path(_io_or_paths()["beh_dir"])
BEH_ROOT = out_dir() / "beh"
BEH_TSVS = sorted(glob.glob(str(BEH_DIR / "*_metadata.tsv")))


rule beh_turn_table:
    input:
        config=CONFIGFILE,
        tsvs=BEH_TSVS
    output:
        csv=str(BEH_ROOT / "turn_table.csv")
    threads:
        light_threads()
    resources:
        mem_mb=light_mem_mb()
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main beh-turn-table --config "{input.config}"
        """
