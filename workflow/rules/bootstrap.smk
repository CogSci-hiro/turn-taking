WORKFLOW_DIR = Path(workflow.basedir)
REPO_ROOT = WORKFLOW_DIR.parent
CONDA_PY_ENV = str(WORKFLOW_DIR / "envs" / "python.yaml")

rule install_turntaking:
    conda: CONDA_PY_ENV
    output:
        stamp = str(WORKFLOW_DIR / ".turntaking_installed.stamp")
    shell:
        r"""
        set -euo pipefail
        python -m pip install -e "{REPO_ROOT}"
        python -c "import turntaking; print('TURNTAKING_OK', turntaking.__file__)"
        touch "{output.stamp}"
        """

rule analyze_erp:
    conda: CONDA_PY_ENV
    input:
        config=CONFIG_PATH,
        stamp=str(WORKFLOW_DIR / ".turntaking_installed.stamp"),
    shell:
        r"""
        set -euo pipefail
        python -m turntaking.cli.main analyze erp --config "{input.config}"
        """
