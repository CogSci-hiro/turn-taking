"""Contracts for Snakemake rule hygiene and output-scope safety."""

from pathlib import Path

import pytest

@pytest.mark.contract
def test_snakemake_rules_do_not_hardcode_config_path() -> None:
    rules_dir = Path("workflow/rules")
    for rule_file in sorted(rules_dir.glob("*.smk")):
        text = rule_file.read_text(encoding="utf-8")
        assert "workflow/config.yaml" not in text, f"Hardcoded config path in {rule_file}"


@pytest.mark.contract
def test_snakemake_rules_avoid_string_concat_out_dir_paths() -> None:
    rules_dir = Path("workflow/rules")
    bad_patterns = ['config["paths"]["out_dir"] +', "config['paths']['out_dir'] +", 'config["io"]["out_dir"] +']
    for rule_file in sorted(rules_dir.glob("*.smk")):
        text = rule_file.read_text(encoding="utf-8")
        for pattern in bad_patterns:
            assert pattern not in text, f"Legacy out_dir string concat in {rule_file}: {pattern}"
