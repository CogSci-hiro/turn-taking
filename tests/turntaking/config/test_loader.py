
"""Tests for strict YAML-to-dataclass config loading."""

from pathlib import Path

import pytest
import yaml

from turntaking.config.loader import load_config


def _valid_config_dict() -> dict:
    return {
        "io": {
            "epoch_dir": "/data/epochs",
            "epoch_pattern": "{subject}_task-{task}_run-{run}_epochs-epo.fif",
            "out_dir": "/data/out",
        },
        "dataset": {
            "subjects": {"mode": "from_epochs", "exclude": ["sub-001"], "include": []},
            "tasks": ["conversation"],
            "runs": [1, 2],
            "invalid_subject_run": [["sub-004", "1"]],
        },
        "constraints": {
            "min_latency": -1.0,
            "max_latency": 1.0,
            "min_response_duration": 0.01,
        },
        "analysis": {
            "contrasts": ["duration", "latency"],
            "bands": ["alpha"],
            "erp": {
                "left_margin": 0.2,
                "right_margin": 0.5,
                "baseline": [-0.2, 0.0],
                "sfreq": 512,
                "n_permutations": 10,
                "threshold": None,
            },
            "tfr": {
                "left_margin": 0.2,
                "right_margin": 0.5,
                "method": "hilbert",
                "sfreq": 128,
                "n_permutations": 10,
                "threshold": None,
            },
            "mixed": {
                "tw1": [-1.3, -0.8],
                "tw2": [-0.7, 0.0],
                "baseline": [-2.0, 0.0],
                "selection": {
                    "min_latency": 0.05,
                    "max_latency": 2.0,
                    "min_self_duration": 0.10,
                },
            },
            "decoding": {"sfreq": 64, "n_splits": 10},
        },
        "execution": {
            "threads_light": 1,
            "threads_heavy": 4,
            "mem_mb_light": 512,
            "mem_mb_heavy": 4096,
        },
        "viz": {
            "erp_timecourse": {
                "xlim_ms": [-1500, 500],
                "ylim_uv": [-2.8, 1.9],
            },
            "behavior": {
                "n_bins": 100,
            },
            "erp_topo": {
                "tmin_s": -2.0,
                "tmax_s": 0.0,
                "step_ms": 100,
            },
            "erp_topomaps": {},
            "tfr_topomaps": {},
            "tfr_topos": {
                "tmin_s": -2.0,
                "tmax_s": 0.0,
                "step_ms": 100,
            },
            "decoding": {
                "figure_profile": "jneuro_2col",
                "p_threshold": 0.05,
                "ymax": 0.65,
                "lim": 0.04,
            },
            "erp_hist": {
                "xlim_ms": [-1500, 500],
                "ylim_uv": [-2.8, 1.9],
            },
        },
    }


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_config_parses_types_and_nested_sections(tmp_path):
    """Validates full schema parsing so CLI/analysis layers can rely on typed config objects."""
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(cfg_path, _valid_config_dict())
    cfg = load_config(cfg_path)

    assert cfg.dataset.subjects.mode == "from_epochs"
    assert cfg.dataset.runs == [1, 2]
    assert cfg.dataset.invalid_subject_run == [("sub-004", "1")]
    assert cfg.analysis.decoding.sfreq == 64
    assert cfg.analysis.erp.baseline == [-0.2, 0.0]
    assert cfg.viz.erp_topomaps.n_latency_maps == 3
    assert cfg.viz.erp_timecourse.duration_long_fif == Path("/data/out/erp/duration/long_ave.fif")
    assert cfg.viz.decoding.out_base == Path("/data/out/figures/main/F_decoding")
    assert cfg.io.epoch_dir == Path("/data/epochs")


def test_load_config_rejects_non_mapping_yaml_root(tmp_path):
    """Defends against malformed YAML roots that cannot be interpreted as configuration mappings."""
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config must be a YAML mapping"):
        load_config(cfg_path)


def test_load_config_rejects_invalid_subject_mode(tmp_path):
    """Ensures subject-mode validation prevents unsupported selection behavior."""
    payload = _valid_config_dict()
    payload["dataset"]["subjects"]["mode"] = "invalid"
    cfg_path = tmp_path / "bad_mode.yaml"
    _write_yaml(cfg_path, payload)

    with pytest.raises(ValueError, match="dataset.subjects.mode"):
        load_config(cfg_path)


def test_load_config_rejects_missing_required_key(tmp_path):
    """Checks strict required-key enforcement to fail fast on incomplete configs."""
    payload = _valid_config_dict()
    del payload["analysis"]["erp"]["sfreq"]
    cfg_path = tmp_path / "missing_key.yaml"
    _write_yaml(cfg_path, payload)

    with pytest.raises(KeyError, match="Missing required key 'sfreq'"):
        load_config(cfg_path)


def test_load_config_rejects_missing_analysis_erp_baseline(tmp_path):
    """Requires explicit ERP baseline window to avoid implicit defaults."""
    payload = _valid_config_dict()
    del payload["analysis"]["erp"]["baseline"]
    cfg_path = tmp_path / "missing_erp_baseline.yaml"
    _write_yaml(cfg_path, payload)

    with pytest.raises(KeyError, match="Missing required key 'baseline'"):
        load_config(cfg_path)


def test_load_config_accepts_new_viz_topomaps_keys(tmp_path):
    """Supports canonical viz.topomaps naming while keeping legacy typed accessors."""
    payload = _valid_config_dict()
    payload["viz"]["topomaps"] = {
        "erp": {
            "static": payload["viz"]["erp_topo"],
            "svg": payload["viz"]["erp_topomaps"],
        },
        "tfr": {
            "static": payload["viz"]["tfr_topos"],
            "svg": payload["viz"]["tfr_topomaps"],
        },
    }
    del payload["viz"]["erp_topo"]
    del payload["viz"]["erp_topomaps"]
    del payload["viz"]["tfr_topos"]
    del payload["viz"]["tfr_topomaps"]

    cfg_path = tmp_path / "topomaps_config.yaml"
    _write_yaml(cfg_path, payload)

    cfg = load_config(cfg_path)
    assert str(cfg.viz.erp_topo.duration_cluster_hdf5).endswith("stats/erp/duration/cluster_results.hdf5")
    assert str(cfg.viz.erp_topomaps.template_svg).endswith("ERP-timeline.svg")


def test_load_config_normalizes_automatic_cluster_threshold_dict(tmp_path):
    """Supports explicit threshold mappings while preserving automatic-threshold behavior."""
    payload = _valid_config_dict()
    payload["analysis"]["erp"]["threshold"] = {"type": "automatic"}
    payload["analysis"]["tfr"]["threshold"] = {"value": None}

    cfg_path = tmp_path / "threshold_auto.yaml"
    _write_yaml(cfg_path, payload)

    cfg = load_config(cfg_path)
    assert cfg.analysis.erp.threshold is None
    assert cfg.analysis.tfr.threshold is None


def test_load_config_viz_artifact_paths_are_inferred_from_out_dir(tmp_path):
    """Ignores explicit viz artifact path overrides and derives paths from io.out_dir."""
    payload = _valid_config_dict()
    payload["viz"]["base_out_dir"] = "/override/out"
    payload["viz"]["erp_timecourse"]["duration_long_fif"] = "/override/out/custom/long_ave.fif"
    payload["viz"]["decoding"]["out_base"] = "/override/out/custom/F_decoding"

    cfg_path = tmp_path / "viz_paths_inferred.yaml"
    _write_yaml(cfg_path, payload)

    cfg = load_config(cfg_path)
    assert cfg.viz.erp_timecourse.duration_long_fif == Path("/data/out/erp/duration/long_ave.fif")
    assert cfg.viz.decoding.out_base == Path("/data/out/figures/main/F_decoding")
