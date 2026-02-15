from __future__ import annotations

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
                "duration_long_fif": "/out/a.fif",
                "duration_short_fif": "/out/b.fif",
                "latency_fast_fif": "/out/c.fif",
                "latency_slow_fif": "/out/d.fif",
                "out_base": "/out/fig.tif",
                "xlim_ms": [-1500, 500],
                "ylim_uv": [-2.8, 1.9],
            },
            "behavior": {
                "duration_offsets_csv": "/out/duration_offsets.csv",
                "latency_offsets_csv": "/out/latency_offsets.csv",
                "turn_table_csv": "/out/turn_table.csv",
                "out_base": "/out/behavior",
                "n_bins": 100,
            },
            "erp_topo": {
                "duration_cluster_hdf5": "/out/duration_cluster.h5",
                "latency_cluster_hdf5": "/out/latency_cluster.h5",
                "info_source_fif": "/out/info.fif",
                "out_duration": "/out/duration.tif",
                "out_latency": "/out/latency.tif",
                "tmin_s": -2.0,
                "tmax_s": 0.0,
                "step_ms": 100,
            },
            "erp_topomaps": {
                "template_svg": "workflow/templates/ERP-timeline.svg",
                "parts_dir": "workflow/results/parts_erp_topomaps",
                "out_svg": "workflow/results/F_erp_topomaps.svg",
                "info_source_fif": "/out/info.fif",
                "duration_cluster_hdf5": "/out/duration_cluster.h5",
                "latency_cluster_hdf5": "/out/latency_cluster.h5",
            },
            "tfr_topomaps": {
                "template_svg": "workflow/templates/TFR-timeline.svg",
                "parts_dir": "workflow/results/parts_tfr_topomaps",
                "out_svg": "workflow/results/F_tfr_topomaps.svg",
                "info_source_fif": "/out/info.fif",
                "alpha_cluster_hdf5": "/out/alpha_cluster.h5",
                "beta_cluster_hdf5": "/out/beta_cluster.h5",
            },
            "tfr_topos": {
                "alpha_duration_cluster_hdf5": "/out/alpha_duration_cluster.h5",
                "alpha_latency_cluster_hdf5": "/out/alpha_latency_cluster.h5",
                "beta_duration_cluster_hdf5": "/out/beta_duration_cluster.h5",
                "beta_latency_cluster_hdf5": "/out/beta_latency_cluster.h5",
                "info_source_fif": "/out/info.fif",
                "out_alpha_duration": "/out/alpha_duration.tif",
                "out_alpha_latency": "/out/alpha_latency.tif",
                "out_beta_duration": "/out/beta_duration.tif",
                "out_beta_latency": "/out/beta_latency.tif",
                "tmin_s": -2.0,
                "tmax_s": 0.0,
                "step_ms": 100,
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
    assert cfg.viz.erp_topomaps.n_latency_maps == 3
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
