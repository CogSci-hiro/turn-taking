
from pathlib import Path

import pytest
import yaml

from turntaking.config.loader import load_config


def _valid_config() -> dict:
    return {
        "io": {
            "epoch_dir": "/tmp/epochs",
            "epoch_pattern": "{subject}_task-diapix_run-{run}_epo.fif",
            "out_dir": "/tmp/out",
        },
        "dataset": {
            "subjects": {"mode": "explicit", "exclude": [], "include": ["sub-006"]},
            "tasks": ["diapix"],
            "runs": [1],
            "invalid_subject_run": [],
        },
        "constraints": {
            "min_latency": -1.0,
            "max_latency": 1.0,
            "min_response_duration": 0.01,
        },
        "analysis": {
            "contrasts": ["duration", "latency"],
            "bands": ["alpha", "beta"],
            "erp": {"left_margin": 0.0, "right_margin": 0.0, "sfreq": 64, "n_permutations": 10, "threshold": None},
            "tfr": {"left_margin": 0.0, "right_margin": 0.0, "method": "hilbert", "sfreq": 64, "n_permutations": 10, "threshold": None},
            "mixed": {
                "tw1": [-0.2, -0.1],
                "tw2": [-0.1, 0.0],
                "baseline": [-0.3, -0.2],
                "selection": {"min_latency": -1.0, "max_latency": 1.0, "min_self_duration": 0.01},
            },
            "decoding": {"sfreq": 64, "n_splits": 2},
        },
        "execution": {"threads_light": 1, "threads_heavy": 1, "mem_mb_light": 256, "mem_mb_heavy": 512},
        "viz": {
            "erp_timecourse": {
                "duration_long_fif": "/tmp/a.fif",
                "duration_short_fif": "/tmp/b.fif",
                "latency_fast_fif": "/tmp/c.fif",
                "latency_slow_fif": "/tmp/d.fif",
                "out_base": "/tmp/erp_timecourse",
                "xlim_ms": [-500, 500],
                "ylim_uv": [-5, 5],
            },
            "behavior": {
                "duration_offsets_csv": "/tmp/duration_offsets.csv",
                "latency_offsets_csv": "/tmp/latency_offsets.csv",
                "turn_table_csv": "/tmp/turn_table.csv",
                "out_base": "/tmp/behavior",
                "n_bins": 10,
            },
            "erp_topo": {
                "duration_cluster_hdf5": "/tmp/duration_cluster.h5",
                "latency_cluster_hdf5": "/tmp/latency_cluster.h5",
                "info_source_fif": "/tmp/info.fif",
                "out_duration": "/tmp/out_duration.tif",
                "out_latency": "/tmp/out_latency.tif",
                "tmin_s": -0.2,
                "tmax_s": 0.1,
                "step_ms": 50,
            },
            "erp_topomaps": {
                "template_svg": "workflow/templates/ERP-timeline.svg",
                "parts_dir": "workflow/results/parts_erp_topomaps",
                "out_svg": "workflow/results/F_erp_topomaps.svg",
                "info_source_fif": "/tmp/info.fif",
                "duration_cluster_hdf5": "/tmp/duration_cluster.h5",
                "latency_cluster_hdf5": "/tmp/latency_cluster.h5",
            },
            "tfr_topomaps": {
                "template_svg": "workflow/templates/TFR-timeline.svg",
                "parts_dir": "workflow/results/parts_tfr_topomaps",
                "out_svg": "workflow/results/F_tfr_topomaps.svg",
                "info_source_fif": "/tmp/info.fif",
                "alpha_cluster_hdf5": "/tmp/alpha_cluster.h5",
                "beta_cluster_hdf5": "/tmp/beta_cluster.h5",
            },
            "tfr_topos": {
                "alpha_duration_cluster_hdf5": "/tmp/a_dur.h5",
                "alpha_latency_cluster_hdf5": "/tmp/a_lat.h5",
                "beta_duration_cluster_hdf5": "/tmp/b_dur.h5",
                "beta_latency_cluster_hdf5": "/tmp/b_lat.h5",
                "info_source_fif": "/tmp/info.fif",
                "out_alpha_duration": "/tmp/out_a_dur.tif",
                "out_alpha_latency": "/tmp/out_a_lat.tif",
                "out_beta_duration": "/tmp/out_b_dur.tif",
                "out_beta_latency": "/tmp/out_b_lat.tif",
                "tmin_s": -0.2,
                "tmax_s": 0.1,
                "step_ms": 50,
            },
            "erp_hist": {
                "duration_long_fif": "/tmp/a.fif",
                "duration_short_fif": "/tmp/b.fif",
                "latency_fast_fif": "/tmp/c.fif",
                "latency_slow_fif": "/tmp/d.fif",
                "hist_table_csv": "/tmp/hist.csv",
                "out_base": "/tmp/hist",
                "xlim_ms": [-500, 500],
                "ylim_uv": [-5, 5],
            },
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_valid_config_loads_to_dataclass(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    _write(cfg_path, _valid_config())
    cfg = load_config(cfg_path)
    assert cfg.io.epoch_pattern == "{subject}_task-diapix_run-{run}_epo.fif"
    assert cfg.analysis.contrasts == ["duration", "latency"]
    assert cfg.dataset.subjects.include == ["sub-006"]


def test_missing_required_field_raises(tmp_path):
    payload = _valid_config()
    del payload["io"]["out_dir"]
    cfg_path = tmp_path / "missing.yaml"
    _write(cfg_path, payload)
    with pytest.raises(KeyError, match="Missing required key 'out_dir'"):
        load_config(cfg_path)


def test_invalid_contrast_value_raises(tmp_path):
    payload = _valid_config()
    payload["analysis"]["contrasts"] = ["duration", "bad_contrast"]
    cfg_path = tmp_path / "bad_contrast.yaml"
    _write(cfg_path, payload)
    with pytest.raises(ValueError, match="unsupported value"):
        load_config(cfg_path)


def test_invalid_epoch_dir_type_raises(tmp_path):
    payload = _valid_config()
    payload["io"]["epoch_dir"] = {"not": "a-path"}
    cfg_path = tmp_path / "bad_epoch_dir.yaml"
    _write(cfg_path, payload)
    with pytest.raises(TypeError):
        load_config(cfg_path)
