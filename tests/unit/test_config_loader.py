
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
            "erp": {
                "left_margin": 0.0,
                "right_margin": 0.0,
                "baseline": [-0.2, 0.0],
                "sfreq": 64,
                "n_permutations": 10,
                "threshold": None,
            },
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
                "xlim_ms": [-500, 500],
                "ylim_uv": [-5, 5],
            },
            "behavior": {
                "n_bins": 10,
            },
            "erp_topo": {
                "tmin_s": -0.2,
                "tmax_s": 0.1,
                "step_ms": 50,
            },
            "erp_topomaps": {},
            "tfr_topomaps": {},
            "tfr_topos": {
                "tmin_s": -0.2,
                "tmax_s": 0.1,
                "step_ms": 50,
            },
            "decoding": {
                "figure_profile": "jneuro_2col",
                "p_threshold": 0.05,
                "ymax": 0.65,
                "lim": 0.04,
            },
            "erp_hist": {
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
    assert cfg.analysis.erp.baseline == [-0.2, 0.0]
    assert cfg.dataset.subjects.include == ["sub-006"]
    assert cfg.viz.erp_timecourse.duration_long_fif == Path("/tmp/out/erp/duration/long_ave.fif")


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


def test_viz_paths_are_derived_from_io_out_dir(tmp_path):
    payload = _valid_config()
    payload["viz"]["base_out_dir"] = "/tmp/override"
    payload["viz"]["erp_timecourse"]["duration_long_fif"] = "/tmp/override/custom.fif"
    cfg_path = tmp_path / "viz_paths.yaml"
    _write(cfg_path, payload)

    cfg = load_config(cfg_path)
    assert cfg.viz.erp_timecourse.duration_long_fif == Path("/tmp/out/erp/duration/long_ave.fif")
