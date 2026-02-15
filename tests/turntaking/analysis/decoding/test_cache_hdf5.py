from __future__ import annotations

"""Tests for decoding feature-cache config parsing and IO closure wiring."""

import numpy as np

from turntaking.analysis.decoding.cache_hdf5 import DecodingCacheConfig, _get_optional, make_hdf5_cache_io


def test_get_optional_returns_nested_value_or_default():
    """Checks dotted-key lookup behavior for optional nested cache config fields."""
    cfg = {"a": {"b": {"c": 10}}}
    assert _get_optional(cfg, "a.b.c", 0) == 10
    assert _get_optional(cfg, "a.b.missing", 99) == 99
    assert _get_optional(cfg, "a.missing.c", "x") == "x"


def test_decoding_cache_config_from_config_dict_defaults_and_values():
    """Ensures cache config parsing yields stable defaults and typed override values."""
    empty = {}
    c1 = DecodingCacheConfig.from_config_dict(empty)
    assert c1.compression == "gzip"
    assert c1.compression_level == 4
    assert c1.dtype == "float32"

    cfg = {"analysis": {"decoding": {"cache": {"compression": None, "compression_level": 7, "dtype": "float16"}}}}
    c2 = DecodingCacheConfig.from_config_dict(cfg)
    assert c2.compression is None
    assert c2.compression_level == 7
    assert c2.dtype == "float16"


def test_make_hdf5_cache_io_wires_load_and_save_functions(monkeypatch, tmp_path):
    """Verifies closure-based cache API calls lower-level load/save with expected argument mapping."""
    captured = {"load": [], "save": []}

    def fake_load_subject_feature_cache_hdf5(*, out_dir, contrast, subject):
        captured["load"].append((out_dir, contrast, subject))
        return np.array([1]), np.array([0]), np.array([0.1])

    def fake_save_subject_feature_cache_hdf5(*, out_dir, contrast, subject, X, y, times_s, cache_params):
        captured["save"].append((out_dir, contrast, subject, cache_params.x_dtype, cache_params.compression))
        assert X.shape == (1, 2, 3)
        assert y.shape == (1,)
        assert times_s.shape == (3,)

    monkeypatch.setattr(
        "turntaking.analysis.decoding.cache_hdf5.load_subject_feature_cache_hdf5",
        fake_load_subject_feature_cache_hdf5,
    )
    monkeypatch.setattr(
        "turntaking.analysis.decoding.cache_hdf5.save_subject_feature_cache_hdf5",
        fake_save_subject_feature_cache_hdf5,
    )

    load_fn, save_fn = make_hdf5_cache_io(
        out_dir=tmp_path,
        contrast="duration",
        cache_cfg=DecodingCacheConfig(compression="gzip", compression_level=1, dtype="float32"),
    )

    X, y, t = load_fn("sub-001")
    np.testing.assert_array_equal(X, np.array([1]))
    np.testing.assert_array_equal(y, np.array([0]))
    np.testing.assert_array_equal(t, np.array([0.1]))

    save_fn("sub-001", np.zeros((1, 2, 3)), np.zeros((1,), dtype=int), np.array([0.0, 0.1, 0.2]))
    assert captured["load"] == [(tmp_path, "duration", "sub-001")]
    assert captured["save"] == [(tmp_path, "duration", "sub-001", "float32", "gzip")]
