from __future__ import annotations

import json

import pytest
from oxocr.config import ModelOptions, OcrConfig, apply_overrides
from oxocr.errors import UnknownParameterError


def test_text_score_clamped():
    cfg = OcrConfig.from_dict({"infer": {"text_score": 5.0}})
    assert cfg.infer.text_score == 1.0
    cfg2 = OcrConfig.from_dict({"infer": {"text_score": -3}})
    assert cfg2.infer.text_score == 0.0


def test_num_threads_min_zero():
    cfg = OcrConfig.from_dict({"model_options": {"num_threads": -8}})
    assert cfg.model_options.num_threads == 0


def test_invalid_mode_falls_back_with_warning():
    with pytest.warns(UserWarning):
        cfg = OcrConfig.from_dict({"postprocess": {"mode": "nonsense"}})
    assert cfg.postprocess.mode == "para"


def test_unknown_top_key_ignored_with_warning():
    with pytest.warns(UserWarning):
        cfg = OcrConfig.from_dict({"totally_unknown": 1, "model": "ppocrv6_tiny"})
    assert cfg.model == "ppocrv6_tiny"


def test_apply_overrides_shortcuts():
    cfg = apply_overrides(
        OcrConfig(),
        {"use_cls": False, "dark_convert": False, "det_limit_side_len": 960, "text_score": 0.7},
    )
    assert cfg.model_options.use_cls is False
    assert cfg.preprocess.options["dark_to_light"]["enabled"] is False
    assert cfg.model_options.params["det_limit_side_len"] == 960
    assert cfg.infer.text_score == 0.7


def test_unknown_override_raises():
    with pytest.raises(UnknownParameterError):
        apply_overrides(OcrConfig(), {"not_a_real_flag": 1})


def test_merge_returns_new_object():
    base = OcrConfig(model="ppocrv6_small")
    merged = base.merge({"model": "ppocrv6_tiny"})
    assert base.model == "ppocrv6_small"
    assert merged.model == "ppocrv6_tiny"


def test_from_json_roundtrip(tmp_path):
    data = {"model": "ppocrv6_tiny", "postprocess": {"mode": "layout"}}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cfg = OcrConfig.from_json(path)
    assert cfg.postprocess.mode == "layout"
    assert "layout" in cfg.to_dict()["postprocess"]["mode"]


def test_env_default_model(monkeypatch):
    monkeypatch.setenv("OXOCR_DEFAULT_MODEL", "ppocrv6_medium")
    assert OcrConfig().model == "ppocrv6_medium"


def test_env_resident_zero(monkeypatch):
    monkeypatch.setenv("OXOCR_RESIDENT", "0")
    assert OcrConfig().resident is False


def test_options_fingerprint_sensitive():
    a = ModelOptions(use_cls=True).fingerprint()
    b = ModelOptions(use_cls=False).fingerprint()
    assert a != b
    assert ModelOptions(num_threads=4).fingerprint() != ModelOptions(num_threads=8).fingerprint()
