from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from oxocr.config import ModelOptions
from oxocr.errors import ModelFileMissingError, ModelNotFoundError, UnknownParameterError
from oxocr.models.base import ModelFile, ModelSpec
from oxocr.registry import registry

REPO = Path(__file__).resolve().parent.parent.parent
MODELS = REPO / "test" / "models"


def test_resolve_builtin_spec():
    spec = registry.resolve_spec("ppocrv6_small")
    assert spec.family == "ppocr"
    assert any(f.role == "det" for f in spec.files)


def test_unknown_model_lists_available():
    with pytest.raises(ModelNotFoundError) as e:
        registry.resolve_spec("nope-not-a-model")
    assert "ppocrv6_small" in str(e.value)


def test_register_spec_and_impl():
    spec = ModelSpec(
        name="unit_fake",
        family="unitfake",
        impl="oxocr.models.rapid_ocr.pipeline:PpOcrModel",
        files=(ModelFile("model", "x.onnx"),),
        defaults={"alpha": 1},
    )
    registry.register_spec(spec)
    try:
        assert registry.resolve_spec("unit_fake") is spec
    finally:
        registry._RUNTIME_SPECS.pop("unit_fake", None)


def test_params_whitelist_violation():
    from oxocr.models.rapid_ocr.pipeline import PpOcrModel

    spec = registry.resolve_spec("ppocrv6_small")
    model = PpOcrModel(spec, {}, ModelOptions(params={"bogus": 1}))
    with pytest.raises(UnknownParameterError):
        model.validate_options(ModelOptions(params={"bogus": 1}))


def test_resolve_files_missing_raises_with_path(tmp_path):
    spec = registry.resolve_spec("ppocrv6_small")
    with pytest.raises(ModelFileMissingError) as e:
        registry.resolve_files(spec, tmp_path, ModelOptions())
    assert "det" in e.value.missing


def test_resolve_files_lm_orient_always_optional(tmp_path):
    spec = ModelSpec(
        name="unit_optional_roles",
        family="unitfake",
        impl="oxocr.models.rapid_ocr.pipeline:PpOcrModel",
        files=(
            ModelFile("det", "d.onnx"),
            ModelFile("rec", "r.onnx"),
            ModelFile("meta", "m.json"),
            ModelFile("lm", "lm.npz"),
            ModelFile("orient", "o.onnx"),
        ),
    )
    (tmp_path / "d.onnx").write_bytes(b"x")
    (tmp_path / "r.onnx").write_bytes(b"x")
    (tmp_path / "m.json").write_bytes(b"{}")
    files = registry.resolve_files(spec, tmp_path, ModelOptions())
    assert set(files) == {"det", "rec", "meta"}  # 缺 lm/orient 不报错
    installed, _ = registry._installed_paths(spec, tmp_path)
    assert installed  # lm/orient 不计入必需
    (tmp_path / "m.json").unlink()
    with pytest.raises(ModelFileMissingError) as e:
        registry.resolve_files(spec, tmp_path, ModelOptions())
    assert "meta" in e.value.missing  # 非 gated role 恒必需


def test_resolve_files_offline_layout(models_dir):
    spec = registry.resolve_spec("ppocrv6_small")
    files = registry.resolve_files(spec, models_dir, ModelOptions())
    assert files["det"].is_file() and files["rec"].is_file() and files["cls"].is_file()


def test_list_models_installed_flag(monkeypatch, models_dir, tmp_path):
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(models_dir))
    names = {m.name: m.installed for m in registry.list_models()}
    assert names["ppocrv6_small"] is True
    assert names["ppocrv5_ch_server"] is True
    monkeypatch.setenv("OXOCR_MODEL_DIR", str(tmp_path))
    assert all(not m.installed for m in registry.list_models())


def test_builtin_specs_metadata_contract():
    """docs/模型信息字段.md 要求的元数据字段必须齐全且自洽。"""
    from oxocr.registry.catalog import BUILTIN_SPECS

    for spec in BUILTIN_SPECS.values():
        assert spec.name_chinese, spec.name
        assert spec.raw_author and spec.raw_license, spec.name
        assert spec.raw_urls and spec.raw_urls[0].startswith("http"), spec.name
        assert spec.model_type in ("text", "latex"), spec.name
        assert len(spec.languages) == len(spec.languages_chinese) == len(spec.languages_english)
        assert spec.size_bytes > 0, spec.name
        assert 0 <= spec.star_rating <= 5, spec.name
        assert spec.model_architecture, spec.name
        assert spec.commercial_use.split("(")[0] in ("yes", "no", "unknown", "restricted")
        # current_* 为空时回退 raw_*
        assert spec.author == (spec.current_author or spec.raw_author)
        assert spec.license == (spec.current_license or spec.raw_license)
        assert spec.urls == (spec.current_urls if spec.current_urls is not None else spec.raw_urls)
        assert ModelSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.model
def test_builtin_size_bytes_matches_offline(models_dir):
    from oxocr.registry.catalog import BUILTIN_SPECS

    for spec in BUILTIN_SPECS.values():
        actual = sum((models_dir / f.filename).stat().st_size for f in spec.files)
        assert actual == spec.size_bytes, spec.name


def test_list_models_import_lightweight():
    """list_models 不得触发 onnxruntime / pyclipper / upolygon 导入。"""
    code = (
        "import sys, oxocr; oxocr.list_models();"
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime';"
        "assert 'pyclipper' not in sys.modules, 'pyclipper';"
        "assert 'upolygon' not in sys.modules, 'upolygon';"
        "print('LIGHT_OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "LIGHT_OK" in out.stdout, out.stderr
