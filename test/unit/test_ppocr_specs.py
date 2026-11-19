"""PP-OCR 清单与方向分类变体的纯数据契约（不加载权重）。"""
from __future__ import annotations

import pytest
from oxocr.config import ModelOptions
from oxocr.errors import UnknownParameterError
from oxocr.models.rapid_ocr.specs import (
    CLS_MODELS,
    SPECS,
    cls_file_for,
    cls_variant,
)
from oxocr.registry import registry


@pytest.mark.model
def test_every_spec_file_exists_offline(models_dir):
    """清单里的每条相对路径都必须能在 test/models 命中，防止再次出现死路径。"""
    missing = {
        name: [f.filename for f in spec.files if not (models_dir / f.filename).is_file()]
        for name, spec in SPECS.items()
    }
    assert not {k: v for k, v in missing.items() if v}


def test_all_versions_covered():
    for version in (3, 4, 5, 6):
        assert any(n.startswith(f"ppocrv{version}_") for n in SPECS)


def test_defaults_match_reference():
    """RapidOCR 对所有版本统一 736/min 与 rec 高度 48。"""
    for spec in SPECS.values():
        assert spec.defaults["det_limit_side_len"] == 736
        assert spec.defaults["rec_image_shape"][1] == 48
        assert spec.family == "ppocr"
        assert spec.impl == "oxocr.models.rapid_ocr.pipeline:PpOcrModel"


def test_cls_default_follows_version():
    assert SPECS["ppocrv5_ch_server"].defaults["cls_model"] == "textline_ori_mobile"
    assert SPECS["ppocrv4_ch_server"].defaults["cls_model"] == "v2_0_cls"
    assert SPECS["ppocrv6_small"].defaults["cls_model"] == "v2_0_cls"


def test_textline_variants_use_static_80x160():
    # ONNX 图里 80×160 是静态维度，与 v2.0 的 48×192 不可互换
    assert CLS_MODELS["textline_ori_mobile"].image_shape == (3, 80, 160)
    assert CLS_MODELS["textline_ori_server"].image_shape == (3, 80, 160)
    assert CLS_MODELS["v2_0_cls"].image_shape == (3, 48, 192)


def test_cls_file_for_default_and_override():
    spec = SPECS["ppocrv6_small"]
    assert cls_file_for(spec, {}).filename == CLS_MODELS["v2_0_cls"].filename
    for name, variant in CLS_MODELS.items():
        assert cls_file_for(spec, {"cls_model": name}).filename == variant.filename


def test_unknown_cls_model_rejected():
    spec = SPECS["ppocrv6_small"]
    with pytest.raises(UnknownParameterError) as e:
        cls_variant("nope", "测试")
    assert "textline_ori_server" in str(e.value)
    with pytest.raises(UnknownParameterError):
        cls_file_for(spec, {"cls_model": "nope"})


@pytest.mark.model
def test_resolve_files_honors_cls_override(models_dir):
    spec = SPECS["ppocrv6_small"]
    for name, variant in CLS_MODELS.items():
        files = registry.resolve_files(spec, models_dir, ModelOptions(params={"cls_model": name}))
        assert files["cls"] == models_dir / variant.filename


def test_cls_model_is_whitelisted_param():
    for spec in SPECS.values():
        assert "cls_model" in spec.defaults
        # 形状由权重决定，不再作为独立参数暴露
        assert "cls_image_shape" not in spec.defaults


def test_v5_server_and_mobile_pair_with_matching_det():
    assert SPECS["ppocrv5_ch_server"].files[0].filename.endswith("ch_PP-OCRv5_det_server.onnx")
    assert SPECS["ppocrv5_ch_mobile"].files[0].filename.endswith("ch_PP-OCRv5_det_mobile.onnx")
    assert SPECS["ppocrv4_ch_doc_server"].files[0].filename.endswith("ch_PP-OCRv4_det_server.onnx")
