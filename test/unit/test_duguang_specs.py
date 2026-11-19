"""读光清单与 det 变体表的纯数据契约（不加载权重）。"""
from __future__ import annotations

import pytest
from oxocr.config import ModelOptions
from oxocr.errors import UnknownParameterError
from oxocr.models.base import OcrModel
from oxocr.models.duguang_ocr.specs import DET_VARIANTS, DUGUANG_SPECS, det_file_for
from oxocr.registry import registry
from oxocr.registry.catalog import BUILTIN_SPECS


def test_three_specs_declared():
    assert set(DUGUANG_SPECS) == {"duguang_ocr_small", "duguang_ocr_large", "duguang_ocr_max"}


def test_spec_fields():
    for spec in DUGUANG_SPECS.values():
        assert spec.family == "duguang_ocr"
        assert spec.impl == "oxocr.models.duguang_ocr.pipeline:DuguangModel"
        assert spec.capabilities.detection is True
        assert spec.capabilities.recognition is True
        assert spec.capabilities.angle_classification is False
        assert spec.capabilities.own_postprocess is False
        assert spec.license == "Apache-2.0"
        assert spec.languages == ("zh", "en")


def test_file_roles():
    for spec in DUGUANG_SPECS.values():
        assert {f.role for f in spec.files} == {"det", "rec", "vocab"}


def test_db_params_whitelist():
    for name in ("duguang_ocr_small", "duguang_ocr_large"):
        spec = DUGUANG_SPECS[name]
        assert set(spec.defaults) == {
            "det_image_size", "det_thresh", "det_box_thresh",
            "det_unclip_ratio", "rec_batch_num",
        }
        assert spec.defaults["det_thresh"] == 0.2
        assert spec.defaults["det_box_thresh"] == 0.3
        assert spec.defaults["det_unclip_ratio"] == 1.5
        assert spec.defaults["rec_batch_num"] == 6


def test_seglink_params_whitelist():
    spec = DUGUANG_SPECS["duguang_ocr_max"]
    assert set(spec.defaults) == {
        "det_image_size", "det_node_thresh", "det_link_thresh", "rec_batch_num",
    }
    assert spec.defaults["det_node_thresh"] == 0.4
    assert spec.defaults["det_link_thresh"] == 0.6


def test_det_variants_table():
    assert DET_VARIANTS["duguang_ocr_small"] == {1024: "duguang-ocr/small/det/model_1024x1024.onnx"}
    assert DET_VARIANTS["duguang_ocr_large"] == {512: "duguang-ocr/large/det/model_512x512.onnx"}
    assert DET_VARIANTS["duguang_ocr_max"] == {512: "duguang-ocr/max/det/model_512x512.onnx"}


def test_default_det_image_size():
    assert DUGUANG_SPECS["duguang_ocr_small"].defaults["det_image_size"] == 1024
    assert DUGUANG_SPECS["duguang_ocr_large"].defaults["det_image_size"] == 512
    assert DUGUANG_SPECS["duguang_ocr_max"].defaults["det_image_size"] == 512


def test_det_file_for_default():
    for name, spec in DUGUANG_SPECS.items():
        size = spec.defaults["det_image_size"]
        f = det_file_for(spec, {})
        assert f.role == "det"
        assert f.filename == DET_VARIANTS[name][size]


def test_det_file_for_explicit():
    spec = DUGUANG_SPECS["duguang_ocr_small"]
    f = det_file_for(spec, {"det_image_size": 1024})
    assert f.filename == DET_VARIANTS["duguang_ocr_small"][1024]


def test_unknown_det_image_size_rejected():
    spec = DUGUANG_SPECS["duguang_ocr_small"]
    with pytest.raises(UnknownParameterError) as e:
        det_file_for(spec, {"det_image_size": 512})
    assert "1024" in str(e.value)
    assert "det_image_size" in str(e.value)


def test_impl_resolves_to_ocr_model():
    cls = registry.import_string(DUGUANG_SPECS["duguang_ocr_small"].impl)
    assert issubclass(cls, OcrModel)


def test_specs_aggregated_in_catalog():
    assert set(DUGUANG_SPECS) <= set(BUILTIN_SPECS)
    assert all(spec.name in BUILTIN_SPECS for spec in DUGUANG_SPECS.values())


@pytest.mark.model
def test_spec_files_exist_offline(duguang_models_dir):
    missing = [
        f.filename
        for spec in DUGUANG_SPECS.values()
        for f in spec.files
        if not (duguang_models_dir / f.filename).is_file()
    ]
    assert not missing


@pytest.mark.model
def test_resolve_files_honors_det_variant(duguang_models_dir):
    spec = DUGUANG_SPECS["duguang_ocr_small"]
    files = registry.resolve_files(spec, duguang_models_dir, ModelOptions())
    assert files["det"] == duguang_models_dir / DET_VARIANTS["duguang_ocr_small"][1024]
    assert files["vocab"].is_file()


@pytest.mark.model
def test_resolve_files_unknown_size_rejected(duguang_models_dir):
    spec = DUGUANG_SPECS["duguang_ocr_small"]
    with pytest.raises(UnknownParameterError):
        registry.resolve_files(
            spec, duguang_models_dir, ModelOptions(params={"det_image_size": 512})
        )
