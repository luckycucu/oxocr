"""anaksor 清单纯数据契约（不加载权重）。"""

from __future__ import annotations

import pytest
from oxocr.config import ModelOptions
from oxocr.models.anaksor_ocr.specs import ANAKSOR_SPECS
from oxocr.models.base import OcrModel
from oxocr.registry import registry
from oxocr.registry.catalog import BUILTIN_SPECS


def test_single_spec_declared():
    assert set(ANAKSOR_SPECS) == {"anaksor_ocr"}


def test_spec_fields():
    spec = ANAKSOR_SPECS["anaksor_ocr"]
    assert spec.family == "anaksor_ocr"
    assert spec.impl == "oxocr.models.anaksor_ocr.pipeline:AnaksorModel"
    assert spec.capabilities.detection is True
    assert spec.capabilities.recognition is True
    assert spec.capabilities.angle_classification is False
    assert spec.capabilities.own_postprocess is False
    assert spec.capabilities.backend == "onnxruntime"
    assert spec.license == "Apache-2.0"
    assert spec.languages == ("km", "en")


def test_file_roles():
    spec = ANAKSOR_SPECS["anaksor_ocr"]
    assert {f.role for f in spec.files} == {"det", "rec", "dict"}


def test_files_have_sha256_and_size():
    for f in ANAKSOR_SPECS["anaksor_ocr"].files:
        assert f.sha256 and len(f.sha256) == 64
        assert f.size and f.size > 0
        assert f.url is None  # 自动下载暂缓


def test_params_whitelist():
    spec = ANAKSOR_SPECS["anaksor_ocr"]
    assert set(spec.defaults) == {
        "det_limit_side_len",
        "det_min_side_len",
        "det_thresh",
        "det_box_thresh",
        "det_unclip_ratio",
        "rec_batch_num",
        "crop_pad_y",
        "crop_pad_x",
        "try_vertical",
        "try_180",
    }
    assert spec.defaults["det_limit_side_len"] == 1280
    assert spec.defaults["det_min_side_len"] == 736
    assert spec.defaults["det_thresh"] == 0.3
    assert spec.defaults["det_box_thresh"] == 0.5
    assert spec.defaults["det_unclip_ratio"] == 1.5
    assert spec.defaults["rec_batch_num"] == 32
    assert spec.defaults["crop_pad_y"] == 0.16
    assert spec.defaults["crop_pad_x"] == 0.10
    assert spec.defaults["try_vertical"] is True
    assert spec.defaults["try_180"] is False


def test_impl_resolves_to_ocr_model():
    cls = registry.import_string(ANAKSOR_SPECS["anaksor_ocr"].impl)
    assert issubclass(cls, OcrModel)


def test_specs_aggregated_in_catalog():
    assert set(ANAKSOR_SPECS) <= set(BUILTIN_SPECS)
    assert BUILTIN_SPECS["anaksor_ocr"] is ANAKSOR_SPECS["anaksor_ocr"]


@pytest.mark.model
def test_spec_files_exist_offline(anaksor_models_dir):
    missing = [
        f.filename
        for spec in ANAKSOR_SPECS.values()
        for f in spec.files
        if not (anaksor_models_dir / f.filename).is_file()
    ]
    assert not missing


@pytest.mark.model
def test_resolve_files_roles(anaksor_models_dir):
    spec = ANAKSOR_SPECS["anaksor_ocr"]
    files = registry.resolve_files(spec, anaksor_models_dir, ModelOptions())
    assert set(files) == {"det", "rec", "dict"}
    assert all(p.is_file() for p in files.values())


@pytest.mark.model
def test_dict_always_required(anaksor_models_dir):
    # dict 非 gated role：use_rec=False 也要求存在
    spec = ANAKSOR_SPECS["anaksor_ocr"]
    files = registry.resolve_files(spec, anaksor_models_dir, ModelOptions(use_rec=False))
    assert "dict" in files
