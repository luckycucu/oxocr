"""openthai 清单纯数据契约（不加载权重）。"""

from __future__ import annotations

import pytest
from oxocr.config import ModelOptions
from oxocr.models.base import OcrModel
from oxocr.models.openthai_ocr.specs import OPENTHAI_SPECS
from oxocr.registry import registry
from oxocr.registry.catalog import BUILTIN_SPECS


def test_single_spec_declared():
    assert set(OPENTHAI_SPECS) == {"openthai2.0_ocr"}


def test_spec_fields():
    spec = OPENTHAI_SPECS["openthai2.0_ocr"]
    assert spec.name == "openthai2.0_ocr"
    assert spec.family == "openthai_ocr"
    assert spec.impl == "oxocr.models.openthai_ocr.pipeline:OpenThaiModel"
    assert spec.capabilities.detection is True
    assert spec.capabilities.recognition is True
    assert spec.capabilities.angle_classification is False
    assert spec.capabilities.own_postprocess is False
    assert spec.capabilities.backend == "onnxruntime"
    assert spec.license == "Apache-2.0"
    assert spec.languages == ("th", "en")
    assert spec.model_type == "text"
    assert spec.commercial_use == "yes"


def test_file_roles():
    spec = OPENTHAI_SPECS["openthai2.0_ocr"]
    assert {f.role for f in spec.files} == {"det", "rec", "dict"}


def test_files_have_sha256_and_size():
    for f in OPENTHAI_SPECS["openthai2.0_ocr"].files:
        assert f.sha256 and len(f.sha256) == 64
        assert f.size and f.size > 0
    assert OPENTHAI_SPECS["openthai2.0_ocr"].size_bytes == sum(
        f.size for f in OPENTHAI_SPECS["openthai2.0_ocr"].files
    )


def test_params_whitelist():
    spec = OPENTHAI_SPECS["openthai2.0_ocr"]
    assert set(spec.defaults) == {
        "det_limit_side_len",
        "det_thresh",
        "det_box_thresh",
        "det_unclip_ratio",
        "split_gap",
        "rec_batch_num",
        "rec_color",
        "drop_below",
        "min_chars",
        "junk_below",
        "junk_invalid",
        "orientation",
        "deskew_above",
    }
    # det 阈值 = det.yml PostProcess 实测覆盖值
    assert spec.defaults["det_limit_side_len"] == 1280
    assert spec.defaults["det_thresh"] == 0.2
    assert spec.defaults["det_box_thresh"] == 0.4
    assert spec.defaults["det_unclip_ratio"] == 1.4
    assert spec.defaults["split_gap"] == 1.2
    # rec 过滤 = config.json 作者推荐值
    assert spec.defaults["rec_batch_num"] == 16
    assert spec.defaults["rec_color"] == "flatten"
    assert spec.defaults["drop_below"] == 0.8
    assert spec.defaults["min_chars"] == 2
    assert spec.defaults["junk_below"] == 0.9
    assert spec.defaults["junk_invalid"] == 0.05
    assert spec.defaults["orientation"] is True
    assert spec.defaults["deskew_above"] == 2.5


def test_impl_resolves_to_ocr_model():
    cls = registry.import_string(OPENTHAI_SPECS["openthai2.0_ocr"].impl)
    assert issubclass(cls, OcrModel)


def test_specs_aggregated_in_catalog():
    assert set(OPENTHAI_SPECS) <= set(BUILTIN_SPECS)
    assert BUILTIN_SPECS["openthai2.0_ocr"] is OPENTHAI_SPECS["openthai2.0_ocr"]


@pytest.mark.model
def test_spec_files_exist_offline(openthai_models_dir):
    missing = [
        f.filename
        for spec in OPENTHAI_SPECS.values()
        for f in spec.files
        if not (openthai_models_dir / f.filename).is_file()
    ]
    assert not missing


@pytest.mark.model
def test_spec_sha256_matches_weights(openthai_models_dir):
    import hashlib

    for spec in OPENTHAI_SPECS.values():
        for f in spec.files:
            data = (openthai_models_dir / f.filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == f.sha256
            assert len(data) == f.size


@pytest.mark.model
def test_resolve_files_roles(openthai_models_dir):
    spec = OPENTHAI_SPECS["openthai2.0_ocr"]
    files = registry.resolve_files(spec, openthai_models_dir, ModelOptions())
    assert set(files) == {"det", "rec", "dict"}
    assert all(p.is_file() for p in files.values())


@pytest.mark.model
def test_dict_always_required(openthai_models_dir):
    # dict 非 gated role：use_rec=False 也要求存在
    spec = OPENTHAI_SPECS["openthai2.0_ocr"]
    files = registry.resolve_files(spec, openthai_models_dir, ModelOptions(use_rec=False))
    assert "dict" in files