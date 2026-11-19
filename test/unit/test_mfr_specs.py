"""Pix2Text MFR 清单契约（纯数据 + 本地权重路径）。"""
from __future__ import annotations

import pytest
from oxocr.models.base import OcrModel
from oxocr.models.pix2text.specs import MFR_SPECS
from oxocr.registry import registry
from oxocr.registry.catalog import BUILTIN_SPECS


def test_three_specs_declared():
    assert set(MFR_SPECS) == {"pix2text_mfr", "pix2text_mfr_1.5", "pix2text_mfr_quantized"}


def test_spec_fields():
    for spec in MFR_SPECS.values():
        assert spec.family == "pix2text"
        assert spec.impl == "oxocr.models.pix2text.pipeline:MfrModel"
        assert spec.capabilities.detection is False
        assert spec.capabilities.recognition is True
        assert spec.capabilities.own_postprocess is True
        assert spec.capabilities.angle_classification is False
        assert spec.license == "MIT"
        assert set(spec.defaults) == {"max_new_tokens", "repeat_limit"}
        assert spec.defaults["max_new_tokens"] == 256


def test_file_roles_and_layout():
    for spec in MFR_SPECS.values():
        dirname = spec.files[0].filename.split("/")[0]
        assert all(f.filename.startswith(dirname + "/") for f in spec.files)
        assert {f.role for f in spec.files} == {"encoder", "decoder", "tokenizer", "config"}


def test_raw_model_ids_match_source():
    # 量化版是 1.0 原版的 int8 encoder 变体（decoder/tokenizer 逐字节相同），不指向 1.5
    assert MFR_SPECS["pix2text_mfr"].raw_model_ids == {"rec": "breezedeus/pix2text-mfr"}
    assert MFR_SPECS["pix2text_mfr_1.5"].raw_model_ids == {"rec": "breezedeus/pix2text-mfr-1.5"}
    assert MFR_SPECS["pix2text_mfr_quantized"].raw_model_ids == {"rec": "breezedeus/pix2text-mfr"}


def test_impl_resolves_to_ocr_model():
    cls = registry.import_string(MFR_SPECS["pix2text_mfr_1.5"].impl)
    assert issubclass(cls, OcrModel)


def test_specs_aggregated_in_catalog():
    assert set(MFR_SPECS) <= set(BUILTIN_SPECS)
    assert all(spec.name in BUILTIN_SPECS for spec in MFR_SPECS.values())


@pytest.mark.model
def test_spec_files_exist_offline(mfr_models_dir):
    """清单里的每条相对路径都必须能在 test/models 命中，防止死路径。"""
    missing = [
        f.filename
        for spec in MFR_SPECS.values()
        for f in spec.files
        if not (mfr_models_dir / f.filename).is_file()
    ]
    assert not missing
