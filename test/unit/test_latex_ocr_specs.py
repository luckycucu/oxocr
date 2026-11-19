"""LaTeX-OCR 清单契约（纯数据 + 本地权重路径）。"""
from __future__ import annotations

import pytest
from oxocr.models.base import OcrModel
from oxocr.models.rapid_latex_ocr.specs import LATEX_OCR_SPECS
from oxocr.registry import registry
from oxocr.registry.catalog import BUILTIN_SPECS


def test_single_spec_declared():
    assert set(LATEX_OCR_SPECS) == {"rapid_latex_ocr"}


def test_spec_fields():
    spec = LATEX_OCR_SPECS["rapid_latex_ocr"]
    assert spec.family == "latex_ocr"
    assert spec.impl == "oxocr.models.rapid_latex_ocr.pipeline:LatexOcrModel"
    assert spec.capabilities.detection is False
    assert spec.capabilities.recognition is True
    assert spec.capabilities.own_preprocess is True
    assert spec.capabilities.own_postprocess is True
    assert spec.capabilities.angle_classification is False
    assert spec.license == "MIT"
    assert spec.languages == ("latex",)
    assert set(spec.defaults) == {"max_new_tokens", "repeat_limit"}
    assert spec.defaults["max_new_tokens"] == 512
    assert spec.defaults["repeat_limit"] == 12


def test_file_roles_and_layout():
    spec = LATEX_OCR_SPECS["rapid_latex_ocr"]
    assert {f.role for f in spec.files} == {"resizer", "encoder", "decoder", "tokenizer"}
    assert all(f.filename.startswith("rapid-latex-ocr/") for f in spec.files)
    by_role = {f.role: f.filename for f in spec.files}
    assert by_role["resizer"] == "rapid-latex-ocr/image_resizer.onnx"
    assert by_role["encoder"] == "rapid-latex-ocr/encoder.onnx"
    assert by_role["decoder"] == "rapid-latex-ocr/decoder.onnx"
    assert by_role["tokenizer"] == "rapid-latex-ocr/tokenizer.json"


def test_impl_resolves_to_ocr_model():
    cls = registry.import_string(LATEX_OCR_SPECS["rapid_latex_ocr"].impl)
    assert issubclass(cls, OcrModel)


def test_specs_aggregated_in_catalog():
    assert set(LATEX_OCR_SPECS) <= set(BUILTIN_SPECS)
    assert all(spec.name in BUILTIN_SPECS for spec in LATEX_OCR_SPECS.values())


@pytest.mark.model
def test_spec_files_exist_offline(latex_ocr_models_dir):
    """清单里的每条相对路径都必须能在 test/models 命中，防止死路径。"""
    missing = [
        f.filename
        for spec in LATEX_OCR_SPECS.values()
        for f in spec.files
        if not (latex_ocr_models_dir / f.filename).is_file()
    ]
    assert not missing
