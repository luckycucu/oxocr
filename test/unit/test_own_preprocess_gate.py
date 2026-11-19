"""own_preprocess 门控：自带前处理的模型默认跳过核心前处理，显式传参可恢复。

与 own_postprocess 门控对称（docs/latex-ocr.md §4）：`pre_explicit` = 调用处传了
preprocess=...，或 Mapping 配置带 preprocess 键；OcrConfig 实例视为未表态。
own_preprocess 且未表态 → 管线按 preprocess.enabled=False 直通（原图送模型）。
"""
from __future__ import annotations

import numpy as np
import pytest
from oxocr.config import OcrConfig
from oxocr.facade import _build_config, _pre_default_off
from oxocr.models.base import InferRequest, ModelOutput
from oxocr.pipeline import OcrPipeline
from oxocr.types import Box, TextItem


def _explicit_of(**kwargs) -> bool:
    return _build_config(
        kwargs.pop("config", None),
        model_name="rapid_latex_ocr",
        mode=kwargs.pop("mode", None),
        output=kwargs.pop("output", None),
        preprocess=kwargs.pop("preprocess", None),
        postprocess=kwargs.pop("postprocess", None),
        model_dir=None,
        backend=None,
        resident=None,
        debug=None,
        overrides={},
    )[4]  # pre_explicit（返回 5 元组：cfg, pre_inst, post_inst, post_explicit, pre_explicit）


def test_not_explicit_by_default():
    assert _explicit_of() is False


@pytest.mark.parametrize("preprocess", [True, False, {"steps": []}, (object(),)])
def test_preprocess_kwarg_is_explicit(preprocess):
    assert _explicit_of(preprocess=preprocess) is True


def test_mapping_config_with_preprocess_key_is_explicit():
    assert _explicit_of(config={"preprocess": {"enabled": True}}) is True
    assert _explicit_of(config={"model": "x"}) is False


def test_ocrconfig_instance_is_not_explicit():
    assert _explicit_of(config=OcrConfig()) is False


def test_pre_default_off_helper():
    from oxocr.registry.catalog import BUILTIN_SPECS

    spec = BUILTIN_SPECS["rapid_latex_ocr"]
    assert _pre_default_off(spec, pre_explicit=False) is True
    assert _pre_default_off(spec, pre_explicit=True) is False
    # 非 own_preprocess 模型（PP-OCR）不受门控影响
    pp = BUILTIN_SPECS["ppocrv6_small"]
    assert _pre_default_off(pp, pre_explicit=False) is False


class _FakeExecutor:
    """记录送入的图像，返回单个整图条目，不加载任何模型。"""

    def __init__(self):
        self.image: np.ndarray | None = None

    @property
    def is_loaded(self):
        return True

    def warm(self):
        return None

    def close(self):
        return None

    def run(self, image, request: InferRequest) -> ModelOutput:
        self.image = image
        h, w = image.shape[:2]
        return ModelOutput(
            items=[TextItem(text="x", score=0.9, box=Box.from_rect(0.0, 0.0, float(w), float(h)))],
            elapse={"rec": 0.1},
        )


def _run_pipeline(pre_default_off: bool):
    image = np.full((40, 60, 3), 255, dtype=np.uint8)  # 小图，默认会触发 pad_border
    executor = _FakeExecutor()
    pipeline = OcrPipeline(OcrConfig(), executor, {}, pre_default_off=pre_default_off)
    pipeline.run(image)
    return executor.image


def test_pipeline_gate_passes_original_image():
    # 门控开启：原图直送，不被 pad_border 补边
    assert _run_pipeline(pre_default_off=True).shape == (40, 60, 3)


def test_pipeline_without_gate_applies_pad_border():
    # 门控关闭：小图被 pad_border 补边，送入尺寸变大
    assert _run_pipeline(pre_default_off=False).shape != (40, 60, 3)
