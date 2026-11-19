"""pad_border 门控：无检测模型（公式识别）默认关闭补边，显式传参可恢复。

补边只为提高文字检测效果服务；`detection=False` 的模型整图直识、无检测步骤，
补边无必要（indra math 甚至要在内部把补边裁掉）。与 own_preprocess/own_postprocess
门控对称：`pre_explicit` = 调用处传了 preprocess=...，或 Mapping 配置带 preprocess 键。
门控只关 pad_border 一步，保留 dark_to_light 等其余前处理。
"""
from __future__ import annotations

import numpy as np
import pytest
from oxocr.config import OcrConfig
from oxocr.facade import _build_config, _pad_default_off
from oxocr.models.base import InferRequest, ModelOutput
from oxocr.pipeline import OcrPipeline
from oxocr.types import Box, TextItem

_NO_DET = ["pix2text_mfr", "pix2text_mfr_1.5", "pix2text_mfr_quantized", "indra_ocr_math", "rapid_latex_ocr"]
_WITH_DET = ["ppocrv6_small", "indra_ocr_latin"]


def _explicit_of(**kwargs) -> bool:
    return _build_config(
        kwargs.pop("config", None),
        model_name="pix2text_mfr_1.5",
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


@pytest.mark.parametrize("name", _NO_DET)
def test_pad_default_off_for_no_detection_models(name):
    from oxocr.registry.catalog import BUILTIN_SPECS

    spec = BUILTIN_SPECS[name]
    assert spec.capabilities.detection is False
    assert _pad_default_off(spec, pre_explicit=False) is True
    # 显式表态（preprocess=...）时不门控，尊重调用方
    assert _pad_default_off(spec, pre_explicit=True) is False


@pytest.mark.parametrize("name", _WITH_DET)
def test_pad_not_gated_for_detection_models(name):
    from oxocr.registry.catalog import BUILTIN_SPECS

    spec = BUILTIN_SPECS[name]
    assert spec.capabilities.detection is True
    assert _pad_default_off(spec, pre_explicit=False) is False


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


def _run_pipeline(pad_default_off: bool, config: OcrConfig | None = None, **flags):
    image = np.full((40, 60, 3), 255, dtype=np.uint8)  # 小图，默认会触发 pad_border
    executor = _FakeExecutor()
    pipeline = OcrPipeline(
        config or OcrConfig(), executor, {}, pad_default_off=pad_default_off, **flags
    )
    result = pipeline.run(image)
    return executor.image, result


def test_gate_passes_original_image():
    # 门控开启：小图原样直送，不被 pad_border 补边
    sent, r = _run_pipeline(pad_default_off=True)
    assert sent.shape == (40, 60, 3)
    assert r.config.preprocess.options["pad_border"]["enabled"] is False


def test_gate_keeps_other_preprocess():
    # 只关 pad_border 一步：核心前处理整体仍启用，dark_to_light 保留
    _, r = _run_pipeline(pad_default_off=True)
    assert r.config.preprocess.enabled is True
    assert "dark_to_light" in r.config.preprocess.steps
    assert r.config.preprocess.options.get("dark_to_light", {}).get("enabled", True) is True


def test_without_gate_applies_pad_border():
    # 门控关闭：小图被 pad_border 补边，送入尺寸变大
    sent, _ = _run_pipeline(pad_default_off=False)
    assert sent.shape != (40, 60, 3)


def test_explicit_config_reenables_pad_border():
    # 显式开启 pad_border（门控不置位）时照常补边
    cfg = OcrConfig().merge({"preprocess": {"options": {"pad_border": {"enabled": True}}}})
    sent, _ = _run_pipeline(pad_default_off=False, config=cfg)
    assert sent.shape != (40, 60, 3)


def test_pre_default_off_takes_precedence():
    # rapid_latex_ocr：own_preprocess 全关前处理优先于 pad 门控（elif 分支）
    sent, r = _run_pipeline(pad_default_off=True, pre_default_off=True)
    assert r.config.preprocess.enabled is False
    assert sent.shape == (40, 60, 3)
