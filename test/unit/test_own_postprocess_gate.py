"""own_postprocess 门控：公式模型默认跳过后处理，显式传参可恢复。

门控规则（docs/mfr.md §5.3）：`post_explicit` = 调用处传了 postprocess=...，
或 Mapping 配置带 postprocess 键；OcrConfig 实例无法区分默认值与显式开启，
视为未表态。own_postprocess 且未表态 → 管线按 enabled=False 直通。
"""
from __future__ import annotations

import numpy as np
import pytest
from oxocr.config import OcrConfig
from oxocr.facade import _build_config, _post_default_off
from oxocr.models.base import InferRequest, ModelOutput
from oxocr.pipeline import OcrPipeline
from oxocr.types import Box, TextItem

_MFR_SPEC = None


def _mfr_spec():
    global _MFR_SPEC
    if _MFR_SPEC is None:
        from oxocr.registry.catalog import BUILTIN_SPECS

        _MFR_SPEC = BUILTIN_SPECS["pix2text_mfr_1.5"]
    return _MFR_SPEC


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
    )[3]  # post_explicit（返回 5 元组：cfg, pre_inst, post_inst, post_explicit, pre_explicit）


def test_not_explicit_by_default():
    assert _explicit_of() is False


@pytest.mark.parametrize(
    "postprocess", [True, False, {"steps": []}, (object(),)]
)
def test_postprocess_kwarg_is_explicit(postprocess):
    assert _explicit_of(postprocess=postprocess) is True


def test_mapping_config_with_postprocess_key_is_explicit():
    assert _explicit_of(config={"postprocess": {"mode": "raw"}}) is True
    assert _explicit_of(config={"model": "x"}) is False


def test_ocrconfig_instance_is_not_explicit():
    # 实例上的默认 enabled=True 与显式开启无法区分，统一视为未表态
    assert _explicit_of(config=OcrConfig()) is False


def test_post_default_off_helper():
    spec = _mfr_spec()
    assert _post_default_off(spec, post_explicit=False) is True
    assert _post_default_off(spec, post_explicit=True) is False


class _FakeExecutor:
    """返回单个整图 LaTeX 条目，不加载任何模型。"""

    def __init__(self):
        self.request: InferRequest | None = None

    @property
    def is_loaded(self):
        return True

    def warm(self):
        return None

    def close(self):
        return None

    def run(self, image, request: InferRequest) -> ModelOutput:
        self.request = request
        h, w = image.shape[:2]
        return ModelOutput(
            items=[
                TextItem(
                    text="\\frac { 1 } { 2 }",
                    score=0.9,
                    box=Box.from_rect(0.0, 0.0, float(w), float(h)),
                )
            ],
            elapse={"rec": 0.1},
        )


def _run_pipeline(post_default_off: bool, config: OcrConfig | None = None):
    cfg = config or OcrConfig()
    image = np.full((40, 60, 3), 255, dtype=np.uint8)
    pipeline = OcrPipeline(cfg, _FakeExecutor(), {}, post_default_off=post_default_off)
    return pipeline.run(image)


def test_pipeline_gate_disables_postprocess():
    r = _run_pipeline(post_default_off=True)
    assert r.config.postprocess.enabled is False
    # raw 直通：LaTeX 原样进 block 与最终文本，未被 normalize_space/corrections 改写
    assert r.text.strip() == "\\frac { 1 } { 2 }"
    assert r.items[0].text == "\\frac { 1 } { 2 }"


def test_pipeline_without_gate_keeps_default_postprocess():
    r = _run_pipeline(post_default_off=False)
    assert r.config.postprocess.enabled is True


def test_pipeline_gate_with_explicit_enabled_config():
    # 门面在显式场景下不会传 post_default_off；这里验证配置直传时后处理生效
    cfg = OcrConfig().merge({"postprocess": {"enabled": True}})
    r = _run_pipeline(post_default_off=False, config=cfg)
    assert r.config.postprocess.enabled is True
