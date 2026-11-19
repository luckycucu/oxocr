"""扩展契约测试：dummy OcrModel + dummy 前处理，验证「写一个 infer() 即接入」
与常驻池语义。不依赖 onnxruntime / 权重。
"""
from __future__ import annotations

import threading

import numpy as np
import oxocr
import pytest
from oxocr.models.base import Capabilities, InferRequest, ModelOutput, ModelSpec, OcrModel
from oxocr.postprocess.base import PostContext
from oxocr.preprocess.base import Affine, PreContext, register_preprocessor
from oxocr.registry import registry
from oxocr.runtime.pool import pool_instance
from oxocr.types import Box, TextItem


class DummyModel(OcrModel):
    family = "dummyfam"
    capabilities = Capabilities(detection=True, recognition=True)
    created = 0
    loads = 0
    infers = 0
    _counter_lock = threading.Lock()

    def __init__(self, spec, files, options):
        super().__init__(spec, files, options)
        with DummyModel._counter_lock:
            DummyModel.created += 1

    def load(self):
        with DummyModel._counter_lock:
            DummyModel.loads += 1
        self._loaded = True

    def unload(self):
        self._loaded = False

    def infer(self, image, request: InferRequest) -> ModelOutput:
        with DummyModel._counter_lock:
            DummyModel.infers += 1
        # 返回送入图坐标下的两个 box（右移/下移 100/50，供逆映射验证）
        items = [
            TextItem("hello", 0.9, Box.from_rect(105, 55, 50, 15)),
            TextItem("world", 0.9, Box.from_rect(160, 55, 50, 15)),
        ]
        return ModelOutput(items=items, elapse={"det": 0.01, "rec": 0.02})


@register_preprocessor("dummy_pad")
class DummyPad:
    """可控前处理：固定补边 (left=100, top=50) 并压入变换栈。"""

    name = "dummy_pad"
    default_options = {"enabled": True}

    def apply(self, ctx: PreContext):
        ctx.image = np.pad(ctx.image, [(50, 10), (100, 10), (0, 0)], mode="constant")
        ctx.transform = ctx.transform.then(Affine.translation(100, 50))
        return ctx


class DummyModifier:
    name = "dummy_modifier"
    default_options = {"enabled": True}

    def on_blocks(self, blocks, ctx: PostContext):
        for b in blocks:
            b.text = b.text.upper()
        return blocks


@pytest.fixture
def dummy_spec(tmp_path):
    DummyModel.created = DummyModel.loads = DummyModel.infers = 0
    spec = ModelSpec(
        name="dummymodel",
        family="dummyfam",
        impl="test.contract.dummy:DummyModel",
        files=(),
        defaults={"det_limit_side_len": 736},
        capabilities=DummyModel.capabilities,
    )
    registry.register_spec(spec)
    registry.register_model_impl("dummyfam", DummyModel)
    pool_instance().unload_all()
    yield spec, tmp_path
    registry._RUNTIME_SPECS.pop("dummymodel", None)
    registry._RUNTIME_IMPLS.pop("dummyfam", None)
    pool_instance().unload_all()


@pytest.fixture
def blank_image():
    return np.zeros((80, 200, 3), dtype=np.uint8)


def test_infer_only_extension(dummy_spec, blank_image):
    spec,mdir = dummy_spec
    r = oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False,
                  postprocess={"mode": "line"})
    assert [it.text for it in r.items] == ["hello", "world"]
    assert DummyModel.created == 1 and DummyModel.loads == 1


def test_coordinate_inverse_map_via_preprocess(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    r = oxocr.ocr(blank_image, model=spec, model_dir=mdir,
                  preprocess=[DummyPad()], postprocess=False)
    # 模型给 105/55（padded 坐标），减去补边 100/50 → 原图 5/5
    assert r.items[0].box.x_min == pytest.approx(5)
    assert r.items[0].box.y_min == pytest.approx(5)


def test_preprocess_disabled_identity_transform(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    r = oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False)
    assert r.items[0].box.x_min == pytest.approx(105)  # 未补边，原样


def test_single_step_toggle(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    r = oxocr.ocr(blank_image, model=spec, model_dir=mdir,
                  preprocess={"steps": ["dummy_pad"],
                              "options": {"dummy_pad": {"enabled": False}}})
    assert r.items[0].box.x_min == pytest.approx(105)  # 该步关闭


def test_modifier_injection(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    r = oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False,
                  postprocess={"mode": "line"})
    assert r.text == "hello world"
    r2 = oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False,
                   postprocess=[DummyModifier()])
    assert "HELLO" in r2.text


def test_as_text_reuses_inference(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    r = oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False,
                  postprocess={"mode": "para"})
    assert DummyModel.infers == 1
    _ = r.as_text("line")
    _ = r.as_text("raw")
    assert DummyModel.infers == 1  # 换排版不重新推理


def test_resident_reuse_single_instance(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False)
    oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False)
    oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False)
    assert DummyModel.created == 1
    assert DummyModel.loads == 1
    assert DummyModel.infers == 3
    snaps = pool_instance().snapshot()
    assert any(s.name == "dummymodel" and s.infer_count == 3 for s in snaps)


def test_different_options_new_instance(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False, use_cls=True)
    oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False, use_cls=False)
    assert DummyModel.created == 2


def test_unload_then_reacquire_reloads(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False)
    assert DummyModel.loads == 1
    oxocr.unload_model("dummymodel", model_dir=mdir)
    assert pool_instance().snapshot() == []
    oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False)
    assert DummyModel.loads == 2  # 重新加载


def test_concurrent_get_or_load_single_instance(dummy_spec, blank_image):
    spec, mdir = dummy_spec
    errors = []

    def work():
        try:
            oxocr.ocr(blank_image, model=spec, model_dir=mdir, preprocess=False)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert DummyModel.created == 1
    assert DummyModel.loads == 1
    assert DummyModel.infers == 8
