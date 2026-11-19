"""LaTeX-OCR 预处理与输出整理单测（不依赖权重 / onnxruntime）。"""
from __future__ import annotations

import json

import numpy as np
import pytest
from oxocr.errors import InferenceError
from oxocr.models.rapid_latex_ocr.pipeline import _compact_latex, _read_bos_eos
from oxocr.models.rapid_latex_ocr.preproc import minmax_size, pad, to_tensor
from PIL import Image


def _gray(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8)).convert("L")


def test_pad_crops_to_text_bbox_and_aligns_32():
    arr = np.full((100, 100), 255, dtype=np.uint8)
    arr[20:61, 10:51] = 0  # 41×41 黑块
    out = pad(_gray(arr))
    assert out.mode == "L"
    assert out.size == (64, 64)  # 41 →  next 32 倍数
    o = np.asarray(out)
    assert o[0, 0] == 0  # 黑块裁到左上角
    assert o[40, 40] == 0
    assert o[41, 41] == 255  # 其余白底
    assert o[63, 63] == 255


def test_pad_inverts_dark_background():
    arr = np.zeros((80, 80), dtype=np.uint8)  # 黑底
    arr[20:40, 20:40] = 255  # 白字
    out = np.asarray(pad(_gray(arr)))
    assert out[0, 0] == 0  # 白字反色为黑，裁到左上角
    assert out[31, 31] == 255  # 其余白底


def test_pad_uniform_image_does_not_crash():
    out = pad(_gray(np.full((50, 50), 255, dtype=np.uint8)))
    assert out.size == (64, 64)
    assert np.asarray(out).min() == 255  # 全白兜底


def test_pad_blank_image_does_not_crash():
    out = pad(_gray(np.zeros((50, 50), dtype=np.uint8)))
    assert out.size[0] % 32 == 0 and out.size[1] % 32 == 0


def test_minmax_downscales_oversize():
    out = minmax_size(_gray(np.full((384, 1344), 255, dtype=np.uint8)))
    assert out.size == (672, 192)  # 按 max(1344/672, 384/192)=2 等比缩


def test_minmax_pads_undersize():
    out = minmax_size(_gray(np.full((10, 10), 255, dtype=np.uint8)))
    assert out.size == (32, 32)


def test_to_tensor_shape_and_normalization():
    t = to_tensor(_gray(np.full((32, 64), 255, dtype=np.uint8)))
    assert t.shape == (1, 1, 32, 64)
    assert t.dtype == np.float32
    expected = (255 - 0.7931 * 255) / (0.1738 * 255)
    assert float(t[0, 0, 0, 0]) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("x ^ { 2 } + y ^ { 2 } = 1", "x^{2}+y^{2}=1"),
        (
            "{ \\frac { x ^ { 2 } } { a ^ { 2 } } } - { \\frac { y ^ { 2 } } { b ^ { 2 } } } = 1",
            "{\\frac{x^{2}}{a^{2}}}-{\\frac{y^{2}}{b^{2}}}=1",
        ),
        ("4 a c", "4a c"),  # 字母↔字母间空格保留（参考行为）
        ("\\mathrm { a b }", "\\mathrm{ab}"),
    ],
)
def test_compact_latex(raw, expected):
    assert _compact_latex(raw) == expected


def test_read_bos_eos_from_added_tokens(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text(
        json.dumps({"added_tokens": [{"id": 1, "content": "[BOS]"}, {"id": 2, "content": "[EOS]"}]}),
        encoding="utf-8",
    )
    assert _read_bos_eos(p) == (1, 2)


def test_read_bos_eos_missing_raises(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps({"added_tokens": [{"id": 0, "content": "[PAD]"}]}), encoding="utf-8")
    with pytest.raises(InferenceError):
        _read_bos_eos(p)
