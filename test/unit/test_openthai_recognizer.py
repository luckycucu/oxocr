"""CTC 解码 / 字典 / 颜色归一（合成数据，无权重）。"""

from __future__ import annotations

import numpy as np
import pytest
from oxocr.models.openthai_ocr.recognizer import (
    COLOR_MODES,
    ctc_greedy,
    prep_crop,
    read_dict,
)


def _probs(seq: list[int], n_classes: int, peak: float = 0.9) -> np.ndarray:
    """每帧 one-hot 软化：指定 id 概率 peak，其余均分。"""
    t = np.full((len(seq), n_classes), (1 - peak) / (n_classes - 1), dtype=np.float32)
    t[np.arange(len(seq)), seq] = peak
    return t[None]


def test_ctc_greedy_collapse_and_blank():
    classes = ["", "a", "b", " "]
    # blank a a blank b b b space → "ab "
    probs = _probs([0, 1, 1, 0, 2, 2, 2, 3], 4)
    (text, score), = ctc_greedy(probs, classes)
    assert text == "ab "
    # 保留帧 = 每段首帧（id 1/2/3 处），概率皆 peak
    assert score == pytest.approx(0.9, abs=1e-5)


def test_ctc_greedy_blank_separated_repeats():
    classes = ["", "a"]
    # a blank a → "aa"（blank 分隔的重复不折叠）
    probs = _probs([1, 0, 1], 2)
    (text, _), = ctc_greedy(probs, classes)
    assert text == "aa"


def test_ctc_greedy_empty():
    probs = _probs([0, 0, 0], 3)
    assert ctc_greedy(probs, ["", "a", "b"]) == [("", 0.0)]


def test_ctc_greedy_score_mean_of_kept():
    classes = ["", "a", "b"]
    probs = _probs([1, 2], 3, peak=0.8)
    probs[0, 0, 1] = 0.6  # 首帧概率降低
    probs[0, 0, 0] = 0.2
    probs[0, 0, 2] = 0.2
    (text, score), = ctc_greedy(probs, classes)
    assert text == "ab"
    assert score == pytest.approx((0.6 + 0.8) / 2, abs=1e-5)


def test_ctc_greedy_batch():
    probs = np.concatenate([_probs([0, 1], 3), _probs([2, 0], 3)], axis=0)
    out = ctc_greedy(probs, ["", "a", "b"])
    assert [t for t, _ in out] == ["a", "b"]


def test_read_dict(tmp_path):
    p = tmp_path / "dict.txt"
    p.write_text("ก\nข\na\n\n", encoding="utf-8")  # 空行过滤
    assert read_dict(p) == ["", "ก", "ข", "a", " "]


def test_prep_crop_bgr_unchanged():
    crop = np.zeros((10, 40, 3), np.uint8)
    crop[:, :, 0] = 100
    assert prep_crop(crop, "bgr") is crop


def test_prep_crop_gray():
    crop = np.zeros((10, 40, 3), np.uint8)
    crop[:, :, 1] = 200  # 纯绿
    out = prep_crop(crop, "gray")
    assert out.shape == crop.shape
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])


def test_prep_crop_flatten_dark_background():
    # 暗底亮字 → 白底黑字
    crop = np.full((24, 120, 3), 30, np.uint8)
    crop[8:16, 20:100] = 220
    out = prep_crop(crop, "flatten")
    g = out[:, :, 0]
    assert g[0, 0] > 200  # 背景变白
    assert g[12, 50] < 100  # 文字变黑


def test_prep_crop_flatten_white_background():
    crop = np.full((24, 120, 3), 250, np.uint8)
    crop[8:16, 20:100] = 20
    out = prep_crop(crop, "flatten")
    g = out[:, :, 0]
    assert g[0, 0] > 200
    assert g[12, 50] < 100


def test_prep_crop_gray_inv_light_on_dark():
    crop = np.full((24, 120, 3), 40, np.uint8)
    crop[8:16, 20:100] = 230
    out = prep_crop(crop, "gray_inv")
    g = out[:, :, 0]
    assert g[0, 0] > 128  # 暗底反相为亮底
    assert g[12, 50] < 128


def test_prep_crop_stretch_low_contrast():
    crop = np.full((24, 120, 3), 120, np.uint8)
    crop[8:16, 20:100] = 140
    out = prep_crop(crop, "stretch")
    g = out[:, :, 0]
    assert g[0, 0] > 200  # 低对比拉伸后背景接近白
    assert g[12, 50] < 60


def test_color_modes_constant():
    assert set(COLOR_MODES) == {"bgr", "gray", "gray_inv", "stretch", "flatten"}