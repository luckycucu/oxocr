"""Indra LineNet 检测单测（合成 logit，无权重）：decode 成框 / page_to_input / 自适应。"""
from __future__ import annotations

import numpy as np
import pytest
from oxocr.models.indra_ocr.detector import (
    ADAPTIVE_MAX_SIDE,
    LineDetector,
    decode,
    page_to_input,
)


def _synthetic_out(h: int = 16, w: int = 32, y0: float = 12.0, y1: float = 20.0):
    """core 带 rows 6..9（输入 y 12..20），cols 4..27；上下距离回归精确。"""
    out = np.full((3, h, w), -10.0, np.float32)
    rows = np.arange(h, dtype=np.float32)
    ys = (rows + 0.5) * 2
    band = (ys >= y0) & (ys <= y1)
    out[0][band, 4:28] = 10.0
    out[1][band, 4:28] = (ys[band] - y0)[:, None]
    out[2][band, 4:28] = (y1 - ys[band])[:, None]
    return out


def test_decode_single_line():
    boxes = decode(_synthetic_out(), thresh=0.5)
    assert boxes.shape == (1, 5)
    x0, y0, x1, y1, score = boxes[0]
    assert y0 == pytest.approx(12.0, abs=0.6)
    assert y1 == pytest.approx(20.0, abs=0.6)
    # core 左右界 4*2 / 28*2，外扩 CORE_X_SHRINK * 行高 = 1.2
    assert x0 == pytest.approx(8 - 1.2, abs=0.1)
    assert x1 == pytest.approx(56 + 1.2, abs=0.1)
    assert score > 0.99


def test_decode_two_lines_and_min_area():
    out = _synthetic_out(h=32)
    # 第二条 core 带 rows 22..25（y 44..52）
    rows = np.arange(32, dtype=np.float32)
    ys = (rows + 0.5) * 2
    band2 = (ys >= 44) & (ys <= 52)
    out[0][band2, 2:20] = 10.0
    out[1][band2, 2:20] = (ys[band2] - 44)[:, None]
    out[2][band2, 2:20] = (52 - ys[band2])[:, None]
    # 单像素噪点（面积 1 < min_area 2）
    out[0, 30, 30] = 10.0
    out[1, 30, 30] = 1.0
    out[2, 30, 30] = 1.0
    boxes = decode(out, thresh=0.5, min_area=2)
    assert len(boxes) == 2
    assert boxes[1][1] == pytest.approx(44.0, abs=0.6)


def test_decode_empty():
    out = np.full((3, 8, 8), -10.0, np.float32)
    assert decode(out).shape == (0, 5)


def test_decode_thresh():
    out = _synthetic_out()
    out[0][6:10, 4:28] = 0.2  # sigmoid(0.2) ≈ 0.55
    assert len(decode(out, thresh=0.5)) == 1
    assert len(decode(out, thresh=0.6)) == 0


def test_page_to_input_pad_and_scale():
    gray = np.full((100, 300), 128, np.uint8)
    x, s = page_to_input(gray, 1024)
    assert s == pytest.approx(1024 / 300)
    assert x.shape[0] % 32 == 0 and x.shape[1] % 32 == 0
    assert x.shape[1] == 1024
    nh = round(100 * s)
    assert x.shape[0] == ((nh + 31) // 32) * 32
    assert x[0, 0] == pytest.approx(1 - 128 / 255, abs=1e-3)
    assert x[nh:, :].sum() == 0  # 白补边 ink=0
    # 恒缩放到 long_side（小图也放大，同示例）
    small = np.zeros((10, 10), np.uint8)
    x2, s2 = page_to_input(small, 32)
    assert s2 == pytest.approx(3.2) and x2.shape == (32, 32)


def test_adaptive_redetect_monkeypatched():
    det = LineDetector.__new__(LineDetector)  # 不建会话
    calls: list[int] = []

    def fake(gray, long_side, thresh):
        calls.append(long_side)
        if long_side == 1024:
            # 8 行、行高 5 < ADAPTIVE_MIN_H(9) → 触发放大
            b = np.zeros((8, 5), np.float32)
            b[:, 1] = np.arange(8) * 10
            b[:, 3] = b[:, 1] + 5
            b[:, 4] = 0.9
            return b, 1.0
        b = np.zeros((8, 5), np.float32)
        b[:, 1] = np.arange(8) * 20
        b[:, 3] = b[:, 1] + 12
        b[:, 4] = 0.9
        return b, 2.0

    det._detect_at = fake
    gray = np.zeros((200, 200), np.uint8)
    boxes = det(gray, long_side=1024, thresh=0.5, adaptive=True)
    assert calls == [1024, ADAPTIVE_MAX_SIDE]  # 1024*12/5=2457 → 上限 2048
    assert boxes[0, 3] - boxes[0, 1] == pytest.approx(6.0)  # 12 / scale 2
    calls.clear()
    det(gray, long_side=1024, thresh=0.5, adaptive=False)
    assert calls == [1024]  # 关闭自适应不重检
