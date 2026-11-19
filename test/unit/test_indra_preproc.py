"""Indra preproc 单测（合成图，无权重）：裁剪/暗行/批编排。"""
from __future__ import annotations

import numpy as np
from oxocr.models.indra_ocr.preproc import (
    crop_line,
    is_dark_line,
    pad_batch,
    to_gray,
    width_batches,
)


def _page_with_line(dark: bool = False) -> tuple[np.ndarray, list[float]]:
    """浅底深字：实心墨行；深底亮字：暗背景 + 亮字块（覆盖率 <50%，中位数是背景）。"""
    if dark:
        gray = np.full((120, 400), 30, np.uint8)
        gray[46:66, 100:300] = 240
        return gray, [50.0, 40.0, 350.0, 72.0]
    gray = np.full((120, 400), 250, np.uint8)
    gray[40:72, 50:350] = 20
    return gray, [50.0, 40.0, 350.0, 71.0]


def test_to_gray_bt601():
    bgr = np.array([[[10, 20, 30], [255, 255, 255]]], np.uint8)
    g = to_gray(bgr)
    assert g.dtype == np.uint8 and g.shape == (1, 2)
    assert g[0, 0] == int(10 * 0.114 + 20 * 0.587 + 30 * 0.299)  # 截断，同核心约定
    assert to_gray(g) is g  # 已是灰度直接返回


def test_is_dark_line():
    gray, box = _page_with_line(dark=True)
    assert is_dark_line(gray, box)
    gray, box = _page_with_line(dark=False)
    assert not is_dark_line(gray, box)
    assert not is_dark_line(gray, [1000, 1000, 1001, 1001])  # 页外空框


def test_crop_line_shape_and_ink():
    gray, box = _page_with_line()
    ink = crop_line(gray, box, height=32)
    assert ink.shape[0] == 32
    assert ink.shape[1] >= 8  # 8 倍数由 pad_batch 保证
    assert ink.dtype == np.float32
    assert 0.0 <= ink.min() and ink.max() <= 1.0
    # 行中心应是墨（ink 高），扩边角含纸（ink 低）
    assert ink[16, ink.shape[1] // 2] > 0.7
    assert ink[0, 0] < 0.2


def test_crop_line_window_tall():
    gray, box = _page_with_line()
    band = crop_line(gray, box, height=32, window=1)
    tall = crop_line(gray, box, height=32, window=3)
    assert tall.shape[0] == 96
    # 高窗中带与行带同尺度（宽度相近）
    assert abs(tall.shape[1] - band.shape[1]) <= 16
    assert tall[48, tall.shape[1] // 2] > 0.7  # 中带中心是墨


def test_crop_line_dark_inverts():
    gray, box = _page_with_line(dark=True)
    auto = crop_line(gray, box, height=32)  # 自动判暗行
    forced = crop_line(gray, box, height=32, dark=True)
    assert np.array_equal(auto, forced)
    assert auto[16, auto.shape[1] // 2] > 0.7  # 亮字反色后成墨
    light = crop_line(gray, box, height=32, dark=False)
    assert np.allclose(light, 1.0 - auto, atol=1e-6)  # 不反色则互补


def test_crop_line_clips_outside_page():
    gray = np.full((60, 60), 250, np.uint8)
    ink = crop_line(gray, [-20.0, -10.0, 80.0, 70.0], height=32)
    assert ink.shape[0] == 32 and ink.shape[1] >= 8


def test_pad_batch():
    lines = [np.ones((32, 17), np.float32), np.full((32, 40), 0.5, np.float32)]
    arr, widths = pad_batch(lines)
    assert arr.shape == (2, 1, 32, 40)  # 40 已是 8 倍数
    assert widths.tolist() == [17, 40]
    assert arr[0, 0, :, :17].sum() == 32 * 17
    assert arr[0, 0, :, 17:].sum() == 0  # 右补零
    odd = [np.ones((32, 17), np.float32)]
    arr, _ = pad_batch(odd)
    assert arr.shape[-1] == 24  # 补到 8 倍数


def test_width_batches_narrow_first_and_ratio():
    widths = [300, 60, 70, 64, 1000]
    batches = width_batches(widths, max_batch=64, max_ratio=1.3)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(5))
    # 窄优先：第一批从最窄开始
    assert batches[0][0] == 1
    # 60/64/70 同批（min_width=64 归一），300 与 1000 各自分批
    assert set(batches[0]) == {1, 2, 3}
    assert [300, 1000] == [widths[b[0]] for b in batches[1:]]


def test_width_batches_max_batch_and_area():
    assert all(len(b) <= 3 for b in width_batches([64] * 10, max_batch=3))
    # 面积上限：(len+1)*width <= max_area
    batches = width_batches([640] * 10, max_batch=64, max_area=64 * 640)
    assert all(len(b) * 640 <= 64 * 640 for b in batches)
    assert width_batches([]) == []
