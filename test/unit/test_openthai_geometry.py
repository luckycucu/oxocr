"""行裁剪 / 切分 / flatten / 中值滤波（合成图；cv2 对照仅 dev 依赖）。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from oxocr.models.openthai_ocr.geometry import (
    crop_line,
    flatten,
    line_height,
    median_blur_1d,
    page_skew,
    split_one,
    to_gray,
    vertical,
)
from oxocr.runtime import imageops


def _quad(x, y, w, h):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def test_to_gray_weights():
    img = np.zeros((2, 2, 3), np.uint8)
    img[0, 0] = (255, 0, 0)  # B
    img[0, 1] = (0, 255, 0)  # G
    img[1, 0] = (0, 0, 255)  # R
    g = to_gray(img)
    assert g[0, 0] == pytest.approx(255 * 0.114, abs=1)
    assert g[0, 1] == pytest.approx(255 * 0.587, abs=1)
    assert g[1, 0] == pytest.approx(255 * 0.299, abs=1)
    assert to_gray(g) is g


def test_median_blur_1d_odd_window():
    row = np.array([1, 9, 2, 8, 3, 7, 4], np.uint8)
    out = median_blur_1d(row, 3)
    # edge pad：首窗 [1,1,9] 中值 1
    assert out[0] == 1
    assert out[1] == 2  # [1,9,2]
    assert out[-1] == 4  # [7,4,4]


def test_flatten_white_bg_black_text():
    g = np.full((24, 120), 250, np.uint8)
    g[8:16, 20:100] = 20
    out = flatten(g)
    assert out[0, 0] > 200
    assert out[12, 50] < 100


def test_flatten_dark_bg_light_text():
    g = np.full((24, 120), 30, np.uint8)
    g[8:16, 20:100] = 220
    out = flatten(g)
    assert out[0, 0] > 200
    assert out[12, 50] < 100


def test_flatten_gradient_bg():
    # 左暗右亮渐变底 + 黑字：逐列背景减除后背景都变白，文字比本地背景暗
    # （拉伸尺度 hi 是整 crop 的 98% 分位，低对比列文字不会拉到纯黑——参考同语义）
    g = np.tile(np.linspace(40, 220, 120).astype(np.uint8), (24, 1))
    g[8:16, 20:100] = np.minimum(g[8:16, 20:100], 30)
    out = flatten(g)
    assert out[0, 0] > 180 and out[0, -1] > 180
    assert out[12, 90] < 128  # 高对比列文字接近黑
    assert out[12, 30] < out[0, 30] - 40  # 低对比列文字仍显著暗于背景


def test_crop_line_axis_aligned():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)
    q = _quad(10, 5, 100, 20)
    crop = crop_line(img, q)
    assert crop.shape == (20, 100, 3)
    ref = img[5:25, 10:110]
    assert np.abs(crop[2:-2, 2:-2].astype(int) - ref[2:-2, 2:-2].astype(int)).max() <= 2


def test_crop_line_out_of_bounds_edge_fill():
    img = np.full((30, 30, 3), 7, np.uint8)
    q = _quad(-5, -5, 20, 20)  # 越界 quad：edge 填充不报错
    crop = crop_line(img, q)
    assert crop.shape == (20, 20, 3)
    assert np.all(crop == 7)


def test_split_one_disabled():
    img = np.full((40, 400, 3), 255, np.uint8)
    b = _quad(0, 0, 400, 40)
    assert split_one(img, b, 0)[0] is b


def test_split_one_two_blocks():
    img = np.full((40, 400, 3), 255, np.uint8)
    img[10:30, 20:120] = 0
    img[10:30, 260:380] = 0
    b = _quad(0, 0, 400, 40)
    pieces = split_one(img, b, 1.2)  # 空白 140px >= 1.2*40
    assert len(pieces) == 2
    assert pieces[0][:, 0].max() < pieces[1][:, 0].min()


def test_split_one_trims_blank_ends():
    img = np.full((40, 400, 3), 255, np.uint8)
    img[10:30, 100:300] = 0
    b = _quad(0, 0, 400, 40)
    pieces = split_one(img, b, 1.2)  # 两端空白 100px >= 48：修剪
    assert len(pieces) == 1
    assert pieces[0] is not b
    assert pieces[0][:, 0].min() > 0
    assert pieces[0][:, 0].max() < 400


def test_split_one_short_box_untouched():
    img = np.full((40, 100, 3), 255, np.uint8)  # w < 4h
    b = _quad(0, 0, 100, 40)
    assert split_one(img, b, 1.2)[0] is b


def test_line_height_ignores_fragments():
    lines = [_quad(0, i * 40, 200, 20) for i in range(4)]
    frags = [_quad(50, i * 40 + 22, 8, 8) for i in range(6)]
    assert line_height(lines + frags) == pytest.approx(20.0)


def test_page_skew():
    b = _quad(0, 0, 200, 20)
    a = math.radians(5)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    tilted = b @ rot.T
    assert page_skew([tilted]) == pytest.approx(5.0, abs=0.01)
    assert page_skew([]) == 0.0


def test_vertical():
    assert vertical(_quad(0, 0, 20, 100), 2.0) == 1
    assert vertical(_quad(0, 0, 100, 20), 2.0) == -1
    assert vertical(_quad(0, 0, 50, 50), 2.0) == 0


def test_rotate_expand_shape_and_fill():
    img = np.full((100, 200, 3), 255, np.uint8)
    out = imageops.rotate(img, 90, fill=(1, 2, 3), expand=True)
    assert out.shape == (200, 100, 3)
    out45 = imageops.rotate(img, 45, expand=True)
    h, w = 100, 200
    s = math.sin(math.radians(45))
    assert out45.shape[0] == int(round(h * abs(s) + w * abs(s)))
    # 不扩画布默认行为不变
    assert imageops.rotate(img, 45).shape == img.shape


# -- cv2 对照（dev-only） --------------------------------------------------------


def test_to_gray_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (32, 64, 3), dtype=np.uint8)
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert np.abs(to_gray(img).astype(int) - ref.astype(int)).max() <= 1


@pytest.mark.parametrize("win", [3, 5, 9, 25, 49])
def test_median_blur_matches_cv2(win):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    row = rng.integers(0, 255, 200, dtype=np.uint8)
    strip = np.repeat(row[None, :], 3, 0)
    ref = cv2.medianBlur(strip, win)[1]
    assert np.array_equal(median_blur_1d(row, win), ref)


def test_flatten_matches_reference():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    g = rng.integers(0, 255, (48, 300), dtype=np.uint8)
    g[12:36, 40:260] = np.clip(g[12:36, 40:260].astype(int) - 80, 0, 255).astype(np.uint8)

    # 参考实现（OpenThai runtime _flatten，cv2 版）
    h, w = g.shape
    k = max(2, h // 6)
    bg = np.median(np.concatenate([g[:k], g[-k:]], 0), axis=0).astype(np.uint8)
    win = max(3, min(h | 1, 255))
    bg = cv2.medianBlur(np.ascontiguousarray(np.repeat(bg[None, :], 3, 0)), win)[1].astype(np.float32)
    d = np.abs(g.astype(np.float32) - bg[None, :])
    hist = np.cumsum(np.bincount(d.astype(np.uint8).ravel(), minlength=256))
    hi = max(float(np.searchsorted(hist, 0.98 * hist[-1])), 24.0)
    ref = (255 - np.clip(d * (255.0 / hi), 0, 255)).astype(np.uint8)
    assert np.array_equal(flatten(g), ref)


def test_rotate_expand_matches_cv2():
    cv2 = pytest.importorskip("cv2")
    # 平滑图：随机噪声会放大 PIL/cv2 双三次插值的边界核差异
    yy, xx = np.mgrid[0:80, 0:120]
    img = np.stack([xx * 2 % 256, yy * 3 % 256, (xx + yy) % 256], -1).astype(np.uint8)
    for angle in (5.0, -12.0, 90.0):
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        c, s = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(round(h * s + w * c)), int(round(h * c + w * s))
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        ref = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(9, 9, 9))
        out = imageops.rotate(img, angle, fill=(9, 9, 9), interpolation="bicubic", expand=True)
        assert out.shape == ref.shape
        # 只比内容内部（腐蚀 3px 去掉边界带）：PIL/cv2 对越界插值 tap 的
        # 填充语义不同（cv2 tap 级 vs PIL 整像素级），边界带差异是已知且可接受的。
        # mask 先 pad 0：cv2.erode 默认把画布外当 +inf，会漏掉贴画布边的内容边界
        mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (nw, nh),
                              flags=cv2.INTER_NEAREST, borderValue=0)
        mask = np.pad(mask, 3, mode="constant", constant_values=0)
        valid = cv2.erode(mask, np.ones((7, 7), np.uint8))[3:-3, 3:-3] > 0
        diff = np.abs(out.astype(int) - ref.astype(int))[valid]
        assert diff.mean() < 2.5  # PIL 双三次定点精度 ~2 灰阶
        assert diff.max() <= 3