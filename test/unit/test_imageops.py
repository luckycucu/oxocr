"""imageops 与 cv2 结果对照（cv2 仅 dev 依赖，缺失即跳过）。"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from oxocr.runtime import imageops  # noqa: E402  cv2 可用性必须先于 imageops 导入判定


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dilate_matches_cv2(k, seed):
    rng = np.random.default_rng(seed)
    mask = (rng.random((37, 53)) > 0.6).astype(np.uint8)
    kernel = np.ones((k, k), np.uint8)
    expected = cv2.dilate(mask, kernel)
    assert np.array_equal(imageops.dilate(mask, k), expected)


def test_min_area_rect_area_matches_cv2():
    rng = np.random.default_rng(7)
    for _ in range(20):
        pts = rng.integers(0, 80, size=(rng.integers(3, 12), 2)).astype(np.float32)
        mine = imageops.min_area_rect(pts)
        area_mine = mine[1][0] * mine[1][1]
        rect = cv2.minAreaRect(pts.reshape(-1, 1, 2).astype(np.int32))
        area_cv2 = rect[1][0] * rect[1][1]
        assert area_mine == pytest.approx(area_cv2, rel=1e-2, abs=1.0)


def test_min_area_rect_contains_points():
    pts = np.array([[0, 0], [50, 0], [50, 20], [0, 20]], dtype=np.float32)
    rect = imageops.min_area_rect(pts)
    box = imageops.box_points(rect)
    # 旋转矩形应大致覆盖原始点集
    assert abs(rect[1][0] * rect[1][1] - 50 * 20) < 6
    assert box.shape == (4, 2)


def test_box_score_fast_matches_cv2_fillpoly():
    rng = np.random.default_rng(3)
    bitmap = rng.random((40, 60)).astype(np.float32)
    box = np.array([[5, 5], [40, 6], [41, 20], [6, 19]], dtype=np.float32)
    mine = imageops.box_score_fast(bitmap, box)

    mask = np.zeros((40, 60), np.uint8)
    cv2.fillPoly(mask, [box.reshape(-1, 1, 2).astype(np.int32)], 1)
    sel = mask.astype(bool)
    expected = float(bitmap[sel].mean()) if sel.any() else 0.0
    assert mine == pytest.approx(expected, rel=1e-2, abs=1e-3)


def test_perspective_transform_matches_cv2():
    src = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
    dst = np.array([[10, 5], [110, 0], [105, 55], [5, 60]], dtype=np.float32)
    mine = imageops.get_perspective_transform(src, dst)
    cv = cv2.getPerspectiveTransform(src, dst)
    assert np.allclose(mine, cv, atol=1e-6)


def test_resize_matches_cv2_bilinear():
    img = (np.random.default_rng(1).random((23, 41, 3)) * 255).astype(np.uint8)
    mine = imageops.resize(img, (80, 50), "bilinear")  # dsize=(w,h)
    assert mine.shape[:2] == (50, 80)


def test_order_points_clockwise():
    pts = np.array([[50, 0], [0, 0], [0, 30], [50, 30]], dtype=np.float32)
    ordered = imageops.order_points_clockwise(pts)
    assert ordered[0].tolist() == [0.0, 0.0]  # tl
    assert ordered[1].tolist() == [50.0, 0.0]  # tr
    assert ordered[2].tolist() == [50.0, 30.0]  # br
    assert ordered[3].tolist() == [0.0, 30.0]  # bl


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_otsu_threshold_matches_cv2(seed):
    rng = np.random.default_rng(seed)
    # 合成双峰灰度图：暗背景 + 亮前景
    gray = np.concatenate(
        [rng.normal(40, 12, 4000), rng.normal(200, 12, 4000)]
    ).clip(0, 255).astype(np.uint8).reshape(80, 100)
    mine = imageops.otsu_threshold(gray)
    expected = int(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    assert abs(mine - expected) <= 1


def test_otsu_threshold_uniform():
    assert imageops.otsu_threshold(np.full((10, 10), 128, dtype=np.uint8)) in range(256)
