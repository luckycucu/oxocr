"""anaksor 几何原语单测（纯 numpy/PIL，无权重）：角点排序、扩边、透视裁剪、包含框剔除。

warp_crop 的 BORDER_REPLICATE 等价与 cv2 对照为 dev-only（缺 cv2 即跳过该用例）。
"""

from __future__ import annotations

import numpy as np
import pytest
from oxocr.models.anaksor_ocr.geometry import (
    drop_contained,
    expand_quad,
    order_quad,
    quad_size,
    warp_crop,
)
from oxocr.runtime import imageops


def _rotate(pts: np.ndarray, deg: float) -> np.ndarray:
    c = pts.mean(0)
    a = np.deg2rad(deg)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts - c) @ r.T + c


# ── order_quad ───────────────────────────────────────────────────────────────


def test_order_quad_axis_aligned():
    pts = np.array([[100, 40], [0, 0], [0, 40], [100, 0]], dtype=np.float32)
    q = order_quad(pts)
    assert np.allclose(q, [[0, 0], [100, 0], [100, 40], [0, 40]])


@pytest.mark.parametrize("deg", [30, 45, 90, 135])
def test_order_quad_rotated(deg):
    base = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], dtype=np.float64)
    rot = _rotate(base, deg)
    q = order_quad(rot[[2, 0, 3, 1]])
    # 起点 x+y 最小（45° 可能有并列最小，比较值而非点）；邻边长为 {100, 40}
    assert q[0].sum() == pytest.approx(rot.sum(1).min(), abs=1e-3)
    d01 = np.linalg.norm(q[1] - q[0])
    d12 = np.linalg.norm(q[2] - q[1])
    assert {round(d01), round(d12)} == {40, 100}
    assert np.allclose(np.sort(q.sum(1)), np.sort(rot.sum(1)), atol=1e-3)


def test_order_quad_more_than_4_points():
    pts = np.array([[0, 0], [50, -5], [100, 0], [100, 40], [50, 45], [0, 40]], dtype=np.float32)
    q = order_quad(pts)
    assert q.shape == (4, 2)
    w, h = quad_size(q)
    assert w == pytest.approx(100, abs=6)
    assert h == pytest.approx(50, abs=6)  # 点集 y 跨度 -5..45


# ── quad_size / expand_quad ──────────────────────────────────────────────────


def test_quad_size_axis_aligned():
    q = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], dtype=np.float32)
    assert quad_size(q) == pytest.approx((100, 40))


def test_quad_size_rotated():
    base = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], dtype=np.float64)
    w, h = quad_size(_rotate(base, 30))
    assert w == pytest.approx(100, abs=1e-3)
    assert h == pytest.approx(40, abs=1e-3)


def test_expand_quad_axis_aligned():
    q = np.array([[10, 10], [110, 10], [110, 50], [10, 50]], dtype=np.float32)
    e = expand_quad(q, 2, 3)
    assert np.allclose(e, [[8, 7], [112, 7], [112, 53], [8, 53]])


def test_expand_quad_rotated_grows_along_own_axes():
    base = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], dtype=np.float64)
    rot = _rotate(base, 30)
    e = expand_quad(rot, 5, 5)
    w, h = quad_size(e)
    assert w == pytest.approx(110, abs=1e-3)
    assert h == pytest.approx(50, abs=1e-3)


# ── warp_crop ────────────────────────────────────────────────────────────────


def test_warp_crop_identity():
    img = np.arange(100 * 60 * 3, dtype=np.uint8).reshape(60, 100, 3)
    q = np.array([[0, 0], [100, 0], [100, 60], [0, 60]], dtype=np.float32)
    crop = warp_crop(img, q)
    assert crop.shape == (60, 100, 3)
    assert np.array_equal(crop, img)


def test_warp_crop_out_of_bounds_edge_fill():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[0, :] = (10, 20, 30)  # 顶边
    img[:, 0] = (40, 50, 60)  # 左边
    img[0, 0] = (70, 80, 90)
    q = np.array([[-3, -3], [7, -3], [7, 7], [-3, 7]], dtype=np.float32)
    crop = warp_crop(img, q)
    assert crop.shape == (10, 10, 3)
    # crop(r,c) 采样原图 (c-3, r-3)；越界复制最近边界像素
    assert np.array_equal(crop[0, 0], (70, 80, 90))  # 左上角双越界 → img[0,0]
    assert np.array_equal(crop[5, 0], (40, 50, 60))  # 仅左越界 → img[2,0]
    assert np.array_equal(crop[0, 5], (10, 20, 30))  # 仅上越界 → img[0,2]


def test_warp_crop_matches_cv2_replicate():
    cv2 = pytest.importorskip("cv2")
    # 平滑渐变图：双三次核差异小，越界若未做 edge 填充则边界差异巨大
    yy, xx = np.mgrid[0:40, 0:60]
    img = np.stack([xx * 4, yy * 6, (xx + yy) * 2], -1).astype(np.uint8)
    q = np.array([[-4, -2], [50, -5], [55, 30], [-2, 35]], dtype=np.float32)
    mine = warp_crop(img, q)
    w, h = quad_size(q)
    w, h = max(int(round(w)), 2), max(int(round(h)), 2)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    expected = cv2.warpPerspective(
        img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    assert mine.shape == expected.shape
    assert np.abs(mine.astype(int) - expected.astype(int)).mean() < 3.0


# ── drop_contained ───────────────────────────────────────────────────────────


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def test_drop_contained_fragment():
    big = _rect(0, 0, 200, 100)
    frag = _rect(10, 10, 30, 30)  # 短边比 0.2 < 0.4 → 碎片
    assert drop_contained([big, frag]) == [0]


def test_drop_contained_duplicate():
    big = _rect(0, 0, 200, 100)
    dup = _rect(0, 0, 200, 90)  # 短边比 0.9 > 0.75 → 重复
    assert drop_contained([big, dup]) == [0]


def test_drop_contained_half_height_line_kept():
    big = _rect(0, 0, 200, 100)
    half = _rect(0, 0, 200, 50)  # 短边比 0.5 ∈ [0.4, 0.75] → 单行保留
    assert drop_contained([big, half]) == [0, 1]


def test_drop_contained_disjoint_kept():
    a = _rect(0, 0, 100, 100)
    b = _rect(200, 0, 300, 100)
    assert drop_contained([a, b]) == [0, 1]


def test_drop_contained_empty_and_single():
    assert drop_contained([]) == []
    assert drop_contained([_rect(0, 0, 10, 10)]) == [0]


# ── imageops.sort_boxes（检测框阅读序，scores 同步）──────────────────────────


def test_sort_boxes_reading_order():
    boxes = np.array(
        [
            [[100, 0], [200, 0], [200, 20], [100, 20]],  # 第一行右
            [[0, 50], [100, 50], [100, 70], [0, 70]],  # 第二行
            [[0, 0], [100, 0], [100, 20], [0, 20]],  # 第一行左
        ],
        dtype=np.int32,
    )
    scores = [0.6, 0.7, 0.8]
    sorted_boxes, sorted_scores = imageops.sort_boxes(boxes, scores)
    assert sorted_scores == [0.8, 0.6, 0.7]
    assert sorted_boxes[0][0][0] == 0 and sorted_boxes[0][0][1] == 0
    assert sorted_boxes[1][0][0] == 100
    assert sorted_boxes[2][0][1] == 50


def test_sort_boxes_empty():
    boxes = np.empty((0, 4, 2), dtype=np.int32)
    b, s = imageops.sort_boxes(boxes, [])
    assert len(b) == 0 and s == []
