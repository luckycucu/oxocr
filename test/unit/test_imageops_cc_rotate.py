"""connected_components / rotate / area-resize 与 cv2 对照（cv2 仅 dev 依赖）。"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from oxocr.runtime import imageops  # noqa: E402


def _partition(labels: np.ndarray, n: int) -> set:
    return {frozenset(map(tuple, np.argwhere(labels == i))) for i in range(1, n)}


@pytest.mark.parametrize("conn", [4, 8])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_connected_components_matches_cv2(conn, seed):
    rng = np.random.default_rng(seed)
    mask = (rng.random((48, 64)) < 0.35).astype(np.uint8)
    n, lab, stats = imageops.connected_components(mask, conn)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(mask, connectivity=conn)
    assert n == n2
    # 编号次序允许不同（cv2 union 策略），分区与每域 stats 必须一致
    assert _partition(lab, n) == _partition(lab2, n2)
    s1 = stats[1:][np.lexsort((stats[1:, 0], stats[1:, 1]))]
    s2 = stats2[1:][np.lexsort((stats2[1:, 0], stats2[1:, 1]))]
    assert np.array_equal(s1, s2)
    for i in range(1, n):
        ys, xs = np.where(lab == i)
        assert stats[i, 0] == xs.min() and stats[i, 1] == ys.min()
        assert stats[i, 2] == xs.max() - xs.min() + 1
        assert stats[i, 3] == ys.max() - ys.min() + 1
        assert stats[i, 4] == len(xs)


def test_connected_components_empty_and_single():
    n, lab, stats = imageops.connected_components(np.zeros((10, 12), np.uint8), 4)
    assert n == 1 and lab.sum() == 0 and stats[0, 4] == 120
    mask = np.zeros((9, 9), np.uint8)
    mask[2:5, 3:7] = 1
    n, lab, stats = imageops.connected_components(mask, 8)
    assert n == 2
    assert np.array_equal(stats[1], [3, 2, 4, 3, 12])


def test_connected_components_diagonal_connectivity():
    mask = np.array([[1, 0], [0, 1]], np.uint8)
    n4, _, _ = imageops.connected_components(mask, 4)
    n8, _, _ = imageops.connected_components(mask, 8)
    assert n4 == 3 and n8 == 2
    with pytest.raises(ValueError):
        imageops.connected_components(mask, 6)


@pytest.mark.parametrize("angle", [0.0, 3.5, -12.0, 27.0])
def test_rotate_smooth_matches_cv2(angle):
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    smooth = cv2.GaussianBlur(noise, (15, 15), 0)
    mine = imageops.rotate(smooth, angle, fill=255)
    m = cv2.getRotationMatrix2D((80, 60), angle, 1.0)
    ref = cv2.warpAffine(
        smooth, m, (160, 120), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    assert mine.shape == smooth.shape
    interior = (slice(20, 100), slice(25, 135))
    diff = np.abs(mine[interior].astype(int) - ref[interior].astype(int))
    assert diff.mean() < 3.0


def test_rotate_direction_and_fill():
    img = np.full((40, 60), 255, np.uint8)
    img[8:13, 48:53] = 0  # 右上角附近的墨块
    out = imageops.rotate(img, 90.0, fill=255)
    assert out.shape == img.shape
    ys, xs = np.where(out < 128)
    # 逆时针 90°：右上 → 左上
    assert xs.mean() < 30 and ys.mean() < 20
    tilted = imageops.rotate(img, 30.0, fill=255)
    assert tilted[0, 0] == 255  # 露出区域填纸白
    ink = np.zeros((40, 60), np.uint8)
    ink[18:23, 28:33] = 200
    assert imageops.rotate(ink, 30.0, fill=0)[0, 0] == 0


def test_resize_area_matches_cv2():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (100, 100), dtype=np.uint8)
    mine = imageops.resize(img, (25, 25), "area")
    ref = cv2.resize(img, (25, 25), interpolation=cv2.INTER_AREA)
    assert np.abs(mine.astype(int) - ref.astype(int)).mean() < 2.0
