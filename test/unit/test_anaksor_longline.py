"""anaksor 长行窗口与字簇拼接单测（合成 chars / 合成图，无权重）。"""

from __future__ import annotations

import numpy as np
import pytest
from oxocr.models.anaksor_ocr.longline import clusters, ink_profile, line_windows, stitch
from oxocr.models.anaksor_ocr.text import COENG

# ── ink_profile ──────────────────────────────────────────────────────────────


def test_ink_profile_dark_on_light():
    img = np.full((20, 60, 3), 255, dtype=np.uint8)
    img[:, 20:30] = 0
    prof = ink_profile(img)
    assert prof.shape == (60,)
    assert prof[20:30].mean() > prof[:20].mean() + 5
    assert prof[:20].max() <= 1


def test_ink_profile_light_on_dark():
    img = np.zeros((20, 60, 3), dtype=np.uint8)
    img[:, 20:30] = 255
    prof = ink_profile(img)
    assert prof[20:30].mean() > prof[:20].mean() + 5


# ── line_windows ─────────────────────────────────────────────────────────────


def test_line_windows_short_single():
    crop = np.full((10, 150, 3), 255, dtype=np.uint8)  # ratio 15 <= 20
    wins, stitches = line_windows(crop, max_ratio=20)
    assert len(wins) == 1 and stitches == []
    piece, x0, x1 = wins[0]
    assert (x0, x1) == (0, 150)
    assert piece.shape == crop.shape


def test_line_windows_wide_overlap():
    crop = np.full((10, 300, 3), 255, dtype=np.uint8)  # ratio 30 > 20
    wins, stitches = line_windows(crop, max_ratio=20)
    assert len(wins) == 2
    assert len(stitches) == 1
    (p0, a0, a1), (p1, b0, b1) = wins
    assert a0 == 0 and b1 == 300  # 覆盖全宽
    assert b0 < a1  # 有重叠
    assert b0 <= stitches[0] <= a1  # stitch 落在重叠区
    assert p0.shape[1] == a1 - a0
    assert p1.shape[1] == b1 - b0


def test_line_windows_stitch_near_low_ink():
    crop = np.full((10, 300, 3), 255, dtype=np.uint8)
    crop[:, 150] = 0  # 重叠区中部一条低墨列
    wins, stitches = line_windows(crop, max_ratio=20)
    assert abs(stitches[0] - 150) <= 6


# ── clusters ─────────────────────────────────────────────────────────────────


def test_clusters_group_marks_and_coeng():
    chars = [
        ("ក", 0.1, 0.9),
        (COENG, 0.15, 0.8),
        ("ម", 0.2, 0.85),
        ("ា", 0.25, 0.7),
        ("ខ", 0.5, 0.6),
    ]
    cs = clusters(chars)
    assert len(cs) == 2
    assert cs[0]["x"] == pytest.approx(0.1)
    assert [c for c, _ in cs[0]["chars"]] == ["ក", COENG, "ម", "ា"]
    assert cs[1]["x"] == pytest.approx(0.5)


def test_clusters_empty():
    assert clusters([]) == []


# ── stitch ───────────────────────────────────────────────────────────────────


def test_stitch_dedup_overlap():
    windows = [(None, 0, 100), (None, 80, 180)]
    stitches = [90]
    results = [
        {"chars": [("a", 0.2, 0.9), ("b", 0.85, 0.8)]},
        {"chars": [("b", 0.05, 0.7), ("c", 0.5, 0.6)]},
    ]
    text, score = stitch(windows, stitches, results)
    assert text == "abc"  # 重叠区的 b 只取一次
    assert score == pytest.approx((0.9 + 0.8 + 0.6) / 3)


def test_stitch_cluster_not_split():
    windows = [(None, 0, 100), (None, 80, 180)]
    stitches = [90]
    results = [
        {"chars": [("ក", 0.88, 0.9), ("ា", 0.92, 0.8)]},
        {"chars": [("ក", 0.05, 0.7), ("ា", 0.09, 0.6), ("ខ", 0.5, 0.5)]},
    ]
    text, score = stitch(windows, stitches, results)
    assert text == "កាខ"  # កា 字簇整体取自窗口 0，不被 stitch 切断
    assert score == pytest.approx((0.9 + 0.8 + 0.5) / 3)


def test_stitch_single_window():
    windows = [(None, 0, 100)]
    results = [{"chars": [("a", 0.3, 0.9), ("b", 0.7, 0.8)]}]
    text, score = stitch(windows, [], results)
    assert text == "ab"
    assert score == pytest.approx(0.85)


def test_stitch_empty():
    text, score = stitch([(None, 0, 100)], [], [{"chars": []}])
    assert text == "" and score == 0.0
