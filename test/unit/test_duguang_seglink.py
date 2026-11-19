"""SegLink 解码 parity golden：与参考 utils_seglink.py 对固定随机输入的输出逐值对比。

golden 向量入库 test/test-data/seglink_golden.npz，不依赖 examples/ 与权重。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from oxocr.models.duguang_ocr.seglink import (
    SegLinkDetector,
    _cal_width,
    _combine_segments,
    _nms,
    _rboxes_to_polygons,
)

_GOLDEN = Path(__file__).resolve().parent.parent / "test-data" / "seglink_golden.npz"


@pytest.fixture(scope="module")
def golden_data():
    if not _GOLDEN.is_file():
        pytest.skip("SegLink golden 向量未找到")
    return np.load(_GOLDEN)


def test_decode_parity(golden_data):
    """完整 _decode 管线与参考实现输出逐值一致。"""
    S = int(golden_data["S"])
    all_maps = [golden_data[f"map_{i}"] for i in range(18)]
    expected = golden_data["rboxes"]

    det = SegLinkDetector.__new__(SegLinkDetector)
    det.image_size = S
    det.node_thresh = 0.4
    det.link_thresh = 0.6
    rboxes = det._decode(all_maps)

    assert rboxes.shape == expected.shape
    np.testing.assert_allclose(rboxes, expected, atol=1e-5)


def test_rboxes_to_polygons_shape():
    rboxes = np.array([[10, 20, 30, 10, 0], [50, 60, 40, 15, 0.5]], dtype=np.float32)
    polygons = _rboxes_to_polygons(rboxes)
    assert polygons.shape == (2, 8)


def test_rboxes_to_polygons_zero_theta():
    rboxes = np.array([[10, 20, 30, 10, 0]], dtype=np.float32)
    polygons = _rboxes_to_polygons(rboxes)
    expected = np.array([[ -5, 15, 25, 15, 25, 25, -5, 25]], dtype=np.float32)
    np.testing.assert_allclose(polygons, expected, atol=1e-5)


def test_nms_suppresses_contained():
    boxes = [
        [0, 0, 100, 0, 100, 10, 0, 10, 100.0],
        [10, 2, 50, 2, 50, 8, 10, 8, 40.0],
    ]
    result = _nms(boxes)
    assert len(result) == 1
    assert result[0][8] == 100.0


def test_nms_keeps_non_overlapping():
    boxes = [
        [0, 0, 10, 0, 10, 10, 0, 10, 10.0],
        [100, 100, 110, 100, 110, 110, 100, 110, 10.0],
    ]
    result = _nms(boxes)
    assert len(result) == 2


def test_cal_width():
    box = [0, 0, 10, 0, 10, 5, 0, 5]
    assert _cal_width(box) == pytest.approx(10.0)


def test_combine_segments_empty():
    segments = np.empty((0, 6), dtype=np.float32)
    group_indices = np.empty((0,), dtype=np.int32)
    rboxes = _combine_segments(segments, group_indices)
    assert rboxes.shape == (0, 5)


def test_sorted_boxes_reading_order():
    det = SegLinkDetector.__new__(SegLinkDetector)
    boxes = np.array([
        [[100, 50], [200, 50], [200, 70], [100, 70]],
        [[10, 10], [50, 10], [50, 30], [10, 30]],
        [[60, 10], [100, 10], [100, 30], [60, 30]],
    ], dtype=np.float32)
    sorted_boxes = det._sorted_boxes(boxes)
    assert sorted_boxes[0, 0, 0] == 10
    assert sorted_boxes[1, 0, 0] == 60
    assert sorted_boxes[2, 0, 0] == 100
