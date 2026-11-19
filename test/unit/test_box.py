from __future__ import annotations

import numpy as np
import pytest
from oxocr.types import Box


def test_from_rect_properties():
    b = Box.from_rect(10, 20, 30, 40)
    assert b.x_min == 10 and b.y_min == 20
    assert b.x_max == 40 and b.y_max == 60
    assert b.width == 30 and b.height == 40
    assert b.center == (25.0, 40.0)
    assert b.y_center == 40.0
    assert b.as_rect() == (10, 20, 30, 40)


def test_from_points_and_to_list():
    b = Box.from_points([[0, 0], [4, 0], [4, 2], [0, 2]])
    assert b.to_list() == [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]
    assert b.points.dtype == np.float32
    with pytest.raises(ValueError):
        Box.from_points([[0, 0], [1, 1], [2, 2]])


def test_shifted():
    b = Box.from_rect(1, 2, 5, 5).shifted(10, -1)
    assert b.x_min == 11 and b.y_min == 1


def test_transformed_affine():
    b = Box.from_rect(0, 0, 2, 2)
    m = np.array([[2, 0, 1], [0, 3, 1], [0, 0, 1]], dtype=float)
    t = b.transformed(m)
    assert t.x_min == pytest.approx(1)
    assert t.y_min == pytest.approx(1)
    assert t.width == pytest.approx(4)
    assert t.height == pytest.approx(6)


def test_iou_identical_and_disjoint():
    b = Box.from_rect(0, 0, 10, 10)
    assert b.iou(b) == pytest.approx(1.0)
    far = Box.from_rect(100, 100, 10, 10)
    assert b.iou(far) == pytest.approx(0.0)


def test_iou_half_overlap():
    a = Box.from_rect(0, 0, 10, 10)
    b = Box.from_rect(5, 0, 10, 10)  # 交集 5x10，并集 15x10
    assert a.iou(b) == pytest.approx(50 / 150)


def test_array_protocol():
    b = Box.from_rect(0, 0, 1, 1)
    arr = np.asarray(b)
    assert arr.shape == (4, 2)
