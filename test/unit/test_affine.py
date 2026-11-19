"""Affine 变换栈：几何改动必须可逆映射回原图（方案核心修正 #4）。"""
from __future__ import annotations

import numpy as np
import pytest
from oxocr.preprocess.base import Affine
from oxocr.types import Box


def test_identity_roundtrip():
    a = Affine.identity()
    assert np.allclose(a.matrix, np.eye(3))
    assert np.allclose(a.inverse.matrix, np.eye(3))


def test_translation_inverse():
    a = Affine.translation(10, 7)
    b = Box.from_rect(3, 4, 20, 12)
    moved = a.apply_to_box(b)
    assert moved.x_min == pytest.approx(13)
    assert moved.y_min == pytest.approx(11)
    back = a.inverse.apply_to_box(moved)
    assert np.allclose(back.points, b.points)


def test_then_composition_order():
    t = Affine.translation(5, 5)
    s = Affine.scaling(2, 2)
    composed = t.then(s)  # 先平移再缩放
    p = composed.apply_to_points(np.array([[0, 0]], dtype=float))
    assert p[0].tolist() == pytest.approx([10.0, 10.0])
    inv = composed.inverse
    assert np.allclose(inv.apply_to_points(p), [[0, 0]])


def test_rotation_inverse_roundtrip():
    r = Affine.rotation(30, (50, 40))
    b = Box.from_rect(10, 10, 20, 8)
    rot = r.apply_to_box(b)
    back = r.inverse.apply_to_box(rot)
    assert np.allclose(back.points, b.points, atol=1e-4)


def test_scale_inverse_roundtrip():
    s = Affine.scaling(0.5, 0.25)
    b = Box.from_points([[0, 0], [8, 0], [8, 4], [0, 4]])
    scaled = s.apply_to_box(b)
    assert scaled.width == pytest.approx(4)
    back = s.inverse.apply_to_box(scaled)
    assert np.allclose(back.points, b.points)


def test_pad_then_inverse_returns_original_coords():
    # 模拟 pad_border：left=10, top=20，模型返回 padded 坐标 box → 逆映射回原图
    pad = Affine.translation(10, 20)
    processed_box = Box.from_rect(25, 30, 40, 10)
    original = pad.inverse.apply_to_box(processed_box)
    assert original.x_min == pytest.approx(15)
    assert original.y_min == pytest.approx(10)
