"""visual 可视化模块测试：纯合成图像，无模型依赖、无 cv2 依赖。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oxocr import visual
from oxocr.errors import OxocrError
from oxocr.runtime.imageops import get_perspective_transform, warp_perspective
from oxocr.types import Box, TextItem

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_FONT = (
    REPO_ROOT / "examples" / "duguang-ocr-onnx-v2.5" / "dgocr" / "AlibabaPuHuiTi-3-45-Light.ttf"
)


@pytest.fixture(scope="module")
def font_path() -> str:
    if BUNDLED_FONT.is_file():
        return str(BUNDLED_FONT)
    try:
        return visual.find_font()
    except OxocrError:
        pytest.skip("无可用字体，跳过文字绘制用例")


@pytest.fixture
def white_img() -> np.ndarray:
    return np.full((120, 200, 3), 255, dtype=np.uint8)  # BGR


@pytest.fixture
def boxes() -> list[list[list[float]]]:
    return [
        [[10, 10], [120, 10], [120, 35], [10, 35]],
        [[20, 60], [180, 58], [181, 88], [21, 90]],
    ]


def test_draw_boxes_marks_red_and_keeps_input(white_img, boxes):
    before = white_img.copy()
    out = visual.draw_boxes(white_img, boxes)
    assert out.shape == white_img.shape
    assert np.array_equal(white_img, before)  # 不修改输入
    red = (out[:, :, 2] == 255) & (out[:, :, 0] == 0) & (out[:, :, 1] == 0)
    assert red.sum() > 0


def test_draw_boxes_drop_score(white_img, boxes):
    out = visual.draw_boxes(white_img, boxes, scores=[0.1, float("nan")])
    assert np.array_equal(out, white_img)


def test_draw_ocr_box_txt_layout(white_img, boxes, font_path):
    out = visual.draw_ocr_box_txt(white_img, boxes, ["你好", "world"], font_path=font_path)
    h, w = white_img.shape[:2]
    assert out.shape == (h, w * 2, 3)
    left, right = out[:, :w], out[:, w:]
    assert not np.array_equal(left, white_img)  # 左半有半透明色块
    dark = (right < 128).all(axis=2)
    assert dark.sum() > 0  # 右半有黑色文字


def test_draw_ocr_box_txt_overlapping_boxes_keep_text(white_img, font_path):
    # 白边填充 + bitwise_and：相邻框的文字不应互相擦除
    boxes = [
        [[10, 10], [100, 10], [100, 40], [10, 40]],
        [[105, 10], [190, 10], [190, 40], [105, 40]],
    ]
    out = visual.draw_ocr_box_txt(white_img, boxes, ["AAA", "BBB"], font_path=font_path)
    right = out[:, white_img.shape[1] :]
    dark = (right < 128).all(axis=2)
    assert dark[:, :100].sum() > 0
    assert dark[:, 100:].sum() > 0


def test_draw_ocr_box_txt_vertical(white_img, font_path):
    boxes = [[[150, 10], [170, 10], [170, 110], [150, 110]]]  # 高 > 2 宽且 > 30
    out = visual.draw_ocr_box_txt(white_img, boxes, ["竖排文字"], font_path=font_path)
    right = out[:, white_img.shape[1] :]
    assert (right < 128).all(axis=2).sum() > 0


def test_draw_ocr_box_txt_without_txts_needs_no_font(white_img, boxes, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("无文字时不应解析字体")

    monkeypatch.setattr(visual, "find_font", boom)
    out = visual.draw_ocr_box_txt(white_img, boxes)
    assert out.shape == (120, 400, 3)


def test_draw_ocr_with_text_panel(white_img, boxes, font_path):
    out = visual.draw_ocr(white_img, boxes, ["你好", "world"], font_path=font_path)
    assert out.shape[1] > 600  # 最长边（宽 200）缩放到 600 后拼接文字面板
    assert out.shape[0] == 360
    assert (out.shape[1] - 600) % 600 == 0


def test_draw_ocr_without_txts_equals_draw_boxes(white_img, boxes):
    assert np.array_equal(
        visual.draw_ocr(white_img, boxes), visual.draw_boxes(white_img, boxes)
    )


def test_text_visual_pagination(font_path):
    texts = [f"第{i}行文本内容测试" for i in range(60)]
    out = visual.text_visual(texts, img_h=200, img_w=600, font_path=font_path)
    assert out.shape[0] == 200
    assert out.shape[1] % 600 == 0
    assert out.shape[1] > 600  # 超高触发横向分页


def test_text_visual_long_line_wrap(font_path):
    out = visual.text_visual(["长" * 200], img_h=400, img_w=600, font_path=font_path)
    assert (out < 128).all(axis=2).sum() > 0


def test_text_visual_score_mismatch(font_path):
    with pytest.raises(ValueError):
        visual.text_visual(["a"], scores=[1.0, 0.9], font_path=font_path)


def test_draw_items_from_text_items(white_img, font_path):
    items = [
        TextItem(text="你好", score=0.99, box=Box.from_rect(10, 10, 110, 25)),
        TextItem(text="world", score=0.95, box=Box.from_rect(20, 60, 160, 30)),
    ]
    out = visual.draw_items(white_img, items, font_path=font_path)
    assert out.shape == (120, 400, 3)


def test_accepts_pil_and_path_inputs(white_img, boxes, tmp_path):
    pil = Image.fromarray(white_img[:, :, ::-1])
    out_pil = visual.draw_boxes(pil, boxes)
    p = tmp_path / "img.png"
    pil.save(p)
    out_path = visual.draw_boxes(str(p), boxes)
    assert np.array_equal(out_pil, out_path)


def test_find_font_explicit_path(font_path):
    assert visual.find_font(font_path) == font_path


def test_find_font_invalid_path():
    with pytest.raises(OSError):
        visual.find_font("no_such_font_file.ttf")


def test_warp_perspective_fillcolor_and_interpolation():
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    src = np.float32([[0, 0], [40, 0], [40, 40], [0, 40]])
    dst = np.float32([[0, 0], [20, 0], [20, 20], [0, 20]])  # 缩到左上角
    matrix = get_perspective_transform(src, dst)
    white = warp_perspective(
        img, matrix, (40, 40), interpolation="nearest", fillcolor=(255, 255, 255)
    )
    assert (white[30, 30] == 255).all()
    default = warp_perspective(img, matrix, (40, 40))
    assert (default[30, 30] == 0).all()  # 默认行为不变：黑边
    assert (white[5, 5] == 100).all()